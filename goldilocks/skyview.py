"""
skyview.py
-----------
Photorealistic ground-to-sky renderer for any planet in an N-star system.

The model is single-scattering atmospheric radiative transfer (Nishita
1993; popularised by O'Neil, GPU Gems 2; scratchapixel "Simulating the
Colors of the Sky"), generalised here to:

  * arbitrary stellar spectra        -- Planck(T_eff) per star, normalised
                                        to the L/d^2 flux the planet sees;
  * arbitrary atmospheres            -- Rayleigh + Mie coefficients derived
                                        from the planet's HabitabilityProfile
                                        (composition, surface pressure,
                                        scale height, haze);
  * any number of suns               -- in-scattering and the direct
                                        attenuated stellar disks are summed
                                        over every star above the horizon;
  * full-spectral colour             -- ~20 wavelengths integrated through
                                        CIE-XYZ colour-matching functions
                                        (Wyman, Sloan & Shirley 2013 fit),
                                        XYZ -> linear sRGB -> ACES tone-map.

Output layer only: depends on system / planets / habitability / stellar /
kepler; touches no physics primitive and pins no sanity value.

References
----------
* Nishita T. et al. 1993, SIGGRAPH (single-scattering sky)
* O'Neil S. 2005, GPU Gems 2, ch. 16 (ray-march form + constants)
* Bruneton & Neyret 2008, CGF 27 (validation of single-scattering look)
* Wyman, Sloan & Shirley 2013, JCGT 2(2) (analytic CIE 1931 CMF fit)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple, Dict

import numpy as np

from goldilocks.habitability import R_EARTH_M, M_EARTH_KG
from goldilocks.kepler import orbital_elements_to_state, orbital_period
from goldilocks.planets import M_EARTH_OVER_M_SUN
from goldilocks.stellar import AU_M, R_SUN_M, G_SI

# ---------------------------------------------------------------------
# Spectral grid + physical constants
# ---------------------------------------------------------------------
N_LAMBDA_DEFAULT = 20
H_PLANCK = 6.62607015e-34  # J s
C_LIGHT = 2.99792458e8  # m / s
K_BOLTZ = 1.380649e-23  # J / K

# Earth sea-level Rayleigh scattering at (440, 550, 680) nm  [m^-1]
# (O'Neil / scratchapixel reference values).
_BETA_R_EARTH = {440.0: 33.1e-6, 550.0: 13.5e-6, 680.0: 5.8e-6}
_BETA_M_EARTH = 21e-6  # m^-1, weakly wavelength dependent
_MIE_EXT_FACT = 1.1  # Mie extinction = 1.1 * scattering

# Lit-night background levels (relative to the unit-power stellar
# spectra used everywhere here).  Ratios follow the moonless night-sky
# budget airglow >> zodiacal >> integrated starlight (Roach & Gordon
# 1973; Haenel et al. 2018): a deep starlit blue-grey, never pure black.
_BG_STAR_GAIN = 2.1e-5
_AIRGLOW_LEVEL = 3.3e-6
_ZODIACAL_LEVEL = 1.5e-6
_MILKYWAY_LEVEL = 3.6e-6


def lambda_grid_nm(n: int = N_LAMBDA_DEFAULT) -> np.ndarray:
    """Visible sampling wavelengths in nm (380..740)."""
    return np.linspace(380.0, 740.0, n)


# ---------------------------------------------------------------------
# Black-body spectrum
# ---------------------------------------------------------------------
def planck_spectral(lam_nm: np.ndarray, teff: float) -> np.ndarray:
    """Planck spectral radiance (arbitrary units) at wavelengths lam_nm."""
    lam = np.asarray(lam_nm, dtype=float) * 1e-9
    teff = max(float(teff), 100.0)
    a = 2.0 * H_PLANCK * C_LIGHT ** 2 / lam ** 5
    expo = H_PLANCK * C_LIGHT / (lam * K_BOLTZ * teff)
    return a / np.expm1(np.clip(expo, 1e-6, 700.0))


# ---------------------------------------------------------------------
# CIE 1931 colour-matching functions (Wyman-Sloan-Shirley 2013 fit)
# ---------------------------------------------------------------------
def _g(x: np.ndarray, mu: float, s1: float, s2: float) -> np.ndarray:
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) * s) ** 2)


def cie_xyz_bar(lam_nm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic CIE 1931 2-deg x_bar, y_bar, z_bar at lam_nm (nm)."""
    x = np.asarray(lam_nm, dtype=float)
    xb = (1.056 * _g(x, 599.8, 0.0264, 0.0323)
          + 0.362 * _g(x, 442.0, 0.0624, 0.0374)
          - 0.065 * _g(x, 501.1, 0.0490, 0.0382))
    yb = (0.821 * _g(x, 568.8, 0.0213, 0.0247)
          + 0.286 * _g(x, 530.9, 0.0613, 0.0322))
    zb = (1.217 * _g(x, 437.0, 0.0845, 0.0278)
          + 0.681 * _g(x, 459.0, 0.0385, 0.0725))
    return xb, yb, zb


# sRGB (linear) <- CIE XYZ
_XYZ_TO_RGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
])


def _aces(x: np.ndarray) -> np.ndarray:
    """ACES filmic tone curve (Narkowicz 2015)."""
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                   0.0, 1.0)


def spectrum_to_srgb(spec: np.ndarray, lam_nm: np.ndarray,
                     exposure: float) -> np.ndarray:
    """(..., Nlam) spectral radiance -> (..., 3) uint8 sRGB.

    Integrates against the CIE CMFs, applies a scalar `exposure`, an ACES
    tone curve and a gamma of 2.2.
    """
    xb, yb, zb = cie_xyz_bar(lam_nm)
    dl = float(lam_nm[1] - lam_nm[0]) if len(lam_nm) > 1 else 1.0
    X = np.tensordot(spec, xb, axes=([-1], [0])) * dl
    Y = np.tensordot(spec, yb, axes=([-1], [0])) * dl
    Z = np.tensordot(spec, zb, axes=([-1], [0])) * dl
    xyz = np.stack([X, Y, Z], axis=-1) * exposure
    rgb = np.tensordot(xyz, _XYZ_TO_RGB.T, axes=([-1], [0]))
    rgb = np.maximum(rgb, 0.0)
    rgb = _aces(rgb)
    rgb = np.power(rgb, 1.0 / 2.2)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------
# Atmosphere model from the planet's HabitabilityProfile
# ---------------------------------------------------------------------
# Relative molecular refractivity (n-1) vs Earth air, by dominant gas.
_REFRACTIVITY = (
    ("CO2", 1.50), ("H2/He", 0.25), ("H2", 0.25), ("N2/CH4", 1.05),
    ("N2/O2", 1.00), ("trace", 0.10),
)


def _comp_factor(dominant_gas: str) -> float:
    g = (dominant_gas or "").upper()
    for key, fac in _REFRACTIVITY:
        if key.upper() in g:
            return fac
    return 1.0


@dataclass
class Atmosphere:
    beta_r: np.ndarray  # Rayleigh scatter coeff per wavelength [1/m]
    beta_m: float  # Mie scatter coeff [1/m]
    mie_g: float  # Henyey-Greenstein asymmetry
    h_rayleigh_m: float
    h_mie_m: float
    r_planet_m: float
    r_atmo_m: float
    ground_albedo: np.ndarray  # per-wavelength Lambert reflectance


def atmosphere_for(planet, lam_nm: np.ndarray) -> Atmosphere:
    """Build an Atmosphere from the planet's attached HabitabilityProfile."""
    prof = planet.habitability
    p_bar = float(getattr(prof, "surface_pressure_bar", 1.0)) if prof else 1.0
    mu_air = float(getattr(prof, "mean_molecular_weight", 29.0)) if prof else 29.0
    H_km = float(getattr(prof, "scale_height_km", 8.5)) if prof else 8.5
    gas = getattr(prof, "dominant_gas", "N2/O2") if prof else "N2/O2"
    albedo_v = float(getattr(prof, "bond_albedo", 0.30)) if prof else 0.30
    storm = float(getattr(prof, "storm_index", 0.3)) if prof else 0.3

    # Rayleigh: Earth lambda^-4 curve, scaled by column density (~ pressure)
    # and molecular refractivity of the dominant gas.
    lam = np.asarray(lam_nm, dtype=float)
    beta_r550 = _BETA_R_EARTH[550.0]
    scale = (p_bar / 1.0) * (29.0 / max(mu_air, 1.0)) * _comp_factor(gas)
    beta_r = beta_r550 * (550.0 / lam) ** 4 * scale

    # Mie: pressure-scaled, boosted for hazy/organic/overcast atmospheres.
    haze = 1.0 + 6.0 * storm
    if "CH4" in gas or "haze" in str(getattr(prof, "sky_description", "")):
        haze *= 3.0
    beta_m = _BETA_M_EARTH * (p_bar / 1.0) * haze
    mie_g = 0.76

    H_r = max(H_km, 0.3) * 1000.0
    H_m = max(H_r / 6.0, 200.0)
    r_p = float(planet.radius_re) * R_EARTH_M
    r_atmo = r_p + max(12.0 * H_r, 60_000.0)

    # Ground reflectance: grey at the Bond albedo, faintly warm.
    g = np.full_like(lam, np.clip(albedo_v, 0.03, 0.7))
    tint = np.interp(lam, [380, 550, 740], [0.85, 1.0, 1.12])
    albedo = np.clip(g * tint, 0.0, 0.95)
    return Atmosphere(beta_r, float(beta_m), mie_g, H_r, H_m,
                      r_p, r_atmo, albedo)


# ---------------------------------------------------------------------
# Rotational flattening (oblate spheroid)
# ---------------------------------------------------------------------
# A rotating self-gravitating body is an oblate spheroid, not a sphere.
# In the slow-rotation Maclaurin limit the geometric flattening is
#   f = (R_eq - R_pol)/R_eq ~ (5/4) m,   m = w^2 R_eq^3 / (G M)
# (Maclaurin 1742; Murray & Dermott 1999, "Solar System Dynamics").
# Sanity: Jupiter f~0.065, Saturn~0.098, Earth~1/298 (~0.0034).
def oblateness_for(planet) -> float:
    """Geometric flattening f of `planet` from its rotation.

    Uses the sidereal spin (tidally-locked => spin = orbital) period
    from the attached HabitabilityProfile; falls back to 24 h."""
    prof = getattr(planet, "habitability", None)
    if prof is not None:
        period_h = float(getattr(prof, "sidereal_day_h", 24.0))
        if not math.isfinite(period_h) or period_h <= 0.0:
            period_h = float(getattr(prof, "rotation_period_h", 24.0))
    else:
        period_h = 24.0
    omega = 2.0 * math.pi / (max(period_h, 1e-6) * 3600.0)
    R_eq = float(planet.radius_re) * R_EARTH_M
    M = float(planet.mass_me) * M_EARTH_KG
    m = omega ** 2 * R_eq ** 3 / (G_SI * max(M, 1e12))
    return float(np.clip(1.25 * m, 0.0, 0.35))


def _ellipsoid_quadratic(O: np.ndarray, d: np.ndarray, r_eq: float,
                         f: float, axis: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quadratic (A, B, C) of |S(O + t d)|^2 = r_eq^2 for an oblate
    spheroid of equatorial radius `r_eq`, flattening `f`, spin `axis`.

    S is the exact affine map that squashes the polar direction by
    1/(1-f), turning the spheroid into a sphere of radius r_eq; the
    near/far roots of A t^2 + B t + C are the true ray hits.
    `O` is (3,) or (...,3), `d` is (...,3), `axis` a unit (3,)-vector.
    """
    k = 1.0 / max(1.0 - f, 1e-6) - 1.0
    ax = axis / (np.linalg.norm(axis) + 1e-15)
    SO = O + k * np.sum(O * ax, axis=-1, keepdims=True) * ax
    Sd = d + k * np.sum(d * ax, axis=-1, keepdims=True) * ax
    A = np.sum(Sd * Sd, axis=-1)
    B = 2.0 * np.sum(Sd * SO, axis=-1)
    C = np.sum(SO * SO, axis=-1) - r_eq ** 2
    return A, B, C


# ---------------------------------------------------------------------
# Geometry: where each star sits in the local sky
# ---------------------------------------------------------------------
@dataclass
class StarSky:
    name: str
    teff: float
    flux_rel: float  # L / d^2 in Earth units
    ang_radius_rad: float
    dir_inertial: np.ndarray  # unit vector planet -> star (orbital frame)


def _planet_position(sys, planet, orbit_phase: float) -> np.ndarray:
    """Planet position (AU) in the system frame at the given mean anomaly."""
    m_p = float(planet.mass_me) * M_EARTH_OVER_M_SUN
    a = float(planet.semi_major_axis_au or 1.0)
    e = float(planet.eccentricity)
    if planet.is_circumbinary():
        m_in = sum(s.mass for s in sys.stars)
        centre = sys.barycentre()
    else:
        host = sys.stars[planet.host_star_index]
        m_in = host.mass
        centre = np.array(host.position, dtype=float)
    r_rel, _ = orbital_elements_to_state(m_in, m_p, a, e,
                                         mean_anomaly=orbit_phase)
    return centre + r_rel


def star_sky_list(sys, planet, t_orbit: float = 0.0,
                  orbit_phase: float = 0.0) -> List[StarSky]:
    """Inertial direction / flux / angular size of every star, as seen
    from the planet at orbital time `t_orbit` and mean anomaly
    `orbit_phase`."""
    sys._update_stellar_positions(t_orbit)
    p_pos = _planet_position(sys, planet, orbit_phase)
    out: List[StarSky] = []
    for s in sys.stars:
        vec = np.array(s.position, dtype=float) - p_pos
        d = float(np.linalg.norm(vec))
        d = max(d, 1e-6)
        r_star_au = (s.radius or 1.0) * R_SUN_M / AU_M
        out.append(StarSky(
            name=s.name, teff=float(s.teff or 5772.0),
            flux_rel=float(s.luminosity) / d ** 2,
            ang_radius_rad=float(math.asin(min(r_star_au / d, 0.999))),
            dir_inertial=vec / d))

    return out


# ---------------------------------------------------------------------
# Reflective sky bodies: sibling planets + every moon
# ---------------------------------------------------------------------
R_EARTH_AU = R_EARTH_M / AU_M


def _lambert_phase(alpha: np.ndarray) -> np.ndarray:
    """Lambert-sphere phase function Phi(alpha), Phi(0)=1, Phi(pi)=0
    (Madhusudhan & Burrows 2012; Cahoy et al. 2010).  The reflected
    contrast scales as A_g (R/d)^2 Phi(alpha)."""
    a = np.clip(alpha, 0.0, math.pi)
    return (np.sin(a) + (math.pi - a) * np.cos(a)) / math.pi


def _hex_spectral_tint(hex_color: str, lam_nm: np.ndarray) -> np.ndarray:
    """Smooth per-wavelength multiplier (mean ~1) that reproduces the
    hue of a #RRGGBB body colour without changing its total power."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0,
                   int(h[4:6], 16) / 255.0)
    except Exception:
        r = g = b = 0.7
    anchors_l = np.array([380.0, 470.0, 550.0, 640.0, 740.0])
    anchors_v = np.array([b, b, g, r, r]) + 1e-3
    t = np.interp(np.asarray(lam_nm, float), anchors_l, anchors_v)
    return t / max(float(np.mean(t)), 1e-6)


def _phase0(name: str) -> float:
    """Deterministic per-body initial mean anomaly so siblings are not
    all phase-aligned at t=0.  Uses a stable hash (Python's str hash is
    salted per process)."""
    import zlib
    return (zlib.crc32(name.encode()) % 100000) / 100000.0 * 2.0 * math.pi


def _planet_pos_msun(sys, planet, mean_anom: float):
    """(position AU, central mass Msun) of `planet` at `mean_anom`."""
    m_p = float(planet.mass_me) * M_EARTH_OVER_M_SUN
    a = float(planet.semi_major_axis_au or 1.0)
    e = float(planet.eccentricity)
    if planet.is_circumbinary():
        m_in = sum(s.mass for s in sys.stars)
        centre = sys.barycentre()
    else:
        host = sys.stars[planet.host_star_index]
        m_in = host.mass
        centre = np.array(host.position, dtype=float)
    r_rel, _ = orbital_elements_to_state(m_in, m_p, a, e,
                                         mean_anomaly=mean_anom)
    return centre + r_rel, float(m_in)


@dataclass
class SkyBody:
    name: str
    kind: str  # 'planet' | 'moon'
    dir_inertial: np.ndarray  # unit, observer -> body (system frame)
    dist_au: float
    ang_radius_rad: float
    refl_spec: np.ndarray  # spectral irradiance at observer (Nlam)
    phase_frac: float  # illuminated fraction 0..1
    sub_star_dir: np.ndarray  # unit, body -> brightest star (sys frame)
    f_oblate: float
    spin_axis: np.ndarray  # unit (system frame), ~orbital normal
    # Filled by render_sky for bodies it actually composited (debug
    # overlay): pixel centre, on-screen radius (px), altitude (deg).
    screen_x: float = -1.0
    screen_y: float = -1.0
    screen_r: float = 0.0
    altitude_deg: float = 0.0


def _albedo_g(body, is_moon: bool) -> float:
    if is_moon:
        rho = float(getattr(body, "density_gcc", 2.5))
        return 0.40 if rho < 2.3 else 0.13  # icy bright vs rocky dark
    prof = getattr(body, "habitability", None)
    a = float(getattr(prof, "bond_albedo", 0.30)) if prof else 0.30
    return float(np.clip(a, 0.05, 0.75))


def sky_bodies(sys, observer_planet, lam_nm: np.ndarray,
               t_orbit: float = 0.0, orbit_phase: float = 0.0
               ) -> List[SkyBody]:
    """Every sibling planet and every moon (including the observer
    planet's own moons) as a reflected-light disk seen from the
    observer.  Stars must already be positioned (caller does
    `_update_stellar_positions`)."""
    obs_pos, _ = _planet_pos_msun(sys, observer_planet, orbit_phase)
    star_pos = [np.array(s.position, dtype=float) for s in sys.stars]
    star_L = [float(s.luminosity) for s in sys.stars]
    star_spec = []
    for s in sys.stars:
        pl = planck_spectral(lam_nm, float(s.teff or 5772.0))
        star_spec.append(pl / np.trapezoid(pl, np.asarray(lam_nm) * 1e-9))

    out: List[SkyBody] = []

    def _emit(body, b_pos, r_body_au, is_moon, tint):
        vec = b_pos - obs_pos
        d_obs = max(float(np.linalg.norm(vec)), 1e-9)
        bdir = vec / d_obs
        ang_r = float(math.asin(min(r_body_au / d_obs, 0.999)))
        A_g = _albedo_g(body, is_moon)
        spec = np.zeros_like(lam_nm, dtype=float)
        bright_i, bright_f = 0, -1.0
        cos_alpha_acc = 0.0
        for i, sp in enumerate(star_pos):
            to_star = sp - b_pos
            d_sb = max(float(np.linalg.norm(to_star)), 1e-9)
            u_star = to_star / d_sb
            cos_a = float(np.dot(u_star, -bdir))
            alpha = math.acos(np.clip(cos_a, -1.0, 1.0))
            phi = float(_lambert_phase(np.array(alpha)))
            refl = (star_L[i] / d_sb ** 2) * A_g \
                   * (r_body_au / d_obs) ** 2 * phi
            spec += refl * star_spec[i]
            if star_L[i] / d_sb ** 2 > bright_f:
                bright_f = star_L[i] / d_sb ** 2
                bright_i = i
                cos_alpha_acc = cos_a
        spec *= tint
        sub = star_pos[bright_i] - b_pos
        sub = sub / (np.linalg.norm(sub) + 1e-15)
        out.append(SkyBody(
            name=body.name, kind=("moon" if is_moon else "planet"),
            dir_inertial=bdir, dist_au=d_obs, ang_radius_rad=ang_r,
            refl_spec=spec, phase_frac=0.5 * (1.0 + cos_alpha_acc),
            sub_star_dir=sub, f_oblate=oblateness_for(body)
            if not is_moon else 0.0,
            spin_axis=np.array([0.0, 0.0, 1.0])))

    for p in sys.planets:
        if p.semi_major_axis_au is None:
            continue
        is_obs = p is observer_planet
        _, m_in0 = _planet_pos_msun(sys, p, 0.0)
        if is_obs:
            M_p = orbit_phase
        else:
            P_p = orbital_period(m_in0,
                                 float(p.mass_me) * M_EARTH_OVER_M_SUN,
                                 float(p.semi_major_axis_au))
            M_p = _phase0(p.name) + 2.0 * math.pi * t_orbit / max(P_p, 1e-9)
        p_pos, m_in = _planet_pos_msun(sys, p, M_p)
        if not is_obs:
            r_au = float(p.radius_re) * R_EARTH_AU
            _emit(p, p_pos, r_au, False,
                  _hex_spectral_tint(
                      p.habitability.sky_color_hex
                      if p.habitability else "#9AA7B5", lam_nm))
        m_planet = float(p.mass_me) * M_EARTH_OVER_M_SUN
        for mn in getattr(p, "moons", []):
            # Skip the faint captured-irregular swarm: invisible specks
            # that would otherwise cost a Kepler solve every frame.
            if mn.a_planet_au <= 0.0 or float(mn.mass_me) < 1e-5:
                continue
            m_moon = float(mn.mass_me) * M_EARTH_OVER_M_SUN
            P_m = orbital_period(m_planet, m_moon, mn.a_planet_au)
            sgn = -1.0 if getattr(mn, "retrograde", False) else 1.0
            M_m = _phase0(mn.name) + sgn * 2.0 * math.pi * t_orbit \
                  / max(P_m, 1e-12)
            off, _ = orbital_elements_to_state(m_planet, m_moon,
                                               mn.a_planet_au,
                                               mn.eccentricity,
                                               mean_anomaly=M_m)
            inc = math.radians(getattr(mn, "inclination_deg", 0.0))
            if inc:
                cy, syi = math.cos(inc), math.sin(inc)
                off = np.array([off[0],
                                cy * off[1] - syi * off[2],
                                syi * off[1] + cy * off[2]])
            _emit(mn, p_pos + off, float(mn.radius_re) * R_EARTH_AU,
                  True, np.ones_like(lam_nm, dtype=float))
    return out


# ---------------------------------------------------------------------
# Lit procedural background: stars + Milky-Way band + airglow floor
# ---------------------------------------------------------------------
# Moonless night-sky budget (Roach & Gordon 1973; Haenel et al. 2018;
# Masana et al. 2021): V ~ 22 mag/arcsec^2, airglow >> zodiacal >>
# integrated starlight >> diffuse galactic light.  The full procedural
# universe (galaxies, voids, dust, cached host-galaxy population) is a
# deferred ROADMAP item; this is the lightweight grounded stand-in.
@lru_cache(maxsize=8)
def background_starfield(seed: int = 2026, n_stars: int = 6000):
    """Deterministic celestial-sphere starfield with a Milky-Way band.

    Returns (dirs (N,3) unit, teff (N,), flux (N,) relative,
    plane_normal (3,)).  Cached so every frame of an animation and
    every phase still share one identical sky."""
    rng = np.random.default_rng(seed)
    n_band = int(0.55 * n_stars)
    n_iso = n_stars - n_band
    # Galactic plane: a fixed tilted great circle.
    pn = np.array([0.30, -0.20, 0.93])
    pn = pn / np.linalg.norm(pn)
    e1 = np.cross(pn, [0.0, 0.0, 1.0]);
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(pn, e1)
    # Isotropic field.
    u = rng.normal(size=(n_iso, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    # Band: small Gaussian galactic latitude about the plane.
    lon = rng.uniform(0.0, 2.0 * math.pi, n_band)
    blat = np.clip(rng.normal(0.0, math.radians(9.0), n_band),
                   -math.pi / 2, math.pi / 2)
    band = (np.cos(blat)[:, None]
            * (np.cos(lon)[:, None] * e1 + np.sin(lon)[:, None] * e2)
            + np.sin(blat)[:, None] * pn)
    dirs = np.vstack([u, band])
    # Colour: mostly cool dwarfs, a few hot stars.
    teff = np.clip(rng.lognormal(math.log(4200.0), 0.45, n_stars),
                   2400.0, 30000.0)
    # Brightness: steep counts -> a few bright, very many faint.
    flux = rng.pareto(1.6, n_stars) + 0.02
    flux = flux / np.percentile(flux, 99.0)
    return dirs, teff, flux, pn


def _local_basis(obliquity_deg: float, latitude_deg: float,
                 rot_phase: float) -> Tuple[np.ndarray, np.ndarray,
np.ndarray]:
    """Return (east, north, zenith) unit vectors of an observer at the
    given latitude and planet rotation phase, in the orbital frame.

    The spin axis is tilted by `obliquity` from the orbital normal (+z)
    toward +x; `rot_phase` (rad) advances the planet's rotation.
    """
    obl = math.radians(obliquity_deg)
    lat = math.radians(latitude_deg)
    n = np.array([math.sin(obl), 0.0, math.cos(obl)])  # spin axis
    ea = np.cross(n, [0.0, 0.0, 1.0])
    if np.linalg.norm(ea) < 1e-8:
        ea = np.array([1.0, 0.0, 0.0])
    ea = ea / np.linalg.norm(ea)
    eb = np.cross(n, ea)
    radial = (math.cos(lat) * (math.cos(rot_phase) * ea
                               + math.sin(rot_phase) * eb)
              + math.sin(lat) * n)
    zenith = radial / np.linalg.norm(radial)
    north = n - np.dot(n, zenith) * zenith
    if np.linalg.norm(north) < 1e-8:
        north = ea.copy()
    north = north / np.linalg.norm(north)
    east = np.cross(north, zenith)
    east = east / np.linalg.norm(east)
    return east, north, zenith


def _altitude(star_dir: np.ndarray, zenith: np.ndarray) -> float:
    return math.asin(float(np.clip(np.dot(star_dir, zenith), -1.0, 1.0)))


# ---------------------------------------------------------------------
# Phase solver: rotation phases for midnight / sunrise / noon / sunset
# ---------------------------------------------------------------------
def phase_rotations(sys, planet, latitude_deg: float = 20.0,
                    t_orbit: float = 0.0, orbit_phase: float = 0.0,
                    n_scan: int = 1440) -> Dict[str, float]:
    """Rotation phases of the four canonical lighting situations, anchored
    to the brightest star.  For a tidally-locked planet the phases are
    observer longitudes (substellar / terminators / antistellar)."""
    stars = star_sky_list(sys, planet, t_orbit, orbit_phase)
    bright = max(range(len(stars)), key=lambda i: stars[i].flux_rel)
    s_dir = stars[bright].dir_inertial
    prof = planet.habitability
    obl = float(getattr(prof, "obliquity_deg", 23.4)) if prof else 23.4

    rho = np.linspace(0.0, 2.0 * math.pi, n_scan, endpoint=False)
    alt = np.empty(n_scan)
    for k, r in enumerate(rho):
        _, _, z = _local_basis(obl, latitude_deg, float(r))
        alt[k] = _altitude(s_dir, z)

    i_noon = int(np.argmax(alt))
    i_mid = int(np.argmin(alt))
    # Zero crossings of altitude (horizon) -> sunrise (rising) / sunset.
    rise = set_ = None
    for k in range(n_scan):
        a0, a1 = alt[k], alt[(k + 1) % n_scan]
        if a0 < 0.0 <= a1 and rise is None:
            rise = 0.5 * (rho[k] + rho[(k + 1) % n_scan])
        if a0 >= 0.0 > a1 and set_ is None:
            set_ = 0.5 * (rho[k] + rho[(k + 1) % n_scan])
    if rise is None:  # polar day / night fallback
        rise = float(rho[i_noon] - 0.4)
    if set_ is None:
        set_ = float(rho[i_noon] + 0.4)
    return {"midnight": float(rho[i_mid]), "sunrise": float(rise),
            "noon": float(rho[i_noon]), "sunset": float(set_)}


# ---------------------------------------------------------------------
# Optical depth of a light ray from a point toward a star
# ---------------------------------------------------------------------
def _light_optical_depth(pos: np.ndarray, s_local: np.ndarray,
                         atmo: Atmosphere, n_light: int
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rayleigh / Mie optical depth from each point in `pos` (M,3) toward
    the (single, shared) star direction `s_local`, and a blocked-mask
    (star below the local horizon / planet in the way)."""
    Xs = pos @ s_local
    r2 = np.einsum("ij,ij->i", pos, pos)
    # Atmosphere-top intersection along +s.
    disc_a = Xs ** 2 - (r2 - atmo.r_atmo_m ** 2)
    u_top = -Xs + np.sqrt(np.maximum(disc_a, 0.0))
    # Planet intersection -> light blocked if it hits the ground first.
    disc_p = Xs ** 2 - (r2 - atmo.r_planet_m ** 2)
    blocked = (disc_p > 0.0) & (Xs < 0.0)
    u = np.linspace(0.0, 1.0, n_light)[:, None] * u_top[None, :]
    seg = np.where(n_light > 1, u_top / (n_light - 1), u_top)
    odr = np.zeros(pos.shape[0])
    odm = np.zeros(pos.shape[0])
    for k in range(n_light):
        P = pos + u[k][:, None] * s_local[None, :]
        h = np.linalg.norm(P, axis=1) - atmo.r_planet_m
        h = np.maximum(h, 0.0)
        odr += np.exp(-h / atmo.h_rayleigh_m) * seg
        odm += np.exp(-h / atmo.h_mie_m) * seg
    return odr, odm, blocked


# ---------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------
def _rayleigh_phase(mu: np.ndarray) -> np.ndarray:
    return 3.0 / (16.0 * math.pi) * (1.0 + mu ** 2)


def _mie_phase(mu: np.ndarray, g: float) -> np.ndarray:
    g2 = g * g
    return (3.0 / (8.0 * math.pi) * ((1.0 - g2) * (1.0 + mu ** 2))
            / ((2.0 + g2) * np.power(1.0 + g2 - 2.0 * g * mu, 1.5)))


def _airglow_spectrum(lam: np.ndarray) -> np.ndarray:
    """Airglow: dominant O I 557.7 nm green line + weaker 630 nm red +
    a faint continuum (the brightest natural night-sky term)."""
    g557 = np.exp(-0.5 * ((lam - 557.7) / 7.0) ** 2)
    g630 = 0.35 * np.exp(-0.5 * ((lam - 630.0) / 8.0) ** 2)
    return 0.25 + g557 + g630


def _to_screen(D: np.ndarray, right, up, fwd, hfov: float,
               W: int, H: int, p_top: float, p_bot: float):
    """Project system-frame directions D (...,3) to (col, row,
    in_front).  Inverse of the per-pixel ray construction."""
    z = D @ fwd
    x = D @ right
    y = D @ up
    yaw = np.arctan2(x, np.where(np.abs(z) < 1e-9, 1e-9, z))
    pit = np.arctan2(y, np.sqrt(x * x + z * z) + 1e-12)
    col = (yaw / hfov + 0.5) * (W - 1)
    row = (p_top - pit) / (p_top - p_bot) * (H - 1)
    return col, row, (z > 1e-6)


def render_sky(sys, planet, *, latitude_deg: float = 20.0,
               rot_phase: float = 0.0, t_orbit: float = 0.0,
               orbit_phase: float = 0.0,
               resolution: Tuple[int, int] = (640, 360),
               vfov_deg: float = 130.0, ground_frac: float = 0.30,
               eye_height_m: float = 2.0,
               n_view: int = 18, n_light: int = 8,
               n_lambda: int = N_LAMBDA_DEFAULT,
               cam_azimuth: Optional[float] = None,
               exposure: Optional[float] = None,
               max_dark_gain: float = 2000.0,
               bg_seed: int = 2026, show_bodies: bool = True
               ) -> Tuple[np.ndarray, float, List[StarSky],
np.ndarray, List[SkyBody]]:
    """Render one ground-to-sky frame.

    Returns (rgb HxWx3 uint8, exposure_used, star_sky_list,
    altitudes_rad, visible_sky_bodies).  A horizon camera looks toward
    `cam_azimuth` (default: the brightest star), with `ground_frac` of
    the vertical FoV below the horizon.  The world is an oblate
    spheroid; sibling planets, every moon, a procedural Milky-Way
    starfield and an airglow floor are composited in.
    """
    W, H = resolution
    lam = lambda_grid_nm(n_lambda)
    atmo = atmosphere_for(planet, lam)
    stars = star_sky_list(sys, planet, t_orbit, orbit_phase)
    prof = planet.habitability
    obl = float(getattr(prof, "obliquity_deg", 23.4)) if prof else 23.4
    east, north, zenith = _local_basis(obl, latitude_deg, rot_phase)

    # Oblate spheroid: spin axis (orbital frame) expressed locally.
    f_obl = oblateness_for(planet)
    n_world = np.array([math.sin(math.radians(obl)), 0.0,
                        math.cos(math.radians(obl))])
    spin_local = np.array([float(n_world @ east), float(n_world @ north),
                           float(n_world @ zenith)])
    spin_local = spin_local / (np.linalg.norm(spin_local) + 1e-15)

    # Star directions / brightness in the local (east, north, up) frame.
    s_local, s_spec, s_phasemu_axis, s_alt = [], [], [], []
    for st in stars:
        d = st.dir_inertial
        v = np.array([float(d @ east), float(d @ north), float(d @ zenith)])
        v = v / np.linalg.norm(v)
        s_local.append(v)
        s_alt.append(math.asin(np.clip(v[2], -1.0, 1.0)))
        pl = planck_spectral(lam, st.teff)
        pl = pl / np.trapezoid(pl, lam * 1e-9)  # unit visible power
        s_spec.append(st.flux_rel * pl)
    s_alt = np.array(s_alt)

    # Everything below is done in the LOCAL (east, north, zenith)
    # frame: zenith = +Z, so the observer is O = (0,0,r_eye) and the
    # star/body/background directions are projected onto (east, north,
    # zenith).  (Star directions are already local as `s_local`.)
    if cam_azimuth is None:
        bi = int(np.argmax([st.flux_rel for st in stars]))
        cam_azimuth = math.atan2(s_local[bi][0], s_local[bi][1])
    fwd = np.array([math.sin(cam_azimuth), math.cos(cam_azimuth), 0.0])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right = right / np.linalg.norm(right)
    up = np.array([0.0, 0.0, 1.0])

    vfov = math.radians(vfov_deg)
    hfov = vfov * (W / H)
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    yaw = (jj / (W - 1) - 0.5) * hfov
    # Keep pitch strictly inside (-90, 90) so the top row never wraps
    # past the zenith (which would render the planet as if from space).
    lim = math.radians(89.0)
    pitch_top = min(vfov * (1.0 - ground_frac), lim)
    pitch_bot = max(-vfov * ground_frac, -lim)
    pitch = pitch_top - (ii / (H - 1)) * (pitch_top - pitch_bot)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    vd = (cp * cy)[..., None] * fwd \
         + (cp * sy)[..., None] * right \
         + sp[..., None] * up
    vd = (vd / np.linalg.norm(vd, axis=-1, keepdims=True)).reshape(-1, 3)
    NP = vd.shape[0]

    # Observer at eye height above the *spheroid* surface along the
    # local zenith (+Z).  The oblate surface radius along zenith is
    # R_eq / sqrt(1 + (k^2+2k) (zhat.spin)^2); placing the eye there
    # (not at R_eq) keeps the observer on the ground at every rotation
    # so the equatorial bulge never spuriously occludes the sky.
    _k = 1.0 / max(1.0 - f_obl, 1e-6) - 1.0
    r_surf = atmo.r_planet_m / math.sqrt(
        1.0 + (_k * _k + 2.0 * _k) * float(spin_local[2]) ** 2)
    r_eye = r_surf + eye_height_m
    O = np.array([0.0, 0.0, r_eye])
    # Oblate spheroid ground / atmosphere top via the exact affine
    # sphere-mapping (Maclaurin flattening f_obl about spin_local).
    Ag, Bg, Cg = _ellipsoid_quadratic(O, vd, atmo.r_planet_m,
                                      f_obl, spin_local)
    disc_g = Bg ** 2 - 4.0 * Ag * Cg
    sq_g = np.sqrt(np.maximum(disc_g, 0.0))
    t_ground = np.where(disc_g >= 0.0,
                        (-Bg - sq_g) / (2.0 * Ag), np.inf)
    hit_ground = (disc_g >= 0.0) & (t_ground > 0.0)
    t_ground = np.where(hit_ground, t_ground, np.inf)
    Aa, Ba, Ca = _ellipsoid_quadratic(O, vd, atmo.r_atmo_m,
                                      f_obl, spin_local)
    t_atmo = (-Ba + np.sqrt(np.maximum(Ba ** 2 - 4.0 * Aa * Ca, 0.0))) \
             / (2.0 * Aa)
    t_end = np.minimum(t_ground, t_atmo)

    spec = np.zeros((NP, n_lambda))
    seg = t_end / n_view
    odr_v = np.zeros(NP)
    odm_v = np.zeros(NP)
    tmid = (np.arange(n_view) + 0.5) / n_view
    for k in range(n_view):
        P = O[None, :] + (tmid[k] * t_end)[:, None] * vd
        h = np.maximum(np.linalg.norm(P, axis=1) - atmo.r_planet_m, 0.0)
        dr = np.exp(-h / atmo.h_rayleigh_m)
        dm = np.exp(-h / atmo.h_mie_m)
        odr_v += dr * seg
        odm_v += dm * seg
        for si, st in enumerate(stars):
            sl = s_local[si]
            odr_l, odm_l, blk = _light_optical_depth(P, sl, atmo, n_light)
            tau = (atmo.beta_r[None, :] * (odr_v + odr_l)[:, None]
                   + (_MIE_EXT_FACT * atmo.beta_m)
                   * (odm_v + odm_l)[:, None])
            trans = np.exp(-np.clip(tau, 0.0, 60.0))
            mu = vd @ sl
            ph_r = _rayleigh_phase(mu)[:, None]
            ph_m = _mie_phase(mu, atmo.mie_g)[:, None]
            contrib = (atmo.beta_r[None, :] * (dr * seg)[:, None] * ph_r
                       + atmo.beta_m * (dm * seg)[:, None] * ph_m)
            vis = (~blk).astype(float)[:, None]
            spec += s_spec[si][None, :] * trans * contrib * vis

    # Ground (Lambert, lit by every star above its local horizon + sky
    # ambient), seen through the intervening atmospheric transmittance.
    if np.any(hit_ground):
        gmask = hit_ground
        Pg = O[None, :] + t_end[gmask, None] * vd[gmask]
        nrm = Pg / np.linalg.norm(Pg, axis=1, keepdims=True)
        gnd = np.zeros((int(gmask.sum()), n_lambda))
        for si, st in enumerate(stars):
            sl = s_local[si]
            ndl = np.clip(nrm @ sl, 0.0, 1.0)
            odr_g, odm_g, blk = _light_optical_depth(Pg, sl, atmo, n_light)
            tg = np.exp(-np.clip(
                atmo.beta_r[None, :] * odr_g[:, None]
                + _MIE_EXT_FACT * atmo.beta_m * odm_g[:, None], 0.0, 60.0))
            gnd += (s_spec[si][None, :] * tg
                    * (ndl * (~blk).astype(float))[:, None])
        amb = 0.10 * np.sum([s.flux_rel for s in stars]) \
              * np.ones(n_lambda)[None, :]
        gnd = (gnd + amb) * atmo.ground_albedo[None, :] / math.pi
        tau_v = (atmo.beta_r[None, :] * odr_v[gmask, None]
                 + _MIE_EXT_FACT * atmo.beta_m * odm_v[gmask, None])
        spec[gmask] += gnd * np.exp(-np.clip(tau_v, 0.0, 60.0))

    # Direct stellar disks (only where no ground occludes the line of
    # sight and the star is above the horizon).
    for si, st in enumerate(stars):
        if s_alt[si] <= 0.0:
            continue
        sl = s_local[si]
        mu = np.clip(vd @ sl, -1.0, 1.0)
        ang = np.arccos(mu)
        disk = (ang < stars[si].ang_radius_rad) & (~hit_ground)
        if np.any(disk):
            tau_v = (atmo.beta_r[None, :] * odr_v[disk, None]
                     + _MIE_EXT_FACT * atmo.beta_m * odm_v[disk, None])
            # Radiance ~ irradiance / solid angle of the stellar disk.
            omega = max(math.pi * stars[si].ang_radius_rad ** 2, 1e-9)
            spec[disk] += (s_spec[si][None, :] / omega
                           * np.exp(-np.clip(tau_v, 0.0, 60.0)))

    # ---- lit procedural background: airglow + Milky Way + stars ----
    lam_m = lam * 1e-9
    trans_v = np.exp(-np.clip(
        atmo.beta_r[None, :] * odr_v[:, None]
        + _MIE_EXT_FACT * atmo.beta_m * odm_v[:, None], 0.0, 60.0))
    # Orbital -> local rotation (rows are the local basis vectors).
    _Rml = np.stack([east, north, zenith])  # (3,3)
    alt_pix = vd[:, 2]  # local zenith comp
    sky_pix = (~hit_ground) & (alt_pix > -0.02)
    dirs0, bteff, bflux, pn0 = background_starfield(bg_seed)
    dirs = dirs0 @ _Rml.T  # -> local frame
    pn = _Rml @ np.asarray(pn0)  # galactic normal
    z_orb = _Rml @ np.array([0.0, 0.0, 1.0])  # ecliptic normal
    if np.any(sky_pix):
        airg = _airglow_spectrum(lam)
        van = np.clip(
            1.0 / np.sqrt(1.0 - 0.93 * (1.0 - alt_pix ** 2)),
            1.0, 2.2)
        bg = _AIRGLOW_LEVEL * van[:, None] * airg[None, :]
        zod = planck_spectral(lam, 5772.0)
        zod = zod / np.trapezoid(zod, lam_m)
        ecl = np.exp(-((vd @ z_orb) ** 2) / (2.0 * 0.35 ** 2))
        bg = bg + _ZODIACAL_LEVEL * ecl[:, None] * zod[None, :]
        mw = planck_spectral(lam, 4500.0)
        mw = mw / np.trapezoid(mw, lam_m)
        band = np.exp(-((vd @ pn) ** 2) / (2.0 * 0.16 ** 2))
        bg = bg + _MILKYWAY_LEVEL * band[:, None] * mw[None, :]
        spec[sky_pix] += bg[sky_pix] * trans_v[sky_pix]

    # discrete background stars (vectorised blackbody splat)
    col, row, infront = _to_screen(dirs, right, up, fwd, hfov, W, H,
                                   pitch_top, pitch_bot)
    balt = dirs[:, 2]
    ci = np.round(col).astype(int)
    ri = np.round(row).astype(int)
    vis = (infront & (balt > 0.01) & (ci >= 0) & (ci < W)
           & (ri >= 0) & (ri < H))
    if np.any(vis):
        flat = ri[vis] * W + ci[vis]
        keep = ~hit_ground[flat]
        if np.any(keep):
            flat = flat[keep]
            tv = bteff[vis][keep]
            fv = bflux[vis][keep]
            X = 1.0 / np.clip(balt[vis][keep], 0.08, 1.0)
            a_pl = 2.0 * H_PLANCK * C_LIGHT ** 2 / lam_m ** 5
            expo = (H_PLANCK * C_LIGHT
                    / (lam_m[None, :] * K_BOLTZ * tv[:, None]))
            pl = a_pl[None, :] / np.expm1(np.clip(expo, 1e-6, 700.0))
            pl = pl / np.trapezoid(pl, lam_m, axis=1)[:, None]
            ext = np.exp(-((550.0 / lam)[None, :] ** 4)
                         * 0.012 * X[:, None])
            np.add.at(spec, flat,
                      (_BG_STAR_GAIN * fv)[:, None] * pl * ext)

    # ---- sibling planets + every moon, as reflected-light disks ----
    vis_bodies: List[SkyBody] = []
    if show_bodies:
        px_per_rad = (H - 1) / (pitch_top - pitch_bot)
        for b in sky_bodies(sys, planet, lam, t_orbit, orbit_phase):
            D = _Rml @ b.dir_inertial  # -> local frame
            if float(D[2]) <= 0.01:
                continue
            c1, r1, infr = _to_screen(D[None, :], right, up, fwd, hfov,
                                      W, H, pitch_top, pitch_bot)
            if not bool(infr[0]):
                continue
            cc, rr = float(c1[0]), float(r1[0])
            rad_px = b.ang_radius_rad * px_per_rad
            rad_eff = max(rad_px, 0.7)
            omega = max(math.pi * b.ang_radius_rad ** 2, 1e-12)
            sdir = _Rml @ b.sub_star_dir  # -> local frame
            ssx, ssy = float(sdir @ right), -float(sdir @ up)
            sn = math.hypot(ssx, ssy) + 1e-9
            ssx, ssy = ssx / sn, ssy / sn
            i0, i1 = max(int(rr - rad_eff - 1), 0), \
                min(int(rr + rad_eff + 2), H)
            j0, j1 = max(int(cc - rad_eff - 1), 0), \
                min(int(cc + rad_eff + 2), W)
            if i1 <= i0 or j1 <= j0:
                continue
            jj2, ii2 = np.meshgrid(np.arange(j0, j1), np.arange(i0, i1))
            dxp, dyp = jj2 - cc, ii2 - rr
            rr2 = dxp ** 2 + dyp ** 2
            inside = rr2 <= rad_eff ** 2
            if not np.any(inside):
                continue
            if rad_px >= 0.7:
                proj = (dxp * ssx + dyp * ssy) / max(rad_eff, 1e-6)
                shade = np.clip(0.5 + 0.5 * proj
                                * (2.0 * b.phase_frac - 1.0)
                                + 1.1 * (b.phase_frac - 0.5),
                                0.03, 1.0)
                rad_disk = b.refl_spec / omega
            else:
                shade = np.exp(-rr2 / (2.0 * 0.55 ** 2))
                rad_disk = b.refl_spec / (math.pi
                                          * (rad_eff / px_per_rad) ** 2 + 1e-12)
            ridx = ii2[inside] * W + jj2[inside]
            tvb = np.exp(-np.clip(
                atmo.beta_r[None, :] * odr_v[ridx, None]
                + _MIE_EXT_FACT * atmo.beta_m * odm_v[ridx, None],
                0.0, 60.0))
            occ = (~hit_ground[ridx]).astype(float)
            np.add.at(spec, ridx,
                      rad_disk[None, :] * shade[inside][:, None]
                      * tvb * occ[:, None])
            b.screen_x = cc
            b.screen_y = rr
            b.screen_r = rad_eff
            b.altitude_deg = math.degrees(math.asin(
                float(np.clip(D[2], -1.0, 1.0))))
            vis_bodies.append(b)

    spec = spec.reshape(H, W, n_lambda)

    _, yb, _ = cie_xyz_bar(lam)
    dl = float(lam[1] - lam[0])
    Y = np.tensordot(spec, yb, axes=([-1], [0])) * dl
    e_auto = 0.55 / max(np.percentile(Y, 92.0), 1e-12)
    if exposure is None:
        # First (noon) call keys the scene exposure.
        exposure = e_auto
    else:
        # Reuse the caller's exposure for faithful relative day light,
        # but let dark frames brighten (dark adaptation / long
        # exposure) up to `max_dark_gain` so the starlit Milky-Way
        # night is visible instead of crushed to black.
        exposure = float(np.clip(e_auto, exposure,
                                 exposure * max_dark_gain))

    rgb = spectrum_to_srgb(spec, lam, exposure)
    return rgb, float(exposure), stars, s_alt, vis_bodies


# ---------------------------------------------------------------------
# Phase stills + day-cycle animation
# ---------------------------------------------------------------------
def _body_line(bodies) -> str:
    """Short overlay line for the brightest visible moons/planets."""
    if not bodies:
        return ""
    ranked = sorted(bodies, key=lambda b: -float(np.sum(b.refl_spec)))
    parts = []
    for b in ranked[:3]:
        ph = b.phase_frac
        tag = ("full" if ph > 0.92 else "new" if ph < 0.08
        else f"{ph:.0%}")
        parts.append(f"{b.name} ({b.kind[:1]}, {tag})")
    return "moons/planets: " + "  ".join(parts)


def _label(sys, planet, phase: str, stars, alts, bodies=None) -> str:
    locked = planet.habitability and not math.isfinite(
        getattr(planet.habitability, "solar_day_h", 24.0))
    nm = {"midnight": "Midnight", "sunrise": "Sunrise",
          "noon": "Noon", "sunset": "Sunset"}[phase]
    if locked:
        nm = {"midnight": "Antistellar (night side)",
              "sunrise": "Terminator (E)", "noon": "Substellar (noon)",
              "sunset": "Terminator (W)"}[phase]
    up = [f"{s.name} {math.degrees(a):+.0f}deg"
          for s, a in zip(stars, alts) if a > -0.05]
    sky = getattr(planet.habitability, "sky_description", "") \
        if planet.habitability else ""
    bl = _body_line(bodies)
    return (f"{sys.name} / {planet.name} -- {nm}\n"
            f"{'  |  '.join(up) if up else 'no star above horizon'}\n"
            f"{sky}" + (f"\n{bl}" if bl else ""))


def render_phases(sys, planet, out_dir: str, *,
                  latitude_deg: float = 20.0, orbit_phase: float = 0.0,
                  resolution: Tuple[int, int] = (900, 506),
                  **kw) -> Dict[str, str]:
    """Write midnight / sunrise / noon / sunset PNGs for one planet.

    Exposure is fixed from the 'noon' frame and reused for the other
    three so the relative light levels are physically faithful.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    phases = phase_rotations(sys, planet, latitude_deg, orbit_phase=orbit_phase)
    _, exp_noon, _, _, _ = render_sky(
        sys, planet, latitude_deg=latitude_deg, rot_phase=phases["noon"],
        orbit_phase=orbit_phase, resolution=resolution, **kw)

    written: Dict[str, str] = {}
    base = f"{sys.name}_{planet.name}".replace(" ", "_").replace("/", "-")
    for phase in ("midnight", "sunrise", "noon", "sunset"):
        rgb, _, stars, alts, bodies = render_sky(
            sys, planet, latitude_deg=latitude_deg,
            rot_phase=phases[phase], orbit_phase=orbit_phase,
            resolution=resolution, exposure=exp_noon, **kw)
        fig = plt.figure(figsize=(resolution[0] / 130, resolution[1] / 130),
                         dpi=130)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(rgb)
        ax.axis("off")
        ax.text(0.012, 0.975,
                _label(sys, planet, phase, stars, alts, bodies),
                transform=ax.transAxes, va="top", ha="left",
                color="white", fontsize=9,
                bbox=dict(boxstyle="round", fc="#000000AA", ec="none"))
        path = os.path.join(out_dir, f"{base}_{phase}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written[phase] = path
    return written


def animate_day(sys, planet, out_path: str, *, n_frames: int = 120,
                fps: int = 24, latitude_deg: float = 20.0,
                orbit_phase: float = 0.0,
                resolution: Tuple[int, int] = (480, 270),
                dpi=300,
                **kw) -> None:
    """Render a full solar-day MP4 (one planet rotation).  Raises if
    ffmpeg is unavailable -- the caller handles the skip, matching the
    repo's other animation helpers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    phases = phase_rotations(sys, planet, latitude_deg, orbit_phase=orbit_phase)
    _, exp_noon, _, _, _ = render_sky(
        sys, planet, latitude_deg=latitude_deg, rot_phase=phases["noon"],
        orbit_phase=orbit_phase, resolution=resolution, **kw)
    rots = np.linspace(0.0, 2.0 * math.pi, n_frames, endpoint=False) \
           + phases["midnight"]

    fig = plt.figure(figsize=(resolution[0] / dpi, resolution[1] / dpi),
                     dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(np.zeros((resolution[1], resolution[0], 3), np.uint8))
    txt = ax.text(0.012, 0.975, "", transform=ax.transAxes, va="top",
                  color="white", fontsize=8,
                  bbox=dict(boxstyle="round", fc="#000000AA", ec="none"))

    def update(fi):
        rgb, _, stars, alts, bodies = render_sky(
            sys, planet, latitude_deg=latitude_deg, rot_phase=float(rots[fi]),
            orbit_phase=orbit_phase, resolution=resolution,
            exposure=exp_noon, **kw)
        im.set_data(rgb)
        up = [f"{s.name} {math.degrees(a):+.0f}deg"
              for s, a in zip(stars, alts) if a > -0.05]
        bl = _body_line(bodies)
        txt.set_text(f"{sys.name} / {planet.name}\n"
                     f"{'  |  '.join(up) if up else 'night'}"
                     + (f"\n{bl}" if bl else ""))
        return [im, txt]

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000.0 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2600, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p",
                                      # Automatically pad width/height to the nearest even number
                                      "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
    anim.save(out_path, writer=writer)
    plt.close(fig)
