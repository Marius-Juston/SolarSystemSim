"""
moon_surface.py
---------------
Realistic moon-surface physics + a shaded-relief renderer for the
body-centric view.

`MoonSurface` mirrors the `HabitabilityProfile` / `profile_for_planet`
pattern: a derived profile attached to a `Moon` (the `Moon` dataclass
only gains one optional `.surface` field).  Every field comes from a
published scaling law so the rendered body is physically believable,
not arbitrary art:

* surface gravity / escape velocity        : Newtonian (textbook)
* ice fraction (rock vs ice split)         : bulk density + the moon's
                                             equilibrium temperature
                                             (T_eq = 278.5 S^1/4)
* tidal-heating resurfacing index          : Peale, Cassen & Reynolds
                                             1979, Science 203 (Io);
                                             ~ e^2 (a_Roche/a)^3 M_planet
* crater simple->complex transition D_sc   : Pike 1980; Melosh 1989 --
                                             D_sc ~ 1/g (Moon ~15-20 km;
                                             smaller on icy / low-g)
* crater density (retention)               : Neukum, Ivanov & Hartmann
                                             2001 production function;
                                             Gault 1970 saturation;
                                             reduced by resurfacing /
                                             atmosphere
* crater size-frequency exponent           : cumulative N(>D) ~ D^-b,
                                             b ~ 2 (Neukum/Hartmann)
* atmosphere retention                     : Jeans parameter
                                             (Catling & Kasting 2017)
* aeolian dune coverage                    : Lorenz et al. 2006;
                                             Burr et al. 2015, Nature
                                             (Titan saltation threshold)
* maximum relief                           : isostatic limit
                                             h ~ sigma / (rho g)
                                             (Melosh, Planetary Surface
                                             Processes; Jeffreys)

The renderer builds an equirectangular height + albedo texture ONCE per
(surface, seed) -- `lru_cache`d like `skyview._od_table` -- then samples
it per on-screen disk pixel, so an animation's frames (rendered in
independent pool workers) all share one identical surface and the cost
is the small disk bounding box, never the whole frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np

from goldilocks import backend as B
from goldilocks import noise as N
from goldilocks.habitability import (_jeans_parameter, M_EARTH_KG,
                                     R_EARTH_M, G_EARTH_SI, V_ESC_EARTH)
from goldilocks.planets import M_JUP_OVER_M_EARTH
from goldilocks.stellar import AU_M

xp = B.xp

_G_MOON_REF = 1.62      # m/s^2  (Moon: D_sc ~ 18 km anchor, Pike 1980)
_D_SC_MOON_KM = 18.0
_SIGMA_ROCK_PA = 1.0e8  # crustal yield strength (Melosh)
_SIGMA_ICE_PA = 3.0e7


# ---------------------------------------------------------------------
@dataclass
class MoonSurface:
    surface_gravity_ms2: float
    escape_velocity_kms: float
    t_eq_k: float
    is_icy: bool
    ice_fraction: float            # 0..1
    tidal_heating_index: float     # 0..1
    crater_transition_km: float    # simple -> complex diameter D_sc
    crater_density: float          # 0..1 retained-crater saturation
    crater_sfd_exponent: float     # cumulative N(>D) ~ D^-b
    has_atmosphere: bool
    dune_coverage: float           # 0..1
    max_relief_km: float
    surface_type: str              # rocky|regolith|icy|volcanic|
    #                                cryovolcanic|dune
    albedo: float
    summary: str = ""


# ---------------------------------------------------------------------
# Profile derivation  (Layer 3: pure physics, no rendering)
# ---------------------------------------------------------------------
def _moon_insolation_searth(planet, sys) -> float:
    """Bolometric insolation at the planet (=moon) in Earth units:
    sum_i L_i / d_i^2 (Lsun / AU^2; 1.0 for Earth)."""
    a_p = float(planet.semi_major_axis_au or 1.0)
    if planet.is_circumbinary():
        centre = np.array(sys.barycentre(), dtype=float)
    else:
        centre = np.array(sys.stars[planet.host_star_index].position,
                          dtype=float)
    p = centre + np.array([a_p, 0.0, 0.0])
    acc = 0.0
    for s in sys.stars:
        d2 = float(np.sum((p - np.array(s.position, dtype=float)) ** 2))
        acc += float(s.luminosity) / max(d2, 1e-12)
    return acc


def moon_surface_for(moon, planet, sys,
                     rng: np.random.Generator) -> MoonSurface:
    """Derive a `MoonSurface` for `moon` orbiting `planet` in `sys`."""
    M = float(moon.mass_me) * M_EARTH_KG
    R = float(moon.radius_re) * R_EARTH_M
    g = G_SI_safe(M, R)
    g_rel = (float(moon.mass_me) / max(float(moon.radius_re) ** 2, 1e-12))
    v_esc = V_ESC_EARTH * math.sqrt(max(float(moon.mass_me)
                                        / max(float(moon.radius_re),
                                              1e-9), 0.0))

    S = _moon_insolation_searth(planet, sys)
    # Bare-rock equilibrium temperature (no greenhouse): A ~ 0.1.
    t_eq = 278.5 * (max(S, 1e-9) ** 0.25) * (0.9 ** 0.25)

    rho = float(getattr(moon, "density_gcc", 2.5))
    # Ice fraction: low density and/or cold favour an ice-rich body.
    rho_ice = float(np.clip((2.6 - rho) / 1.4, 0.0, 1.0))
    cold = float(np.clip((180.0 - t_eq) / 120.0, 0.0, 1.0))
    ice_fraction = float(np.clip(0.55 * rho_ice + 0.45 * cold, 0.0, 1.0))
    is_icy = ice_fraction > 0.4

    # Tidal heating (Peale, Cassen & Reynolds 1979): satellite heating
    # rate ~ M_planet^5/2 R_moon^5 e^2 / a^15/2, normalised to Io = 1.
    a_m = max(float(moon.a_planet_au), 1e-9)
    e_m = float(moon.eccentricity)
    mp_mjup = float(planet.mass_me) / M_JUP_OVER_M_EARTH

    def _heat(mp, rm, e, a):
        return (max(mp, 1e-6) ** 2.5) * (max(rm, 1e-6) ** 5) \
            * (e ** 2) / (max(a, 1e-9) ** 7.5)

    raw_io = _heat(1.0, 0.286, 0.0041, 0.00282)  # Io reference
    tidal_heating_index = float(np.clip(
        _heat(mp_mjup, float(moon.radius_re), e_m, a_m) / raw_io,
        0.0, 1.0))

    # Simple -> complex transition diameter: D_sc ~ 1/g (Pike 1980);
    # icy crust transitions at a smaller diameter.
    g_eff = max(g, 1e-3)
    d_sc = _D_SC_MOON_KM * (_G_MOON_REF / g_eff)
    if is_icy:
        d_sc *= 0.5
    d_body_km = 2.0 * float(moon.radius_re) * R_EARTH_M / 1000.0
    crater_transition_km = float(np.clip(d_sc, 0.3, 0.6 * d_body_km))

    # A thick atmosphere needs strong Jeans retention AND a cold,
    # massive, volatile-rich (icy/organic) body that is not being
    # resurfaced/stripped by intense tidal heating -- i.e. Titan-like.
    lam_n2 = _jeans_parameter(28.0, max(t_eq, 30.0),
                              float(moon.mass_me), float(moon.radius_re))
    has_atmosphere = bool(lam_n2 > 110.0 and t_eq < 130.0
                          and float(moon.mass_me) > 0.018
                          and ice_fraction > 0.4
                          and tidal_heating_index < 0.3)

    # Crater retention: saturated, then erased by resurfacing/burial.
    crater_density = 1.0
    crater_density *= (1.0 - 0.85 * tidal_heating_index)
    if has_atmosphere:
        crater_density *= 0.45
    crater_density = float(np.clip(crater_density, 0.05, 1.0))
    crater_sfd_exponent = 2.0

    # Aeolian dunes: need an atmosphere + mobile sediment (organic/ice
    # fines).  Titan-like only.
    dune_coverage = float(np.clip(0.55 * ice_fraction, 0.0, 0.6)) \
        if has_atmosphere else 0.0

    # Isostatic maximum relief h ~ sigma / (rho g).
    sigma = _SIGMA_ICE_PA if is_icy else _SIGMA_ROCK_PA
    rho_si = max(rho, 0.3) * 1000.0
    max_relief_km = float(np.clip(
        sigma / (rho_si * max(g, 1e-3)) / 1000.0, 0.05, 60.0))

    # Surface classification + characteristic albedo.
    if tidal_heating_index > 0.30 and not is_icy:
        surface_type, albedo = "volcanic", 0.18      # Io-like
    elif tidal_heating_index > 0.15 and is_icy:
        surface_type, albedo = "cryovolcanic", 0.7   # Europa/Enceladus
    elif dune_coverage > 0.05:
        surface_type, albedo = "dune", 0.12          # Titan-like
    elif is_icy:
        surface_type, albedo = "icy", 0.6
    elif float(moon.mass_me) < 1e-4:
        surface_type, albedo = "regolith", 0.06      # dark captured
    else:
        surface_type, albedo = "rocky", 0.13
    albedo = float(np.clip(rng.normal(albedo, 0.02), 0.04, 0.92))

    s = MoonSurface(
        surface_gravity_ms2=g, escape_velocity_kms=v_esc, t_eq_k=t_eq,
        is_icy=is_icy, ice_fraction=ice_fraction,
        tidal_heating_index=tidal_heating_index,
        crater_transition_km=crater_transition_km,
        crater_density=crater_density,
        crater_sfd_exponent=crater_sfd_exponent,
        has_atmosphere=has_atmosphere, dune_coverage=dune_coverage,
        max_relief_km=max_relief_km, surface_type=surface_type,
        albedo=albedo)
    s.summary = (
        f"{moon.name}: {surface_type}, g={g:.2f} m/s^2, "
        f"T_eq~{t_eq:.0f} K, ice={ice_fraction:.2f}, "
        f"D_sc~{crater_transition_km:.1f} km, "
        f"crater_density={crater_density:.2f}, "
        f"relief<={max_relief_km:.1f} km"
        f"{', atmosphere' if has_atmosphere else ''}"
        f"{', dunes' if dune_coverage > 0.05 else ''}")
    return s


def G_SI_safe(M_kg: float, R_m: float) -> float:
    from goldilocks.stellar import G_SI
    return G_SI * max(M_kg, 1e3) / max(R_m, 1.0) ** 2


# ---------------------------------------------------------------------
# Equirectangular surface texture  (cached -- _od_table pattern)
# ---------------------------------------------------------------------
def _sig(s: MoonSurface) -> Tuple:
    return (round(s.crater_transition_km, 3), round(s.crater_density, 3),
            round(s.crater_sfd_exponent, 3), round(s.max_relief_km, 3),
            round(s.dune_coverage, 3), round(s.ice_fraction, 3),
            s.surface_type, round(s.albedo, 4))


@lru_cache(maxsize=24)
def _surface_tex(sig: Tuple, seed: int, n_lat: int = 320,
                 n_lon: int = 640):
    """Build (height (n_lat,n_lon) in km, albedo (n_lat,n_lon)) on the
    active backend.  Equirectangular: lat in [-pi/2, pi/2], lon in
    [0, 2pi).  Deterministic for (sig, seed)."""
    (d_sc, c_den, c_b, relief, dunes, ice, stype, alb) = sig
    rng = np.random.default_rng(seed)
    lat = np.linspace(-math.pi / 2, math.pi / 2, n_lat)
    lon = np.linspace(0.0, 2.0 * math.pi, n_lon, endpoint=False)
    LON, LAT = np.meshgrid(lon, lat)
    # Map lon/lat onto a noise plane (lon scaled by cos lat to limit
    # polar pinching).
    px = (LON * np.cos(LAT)) * 1.4
    py = LAT * 2.6
    base = B.asnumpy(N.fbm(B.asarray(px * 1.0), B.asarray(py * 1.0),
                           seed=seed + 1, octaves=5))
    height = 0.5 * relief * base  # km, low-frequency relief

    # Crater field: cumulative N(>D) ~ D^-b (Neukum/Hartmann).  Diameters
    # are angular (fraction of pi); biggest ~ a basin, smallest resolved
    # ~ 0.012.  Each crater is stamped only inside its own lon/lat index
    # window (the disk-bbox trick) so cost is O(sum of crater areas), not
    # O(n_craters * whole map).
    d_max, d_min = 0.9, 0.012
    n_big = int(round(70 * c_den))
    diam, d_hi = [], d_max
    while d_hi > d_min and len(diam) < 600:
        d_lo = d_hi / 1.7
        cnt = int(round(n_big * ((d_lo / d_max) ** (-c_b)
                                 - (d_hi / d_max) ** (-c_b))))
        for _ in range(min(max(cnt, 1 if d_hi < d_max else 0), 220)):
            diam.append(rng.uniform(d_lo, d_hi))
        d_hi = d_lo
    dlat = math.pi / (n_lat - 1)
    dlon = 2.0 * math.pi / n_lon
    for D in diam:
        la = math.asin(rng.uniform(-1.0, 1.0))
        lo = rng.uniform(0.0, 2.0 * math.pi)
        rad = 0.5 * D
        reach = rad * 1.3
        ic = (la + math.pi / 2) / math.pi * (n_lat - 1)
        hl = int(math.ceil(reach / dlat)) + 1
        i_lo = max(int(ic) - hl, 0)
        i_hi = min(int(ic) + hl + 1, n_lat)
        lon_hw = reach / max(math.cos(la), 0.12)
        if lon_hw >= math.pi:                       # spans all longitudes
            jcols = np.arange(n_lon)
        else:
            hj = int(math.ceil(lon_hw / dlon)) + 1
            jc = int(round(lo / dlon))
            jcols = (np.arange(jc - hj, jc + hj + 1)) % n_lon
        sub_lat = lat[i_lo:i_hi][:, None]
        sub_lon = lon[jcols][None, :]
        dl = np.abs(((sub_lon - lo + math.pi) % (2 * math.pi)) - math.pi)
        ca = (np.sin(sub_lat) * math.sin(la)
              + np.cos(sub_lat) * math.cos(la) * np.cos(dl))
        ang = np.arccos(np.clip(ca, -1.0, 1.0))
        t = ang / max(rad, 1e-6)
        depth = 0.06 * relief * (1.0 + 4.0 * rad)
        floor = -(1.0 - np.clip(t, 0.0, 1.0) ** 2)
        if (D * 30.0) > d_sc:                       # complex morphology
            floor = np.where(t < 0.45, -0.7, floor)
            floor = floor + np.where(t < 0.18,
                                     0.5 * (0.18 - t) / 0.18, 0.0)
        rim = np.exp(-((t - 1.0) / 0.16) ** 2) * 0.6
        prof = np.where(t <= 1.0, floor, 0.0) + rim
        m = t < 1.3
        rows = np.arange(i_lo, i_hi)
        sel = np.ix_(rows, jcols)
        block = height[sel]
        height[sel] = np.where(m, block + depth * prof, block)

    # Aeolian dune bands: low transverse ripples in equatorial latitudes.
    if dunes > 0.01:
        band = np.exp(-(LAT / math.radians(35.0)) ** 2)
        ripple = np.sin(LON * 60.0 + 6.0 * base) \
            * np.sin(LAT * 40.0)
        height += dunes * 0.04 * relief * band * ripple

    # Albedo map: surface-type base + mottling; bright polar ice caps
    # for cold/icy worlds.
    mott = B.asnumpy(N.fbm(B.asarray(px * 2.0), B.asarray(py * 2.0),
                           seed=seed + 9, octaves=4))
    albedo = np.clip(alb * (1.0 + 0.35 * mott), 0.02, 0.95)
    if stype in ("volcanic",):
        lava = B.asnumpy(N.fbm(B.asarray(px * 3.0), B.asarray(py * 3.0),
                               seed=seed + 4, octaves=3))
        albedo = np.where(lava > 0.35, albedo * 0.5, albedo)
    if ice > 0.35 or stype in ("icy", "cryovolcanic"):
        cap = np.clip((np.abs(LAT) - math.radians(58.0))
                      / math.radians(32.0), 0.0, 1.0)
        albedo = np.clip(albedo + 0.55 * cap * (0.5 + 0.5 * ice),
                         0.02, 0.97)
    return B.asarray(height), B.asarray(albedo)


# ---------------------------------------------------------------------
# Disk renderer  (Layer 5: composites into the bodyview `spec` buffer)
# ---------------------------------------------------------------------
def render_moon_disk(spec, surface: MoonSurface, moon, lam_nm,
                     *, center_px: Tuple[float, float],
                     radius_px: float, W: int, H: int,
                     cam_right, cam_up, cam_fwd,
                     sun_dirs_local: List[np.ndarray],
                     sun_specs: List[np.ndarray],
                     spin_axis_local: np.ndarray,
                     rot_phase: float,
                     illum_scale: float = 1.0,
                     illum_spec_override: Optional[np.ndarray] = None
                     ) -> None:
    """Shaded-relief render of `moon` as a sphere filling a screen disk.

    `spec` is the (H*W, n_lambda) bodyview radiance buffer; this
    `scatter_add`s the lit, textured hemisphere into it (the same
    compositing mechanism as the `skyview.sky_bodies` splat).  All
    geometry is in the camera-local (right, up, fwd) frame.
    """
    cc, rr = center_px
    rad = max(float(radius_px), 1.5)
    i0, i1 = max(int(rr - rad - 1), 0), min(int(rr + rad + 2), H)
    j0, j1 = max(int(cc - rad - 1), 0), min(int(cc + rad + 2), W)
    if i1 <= i0 or j1 <= j0:
        return
    jj, ii = xp.meshgrid(xp.arange(j0, j1), xp.arange(i0, i1))
    nx = (jj - cc) / rad
    ny = -(ii - rr) / rad
    rho2 = nx ** 2 + ny ** 2
    inside = rho2 <= 1.0
    if not bool(xp.any(inside)):
        return
    nz = xp.sqrt(xp.clip(1.0 - rho2, 0.0, 1.0))

    cr = B.asarray(np.asarray(cam_right, float))
    cu = B.asarray(np.asarray(cam_up, float))
    cf = B.asarray(np.asarray(cam_fwd, float))
    # Outward surface normal (camera frame): hemisphere faces observer
    # (-fwd).  n = nx*right + ny*up - nz*fwd.
    n = (nx[..., None] * cr + ny[..., None] * cu
         - nz[..., None] * cf)
    n = n / (xp.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)

    # Body-fixed lon/lat for texture lookup.
    pole = B.asarray(np.asarray(spin_axis_local, float))
    pole = pole / (xp.linalg.norm(pole) + 1e-12)
    e1 = xp.cross(pole, cf)
    if float(xp.linalg.norm(e1)) < 1e-6:
        e1 = xp.cross(pole, cr)
    e1 = e1 / (xp.linalg.norm(e1) + 1e-12)
    e2 = xp.cross(pole, e1)
    nlat = xp.arcsin(xp.clip(xp.sum(n * pole, axis=-1), -1.0, 1.0))
    nlon = xp.arctan2(xp.sum(n * e2, axis=-1),
                      xp.sum(n * e1, axis=-1)) + float(rot_phase)

    hgt, alb = _surface_tex(_sig(surface),
                            seed=(abs(hash(moon.name)) % 100000) + 1)
    NLAT, NLON = hgt.shape
    fi = (nlat + math.pi / 2) / math.pi * (NLAT - 1)
    fj = (nlon % (2.0 * math.pi)) / (2.0 * math.pi) * NLON
    ia = xp.clip(xp.floor(fi).astype(xp.int64), 0, NLAT - 2)
    ja = (xp.floor(fj).astype(xp.int64)) % NLON
    jb = (ja + 1) % NLON
    ta = fi - ia
    tb = fj - xp.floor(fj)

    def _samp(T):
        return ((T[ia, ja] * (1 - tb) + T[ia, jb] * tb) * (1 - ta)
                + (T[ia + 1, ja] * (1 - tb) + T[ia + 1, jb] * tb) * ta)

    albed = _samp(alb)
    # Relief shading: perturb the normal by a *centred* height gradient
    # (smoother than a one-sided difference -> no lattice speckle).
    jm = (ja - 1) % NLON
    iu = xp.clip(ia + 1, 0, NLAT - 1)
    idn = xp.clip(ia - 1, 0, NLAT - 1)
    dhx = 0.5 * (hgt[ia, jb] - hgt[ia, jm])
    dhy = 0.5 * (hgt[iu, ja] - hgt[idn, ja])
    relief_k = max(surface.max_relief_km, 1e-3)
    bump = 0.9 / relief_k
    n = n - bump * (dhx[..., None] * cr + dhy[..., None] * cu)
    n = n / (xp.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)

    nlam = len(lam_nm)
    acc = xp.zeros(n.shape[:-1] + (nlam,))
    for sd, sp in zip(sun_dirs_local, sun_specs):
        sdl = B.asarray(np.asarray(sd, float))
        ndl = xp.clip(xp.sum(n * sdl, axis=-1), 0.0, 1.0)
        src = (B.asarray(np.asarray(illum_spec_override, float))
               if illum_spec_override is not None
               else B.asarray(np.asarray(sp, float)))
        acc = acc + ndl[..., None] * src[None, None, :]
    acc = acc * (albed[..., None] / math.pi) * float(illum_scale)

    sel = inside
    ridx = (ii[sel] * W + jj[sel]).astype(xp.int64)
    B.scatter_add(spec, ridx, acc[sel])
