"""
habitability.py
----------------
Per-planet physical & habitability profile for the random solar-system
generator.  Every field is derived from a published scaling law so the
numbers feed a believable "what would it be like to live there?" picture
while still being interesting (heavy/light gravity, long/short days,
extreme seasons, banded storm worlds, exotic sky colours, weak/strong
magnetic shielding).

Grounding references
--------------------
* Surface gravity / escape velocity        : Newtonian (textbook)
* Equilibrium temperature  T_eq = 278.5 S^1/4 (1-A)^1/4
                                            : Catling & Kasting 2017, eq. 3.9
* Tidal de-spin timescale                  : Gladman et al. 1996, Icarus 122;
                                              Peale 1977
* Solar vs sidereal day                    : kinematics (textbook)
* Obliquity stabilisation by a large moon  : Laskar, Joutel & Robutel 1993,
                                              Nature 361, 615
* Atmospheric escape (Jeans parameter)     : Catling & Kasting 2017, ch. 5
* Atmospheric scale height H = kT/(mu g)    : hydrostatic equilibrium
* Dynamo / magnetic-moment scaling          : Olson & Christensen 2006,
                                              EPSL 250; Zuluaga et al. 2013
* Hadley/jet & superrotation regime         : Held & Hou 1980, JAS 37;
                                              Showman & Polvani 2011, JAS 68
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from goldilocks.planets import is_gas_giant
from goldilocks.stellar import G_SI, M_SUN_KG, AU_M

# Physical constants (SI) ------------------------------------------------
K_B = 1.380649e-23  # J/K
M_H = 1.6735575e-27  # kg (hydrogen atom)
G_EARTH_SI = 9.80665  # m/s^2
M_EARTH_KG = 5.972168e24
R_EARTH_M = 6.378137e6
V_ESC_EARTH = 11.186  # km/s
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_YEAR = 3.15576e7
GYR_S = 3.15576e16


@dataclass
class HabitabilityProfile:
    # Bulk / gravity
    surface_gravity_g: float = 1.0  # in Earth g
    surface_gravity_ms2: float = G_EARTH_SI
    escape_velocity_kms: float = V_ESC_EARTH
    # Radiation / temperature
    mean_insolation_searth: float = 1.0  # relative to Earth (S0)
    bond_albedo: float = 0.30
    t_eq_k: float = 255.0
    t_surface_k: float = 288.0
    # Rotation / day-night
    rotation_period_h: float = 24.0
    tidally_locked: bool = False
    sidereal_day_h: float = 24.0
    solar_day_h: float = 24.0
    day_night_note: str = ""
    # Axial tilt / seasons
    obliquity_deg: float = 23.4
    seasonality: float = 0.40  # 0..1, ~ sin(obliquity)
    chaotic_obliquity: bool = False
    # Magnetosphere
    magnetic_moment_rel: float = 1.0  # relative to Earth
    magnetosphere: str = "moderate"
    # Atmosphere
    dominant_gas: str = "N2/O2"
    mean_molecular_weight: float = 29.0
    scale_height_km: float = 8.5
    surface_pressure_bar: float = 1.0
    sky_color_hex: str = "#7FB2FF"
    sky_description: str = "clear pale-blue Rayleigh sky"
    # Weather
    wind_regime: str = "Earth-like Hadley + mid-latitude jets"
    storm_index: float = 0.30  # 0..1
    # Overall
    in_phz: bool = False
    biosphere_score: float = 0.0  # 0..1
    summary: str = ""


# ---------------------------------------------------------------------
# Flux
# ---------------------------------------------------------------------
def _mean_bolometric_flux_searth(planet, sys,
                                 n_phase: int = 16,
                                 n_theta: int = 12) -> float:
    """Orbit-averaged bolometric insolation in Earth units (S0).

    Sum_i L_i / d_i^2 (Lsun / AU^2; = 1.0 for Earth) averaged over the
    stellar phase and the planet's azimuth.
    """
    a_p = planet.semi_major_axis_au
    multi = len(sys.stars) >= 2 and bool(sys.stellar_orbits)
    P = sys.stellar_orbit_period(0) if multi else 1.0
    times = np.linspace(0.0, P, n_phase, endpoint=False) if multi else [0.0]
    thetas = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)

    acc, cnt = 0.0, 0
    for t in times:
        sys._update_stellar_positions(t)
        star_pos = np.array([s.position for s in sys.stars])
        star_L = np.array([s.luminosity for s in sys.stars])
        if planet.is_circumbinary():
            centre = sys.barycentre()
        else:
            centre = np.array(sys.stars[planet.host_star_index].position)
        for th in thetas:
            p = centre + a_p * np.array([math.cos(th), math.sin(th), 0.0])
            d2 = np.sum((p[None, :] - star_pos) ** 2, axis=1)
            acc += float(np.sum(star_L / np.maximum(d2, 1e-12)))
            cnt += 1
    sys._update_stellar_positions(0.0)
    return acc / max(cnt, 1)


# ---------------------------------------------------------------------
# Tidal de-spin (Gladman et al. 1996)
# ---------------------------------------------------------------------
def _tidal_lock(planet, host_mass_msun: float, w0_rad_s: float,
                age_gyr: float, Q: float = 100.0, k2: float = 0.3) -> bool:
    """Return True if the planet's spin is locked within the system age.

    t_despin ~ (w a^6 I Q) / (3 G m*^2 k2 R^5),  I = 0.33 M R^2
    (Gladman et al. 1996, Icarus 122, 166; Peale 1977).
    """
    a = planet.semi_major_axis_au * AU_M
    R = planet.radius_re * R_EARTH_M
    M = planet.mass_me * M_EARTH_KG
    Ms = host_mass_msun * M_SUN_KG
    I = 0.33 * M * R * R
    t_despin = (w0_rad_s * a ** 6 * I * Q) / (3.0 * G_SI * Ms * Ms * k2 * R ** 5)
    return t_despin < age_gyr * GYR_S


# ---------------------------------------------------------------------
# Atmosphere inference
# ---------------------------------------------------------------------
def _jeans_parameter(mu: float, T_k: float,
                     mass_me: float, radius_re: float) -> float:
    """lambda = G M mu m_H / (k T R).  lambda >~ 30 => gas retained."""
    M = mass_me * M_EARTH_KG
    R = radius_re * R_EARTH_M
    return (G_SI * M * mu * M_H) / (K_B * max(T_k, 1.0) * R)


def _infer_atmosphere(planet, t_eq_k: float, g_si: float):
    """Return (dominant_gas, mu, P_surface_bar, sky_hex, sky_desc)."""
    if is_gas_giant(planet):
        mu = 2.3  # H2/He
        return ("H2/He (+CH4, NH3)", mu, 1e4,
                "#D9C7A3", "deep banded H2/He haze, ammonia/methane tinted")

    lam_h2 = _jeans_parameter(2.0, t_eq_k, planet.mass_me, planet.radius_re)
    lam_n2 = _jeans_parameter(28.0, t_eq_k, planet.mass_me, planet.radius_re)

    if lam_h2 > 30.0 and t_eq_k < 250.0:
        mu = 4.0
        return ("H2/He-rich (sub-Neptune)", mu, 5e2,
                "#A9C6D8", "thick hydrogen haze, washed-out white sky")
    if lam_n2 < 6.0:
        # Light envelope largely lost -> tenuous CO2 / airless
        mu = 44.0
        return ("trace CO2 (near-airless)", mu, 0.01,
                "#1A1326", "near-vacuum: black sky, sharp shadows")

    # Earth/Venus/Mars-like regime: pressure scales with gravity + warmth
    p_bar = float(np.clip(1.0 * (g_si / G_EARTH_SI) ** 1.5
                          * (t_eq_k / 255.0) ** 2, 0.1, 90.0))
    if t_eq_k > 320.0 and p_bar > 5.0:
        return ("CO2 (runaway-greenhouse)", 44.0, p_bar,
                "#E8C16A", "dense CO2 overcast, hazy yellow-orange sky")
    if t_eq_k < 200.0:
        return ("N2/CH4 (cold, reducing)", 28.0, p_bar,
                "#C8772E", "orange organic-haze sky (Titan-like)")
    # Temperate N2/O2/CO2: Rayleigh-blue, deeper blue at higher pressure
    mu = 29.0
    deep = float(np.clip(p_bar, 0.3, 3.0))
    r = int(np.interp(deep, [0.3, 1.0, 3.0], [150, 127, 70]))
    g = int(np.interp(deep, [0.3, 1.0, 3.0], [190, 178, 130]))
    b = int(np.interp(deep, [0.3, 1.0, 3.0], [235, 255, 240]))
    return ("N2/O2 (temperate)", mu, p_bar, f"#{r:02X}{g:02X}{b:02X}",
            "clear Rayleigh-scattered blue sky")


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def profile_for_planet(planet, sys,
                       rng: np.random.Generator,
                       in_phz: bool = False,
                       age_gyr: float = 5.0) -> HabitabilityProfile:
    M_me, R_re = planet.mass_me, planet.radius_re
    giant = is_gas_giant(planet)

    # ----- gravity -----
    g_g = M_me / (R_re ** 2)
    g_si = g_g * G_EARTH_SI
    v_esc = V_ESC_EARTH * math.sqrt(M_me / R_re)

    # ----- insolation / temperature -----
    S = _mean_bolometric_flux_searth(planet, sys)
    A = float(np.clip(rng.normal(0.30, 0.08), 0.05, 0.7))
    t_eq = 278.5 * (S ** 0.25) * ((1.0 - A) ** 0.25)
    if giant:
        host_mass = sum(s.mass for s in sys.stars)
    elif planet.is_circumbinary():
        host_mass = sum(s.mass for s in sys.stars)
    else:
        host_mass = sys.stars[planet.host_star_index].mass

    # ----- rotation -----
    if giant:
        rot_h = float(rng.uniform(6.0, 20.0))
    else:
        rot_h = float(np.clip(10 ** rng.normal(math.log10(24.0), 0.45),
                              4.0, 600.0))
    w0 = 2.0 * math.pi / (rot_h * SECONDS_PER_HOUR)
    locked = (not giant) and _tidal_lock(planet, host_mass, w0, age_gyr)

    P_orb_h = (math.sqrt(planet.semi_major_axis_au ** 3 / max(host_mass, 1e-6))
               * SECONDS_PER_YEAR / SECONDS_PER_HOUR)
    if locked:
        sidereal_h = P_orb_h
        solar_h = float("inf")
        dn_note = ("1:1 spin-orbit lock: permanent day side / night side, "
                   "habitable terminator ring")
    else:
        sidereal_h = rot_h
        inv = 1.0 / sidereal_h - 1.0 / P_orb_h
        solar_h = abs(1.0 / inv) if abs(inv) > 1e-12 else float("inf")
        dn_note = (f"sidereal day {sidereal_h:.1f} h -> "
                   f"solar day {solar_h:.1f} h")

    # ----- obliquity / seasons -----
    obliq = float(np.clip(abs(rng.normal(25.0, 22.0)), 0.0, 89.0))
    has_big_moon = any(mn.mass_me > 0.005 and mn.kind == "regular"
                       for mn in planet.moons)
    chaotic = (not giant) and (not has_big_moon) and obliq > 10.0
    seasonality = math.sin(math.radians(obliq))

    # ----- atmosphere -----
    gas, mu, p_bar, sky_hex, sky_desc = _infer_atmosphere(planet, t_eq, g_si)
    H_km = (K_B * max(t_eq, 50.0) / (mu * M_H * max(g_si, 0.1))) / 1000.0
    # crude greenhouse offset ~ proportional to log surface pressure
    dT_gh = 33.0 * math.log10(max(p_bar, 0.01) / 1.0 + 1.0) \
        if not giant else 0.0
    t_surf = t_eq + max(dT_gh, 0.0)

    # ----- magnetosphere (Olson & Christensen 2006-style scaling) -----
    # dipole moment ~ rho_c^1/2 (rotation)~ ; relative proxy with a
    # rotation and core-mass dependence, normalised to Earth = 1.
    rot_factor = (24.0 / max(rot_h, 1.0)) ** 0.3
    mag_rel = (M_me ** 0.6) * rot_factor
    if locked:
        mag_rel *= 0.25
    if mag_rel > 1.5:
        mag = "strong (broad shielding magnetosphere)"
    elif mag_rel > 0.3:
        mag = "moderate (Earth-like shielding)"
    else:
        mag = "weak (poor atmospheric/biological shielding)"

    # ----- winds / storms (Held-Hou + Showman-Polvani regime) -----
    # Rossby-like: fast rotation + strong insolation gradient -> banded
    # jets & frequent storms; slow rotation -> superrotation/Hadley.
    storm = float(np.clip(0.12 * (S ** 0.5)
                          + 0.22 * (24.0 / max(rot_h, 1.0)) ** 0.5
                          + 0.18 * seasonality - (0.25 if locked else 0.0),
                          0.0, 1.0))
    if giant:
        wind = "deep zonal jet bands, century-scale anticyclonic storms"
        storm = float(np.clip(storm + 0.3, 0.0, 1.0))
    elif rot_h < 12.0:
        wind = "many narrow zonal jets, frequent cyclones (fast rotator)"
    elif rot_h > 200.0 or locked:
        wind = "global superrotation, weak Coriolis, day-night Hadley flow"
    else:
        wind = "Earth-like Hadley cells + mid-latitude jet streams"

    # ----- biosphere / quality-of-life score -----
    g_comfort = math.exp(-((g_g - 1.0) ** 2) / (2 * 0.35 ** 2))
    t_comfort = math.exp(-((t_surf - 288.0) ** 2) / (2 * 30.0 ** 2))
    day_comfort = math.exp(-((math.log10(max(solar_h, 1e-3))
                              - math.log10(24.0)) ** 2) / (2 * 0.5 ** 2)) \
        if math.isfinite(solar_h) else 0.15
    shield = float(np.clip(mag_rel / 1.0, 0.0, 1.0))
    calm = 1.0 - storm
    bio = 0.0 if giant else float(np.clip(
        (0.30 * t_comfort + 0.20 * g_comfort + 0.15 * day_comfort
         + 0.15 * shield + 0.10 * calm + 0.10 * (1.0 if in_phz else 0.0)),
        0.0, 1.0))

    prof = HabitabilityProfile(
        surface_gravity_g=g_g, surface_gravity_ms2=g_si,
        escape_velocity_kms=v_esc,
        mean_insolation_searth=S, bond_albedo=A,
        t_eq_k=t_eq, t_surface_k=t_surf,
        rotation_period_h=rot_h, tidally_locked=locked,
        sidereal_day_h=sidereal_h, solar_day_h=solar_h,
        day_night_note=dn_note,
        obliquity_deg=obliq, seasonality=seasonality,
        chaotic_obliquity=chaotic,
        magnetic_moment_rel=mag_rel, magnetosphere=mag,
        dominant_gas=gas, mean_molecular_weight=mu,
        scale_height_km=H_km, surface_pressure_bar=p_bar,
        sky_color_hex=sky_hex, sky_description=sky_desc,
        wind_regime=wind, storm_index=storm,
        in_phz=in_phz, biosphere_score=bio)
    prof.summary = _summarize(planet, prof)
    return prof


def _summarize(planet, p: HabitabilityProfile) -> str:
    kind = "gas/ice giant" if is_gas_giant(planet) else "terrestrial"
    sol = ("permanent (locked)" if not math.isfinite(p.solar_day_h)
           else f"{p.solar_day_h:.1f} h")
    return (
        f"{planet.name} ({kind}, {len(planet.moons)} moons)\n"
        f"  gravity      : {p.surface_gravity_g:.2f} g "
        f"({p.surface_gravity_ms2:.1f} m/s^2), v_esc {p.escape_velocity_kms:.1f} km/s\n"
        f"  insolation   : {p.mean_insolation_searth:.2f} S_Earth, "
        f"T_eq {p.t_eq_k:.0f} K, T_surf ~{p.t_surface_k:.0f} K\n"
        f"  day/night    : {p.day_night_note}; solar day {sol}\n"
        f"  axial tilt   : {p.obliquity_deg:.0f} deg "
        f"(seasonality {p.seasonality:.2f}"
        f"{', CHAOTIC' if p.chaotic_obliquity else ''})\n"
        f"  atmosphere   : {p.dominant_gas}, P~{p.surface_pressure_bar:.2g} bar, "
        f"H {p.scale_height_km:.1f} km; {p.sky_description}\n"
        f"  magnetosphere: {p.magnetosphere} (M~{p.magnetic_moment_rel:.2f} Earth)\n"
        f"  weather      : {p.wind_regime}; storm index {p.storm_index:.2f}\n"
        f"  biosphere    : {p.biosphere_score:.2f}"
        f"{'  [in PHZ]' if p.in_phz else ''}")
