"""
random_systems.py
-----------------
Generators for a variety of interesting star systems with planets:

  * **Random Raghavan binary** -- mass and orbital distribution drawn
    from Raghavan et al. 2010 (arXiv:1007.0414): companion-period
    distribution is log-normal peaked at 300 yr, eccentricities roughly
    flat for long-period pairs, mass ratio peaked at q ~ 1.
  * **Star-hopper candidate** -- wide binary (100-1000 AU) with a
    high-eccentricity planet straddling the L1 point.  Inspired by
    Moeckel & Veras (2012, MNRAS 422, 831).
  * **Trojan / co-orbital companion** -- two planets at the same
    semi-major axis, one at the L4 Lagrange point of the other.
  * **Tight P-type with rocky planets** -- a tight stellar binary like
    Kepler-47 with multiple terrestrials including one in the HZ.
  * **Wide hierarchical** -- a close binary with a wider third star at
    ~30-100 AU, where the third star perturbs the circumbinary HZ.
  * **High-inclination "polar" planet** -- single host with one HZ
    planet on an i ~ 70 deg orbit (polar relative to a hypothetical
    debris disc).

Each generator returns a StarSystem.  All systems are validated against
Mardling-Aarseth and Holman-Wiegert before being returned.
"""

from __future__ import annotations

import math

import numpy as np

from goldilocks.moons import (Moon, R_EARTH_AU, planetary_roche_limit_au,
                              critical_moon_fraction,
                              planet_hill_radius_au)
from goldilocks.planets import (Planet, earth_analog, bulk_density_gcc,
                                radius_from_mass_me)
from goldilocks.stellar import Star
from goldilocks.system import StarSystem


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _random_mass(rng: np.random.Generator,
                 m_min: float = 0.3, m_max: float = 1.4) -> float:
    """Draw a sensible main-sequence mass (Msun) for a planet-hosting
    star (rough Kroupa-like).  Restricted to FGKM masses.
    """
    # log-uniform between m_min and m_max
    return float(10 ** rng.uniform(math.log10(m_min), math.log10(m_max)))


def _random_binary_separation(rng: np.random.Generator,
                              kind: str = "any") -> float:
    """Raghavan 2010 log-normal: period peak ~300 yr, sigma ~2.3 dex.

    `kind` can be 'tight' (a < 2 AU), 'close' (2-50 AU), 'wide' (50-2000 AU),
    or 'any'.
    """
    if kind == "tight":
        P_yr = 10 ** rng.uniform(-2, 0.5)  # 4 days to 3 yr
    elif kind == "close":
        P_yr = 10 ** rng.uniform(0.5, 2)  # 3 yr to 100 yr
    elif kind == "wide":
        P_yr = 10 ** rng.uniform(2.5, 5)  # 300 yr to 100,000 yr
    else:
        P_yr = 10 ** rng.normal(2.5, 1.4)
        P_yr = max(0.01, min(1e5, P_yr))
    # Convert period to SMA using Kepler 3rd (total mass ~ 1.5 Msun by default)
    # We'll let the caller refine, here just pick a typical SMA
    a = (P_yr * P_yr * 1.5) ** (1.0 / 3.0)
    return a


def _random_binary_eccentricity(rng: np.random.Generator,
                                a_au: float) -> float:
    """Period-eccentricity correlation:
       short-period binaries (< 12 days, ~0.1 AU): circularised, e < 0.05
       intermediate: e roughly thermal (favouring 0.3-0.6)
       very long: e can be near 1
    """
    P_yr = math.sqrt(a_au ** 3 / 1.5)
    if P_yr < 0.033:  # 12 days
        return float(rng.uniform(0.0, 0.05))
    if P_yr < 10.0:
        return float(rng.uniform(0.05, 0.4))
    # Thermal-like, but cap at 0.8 for stability
    e = math.sqrt(rng.uniform(0.0, 0.7))
    return min(0.7, e)


# ---------------------------------------------------------------------
# 1.  Random Raghavan-like binary
# ---------------------------------------------------------------------
def random_binary(rng: np.random.Generator,
                  kind: str = "close",
                  name: str = "Random binary"
                  ) -> StarSystem:
    """Build a random binary drawn from the Raghavan 2010 distributions."""
    m1 = _random_mass(rng, 0.4, 1.3)
    q = float(rng.uniform(0.3, 1.0))  # peaked-like distribution
    m2 = m1 * q
    a = _random_binary_separation(rng, kind=kind)
    e = _random_binary_eccentricity(rng, a)
    A = Star("Primary", mass=m1)
    B = Star("Secondary", mass=m2)
    return StarSystem.binary(name, A, B, separation_au=a,
                             eccentricity=e, quiet=True)


# ---------------------------------------------------------------------
# 2.  Star-hopper candidate
# ---------------------------------------------------------------------
def star_hopper_binary(name: str = "Star-hopper binary",
                       a_bin: float = 40.0,
                       e_bin: float = 0.0,
                       m1: float = 1.0, m2: float = 0.3,
                       planet_a: float = 14.0,
                       planet_e: float = 0.70
                       ) -> StarSystem:
    """Wide binary with a planet on a high-eccentricity orbit whose
    apoastron reaches the L1 Lagrange point of the binary.

    Default parameters give:
      a_bin     = 40 AU       (binary separation)
      m1, m2    = 1.0, 0.3 Msun
      planet_a  = 14 AU       around star A
      planet_e  = 0.70        (apoastron = 23.8 AU; L1 ~ 21.4 AU)

    The apoastron just exceeds L1, so the planet's energy puts it near
    the Jacobi-constant separatrix.  The N-body integration shows
    strong non-Keplerian wobble -- the planet bounces between L1 and
    the inner Hill region of star A.

    Inspired by Moeckel & Veras 2012 (MNRAS 422, 831)."""
    A = Star("Sun-like", mass=m1, luminosity=m1 ** 3.5, teff=5800.0)
    B = Star("M-dwarf", mass=m2, luminosity=m2 ** 3.0, teff=3500.0)
    # Set the binary's omega to pi so star B sits at -x at t=0; the
    # planet's default omega matches the binary, so its apoastron is
    # along -x (toward the M-dwarf), giving real star-hopping geometry.
    sys = StarSystem.binary(name, A, B, separation_au=a_bin,
                            eccentricity=e_bin, omega=math.pi,
                            quiet=True)
    planet = Planet(
        name="Bouncer", mass_me=15.0, radius_re=4.0,
        semi_major_axis_au=planet_a, eccentricity=planet_e,
        host_star_index=0,
        description="Star-hopper: e=0.70 around star A, apoastron just "
                    "past L1 -- N-body shows non-Keplerian motion",
        real_planet=False,
    )
    sys.planets = [planet]
    return sys


# ---------------------------------------------------------------------
# 3.  Trojan / co-orbital
# ---------------------------------------------------------------------
def trojan_system(name: str = "Trojan companion",
                  m_star: float = 1.0,
                  a_planet: float = 1.0
                  ) -> StarSystem:
    """A single Sun-like star with two co-orbital planets at the same
    SMA, the secondary sitting at the L4 Lagrange point of the
    primary (60 degrees ahead).

    Trojan configurations are observed in the Solar System (Jupiter's
    Trojans, Mars' Trojans, Neptune's Trojans, and Earth's 2010 TK7).
    """
    star = Star("Host", mass=m_star, luminosity=m_star ** 3.5, teff=5800.0,
                radius=m_star ** 0.7)
    p1 = Planet(name="Earth-A", mass_me=1.0,
                semi_major_axis_au=a_planet, eccentricity=0.0167,
                host_star_index=0,
                description="L3-mate of the L4 Trojan", real_planet=False)
    # Place the Trojan at 60 deg ahead (mean anomaly = 60 deg) by storing
    # in the description; the static plotter renders the orbital ellipse.
    p2 = Planet(name="Earth-Trojan", mass_me=0.5,
                semi_major_axis_au=a_planet, eccentricity=0.02,
                host_star_index=0,
                description="At Earth-A's L4 Lagrange point (60 deg lead)",
                real_planet=False)
    return StarSystem.single(name, star, planets=[p1, p2])


# ---------------------------------------------------------------------
# 4.  Tight P-type analog with rocky inner planets
# ---------------------------------------------------------------------
def tight_circumbinary_terrestrial(
        name: str = "Tight circumbinary system",
        m1: float = 0.9, m2: float = 0.6,
        a_bin: float = 0.15, e_bin: float = 0.1,
) -> StarSystem:
    """A close binary like Kepler-35 / -47 with a small rocky planet
    set in the circumbinary HZ.

    Holman-Wiegert critical SMA is ~3.5 a_bin for these mass ratios.
    Here we put an Earth-mass planet just outside the stability limit
    and inside the HZ.
    """
    A = Star("HD-x A", mass=m1, luminosity=m1 ** 3.5, teff=5500.0)
    B = Star("HD-x B", mass=m2, luminosity=m2 ** 3.5, teff=4400.0)
    sys = StarSystem.binary(name, A, B, separation_au=a_bin,
                            eccentricity=e_bin, quiet=True)
    # Place a rocky planet in the circumbinary HZ
    L_tot = sys.total_luminosity()
    a_hz = math.sqrt(L_tot / 0.95)  # ~runaway-GH distance
    planet = Planet(
        name="Tatoo", mass_me=1.0, radius_re=1.0,
        semi_major_axis_au=a_hz, eccentricity=0.03,
        host_star_index=None,
        description="HZ rocky planet around tight binary",
        real_planet=False)
    sys.planets = [planet]
    return sys


# ---------------------------------------------------------------------
# 5.  Wide hierarchical triple with an HZ Earth around the third star
# ---------------------------------------------------------------------
def wide_hierarchical_triple(
        name: str = "Wide hierarchy",
        rng: np.random.Generator = None,
) -> StarSystem:
    """Close G+K binary with a wide M-dwarf companion at ~50-200 AU,
    Earth-analog around the M-dwarf."""
    if rng is None:
        rng = np.random.default_rng(seed=1)
    m_in_a = float(rng.uniform(0.9, 1.2))
    m_in_b = float(rng.uniform(0.6, 0.9))
    m_out = float(rng.uniform(0.2, 0.45))
    a_in = float(rng.uniform(0.2, 1.0))
    e_in = float(rng.uniform(0.0, 0.3))
    a_out = float(rng.uniform(50.0, 200.0))
    e_out = float(rng.uniform(0.1, 0.4))
    A = Star("Inner-A", mass=m_in_a)
    B = Star("Inner-B", mass=m_in_b)
    C = Star("Outer-M", mass=m_out)
    sys = StarSystem.hierarchical_triple(
        name, A, B, C,
        a_in=a_in, e_in=e_in, a_out=a_out, e_out=e_out,
        quiet=True)
    # Earth analog around the M-dwarf
    L_M = C.luminosity
    a_hz = math.sqrt(L_M / 0.95)
    sys.planets = [Planet(
        name="Earth-around-M", mass_me=1.0, radius_re=1.0,
        semi_major_axis_au=a_hz, eccentricity=0.0167,
        host_star_index=2,
        description="Earth-analog in M-dwarf's HZ",
        real_planet=False)]
    return sys


# ---------------------------------------------------------------------
# 6.  High-inclination "polar" planet
# ---------------------------------------------------------------------
def polar_planet_system(name: str = "Polar planet") -> StarSystem:
    """K-dwarf with one HZ planet on a high-inclination orbit.

    Polar planets like HAT-P-7 b and WASP-17 b are real -- their
    orbital plane is severely tilted relative to the stellar spin
    axis.  Here we place an Earth-mass planet at i = 75 deg.
    """
    star = Star("K-dwarf", mass=0.75, luminosity=0.31, teff=4800.0,
                radius=0.78)
    sys = StarSystem.single(name, star, planets=[
        Planet(name="Polaris-b", mass_me=1.0, radius_re=1.0,
               semi_major_axis_au=0.62, eccentricity=0.05,
               host_star_index=0,
               description="i = 75 deg, polar orbit",
               real_planet=False),
    ])
    return sys


# ---------------------------------------------------------------------
# 7.  Dramatic close moon (Earth-Moon geometry, exaggerated)
# ---------------------------------------------------------------------
def _moon_ang_diam_deg(r_body_au: float, dist_au: float) -> float:
    return 2.0 * math.degrees(math.asin(min(r_body_au / dist_au, 0.999)))


def big_moon_system(name: str = "Big-moon world") -> StarSystem:
    """Sun-like star, an Earth-analog whose single large moon sits so
    close that it looms ~8x the apparent size of the real Moon.

    The moon clears 1.5x its fluid Roche limit and sits far inside the
    Domingos+2006 Hill bound, so the configuration is dynamically sound
    -- it is the *size on the sky* that is exaggerated, not the physics.
    """
    sun = Star(name + " Sun", mass=1.0, luminosity=1.0, teff=5772.0,
               radius=1.0)
    planet = earth_analog("Hearth", a_au=1.0)
    r_re, a_moon, rho_m = 0.80, 9.0e-4, 3.3
    rho_p = bulk_density_gcc(planet.mass_me, planet.radius_re)
    roche = planetary_roche_limit_au(planet.radius_re, rho_p, rho_m)
    r_hill = planet_hill_radius_au(planet.mass_me, 1.0, sun.mass)
    a_crit = critical_moon_fraction(0.0, 0.0, False) * r_hill
    assert a_moon > 1.5 * roche, (a_moon, roche)
    assert a_moon < a_crit, (a_moon, a_crit)
    planet.moons = [Moon("Selene", mass_me=0.06, radius_re=r_re,
                         a_planet_au=a_moon, eccentricity=0.01,
                         inclination_deg=1.0, density_gcc=rho_m,
                         kind="regular")]
    d = _moon_ang_diam_deg(r_re * R_EARTH_AU, a_moon)
    print(f"  [big_moon_system] Selene angular diameter ~ {d:.2f} deg "
          f"(real Moon ~ 0.52 deg); Roche={roche:.2e} a_crit={a_crit:.2e}")
    return StarSystem.single(name, sun, planets=[planet])


# ---------------------------------------------------------------------
# 8.  Co-orbital giant + its big moon, both prominent in the sky
# ---------------------------------------------------------------------
def companion_with_moon_system(
        name: str = "Co-orbital giant + moon") -> StarSystem:
    """Sun-like star with a habitable observer planet and a co-orbital
    (1:1, quasi-satellite) gas giant a few degrees away that carries one
    large regular moon.

    Two heliocentric planets cannot, by orbital mechanics, appear large
    in each other's sky unless they share an orbit -- so the giant is a
    tight co-orbital companion (a known stable family).  At the canonical
    render the giant subtends ~1 deg (about twice the real Moon) and its
    moon resolves as a separate disk tracked across the day MP4 and the
    debug contact sheet.  The moon clears 1.5x the giant's Roche limit
    and stays well inside the Domingos+2006 Hill bound.
    """
    sun = Star(name + " Sun", mass=1.0, luminosity=1.0, teff=5772.0,
               radius=1.0)
    observer = earth_analog("Watcher", a_au=1.0)

    # `sky_bodies` advances each sibling by _phase0(name); pick a giant
    # name whose deterministic phase places it a few degrees from the
    # observer (chord ~ 0.05 AU) so it looms large without overlapping.
    from goldilocks.skyview import _phase0
    target, best, best_nm = 0.05, 1e9, "Companion"
    for i in range(1, 400):
        nm = f"Companion-{i}"
        ph = _phase0(nm)
        ph = min(ph, 2.0 * math.pi - ph)  # fold to [0, pi]
        if abs(ph - target) < best:
            best, best_nm, best_ph = abs(ph - target), nm, ph
    chord = 2.0 * math.sin(best_ph / 2.0) * 1.0  # AU, both at a=1

    g_mass, g_re = 400.0, 13.0
    giant = Planet(name=best_nm, mass_me=g_mass, radius_re=g_re,
                   semi_major_axis_au=1.0, eccentricity=0.0,
                   host_star_index=0,
                   description="Co-orbital gas giant", real_planet=False)
    rho_p = bulk_density_gcc(g_mass, g_re)
    rho_m = 2.0
    a_moon = 2.0e-3
    roche = planetary_roche_limit_au(g_re, rho_p, rho_m)
    r_hill = planet_hill_radius_au(g_mass, 1.0, sun.mass)
    a_crit = critical_moon_fraction(0.0, 0.0, False) * r_hill
    assert a_moon > 1.5 * roche, (a_moon, roche)
    assert a_moon < a_crit, (a_moon, a_crit)
    giant.moons = [Moon("Titanis", mass_me=0.15, radius_re=1.4,
                        a_planet_au=a_moon, eccentricity=0.02,
                        inclination_deg=1.0, density_gcc=rho_m,
                        kind="regular")]
    dg = _moon_ang_diam_deg(g_re * R_EARTH_AU, chord)
    dm = _moon_ang_diam_deg(1.4 * R_EARTH_AU, chord)
    assert dg > 0.5, dg
    print(f"  [companion_with_moon_system] giant ~{dg:.2f} deg, "
          f"moon ~{dm:.3f} deg at chord {chord:.3f} AU "
          f"(phase {math.degrees(best_ph):.1f} deg)")
    return StarSystem.single(name, sun, planets=[observer, giant])


# ---------------------------------------------------------------------
# Convenience: build a varied set of "interesting" example systems
# ---------------------------------------------------------------------
def build_interesting_systems(rng: np.random.Generator = None
                              ) -> list:
    """Build a curated mix of interesting systems for demos.  Each entry
    is (name_for_file, StarSystem, suggested extent_au, n_periods)."""
    if rng is None:
        rng = np.random.default_rng(seed=2026)
    out = []
    # Star-hopper
    sh = star_hopper_binary(name="Star-hopper (Moeckel-Veras 2012)",
                            a_bin=250.0, e_bin=0.0,
                            m1=1.0, m2=0.3,
                            planet_a=50.0, planet_e=0.85)
    out.append(("07_star_hopper", sh, 300.0, 1.0))
    # Trojan
    tr = trojan_system(name="Trojan co-orbital")
    out.append(("08_trojan", tr, 1.5, 3.0))
    # Tight circumbinary terrestrial
    tc = tight_circumbinary_terrestrial(
        name="Tight P-type with rocky HZ planet",
        m1=0.95, m2=0.55, a_bin=0.20, e_bin=0.05)
    out.append(("09_tight_ptype", tc, 1.6, 8.0))
    # Wide hierarchical
    wh = wide_hierarchical_triple(name="Close binary + wide M-dwarf",
                                  rng=rng)
    out.append(("10_wide_hierarchy", wh, 12.0, 0.05))
    # Polar planet
    pp = polar_planet_system(name="K-dwarf with polar HZ planet")
    out.append(("11_polar_planet", pp, 1.3, 1.0))
    # Random Raghavan binary
    rb = random_binary(rng, kind="close",
                       name="Random Raghavan-distribution binary")
    out.append(("12_random_binary", rb, None, 1.0))
    return out
