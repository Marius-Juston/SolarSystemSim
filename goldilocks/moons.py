"""
moons.py
--------
Moon generation and moon-orbit stability for the random solar-system
generator.

Two distinct stability bounds bracket where a moon can live around its
planet:

1.  **Inner edge -- the planetary Roche limit** (density-based, fluid
    body, Roche 1849; Murray & Dermott 1999, sec. 4.13):

        d_Roche = 2.44 R_p (rho_p / rho_m)^(1/3)

    A satellite straying inside d_Roche is tidally shredded into a ring.
    (This is *not* the Eggleton stellar-binary Roche lobe in roche.py --
    that is a mass-ratio equipotential between two stars; this is the
    classical density-based satellite-disruption distance.)

2.  **Outer edge -- a fraction of the planet's Hill radius**
    (Domingos, Winter & Yokoyama 2006, MNRAS 373, 1227).  A satellite is
    bound long-term only if its semi-major axis stays inside

        a_crit = f * R_Hill,
        f_prograde   = 0.4895 (1 - 1.0305 e_p - 0.2738 e_m)
        f_retrograde = 0.9309 (1 - 1.0764 e_p - 0.9812 e_m)

    Retrograde satellites are stable roughly twice as far out as
    prograde ones -- which is exactly why the distant irregular moons of
    Jupiter and Saturn are overwhelmingly retrograde.

Moon-count statistics follow the Solar System: terrestrial planets have
0-2 moons, gas/ice giants have dozens to well over a hundred (a few large
*regular* moons close in, plus a large swarm of small, often retrograde,
*irregular* captured moons further out).

References
----------
Roche E. 1849, Acad. Sci. Montpellier 1, 243
Murray C.D. & Dermott S.F. 1999, "Solar System Dynamics", sec. 4.13
Domingos R.C., Winter O.C., Yokoyama T. 2006, MNRAS 373, 1227
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from goldilocks import stability as stab
from goldilocks.planets import (radius_from_mass_me, bulk_density_gcc,
                                is_gas_giant, M_EARTH_OVER_M_SUN,
                                R_EARTH_KM)

AU_KM = 1.495978707e8
R_EARTH_AU = R_EARTH_KM / AU_KM


# ---------------------------------------------------------------------
# Moon data class
# ---------------------------------------------------------------------
@dataclass
class Moon:
    name: str
    mass_me: float  # Earth masses
    radius_re: Optional[float] = None  # Earth radii
    a_planet_au: float = 0.0  # SMA relative to the host planet
    eccentricity: float = 0.0
    inclination_deg: float = 0.0
    retrograde: bool = False
    density_gcc: float = 2.0
    kind: str = "regular"  # "regular" | "irregular"

    def __post_init__(self) -> None:
        if self.radius_re is None:
            self.radius_re = radius_from_mass_me(self.mass_me)

    def __repr__(self) -> str:
        d = "retro" if self.retrograde else "pro"
        return (f"Moon({self.name!r}, M={self.mass_me:.2e} Me, "
                f"a={self.a_planet_au:.3e} AU, e={self.eccentricity:.2f}, "
                f"{self.kind}, {d})")


# ---------------------------------------------------------------------
# Stability bounds
# ---------------------------------------------------------------------
def planetary_roche_limit_au(radius_re: float,
                             rho_planet_gcc: float,
                             rho_moon_gcc: float) -> float:
    """Fluid-body Roche limit (AU): d = 2.44 R_p (rho_p/rho_m)^(1/3)."""
    r_p_au = radius_re * R_EARTH_AU
    return 2.44 * r_p_au * (rho_planet_gcc / rho_moon_gcc) ** (1.0 / 3.0)


def critical_moon_fraction(e_planet: float, e_moon: float,
                           retrograde: bool) -> float:
    """Domingos+2006 critical a_moon / R_Hill for long-term stability."""
    if retrograde:
        f = 0.9309 * (1.0 - 1.0764 * e_planet - 0.9812 * e_moon)
    else:
        f = 0.4895 * (1.0 - 1.0305 * e_planet - 0.2738 * e_moon)
    return max(0.0, f)


def planet_hill_radius_au(planet_mass_me: float,
                          planet_a_au: float,
                          host_mass_msun: float) -> float:
    """Planet Hill radius (AU).  `host_mass_msun` is the star (S-type) or
    the total system mass (circumbinary)."""
    return stab.hill_radius_single(planet_mass_me, planet_a_au,
                                   host_mass_msun)


# ---------------------------------------------------------------------
# Moon generation
# ---------------------------------------------------------------------
def _moon_density(rng: np.random.Generator, icy: bool) -> float:
    if icy:
        return float(rng.uniform(1.2, 2.2))  # ice-rich (Ganymede-like)
    return float(rng.uniform(2.8, 3.8))  # rocky (Io/Moon-like)


def generate_moons(planet,
                   host_mass_msun: float,
                   rng: np.random.Generator) -> List[Moon]:
    """Generate a globally-stable moon system for `planet`.

    Every returned moon satisfies, by construction:
      * a_moon > 1.3 x planetary Roche limit (no tidal disruption),
      * a_moon < f_crit(e) x R_Hill  (Domingos+2006 long-term bound),
      * regular moons are mutually separated by >= 8 mutual Hill radii
        (Smith & Lissauer 2009 long-term packing limit).
    """
    a_p = planet.semi_major_axis_au
    if a_p is None or a_p <= 0.0:
        return []
    e_p = planet.eccentricity
    rho_p = bulk_density_gcc(planet.mass_me, planet.radius_re)
    R_hill = planet_hill_radius_au(planet.mass_me, a_p, host_mass_msun)
    giant = is_gas_giant(planet)

    # ----- moon count -----
    if giant:
        # lognormal-ish: dozens to >100 (Jupiter 95, Saturn 146 known)
        n_total = int(np.clip(rng.normal(85.0, 35.0), 24, 170))
        n_regular = int(rng.integers(3, 9))  # Galilean/Saturnian-like
    else:
        n_total = int(rng.integers(0, 3))  # 0, 1 or 2
        n_regular = n_total
    n_total = max(n_total, 0)
    n_regular = min(n_regular, n_total)
    n_irregular = n_total - n_regular
    if n_total == 0:
        return []

    moons: List[Moon] = []

    # ----- regular (large, close-in, prograde, low-e) moons -----
    # Real regular satellites sit FAR inside the Domingos outer bound
    # (the Galileans are at ~0.002-0.013 R_Hill, vs f_crit ~ 0.49).  Cap
    # the outer regular at 0.33 f_crit so they are robustly bound under
    # the full N-body field, not perched on the stability separatrix.
    f_pro = 0.33 * critical_moon_fraction(e_p, 0.01, False) * R_hill
    a_cursor = None
    for i in range(n_regular):
        icy = bool(rng.random() < 0.5)
        rho_m = _moon_density(rng, icy)
        if giant:
            m_me = float(10 ** rng.uniform(-3.3, -1.5))  # ~5e-4..0.03 Me
        else:
            m_me = float(10 ** rng.uniform(-2.5, -1.8))  # up to ~Moon mass
        r_re = radius_from_mass_me(m_me)
        roche = planetary_roche_limit_au(planet.radius_re, rho_p, rho_m)
        a_min = 1.5 * roche
        if a_cursor is None:
            a = a_min * rng.uniform(1.0, 1.6)
        else:
            # >= 8 mutual Hill radii beyond the previous regular moon
            prev = moons[-1]
            rh = stab.mutual_hill_radius(
                prev.mass_me, m_me, a_cursor, a_cursor,
                planet.mass_me * M_EARTH_OVER_M_SUN)
            a = a_cursor + max(8.0 * rh, 0.25 * a_cursor)
        if a >= f_pro:
            break  # ran out of stable room
        e = float(abs(rng.normal(0.0, 0.01)))
        inc = float(abs(rng.normal(0.0, 1.5)))
        moons.append(Moon(
            name=f"{planet.name} {_roman(i + 1)}",
            mass_me=m_me, radius_re=r_re,
            a_planet_au=a, eccentricity=e, inclination_deg=inc,
            retrograde=False, density_gcc=rho_m, kind="regular"))
        a_cursor = a

    # ----- irregular (small, distant, eccentric, often retrograde) -----
    for j in range(n_irregular):
        retro = bool(rng.random() < 0.75)  # capture favours retro
        icy = bool(rng.random() < 0.4)
        rho_m = _moon_density(rng, icy)
        m_me = float(10 ** rng.uniform(-7.0, -3.5))
        r_re = radius_from_mass_me(m_me)
        e = float(rng.uniform(0.05, 0.35))
        inc = float(rng.uniform(100.0, 165.0) if retro
                    else rng.uniform(5.0, 45.0))
        f_crit = critical_moon_fraction(e_p, e, retro)
        roche = planetary_roche_limit_au(planet.radius_re, rho_p, rho_m)
        a_lo = max(0.12 * R_hill, 1.5 * roche)
        # Stay well inside the Domingos boundary (0.6 f_crit) so captured
        # irregulars are robustly bound, not on the stability separatrix.
        a_hi = 0.60 * f_crit * R_hill
        if a_hi <= a_lo:
            continue
        a = float(rng.uniform(a_lo, a_hi))
        moons.append(Moon(
            name=f"{planet.name} S/{j + 1:03d}",
            mass_me=m_me, radius_re=r_re,
            a_planet_au=a, eccentricity=e, inclination_deg=inc,
            retrograde=retro, density_gcc=rho_m, kind="irregular"))

    return moons


_ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def _roman(n: int) -> str:
    return _ROMAN[n] if 0 <= n < len(_ROMAN) else str(n)
