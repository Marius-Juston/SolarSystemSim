"""
planets.py
----------
Planet objects, a catalogue of known interesting planets (terrestrial
analogs, circumbinary planets, etc.) and a small library of "candidate"
classes useful for asking "could a planet like Earth fit here?".

Each Planet carries the mass, radius, and any known orbital
parameters.  When passed to a StarSystem, the planet's habitability is
checked against the system's Goldilocks zone (including dynamical
forcing).

Mass-radius relation
--------------------
For unknown radii we use Otegi+20 / Chen-Kipping 2017 piecewise:
   R/Re = M^0.279     for  M <= 2 Me      (rocky)
   R/Re = 1.30 M^0.270 for 2 < M <= 130 Me (sub-Neptune / Neptune)
   R/Re = 17.74 M^-0.044 for M > 130 Me   (Jupiter-like, ~constant)
We use these only when an explicit radius isn't available.

References
----------
Otegi J.F., Bouchy F., Helled R., 2020, A&A 634, A43
Chen J. & Kipping D., 2017, ApJ 834, 17
Orosz J.A. et al., 2012, Science 337, 1511 (Kepler-47 b, c)
Orosz J.A. et al., 2019, AJ 157, 174     (Kepler-47 d)
Doyle L.R. et al., 2011, Science 333, 1602  (Kepler-16 b)
Welsh W.F. et al., 2012, Nature 481, 475    (Kepler-34 b, 35 b)
Orosz J.A. et al., 2012, ApJ 758, 87        (Kepler-38 b)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


# Conversion factors -----------------------------------------------------
M_EARTH_OVER_M_SUN  = 3.0034893e-6
M_JUP_OVER_M_EARTH  = 317.83
R_EARTH_KM          = 6378.137
EARTH_BULK_DENSITY_GCC = 5.514          # g/cm^3 (mean Earth density)


# -----------------------------------------------------------------------
# Mass-radius relation
# -----------------------------------------------------------------------
def radius_from_mass_me(mass_me: float) -> float:
    """Return planet radius (Rearth) from mass (Mearth) using a piecewise
    fit (Otegi+20 / Chen-Kipping 2017)."""
    if mass_me <= 2.0:
        return mass_me ** 0.279
    if mass_me <= 130.0:
        return 1.30 * mass_me ** 0.270
    # Jupiter-like, weak dependence -> nearly constant ~11.2 Rearth
    return 17.74 * mass_me ** -0.044


def bulk_density_gcc(mass_me: float, radius_re: float) -> float:
    """Mean bulk density (g/cm^3) from Earth-relative mass and radius.

    rho = rho_Earth * (M/Me) / (R/Re)^3.  Used for the planetary
    (density-based) Roche limit of moons.
    """
    return EARTH_BULK_DENSITY_GCC * mass_me / (radius_re ** 3)


# Giant if heavier than ~0.1 M_Jup or puffier than ~4 R_earth.  Below this
# the Otegi/Chen-Kipping relation is in the rocky/sub-Neptune regime.
GAS_GIANT_MASS_ME   = 30.0
GAS_GIANT_RADIUS_RE = 4.0


def is_gas_giant(planet: "Planet") -> bool:
    """Classify a planet as a gas/ice giant (vs terrestrial)."""
    return (planet.mass_me >= GAS_GIANT_MASS_ME
            or (planet.radius_re or 0.0) >= GAS_GIANT_RADIUS_RE)


# -----------------------------------------------------------------------
# Planet data class
# -----------------------------------------------------------------------
@dataclass
class Planet:
    name: str
    mass_me: float                  # Earth masses
    radius_re: Optional[float] = None  # Earth radii
    # Orbital elements relative to host (S-type) or to barycentre (P-type)
    semi_major_axis_au: Optional[float] = None
    eccentricity: float = 0.0
    host_star_index: Optional[int] = None  # None -> circumbinary (P-type)
    description: str = ""
    real_planet: bool = False
    # Populated by the random-solar-system generator (moons.py / habitability.py).
    moons: List[Any] = field(default_factory=list)
    habitability: Optional[Any] = None
    # 0.0 = prograde rotation; spin obliquity etc. live in `habitability`.
    inclination_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.radius_re is None:
            self.radius_re = radius_from_mass_me(self.mass_me)

    def is_circumbinary(self) -> bool:
        return self.host_star_index is None

    def __repr__(self) -> str:
        kind = "P-type" if self.is_circumbinary() else f"S-type (star {self.host_star_index})"
        if self.semi_major_axis_au is not None:
            return (f"Planet({self.name!r}, M={self.mass_me:.2g} Me, "
                    f"R={self.radius_re:.2g} Re, a={self.semi_major_axis_au:.3f} AU, "
                    f"e={self.eccentricity:.2f}, {kind})")
        return (f"Planet({self.name!r}, M={self.mass_me:.2g} Me, "
                f"R={self.radius_re:.2g} Re, {kind})")


# -----------------------------------------------------------------------
# Candidate templates for "could a planet of this type fit?"
# -----------------------------------------------------------------------
def earth_analog(name: str = "Earth-analog",
                 a_au: float = 1.0,
                 e: float = 0.0167,
                 host_star_index: int = 0) -> Planet:
    """1 Mearth, 1 Rearth planet with Earth's actual eccentricity (0.0167).

    Set a_au and host_star_index to place the analog at any host."""
    return Planet(name=name, mass_me=1.0, radius_re=1.0,
                  semi_major_axis_au=a_au, eccentricity=e,
                  host_star_index=host_star_index,
                  description="Earth-like terrestrial planet "
                              "(e = 0.0167, the real Earth value)")


def super_earth(name: str = "Super-Earth") -> Planet:
    return Planet(name=name, mass_me=5.0,
                  description="Rocky super-Earth, 5 Mearth")


def mini_earth(name: str = "Mini-Earth") -> Planet:
    return Planet(name=name, mass_me=0.1,
                  description="0.1 Mearth, Mars-mass")


def mars_analog(name: str = "Mars-analog") -> Planet:
    return Planet(name=name, mass_me=0.107, radius_re=0.532,
                  description="Mars-like rocky planet")


def venus_analog(name: str = "Venus-analog") -> Planet:
    return Planet(name=name, mass_me=0.815, radius_re=0.949,
                  description="Venus-like rocky planet")


# -----------------------------------------------------------------------
# Real circumbinary planets (Kepler-16, 34, 35, 38, 47)
# Parameters from the NASA Exoplanet Archive and the original
# discovery papers.
# -----------------------------------------------------------------------
def kepler16_b() -> Planet:
    return Planet(
        name="Kepler-16 b", mass_me=105.0, radius_re=8.27,
        semi_major_axis_au=0.7048, eccentricity=0.0069,
        host_star_index=None,
        description="Saturn-mass circumbinary planet, Doyle+11",
        real_planet=True)


def kepler34_b() -> Planet:
    return Planet(
        name="Kepler-34 b", mass_me=69.9, radius_re=8.38,
        semi_major_axis_au=1.0896, eccentricity=0.182,
        host_star_index=None,
        description="0.22 MJ circumbinary planet, Welsh+12",
        real_planet=True)


def kepler35_b() -> Planet:
    return Planet(
        name="Kepler-35 b", mass_me=40.4, radius_re=8.16,
        semi_major_axis_au=0.6035, eccentricity=0.042,
        host_star_index=None,
        description="0.13 MJ circumbinary planet, Welsh+12",
        real_planet=True)


def kepler38_b() -> Planet:
    return Planet(
        name="Kepler-38 b", mass_me=21.0, radius_re=4.20,
        semi_major_axis_au=0.4644, eccentricity=0.032,
        host_star_index=None,
        description="Sub-Neptune circumbinary planet, Orosz+12",
        real_planet=True)


def kepler47_b() -> Planet:
    return Planet(
        name="Kepler-47 b", mass_me=2.07, radius_re=2.98,
        semi_major_axis_au=0.2877, eccentricity=0.021,
        host_star_index=None,
        description="Inner circumbinary planet, Orosz+12/19",
        real_planet=True)


def kepler47_d() -> Planet:
    return Planet(
        name="Kepler-47 d", mass_me=19.0, radius_re=7.04,
        semi_major_axis_au=0.6992, eccentricity=0.024,
        host_star_index=None,
        description="Middle circumbinary planet, Orosz+19",
        real_planet=True)


def kepler47_c() -> Planet:
    return Planet(
        name="Kepler-47 c", mass_me=3.17, radius_re=4.65,
        semi_major_axis_au=0.9638, eccentricity=0.044,
        host_star_index=None,
        description="Outer (HZ) circumbinary planet, Orosz+12/19",
        real_planet=True)


# -----------------------------------------------------------------------
# Convenience: pick a planet template by name
# -----------------------------------------------------------------------
CATALOG: Dict[str, callable] = {
    "earth"       : earth_analog,
    "venus"       : venus_analog,
    "mars"        : mars_analog,
    "super-earth" : super_earth,
    "mini-earth"  : mini_earth,
    "kepler-16 b" : kepler16_b,
    "kepler-34 b" : kepler34_b,
    "kepler-35 b" : kepler35_b,
    "kepler-38 b" : kepler38_b,
    "kepler-47 b" : kepler47_b,
    "kepler-47 c" : kepler47_c,
    "kepler-47 d" : kepler47_d,
}


def list_catalog() -> None:
    print("Available planet templates:")
    for key, fn in CATALOG.items():
        print(f"  {key:>14s} -> {fn().description}")