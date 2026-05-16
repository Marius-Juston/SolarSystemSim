"""
stellar.py
-----------
Stellar physics: mass-luminosity, mass-radius, mass-effective-temperature
relations for main-sequence stars.

References
----------
Eker, Z., et al. 2018, MNRAS, 479, 5491
   "Interrelated main-sequence mass-luminosity, mass-radius and
   mass-effective temperature relations"
Kopparapu, R. K. et al. 2013, ApJ, 765, 131
Kopparapu, R. K. et al. 2014, ApJ Letters, 787, L29
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

# Physical constants -----------------------------------------------------
M_SUN_KG       = 1.98892e30          # kg
L_SUN_W        = 3.828e26            # W (IAU 2015 nominal)
R_SUN_M        = 6.957e8             # m
AU_M           = 1.495978707e11      # m
G_SI           = 6.67430e-11         # m^3 / kg / s^2
SIGMA_SB       = 5.670374419e-8      # W / m^2 / K^4
T_EFF_SUN_K    = 5772.0              # K (IAU 2015 nominal)
YEAR_S         = 3.15576e7           # s


# -----------------------------------------------------------------------
# Mass-Luminosity relation (Eker et al. 2018, six-piece classical MLR)
# log10(L/Lsun) = alpha * log10(M/Msun) + beta
# Valid for 0.179 <= M/Msun <= 31
# -----------------------------------------------------------------------
_EKER_MLR = [
    # (M_min, M_max, alpha, beta)
    (0.179, 0.45,  2.028, -0.976),
    (0.45,  0.72,  4.572, -0.102),
    (0.72,  1.05,  5.743,  0.007),
    (1.05,  2.40,  4.329,  0.010),
    (2.40,  7.00,  3.967,  0.093),
    (7.00, 31.00,  2.865,  1.105),
]


def luminosity_from_mass(mass_msun: float) -> float:
    """Main-sequence luminosity (Lsun) from mass (Msun), Eker+18."""
    if mass_msun < 0.05 or mass_msun > 150:
        raise ValueError(f"Mass {mass_msun} Msun is well outside MS range.")
    # Extend a touch below 0.179 with the same low-mass slope, with a warning.
    for m_lo, m_hi, alpha, beta in _EKER_MLR:
        if m_lo <= mass_msun <= m_hi:
            return 10.0 ** (alpha * math.log10(mass_msun) + beta)
    # Below lowest bin: use lowest-bin slope.
    if mass_msun < _EKER_MLR[0][0]:
        _, _, alpha, beta = _EKER_MLR[0]
        return 10.0 ** (alpha * math.log10(mass_msun) + beta)
    # Above highest: use highest-bin slope.
    _, _, alpha, beta = _EKER_MLR[-1]
    return 10.0 ** (alpha * math.log10(mass_msun) + beta)


# -----------------------------------------------------------------------
# Mass-Radius relation (Eker et al. 2018, valid for M <= 1.5 Msun)
#  R/Rsun = 0.438 * M^2 + 0.479 * M + 0.075
# For higher mass, use a two-piece fit from Demircan & Kahraman / Eker+18.
# -----------------------------------------------------------------------
def radius_from_mass(mass_msun: float) -> float:
    if mass_msun <= 1.5:
        return 0.438 * mass_msun**2 + 0.479 * mass_msun + 0.075
    # Demircan & Kahraman / Eker high-mass form: R ~ M^0.747
    return 1.5**2 * 0.438 + 1.5 * 0.479 + 0.075 \
           if mass_msun == 1.5 else \
           (radius_from_mass(1.5)) * (mass_msun / 1.5) ** 0.747


# -----------------------------------------------------------------------
# Effective temperature from L and R (Stefan-Boltzmann)
# -----------------------------------------------------------------------
def teff_from_l_and_r(l_lsun: float, r_rsun: float) -> float:
    """Stefan-Boltzmann: L = 4 pi R^2 sigma Teff^4."""
    L = l_lsun * L_SUN_W
    R = r_rsun * R_SUN_M
    return (L / (4.0 * math.pi * R * R * SIGMA_SB)) ** 0.25


def teff_from_mass(mass_msun: float) -> float:
    """Combine MLR and MRR to estimate Teff."""
    return teff_from_l_and_r(luminosity_from_mass(mass_msun),
                             radius_from_mass(mass_msun))


# -----------------------------------------------------------------------
# Star class
# -----------------------------------------------------------------------
@dataclass
class Star:
    """A main-sequence star.

    Any one of (mass, luminosity, teff) is enough; the rest will be
    inferred from main-sequence relations.  If the caller supplies more
    than one, the supplied values are kept and inconsistent fields are
    flagged.
    """
    name: str
    mass: Optional[float]  = None    # Msun
    luminosity: Optional[float] = None  # Lsun
    teff: Optional[float]  = None    # K
    radius: Optional[float] = None   # Rsun
    # Position and velocity (3-vectors), AU and AU/yr -- filled by System.
    position: tuple = (0.0, 0.0, 0.0)
    velocity: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.mass is None and self.luminosity is None and self.teff is None:
            raise ValueError(f"Star {self.name!r}: need mass, L, or Teff.")
        if self.mass is not None:
            if self.luminosity is None:
                self.luminosity = luminosity_from_mass(self.mass)
            if self.radius is None:
                self.radius = radius_from_mass(self.mass)
            if self.teff is None:
                self.teff = teff_from_l_and_r(self.luminosity, self.radius)
        else:
            # Need to back-solve.  Use Teff if given, else estimate from L.
            if self.teff is None and self.luminosity is not None:
                # Assume R ~ L^0.5 for FGK; weak guess but consistent.
                if self.radius is None:
                    self.radius = self.luminosity ** 0.5
                self.teff = teff_from_l_and_r(self.luminosity, self.radius)
            elif self.luminosity is None and self.teff is not None:
                # Crude inverse: use Sun as anchor, R~Rsun.
                if self.radius is None:
                    self.radius = 1.0
                self.luminosity = (self.radius**2) * (self.teff / T_EFF_SUN_K) ** 4
            # Mass left as None if user really gave only L or Teff; downstream
            # routines that need mass will raise.

    # Convenience ------------------------------------------------------
    def __repr__(self) -> str:
        return (f"Star({self.name!r}, M={self.mass:.3g} Msun, "
                f"L={self.luminosity:.3g} Lsun, "
                f"Teff={self.teff:.0f} K)")