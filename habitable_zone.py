"""
habitable_zone.py
-----------------
Habitable-zone (Goldilocks-zone) calculations for single, binary, and
N-star systems.  Includes three flavours of HZ:

* **Classical (snapshot)** HZ from Kopparapu et al. 2013/2014, valid
  for an Earth-mass planet on a fixed circular orbit.
* **Multi-star** HZ via the spectrally-weighted-flux scheme of Mueller
  & Haghighipour 2014.
* **Permanently Habitable Zone (PHZ)** of Eggl et al. 2012:
  the planet's eccentricity oscillation (from binary forcing and
  planet-planet coupling) is allowed to vary the instantaneous flux,
  and the planet is required to stay habitable at periastron AND
  apastron simultaneously.

Distance / flux equations
=========================
For an isolated single star:

    S_eff(T_*) = polynomial(T_eff)                  # Kopparapu 2013
    d         = sqrt(L / S_eff)                     # AU, for S in S_0

For a planet on an eccentric orbit of semi-major axis a_p and
eccentricity e_p:

    r_peri = a_p (1 - e_p)
    r_apo  = a_p (1 + e_p)

The maximum (periastron) and minimum (apastron) fluxes a single-star
planet receives are S_max = L / r_peri^2 and S_min = L / r_apo^2.
For PHZ habitability we require

    S_min  >=  S_outer_threshold   (planet never freezes globally)
    S_max  <=  S_inner_threshold   (planet never runs away)

For N-star systems, S_max and S_min must be taken over the *full
trajectory* of the planet which moves in the time-dependent potential
of N stars.  We approximate this by sampling the binary configuration
between periastron and apastron and the planet trajectory between its
own periastron and apastron (see system.py).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import numpy as np

from stellar import Star


# -----------------------------------------------------------------------
# Kopparapu 2013/2014 polynomial coefficients
# (from Kopparapu's published Fortran/Python code at PSU)
# -----------------------------------------------------------------------
# Order:   RecentVenus, RunawayGH, MaxGH, EarlyMars, RG_5Me, RG_0.1Me
_SEFF_SUN = np.array([1.776, 1.107, 0.356, 0.320, 1.188, 0.99])
_A        = np.array([2.136e-4,  1.332e-4,  6.171e-5,  5.547e-5,  1.433e-4,  1.209e-4])
_B        = np.array([2.533e-8,  1.580e-8,  1.698e-9,  1.526e-9,  1.707e-8,  1.404e-8])
_C        = np.array([-1.332e-11, -8.308e-12, -3.198e-12, -2.874e-12, -8.968e-12, -7.418e-12])
_D        = np.array([-3.097e-15, -1.931e-15, -5.575e-16, -5.011e-16, -2.084e-15, -1.713e-15])

LIMIT_NAMES = (
    "RecentVenus",       # optimistic inner
    "RunawayGreenhouse", # conservative inner (1 Me)
    "MaxGreenhouse",     # conservative outer
    "EarlyMars",         # optimistic outer
    "RG_5Me",            # conservative inner (5 Me)
    "RG_0.1Me",          # conservative inner (0.1 Me)
)
LIMIT_INDEX = {n: i for i, n in enumerate(LIMIT_NAMES)}


def seff_for_limit(teff_k: float, limit: str = "RunawayGreenhouse") -> float:
    """Effective stellar flux (in units of S_0 at Earth) for a given
    Kopparapu HZ limit, valid for 2600 <= Teff <= 7200 K."""
    i = LIMIT_INDEX[limit]
    t = max(2600.0, min(7200.0, teff_k)) - 5780.0
    return float(_SEFF_SUN[i] + _A[i]*t + _B[i]*t**2 + _C[i]*t**3 + _D[i]*t**4)


def hz_distance(star: Star, limit: str = "RunawayGreenhouse") -> float:
    """Distance (AU) at which a star delivers the flux threshold."""
    if star.luminosity is None:
        raise ValueError(f"Star {star.name!r} needs a luminosity.")
    return math.sqrt(star.luminosity / seff_for_limit(star.teff, limit))


# -----------------------------------------------------------------------
# Snapshot HZ for an isolated star
# -----------------------------------------------------------------------
@dataclass
class HZBoundaries:
    optimistic_inner: float
    conservative_inner: float
    conservative_outer: float
    optimistic_outer: float

    def contains(self, r_au: float, optimistic: bool = False) -> bool:
        if optimistic:
            return self.optimistic_inner <= r_au <= self.optimistic_outer
        return self.conservative_inner <= r_au <= self.conservative_outer


def single_star_hz(star: Star,
                   planet_mass_me: float = 1.0) -> HZBoundaries:
    """Boundaries of the snapshot HZ around an isolated star."""
    if abs(planet_mass_me - 1.0) < 0.1:
        inner_lim = "RunawayGreenhouse"
    elif planet_mass_me >= 3.0:
        inner_lim = "RG_5Me"
    else:
        inner_lim = "RG_0.1Me"
    return HZBoundaries(
        optimistic_inner   = hz_distance(star, "RecentVenus"),
        conservative_inner = hz_distance(star, inner_lim),
        conservative_outer = hz_distance(star, "MaxGreenhouse"),
        optimistic_outer   = hz_distance(star, "EarlyMars"),
    )


# -----------------------------------------------------------------------
# Spectral weighting for multi-star systems (Mueller & Haghighipour 2014)
# -----------------------------------------------------------------------
def spectral_weight(teff_k: float, limit: str = "RunawayGreenhouse") -> float:
    """W(T*) = S_eff_sun / S_eff(T*).

    A photon from a star of arbitrary T_eff is "worth" 1/S_eff(T_eff)
    times one from the Sun because the limit S_eff depends on T_eff.
    The combined criterion then has the *same* threshold (S_eff_sun)
    regardless of stellar mix.
    """
    i = LIMIT_INDEX[limit]
    s_sun  = float(_SEFF_SUN[i])
    s_star = seff_for_limit(teff_k, limit)
    return s_sun / s_star


def weighted_flux_at(point_au: np.ndarray,
                     stars: Iterable[Star],
                     limit: str = "RunawayGreenhouse") -> float:
    """Spectrally weighted flux from all `stars` at a single 3-D point."""
    p = np.asarray(point_au, dtype=float)
    total = 0.0
    for s in stars:
        r2 = np.sum((p - np.asarray(s.position, dtype=float)) ** 2)
        if r2 <= 0.0:
            return float("inf")
        w = spectral_weight(s.teff, limit)
        total += w * s.luminosity / r2
    return float(total)


def hz_mask(grid_au: np.ndarray,
            stars: Iterable[Star],
            optimistic: bool = False) -> np.ndarray:
    """Vectorised habitability test on a grid of 3-D points."""
    inner_lim = "RecentVenus"        if optimistic else "RunawayGreenhouse"
    outer_lim = "EarlyMars"          if optimistic else "MaxGreenhouse"

    grid = np.asarray(grid_au, dtype=float)
    flat = grid.reshape(-1, 3)

    flux_inner = np.zeros(flat.shape[0])
    flux_outer = np.zeros(flat.shape[0])

    s_sun_inner = float(_SEFF_SUN[LIMIT_INDEX[inner_lim]])
    s_sun_outer = float(_SEFF_SUN[LIMIT_INDEX[outer_lim]])

    for s in stars:
        pos = np.asarray(s.position, dtype=float)
        diff = flat - pos
        r2 = np.einsum("ij,ij->i", diff, diff)
        r2 = np.where(r2 > 0.0, r2, np.nan)
        w_in = spectral_weight(s.teff, inner_lim)
        w_out = spectral_weight(s.teff, outer_lim)
        flux_inner += w_in  * s.luminosity / r2
        flux_outer += w_out * s.luminosity / r2

    hab = (flux_inner <= s_sun_inner) & (flux_outer >= s_sun_outer)
    hab = np.where(np.isnan(flux_inner) | np.isnan(flux_outer), False, hab)
    return hab.reshape(grid.shape[:-1])