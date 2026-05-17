"""
stability.py
------------
Dynamical-stability criteria.

For binaries
------------
Holman & Wiegert 1999 critical SMA:

  S-type:  a_p stable for  a_p / a_b  <  a_c
           a_c = 0.464 - 0.380 mu - 0.631 e
                 + 0.586 mu e + 0.150 e^2 - 0.198 mu e^2
           (mu = m_other / (m_host + m_other))

  P-type:  a_p stable for  a_p / a_b  >  a_c
           a_c = 1.60 + 5.10 e - 2.22 e^2
                 + 4.12 mu - 4.27 e mu - 5.09 mu^2 + 4.61 e^2 mu^2

For triples / hierarchies
-------------------------
Mardling & Aarseth 2001:

  (a_out / a_in)_crit = 2.8 * [(1 + q_out)(1 + e_out) / sqrt(1 - e_out)]^{2/5}
                          * (1 - 0.3 i_mut / 180)
  q_out = m3 / (m1 + m2)

For planet packing
------------------
Mutual Hill radius (Gladman 1993; Smith & Lissauer 2009):

  R_H,mutual = ((m1 + m2)/(3 M*))^{1/3} * (a1 + a2)/2

  Hill stability:    Delta a >= 2 sqrt(3) R_H ~ 3.46 R_H
  Lagrange Gyr:      Delta a >= ~ 8-10 R_H  (empirical, Smith & Lissauer 2009)

References
----------
Holman M.J. & Wiegert P.A., 1999, AJ 117, 621
Mardling R.A. & Aarseth S.J., 2001, MNRAS 321, 398
Gladman B., 1993, Icarus 106, 247
Smith A.W. & Lissauer J.J., 2009, Icarus 201, 381
"""

from __future__ import annotations

import math
from typing import List

M_EARTH_OVER_M_SUN = 3.0034893e-6


# -----------------------------------------------------------------------
# Holman & Wiegert (1999)
# -----------------------------------------------------------------------
def holman_wiegert_stype(mu: float, e_bin: float) -> float:
    e = e_bin
    return (0.464
            - 0.380 * mu
            - 0.631 * e
            + 0.586 * mu * e
            + 0.150 * e * e
            - 0.198 * mu * e * e)


def holman_wiegert_ptype(mu: float, e_bin: float) -> float:
    e = e_bin
    return (1.60
            + 5.10 * e
            - 2.22 * e * e
            + 4.12 * mu
            - 4.27 * e * mu
            - 5.09 * mu * mu
            + 4.61 * e * e * mu * mu)


# -----------------------------------------------------------------------
# Mardling-Aarseth hierarchical-triple stability
# -----------------------------------------------------------------------
def mardling_aarseth_stable(a_in: float, a_out: float,
                            e_out: float,
                            m1: float, m2: float, m3: float,
                            i_mut_deg: float = 0.0) -> bool:
    """Return True if the hierarchical triple is dynamically stable."""
    q_out = m3 / (m1 + m2)
    crit = 2.8 * ((1.0 + q_out) * (1.0 + e_out) / math.sqrt(1.0 - e_out)) ** 0.4
    crit *= (1.0 - 0.3 * i_mut_deg / 180.0)
    return (a_out / a_in) > crit


def mardling_aarseth_ratio(a_in: float, e_out: float,
                           m1: float, m2: float, m3: float,
                           i_mut_deg: float = 0.0) -> float:
    """Return the critical outer/inner SMA ratio."""
    q_out = m3 / (m1 + m2)
    crit = 2.8 * ((1.0 + q_out) * (1.0 + e_out) / math.sqrt(1.0 - e_out)) ** 0.4
    crit *= (1.0 - 0.3 * i_mut_deg / 180.0)
    return crit


# -----------------------------------------------------------------------
# Hill radii and planet packing
# -----------------------------------------------------------------------
def mutual_hill_radius(m1_me: float, m2_me: float,
                       a1_au: float, a2_au: float,
                       m_star_msun: float) -> float:
    factor = (m1_me + m2_me) * M_EARTH_OVER_M_SUN / (3.0 * m_star_msun)
    return ((a1_au + a2_au) / 2.0) * factor ** (1.0 / 3.0)


def hill_radius_single(m_planet_me: float, a_au: float,
                       m_star_msun: float) -> float:
    """Hill radius of a single planet (used in Roche-lobe-style cuts)."""
    factor = m_planet_me * M_EARTH_OVER_M_SUN / (3.0 * m_star_msun)
    return a_au * factor ** (1.0 / 3.0)


def max_planets_in_zone(inner_au: float, outer_au: float,
                        m_star_msun: float,
                        planet_mass_me: float = 1.0,
                        delta: float = 10.0) -> int:
    """Maximum equal-mass planets between inner_au and outer_au, separated
    by `delta` mutual Hill radii.

    delta = 10 is typical for Lagrange/Gyr stability (Smith & Lissauer 2009);
    delta ~ 3.5 is the Hill stability limit (Gladman 1993).
    """
    if outer_au <= inner_au:
        return 0
    k = ((2.0 * planet_mass_me) * M_EARTH_OVER_M_SUN
         / (3.0 * m_star_msun)) ** (1.0 / 3.0)
    half = 0.5 * delta * k
    if half >= 1.0:
        return 1
    ratio = (1.0 + half) / (1.0 - half)

    a = inner_au
    n = 1
    while True:
        a_next = a * ratio
        if a_next > outer_au:
            return n
        a = a_next
        n += 1


def packing_positions(inner_au: float, outer_au: float,
                      m_star_msun: float,
                      planet_mass_me: float = 1.0,
                      delta: float = 10.0) -> List[float]:
    """Return the geometric-progression SMAs for `max_planets_in_zone`."""
    k = ((2.0 * planet_mass_me) * M_EARTH_OVER_M_SUN
         / (3.0 * m_star_msun)) ** (1.0 / 3.0)
    half = 0.5 * delta * k
    if half >= 1.0:
        return [0.5 * (inner_au + outer_au)]
    ratio = (1.0 + half) / (1.0 - half)
    positions = [inner_au]
    while positions[-1] * ratio <= outer_au:
        positions.append(positions[-1] * ratio)
    return positions
