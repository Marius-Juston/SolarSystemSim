"""
roche.py
--------
Roche-lobe geometry for binary star systems.

The Roche lobe is the teardrop-shaped equipotential bounding the
material gravitationally bound to one star of a binary.  We use the
Eggleton (1983) approximation, accurate to 1% over the entire mass
range:

    r_L / A = 0.49 q^(2/3) / [0.6 q^(2/3) + ln(1 + q^(1/3))]

where q = m1 / m2 and A is the orbital separation.  This is the
radius of a sphere with the same volume as the Roche lobe of star 1.

Why we need it
--------------
The Roche lobe defines the *physical* upper bound to S-type orbits:
any planet whose semi-major axis approaches a sizeable fraction of
the host star's Roche lobe radius will be tidally stripped at
periastron, regardless of what Holman & Wiegert allow.  In practice
the dynamical instability limit (Holman-Wiegert) kicks in first for
typical mass ratios, but for very unequal mass binaries the Roche
lobe becomes the relevant cut-off near periastron of an eccentric
orbit.

Reference
---------
Eggleton P.P. 1983, ApJ 268, 368.
"""

from __future__ import annotations

import math


def eggleton_roche_radius(m_host: float, m_other: float,
                          separation_au: float) -> float:
    """Roche-lobe radius (AU) of the host star.

    Parameters
    ----------
    m_host  : mass of the star whose Roche lobe is being computed (Msun)
    m_other : mass of the companion star (Msun)
    separation_au : instantaneous separation A (AU)

    Returns
    -------
    r_L  : radius (AU) of an equivalent sphere with the same volume
           as the Roche lobe of the host star.
    """
    q = m_host / m_other
    q23 = q ** (2.0 / 3.0)
    q13 = q ** (1.0 / 3.0)
    return separation_au * 0.49 * q23 / (0.6 * q23 + math.log1p(q13))


def roche_lobe_periastron(m_host: float, m_other: float,
                          a_bin: float, e_bin: float) -> float:
    """Roche-lobe radius at periastron of an eccentric binary (worst case)."""
    return eggleton_roche_radius(m_host, m_other, a_bin * (1.0 - e_bin))


def l1_distance(m_host: float, m_other: float,
                separation_au: float) -> float:
    """Distance from the host star to the L1 Lagrange point.

    Approximation good to <1% over reasonable mass ratios.  This is
    where mass transfer would occur if the host star filled its Roche
    lobe.  L1 is always slightly closer to the host than r_L.
    """
    mu = m_other / (m_host + m_other)
    # Standard 5th-order expansion of L1 position (Hill / Roche).
    # x = (mu/3)^{1/3} - (1/3)*(mu/3)^{2/3} - (1/9)*(mu/3) + ...
    x = (mu / 3.0) ** (1.0 / 3.0)
    x -= (1.0 / 3.0) * x * x
    x -= (1.0 / 9.0) * x * x * x
    return separation_au * (1.0 - x)
