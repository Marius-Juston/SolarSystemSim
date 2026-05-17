"""
kepler.py
---------
Closed-form Kepler solver for the two-body problem and a simple
symplectic N-body integrator for N >= 3.

The classical two-body problem has an exact solution; the relative
position (r, theta) at time t follows from Kepler's equation

    M = E - e * sin(E),       M = n (t - t_p)

with mean motion  n = sqrt( G (m1 + m2) / a^3 ).  The eccentric
anomaly E is solved for numerically (one or two Newton iterations is
enough for double precision); the true anomaly and radial distance
follow analytically.

For N >= 3 there is no closed form (Poincare 1890), and we fall back
on a velocity-Verlet / kick-drift-kick leapfrog integrator, which is
symplectic and conserves energy to bounded error for the timescales
relevant here (~1 Myr).

Units
-----
Throughout we use AU, solar masses, and years -- in which
G = 4 pi^2 / M_sun (Kepler's third law for a = 1 AU, m = 1 Msun
gives P = 1 yr).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

G_AU3_MSUN_YR2 = 4.0 * math.pi * math.pi  # AU^3 / (Msun * yr^2)


# ----------------------------------------------------------------------
# Kepler equation solver
# ----------------------------------------------------------------------
def solve_kepler(M: float, e: float, tol: float = 1e-12,
                 max_iter: int = 50) -> float:
    """Return the eccentric anomaly E such that M = E - e sin E.

    Uses a Halley iteration with a Danby-style starting guess; converges
    quadratically (effectively cubically) and reaches machine precision
    in 2-4 iterations for e < 0.95.
    """
    # Wrap M to [-pi, pi] for numerical stability.
    M = (M + math.pi) % (2.0 * math.pi) - math.pi
    # Danby's starting guess.
    E = M + 0.85 * e * math.copysign(1.0, math.sin(M))
    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        fpp = e * math.sin(E)
        # Halley step
        delta = -f / (fp - 0.5 * f * fpp / fp)
        E += delta
        if abs(delta) < tol:
            return E
    return E


# ----------------------------------------------------------------------
# Two-body Kepler closed-form
# ----------------------------------------------------------------------
def kepler_two_body(m1: float, m2: float,
                    a: float, e: float, t: float,
                    omega: float = 0.0,
                    t_peri: float = 0.0
                    ) -> Tuple[np.ndarray, np.ndarray,
np.ndarray, np.ndarray]:
    """Closed-form positions and velocities of two bodies at time t.

    Parameters
    ----------
    m1, m2 : Msun
    a      : semi-major axis of the relative orbit (AU)
    e      : eccentricity
    t      : time (yr) since t = 0
    omega  : argument of pericentre measured from +x (rad)
    t_peri : time of pericentre passage (yr)

    Returns
    -------
    r1, r2, v1, v2 : ndarrays, shape (3,)
        Positions (AU) and velocities (AU/yr) of body 1 and body 2 in
        the barycentric frame, with the orbital plane = (x, y).
    """
    if a <= 0 or e < 0 or e >= 1.0:
        raise ValueError("Need a > 0 and 0 <= e < 1 for bound orbit.")
    M_total = m1 + m2
    n = math.sqrt(G_AU3_MSUN_YR2 * M_total / a ** 3)  # rad / yr
    M = n * (t - t_peri)
    E = solve_kepler(M, e)
    cosE, sinE = math.cos(E), math.sin(E)
    # Position in the orbital plane (pericentre on +x axis before rotation by omega)
    x_p = a * (cosE - e)
    y_p = a * math.sqrt(1.0 - e * e) * sinE
    # Velocity in the orbital plane (analytic derivative of Kepler eq.)
    factor = a * n / (1.0 - e * cosE)
    vx_p = -factor * sinE
    vy_p = factor * math.sqrt(1.0 - e * e) * cosE
    # Rotate by omega.
    co, so = math.cos(omega), math.sin(omega)
    r_rel = np.array([co * x_p - so * y_p, so * x_p + co * y_p, 0.0])
    v_rel = np.array([co * vx_p - so * vy_p, so * vx_p + co * vy_p, 0.0])
    # Split into barycentric coordinates.
    f1 = -m2 / M_total
    f2 = m1 / M_total
    return f1 * r_rel, f2 * r_rel, f1 * v_rel, f2 * v_rel


def orbital_period(m1: float, m2: float, a: float) -> float:
    """Keplerian period (yr) for SMA a (AU) and total mass m1+m2 (Msun)."""
    return 2.0 * math.pi * math.sqrt(a ** 3 / (G_AU3_MSUN_YR2 * (m1 + m2)))


# ----------------------------------------------------------------------
# Initial conditions from osculating elements for >= 3 bodies
# ----------------------------------------------------------------------
def orbital_elements_to_state(m_in: float, m_orbiting: float,
                              a: float, e: float,
                              omega: float = 0.0,
                              mean_anomaly: float = 0.0
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (r, v) of the orbiting body relative to the central mass
    given osculating Keplerian elements.  Coplanar (z = 0)."""
    M = mean_anomaly
    E = solve_kepler(M, e)
    cosE, sinE = math.cos(E), math.sin(E)
    n = math.sqrt(G_AU3_MSUN_YR2 * (m_in + m_orbiting) / a ** 3)
    x_p = a * (cosE - e)
    y_p = a * math.sqrt(1.0 - e * e) * sinE
    factor = a * n / (1.0 - e * cosE)
    vx_p = -factor * sinE
    vy_p = factor * math.sqrt(1.0 - e * e) * cosE
    co, so = math.cos(omega), math.sin(omega)
    r = np.array([co * x_p - so * y_p, so * x_p + co * y_p, 0.0])
    v = np.array([co * vx_p - so * vy_p, so * vx_p + co * vy_p, 0.0])
    return r, v


# ----------------------------------------------------------------------
# Simple, fast symplectic N-body integrator (kick-drift-kick leapfrog)
# ----------------------------------------------------------------------
def nbody_step(positions: np.ndarray, velocities: np.ndarray,
               masses: np.ndarray, dt: float) -> None:
    """Advance an N-body system in-place by one leapfrog (KDK) step.

    Arrays are shape (N, 3); masses shape (N,).  Soft-coreless: the
    code assumes the bodies never approach within machine-epsilon of
    each other (which is true for stellar systems on bound orbits).
    """
    a = _accelerations(positions, masses)
    velocities += 0.5 * dt * a
    positions += dt * velocities
    a = _accelerations(positions, masses)
    velocities += 0.5 * dt * a


def _accelerations(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    n = positions.shape[0]
    a = np.zeros_like(positions)
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            dr = positions[j] - positions[i]
            r2 = dr.dot(dr)
            inv_r3 = r2 ** -1.5
            a[i] += G_AU3_MSUN_YR2 * masses[j] * dr * inv_r3
    return a


def nbody_orbit(positions0: np.ndarray, velocities0: np.ndarray,
                masses: np.ndarray,
                t_end: float, n_samples: int = 256
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the system from t=0 to t=t_end and return positions at
    `n_samples` equally-spaced times.

    Returns
    -------
    times      : ndarray, shape (n_samples,)
    snapshots  : ndarray, shape (n_samples, N, 3)
    """
    n = positions0.shape[0]
    positions = positions0.copy()
    velocities = velocities0.copy()
    # Adaptive dt selection: 1/200 of the shortest pair orbital period.
    pmin = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            p = 2.0 * math.pi * math.sqrt(r ** 3 /
                                          (G_AU3_MSUN_YR2 *
                                           (masses[i] + masses[j])))
            pmin = min(pmin, p)
    dt = pmin / 200.0
    n_steps_total = max(int(t_end / dt) + 1, n_samples * 2)
    dt = t_end / n_steps_total
    snapshot_interval = max(1, n_steps_total // n_samples)
    snapshots = []
    times = []
    t = 0.0
    for step in range(n_steps_total + 1):
        if step % snapshot_interval == 0 and len(snapshots) < n_samples:
            snapshots.append(positions.copy())
            times.append(t)
        if step < n_steps_total:
            nbody_step(positions, velocities, masses, dt)
            t += dt
    return np.array(times), np.array(snapshots)
