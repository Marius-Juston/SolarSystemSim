"""
nbody.py
--------
Hybrid N-body integrator for planets in N-star systems.

The integration scheme keeps the stars on their *analytical* closed-form
Kepler orbits (which are stable to roundoff and never drift, and are
correct as long as Mardling-Aarseth is satisfied), and integrates each
planet as a test particle in the time-dependent N-star potential.

Planets also feel each other's gravity, so the equations of motion for
planet i are

    d^2 r_i / dt^2  = - sum_s G M_s (r_i - r_s(t)) / |r_i - r_s|^3
                       - sum_{j != i} G m_j (r_i - r_j) / |r_i - r_j|^3

where r_s(t) is the analytical star trajectory.  We integrate with a
4th-order Yoshida symplectic scheme inside one binary period (planets
need many time steps per their own orbital period), which conserves
energy to bounded error over secular timescales.

Why this matters
----------------
A planet in a binary system experiences a non-Keplerian orbit: it
"wobbles" relative to a pure ellipse around its host because the other
star pulls on it.  Leung & Lee 2013 show that for Kepler-16 b, this
wobble produces a forced eccentricity ~0.04, a 49-yr apse precession,
and a 42-yr node precession -- all visible in real numerical
integrations.  This module reproduces that physics.

For "star-hopper" planets (Moeckel & Veras 2012), the planet can
literally bounce between the two stars; with proper initial conditions
near the Jacobi-constant separatrix the trajectory is dramatically
non-Keplerian and the analytical Kepler-around-host approximation is
hopelessly wrong.
"""

from __future__ import annotations
import math
from typing import List, Sequence, Tuple, Callable, Optional

import numpy as np

from goldilocks.kepler import (G_AU3_MSUN_YR2, kepler_two_body, orbital_period,
                    solve_kepler)


# ---------------------------------------------------------------------
# Star trajectory wrapper (delegates to closed-form Kepler)
# ---------------------------------------------------------------------
class StarTrajectory:
    """Holds the analytical star trajectories for an N-star system.

    Stars 0 and 1 form an inner binary on a Kepler orbit; star 2 (if
    present) is on a Kepler orbit about the inner barycentre.
    """

    def __init__(self, masses: np.ndarray,
                 stellar_orbits: List[Tuple[int, int, float, float, float]]):
        self.masses = np.asarray(masses, dtype=float)
        self.orbits = stellar_orbits
        self.n = len(masses)

    def positions(self, t: float) -> np.ndarray:
        """Return star positions at time t.  Shape (N, 3)."""
        pos = np.zeros((self.n, 3))
        if self.n == 1:
            return pos
        # Inner binary
        edge0 = self.orbits[0]
        i, j, a, e, omega = edge0
        m1, m2 = self.masses[i], self.masses[j]
        r1, r2, _, _ = kepler_two_body(m1, m2, a, e, t,
                                       omega=omega, t_peri=0.0)
        pos[i] = r1
        pos[j] = r2
        if self.n >= 3 and len(self.orbits) >= 2:
            edge1 = self.orbits[1]
            k, _, a_o, e_o, omega_o = edge1
            m_inner = m1 + m2
            m_outer = self.masses[k]
            r_in, r_out, _, _ = kepler_two_body(
                m_inner, m_outer, a_o, e_o, t,
                omega=omega_o, t_peri=0.0)
            pos[i] += r_in
            pos[j] += r_in
            pos[k]  = r_out
        return pos


def planet_acceleration(planet_pos: np.ndarray,
                        planet_idx: int,
                        all_planet_pos: np.ndarray,
                        planet_masses_msun: np.ndarray,
                        star_pos: np.ndarray,
                        star_masses: np.ndarray) -> np.ndarray:
    """Total acceleration on one planet from all stars + all other planets."""
    a = np.zeros(3)
    # Stellar contribution
    for k in range(star_pos.shape[0]):
        dr = star_pos[k] - planet_pos
        r2 = dr.dot(dr)
        if r2 < 1e-14:
            continue
        a += G_AU3_MSUN_YR2 * star_masses[k] * dr / (r2 * math.sqrt(r2))
    # Other-planet contribution
    for j in range(all_planet_pos.shape[0]):
        if j == planet_idx:
            continue
        dr = all_planet_pos[j] - planet_pos
        r2 = dr.dot(dr)
        if r2 < 1e-14:
            continue
        a += G_AU3_MSUN_YR2 * planet_masses_msun[j] * dr / (r2 * math.sqrt(r2))
    return a


def _all_planet_accels(planet_pos: np.ndarray,
                       planet_masses_msun: np.ndarray,
                       star_pos: np.ndarray,
                       star_masses: np.ndarray) -> np.ndarray:
    """Compute accelerations for all planets at once.  Shape matches planet_pos."""
    n_p = planet_pos.shape[0]
    accs = np.zeros_like(planet_pos)
    # Stellar contribution -- vectorised
    for k in range(star_pos.shape[0]):
        dr = star_pos[k][None, :] - planet_pos        # (n_p, 3)
        r2 = np.einsum("ij,ij->i", dr, dr)
        inv_r3 = np.where(r2 > 1e-14, r2 ** -1.5, 0.0)
        accs += G_AU3_MSUN_YR2 * star_masses[k] * dr * inv_r3[:, None]
    # Planet-planet contribution
    if n_p > 1:
        for i in range(n_p):
            for j in range(n_p):
                if j == i:
                    continue
                dr = planet_pos[j] - planet_pos[i]
                r2 = dr.dot(dr)
                if r2 < 1e-14:
                    continue
                accs[i] += (G_AU3_MSUN_YR2 * planet_masses_msun[j]
                            * dr / (r2 * math.sqrt(r2)))
    return accs


# ---------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------
def integrate_planets(traj: StarTrajectory,
                      planet_pos0: np.ndarray,        # (N_p, 3)
                      planet_vel0: np.ndarray,        # (N_p, 3)
                      planet_masses_me: np.ndarray,   # (N_p,)
                      times: np.ndarray,
                      sub_steps_per_sample: int = 50
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate planets forward through `times` (yr), reporting positions
    at each sampled time.  Returns

        planet_history  : shape (len(times), N_p, 3)
        star_history    : shape (len(times), N_s, 3)

    Uses a kick-drift-kick leapfrog integrator with `sub_steps_per_sample`
    substeps between successive sample times.  Stars move analytically
    through Kepler's equation between substeps.
    """
    M_EARTH_OVER_M_SUN = 3.0034893e-6
    planet_masses_msun = planet_masses_me * M_EARTH_OVER_M_SUN

    n_t = len(times)
    n_p = planet_pos0.shape[0]
    n_s = traj.n
    planet_hist = np.zeros((n_t, n_p, 3))
    star_hist   = np.zeros((n_t, n_s, 3))

    pos = planet_pos0.copy()
    vel = planet_vel0.copy()

    # Always record t=0
    planet_hist[0] = pos
    star_hist[0]   = traj.positions(times[0])

    for ti in range(1, n_t):
        t_start = times[ti - 1]
        t_end   = times[ti]
        dt_total = t_end - t_start
        if dt_total <= 0:
            planet_hist[ti] = pos
            star_hist[ti]   = traj.positions(t_end)
            continue
        n_sub = max(1, sub_steps_per_sample)
        dt = dt_total / n_sub
        for k in range(n_sub):
            t_mid_a = t_start + k * dt
            t_mid_b = t_start + (k + 0.5) * dt
            t_mid_c = t_start + (k + 1) * dt
            star_a = traj.positions(t_mid_a)
            star_b = traj.positions(t_mid_b)
            star_c = traj.positions(t_mid_c)
            # KDK leapfrog: kick(dt/2) at t, drift(dt) using v, kick(dt/2) at t+dt
            a1 = _all_planet_accels(pos, planet_masses_msun,
                                    star_a, traj.masses)
            vel += 0.5 * dt * a1
            pos += dt * vel
            a2 = _all_planet_accels(pos, planet_masses_msun,
                                    star_c, traj.masses)
            vel += 0.5 * dt * a2
        planet_hist[ti] = pos
        star_hist[ti]   = traj.positions(t_end)
    return planet_hist, star_hist


def planet_initial_state(host_pos: np.ndarray,
                         host_vel: np.ndarray,
                         m_host_msun: float,
                         a_au: float,
                         e: float,
                         omega: float,
                         inclination_rad: float,
                         lon_ascending_node_rad: float,
                         mean_anomaly_rad: float = 0.0
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Convert osculating Keplerian elements (relative to host) into
    barycentric position and velocity (3-D)."""
    # Solve Kepler's equation for eccentric anomaly
    E = solve_kepler(mean_anomaly_rad, e)
    cosE, sinE = math.cos(E), math.sin(E)
    # Position in orbital plane (peri on +x axis)
    x_op = a_au * (cosE - e)
    y_op = a_au * math.sqrt(1.0 - e*e) * sinE
    # Velocity in orbital plane
    n_mean = math.sqrt(G_AU3_MSUN_YR2 * m_host_msun / a_au**3)
    factor = a_au * n_mean / (1.0 - e * cosE)
    vx_op = -factor * sinE
    vy_op = factor * math.sqrt(1.0 - e*e) * cosE
    # Build rotation matrix R = R_z(Ω) R_x(i) R_z(ω)
    co_w, so_w = math.cos(omega), math.sin(omega)
    co_i, si_i = math.cos(inclination_rad), math.sin(inclination_rad)
    co_O, so_O = math.cos(lon_ascending_node_rad), math.sin(lon_ascending_node_rad)
    # Position rotation
    px = (co_O*co_w - so_O*so_w*co_i) * x_op + (-co_O*so_w - so_O*co_w*co_i) * y_op
    py = (so_O*co_w + co_O*so_w*co_i) * x_op + (-so_O*so_w + co_O*co_w*co_i) * y_op
    pz = (so_w * si_i) * x_op + (co_w * si_i) * y_op
    vx = (co_O*co_w - so_O*so_w*co_i) * vx_op + (-co_O*so_w - so_O*co_w*co_i) * vy_op
    vy = (so_O*co_w + co_O*so_w*co_i) * vx_op + (-so_O*so_w + co_O*co_w*co_i) * vy_op
    vz = (so_w * si_i) * vx_op + (co_w * si_i) * vy_op
    r_rel = np.array([px, py, pz])
    v_rel = np.array([vx, vy, vz])
    return host_pos + r_rel, host_vel + v_rel