"""
nbody_moons.py
--------------
Full *nested* N-body integrator for the random solar-system generator.

Hierarchy
---------
    stars  -- analytical closed-form Kepler orbits (never drift; exact for
              N<=2 and for Mardling-Aarseth-stable triples), reusing
              nbody.StarTrajectory.
    planets -- integrated numerically; feel every star + every other
               planet + their own moons (back-reaction).
    moons   -- integrated numerically; feel their host planet + sibling
               moons + every star + every other planet.

So every moon is a genuine N-body particle in the *combined,
time-varying* field of the whole system -- this is what verifies that
the generated moon systems are globally stable, not just pairwise.

Integration scheme
------------------
Kick-drift-kick leapfrog (symplectic, bounded energy error over secular
timescales -- identical scheme/justification as nbody.integrate_planets),
with the sub-step count fixed so dt <= P_shortest_moon / 40 (the same
"40 steps per shortest period" rule used in animation.py).  Pairwise
body-body accelerations are fully vectorised.

Because a close-in giant moon has a ~1-day period while the long-term
horizon is many years, integrating *every* one of 100+ moons over the
full horizon is deliberately costly.  `max_integrated_moons_per_planet`
caps how many of the dynamically-relevant moons (largest regulars +
a sample of irregulars) enter the heavy integration; the remainder are
already guaranteed stable analytically by construction in moons.py
(Roche + Domingos-2006 + mutual-Hill).  The report records the verified
horizon and body count explicitly.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from goldilocks.kepler import G_AU3_MSUN_YR2, orbital_period
from goldilocks.nbody import StarTrajectory, planet_initial_state
from goldilocks.planets import M_EARTH_OVER_M_SUN, is_gas_giant


# ---------------------------------------------------------------------
# Body bookkeeping
# ---------------------------------------------------------------------
@dataclass
class _Bodies:
    pos0: np.ndarray            # (Nb, 3) AU
    vel0: np.ndarray            # (Nb, 3) AU/yr
    mass_msun: np.ndarray       # (Nb,)
    kind: List[str]             # "planet" | "moon"
    center: List[int]           # planet idx of moon's host, else -1
    label: List[str]
    planet_rows: List[int]      # row index of each planet (order = sys.planets)


def _build_bodies(sys, rng: np.random.Generator,
                  max_moons_per_planet: int) -> _Bodies:
    sys._update_stellar_positions(0.0)
    star_masses = np.array([s.mass for s in sys.stars])
    m_tot = float(star_masses.sum())
    bary = sys.barycentre()

    pos, vel, mass, kind, center, label = [], [], [], [], [], []
    planet_rows: List[int] = []

    for p in sys.planets:
        if p.semi_major_axis_au is None:
            planet_rows.append(-1)
            continue
        if p.is_circumbinary():
            host_pos = np.array(bary)
            host_vel = np.zeros(3)
            m_host = m_tot
        else:
            h = sys.stars[p.host_star_index]
            host_pos = np.array(h.position)
            host_vel = np.array(h.velocity)
            m_host = h.mass
        r, v = planet_initial_state(
            host_pos=host_pos, host_vel=host_vel, m_host_msun=m_host,
            a_au=p.semi_major_axis_au, e=p.eccentricity, omega=0.0,
            inclination_rad=math.radians(p.inclination_deg),
            lon_ascending_node_rad=float(rng.uniform(0, 2 * math.pi)),
            mean_anomaly_rad=float(rng.uniform(0, 2 * math.pi)))
        p_row = len(pos)
        planet_rows.append(p_row)
        pos.append(r); vel.append(v)
        mass.append(p.mass_me * M_EARTH_OVER_M_SUN)
        kind.append("planet"); center.append(-1); label.append(p.name)

        # ----- moons of this planet -----
        chosen = _select_moons(p.moons, max_moons_per_planet)
        m_planet_msun = p.mass_me * M_EARTH_OVER_M_SUN
        for mn in chosen:
            inc = math.radians(mn.inclination_deg)   # >90deg => retrograde
            rm, vm = planet_initial_state(
                host_pos=r, host_vel=v, m_host_msun=m_planet_msun,
                a_au=mn.a_planet_au, e=mn.eccentricity, omega=0.0,
                inclination_rad=inc,
                lon_ascending_node_rad=float(rng.uniform(0, 2 * math.pi)),
                mean_anomaly_rad=float(rng.uniform(0, 2 * math.pi)))
            pos.append(rm); vel.append(vm)
            mass.append(mn.mass_me * M_EARTH_OVER_M_SUN)
            kind.append("moon"); center.append(p_row); label.append(mn.name)

    if not pos:
        return _Bodies(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0),
                       [], [], [], planet_rows)
    return _Bodies(np.array(pos), np.array(vel), np.array(mass),
                   kind, center, label, planet_rows)


def _select_moons(moons: list, cap: int) -> list:
    """Largest regulars first, then a sample of irregulars, up to `cap`."""
    if cap <= 0 or not moons:
        return []
    regs = sorted([m for m in moons if m.kind == "regular"],
                  key=lambda m: -m.mass_me)
    irrs = sorted([m for m in moons if m.kind != "regular"],
                  key=lambda m: -m.mass_me)
    out = regs[:cap]
    if len(out) < cap:
        out += irrs[: cap - len(out)]
    return out


# ---------------------------------------------------------------------
# Vectorised acceleration: stars (analytic) + all bodies (pairwise)
# ---------------------------------------------------------------------
def _accels(body_pos: np.ndarray, body_mass: np.ndarray,
            star_pos: np.ndarray, star_mass: np.ndarray) -> np.ndarray:
    n = body_pos.shape[0]
    acc = np.zeros_like(body_pos)
    # Stars (analytic positions supplied for this instant)
    for k in range(star_pos.shape[0]):
        dr = star_pos[k][None, :] - body_pos
        r2 = np.einsum("ij,ij->i", dr, dr)
        inv = np.where(r2 > 1e-18, r2 ** -1.5, 0.0)
        acc += G_AU3_MSUN_YR2 * star_mass[k] * dr * inv[:, None]
    # Body-body, fully vectorised O(n^2).  d[i,j] = r_j - r_i.
    if n > 1:
        d = body_pos[None, :, :] - body_pos[:, None, :]      # (i,j,3)
        r2 = np.einsum("ijk,ijk->ij", d, d)
        np.fill_diagonal(r2, np.inf)
        inv = r2 ** -1.5
        acc += G_AU3_MSUN_YR2 * np.einsum("ijk,ij,j->ik", d, inv, body_mass)
    return acc


def integrate_solar_system(sys,
                           duration_yr: float,
                           n_samples: int = 600,
                           max_integrated_moons_per_planet: int = 12,
                           sub_steps_floor: int = 60,
                           rng: Optional[np.random.Generator] = None
                           ) -> dict:
    """Integrate the full nested system and return histories + report.

    Returns a dict with: times, star_hist (T,Ns,3), planet_hist (T,Np,3),
    moon arrays, body metadata, and `report` (per-body stability verdict).
    """
    if rng is None:
        rng = np.random.default_rng(2026)
    star_masses = np.array([s.mass for s in sys.stars])
    traj = StarTrajectory(star_masses, sys.stellar_orbits)

    B = _build_bodies(sys, rng, max_integrated_moons_per_planet)
    times = np.linspace(0.0, float(duration_yr), n_samples)

    n_s = traj.n
    star_hist = np.array([traj.positions(t) for t in times])
    if B.pos0.shape[0] == 0:
        return {"times": times, "star_hist": star_hist,
                "body": B, "body_hist": np.zeros((n_samples, 0, 3)),
                "report": {"stable": True, "bodies": [],
                           "horizon_yr": float(duration_yr)}}

    # dt from the shortest moon (or planet) period.
    min_P = np.inf
    for row, (k, c) in enumerate(zip(B.kind, B.center)):
        if k == "moon":
            m_host = B.mass_msun[c]
            a = float(np.linalg.norm(B.pos0[row] - B.pos0[c]))
        else:
            m_host = max(float(star_masses.sum()), 1e-6)
            a = float(np.linalg.norm(B.pos0[row]))
        if a > 0 and m_host > 0:
            min_P = min(min_P, orbital_period(m_host, 0.0, a))
    if not np.isfinite(min_P):
        min_P = duration_yr
    dt_sample = duration_yr / max(n_samples - 1, 1)
    n_sub = max(sub_steps_floor, int(math.ceil(dt_sample / (min_P / 40.0))))
    n_sub = min(n_sub, 4000)

    pos = B.pos0.copy()
    vel = B.vel0.copy()
    body_hist = np.zeros((n_samples, B.pos0.shape[0], 3))
    vel_hist = np.zeros_like(body_hist)
    body_hist[0] = pos
    vel_hist[0] = vel

    for ti in range(1, n_samples):
        t0, t1 = times[ti - 1], times[ti]
        dt = (t1 - t0) / n_sub
        for s in range(n_sub):
            ta = t0 + s * dt
            tc = t0 + (s + 1) * dt
            a1 = _accels(pos, B.mass_msun, traj.positions(ta), star_masses)
            vel += 0.5 * dt * a1
            pos += dt * vel
            a2 = _accels(pos, B.mass_msun, traj.positions(tc), star_masses)
            vel += 0.5 * dt * a2
        body_hist[ti] = pos
        vel_hist[ti] = vel

    report = _stability_report(sys, B, body_hist, vel_hist, star_hist,
                               star_masses, traj, times,
                               float(duration_yr), n_sub)
    return {"times": times, "star_hist": star_hist, "body": B,
            "body_hist": body_hist, "vel_hist": vel_hist, "report": report}


def _analytic_star_vel(traj: StarTrajectory, star_masses: np.ndarray,
                        times: np.ndarray) -> np.ndarray:
    """Analytic-orbit star velocities via a fine central difference of the
    closed-form trajectory (independent of the sample cadence)."""
    P_min = np.inf
    for (i, j, a, e, _) in traj.orbits:
        m1 = traj.masses[i]
        m2 = (traj.masses[j] if j >= 0
              else float(traj.masses.sum()) - traj.masses[i])
        P_min = min(P_min, orbital_period(m1, m2, a))
    h = 1e-4 * (P_min if np.isfinite(P_min) else 1.0)
    h = max(h, 1e-7)
    v = np.zeros((len(times), traj.n, 3))
    for i, t in enumerate(times):
        v[i] = (traj.positions(t + h) - traj.positions(t - h)) / (2 * h)
    return v


def _center_state(sys, B: _Bodies, body_hist: np.ndarray,
                  vel_hist: np.ndarray, star_hist: np.ndarray,
                  star_vel: np.ndarray, star_masses: np.ndarray,
                  row: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Centre position, centre velocity, and GM (AU^3/yr^2) for body `row`.

    Returns (pos_t, vel_t, mu) where mu = G(M_centre + m_body)."""
    if B.kind[row] == "moon":
        c = B.center[row]
        mu = G_AU3_MSUN_YR2 * (B.mass_msun[c] + B.mass_msun[row])
        return body_hist[:, c, :], vel_hist[:, c, :], mu
    # planet -> host star, or stellar barycentre (circumbinary)
    planet = None
    for pp in sys.planets:
        if (pp.semi_major_axis_au is not None
                and B.planet_rows[sys.planets.index(pp)] == row):
            planet = pp
            break
    if planet is None or planet.is_circumbinary():
        w = star_masses / star_masses.sum()
        cpos = np.einsum("tsk,s->tk", star_hist, w)
        cvel = np.einsum("tsk,s->tk", star_vel, w)
        m_c = float(star_masses.sum())
    else:
        cpos = star_hist[:, planet.host_star_index, :]
        cvel = star_vel[:, planet.host_star_index, :]
        m_c = float(star_masses[planet.host_star_index])
    mu = G_AU3_MSUN_YR2 * (m_c + B.mass_msun[row])
    return cpos, cvel, mu


def _stability_report(sys, B: _Bodies, body_hist: np.ndarray,
                       vel_hist: np.ndarray, star_hist: np.ndarray,
                       star_masses: np.ndarray,
                       traj: StarTrajectory, times: np.ndarray,
                       horizon_yr: float, n_sub: int) -> dict:
    """Verdict from period-independent osculating elements.

    a_osc from vis-viva (1/a = 2/r - v^2/mu) is invariant over an orbit,
    so a long-period outer planet sampled for < 1 period is *not*
    spuriously flagged.  Instability = unbound (a<=0), e->1, or a
    real secular spread in a_osc.
    """
    star_vel = _analytic_star_vel(traj, star_masses, times)
    bodies = []
    all_stable = True
    for row in range(B.pos0.shape[0]):
        cpos, cvel, mu = _center_state(sys, B, body_hist, vel_hist,
                                       star_hist, star_vel, star_masses,
                                       row)
        rvec = body_hist[:, row, :] - cpos
        vvec = vel_hist[:, row, :] - cvel
        r = np.linalg.norm(rvec, axis=1)
        v2 = np.einsum("ij,ij->i", vvec, vvec)
        inv_a = 2.0 / np.maximum(r, 1e-12) - v2 / mu
        bound = inv_a > 0
        a_osc = np.where(bound, 1.0 / np.where(bound, inv_a, 1.0), np.nan)
        h = np.linalg.norm(np.cross(rvec, vvec), axis=1)
        with np.errstate(invalid="ignore"):
            e_osc = np.sqrt(np.clip(1.0 - h * h / (mu * a_osc), 0.0, None))
        a_med = float(np.nanmedian(a_osc)) if np.any(bound) else -1.0
        e_med = float(np.nanmedian(e_osc)) if np.any(bound) else 1.0
        n_t = a_osc.shape[0]
        w = max(2, n_t // 4)
        if a_med > 0 and np.any(bound):
            a_fin = a_osc[np.isfinite(a_osc)]
            libration = float((a_fin.max() - a_fin.min()) / a_med)
            # Secular drift: median over the first vs last quarter (each
            # spanning many orbits) averages out bounded libration, so a
            # stable Hill-region moon is not flagged -- only a genuine
            # runaway in <a> is.
            with np.errstate(invalid="ignore"):
                a_head = np.nanmedian(a_osc[:w]) if np.any(
                    np.isfinite(a_osc[:w])) else np.nan
                a_tail = np.nanmedian(a_osc[-w:]) if np.any(
                    np.isfinite(a_osc[-w:])) else np.nan
            secular = (abs(a_tail - a_head) / a_med
                       if np.isfinite(a_head) and np.isfinite(a_tail)
                       else 9.99)
        else:
            libration, secular = 9.99, 9.99
        ejected = bool((~bound).mean() > 0.1) or a_med <= 0.0
        stable = (not ejected) and secular < 0.15 and e_med < 0.9
        all_stable &= stable
        bodies.append({
            "label": B.label[row], "kind": B.kind[row],
            "a_au": max(a_med, 0.0), "e": min(e_med, 1.0),
            "rel_drift": secular, "a_libration": libration,
            "ejected": ejected, "stable": stable})
    return {"stable": all_stable, "bodies": bodies,
            "horizon_yr": horizon_yr, "sub_steps": n_sub,
            "n_bodies": B.pos0.shape[0]}
