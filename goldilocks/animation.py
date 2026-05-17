"""
animation.py
------------
Time-evolution animation of an N-star + N-planet system with:

  * **Real N-body planet dynamics**.  Each planet is integrated as a
    test particle in the *combined* gravitational field of every star
    plus every other planet.  This produces actual non-Keplerian
    wobble -- forced eccentricity, apse and node precession, and (in
    star-hopper systems) the dramatic bouncing motion documented by
    Moeckel & Veras 2012.

  * **3-D inclinations**.  Each planet has a real inclination and
    longitude of ascending node, drawn from realistic distributions
    (~1-8 deg mutual inclination per Solar System and exoplanet
    statistics).  An edge-on side view (x-z) is rendered alongside the
    top-down view so inclinations are visible.

  * **Multi-scale views** for hierarchical systems.  Alpha Cen + Proxima
    has 23 AU inner separation and 8700 AU outer separation -- nearly
    four orders of magnitude.  When the outer-to-inner ratio exceeds
    10, a separate "wide view" panel is added showing the outer body.

  * **Barycentre recentering**.  In hierarchical triples, the inner
    binary's barycentre is offset from the system barycentre by the
    outer Kepler orbit.  The inner-zoom panels are always re-centered
    on the inner-binary barycentre so the inner system is visible
    regardless of where in the outer orbit we are.

Output is an MP4 via ffmpeg.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple, List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from   matplotlib.animation import FuncAnimation, FFMpegWriter

from goldilocks.system           import StarSystem
import goldilocks.habitable_zone as hz
import goldilocks.secular        as sec
from goldilocks.kepler           import (orbital_period, solve_kepler, kepler_two_body,
                              G_AU3_MSUN_YR2)
from goldilocks.nbody            import (StarTrajectory, planet_initial_state,
                              integrate_planets)


M_EARTH_OVER_M_SUN = 3.0034893e-6


STAR_COLORS = {
    "O": "#9BB0FF", "B": "#AABFFF", "A": "#CAD7FF",
    "F": "#F8F7FF", "G": "#FFF4EA", "K": "#FFD2A1", "M": "#FF8855",
}


def _star_color(teff_k: float) -> str:
    if teff_k > 30000: return STAR_COLORS["O"]
    if teff_k > 10000: return STAR_COLORS["B"]
    if teff_k >  7500: return STAR_COLORS["A"]
    if teff_k >  6000: return STAR_COLORS["F"]
    if teff_k >  5200: return STAR_COLORS["G"]
    if teff_k >  3700: return STAR_COLORS["K"]
    return STAR_COLORS["M"]


def _star_marker_size(L: float, scale: float = 1.0) -> float:
    """Marker area in points^2; scaled by `scale` for inset panels."""
    return scale * max(120.0, 120.0 * math.log10(L + 1.5) + 130.0)


# -----------------------------------------------------------------------
# Compute initial states for all planets, including planets-per-host
# from the PHZ packing and any catalogued "known" planets (Kepler-16 b
# etc.)
# -----------------------------------------------------------------------
def _setup_planets(sys: StarSystem,
                   res: dict,
                   t0: float,
                   rng: np.random.Generator
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:

    sys._update_stellar_positions(t0)
    omega_bin = sys.stellar_orbits[0][4] if sys.stellar_orbits else 0.0
    a_bin     = sys.stellar_orbits[0][2] if sys.stellar_orbits else None
    e_bin     = sys.stellar_orbits[0][3] if sys.stellar_orbits else 0.0

    positions, velocities, masses, meta = [], [], [], []

    # ----- S-type packed planets -----
    for host_idx, entry in enumerate(res["stars"]):
        if entry["stable_HZ"] is None or not entry.get("positions"):
            continue
        host = sys.stars[host_idx]
        host_pos = np.array(host.position)
        host_vel = np.array(host.velocity)
        for i, (a_p, e_max) in enumerate(zip(entry["positions"],
                                              entry["e_max"])):
            if a_bin is not None:
                e0 = sec.heppenheimer_e_forced_stype(a_p, a_bin, e_bin)
            else:
                # Single-star: small "natural" eccentricity
                e0 = float(rng.uniform(0.01, 0.08))
            # Visible mutual inclination 3-8 deg
            inc = math.radians(rng.uniform(3.0, 8.0))
            Omega = rng.uniform(0.0, 2.0 * math.pi)
            M = rng.uniform(0.0, 2.0 * math.pi)
            r, v = planet_initial_state(
                host_pos=host_pos, host_vel=host_vel,
                m_host_msun=host.mass, a_au=a_p, e=e0,
                omega=omega_bin, inclination_rad=inc,
                lon_ascending_node_rad=Omega, mean_anomaly_rad=M)
            positions.append(r)
            velocities.append(v)
            masses.append(1.0)
            meta.append({"kind": "S", "host_idx": host_idx,
                          "a": a_p, "e0": e0, "inc_rad": inc,
                          "color": "#56C7FF",
                          "label": f"{host.name} p{i+1}"})

    # ----- P-type packed planets -----
    if res.get("circumbinary") and res["circumbinary"].get("positions"):
        bary = sys.barycentre()
        m_tot = sum(s.mass for s in sys.stars)
        mu = (min(sys.stars[0].mass, sys.stars[1].mass) / m_tot
              if len(sys.stars) >= 2 else 0.0)
        for i, (a_p, e_max) in enumerate(zip(res["circumbinary"]["positions"],
                                              res["circumbinary"]["e_max"])):
            if a_bin is not None:
                e0 = sec.leung_lee_e_forced_ptype(a_p, a_bin, e_bin, mu)
            else:
                e0 = 0.0
            inc = math.radians(rng.uniform(1.5, 5.0))
            Omega = rng.uniform(0.0, 2.0 * math.pi)
            M = rng.uniform(0.0, 2.0 * math.pi)
            r, v = planet_initial_state(
                host_pos=bary, host_vel=np.zeros(3),
                m_host_msun=m_tot, a_au=a_p, e=e0,
                omega=omega_bin, inclination_rad=inc,
                lon_ascending_node_rad=Omega, mean_anomaly_rad=M)
            positions.append(r)
            velocities.append(v)
            masses.append(1.0)
            meta.append({"kind": "P", "host_idx": None,
                          "a": a_p, "e0": e0, "inc_rad": inc,
                          "color": "#FF8A33",
                          "label": f"CBP {i+1}"})

    # ----- Known catalogued planets -----
    for p in sys.planets:
        if p.semi_major_axis_au is None:
            continue
        a_p = p.semi_major_axis_au
        e_p = p.eccentricity
        # Use modest inclination for visualization
        inc = math.radians(rng.uniform(1.0, 4.0))
        Omega = rng.uniform(0.0, 2.0 * math.pi)
        M = 0.0
        if p.is_circumbinary():
            bary = sys.barycentre()
            m_host = sum(s.mass for s in sys.stars)
            r, v = planet_initial_state(
                host_pos=bary, host_vel=np.zeros(3),
                m_host_msun=m_host, a_au=a_p, e=e_p,
                omega=omega_bin, inclination_rad=inc,
                lon_ascending_node_rad=Omega, mean_anomaly_rad=M)
        else:
            host = sys.stars[p.host_star_index]
            r, v = planet_initial_state(
                host_pos=np.array(host.position),
                host_vel=np.array(host.velocity),
                m_host_msun=host.mass, a_au=a_p, e=e_p,
                omega=omega_bin, inclination_rad=inc,
                lon_ascending_node_rad=Omega, mean_anomaly_rad=M)
        positions.append(r)
        velocities.append(v)
        masses.append(p.mass_me)
        meta.append({"kind": "KNOWN", "host_idx": p.host_star_index,
                      "a": a_p, "e0": e_p, "inc_rad": inc,
                      "color": "#FF44AA", "label": p.name})

    if not positions:
        return (np.zeros((0, 3)), np.zeros((0, 3)),
                np.zeros(0), [])
    return (np.array(positions), np.array(velocities),
            np.array(masses), meta)


# -----------------------------------------------------------------------
# Compute the inner-binary barycentre at time t (for recentering the
# inner-zoom panels of hierarchical systems)
# -----------------------------------------------------------------------
def _inner_bary(star_pos: np.ndarray,
                star_masses: np.ndarray,
                n_inner: int) -> np.ndarray:
    """Mass-weighted barycentre of the first n_inner stars."""
    if n_inner <= 0:
        return np.zeros(3)
    w = star_masses[:n_inner]
    return (w[:, None] * star_pos[:n_inner]).sum(0) / w.sum()


def _system_scales(sys: StarSystem,
                   extent_au: float
                   ) -> Tuple[float, Optional[float], int]:
    """Return (inner_extent, outer_extent_or_None, n_inner_stars).

    n_inner is how many stars belong to the inner binary (2 for triples,
    or = N for non-hierarchical systems).
    """
    if len(sys.stellar_orbits) < 2:
        return extent_au, None, len(sys.stars)
    inner_a = sys.stellar_orbits[0][2]
    outer_a = sys.stellar_orbits[1][2]
    if outer_a / inner_a < 8.0:
        # Not strongly hierarchical -- show everything together
        e_out = sys.stellar_orbits[1][3]
        out_extent = 1.5 * outer_a * (1.0 + e_out)
        return out_extent, None, len(sys.stars)
    e_out = sys.stellar_orbits[1][3]
    outer_extent = 1.4 * outer_a * (1.0 + e_out)
    return extent_au, outer_extent, 2


# -----------------------------------------------------------------------
# Main animation
# -----------------------------------------------------------------------
def animate_system(sys: StarSystem,
                   save_path: str,
                   planet_mass_me: float = 1.0,
                   delta: float = 10.0,
                   extent_au: Optional[float] = None,
                   n_frames: int = 240,
                   n_periods: float = 1.0,
                   duration_yr: Optional[float] = None,
                   n_grid: int = 121,
                   fps: int = 24,
                   optimistic: bool = False,
                   sub_steps: int = 60,
                   show_side_view: bool = True,
                   ) -> None:
    """Build an MP4 animation.

    Parameters
    ----------
    n_periods   : how many binary periods to span (used if duration_yr
                  not given)
    duration_yr : if set, overrides n_periods (use for single-star
                  systems or to set a custom duration)
    """
    if extent_au is None:
        L_tot = sys.total_luminosity()
        extent_au = max(3.0, 2.5 * math.sqrt(L_tot / 0.30))

    inner_extent, outer_extent, n_inner = _system_scales(sys, extent_au)

    # Timeline
    if duration_yr is not None:
        duration = float(duration_yr)
    elif sys.stellar_orbits:
        P_bin = sys.stellar_orbit_period(0)
        duration = n_periods * P_bin
    else:
        duration = 5.0
    times = np.linspace(0.0, duration, n_frames)
    P_inner = (sys.stellar_orbit_period(0) if sys.stellar_orbits
                else duration)

    # Pre-compute the packing
    res = sys.count_habitable_planets(planet_mass_me=planet_mass_me,
                                       delta=delta,
                                       optimistic=optimistic,
                                       use_phz=True)

    # Set up N-body planet initial conditions
    rng = np.random.default_rng(seed=42)
    pos0, vel0, p_masses_me, meta = _setup_planets(sys, res, t0=0.0,
                                                    rng=rng)

    # Star trajectory
    star_masses = np.array([s.mass for s in sys.stars])
    traj = StarTrajectory(star_masses, sys.stellar_orbits)

    # Run N-body integration up front (faster frame rendering after)
    if pos0.shape[0] > 0:
        # Adaptive sub_steps: ensure we have at least N steps per
        # shortest planet period to avoid integrator error
        min_a = min(m["a"] for m in meta) if meta else 1.0
        # Approximate min host mass
        min_host_mass = min(
            star_masses[m["host_idx"]] if m["kind"] != "P"
            and m["host_idx"] is not None
            else float(np.sum(star_masses)) for m in meta)
        P_min = orbital_period(min_host_mass, 0.0, min_a)
        dt_max = P_min / 40.0  # at least 40 steps per shortest period
        n_sub_needed = max(sub_steps, int(np.ceil(
            duration / n_frames / dt_max)))
        # Cap to avoid pathological cost
        n_sub_needed = min(n_sub_needed, 2000)
        print(f"  N-body integrating {pos0.shape[0]} planets, "
              f"{n_frames} samples, {n_sub_needed} sub-steps each "
              f"(min P_planet = {P_min*365.25:.1f} d)...")
        planet_hist, star_hist = integrate_planets(
            traj, pos0, vel0, p_masses_me, times,
            sub_steps_per_sample=n_sub_needed)
    else:
        planet_hist = np.zeros((n_frames, 0, 3))
        star_hist = np.array([traj.positions(t) for t in times])

    # Pre-compute inner barycentres at every frame (for recentering)
    if n_inner < len(sys.stars):
        inner_bary_hist = np.array([
            _inner_bary(star_hist[i], star_masses, n_inner)
            for i in range(n_frames)])
    else:
        inner_bary_hist = np.zeros((n_frames, 3))

    # PHZ ring definitions (radii fixed; centres move with hosts/bary)
    phz_stype = []
    for host_idx, entry in enumerate(res["stars"]):
        if entry["stable_HZ"] is None:
            continue
        phz_stype.append((host_idx, entry["stable_HZ"][0],
                           entry["stable_HZ"][1]))
    phz_ptype = None
    if res.get("circumbinary"):
        phz_ptype = (res["circumbinary"]["stable_HZ"][0],
                     res["circumbinary"]["stable_HZ"][1])

    # Grid for snapshot HZ (drawn in inner-bary frame for inner view)
    X = np.linspace(-inner_extent, inner_extent, n_grid)
    Y = np.linspace(-inner_extent, inner_extent, n_grid)
    XX, YY = np.meshgrid(X, Y)
    grid = np.stack([XX, YY, np.zeros_like(XX)], axis=-1)

    # =================== Figure layout ===================
    if outer_extent is not None and show_side_view:
        fig = plt.figure(figsize=(18.0, 11.0))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0],
                              height_ratios=[1.0, 0.55],
                              hspace=0.28, wspace=0.18)
        ax_top   = fig.add_subplot(gs[0, 0])
        ax_outer = fig.add_subplot(gs[0, 1])
        ax_side  = fig.add_subplot(gs[1, :])
    elif show_side_view:
        fig = plt.figure(figsize=(11.0, 13.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.55],
                              hspace=0.22)
        ax_top  = fig.add_subplot(gs[0])
        ax_side = fig.add_subplot(gs[1])
        ax_outer = None
    elif outer_extent is not None:
        fig = plt.figure(figsize=(16.0, 8.0))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.2)
        ax_top   = fig.add_subplot(gs[0])
        ax_outer = fig.add_subplot(gs[1])
        ax_side  = None
    else:
        fig, ax_top = plt.subplots(figsize=(9.5, 9.5))
        ax_side  = None
        ax_outer = None

    fig.patch.set_facecolor("#15161c")
    for a in (ax_top, ax_side, ax_outer):
        if a is None: continue
        a.set_facecolor("#0c0d11")
        a.tick_params(colors="white")
        for sp in a.spines.values():
            sp.set_color("white")

    # Top panel setup
    ax_top.set_aspect("equal", "box")
    ax_top.set_xlim(-inner_extent, inner_extent)
    ax_top.set_ylim(-inner_extent, inner_extent)
    ax_top.set_xlabel("x  [AU]", color="white")
    ax_top.set_ylabel("y  [AU]", color="white")
    title = ax_top.set_title(f"{sys.name}  --  t = 0.000 yr"
                              f"  (P_bin = {P_inner:.3g} yr)",
                              color="white", fontsize=12)

    if ax_side is not None:
        ax_side.set_xlim(-inner_extent, inner_extent)
        z_ext = max(0.25 * inner_extent, 0.5)
        ax_side.set_ylim(-z_ext, z_ext)
        ax_side.set_xlabel("x  [AU]", color="white")
        ax_side.set_ylabel("z  [AU]", color="white")
        ax_side.set_title("Edge-on view: orbital inclinations visible",
                           color="white", fontsize=11)

    if ax_outer is not None:
        ax_outer.set_aspect("equal", "box")
        ax_outer.set_xlim(-outer_extent, outer_extent)
        ax_outer.set_ylim(-outer_extent, outer_extent)
        ax_outer.set_xlabel("x  [AU]", color="white")
        ax_outer.set_ylabel("y  [AU]", color="white")
        ax_outer.set_title(f"Hierarchical view (+/-{outer_extent:.0f} AU)",
                            color="white", fontsize=11)

    # PHZ rings (top view, recentered on inner bary frame)
    theta_circle = np.linspace(0, 2*math.pi, 200)
    cos_t = np.cos(theta_circle)
    sin_t = np.sin(theta_circle)
    phz_rings_top = []
    for host_idx, r_in, r_out in phz_stype:
        poly = ax_top.fill(np.zeros(2*len(theta_circle)+1),
                            np.zeros(2*len(theta_circle)+1),
                            color="#1B8B1B", alpha=0.25, zorder=3,
                            edgecolor="#3FAE3F", linewidth=0.7)[0]
        phz_rings_top.append((host_idx, r_in, r_out, poly))
    phz_ring_ptype = None
    if phz_ptype is not None:
        r_in, r_out = phz_ptype
        poly = ax_top.fill(np.zeros(2*len(theta_circle)+1),
                            np.zeros(2*len(theta_circle)+1),
                            color="#1B8B1B", alpha=0.25, zorder=3,
                            edgecolor="#FF8A33", linewidth=0.7)[0]
        phz_ring_ptype = (r_in, r_out, poly)

    dynamic_hz_contour = [None]

    # Star markers + trails per panel
    star_scatters_top, star_trails_top = [], []
    star_scatters_side, star_trails_side = [], []
    star_scatters_outer, star_trails_outer = [], []
    star_history = [[[], [], []] for _ in sys.stars]
    star_history_inner = [[[], [], []] for _ in sys.stars]  # recentered

    for s in sys.stars:
        c = _star_color(s.teff)
        sz = _star_marker_size(s.luminosity)
        sc = ax_top.scatter([], [], s=sz, c=c, edgecolors="white",
                            linewidths=1.2, zorder=10)
        star_scatters_top.append(sc)
        tr, = ax_top.plot([], [], color="white", lw=0.5, alpha=0.40,
                           zorder=6)
        star_trails_top.append(tr)
        if ax_side is not None:
            sc = ax_side.scatter([], [], s=sz, c=c, edgecolors="white",
                                  linewidths=1.0, zorder=10)
            star_scatters_side.append(sc)
            tr, = ax_side.plot([], [], color="white", lw=0.5, alpha=0.35,
                                zorder=6)
            star_trails_side.append(tr)
        if ax_outer is not None:
            sc = ax_outer.scatter([], [], s=max(40, 0.7*sz), c=c,
                                   edgecolors="white", linewidths=1.0,
                                   zorder=10)
            star_scatters_outer.append(sc)
            tr, = ax_outer.plot([], [], color="white", lw=0.5, alpha=0.35,
                                 zorder=6)
            star_trails_outer.append(tr)

    # Planet markers + trails
    planet_scatters_top  = []
    planet_trails_top    = []
    planet_scatters_side = []
    planet_trails_side   = []
    planet_scatters_outer = []
    planet_trails_outer = []
    planet_history_inner = [[[], [], []] for _ in meta]
    planet_history_abs   = [[[], [], []] for _ in meta]
    for m in meta:
        color = m["color"]
        size = 40 if m["kind"] == "KNOWN" else 28
        sc = ax_top.scatter([], [], s=size, c=color, edgecolors="white",
                            linewidths=0.7, zorder=12)
        planet_scatters_top.append(sc)
        tr, = ax_top.plot([], [], color=color, lw=0.9, alpha=0.85,
                           zorder=7)
        planet_trails_top.append(tr)
        if ax_side is not None:
            sc = ax_side.scatter([], [], s=size, c=color,
                                  edgecolors="white", linewidths=0.7,
                                  zorder=12)
            planet_scatters_side.append(sc)
            tr, = ax_side.plot([], [], color=color, lw=0.9, alpha=0.85,
                                zorder=7)
            planet_trails_side.append(tr)
        if ax_outer is not None:
            sc = ax_outer.scatter([], [], s=max(8, 0.5*size), c=color,
                                   edgecolors="white", linewidths=0.4,
                                   zorder=12)
            planet_scatters_outer.append(sc)
            tr, = ax_outer.plot([], [], color=color, lw=0.6, alpha=0.7,
                                 zorder=7)
            planet_trails_outer.append(tr)

    _make_legend(ax_top, meta, phz_ptype is not None)
    if ax_outer is not None:
        ax_outer.text(0.02, 0.97,
                       "Full hierarchy (absolute coords)",
                       transform=ax_outer.transAxes, color="white",
                       fontsize=10, va="top",
                       bbox=dict(boxstyle="round,pad=0.25",
                                 facecolor="#15161c", alpha=0.5,
                                 ec="none"))

    def update(frame_idx):
        # Absolute star positions for this frame
        star_pos = star_hist[frame_idx]
        # Inner-binary barycentre (for recentering inner panels)
        bary_i = inner_bary_hist[frame_idx]

        # ----- Stars in top view (recentered on inner bary) -----
        for si in range(star_pos.shape[0]):
            x_abs, y_abs, z_abs = star_pos[si]
            # Recentered for inner views
            x_in = x_abs - bary_i[0]
            y_in = y_abs - bary_i[1]
            z_in = z_abs - bary_i[2]
            star_history_inner[si][0].append(x_in)
            star_history_inner[si][1].append(y_in)
            star_history_inner[si][2].append(z_in)
            star_history[si][0].append(x_abs)
            star_history[si][1].append(y_abs)
            star_history[si][2].append(z_abs)

            star_scatters_top[si].set_offsets([[x_in, y_in]])
            # Keep only recent trail to avoid clutter
            keep_inner = max(60, int(0.6 * P_inner / duration * n_frames))
            xs = star_history_inner[si][0][-keep_inner:]
            ys = star_history_inner[si][1][-keep_inner:]
            zs = star_history_inner[si][2][-keep_inner:]
            star_trails_top[si].set_data(xs, ys)
            if ax_side is not None:
                star_scatters_side[si].set_offsets([[x_in, z_in]])
                star_trails_side[si].set_data(xs, zs)
            if ax_outer is not None:
                star_scatters_outer[si].set_offsets([[x_abs, y_abs]])
                star_trails_outer[si].set_data(star_history[si][0],
                                                star_history[si][1])

        # ----- PHZ rings (inner view, recentered) -----
        for host_idx, r_in, r_out, poly in phz_rings_top:
            cx = star_pos[host_idx][0] - bary_i[0]
            cy = star_pos[host_idx][1] - bary_i[1]
            x_out = cx + r_out * cos_t
            y_out = cy + r_out * sin_t
            x_in  = cx + r_in  * cos_t[::-1]
            y_in  = cy + r_in  * sin_t[::-1]
            xy = np.column_stack([
                np.concatenate([x_out, x_in, [x_out[0]]]),
                np.concatenate([y_out, y_in, [y_out[0]]])])
            poly.set_xy(xy)
        if phz_ring_ptype is not None:
            r_in, r_out, poly = phz_ring_ptype
            # P-type ring centred on inner barycentre of inner pair
            bary = _inner_bary(star_pos, star_masses, min(2, len(sys.stars)))
            cx = bary[0] - bary_i[0]
            cy = bary[1] - bary_i[1]
            x_out = cx + r_out * cos_t
            y_out = cy + r_out * sin_t
            x_in  = cx + r_in  * cos_t[::-1]
            y_in  = cy + r_in  * sin_t[::-1]
            xy = np.column_stack([
                np.concatenate([x_out, x_in, [x_out[0]]]),
                np.concatenate([y_out, y_in, [y_out[0]]])])
            poly.set_xy(xy)

        # ----- Dynamic snapshot HZ contour (every few frames; expensive) -----
        if frame_idx % 6 == 0:
            if dynamic_hz_contour[0] is not None:
                try:
                    dynamic_hz_contour[0].remove()
                except (AttributeError, ValueError):
                    pass
            # Move stars into shifted (inner-bary) frame for the HZ calc
            for si, s in enumerate(sys.stars):
                s.position = (star_pos[si][0] - bary_i[0],
                              star_pos[si][1] - bary_i[1],
                              star_pos[si][2] - bary_i[2])
            mask = hz.hz_mask(grid, sys.stars, optimistic=optimistic)
            dynamic_hz_contour[0] = ax_top.contour(
                XX, YY, mask.astype(float),
                levels=[0.5], colors=["#F0D060"], linewidths=0.9,
                alpha=0.85)

        # ----- Planets -----
        for pi, m in enumerate(meta):
            x_abs, y_abs, z_abs = planet_hist[frame_idx, pi]
            # Recentered for inner views
            x_in = x_abs - bary_i[0]
            y_in = y_abs - bary_i[1]
            z_in = z_abs - bary_i[2]
            planet_history_inner[pi][0].append(x_in)
            planet_history_inner[pi][1].append(y_in)
            planet_history_inner[pi][2].append(z_in)
            planet_history_abs[pi][0].append(x_abs)
            planet_history_abs[pi][1].append(y_abs)
            planet_history_abs[pi][2].append(z_abs)

            # Trail length: ~1 planet period of history (more for short
            # P_p ratios so wobble is visible)
            host_mass = (star_masses[m["host_idx"]]
                         if m["kind"] != "P" and m["host_idx"] is not None
                         else float(np.sum(star_masses)))
            P_p = orbital_period(host_mass, 0.0, m["a"])
            n_orbits_shown = max(1.5, 1.5 * min(P_inner / P_p, 5.0))
            max_history = max(20, int(n_orbits_shown * P_p
                                       / duration * n_frames))
            for arr in (planet_history_inner[pi],
                        planet_history_abs[pi]):
                arr[0] = arr[0][-max_history:]
                arr[1] = arr[1][-max_history:]
                arr[2] = arr[2][-max_history:]

            planet_scatters_top[pi].set_offsets([[x_in, y_in]])
            planet_trails_top[pi].set_data(planet_history_inner[pi][0],
                                            planet_history_inner[pi][1])
            if ax_side is not None:
                planet_scatters_side[pi].set_offsets([[x_in, z_in]])
                planet_trails_side[pi].set_data(planet_history_inner[pi][0],
                                                 planet_history_inner[pi][2])
            if ax_outer is not None:
                planet_scatters_outer[pi].set_offsets([[x_abs, y_abs]])
                planet_trails_outer[pi].set_data(planet_history_abs[pi][0],
                                                  planet_history_abs[pi][1])

        title.set_text(f"{sys.name}  --  t = {times[frame_idx]:.3g} yr"
                       f"  (P_bin = {P_inner:.3g} yr)")
        return []

    print(f"  Rendering {n_frames} frames at {fps} fps -> {save_path}")
    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000.0/fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2800,
                          codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p"])
    anim.save(save_path, writer=writer,
              savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)


def _make_legend(ax, meta, has_ptype):
    items = [("PHZ (Goldilocks)",  "#1B8B1B"),
             ("Snapshot HZ",       "#F0D060")]
    if any(m["kind"] == "S" for m in meta):
        items.append(("S-type planet (N-body)", "#56C7FF"))
    if has_ptype:
        items.append(("P-type planet (N-body)", "#FF8A33"))
    if any(m["kind"] == "KNOWN" for m in meta):
        items.append(("Catalogued planet", "#FF44AA"))
    for i, (label, color) in enumerate(items):
        ax.text(0.02, 0.97 - 0.042 * i, label, transform=ax.transAxes,
                color=color, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="#15161c", alpha=0.5, ec="none"))