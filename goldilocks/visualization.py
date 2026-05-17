"""
visualization.py
----------------
Static matplotlib visualization of an N-star + N-planet system.

The figure shows:
  * Snapshot HZ + Permanently Habitable Zone (top-down)
  * Side view (x-z) showing real orbital inclinations
  * Per-planet eccentricity envelope
  * For hierarchical systems with widely-separated outer companions,
    an additional "wide view" panel showing the full hierarchy

Planet orbital paths come from a short N-body integration, so the
plotted orbits show *real* non-Keplerian wobble (forced eccentricity,
apse precession), not just pure Kepler ellipses.

Inner-zoom panels are recentered on the inner-binary barycentre so the
inner system stays in frame regardless of the outer Kepler phase.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from goldilocks import habitable_zone as hz
from goldilocks import secular as sec
from goldilocks.kepler import orbital_period
from goldilocks.nbody import (StarTrajectory, planet_initial_state,
                              integrate_planets)
from goldilocks.system import StarSystem

HZ_COLOR_CONS = "#3FAE3F"
HZ_COLOR_OPT = "#A6E0A6"
PHZ_COLOR = "#1B8B1B"
PLANET_COLOR = "#2266FF"
CBP_COLOR = "#FF8A33"
KNOWN_COLOR = "#FF44AA"

STAR_COLORS = {
    "O": "#9BB0FF", "B": "#AABFFF", "A": "#CAD7FF",
    "F": "#F8F7FF", "G": "#FFF4EA", "K": "#FFD2A1", "M": "#FF8855",
}


def _star_color(teff_k: float) -> str:
    if teff_k > 30000: return STAR_COLORS["O"]
    if teff_k > 10000: return STAR_COLORS["B"]
    if teff_k > 7500: return STAR_COLORS["A"]
    if teff_k > 6000: return STAR_COLORS["F"]
    if teff_k > 5200: return STAR_COLORS["G"]
    if teff_k > 3700: return STAR_COLORS["K"]
    return STAR_COLORS["M"]


def _star_size(L: float, scale: float = 1.0) -> float:
    return scale * max(120.0, 120.0 * math.log10(L + 1.5) + 130.0)


def _inner_bary(star_pos: np.ndarray, masses: np.ndarray,
                n_inner: int) -> np.ndarray:
    if n_inner <= 0:
        return np.zeros(3)
    w = masses[:n_inner]
    return (w[:, None] * star_pos[:n_inner]).sum(0) / w.sum()


def _system_scales(sys: StarSystem, extent_au: float
                   ) -> Tuple[float, Optional[float], int]:
    """Same logic as in animation.py."""
    if len(sys.stellar_orbits) < 2:
        return extent_au, None, len(sys.stars)
    inner_a = sys.stellar_orbits[0][2]
    outer_a = sys.stellar_orbits[1][2]
    if outer_a / inner_a < 8.0:
        e_out = sys.stellar_orbits[1][3]
        out_extent = 1.5 * outer_a * (1.0 + e_out)
        return out_extent, None, len(sys.stars)
    e_out = sys.stellar_orbits[1][3]
    outer_extent = 1.4 * outer_a * (1.0 + e_out)
    return extent_au, outer_extent, 2


def plot_system(sys: StarSystem,
                planet_mass_me: float = 1.0,
                delta: float = 10.0,
                extent_au: Optional[float] = None,
                n_grid: int = 301,
                save_path: Optional[str] = None,
                trail_periods: float = 1.5) -> plt.Figure:
    if extent_au is None:
        L_tot = sys.total_luminosity()
        extent_au = max(3.0, 2.5 * math.sqrt(L_tot / 0.30))

    inner_extent, outer_extent, n_inner = _system_scales(sys, extent_au)

    # Snapshot HZ at t=0 (in absolute frame)
    sys._update_stellar_positions(0.0)
    star_masses = np.array([s.mass for s in sys.stars])
    bary_in = _inner_bary(
        np.array([s.position for s in sys.stars]),
        star_masses, n_inner)

    # PHZ: intersect HZ over a binary period
    if len(sys.stars) >= 2 and sys.stellar_orbits:
        P_bin = sys.stellar_orbit_period(0)
        sample_t = np.linspace(0.0, P_bin, 24, endpoint=False)
    else:
        sample_t = [0.0]

    # Compute on a grid in the INNER-BARY frame (recentered)
    Xg = np.linspace(-inner_extent, inner_extent, n_grid)
    Yg = np.linspace(-inner_extent, inner_extent, n_grid)
    XX, YY = np.meshgrid(Xg, Yg)
    Z0 = np.zeros_like(XX)
    grid = np.stack([XX, YY, Z0], axis=-1)

    # Temporarily shift stars into the inner-bary frame for HZ calc
    sys._update_stellar_positions(0.0)
    orig_positions = [s.position for s in sys.stars]
    bary0 = _inner_bary(np.array(orig_positions), star_masses, n_inner)
    for i, s in enumerate(sys.stars):
        p = orig_positions[i]
        s.position = (p[0] - bary0[0], p[1] - bary0[1], p[2] - bary0[2])
    mask_t0 = hz.hz_mask(grid, sys.stars, optimistic=False)
    if len(sample_t) > 1:
        mask_phz = np.ones_like(mask_t0, dtype=bool)
        for t in sample_t:
            sys._update_stellar_positions(t)
            # Snapshot all star positions BEFORE shifting any of them
            star_positions_now = np.array([s.position for s in sys.stars])
            bary_t = _inner_bary(star_positions_now, star_masses, n_inner)
            for i, s in enumerate(sys.stars):
                p = star_positions_now[i]
                s.position = (p[0] - bary_t[0], p[1] - bary_t[1],
                              p[2] - bary_t[2])
            mask_phz &= hz.hz_mask(grid, sys.stars, optimistic=False)
        sys._update_stellar_positions(0.0)
    else:
        mask_phz = mask_t0
    # Restore
    for i, s in enumerate(sys.stars):
        s.position = orig_positions[i]

    # Planet packing result
    res = sys.count_habitable_planets(planet_mass_me, delta,
                                      optimistic=False, use_phz=True)

    # N-body planet paths
    paths = _nbody_orbit_paths(sys, res, trail_periods)

    # Figure layout: 2x3 if outer needed, else 2x2
    if outer_extent is not None:
        fig = plt.figure(figsize=(20.0, 12.5))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0],
                              height_ratios=[1.0, 0.75],
                              hspace=0.28, wspace=0.22)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_phz = fig.add_subplot(gs[0, 1])
        ax_outer = fig.add_subplot(gs[0, 2])
        ax_side = fig.add_subplot(gs[1, 0:2])
        ax_e = fig.add_subplot(gs[1, 2])
    else:
        fig = plt.figure(figsize=(15.5, 13.0))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0],
                              height_ratios=[1.0, 0.85],
                              hspace=0.27, wspace=0.20)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_phz = fig.add_subplot(gs[0, 1])
        ax_side = fig.add_subplot(gs[1, 0])
        ax_e = fig.add_subplot(gs[1, 1])
        ax_outer = None

    fig.patch.set_facecolor("#15161c")
    for a in (ax_top, ax_phz, ax_outer, ax_side, ax_e):
        if a is None: continue
        a.set_facecolor("#0c0d11")
        a.tick_params(colors="white")
        for sp in a.spines.values():
            sp.set_color("white")

    # TOP-LEFT: snapshot HZ + planet ellipses (inner-bary frame)
    ax_top.contourf(XX, YY, mask_t0.astype(float), levels=[0.5, 1.5],
                    colors=[HZ_COLOR_CONS], alpha=0.45)
    _draw_stellar_orbits(ax_top, sys, bary0, plane="xy")
    _draw_stars(ax_top, sys, bary0, plane="xy", show_labels=True)
    _draw_planet_paths(ax_top, paths, bary0, plane="xy")
    _draw_known_planets(ax_top, sys, bary0, plane="xy")
    ax_top.set_xlim(-inner_extent, inner_extent)
    ax_top.set_ylim(-inner_extent, inner_extent)
    ax_top.set_aspect("equal", "box")
    ax_top.set_xlabel("x  [AU]", color="white")
    ax_top.set_ylabel("y  [AU]", color="white")
    ax_top.set_title(f"{sys.name} -- snapshot HZ (t=0)",
                     color="white", fontsize=12)

    # TOP-RIGHT: PHZ
    ax_phz.contourf(XX, YY, mask_t0.astype(float), levels=[0.5, 1.5],
                    colors=[HZ_COLOR_OPT], alpha=0.18)
    ax_phz.contourf(XX, YY, mask_phz.astype(float), levels=[0.5, 1.5],
                    colors=[PHZ_COLOR], alpha=0.7)
    _draw_stellar_orbits(ax_phz, sys, bary0, plane="xy")
    _draw_stars(ax_phz, sys, bary0, plane="xy", show_labels=False)
    _draw_planet_paths(ax_phz, paths, bary0, plane="xy")
    _draw_known_planets(ax_phz, sys, bary0, plane="xy")
    ax_phz.set_xlim(-inner_extent, inner_extent)
    ax_phz.set_ylim(-inner_extent, inner_extent)
    ax_phz.set_aspect("equal", "box")
    ax_phz.set_xlabel("x  [AU]", color="white")
    ax_phz.set_ylabel("y  [AU]", color="white")
    ax_phz.set_title("Permanently Habitable Zone (1 binary period)",
                     color="white", fontsize=12)

    # Outer hierarchical view (absolute coords)
    if ax_outer is not None:
        # Draw outer Kepler orbit for the outer companion
        if len(sys.stellar_orbits) >= 2:
            _draw_outer_orbit(ax_outer, sys)
        _draw_stars(ax_outer, sys, np.zeros(3), plane="xy",
                    show_labels=True, size_scale=0.7)
        ax_outer.set_xlim(-outer_extent, outer_extent)
        ax_outer.set_ylim(-outer_extent, outer_extent)
        ax_outer.set_aspect("equal", "box")
        ax_outer.set_xlabel("x  [AU]", color="white")
        ax_outer.set_ylabel("y  [AU]", color="white")
        ax_outer.set_title(f"Hierarchical view (+/-{outer_extent:.0f} AU)",
                           color="white", fontsize=11)
        # Mark the inner-binary box for context
        from matplotlib.patches import Rectangle
        rect = Rectangle((-inner_extent, -inner_extent),
                         2 * inner_extent, 2 * inner_extent,
                         fill=False, edgecolor="#3FAE3F",
                         linewidth=0.8, alpha=0.7, linestyle="--",
                         zorder=2)
        ax_outer.add_patch(rect)
        ax_outer.text(inner_extent * 1.05, inner_extent * 1.05,
                      "inner zoom region", color="#3FAE3F",
                      fontsize=8, ha="left", va="bottom")

    # SIDE view (x-z), inner-bary frame
    _draw_stars(ax_side, sys, bary0, plane="xz", show_labels=False)
    _draw_planet_paths(ax_side, paths, bary0, plane="xz")
    _draw_known_planets(ax_side, sys, bary0, plane="xz")
    ax_side.set_xlim(-inner_extent, inner_extent)
    z_ext = max(0.30 * inner_extent, 0.5)
    ax_side.set_ylim(-z_ext, z_ext)
    ax_side.set_xlabel("x  [AU]", color="white")
    ax_side.set_ylabel("z  [AU]  (side view)", color="white")
    ax_side.set_title("Edge-on view: real orbital inclinations",
                      color="white", fontsize=12)

    # Eccentricity bar
    bars_lab, bars_val, bars_clr = [], [], []
    for entry in res["stars"]:
        if entry["stable_HZ"] is None or not entry.get("positions"):
            continue
        for i, (a_p, e_p) in enumerate(zip(entry["positions"],
                                           entry["e_max"])):
            bars_lab.append(f"{entry['name']}\np{i + 1}\n(a={a_p:.2f})")
            bars_val.append(e_p)
            bars_clr.append(PLANET_COLOR)
    if res["circumbinary"] and res["circumbinary"].get("positions"):
        for i, (a_p, e_p) in enumerate(zip(res["circumbinary"]["positions"],
                                           res["circumbinary"]["e_max"])):
            bars_lab.append(f"CBP\np{i + 1}\n(a={a_p:.2f})")
            bars_val.append(e_p)
            bars_clr.append(CBP_COLOR)
    if bars_lab:
        idx = np.arange(len(bars_lab))
        ax_e.bar(idx, bars_val, color=bars_clr, edgecolor="white",
                 linewidth=0.5)
        ax_e.set_xticks(idx)
        ax_e.set_xticklabels(bars_lab, color="white", fontsize=7)
        ax_e.set_ylabel("Max eccentricity  e_max", color="white")
        ax_e.set_ylim(0, max(0.15, 1.2 * max(bars_val) if bars_val else 0.15))
        ax_e.set_title("Planet eccentricity envelope\n"
                       "(Heppenheimer + Laplace-Lagrange)",
                       color="white", fontsize=10)
        ax_e.grid(True, axis="y", color="#3a3c46", alpha=0.3)
    else:
        ax_e.text(0.5, 0.5, "No habitable planets fit", color="white",
                  ha="center", va="center", transform=ax_e.transAxes,
                  fontsize=11)
        ax_e.set_xticks([]);
        ax_e.set_yticks([])

    # Footer
    lines = []
    for entry in res["stars"]:
        if entry["stable_HZ"] is None:
            lines.append(f"{entry['name']}: no stable HZ")
        else:
            lo, hi = entry["stable_HZ"]
            lines.append(f"{entry['name']}: r in [{lo:.2f},{hi:.2f}] AU "
                         f"=> {entry['n_planets']} planet(s)")
    if res["circumbinary"]:
        lo, hi = res["circumbinary"]["stable_HZ"]
        lines.append(f"Circumbinary: r in [{lo:.2f},{hi:.2f}] AU "
                     f"=> {res['circumbinary']['n_planets']} planet(s)")
    fig.text(0.5, 0.005, "    |    ".join(lines),
             ha="center", color="white", fontsize=10)

    if save_path:
        fig.savefig(save_path, dpi=130, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
    return fig


# =====================================================================
# Helpers
# =====================================================================
def _proj(plane: str, xyz: np.ndarray) -> Tuple[float, float]:
    if plane == "xy":
        return xyz[0], xyz[1]
    if plane == "xz":
        return xyz[0], xyz[2]
    raise ValueError("plane must be 'xy' or 'xz'")


def _draw_stars(ax, sys: StarSystem, bary: np.ndarray, plane: str,
                show_labels: bool = True, size_scale: float = 1.0) -> None:
    for s in sys.stars:
        pos = np.array(s.position) - bary
        x, y = _proj(plane, pos)
        ax.scatter(x, y, s=_star_size(s.luminosity, size_scale),
                   c=_star_color(s.teff), edgecolors="white",
                   linewidths=1.2, zorder=10)
        if show_labels and plane == "xy":
            ax.annotate(s.name, xy=(x, y), xytext=(8, 8),
                        textcoords="offset points", color="white",
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="#15161c", alpha=0.6,
                                  ec="none"),
                        zorder=11)


def _draw_stellar_orbits(ax, sys: StarSystem, bary: np.ndarray,
                         plane: str) -> None:
    if not sys.stellar_orbits:
        return
    edge = sys.stellar_orbits[0]
    i, j, a, e, omega = edge
    m1, m2 = sys.stars[i].mass, sys.stars[j].mass
    nu = np.linspace(0, 2 * math.pi, 200)
    r_x = a * (np.cos(nu) - e)
    r_y = a * math.sqrt(1 - e * e) * np.sin(nu)
    co, so = math.cos(omega), math.sin(omega)
    rx = co * r_x - so * r_y
    ry = so * r_x + co * r_y
    f1 = -m2 / (m1 + m2)
    f2 = m1 / (m1 + m2)
    if plane == "xy":
        # Orbits are in the inner-bary frame already (centred on 0,0)
        ax.plot(f1 * rx, f1 * ry, color="white", lw=0.5, alpha=0.4)
        ax.plot(f2 * rx, f2 * ry, color="white", lw=0.5, alpha=0.4)


def _draw_outer_orbit(ax, sys: StarSystem) -> None:
    """Draw the outer companion's Kepler ellipse in the absolute frame."""
    if len(sys.stellar_orbits) < 2:
        return
    edge = sys.stellar_orbits[1]
    k, _, a, e, omega = edge
    nu = np.linspace(0, 2 * math.pi, 200)
    r_x = a * (np.cos(nu) - e)
    r_y = a * math.sqrt(1 - e * e) * np.sin(nu)
    co, so = math.cos(omega), math.sin(omega)
    rx = co * r_x - so * r_y
    ry = so * r_x + co * r_y
    # Star k's orbit around the inner-binary barycentre
    m_in = sys.stars[0].mass + sys.stars[1].mass
    m_out = sys.stars[k].mass
    f_out = m_in / (m_in + m_out)
    ax.plot(f_out * rx, f_out * ry, color="white", lw=0.5, alpha=0.5)


def _draw_planet_paths(ax, paths: list, bary: np.ndarray,
                       plane: str) -> None:
    for p in paths:
        traj = p["path"] - bary[None, :]
        if plane == "xy":
            ax.plot(traj[:, 0], traj[:, 1], color=p["meta"]["color"],
                    lw=0.9, alpha=0.85, zorder=8)
            ax.scatter(traj[0, 0], traj[0, 1], s=22,
                       color=p["meta"]["color"], edgecolors="white",
                       linewidths=0.5, zorder=12)
        else:
            ax.plot(traj[:, 0], traj[:, 2], color=p["meta"]["color"],
                    lw=0.9, alpha=0.85, zorder=8)
            ax.scatter(traj[0, 0], traj[0, 2], s=22,
                       color=p["meta"]["color"], edgecolors="white",
                       linewidths=0.5, zorder=12)


def _draw_known_planets(ax, sys: StarSystem, bary: np.ndarray,
                        plane: str) -> None:
    for p in sys.planets:
        if p.semi_major_axis_au is None:
            continue
        theta = np.linspace(0, 2 * math.pi, 400)
        if p.is_circumbinary():
            centre = sys.barycentre()
        else:
            centre = np.array(sys.stars[p.host_star_index].position)
        a, e_p = p.semi_major_axis_au, p.eccentricity
        r = a * (1 - e_p * e_p) / (1 + e_p * np.cos(theta))
        i = math.radians(3.0)
        xs = (centre[0] - bary[0]) + r * np.cos(theta)
        ys_planar = r * np.sin(theta)
        ys = (centre[1] - bary[1]) + ys_planar * math.cos(i)
        zs = (centre[2] - bary[2]) + ys_planar * math.sin(i)
        if plane == "xy":
            ax.plot(xs, ys, color=KNOWN_COLOR, lw=1.0,
                    alpha=0.9, ls="--", zorder=9)
            peri_x = (centre[0] - bary[0]) + a * (1 - e_p)
            peri_y = (centre[1] - bary[1])
            ax.scatter(peri_x, peri_y, s=30, color=KNOWN_COLOR,
                       edgecolors="white", linewidths=0.6, zorder=12)
            ax.annotate(p.name, xy=(peri_x, peri_y),
                        xytext=(6, -10), textcoords="offset points",
                        color=KNOWN_COLOR, fontsize=8, zorder=13)
        else:
            ax.plot(xs, zs, color=KNOWN_COLOR, lw=1.0,
                    alpha=0.9, ls="--", zorder=9)
            peri_x = (centre[0] - bary[0]) + a * (1 - e_p)
            ax.scatter(peri_x, 0.0, s=30, color=KNOWN_COLOR,
                       edgecolors="white", linewidths=0.6, zorder=12)


def _nbody_orbit_paths(sys: StarSystem, res: dict,
                       trail_periods: float = 1.5) -> list:
    """Integrate each packed planet for ~trail_periods of its own orbit
    to get a realistic non-Keplerian path."""
    sys._update_stellar_positions(0.0)
    star_masses = np.array([s.mass for s in sys.stars])
    traj = StarTrajectory(star_masses, sys.stellar_orbits)
    omega_bin = sys.stellar_orbits[0][4] if sys.stellar_orbits else 0.0
    a_bin = sys.stellar_orbits[0][2] if sys.stellar_orbits else None
    e_bin = sys.stellar_orbits[0][3] if sys.stellar_orbits else 0.0

    rng = np.random.default_rng(seed=17)
    positions, velocities, masses, meta = [], [], [], []

    for host_idx, entry in enumerate(res["stars"]):
        if entry["stable_HZ"] is None or not entry.get("positions"):
            continue
        host = sys.stars[host_idx]
        host_pos = np.array(host.position)
        host_vel = np.array(host.velocity)
        for i, a_p in enumerate(entry["positions"]):
            if a_bin is not None:
                e0 = sec.heppenheimer_e_forced_stype(a_p, a_bin, e_bin)
            else:
                e0 = float(rng.uniform(0.01, 0.08))
            inc = math.radians(rng.uniform(3.0, 8.0))
            Omega = rng.uniform(0.0, 2.0 * math.pi)
            M_anom = rng.uniform(0.0, 2.0 * math.pi)
            r, v = planet_initial_state(
                host_pos=host_pos, host_vel=host_vel,
                m_host_msun=host.mass, a_au=a_p, e=e0,
                omega=omega_bin, inclination_rad=inc,
                lon_ascending_node_rad=Omega, mean_anomaly_rad=M_anom)
            positions.append(r)
            velocities.append(v)
            masses.append(1.0)
            meta.append({"kind": "S", "host_idx": host_idx, "a": a_p,
                         "color": PLANET_COLOR})

    if res["circumbinary"] and res["circumbinary"].get("positions"):
        m_tot = sum(s.mass for s in sys.stars)
        mu = min(sys.stars[0].mass, sys.stars[1].mass) / m_tot
        bary = sys.barycentre()
        for i, a_p in enumerate(res["circumbinary"]["positions"]):
            if a_bin is not None:
                e0 = sec.leung_lee_e_forced_ptype(a_p, a_bin, e_bin, mu)
            else:
                e0 = 0.0
            inc = math.radians(rng.uniform(1.5, 5.0))
            Omega = rng.uniform(0.0, 2.0 * math.pi)
            M_anom = rng.uniform(0.0, 2.0 * math.pi)
            r, v = planet_initial_state(
                host_pos=bary, host_vel=np.zeros(3),
                m_host_msun=m_tot, a_au=a_p, e=e0,
                omega=omega_bin, inclination_rad=inc,
                lon_ascending_node_rad=Omega, mean_anomaly_rad=M_anom)
            positions.append(r)
            velocities.append(v)
            masses.append(1.0)
            meta.append({"kind": "P", "host_idx": None, "a": a_p,
                         "color": CBP_COLOR})

    paths = []
    for j, m in enumerate(meta):
        if m["kind"] == "S":
            host_mass = star_masses[m["host_idx"]]
        else:
            host_mass = float(np.sum(star_masses))
        P_p = orbital_period(host_mass, 0.0, m["a"])
        T = trail_periods * P_p
        n_samples = 360
        times = np.linspace(0.0, T, n_samples)
        pos0 = np.array([positions[j]])
        vel0 = np.array([velocities[j]])
        p_hist, _ = integrate_planets(
            traj, pos0, vel0, np.array([masses[j]]),
            times, sub_steps_per_sample=30)
        paths.append({"path": p_hist[:, 0], "meta": m})
    return paths
