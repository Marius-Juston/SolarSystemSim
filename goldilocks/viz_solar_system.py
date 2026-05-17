"""
viz_solar_system.py
--------------------
Static + animated visualisation for a randomly generated solar system:

  * plot_overview          -- whole system top-down: stars, every planet
                              orbit, the PHZ band and the snow line.
  * plot_all_planet_zooms  -- one detailed card per planet: the planet
                              tinted by its sky colour, every moon orbit
                              scaled against the Roche limit and the
                              Domingos Hill-fraction, plus the full
                              habitability profile as text.
  * plot_longterm_stability-- the long-term nested-N-body verification:
                              whole-system trajectory + per-planet
                              centre-distance(t) envelopes + verdict.
  * animate_solar_system   -- system-wide MP4 (+ one gas-giant moon-zoom
                              MP4) from the nested-N-body history.

Dark-theme styling and the star colour/size helpers are reused from
visualization.py.
"""

from __future__ import annotations

import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter

from goldilocks.moons import (planetary_roche_limit_au, planet_hill_radius_au,
                              critical_moon_fraction)
from goldilocks.planets import bulk_density_gcc
from goldilocks.system import StarSystem
from goldilocks.visualization import _star_color, _star_size

BG = "#15161c"
PANEL = "#0c0d11"


def _dark(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("white")


def _orbit_xy(a, e, n=400):
    th = np.linspace(0, 2 * math.pi, n)
    r = a * (1 - e * e) / (1 + e * np.cos(th))
    return r * np.cos(th), r * np.sin(th)


# ---------------------------------------------------------------------
def plot_overview(sys: StarSystem, save_path: Optional[str] = None):
    sys._update_stellar_positions(0.0)
    kind, host_idx, band = getattr(sys, "phz_band", ("S", 0, (0.9, 1.1)))
    if kind == "S":
        centre = np.array(sys.stars[host_idx].position)
        L = sys.stars[host_idx].luminosity
    else:
        centre = sys.barycentre()
        L = sys.total_luminosity()
    snow = 2.7 * math.sqrt(max(L, 1e-4))

    apo = max((p.semi_major_axis_au * (1 + p.eccentricity)
               for p in sys.planets), default=2.0)
    ext = 1.15 * max(apo, snow)

    fig, ax = plt.subplots(figsize=(11, 11))
    fig.patch.set_facecolor(BG)
    _dark(ax)

    # PHZ annulus + snow line
    th = np.linspace(0, 2 * math.pi, 200)
    ax.fill(np.r_[centre[0] + band[1] * np.cos(th),
                  centre[0] + band[0] * np.cos(th[::-1])],
            np.r_[centre[1] + band[1] * np.sin(th),
                  centre[1] + band[0] * np.sin(th[::-1])],
            color="#1B8B1B", alpha=0.25, zorder=1, label="PHZ")
    ax.plot(centre[0] + snow * np.cos(th), centre[1] + snow * np.sin(th),
            ls="--", color="#7FB0FF", lw=1.0, alpha=0.7, label="snow line")

    for s in sys.stars:
        x, y, _ = s.position
        ax.scatter(x, y, s=_star_size(s.luminosity), c=_star_color(s.teff),
                   edgecolors="white", linewidths=1.2, zorder=10)
        ax.annotate(s.name, (x, y), textcoords="offset points",
                    xytext=(7, 7), color="white", fontsize=8)

    # Inner-region inset so the crowded inner planets are legible.
    inner_ext = 1.25 * max(band[1], snow * 0.6)
    show_inset = ext > 2.2 * inner_ext
    axin = None
    if show_inset:
        axin = ax.inset_axes([0.62, 0.62, 0.36, 0.36])
        _dark(axin)
        axin.fill(np.r_[centre[0] + band[1] * np.cos(th),
                        centre[0] + band[0] * np.cos(th[::-1])],
                  np.r_[centre[1] + band[1] * np.sin(th),
                        centre[1] + band[0] * np.sin(th[::-1])],
                  color="#1B8B1B", alpha=0.30, zorder=1)
        axin.plot(centre[0] + snow * np.cos(th),
                  centre[1] + snow * np.sin(th), ls="--",
                  color="#7FB0FF", lw=0.8, alpha=0.7)
        axin.scatter(*centre[:2], s=90, c="#FFE08A",
                     edgecolors="white", zorder=10)
        axin.set_xlim(centre[0] - inner_ext, centre[0] + inner_ext)
        axin.set_ylim(centre[1] - inner_ext, centre[1] + inner_ext)
        axin.set_aspect("equal", "box")
        axin.set_title("inner-zone zoom", color="white", fontsize=8)

    for idx, p in enumerate(sys.planets):
        c = (np.array(sys.stars[p.host_star_index].position)
             if not p.is_circumbinary() else sys.barycentre())
        ox, oy = _orbit_xy(p.semi_major_axis_au, p.eccentricity)
        inp = p.habitability is not None and p.habitability.in_phz
        col = (p.habitability.sky_color_hex if p.habitability
               else "#8899AA")
        msize = 30 + 26 * math.log10(max(p.radius_re, 0.3) + 1)
        for a_ in (ax,) + ((axin,) if axin is not None else ()):
            a_.plot(c[0] + ox, c[1] + oy, color=col,
                    lw=1.6 if inp else 0.8,
                    alpha=0.95 if inp else 0.55, zorder=5)
            a_.scatter(c[0] + p.semi_major_axis_au, c[1], s=msize,
                       color=col,
                       edgecolors=("#FFFFFF" if inp else "#888"),
                       linewidths=1.3 if inp else 0.6, zorder=8)
        # Alternating leader-line labels on the main axis only.
        dy = (1 if idx % 2 == 0 else -1) * (0.30 + 0.12 * (idx % 3)) * ext
        tag = f"{p.name}" + ("  [PHZ]" if inp else "")
        ax.annotate(
            tag, xy=(c[0] + p.semi_major_axis_au, c[1]),
            xytext=(c[0] + p.semi_major_axis_au, c[1] + dy),
            color=col, fontsize=7.5, ha="center",
            arrowprops=dict(arrowstyle="-", color=col, lw=0.5,
                            alpha=0.5))

    # ax.set_xlim(-ext + centre[0], ext + centre[0])
    # ax.set_ylim(-ext + centre[1], ext + centre[1])
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x  [AU]", color="white")
    ax.set_ylabel("y  [AU]", color="white")
    ax.set_title(f"{sys.name}\n{getattr(sys, 'generation_note', '')}",
                 color="white", fontsize=11)
    ax.legend(facecolor=PANEL, edgecolor="white", labelcolor="white",
              fontsize=8, loc="upper left")
    if save_path:
        fig.savefig(save_path, dpi=130, facecolor=BG, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------
def plot_all_planet_zooms(sys: StarSystem, save_path: Optional[str] = None):
    planets = [p for p in sys.planets if p.semi_major_axis_au]
    n = len(planets)
    cols = 3
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.6 * rows))
    fig.patch.set_facecolor(BG)
    axes = np.atleast_1d(axes).ravel()

    for k, p in enumerate(planets):
        ax = axes[k]
        _dark(ax)
        host_mass = (sum(s.mass for s in sys.stars) if p.is_circumbinary()
                     else sys.stars[p.host_star_index].mass)
        R_hill = planet_hill_radius_au(p.mass_me, p.semi_major_axis_au,
                                       host_mass)
        rho_p = bulk_density_gcc(p.mass_me, p.radius_re)
        roche = planetary_roche_limit_au(p.radius_re, rho_p, 2.0)
        sky = p.habitability.sky_color_hex if p.habitability else "#88AACC"

        # planet disc + Roche (red) and prograde/retrograde Hill rings
        ax.add_patch(plt.Circle((0, 0), roche, color="#552222",
                                alpha=0.5, zorder=1))
        ax.scatter([0], [0], s=420, color=sky, edgecolors="white",
                   linewidths=1.4, zorder=6)
        for frac, sty, lab in (
                (critical_moon_fraction(p.eccentricity, 0.0, False),
                 "-", "prograde stability"),
                (critical_moon_fraction(p.eccentricity, 0.3, True),
                 ":", "retrograde stability")):
            rr = frac * R_hill
            th = np.linspace(0, 2 * math.pi, 200)
            ax.plot(rr * np.cos(th), rr * np.sin(th), ls=sty,
                    color="#3FAE3F", lw=1.0, alpha=0.8)

        for mn in p.moons:
            th = np.linspace(0, 2 * math.pi, 160)
            r = mn.a_planet_au * (1 - mn.eccentricity ** 2) / \
                (1 + mn.eccentricity * np.cos(th))
            col = "#FF8A33" if mn.retrograde else "#56C7FF"
            lw = 0.9 if mn.kind == "regular" else 0.4
            al = 0.9 if mn.kind == "regular" else 0.45
            ax.plot(r * np.cos(th), r * np.sin(th), color=col, lw=lw,
                    alpha=al, zorder=4)

        lim = 1.05 * critical_moon_fraction(p.eccentricity, 0.3, True) * R_hill
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", "box")
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.set_title(p.name, color="white", fontsize=10)
        txt = (p.habitability.summary if p.habitability else "")
        ax.text(0.015, 0.985, txt, transform=ax.transAxes,
                color="#E2E2E2", fontsize=5.8, va="top", ha="left",
                family="monospace", zorder=20,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#05060a",
                          alpha=0.78, edgecolor="#333"))

    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"{sys.name} -- per-planet detail "
                 f"(blue=prograde moon, orange=retrograde; "
                 f"red=Roche, green=Hill stability)",
                 color="white", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path:
        fig.savefig(save_path, dpi=125, facecolor=BG, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------
def plot_longterm_stability(result: dict, sys: StarSystem,
                            save_path: Optional[str] = None):
    body = result["body"]
    bh = result["body_hist"]
    rep = result["report"]
    times = result["times"]

    fig = plt.figure(figsize=(17, 8))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.2)
    ax_traj = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])
    _dark(ax_traj)
    _dark(ax_r)

    star_masses = np.array([s.mass for s in sys.stars])
    w = star_masses / star_masses.sum()
    sys_bary = np.einsum("tsk,s->tk", result["star_hist"], w)

    planet_rows = [r for r, k in enumerate(body.kind) if k == "planet"]
    for r in planet_rows:
        tr = bh[:, r, :] - sys_bary
        ax_traj.plot(tr[:, 0], tr[:, 1], lw=0.7, alpha=0.8)
    ax_traj.scatter([0], [0], s=120, c="#FFE08A", edgecolors="white",
                    zorder=10)
    ax_traj.set_aspect("equal", "box")
    ax_traj.set_xlabel("x [AU]", color="white")
    ax_traj.set_ylabel("y [AU]", color="white")
    ax_traj.set_title(f"Whole-system trajectory over "
                      f"{rep['horizon_yr']:.0f} yr", color="white",
                      fontsize=11)

    # per-planet centre-distance envelope vs time
    for r in planet_rows:
        # planet vs system barycentre is a fine bounded-orbit proxy
        d = np.linalg.norm(bh[:, r, :] - sys_bary, axis=1)
        ax_r.plot(times, d, lw=1.0, label=body.label[r])
    ax_r.set_xlabel("time [yr]", color="white")
    ax_r.set_ylabel("distance to barycentre [AU]", color="white")
    ax_r.set_title("Per-planet orbital envelope (bounded => stable)",
                   color="white", fontsize=11)
    ax_r.legend(facecolor=PANEL, edgecolor="white", labelcolor="white",
                fontsize=7, ncol=2)

    verdict = "STABLE" if rep["stable"] else "UNSTABLE"
    vcol = "#3FAE3F" if rep["stable"] else "#FF5555"
    n_unstable = sum(1 for b in rep["bodies"] if not b["stable"])
    fig.suptitle(
        f"{sys.name} -- long-term nested N-body verification: {verdict}  "
        f"({rep['n_bodies']} bodies, {rep['sub_steps']} sub-steps/sample, "
        f"{n_unstable} flagged)", color=vcol, fontsize=13)
    if save_path:
        fig.savefig(save_path, dpi=125, facecolor=BG, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------
def animate_solar_system(result: dict, sys: StarSystem,
                         save_path: str, fps: int = 24,
                         max_frames: int = 240):
    body = result["body"]
    bh = result["body_hist"]
    star_hist = result["star_hist"]
    n_t = bh.shape[0]
    step = max(1, n_t // max_frames)
    idxs = list(range(0, n_t, step))

    star_masses = np.array([s.mass for s in sys.stars])
    w = star_masses / star_masses.sum()
    sys_bary = np.einsum("tsk,s->tk", star_hist, w)

    planet_rows = [r for r, k in enumerate(body.kind) if k == "planet"]
    pts = np.concatenate([bh[:, r, :2] - sys_bary[:, :2]
                          for r in planet_rows]) if planet_rows else \
        np.zeros((1, 2))
    ext = 1.15 * float(np.max(np.linalg.norm(pts, axis=1))) if len(pts) else 5.0

    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    fig.patch.set_facecolor(BG)
    _dark(ax)
    ax.set_xlim(-ext, ext)
    ax.set_ylim(-ext, ext)
    ax.set_aspect("equal", "box")
    title = ax.set_title("", color="white", fontsize=11)

    star_sc = ax.scatter([], [], s=140, c="#FFE08A", edgecolors="white",
                         zorder=10)
    p_sc = ax.scatter([], [], s=30, c="#56C7FF", edgecolors="white",
                      linewidths=0.5, zorder=8)
    trails = [ax.plot([], [], lw=0.6, alpha=0.7)[0] for _ in planet_rows]

    def update(fi):
        t = idxs[fi]
        b = sys_bary[t]
        sp = star_hist[t] - b
        star_sc.set_offsets(sp[:, :2])
        pp = np.array([bh[t, r, :2] - b[:2] for r in planet_rows])
        if len(pp):
            p_sc.set_offsets(pp)
        for j, r in enumerate(planet_rows):
            seg = bh[max(0, t - 400):t + 1, r, :2] - sys_bary[
                max(0, t - 400):t + 1, :2]
            trails[j].set_data(seg[:, 0], seg[:, 1])
        title.set_text(f"{sys.name} -- t = {result['times'][t]:.1f} yr")
        return []

    anim = FuncAnimation(fig, update, frames=len(idxs),
                         interval=1000.0 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2600, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p"])
    anim.save(save_path, writer=writer,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
