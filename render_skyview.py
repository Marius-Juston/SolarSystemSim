"""
render_skyview.py
-----------------
Photorealistic ground-to-sky renderer driver.  For each test system it
attaches a habitability profile to a habitable planet, then writes the
four canonical lighting situations (midnight / sunrise / noon / sunset)
as PNGs plus a full solar-day MP4.

What this exercises (skyview v2):
  1. Single   -- Sun + Earth-analog (+ a Moon): blue Rayleigh sky, the
                  Moon shows the correct phase and reddens at the
                  horizon; the night is a starlit Milky-Way sky, not
                  pure black.
  2. Binary   -- Alpha Cen A/B; S-type planet round A (two-sun sky).
  3. Triple   -- Alpha Cen A/B + Proxima hierarchical triple.
  4. Random   -- two seeded `random_solar_system`s: sibling planets and
                  every moon appear in the sky as reflected-light disks.
  5. Gallery  -- one labelled grid comparing the sky colour of every
                  atmosphere regime across many random systems.

Usage:  uv run python render_skyview.py [seed]
Output: figures/skyview/  and  animations/skyview/  (repo-relative).
"""

import os
import sys as _sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from goldilocks.stellar import Star, T_EFF_SUN_K
from goldilocks.system import StarSystem
from goldilocks.planets import earth_analog, is_gas_giant, Planet
from goldilocks.moons import Moon
from goldilocks.habitability import profile_for_planet, HabitabilityProfile
from goldilocks.solar_system import random_solar_system
from goldilocks.skyview import (render_phases, animate_day, render_sky,
                                phase_rotations, planck_spectral,
                                spectrum_to_srgb, lambda_grid_nm,
                                cie_xyz_bar)

_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_FIG = os.path.join(_ROOT, "figures", "skyview")
OUT_ANI = os.path.join(_ROOT, "animations", "skyview")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_ANI, exist_ok=True)


def _build(seed: int):
    rng = np.random.default_rng(seed)

    sun = Star("Sun", mass=1.0, luminosity=1.0, teff=T_EFF_SUN_K, radius=1.0)
    earth = earth_analog("Earth")
    # A real Moon analog so the canonical sky actually has a Moon.
    earth.moons = [Moon("Luna", mass_me=0.0123, radius_re=0.273,
                        a_planet_au=0.00257, eccentricity=0.0549,
                        density_gcc=3.34, kind="regular")]
    sys1 = StarSystem.single("Sol", sun, planets=[earth])

    A = Star("Alpha Cen A", mass=1.10)
    B = Star("Alpha Cen B", mass=0.907)
    sys2 = StarSystem.binary("AlphaCenAB", A, B,
                             separation_au=23.4, eccentricity=0.52,
                             planets=[earth_analog("Prox-b-analog",
                                                   a_au=1.25,
                                                   host_star_index=0)],
                             quiet=True)

    A2 = Star("Alpha Cen A", mass=1.10)
    B2 = Star("Alpha Cen B", mass=0.907)
    Pr = Star("Proxima", mass=0.122)
    sys3 = StarSystem.hierarchical_triple(
        "AlphaCenTriple", A2, B2, Pr,
        a_in=23.4, e_in=0.52, a_out=8700.0, e_out=0.50,
        planets=[earth_analog("Triple-HZ", a_au=1.25, host_star_index=0)],
        quiet=True)

    cases = []
    for sysX in (sys1, sys2, sys3):
        p = sysX.planets[0]
        p.habitability = profile_for_planet(
            p, sysX, np.random.default_rng(seed + 7), in_phz=True)
        cases.append((sysX, p))

    # Two random full solar systems (sibling planets + moons in the sky).
    for k in (1, 2):
        rsys = random_solar_system(np.random.default_rng(seed * 10 + k),
                                   name=f"Aurelia-{seed}-{k}")
        phz = [p for p in rsys.planets
               if p.habitability and p.habitability.in_phz]
        cases.append((rsys, (phz or rsys.planets)[0]))
    return cases


# Explicit per-regime atmosphere profiles (mirror the documented
# outputs of habitability._infer_atmosphere) so every gallery tile is
# deterministically the intended regime -- a robust stress test, not a
# random draw that may collapse to one atmosphere.
_REGIMES = {
    "n2o2":  dict(dominant_gas="N2/O2 (temperate)", mean_molecular_weight=29.0,
                  surface_pressure_bar=1.0, scale_height_km=8.5,
                  sky_color_hex="#7FB2FF", bond_albedo=0.30,
                  sky_description="clear Rayleigh-scattered blue sky",
                  t_eq_k=255.0, t_surface_k=288.0),
    "dense": dict(dominant_gas="N2/O2 (temperate)", mean_molecular_weight=29.0,
                  surface_pressure_bar=2.4, scale_height_km=7.0,
                  sky_color_hex="#3A6BE0", bond_albedo=0.25,
                  sky_description="deep Rayleigh blue (high-pressure)",
                  t_eq_k=270.0, t_surface_k=320.0),
    "co2":   dict(dominant_gas="CO2 (runaway-greenhouse)",
                  mean_molecular_weight=44.0, surface_pressure_bar=12.0,
                  scale_height_km=15.0, sky_color_hex="#E8C16A",
                  bond_albedo=0.65, storm_index=0.5,
                  sky_description="dense CO2 overcast, hazy "
                  "yellow-orange sky", t_eq_k=360.0, t_surface_k=560.0),
    "titan": dict(dominant_gas="N2/CH4 (cold, reducing)",
                  mean_molecular_weight=28.0, surface_pressure_bar=1.5,
                  scale_height_km=20.0, sky_color_hex="#C8772E",
                  bond_albedo=0.20, storm_index=0.4,
                  sky_description="orange organic-haze sky (Titan-like)",
                  t_eq_k=95.0, t_surface_k=94.0),
    "airless": dict(dominant_gas="trace CO2 (near-airless)",
                    mean_molecular_weight=44.0, surface_pressure_bar=0.005,
                    scale_height_km=11.0, sky_color_hex="#1A1326",
                    bond_albedo=0.12,
                    sky_description="near-vacuum: black sky, sharp "
                    "shadows", t_eq_k=255.0, t_surface_k=255.0),
    "subnep": dict(dominant_gas="H2/He-rich (sub-Neptune)",
                   mean_molecular_weight=4.0, surface_pressure_bar=200.0,
                   scale_height_km=30.0, sky_color_hex="#A9C6D8",
                   bond_albedo=0.40,
                   sky_description="thick hydrogen haze, washed-out "
                   "white sky", t_eq_k=200.0, t_surface_k=400.0),
    "giant": dict(dominant_gas="H2/He (+CH4, NH3)",
                  mean_molecular_weight=2.3, surface_pressure_bar=1.0e4,
                  scale_height_km=27.0, sky_color_hex="#D9C7A3",
                  bond_albedo=0.34, storm_index=0.8,
                  sky_description="deep banded H2/He haze, "
                  "ammonia/methane tinted", t_eq_k=120.0,
                  t_surface_k=165.0),
}


def _curated_world(name, regime, mass_me, radius_re, a_au, *,
                   star=None, moons=None, obliquity=20.0,
                   sidereal_h=24.0):
    """Single-star world with an explicit, deterministic atmosphere
    regime profile (so the gallery reliably spans every sky colour)."""
    s = star or Star("G2V Sun", mass=1.0, luminosity=1.0,
                      teff=T_EFF_SUN_K, radius=1.0)
    p = Planet(name=name, mass_me=mass_me, radius_re=radius_re,
               semi_major_axis_au=a_au, eccentricity=0.0,
               host_star_index=0)
    if moons:
        p.moons = moons
    sysX = StarSystem.single(name, s, planets=[p])
    prof = HabitabilityProfile(
        obliquity_deg=obliquity, sidereal_day_h=sidereal_h,
        solar_day_h=sidereal_h, rotation_period_h=sidereal_h,
        in_phz=True, **_REGIMES[regime])
    p.habitability = prof
    return sysX, p


def _star_colour_strip():
    """A horizontal RGB bar of true black-body star colours from cool
    M dwarfs (red) through G (white) to hot O/B (blue)."""
    lam = lambda_grid_nm()
    _, yb, _ = cie_xyz_bar(lam)
    dl = float(lam[1] - lam[0])
    teffs = np.array([2800, 3400, 4200, 5200, 5772, 7000, 9500,
                      15000, 28000.0])
    specs, Ys = [], []
    for T in teffs:
        pl = planck_spectral(lam, float(T))
        pl = pl / np.trapezoid(pl, lam * 1e-9)
        specs.append(pl)
        Ys.append(float(np.dot(pl, yb) * dl))
    # One shared exposure: cool stars stay dim/red, hot stars
    # bright/blue -- representative of the real background field.
    exp = 0.7 / max(float(np.median(Ys)), 1e-12)
    cols = [spectrum_to_srgb(s[None, :], lam, exp)[0] for s in specs]
    return teffs, np.array(cols, dtype=np.uint8)


def sky_colour_gallery(out_path: str):
    """Robust labelled grid: deterministically spans every atmosphere
    regime, multi-sun skies, an extreme oblate fast-rotator and a
    deep-night multi-colour starfield, plus a black-body star-colour
    strip -- a representative stress test that the renderer works."""
    sun = Star("G2V Sun", mass=1.0, luminosity=1.0,
               teff=T_EFF_SUN_K, radius=1.0)
    luna = [Moon("Moon", mass_me=0.0123, radius_re=0.273,
                 a_planet_au=0.00257, eccentricity=0.055,
                 density_gcc=3.34, kind="regular")]

    tiles = []  # (title, system, planet, render_kw, phase)

    skw = dict(latitude_deg=0.0, vfov_deg=120.0, ground_frac=0.16)

    # --- every atmosphere regime, deterministically ---
    tiles.append(("Earth-like  N2/O2 (blue)",
                  *_curated_world("Terra", "n2o2", 1.0, 1.0, 1.0,
                                  moons=luna), skw, "noon"))
    tiles.append(("Dense N2/O2 (deep blue)",
                  *_curated_world("Heavy", "dense", 4.0, 1.5, 1.05),
                  skw, "noon"))
    tiles.append(("Runaway CO2 (yellow-orange)",
                  *_curated_world("Cythera", "co2", 4.0, 1.4, 0.7),
                  skw, "noon"))
    tiles.append(("Cold N2/CH4 haze (Titan orange)",
                  *_curated_world("Tholin", "titan", 0.5, 0.8, 2.6),
                  skw, "noon"))
    tiles.append(("Near-airless (black sky)",
                  *_curated_world("Cinder", "airless", 0.1, 0.46, 1.0),
                  skw, "noon"))
    tiles.append(("H2/He sub-Neptune (white)",
                  *_curated_world("Nivis", "subnep", 9.0, 2.6, 2.2),
                  skw, "noon"))
    tiles.append(("Gas giant  H2/He (banded)",
                  *_curated_world("Brobding", "giant", 120.0, 11.0,
                                  1.3), skw, "noon"))

    # --- robustness / challenge cases ---
    # Extreme fast rotator -> strongly oblate world.
    sfast, pfast = _curated_world("Spinster", "n2o2", 2.0, 1.3, 1.0,
                                  sidereal_h=3.0)
    tiles.append(("Fast rotator (oblate world)",
                  sfast, pfast, skw, "noon"))
    # Binary: two suns in the sky at once.
    A = Star("Alpha A", mass=1.10)
    B = Star("Beta B", mass=0.90)
    sbin = StarSystem.binary("Twin-Sun", A, B, separation_au=0.4,
                             eccentricity=0.1,
                             planets=[earth_analog("Helios",
                                                   a_au=1.4,
                                                   host_star_index=0)],
                             quiet=True)
    pbin = sbin.planets[0]
    pbin.habitability = profile_for_planet(
        pbin, sbin, np.random.default_rng(9), in_phz=True)
    tiles.append(("Binary: two suns", sbin, pbin, skw, "noon"))
    # Deep night: multi-colour starfield + Milky-Way band + Moon.
    snight, pnight = _curated_world("Nocturne", "n2o2", 1.0, 1.0, 1.0,
                                    moons=luna)
    tiles.append(("Deep night: stars + Milky Way",
                  snight, pnight, dict(latitude_deg=0.0), "midnight"))

    cols = 3
    rows = int(np.ceil((len(tiles) + 1) / cols))   # +1 = star strip
    fig, axes = plt.subplots(rows, cols,
                             figsize=(6.0 * cols, 3.6 * rows))
    fig.patch.set_facecolor("#0b0c10")
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    for k, (title, sysX, p, kw, phase) in enumerate(tiles):
        ph = phase_rotations(sysX, p)
        _, exp, _, _, _ = render_sky(sysX, p, rot_phase=ph["noon"],
                                     resolution=(560, 315))
        rgb, _, _, _, bodies = render_sky(
            sysX, p, rot_phase=ph[phase], exposure=exp,
            resolution=(560, 315), **kw)
        ax = axes[k]
        ax.imshow(rgb)
        ax.set_title(title, color="white", fontsize=9)
        sd = p.habitability.sky_description if p.habitability else ""
        extra = ""
        if bodies:
            extra = "\nsky bodies: " + ", ".join(
                b.name for b in bodies[:3])
        ax.text(0.02, 0.03,
                f"{p.habitability.dominant_gas} | "
                f"P={p.habitability.surface_pressure_bar:.2g} bar\n{sd}"
                + extra,
                transform=ax.transAxes, color="white", fontsize=7,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round", fc="#000000AA", ec="none"))

    # Black-body star-colour strip (proves multi-colour stars).
    teffs, swatches = _star_colour_strip()
    axs = axes[len(tiles)]
    axs.axis("on")
    axs.imshow(swatches[None, :, :], aspect="auto",
               extent=[0, len(teffs), 0, 1])
    axs.set_yticks([])
    axs.set_xticks(np.arange(len(teffs)) + 0.5)
    axs.set_xticklabels([f"{int(t)}K" for t in teffs],
                        color="white", fontsize=7, rotation=45)
    axs.set_title("Black-body star colours (background field)",
                  color="white", fontsize=9)
    for sp in axs.spines.values():
        sp.set_color("#444")

    fig.suptitle("Skyview robustness: atmosphere colours, multi-sun, "
                 "oblate world, multi-colour starfield",
                 color="white", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130, facecolor="#0b0c10",
                bbox_inches="tight")
    plt.close(fig)
    return out_path


def debug_sky_bodies(sysX, planet, base, *, latitude_deg=15.0,
                     n_scan=72, res=(640, 360)):
    """Debug visualisation that PROVES sibling planets + moons render in
    the sky: sweep a full day, mark and label every composited body
    (name, kind, phase, altitude) on the frame.  Writes an annotated
    contact sheet PNG and (if ffmpeg present) an annotated MP4 where the
    bodies visibly track across the sky."""
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    ph = phase_rotations(sysX, planet, latitude_deg=latitude_deg)
    _, exp, _, _, _ = render_sky(sysX, planet, rot_phase=ph["noon"],
                                 latitude_deg=latitude_deg, resolution=res)
    rots = np.linspace(0.0, 2.0 * np.pi, n_scan, endpoint=False) \
        + ph["midnight"]

    frames = []          # (rgb, [(name,kind,x,y,r,alt,phase), ...])
    for rp in rots:
        rgb, _, _, _, bodies = render_sky(
            sysX, planet, rot_phase=float(rp), latitude_deg=latitude_deg,
            exposure=exp, resolution=res)
        marks = [(b.name, b.kind, b.screen_x, b.screen_y,
                  max(b.screen_r, 7.0), b.altitude_deg, b.phase_frac)
                 for b in bodies]
        frames.append((rgb, marks))

    total = sorted({m[0] for _, ms in frames for m in ms})
    print(f"  debug: {len(total)} distinct sky bodies seen "
          f"({', '.join(total[:10])}{' ...' if len(total) > 10 else ''})")

    def _annot(ax, rgb, marks, title):
        ax.imshow(rgb)
        ax.axis("off")
        for nm, kind, x, y, r, alt, phf in marks:
            col = "#5CE0FF" if kind == "moon" else "#FFC04D"
            ax.add_patch(Circle((x, y), r + 4, fill=False,
                                edgecolor=col, lw=1.4))
            ax.annotate(f"{nm} ({kind}, {phf:.0%}, {alt:+.0f}°)",
                        (x, y), xytext=(x + r + 7, y - r - 7),
                        color=col, fontsize=7,
                        arrowprops=dict(arrowstyle="-", color=col,
                                        lw=0.6))
        ax.text(0.012, 0.975, title, transform=ax.transAxes, va="top",
                color="white", fontsize=8,
                bbox=dict(boxstyle="round", fc="#000000AA", ec="none"))

    # Contact sheet: the 6 frames with the most labelled bodies.
    best = sorted(range(len(frames)),
                  key=lambda i: -len(frames[i][1]))[:6]
    best = sorted(best)
    fig, axes = plt.subplots(2, 3, figsize=(18, 7))
    fig.patch.set_facecolor("#0b0c10")
    for ax, fi in zip(axes.ravel(), best):
        rgb, marks = frames[fi]
        _annot(ax, rgb, marks,
               f"{sysX.name}/{planet.name}  frame {fi}/{n_scan}  "
               f"{len(marks)} bodies")
    for ax in axes.ravel()[len(best):]:
        ax.axis("off")
    fig.suptitle(f"DEBUG: sibling planets + moons in the sky of "
                 f"{planet.name}  (orange=planet, cyan=moon)",
                 color="white", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = base + "_bodies_debug.png"
    fig.savefig(png, dpi=120, facecolor="#0b0c10", bbox_inches="tight")
    plt.close(fig)

    mp4 = None
    try:
        figA = plt.figure(figsize=(res[0] / 100, res[1] / 100), dpi=100)
        axA = figA.add_axes([0, 0, 1, 1])

        def _update(fi):
            axA.clear()
            rgb, marks = frames[fi]
            _annot(axA, rgb, marks,
                   f"{sysX.name}/{planet.name}  t={fi}/{n_scan}  "
                   f"{len(marks)} sky bodies")
            return []

        anim = FuncAnimation(figA, _update, frames=len(frames),
                             interval=80, blit=False)
        mp4 = base + "_bodies_debug.mp4"
        anim.save(mp4, writer=FFMpegWriter(
            fps=12, bitrate=2600, codec="libx264",
            extra_args=["-pix_fmt", "yuv420p",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]))
        plt.close(figA)
    except Exception as e:
        mp4 = None
        print(f"  debug MP4 skipped: {e}")
    return png, mp4


def main():
    seed = int(_sys.argv[1]) if len(_sys.argv) > 1 else 2026
    for sysX, planet in _build(seed):
        t0 = time.time()
        print(f"=== {sysX.name} / {planet.name} "
              f"({len(sysX.stars)} star(s), "
              f"{len(getattr(planet, 'moons', []))} moons) ===")
        if planet.habitability:
            print("  " + planet.habitability.sky_description
                  + f"  | P={planet.habitability.surface_pressure_bar:.2g} bar"
                  + f"  | obliquity {planet.habitability.obliquity_deg:.0f}deg")

        written = render_phases(sysX, planet, OUT_FIG)
        for phase, path in written.items():
            print(f"  {phase:<9s} -> {os.path.relpath(path, _ROOT)}")

        mp4 = os.path.join(
            OUT_ANI,
            f"{sysX.name}_{planet.name}".replace(" ", "_")
            .replace("/", "-") + "_day.mp4")
        try:
            animate_day(sysX, planet, mp4, n_frames=120)
            print(f"  day MP4   -> {os.path.relpath(mp4, _ROOT)}")
        except Exception as e:                       # ffmpeg missing etc.
            print(f"  day MP4 skipped: {e}")
        print(f"  [{time.time() - t0:.1f}s]\n")

    print("=== sky-colour gallery ===")
    t0 = time.time()
    gal = sky_colour_gallery(os.path.join(OUT_FIG,
                                          "sky_colour_gallery.png"))
    if gal:
        print(f"  gallery   -> {os.path.relpath(gal, _ROOT)}"
              f"  [{time.time() - t0:.1f}s]\n")

    # --- debug: sibling planets + moons in the sky (still + MP4) ---
    print("=== debug: sky bodies (moons / sibling planets) ===")
    sun = Star("Sun", mass=1.0, luminosity=1.0, teff=T_EFF_SUN_K,
               radius=1.0)
    earth = earth_analog("Earth")
    earth.moons = [Moon("Luna", mass_me=0.0123, radius_re=0.273,
                        a_planet_au=0.00257, eccentricity=0.0549,
                        density_gcc=3.34, kind="regular")]
    siblings = [earth, earth_analog("Marsish", a_au=1.6),
                Planet("Jovian", mass_me=180.0, radius_re=11.0,
                       semi_major_axis_au=3.2, host_star_index=0)]
    sdbg = StarSystem.single("Sol-dbg", sun, planets=siblings)
    for q in sdbg.planets:
        q.habitability = profile_for_planet(
            q, sdbg, np.random.default_rng(1), in_phz=(q is earth))
    rdbg = random_solar_system(np.random.default_rng(7), name="Aurelia-7")
    rphz = [p for p in rdbg.planets
            if p.habitability and p.habitability.in_phz] or rdbg.planets
    for tag, sx, pl in (("Sol-dbg", sdbg, earth),
                        ("Aurelia-7", rdbg, rphz[0])):
        t0 = time.time()
        png, mp4 = debug_sky_bodies(
            sx, pl, os.path.join(OUT_ANI, tag.replace(" ", "_")))
        print(f"  {tag}: still -> {os.path.relpath(png, _ROOT)}"
              + (f" ; mp4 -> {os.path.relpath(mp4, _ROOT)}" if mp4
                 else "")
              + f"  [{time.time() - t0:.1f}s]")

    print(f"\nfigures    -> {OUT_FIG}")
    print(f"animations -> {OUT_ANI}")


if __name__ == "__main__":
    main()
