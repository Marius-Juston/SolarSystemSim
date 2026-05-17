"""
render_skyview.py
-----------------
Photorealistic ground-to-sky renderer driver.  For each test system it
attaches a habitability profile to a habitable planet, then writes the
four canonical lighting situations (midnight / sunrise / noon / sunset)
as PNGs plus a full solar-day MP4.

Tested regimes:
  1. Single   -- Sun + Earth-analog (blue Rayleigh sky baseline).
  2. Binary   -- Alpha Cen A/B; S-type planet round A (two-sun sky: a
                  brilliant companion sits at an independent altitude).
  3. Triple   -- Alpha Cen A/B + Proxima hierarchical triple.

Usage:  uv run python render_skyview.py [seed]
Output: figures/skyview/  and  animations/skyview/  (repo-relative).
"""

import os
import sys as _sys
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np

from goldilocks.stellar import Star, T_EFF_SUN_K
from goldilocks.system import StarSystem
from goldilocks.planets import earth_analog
from goldilocks.habitability import profile_for_planet
from goldilocks.skyview import render_phases, animate_day

_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_FIG = os.path.join(_ROOT, "figures", "skyview")
OUT_ANI = os.path.join(_ROOT, "animations", "skyview")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_ANI, exist_ok=True)


def _build(seed: int):
    rng = np.random.default_rng(seed)

    sun = Star("Sun", mass=1.0, luminosity=1.0, teff=T_EFF_SUN_K, radius=1.0)
    sys1 = StarSystem.single("Sol", sun, planets=[earth_analog("Earth")])

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
    return cases


def main():
    seed = int(_sys.argv[1]) if len(_sys.argv) > 1 else 2026
    for sysX, planet in _build(seed):
        t0 = time.time()
        print(f"=== {sysX.name} / {planet.name} "
              f"({len(sysX.stars)} star(s)) ===")
        if planet.habitability:
            print("  " + planet.habitability.sky_description
                  + f"  | P={planet.habitability.surface_pressure_bar:.2g} bar"
                  + f"  | obliquity {planet.habitability.obliquity_deg:.0f}deg")

        written = render_phases(sysX, planet, OUT_FIG)
        for phase, path in written.items():
            print(f"  {phase:<9s} -> {os.path.relpath(path, _ROOT)}")

        mp4 = os.path.join(
            OUT_ANI,
            f"{sysX.name}_{planet.name}".replace(" ", "_") + "_day.mp4")
        try:
            animate_day(sysX, planet, mp4, n_frames=120)
            print(f"  day MP4   -> {os.path.relpath(mp4, _ROOT)}")
        except Exception as e:                       # ffmpeg missing etc.
            print(f"  day MP4 skipped: {e}")
        print(f"  [{time.time() - t0:.1f}s]\n")

    print(f"figures    -> {OUT_FIG}")
    print(f"animations -> {OUT_ANI}")

if __name__ == "__main__":
    main()