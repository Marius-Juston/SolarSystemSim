"""
generate_solar_system.py
------------------------
End-to-end driver: generate a random full solar system (Raghavan-2010
star multiplicity), guaranteed to have >= 1 planet in the PHZ, populate
it with terrestrial + giant planets and their moons, attach a
comprehensive habitability profile to every planet, verify global
stability with a full *nested* N-body integration (moons in planets in
analytic stars), and render static PNGs + MP4 animations.

Usage:  uv run python generate_solar_system.py [seed]
Output: figures/  and  animations/  (repo-relative; no hardcoded paths).
"""

import os
import sys as _sys
import time

import matplotlib

matplotlib.use("Agg")

import numpy as np

from goldilocks.solar_system import random_solar_system
from goldilocks.nbody_moons import integrate_solar_system
from goldilocks.viz_solar_system import (plot_overview, plot_all_planet_zooms,
                                         plot_longterm_stability,
                                         animate_solar_system)

_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_FIG = os.path.join(_ROOT, "figures")
OUT_ANI = os.path.join(_ROOT, "animations")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_ANI, exist_ok=True)


def main():
    seed = int(_sys.argv[1]) if len(_sys.argv) > 1 else 2026

    for n_stars in range(1, 4):
        gen_system(seed * n_stars, n_stars)


def gen_system(seed, n):
    rng = np.random.default_rng(seed)
    t0 = time.time()

    print(f"=== Generating random solar system (seed={seed}) with {n} stars ===")
    system = random_solar_system(rng, name=f"Aurelia-{seed}", n_stars=n)
    print(system.summary(use_phz=True))
    print("\n" + system.generation_note + "\n")
    for p in system.planets:
        if p.habitability:
            print(p.habitability.summary)
    print(f"\n[generated in {time.time() - t0:.1f}s]")

    # ----- static figures -----
    print("Rendering overview + per-planet zoom cards...")
    plot_overview(system, os.path.join(OUT_FIG, f"{seed}_{n}_overview.png"))
    plot_all_planet_zooms(system, os.path.join(OUT_FIG, f"{seed}_{n}_planets.png"))

    # ----- long-term nested N-body verification -----
    print("Running nested N-body stability verification...")
    t1 = time.time()
    result = integrate_solar_system(
        system, duration_yr=60.0, n_samples=500,
        max_integrated_moons_per_planet=10,
        rng=np.random.default_rng(seed + 1))
    rep = result["report"]
    print(f"  verdict: {'STABLE' if rep['stable'] else 'UNSTABLE'}  "
          f"({rep['n_bodies']} bodies, horizon {rep['horizon_yr']:.0f} yr, "
          f"{rep['sub_steps']} sub-steps/sample) [{time.time() - t1:.1f}s]")
    for b in rep["bodies"]:
        if b["kind"] == "planet" or not b["stable"]:
            flag = "" if b["stable"] else "  <-- FLAGGED"
            print(f"    {b['label']:<16s} {b['kind']:<7s} "
                  f"a~{b['a_au']:.3g} AU  e~{b['e']:.3f}  "
                  f"drift {b['rel_drift'] * 100:.2f}%{flag}")

    plot_longterm_stability(result, system,
                            os.path.join(OUT_FIG, f"{seed}_{n}_stability.png"))

    # ----- animations -----
    print("Rendering MP4 animations (requires ffmpeg)...")
    try:
        animate_solar_system(
            result, system,
            os.path.join(OUT_ANI, f"{seed}_{n}_system.mp4"))
        print("  system MP4 done")
    except Exception as e:  # ffmpeg missing etc.
        print(f"  animation skipped: {e}")

    print(f"\nDONE in {time.time() - t0:.1f}s")
    print(f"  figures    -> {OUT_FIG}")
    print(f"  animations -> {OUT_ANI}")


if __name__ == "__main__":
    main()
