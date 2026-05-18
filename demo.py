"""
demo.py
-------
End-to-end demonstration of the Goldilocks calculator on TEN systems:

  Core six (covers the main physics regimes):
    1. The Sun + Earth (e = 0.0167, real value)
    2. Alpha Centauri AB (close binary, S-type PHZ on each star)
    3. Kepler-16 (tight circumbinary, real planet at PHZ edge)
    4. Kepler-47 (multi-planet circumbinary)
    5. Alpha Centauri + Proxima (hierarchical triple, multi-scale view)
    6. Synthetic G+G+M wide hierarchy

  Four "interesting" cases:
    7. Star-hopper (Moeckel & Veras 2012): e=0.7 planet bouncing near L1
    8. Trojan / co-orbital companion
    9. Wide hierarchical binary with M-dwarf companion + HZ Earth
   10. Polar planet (i = 75 deg)

Each system gets a static PNG figure AND an animated MP4.
"""

import os
import time

import matplotlib

matplotlib.use("Agg")

from goldilocks.stellar import Star, T_EFF_SUN_K
from goldilocks.planets import (earth_analog, kepler16_b, kepler47_b,
                                kepler47_c, kepler47_d)
from goldilocks.system import StarSystem
from goldilocks.visualization import plot_system
from goldilocks.animation import animate_system
from goldilocks.random_systems import (star_hopper_binary, trojan_system,
                                       wide_hierarchical_triple,
                                       polar_planet_system,
                                       big_moon_system)
from goldilocks import parallel as _P

_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_FIG = os.path.join(_ROOT, "figures")
OUT_ANI = os.path.join(_ROOT, "animations")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_ANI, exist_ok=True)


def make(sys, base_name, extent=None, animate=True,
         n_frames=120, n_periods=1.0, duration_yr=None,
         sub_steps=40):
    t0 = time.time()
    print(f"\n=== {base_name} : {sys.name} ===")
    print(sys.summary(planet_mass_me=1.0, delta=10.0, use_phz=True))
    fig = plot_system(sys, planet_mass_me=1.0, delta=10.0,
                      extent_au=extent,
                      save_path=os.path.join(OUT_FIG, f"{base_name}.png"))
    matplotlib.pyplot.close(fig)
    print(f"  static PNG done ({time.time() - t0:.1f}s)")
    if animate:
        t1 = time.time()
        animate_system(sys,
                       save_path=os.path.join(OUT_ANI, f"{base_name}.mp4"),
                       extent_au=extent,
                       n_frames=n_frames, n_periods=n_periods,
                       duration_yr=duration_yr,
                       sub_steps=sub_steps)
        print(f"  animation MP4 done ({time.time() - t1:.1f}s)")
    return base_name


def _run(job):
    """Picklable pool entry point (one system per pool task)."""
    sys = job.pop("sys")
    return make(sys, **job)


def _build_jobs():
    """All demo systems as picklable (sys, base_name, kwargs) jobs."""
    import numpy as np
    jobs = []

    sun = Star("Sun", mass=1.0, luminosity=1.0, teff=T_EFF_SUN_K,
               radius=1.0)
    jobs.append(dict(sys=StarSystem.single("Sun + Earth", sun,
                                            planets=[earth_analog("Earth")]),
                      base_name="01_sun", extent=1.8, n_frames=80,
                      duration_yr=2.0, sub_steps=200))

    A = Star("Alpha Cen A", mass=1.10)
    B = Star("Alpha Cen B", mass=0.907)
    jobs.append(dict(sys=StarSystem.binary("Alpha Cen AB", A, B,
                                            separation_au=23.4,
                                            eccentricity=0.52),
                      base_name="02_alphacen", extent=30.0,
                      n_frames=120, n_periods=1.0, sub_steps=40))

    k16a = Star("Kepler-16 A", mass=0.6897, luminosity=0.148, teff=4450.0)
    k16b = Star("Kepler-16 B", mass=0.2026, luminosity=0.0057, teff=3310.0)
    jobs.append(dict(sys=StarSystem.binary("Kepler-16", k16a, k16b,
                                            separation_au=0.2243,
                                            eccentricity=0.1594,
                                            planets=[kepler16_b()]),
                      base_name="03_kepler16", extent=1.4, n_frames=120,
                      n_periods=8.0, sub_steps=40))

    k47a = Star("Kepler-47 A", mass=1.04, luminosity=0.84, teff=5636.0)
    k47b = Star("Kepler-47 B", mass=0.342, luminosity=0.0177, teff=3357.0)
    jobs.append(dict(sys=StarSystem.binary(
        "Kepler-47", k47a, k47b, separation_au=0.0836,
        eccentricity=0.0234,
        planets=[kepler47_b(), kepler47_d(), kepler47_c()]),
        base_name="04_kepler47", extent=1.5, n_frames=120,
        n_periods=40.0, sub_steps=40))

    A2 = Star("Alpha Cen A", mass=1.10)
    B2 = Star("Alpha Cen B", mass=0.907)
    Pr = Star("Proxima", mass=0.122)
    jobs.append(dict(sys=StarSystem.hierarchical_triple(
        "Alpha Cen triple", A2, B2, Pr, a_in=23.4, e_in=0.52,
        a_out=8700.0, e_out=0.50),
        base_name="05_alphacen_triple", extent=30.0, n_frames=120,
        n_periods=1.0, sub_steps=40))

    G1 = Star("Gamma A", mass=1.048, luminosity=1.49, teff=5913.0)
    G2 = Star("Gamma B", mass=1.021, luminosity=1.28, teff=5867.0)
    Mc = Star("Gamma C", mass=0.30, luminosity=0.012, teff=3500.0)
    jobs.append(dict(sys=StarSystem.hierarchical_triple(
        "G+G + M wide-hierarchy", G1, G2, Mc, a_in=0.23, e_in=0.52,
        a_out=30.0, e_out=0.20),
        base_name="06_quadruple", extent=10.0, n_frames=120,
        n_periods=0.5, sub_steps=40))

    jobs.append(dict(sys=star_hopper_binary(
        name="Star-hopper (Moeckel & Veras 2012)", a_bin=40.0,
        e_bin=0.0, m1=1.0, m2=0.3, planet_a=14.0, planet_e=0.70),
        base_name="07_star_hopper", extent=45.0, n_frames=180,
        duration_yr=120.0, sub_steps=200))

    jobs.append(dict(sys=trojan_system(name="Trojan / co-orbital pair"),
                     base_name="08_trojan", extent=1.5, n_frames=80,
                     duration_yr=4.0, sub_steps=80))

    jobs.append(dict(sys=wide_hierarchical_triple(
        name="Wide hierarchy with M-dwarf HZ Earth",
        rng=np.random.default_rng(seed=7)),
        base_name="09_wide_hierarchy", extent=2.0, n_frames=120,
        n_periods=0.05, sub_steps=40))

    jobs.append(dict(sys=polar_planet_system(
        name="K-dwarf with polar HZ planet (i=75 deg)"),
        base_name="10_polar_planet", extent=1.0, n_frames=80,
        duration_yr=1.0, sub_steps=100))

    jobs.append(dict(sys=big_moon_system(
        name="Big-moon world (dramatic close moon)"),
        base_name="11_big_moon", extent=1.6, n_frames=80,
        duration_yr=1.0, sub_steps=80))
    return jobs


if __name__ == "__main__":
    jobs = _build_jobs()
    # Whole systems are independent -> fan across the pool (one worker
    # per GPU, or a large CPU process pool).
    done = _P.map_ordered(_run, jobs)

    print("\n\n" + "=" * 70)
    print(f"DONE -- {len(done)} systems: {', '.join(done)}")
    print("=" * 70)
    print(f"Static figures -> {OUT_FIG}")
    print(f"Animations     -> {OUT_ANI}")
