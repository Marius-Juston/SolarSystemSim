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

import os, math, time
import matplotlib
matplotlib.use("Agg")

from stellar       import Star, T_EFF_SUN_K
from planets       import (earth_analog, kepler16_b, kepler47_b,
                            kepler47_c, kepler47_d)
from system        import StarSystem
from visualization import plot_system
from animation     import animate_system
from random_systems import (star_hopper_binary, trojan_system,
                              wide_hierarchical_triple,
                              polar_planet_system)


OUT_FIG = "/home/claude/goldilocks/figures"
OUT_ANI = "/home/claude/goldilocks/animations"
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
    print(f"  static PNG done ({time.time()-t0:.1f}s)")
    if animate:
        t1 = time.time()
        animate_system(sys,
                       save_path=os.path.join(OUT_ANI, f"{base_name}.mp4"),
                       extent_au=extent,
                       n_frames=n_frames, n_periods=n_periods,
                       duration_yr=duration_yr,
                       sub_steps=sub_steps)
        print(f"  animation MP4 done ({time.time()-t1:.1f}s)")


# ---------------------------------------------------------------------
# 1. Sun + Earth (real e = 0.0167)
# ---------------------------------------------------------------------
sun  = Star("Sun", mass=1.0, luminosity=1.0, teff=T_EFF_SUN_K, radius=1.0)
sys1 = StarSystem.single("Sun + Earth", sun,
                          planets=[earth_analog("Earth")])
make(sys1, "01_sun", extent=1.8, n_frames=80,
     duration_yr=2.0, sub_steps=200)


# ---------------------------------------------------------------------
# 2. Alpha Centauri AB
# ---------------------------------------------------------------------
A = Star("Alpha Cen A", mass=1.10)
B = Star("Alpha Cen B", mass=0.907)
sys2 = StarSystem.binary("Alpha Cen AB", A, B,
                          separation_au=23.4, eccentricity=0.52)
make(sys2, "02_alphacen", extent=30.0, n_frames=120, n_periods=1.0,
     sub_steps=40)


# ---------------------------------------------------------------------
# 3. Kepler-16 b
# ---------------------------------------------------------------------
k16a = Star("Kepler-16 A", mass=0.6897, luminosity=0.148, teff=4450.0)
k16b = Star("Kepler-16 B", mass=0.2026, luminosity=0.0057, teff=3310.0)
sys3 = StarSystem.binary("Kepler-16", k16a, k16b,
                          separation_au=0.2243, eccentricity=0.1594,
                          planets=[kepler16_b()])
make(sys3, "03_kepler16", extent=1.4, n_frames=120, n_periods=8.0,
     sub_steps=40)


# ---------------------------------------------------------------------
# 4. Kepler-47 (3 real circumbinary planets)
# ---------------------------------------------------------------------
k47a = Star("Kepler-47 A", mass=1.04, luminosity=0.84, teff=5636.0)
k47b = Star("Kepler-47 B", mass=0.342, luminosity=0.0177, teff=3357.0)
sys4 = StarSystem.binary("Kepler-47", k47a, k47b,
                          separation_au=0.0836, eccentricity=0.0234,
                          planets=[kepler47_b(), kepler47_d(), kepler47_c()])
make(sys4, "04_kepler47", extent=1.5, n_frames=120, n_periods=40.0,
     sub_steps=40)


# ---------------------------------------------------------------------
# 5. Alpha Cen + Proxima (hierarchical triple, multi-scale view)
# ---------------------------------------------------------------------
A2 = Star("Alpha Cen A", mass=1.10)
B2 = Star("Alpha Cen B", mass=0.907)
Pr = Star("Proxima",     mass=0.122)
sys5 = StarSystem.hierarchical_triple("Alpha Cen triple",
                                       A2, B2, Pr,
                                       a_in=23.4, e_in=0.52,
                                       a_out=8700.0, e_out=0.50)
make(sys5, "05_alphacen_triple", extent=30.0, n_frames=120, n_periods=1.0,
     sub_steps=40)


# ---------------------------------------------------------------------
# 6. Synthetic G+G+M hierarchical (outer 30 AU, more compact)
# ---------------------------------------------------------------------
G1 = Star("Gamma A",  mass=1.048, luminosity=1.49,  teff=5913.0)
G2 = Star("Gamma B",  mass=1.021, luminosity=1.28,  teff=5867.0)
Mc = Star("Gamma C",  mass=0.30,  luminosity=0.012, teff=3500.0)
sys6 = StarSystem.hierarchical_triple("G+G + M wide-hierarchy",
                                       G1, G2, Mc,
                                       a_in=0.23, e_in=0.52,
                                       a_out=30.0, e_out=0.20)
make(sys6, "06_quadruple", extent=10.0, n_frames=120, n_periods=0.5,
     sub_steps=40)


# ---------------------------------------------------------------------
# 7. Star-hopper (Moeckel-Veras 2012) - bouncing between L1 and host
# ---------------------------------------------------------------------
sys7 = star_hopper_binary(name="Star-hopper (Moeckel & Veras 2012)",
                           a_bin=40.0, e_bin=0.0,
                           m1=1.0, m2=0.3,
                           planet_a=14.0, planet_e=0.70)
# Period ~ 14^1.5 / sqrt(1.3) = 46 yr.  Let's animate 100 yr (~ 2 planet
# periods); plenty to see non-Keplerian bouncing.
make(sys7, "07_star_hopper", extent=45.0, n_frames=180,
     duration_yr=120.0, sub_steps=200)


# ---------------------------------------------------------------------
# 8. Trojan co-orbital
# ---------------------------------------------------------------------
sys8 = trojan_system(name="Trojan / co-orbital pair")
make(sys8, "08_trojan", extent=1.5, n_frames=80,
     duration_yr=4.0, sub_steps=80)


# ---------------------------------------------------------------------
# 9. Wide hierarchical with HZ Earth around M-dwarf
# ---------------------------------------------------------------------
import numpy as np
sys9 = wide_hierarchical_triple(name="Wide hierarchy with M-dwarf HZ Earth",
                                  rng=np.random.default_rng(seed=7))
make(sys9, "09_wide_hierarchy", extent=2.0, n_frames=120, n_periods=0.05,
     sub_steps=40)


# ---------------------------------------------------------------------
# 10. Polar planet
# ---------------------------------------------------------------------
sys10 = polar_planet_system(name="K-dwarf with polar HZ planet (i=75 deg)")
make(sys10, "10_polar_planet", extent=1.0, n_frames=80,
     duration_yr=1.0, sub_steps=100)


print("\n\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"Static figures -> {OUT_FIG}")
print(f"Animations     -> {OUT_ANI}")