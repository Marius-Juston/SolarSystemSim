"""Sanity tests against known reference values."""
import math
import warnings
import numpy as np

from goldilocks.stellar           import Star, T_EFF_SUN_K
from goldilocks.habitable_zone    import single_star_hz, seff_for_limit, hz_distance
from goldilocks.stability         import (holman_wiegert_stype, holman_wiegert_ptype,
                               mardling_aarseth_stable,
                               max_planets_in_zone)
from goldilocks.roche             import eggleton_roche_radius, roche_lobe_periastron
from goldilocks.secular           import (heppenheimer_e_forced_stype,
                               leung_lee_e_forced_ptype,
                               laplace_coefficient,
                               secular_max_eccentricities)
from goldilocks.kepler            import kepler_two_body, orbital_period
from goldilocks.system            import StarSystem
from goldilocks.planets           import (earth_analog, kepler16_b, kepler34_b,
                               kepler35_b, kepler38_b,
                               kepler47_b, kepler47_d, kepler47_c)


warnings.simplefilter("default")


print("=" * 72)
print("1.  Single-star HZ (the Sun)")
print("=" * 72)
sun = Star("Sun", mass=1.0, luminosity=1.0, teff=T_EFF_SUN_K, radius=1.0)
bnd = single_star_hz(sun)
print(f"  Recent Venus  : {bnd.optimistic_inner:.3f} AU  (expect ~0.75)")
print(f"  Runaway GH    : {bnd.conservative_inner:.3f} AU  (expect ~0.95)")
print(f"  Maximum GH    : {bnd.conservative_outer:.3f} AU  (expect ~1.67)")
print(f"  Early Mars    : {bnd.optimistic_outer:.3f} AU  (expect ~1.77)")

print()
print("=" * 72)
print("2.  Eggleton 1983 Roche-lobe formula")
print("=" * 72)
# Sun + Earth: q = 1Msun / 3e-6 Msun = 3.3e5, r_L/A ~ 0.62 for q -> infinity
print(f"  r_L(Sun, Earth, 1 AU) = {eggleton_roche_radius(1.0, 3e-6, 1.0):.4f} AU"
      f"  (~0.6 AU)")
# Equal-mass binary: r_L/A = 0.380 (textbook)
print(f"  r_L(M=M, A=1 AU)      = {eggleton_roche_radius(1.0, 1.0, 1.0):.4f} AU"
      f"  (expect 0.380)")

print()
print("=" * 72)
print("3.  Kepler 2-body closed-form solver")
print("=" * 72)
# Earth-Sun: at t = 0 (periastron, e=0.0167), Earth should be at ~0.983 AU
r1, r2, v1, v2 = kepler_two_body(1.0, 3e-6, 1.0, 0.0167, 0.0)
# r2 is the Earth's position relative to the barycentre.
print(f"  Earth periastron (rel. barycentre): {np.linalg.norm(r2):.6f} AU")
print(f"  Earth speed at periastron:          {np.linalg.norm(v2):.5f} AU/yr"
      f"  (expect ~6.39)")  # 2*pi*1 / 1yr for circular = 6.28

# Period of Alpha Cen AB:
P = orbital_period(1.10, 0.907, 23.4)
print(f"  Alpha Cen AB period (a=23.4 AU):    {P:.1f} yr  (expect 79.9)")

print()
print("=" * 72)
print("4.  Heppenheimer + Leung-Lee forced eccentricities")
print("=" * 72)
print(f"  e_f(a_p=1, a_b=20, e_b=0.5)   = "
      f"{heppenheimer_e_forced_stype(1.0, 20.0, 0.5):.4f}")
print(f"  e_f(a_p=2, a_b=20, e_b=0.5)   = "
      f"{heppenheimer_e_forced_stype(2.0, 20.0, 0.5):.4f}  (2x)")
print(f"  e_f_P(a_p=1, a_b=0.22, e_b=0.16, mu=0.23) = "
      f"{leung_lee_e_forced_ptype(1.0, 0.22, 0.16, 0.23):.4f}  (Kepler-16 b)")

print()
print("=" * 72)
print("5.  Laplace-Lagrange coupling: 2-planet test")
print("=" * 72)
# Earth + Jupiter-mass at 1, 5 AU around Sun: classical case
# Earth's forced e ~ a few * 1e-2 from Jupiter, modes are well known.
masses = [1.0, 317.83]                # Earth, Jupiter (Mearth)
sma    = [1.0, 5.20]
e_max  = secular_max_eccentricities(masses, sma, 1.0,
                                    initial_eccentricities=[0.0167, 0.0489])
print(f"  Earth-Jupiter e_max: Earth={e_max[0]:.4f}, Jupiter={e_max[1]:.4f}")
print(f"  (expected: Earth ~0.063 (max), Jupiter ~0.061)")

print()
print("=" * 72)
print("6.  Alpha Centauri AB binary (close enough to perturb HZ)")
print("=" * 72)
A = Star("Alpha Cen A", mass=1.10)
B = Star("Alpha Cen B", mass=0.907)
ac = StarSystem.binary("Alpha Cen AB", A, B, separation_au=23.4,
                       eccentricity=0.52)
print(ac.summary(planet_mass_me=1.0, delta=10.0, use_phz=True))

print()
print("=" * 72)
print("7.  Wide-binary warning (should warn it's effectively independent)")
print("=" * 72)
P1 = Star("Wide-A", mass=1.0)
P2 = Star("Wide-B", mass=0.5)
# 1000 AU is way wider than any HZ, should warn.
with warnings.catch_warnings(record=True) as w:
    wide = StarSystem.binary("Wide", P1, P2, separation_au=1000.0, eccentricity=0.0)
    if any("effectively independent" in str(ww.message) for ww in w):
        print("  Warning correctly raised for 1000-AU 'binary'.")

print()
print("=" * 72)
print("8.  Mardling-Aarseth hierarchical stability check")
print("=" * 72)
# Alpha Cen A+B at 23.4 AU + Proxima at 8700 AU, e_out = 0.5
ok = mardling_aarseth_stable(23.4, 8700.0, 0.50,
                              1.10, 0.907, 0.122)
print(f"  Alpha Cen triple stable?  {ok}  (expect True)")
ok = mardling_aarseth_stable(23.4, 50.0, 0.50,
                              1.10, 0.907, 0.122)
print(f"  Same triple at a_out=50: {ok}  (expect False)")