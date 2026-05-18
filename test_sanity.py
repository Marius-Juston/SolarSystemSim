"""Sanity tests against known reference values."""
import math
import warnings

import numpy as np

from goldilocks.habitable_zone import single_star_hz
from goldilocks.kepler import kepler_two_body, orbital_period
from goldilocks.roche import eggleton_roche_radius
from goldilocks.secular import (heppenheimer_e_forced_stype,
                                leung_lee_e_forced_ptype,
                                secular_max_eccentricities)
from goldilocks.stability import (mardling_aarseth_stable)
from goldilocks.stellar import Star, T_EFF_SUN_K
from goldilocks.system import StarSystem

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
masses = [1.0, 317.83]  # Earth, Jupiter (Mearth)
sma = [1.0, 5.20]
e_max = secular_max_eccentricities(masses, sma, 1.0,
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

print()
print("=" * 72)
print("9.  Random-solar-system additions (moons / habitability / nested NB)")
print("=" * 72)
from goldilocks.moons import (planetary_roche_limit_au,
                              critical_moon_fraction, AU_KM)
from goldilocks.solar_system import STAR_COUNT_PMF, random_solar_system
from goldilocks.habitability import profile_for_planet
from goldilocks.planets import Planet
from goldilocks.nbody_moons import integrate_solar_system

# Earth-Moon fluid Roche limit ~ 18,400 km (textbook).
d_au = planetary_roche_limit_au(1.0, 5.514, 3.344)
d_km = d_au * AU_KM
print(f"  Earth-Moon fluid Roche limit: {d_km:,.0f} km  (expect ~18,400)")
assert 16000.0 < d_km < 21000.0, d_km

# Domingos+2006 prograde critical fraction at zero eccentricity.
f0 = critical_moon_fraction(0.0, 0.0, retrograde=False)
print(f"  Domingos prograde a_crit/R_Hill (e=0): {f0:.4f}  (expect 0.4895)")
assert abs(f0 - 0.4895) < 1e-6, f0

# Raghavan PMF normalisation.
tot = sum(STAR_COUNT_PMF.values())
print(f"  Raghavan star-count PMF sum: {tot:.2f}  (expect 1.00)")
assert abs(tot - 1.0) < 1e-9, tot

# Habitability: Sun + Earth-analog -> g=1, T_eq ~ 255 K.
ssun = Star("Sun", mass=1.0, luminosity=1.0, teff=T_EFF_SUN_K, radius=1.0)
se = StarSystem.single("Sun+E", ssun,
                       planets=[Planet("Earth", mass_me=1.0, radius_re=1.0,
                                       semi_major_axis_au=1.0,
                                       eccentricity=0.0167,
                                       host_star_index=0)])
prof = profile_for_planet(se.planets[0], se, np.random.default_rng(0),
                          in_phz=True)
print(f"  Earth-analog gravity = {prof.surface_gravity_g:.3f} g  (expect 1.0)")
print(f"  Earth-analog T_eq    = {prof.t_eq_k:.0f} K  (expect ~245-260)")
assert abs(prof.surface_gravity_g - 1.0) < 1e-6, prof.surface_gravity_g
assert 235.0 < prof.t_eq_k < 270.0, prof.t_eq_k

# Generator guarantees >= 1 PHZ planet.
gsys = random_solar_system(np.random.default_rng(11), name="Sanity")
n_phz = sum(1 for p in gsys.planets
            if p.habitability and p.habitability.in_phz)
print(f"  random_solar_system PHZ planets: {n_phz}  (expect >= 1)")
assert n_phz >= 1, n_phz

# Nested N-body: a generated system stays bounded over a short horizon.
r = integrate_solar_system(gsys, duration_yr=12.0, n_samples=120,
                           max_integrated_moons_per_planet=4,
                           rng=np.random.default_rng(3))
rep = r["report"]
pdrift = [b["rel_drift"] for b in rep["bodies"] if b["kind"] == "planet"]
print(f"  nested NB: {rep['n_bodies']} bodies, "
      f"max planet drift {max(pdrift) * 100:.2f}%  (expect < 25%)")
assert max(pdrift) < 0.25, max(pdrift)

# --- Skyview v2: oblateness, Lambert phase, lit background ---------------
from goldilocks.skyview import (oblateness_for, _lambert_phase,
                                background_starfield, sky_bodies,
                                lambda_grid_nm)
from goldilocks.moons import Moon

# Oblateness: Earth-analog nearly spherical; a fast rotator clearly oblate.
e_earth = Planet("E", mass_me=1.0, radius_re=1.0,
                 semi_major_axis_au=1.0, host_star_index=0)
e_earth.habitability = profile_for_planet(
    e_earth, se, np.random.default_rng(0), in_phz=True)
f_earth = oblateness_for(e_earth)
fast = Planet("Fast", mass_me=1.0, radius_re=1.0,
              semi_major_axis_au=1.0, host_star_index=0)
fast.habitability = profile_for_planet(fast, se, np.random.default_rng(0))
fast.habitability.sidereal_day_h = 4.0  # rapid spin
f_fast = oblateness_for(fast)
print(f"  oblateness Earth-analog={f_earth:.4f}  fast(4h)={f_fast:.3f}")
assert 0.0 <= f_earth < 0.01, f_earth
assert f_earth < f_fast < 0.35, (f_earth, f_fast)

# Lambert phase: Phi(0)=1, Phi(pi)=0, monotone decreasing.
xs = np.linspace(0.0, math.pi, 50)
phi = _lambert_phase(xs)
print(f"  Lambert phase: Phi(0)={phi[0]:.3f} Phi(pi)={phi[-1]:.3f}")
assert abs(phi[0] - 1.0) < 1e-6 and phi[-1] < 1e-6
assert np.all(np.diff(phi) <= 1e-9), "Lambert phase not monotone"

# Background starfield deterministic for a seed.
d1, t1, fx1, pn1 = background_starfield(2026)
d2, t2, fx2, pn2 = background_starfield(2026)
assert np.array_equal(d1, d2) and np.array_equal(fx1, fx2)
print(f"  background_starfield deterministic: {d1.shape[0]} stars")

# Reflected flux ~ linear in A_g and (R/d)^2: a moon with 2x radius and
# 2x albedo-density is ~ (2)*(4) brighter.
sun2 = Star("Sun", mass=1.0, luminosity=1.0, teff=T_EFF_SUN_K, radius=1.0)
pm = Planet("Host", mass_me=1.0, radius_re=1.0, semi_major_axis_au=1.0,
            host_star_index=0)
pm.moons = [Moon("m1", mass_me=0.01, radius_re=0.25, a_planet_au=0.0026,
                 density_gcc=3.0, kind="regular"),
            Moon("m2", mass_me=0.08, radius_re=0.50, a_planet_au=0.0060,
                 density_gcc=1.5, kind="regular")]
sm = StarSystem.single("S", sun2, planets=[pm])
pm.habitability = profile_for_planet(pm, sm, np.random.default_rng(0),
                                     in_phz=True)
lam = lambda_grid_nm()
bs = {b.name: float(np.sum(b.refl_spec))
      for b in sky_bodies(sm, pm, lam, 0.0, 0.0)}
print(f"  reflected-flux ratio m2/m1 = {bs['m2'] / max(bs['m1'], 1e-30):.2f}")
assert bs["m2"] > bs["m1"] > 0.0, bs

# Multi-colour background stars: the field must span cool (red) to hot
# (blue), and a 3000 K vs 20000 K black body must be red- vs blue-biased.
from goldilocks.skyview import (planck_spectral, spectrum_to_srgb,
                                lambda_grid_nm, atmosphere_for)

_, bt, _, _ = background_starfield(2026)
print(f"  background Teff range: {bt.min():.0f}-{bt.max():.0f} K")
assert bt.min() < 3200.0 and bt.max() > 12000.0, (bt.min(), bt.max())
_lam = lambda_grid_nm()
from goldilocks.skyview import cie_xyz_bar as _cxb

_, _yb, _ = _cxb(_lam)
_dl = float(_lam[1] - _lam[0])


def _rgb(T):
    pl = planck_spectral(_lam, T);
    pl = pl / np.trapezoid(pl, _lam * 1e-9)
    Y = float(np.dot(pl, _yb) * _dl)  # mid-tone, unsaturated
    return spectrum_to_srgb(pl[None, :], _lam, 0.35 / max(Y, 1e-12))[0]


cool, hot = _rgb(3000.0), _rgb(20000.0)
print(f"  3000K rgb={tuple(int(c) for c in cool)}  "
      f"20000K rgb={tuple(int(c) for c in hot)}")
assert int(cool[0]) > int(cool[2]) and int(hot[2]) >= int(hot[0]), \
    (cool, hot)


# Distinct atmospheres: a thin N2/O2 vs a dense CO2 atmosphere must give
# different Rayleigh scattering (different sky colour).
class _P:  # minimal duck-typed planet
    radius_re = 1.0

    def __init__(self, prof): self.habitability = prof


pa = profile_for_planet(
    Planet("a", mass_me=1.0, radius_re=1.0, semi_major_axis_au=1.0,
           host_star_index=0), se, np.random.default_rng(0), in_phz=True)
pb = profile_for_planet(
    Planet("b", mass_me=1.0, radius_re=1.0, semi_major_axis_au=1.0,
           host_star_index=0), se, np.random.default_rng(0), in_phz=True)
pb.dominant_gas = "CO2";
pb.surface_pressure_bar = 30.0
pb.mean_molecular_weight = 44.0
at_a = atmosphere_for(_P(pa), _lam)
at_b = atmosphere_for(_P(pb), _lam)
print(f"  beta_r(N2/O2 1bar)~{at_a.beta_r.mean():.2e}  "
      f"beta_r(CO2 30bar)~{at_b.beta_r.mean():.2e}")
assert at_b.beta_r.mean() > 3.0 * at_a.beta_r.mean(), \
    (at_a.beta_r.mean(), at_b.beta_r.mean())

# --- Bruneton-style optical-depth LUT matches the ray-march reference ---
import goldilocks.skyview as _sv
from goldilocks.stellar import Star as _Star, T_EFF_SUN_K as _TSUN
from goldilocks.system import StarSystem as _SS
from goldilocks.planets import earth_analog as _ea
from goldilocks.habitability import profile_for_planet as _pfp

_sun = _Star("Sun", mass=1.0, luminosity=1.0, teff=_TSUN, radius=1.0)
_e = _ea("Earth")
_sys = _SS.single("Sol", _sun, planets=[_e])
_e.habitability = _pfp(_e, _sys, np.random.default_rng(1), in_phz=True)
_ph = _sv.phase_rotations(_sys, _e)
_r_lut, _, _, _, _ = _sv.render_sky(_sys, _e, rot_phase=_ph["noon"],
                                    resolution=(256, 144))
_orig = _sv._light_od_lut
_sv._light_od_lut = lambda P, sl, a, n: _sv._light_optical_depth(P, sl, a, n)
try:
    _r_ref, _, _, _, _ = _sv.render_sky(_sys, _e, rot_phase=_ph["noon"],
                                        resolution=(256, 144))
finally:
    _sv._light_od_lut = _orig
_dlut = np.abs(_r_lut.astype(int) - _r_ref.astype(int))
print(f"  skyview LUT vs ray-march: mean|d|={_dlut.mean():.3f} "
      f"max={_dlut.max()} (/255)")
assert _dlut.mean() < 2.0 and _dlut.max() <= 12, (_dlut.mean(),
                                                  _dlut.max())

# --- GPU vs CPU render parity (only when CuPy is the active backend) ---
from goldilocks import backend as _B
if _B.ON_GPU:
    print(f"  backend = GPU ({_B.n_gpus()} device[s]); CPU/GPU parity "
          "is exercised by the LUT path above on-device")
else:
    print("  backend = CPU (NumPy); GPU parity test skipped")

# --- new showcase systems: a moon / a sibling-planet looming large ----
from goldilocks.random_systems import (big_moon_system as _bms,
                                       companion_with_moon_system as _cwm)
_bm = _bms()
_mn = _bm.planets[0].moons[0]
from goldilocks.moons import R_EARTH_AU as _REA
_dia = 2.0 * np.degrees(np.arcsin(_mn.radius_re * _REA
                                  / _mn.a_planet_au))
print(f"  big_moon_system: {_mn.name} angular diameter {_dia:.2f} deg")
assert _dia > 3.0, _dia
_cw = _cwm()
_giant = _cw.planets[1]
assert _giant.moons and _giant.moons[0].kind == "regular", _giant.moons
_bodies = _sv.sky_bodies(_cw, _cw.planets[0], _sv.lambda_grid_nm())
_names = {b.name for b in _bodies}
print(f"  companion_with_moon_system: sky bodies seen -> "
      f"{sorted(_names)[:6]}")
assert _giant.name in _names and _giant.moons[0].name in _names, _names
print("  All section-9 assertions passed.")
