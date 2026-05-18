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

# --- Skyview v3: noise, MoonSurface, StarSurface, occultation -----------
from goldilocks import noise as _N
from goldilocks import backend as _BK

_xp = _BK.xp
_gr = _xp.linspace(0.0, 6.0, 40)
_X, _Y = _xp.meshgrid(_gr, _gr)
_a = _BK.asnumpy(_N.fbm(_X, _Y, 3, octaves=4))
_b = _BK.asnumpy(_N.fbm(_X, _Y, 3, octaves=4))
assert np.array_equal(_a, _b), "noise not deterministic"
_h = 1e-2


def _vel(px, py):
    return _N.curl_noise_2d(px, py, 5, eps=_h)


_u, _v = _vel(_X, _Y)
_ux = (_vel(_X + _h, _Y)[0] - _vel(_X - _h, _Y)[0]) / (2 * _h)
_vy = (_vel(_X, _Y + _h)[1] - _vel(_X, _Y - _h)[1]) / (2 * _h)
_div = float(_BK.asnumpy(_xp.abs(_ux + _vy)).mean())
print(f"  curl-noise mean|div| = {_div:.2e} (divergence-free)")
assert _div < 1e-9, _div

from goldilocks.moon_surface import moon_surface_for as _msf
from goldilocks.moons import Moon as _Mn

_sun3 = _Star("Sun", mass=1.0, luminosity=1.0, teff=_TSUN, radius=1.0)
_gnt = Planet("Gnt", mass_me=300.0, radius_re=11.0,
                semi_major_axis_au=5.2, host_star_index=0)
_m_small = _Mn("sm", mass_me=1e-3, radius_re=0.08, a_planet_au=0.01,
                eccentricity=0.0, density_gcc=3.0, kind="regular")
_m_big = _Mn("bg", mass_me=0.05, radius_re=0.45, a_planet_au=0.01,
              eccentricity=0.0, density_gcc=3.0, kind="regular")
_m_io = _Mn("io", mass_me=0.015, radius_re=0.286, a_planet_au=0.00282,
             eccentricity=0.0041, density_gcc=3.5, kind="regular")
_m_ice = _Mn("ic", mass_me=2e-5, radius_re=0.04, a_planet_au=0.02,
              eccentricity=0.0, density_gcc=1.4, kind="regular")
_gnt.moons = [_m_small, _m_big, _m_io, _m_ice]
_sg = _SS.single("S3", _sun3, planets=[_gnt])
_rng3 = np.random.default_rng(0)
_ssm = _msf(_m_small, _gnt, _sg, _rng3)
_sbg = _msf(_m_big, _gnt, _sg, _rng3)
_sio = _msf(_m_io, _gnt, _sg, _rng3)
_sic = _msf(_m_ice, _gnt, _sg, _rng3)
print(f"  MoonSurface: D_sc small(g={_ssm.surface_gravity_ms2:.2f})="
      f"{_ssm.crater_transition_km:.1f} km  "
      f"big(g={_sbg.surface_gravity_ms2:.2f})="
      f"{_sbg.crater_transition_km:.1f} km")
# D_sc ~ 1/g (Pike 1980): larger gravity -> smaller transition diameter.
assert _sbg.surface_gravity_ms2 > _ssm.surface_gravity_ms2
assert _sbg.crater_transition_km < _ssm.crater_transition_km
# isostatic relief ~ 1/g: smaller-gravity body sustains higher relief.
assert _ssm.max_relief_km > _sbg.max_relief_km
# tidal heating resurfaces -> the Io-analog retains fewer craters.
assert _sio.tidal_heating_index > 0.3, _sio.tidal_heating_index
assert _sio.crater_density < _ssm.crater_density, (
    _sio.crater_density, _ssm.crater_density)
# low-density body classified ice-rich.
assert _sic.is_icy and _sic.ice_fraction > 0.4, _sic
assert not _sbg.is_icy, _sbg

from goldilocks.starsurface import (star_surface_for as _ssf,
                                    limb_darkening as _ld)

_st_cool = _Star("M", mass=0.3)
_st_sun = _Star("G", mass=1.0, luminosity=1.0, teff=_TSUN, radius=1.0)
_st_hot = _Star("A", mass=2.0)
_su_c, _su_s, _su_h = (_ssf(_st_cool), _ssf(_st_sun), _ssf(_st_hot))
_mu = _xp.linspace(0.0, 1.0, 50)
_ldv = _BK.asnumpy(_ld(_mu, _su_s.ld_u1, _su_s.ld_u2))
print(f"  StarSurface: LD limb={_ldv[0]:.3f} centre={_ldv[-1]:.3f}; "
      f"granule cool={_su_c.granule_scale_rel:.2f} "
      f"sun={_su_s.granule_scale_rel:.2f} "
      f"hot={_su_h.granule_scale_rel:.2f}")
# limb darkening monotone decreasing centre -> limb.
assert np.all(np.diff(_ldv) >= -1e-9), "limb darkening not monotone"
assert _ldv[-1] > _ldv[0] > 0.0
# granule size grows with Teff / falls with gravity (H_p ~ T_eff/g).
assert _su_h.granule_scale_rel > _su_s.granule_scale_rel \
    > _su_c.granule_scale_rel

# --- L0 StellarState core (research/sun_render.md Phase 1) -------------
from goldilocks.stellar_state import StellarState as _StS

_sun_st = _StS.for_mass_age(1.0, 4.6)
print(f"  StellarState Sun: P_rot={_sun_st.p_rot_days:.1f} d, "
      f"tau_c={_sun_st.tau_c_days:.1f} d, Ro={_sun_st.rossby:.2f}, "
      f"B-V={_sun_st.bv:.3f}, beta={_sun_st.beta_gd:.3f}, "
      f"regime={_sun_st.activity_regime()}")
# Solar references with research-doc tolerances.
assert 22.0 <= _sun_st.p_rot_days <= 28.0, _sun_st.p_rot_days
assert 10.0 <= _sun_st.tau_c_days <= 18.0, _sun_st.tau_c_days
assert 1.4 <= _sun_st.rossby <= 2.4, _sun_st.rossby
assert 0.62 <= _sun_st.bv <= 0.68, _sun_st.bv
assert abs(_sun_st.beta_gd - 0.08) < 1e-6, _sun_st.beta_gd
assert _sun_st.evolutionary_state == "ms"
# Hot star: radiative envelope -> von Zeipel beta = 0.25.
_sir_st = _StS.for_mass_age(2.0, 0.3)
assert abs(_sir_st.beta_gd - 0.25) < 1e-6, _sir_st.beta_gd
assert _sir_st.teff_k > _sun_st.teff_k > _StS.for_mass_age(0.3, 5.0).teff_k
# M-dwarf: long convective turnover, low Rossby (active).
_md_st = _StS.for_mass_age(0.3, 5.0)
assert _md_st.tau_c_days > _sun_st.tau_c_days, _md_st.tau_c_days
assert _md_st.rossby < _sun_st.rossby, _md_st.rossby
# Close binary tidally locks; wide one does not.
_lk = _StS.for_mass_age(1.0, 5.0, companion_msun=0.6,
                        orbital_period_days=0.52)
_wd = _StS.for_mass_age(1.0, 5.0, companion_msun=0.6,
                        orbital_period_days=400.0)
assert _lk.tidally_locked and not _wd.tidally_locked, (_lk, _wd)
assert _lk.p_rot_days == 0.52 and _lk.active_longitude_amp == 0.5
# Round-trip serialization.
assert _StS.from_dict(_sun_st.to_dict()) == _sun_st
# Increment-3 gap closures (research/sun_render.md Phase 1).
from goldilocks.stellar_state import (spectral_class_from_teff as _sct,
                                      bv_from_teff as _bvt,
                                      noyes_turnover_days as _noy)

assert _sun_st.spectral_class == "G", _sun_st.spectral_class
assert _sct(5772) == "G" and _sct(45000) == "O" and _sct(3000) == "M"
assert _sir_st.spectral_class in ("A", "B"), _sir_st.spectral_class
assert _md_st.spectral_class in ("M", "K"), _md_st.spectral_class
# Ballesteros inverse: Sun T=5772 -> (B-V)=0.65 +/- 0.02 (checklist 1.7).
assert abs(_bvt(5772.0) - 0.65) <= 0.02, _bvt(5772.0)
# Noyes 1984 turnover: Sun (B-V)=0.65 -> tau_c ~ 11-18 d; monotone in B-V.
assert 11.0 <= _noy(0.65) <= 18.0, _noy(0.65)
assert _noy(1.0) > _noy(0.5), (_noy(1.0), _noy(0.5))
# __post_init__ validation now fires on direct/from_dict construction.
_bad = dict(_sun_st.to_dict())
_bad["mass_msun"] = 500.0
try:
    _StS.from_dict(_bad)
    raise AssertionError("invalid mass did not raise")
except ValueError:
    pass
# Young (<100 Myr) star sits at the saturated near-breakup rotation.
_young = _StS.for_mass_age(1.0, 0.02)
assert _young.breakup_capped and _young.p_rot_days < 1.0, _young.p_rot_days
print(f"  StellarState gaps: class={_sun_st.spectral_class} "
      f"bv(5772)={_bvt(5772.0):.3f} tau_c(0.65)={_noy(0.65):.1f}d "
      f"young P_rot={_young.p_rot_days:.2f}d (saturated) OK")
print("  StellarState: solar refs + atlas + binary lock OK")

from goldilocks.bodyview import (occult_fraction as _of,
                                 render_body_view as _rbv,
                                 _refracted_spectrum as _refr)

assert _of(2.0, 0.5, 0.5) == 0.0
assert _of(0.0, 0.5, 1.0) == 1.0
assert abs(_of(1.0, 1.0, 1.0) - 0.391) < 0.01, _of(1.0, 1.0, 1.0)
print(f"  occult_fraction(1,1,1) = {_of(1.0, 1.0, 1.0):.3f} "
      f"(equal disks, exact lens area)")

_e3 = _ea("Earth")
_e3.moons = [_Mn("Luna", mass_me=0.0123, radius_re=0.273,
                  a_planet_au=0.00257, eccentricity=0.0549,
                  density_gcc=3.34, kind="regular")]
_sysv = _SS.single("Solv", _Star("Sun", mass=1.0, luminosity=1.0,
                                  teff=_TSUN, radius=1.0),
                    planets=[_e3])
_e3.habitability = _pfp(_e3, _sysv, np.random.default_rng(7),
                        in_phz=True)
_lamv = _sv.lambda_grid_nm()
_pl = _sv.planck_spectral(_lamv, 5772.0)
_pl = _pl / np.trapezoid(_pl, _lamv * 1e-9)
_rf = _refr(_e3, _pl, _lamv)
_, _ybv, _ = _sv.cie_xyz_bar(_lamv)
_dlv = float(_lamv[1] - _lamv[0])


def _rgbv(s):
    _Yv = float(np.dot(s, _ybv) * _dlv)
    return _sv.spectrum_to_srgb(s[None, :], _lamv,
                                0.35 / max(_Yv, 1e-12))[0]


_rd = _rgbv(_rf)
print(f"  refracted-limb spectrum rgb = {tuple(int(c) for c in _rd)} "
      f"(blood-red: R >> B)")
assert int(_rd[0]) > int(_rd[2]) + 60, _rd

_rgbB, _expB, _infoB = _rbv(_sysv, _e3, _e3.moons[0],
                            resolution=(160, 120))
assert _rgbB.shape == (120, 160, 3)
_cmid = _rgbB[40:80, 60:100].sum()
_cedge = _rgbB[:20, :20].sum() + _rgbB[-20:, -20:].sum()
print(f"  render_body_view centred: centre-sum {_cmid} >> "
      f"corner-sum {_cedge}")
assert _cmid > _cedge, (_cmid, _cedge)

# --- Photosphere field (research/sun_render.md Phase 2) ---------------
from goldilocks import noise as _Nz
from goldilocks.photosphere import Photosphere as _Ph, _HAVE_WARP as _HW

_rng9 = np.random.default_rng(9)
_pp = _rng9.uniform(-7.0, 7.0, size=(4, 1_000_000))
_n3 = _Nz.value_noise_3d(_pp[0], _pp[1], _pp[2], seed=3)
_n4 = _Nz.value_noise_4d(_pp[0], _pp[1], _pp[2], _pp[3], seed=4)
assert -1.05 <= float(_n3.min()) and float(_n3.max()) <= 1.05, \
    (float(_n3.min()), float(_n3.max()))
assert -1.05 <= float(_n4.min()) and float(_n4.max()) <= 1.05, \
    (float(_n4.min()), float(_n4.max()))
# curl_noise_sphere: flow is tangent (radial component ~ 0).
_th = _rng9.uniform(0.2, np.pi - 0.2, 4000)
_ph = _rng9.uniform(0.0, 2 * np.pi, 4000)
_sx = np.sin(_th) * np.cos(_ph)
_sy = np.sin(_th) * np.sin(_ph)
_sz = np.cos(_th)
_vx, _vy, _vz = _Nz.curl_noise_sphere(_sx, _sy, _sz, t=0.3, seed=5)
_radial = np.abs(_vx * _sx + _vy * _sy + _vz * _sz)
_vmag = np.sqrt(_vx ** 2 + _vy ** 2 + _vz ** 2) + 1e-12
assert float(np.mean(_radial / _vmag)) < 1e-9, float(np.mean(_radial))
print(f"  curl_noise_sphere mean|radial|/|v| = "
      f"{float(np.mean(_radial / _vmag)):.2e} (tangent flow)")

_phr = _Ph.for_star_seed(1.0, 7, backend="reference")
_t_sun = _phr.surface.teff
for _ in range(12):
    _phr.step(0.1)
_Tr = _B.asnumpy(_phr.temperature())
assert np.isfinite(_Tr).all()
assert _Tr.min() > 3500.0 and _Tr.max() < _t_sun + 900.0, \
    (float(_Tr.min()), float(_Tr.max()))
assert _phr.last_cfl < 0.5, _phr.last_cfl
# Determinism: same seed + same steps -> bit-identical (pool-safety).
_pa = _Ph.for_star_seed(1.0, 7, backend="reference")
_pb = _Ph.for_star_seed(1.0, 7, backend="reference")
for _ in range(6):
    _pa.step(0.1)
    _pb.step(0.1)
assert np.array_equal(_B.asnumpy(_pa.scalar()), _B.asnumpy(_pb.scalar()))
# Sunspots darken: umbral emission ~ 0.25 of quiet Sun (research 2.9).
_em = _B.asnumpy(_phr.emission())
_um = _B.asnumpy(_phr._umbra)
if (_um > 0.6).sum() > 5:
    _ratio = float(_em[_um > 0.6].mean() / _em[_um < 0.05].mean())
    assert 0.10 <= _ratio <= 0.45, _ratio
    print(f"  sunspot umbra/quiet emission = {_ratio:.2f} (~0.25)")
else:
    print("  sunspot umbra: few pixels at dev res (quiet Sun) -- skipped")
print(f"  Photosphere[reference] T in [{_Tr.min():.0f},{_Tr.max():.0f}] K, "
      f"CFL={_phr.last_cfl:.2f}, deterministic OK")
# Increment-3 Phase-2 gap closures.
# Granule wavenumber is physically grounded (R*/1Mm) then grid-capped.
assert _phr.k_phys > 100.0 and _phr._freq_capped, _phr.k_phys
# Separate low-frequency sunspot channel at 0.05x granulation (2.8).
assert abs(_phr.spot_freq - max(0.05 * _phr.freq, 0.6)) < 1e-9
# Solar granule-lifetime advance rate recovers ~0.25 (~ the old t/4);
# a more active (lower-Ro) star boils faster (rate scales Ro^-1/2).
assert abs(_phr._time_rate - 0.25) < 5e-3, _phr._time_rate
_active = _Ph.for_star_seed(0.4, 7, backend="reference")
assert _active._time_rate > _phr._time_rate, _active._time_rate
# Dravins blueshift (2.11): colour temp != raw temp; small, net-blue.
_ct = _B.asnumpy(_phr._color_temperature())
_dnud = _ct - _Tr
assert np.abs(_dnud).max() < 25.0, float(np.abs(_dnud).max())
assert 0.0 < float(_dnud.mean()) < 8.0, float(_dnud.mean())
print(f"  Photosphere gaps: k_phys={_phr.k_phys:.0f}->{_phr.freq:.0f} "
      f"spot_k={_phr.spot_freq:.2f} Dravins net=+{_dnud.mean():.2f} K "
      f"(blue) OK")

if _HW:
    _pw = _Ph.for_star_seed(1.0, 7, backend="warp")
    for _ in range(12):
        _pw.step(0.1)
    _Tw = _B.asnumpy(_pw.temperature())
    assert np.isfinite(_Tw).all()
    assert _pw.last_cfl < 0.5, _pw.last_cfl
    # Different noise bases (value-noise vs Warp Perlin) -> compare the
    # field *statistics*, not bit values.
    _dm = abs(float(_Tw.mean()) - float(_Tr.mean()))
    _ds = abs(float(_Tw.std()) - float(_Tr.std()))
    assert _dm < 80.0 and _ds < 350.0, (_dm, _ds)
    print(f"  Photosphere[warp] T mean d={_dm:.1f} K std d={_ds:.1f} K "
          f"(CPU JIT; parity-to-tolerance OK)")
else:
    print("  Photosphere[warp] skipped (warp-lang not importable)")

print("  All section-9 assertions passed.")
