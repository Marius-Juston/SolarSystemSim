"""
starsurface.py
--------------
High-detail star-disk renderer for the body-centric view.

`StarSurface` is a derived profile (no new `Star` fields) grounding the
look in real stellar-surface physics:

* surface gravity / log g                  : g = G M / R^2 (textbook)
* granule angular scale                    : convective cell size ~ the
                                             photospheric pressure scale
                                             height H_p = kT/(mu m g)
                                             ~ T_eff / g (Nordlund &
                                             Stein; Trampedach et al.
                                             2013; the Stagger-grid)
* limb darkening                           : quadratic law
                                             I(mu)/I(1) =
                                             1 - u1(1-mu) - u2(1-mu)^2
                                             (Claret 2000; Sing 2010),
                                             T_eff-interpolated
* spots / faculae                          : activity-rotation relation
                                             (Noyes et al. 1984;
                                             Wright et al. 2011) -- fast
                                             rotators / cool stars are
                                             more active; spots are
                                             ~1500 K-cooler black bodies
* prominences / flares with trails         : magnetic-loop eruptions and
                                             a flare frequency-energy
                                             power law; the multi-
                                             temperature solar structure
                                             of NASA SVS 11418
                                             ("Solar Continuum")

Granulation swirls because the brightness field is advected one step
along a divergence-free `noise.curl_noise_2d` flow (Bridson et al.
2007).  Every time dependence is a pure function of the phase argument
`t` -- there is no inter-frame buffer -- so animation frames rendered in
independent `parallel` pool workers stay perfectly consistent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from goldilocks import backend as B
from goldilocks import noise as N
from goldilocks.stellar import G_SI, M_SUN_KG, R_SUN_M, T_EFF_SUN_K
from goldilocks.stellar_state import StellarState

xp = B.xp

# Quadratic limb-darkening anchor points (u1, u2) vs T_eff
# (representative Claret/Sing visual-band values; cool stars darken
# most, hot stars least).  Linearly interpolated in T_eff.
_LD_TEFF = np.array([3000.0, 4000.0, 5772.0, 7000.0, 9500.0, 20000.0])
_LD_U1 = np.array([0.92, 0.78, 0.40, 0.32, 0.26, 0.20])
_LD_U2 = np.array([-0.22, -0.05, 0.26, 0.30, 0.32, 0.34])


@dataclass
class StarSurface:
    teff: float
    log_g_cgs: float
    granule_scale_rel: float  # angular cell size relative to the Sun
    ld_u1: float
    ld_u2: float
    spot_coverage: float  # 0..1
    spot_dt_k: float  # spot temperature deficit (negative)
    flare_rate: float  # 0..1 activity proxy
    prominence_index: float  # 0..1
    # L0-derived geometry / activity (research §3, §4.1).  All defaulted
    # so the dataclass stays backward-compatible with legacy callers.
    rossby: float = 2.21  # solar default
    p_rot_days: float = 25.0
    beta_gd: float = 0.08  # gravity-darkening exponent
    spot_lat_deg: float = 7.0  # butterfly band centre (|latitude|)
    evershed_kms: float = 4.0
    active_long_amp: float = 0.0  # binary active-longitude contrast
    phi_sub: float = 0.0  # sub-stellar longitude (binary)
    # Rotational geometry (research/sun_render.md Phase 3).
    mass_msun: float = 1.0
    radius_rsun: float = 1.0
    omega_ratio: float = 0.0  # Omega / Omega_crit (Roche)
    oblateness: float = 0.0  # r_eq/r_pole - 1
    # Chromosphere (research/sun_render.md Phase 4).
    chromo_thickness_rel: float = 0.003  # shell thickness / R*
    chromo_activity: float = 0.2  # Rossby-driven activity proxy [0,1]
    summary: str = ""


def star_surface_for(star, rotation_period_h: Optional[float] = None,
                     magnetic_rel: Optional[float] = None,
                     *, age_gyr: float = 4.6,
                     companion_msun: Optional[float] = None,
                     orbital_period_days: Optional[float] = None,
                     ) -> StarSurface:
    """Derive a `StarSurface` from a `Star` (+ optional activity hints).

    Activity is now grounded in the L0 `StellarState` Rossby number
    (research §3/§4.1) rather than an ad-hoc heuristic.  The spot
    coverage follows the saturated Rossby activity law
    (Pizzolato+ 2003 / Wright+ 2011): below the saturation Rossby
    number `Ro_sat ~ 0.13` activity is flat at maximum; above it,
    activity ~ (Ro/Ro_sat)^-2.

    `rotation_period_h` / `magnetic_rel` come from the host planet's
    `HabitabilityProfile` when available; the former *overrides* the
    Skumanich rotation period, the latter multiplicatively sharpens the
    activity level.  Both remain optional so legacy callers are
    unchanged.  `companion_msun`/`orbital_period_days` enable the binary
    tidal-locking + active-longitude path.
    """
    teff = float(star.teff or T_EFF_SUN_K)
    M = float(star.mass or 1.0) * M_SUN_KG
    R = float(star.radius or 1.0) * R_SUN_M
    g = G_SI * M / max(R, 1.0) ** 2
    log_g = math.log10(max(g, 1e-3) * 100.0)  # cgs

    # Granule size ~ pressure scale height H_p ~ T_eff / g, normalised
    # to the Sun (=1): hotter / lower-gravity stars have coarser cells.
    g_sun = G_SI * M_SUN_KG / R_SUN_M ** 2
    granule_scale_rel = float(np.clip(
        (teff / T_EFF_SUN_K) * (g_sun / max(g, 1e-3)), 0.15, 12.0))

    u1 = float(np.interp(teff, _LD_TEFF, _LD_U1))
    u2 = float(np.interp(teff, _LD_TEFF, _LD_U2))

    # --- L0 state: real Rossby-number activity (research §3/§4.1) ---
    # This is an MS-only surface renderer; a post-MS age for a massive
    # catalogue star is expected here, so silence that advisory (it
    # still fires for direct StellarState physics queries).
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        state = StellarState.from_star(
            star, age_gyr=age_gyr, companion_msun=companion_msun,
            orbital_period_days=orbital_period_days)

    ro = state.rossby
    # Optional habitability-driven rotation overrides Skumanich.
    if rotation_period_h and math.isfinite(rotation_period_h) \
            and rotation_period_h > 0:
        p_rot_days = float(rotation_period_h) / 24.0
        ro = p_rot_days / max(state.tau_c_days, 1e-6)
    else:
        p_rot_days = state.p_rot_days

    # Saturated Rossby activity law (Pizzolato+ 2003 / Wright+ 2011):
    #   Ro <= Ro_sat : A = 1            (saturated dynamo)
    #   Ro >  Ro_sat : A = (Ro/Ro_sat)^-2
    ro_sat = 0.13
    activity = 1.0 if ro <= ro_sat else (ro / ro_sat) ** -2.0
    # Magnetic hint multiplicatively sharpens (kept from legacy API).
    mag = float(np.clip(magnetic_rel if magnetic_rel else 1.0, 0.1, 4.0))
    activity = float(np.clip(activity * (0.6 + 0.4 * mag), 0.0, 1.0))

    spot_coverage = float(np.clip(0.02 + 0.40 * activity, 0.0, 0.45))
    flare_rate = activity
    prominence_index = float(np.clip(0.25 + 0.75 * activity, 0.0, 1.0))
    # Butterfly band: quiet stars spot near the equator, active ones at
    # higher latitudes (Maunder; research checklist 2.8/5.1).
    spot_lat_deg = float(np.clip(4.0 + 26.0 * activity, 2.0, 35.0))

    s = StarSurface(
        teff=teff, log_g_cgs=log_g,
        granule_scale_rel=granule_scale_rel, ld_u1=u1, ld_u2=u2,
        spot_coverage=spot_coverage, spot_dt_k=-1500.0,
        flare_rate=flare_rate, prominence_index=prominence_index,
        rossby=float(ro), p_rot_days=float(p_rot_days),
        beta_gd=float(state.beta_gd), spot_lat_deg=spot_lat_deg,
        evershed_kms=4.0,
        active_long_amp=float(state.active_longitude_amp),
        phi_sub=float(state.phi_sub_rad),
        mass_msun=float(state.mass_msun),
        radius_rsun=float(state.radius_rsun),
        omega_ratio=float(state.omega_ratio),
        oblateness=float(state.oblateness),
        chromo_thickness_rel=float(state.chromosphere_thickness_rel),
        chromo_activity=float(state.chromo_activity))
    locked = " locked" if state.tidally_locked else ""
    s.summary = (f"{star.name}: Teff {teff:.0f} K, log g {log_g:.2f}, "
                 f"granule x{granule_scale_rel:.2f} Sun, "
                 f"LD (u1={u1:.2f}, u2={u2:.2f}), "
                 f"Ro {ro:.2f} ({state.activity_regime()}{locked}), "
                 f"spots {spot_coverage:.0%} @|lat|~{spot_lat_deg:.0f}deg, "
                 f"beta {state.beta_gd:.2f}, "
                 f"prominence {prominence_index:.2f}")
    return s


def limb_darkening(mu, u1: float, u2: float):
    """Quadratic law I(mu)/I(1) = 1 - u1(1-mu) - u2(1-mu)^2."""
    one_m = 1.0 - mu
    return xp.clip(1.0 - u1 * one_m - u2 * one_m ** 2, 0.0, 1.5)


# Claret (2000, A&A 363 1081) 4-parameter coefficients, coarse Teff
# grid (a1..a4); I(mu)/I(1) = 1 - sum a_k (1 - mu^(k/2)).
_CL_TEFF = np.array([3500.0, 5000.0, 5772.0, 7000.0, 9500.0])
_CL_A = np.array([
    [0.65, -0.55, 1.05, -0.42],
    [0.62, -0.40, 0.88, -0.36],
    [0.58, -0.28, 0.74, -0.31],
    [0.52, -0.20, 0.60, -0.26],
    [0.44, -0.12, 0.45, -0.20],
])


def limb_darkening_law(mu, surface: "StarSurface", *, law: str = "quadratic"):
    """Limb-darkening I(mu)/I(1) by law (research/checklist 3.4).

    ``quadratic`` -> the pinned `limb_darkening`; ``eddington`` ->
    Eddington grey 0.4 + 0.6 mu; ``claret4`` -> Claret (2000) 4-param.
    """
    if law == "quadratic":
        return limb_darkening(mu, surface.ld_u1, surface.ld_u2)
    if law == "eddington":
        return xp.clip(0.4 + 0.6 * mu, 0.0, 1.5)
    if law == "claret4":
        a = [float(np.interp(surface.teff, _CL_TEFF, _CL_A[:, k]))
             for k in range(4)]
        sm = mu * 0.0
        for k in range(4):
            sm = sm + a[k] * (1.0 - mu ** (0.5 * (k + 1)))
        return xp.clip(1.0 - sm, 0.0, 1.5)
    raise ValueError(f"unknown limb-darkening law {law!r}")


def ld_flux_factor(surface: "StarSurface", law: str = "quadratic") -> float:
    """Disk-integrated flux factor 2 ∫_0^1 I(mu) mu d mu.

    For a bare star L = 4 pi R^2 sigma Teff^4 corresponds to this factor
    being ~1 (the law only redistributes, not creates, flux); used by
    the section-9 flux-conservation check."""
    mu = np.linspace(0.0, 1.0, 4096)
    inten = np.asarray(B.asnumpy(limb_darkening_law(B.asarray(mu),
                                                    surface, law=law)))
    raw = 2.0 * float(np.trapezoid(inten * mu, mu))
    # normalise by the same integral of a uniform unit disk (=1) so the
    # *shape* is what is tested; a well-formed law keeps total flux ~1.
    return raw


# ---------------------------------------------------------------------
# Roche geometry + Espinosa-Lara & Rieutord (2011) gravity darkening
# ---------------------------------------------------------------------
def _roche_x(theta: float, f: float) -> float:
    """Dimensionless Roche radius r/R_pole at colatitude ``theta``.

    Solves the equipotential ``1 = 1/x + f x^2 sin^2(theta)`` (Maeder
    1999); x=1 at the pole, ->1.5 at the equatorial break-up limit.
    """
    s2 = math.sin(theta) ** 2
    if f * s2 < 1.0e-12:
        return 1.0
    lo, hi = 1.0, 1.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if 1.0 / mid + f * mid * mid * s2 - 1.0 > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _elr_phi(theta: float, w2: float, x: float) -> float:
    """Solve the ELR2011 transcendental for the auxiliary angle.

    cos(phi) + ln tan(phi/2) = (1/3) w2 x^3 cos^3(theta)
                               + cos(theta) + ln tan(theta/2)
    (Espinosa-Lara & Rieutord 2011, A&A 533 A43).  The LHS is strictly
    increasing in phi on (0, pi), so a bisection is robust.
    """
    rhs = ((1.0 / 3.0) * w2 * x ** 3 * math.cos(theta) ** 3
           + math.cos(theta) + math.log(math.tan(0.5 * theta)))
    lo, hi = 1.0e-6, math.pi - 1.0e-6
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lhs = math.cos(mid) + math.log(math.tan(0.5 * mid))
        if lhs < rhs:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@lru_cache(maxsize=64)
def _elr_table(omega_ratio: float, n: int = 256):
    """Relative T_eff^4(colatitude) from the ELR2011 model.

    Returns (theta_grid, t4_rel) over one hemisphere.  ``w2`` =
    Omega^2 R_p^3 / (GM) = 2 f.  Reduces to a uniform sphere (t4_rel
    constant) as omega_ratio -> 0, so a slow rotator is unchanged.
    """
    # omega_ratio = Omega/Omega_crit, Omega_crit^2 = 8GM/27R^3
    # -> w2 = Omega^2 R^3 / GM = (8/27) omega_ratio^2
    w2 = (8.0 / 27.0) * float(omega_ratio) ** 2
    th = np.linspace(1.0e-3, math.pi / 2.0 - 1.0e-3, n)
    t4 = np.empty(n)
    for i, theta in enumerate(th):
        x = _roche_x(theta, 0.5 * w2)  # f = w2/2
        s, c = math.sin(theta), math.cos(theta)
        # effective gravity in units of GM/R_p^2 (Roche potential grad)
        g_r = -1.0 / (x * x) + w2 * x * s * s
        g_t = w2 * x * s * c
        g_eff = math.hypot(g_r, g_t)
        if w2 < 1.0e-9:
            fw = 1.0
        else:
            phi = _elr_phi(theta, w2, x)
            fw = (math.tan(phi) / math.tan(theta)) ** 2
        t4[i] = g_eff * fw
    return th, t4


def gravity_darkening_factor(blat, surface: "StarSurface", star=None, *,
                             model: str = "elr"):
    """Per-pixel bolometric multiplier from gravity darkening.

    ``model='elr'`` (default) uses Espinosa-Lara & Rieutord 2011 (valid
    at any rotation rate); ``'vonzeipel'`` the classic T ~ g_eff^beta
    (kept for the over-estimate comparison test).  The convective vs
    radiative envelope is folded in through ``surface.beta_gd`` (Lucy
    0.08 -> von Zeipel 0.25): the pole-equator contrast is scaled by
    beta_gd/0.25, so a convective star is only weakly gravity-darkened.
    A slow rotator returns ~1 everywhere (Sun: <1e-4).
    """
    m_msun = (float(getattr(star, "mass", None) or 0.0)
              or float(surface.mass_msun))
    r_rsun = (float(getattr(star, "radius", None) or 0.0)
              or float(surface.radius_rsun))
    M_kg = m_msun * M_SUN_KG
    R_m = r_rsun * R_SUN_M
    g_grav = G_SI * M_kg / max(R_m, 1.0) ** 2
    omega = 2.0 * math.pi / (max(float(surface.p_rot_days), 1e-3) * 86400.0)
    omega_crit = math.sqrt(8.0 * G_SI * M_kg / (27.0 * R_m ** 3))
    wr = min(max(omega / omega_crit, 0.0), 1.0)
    theta = (math.pi / 2.0) - blat  # colatitude
    s_attn = float(np.clip(surface.beta_gd / 0.25, 0.0, 1.0))

    if wr < 1.0e-4:
        return xp.clip(1.0 + 0.0 * blat, 0.2, 3.0)

    if model == "vonzeipel":
        w2r = omega * omega * R_m
        sin_t = xp.sin(theta)
        cos_t = xp.cos(theta)
        g_eff = xp.sqrt((g_grav - w2r * sin_t ** 2) ** 2
                        + (w2r * sin_t * cos_t) ** 2)
        _th = np.linspace(1e-3, math.pi - 1e-3, 256)
        _ge = np.sqrt((g_grav - w2r * np.sin(_th) ** 2) ** 2
                      + (w2r * np.sin(_th) * np.cos(_th)) ** 2)
        g_bar = float(np.trapezoid(_ge * np.sin(_th), _th)
                      / np.trapezoid(np.sin(_th), _th))
        m = (g_eff / max(g_bar, 1e-9)) ** (4.0 * float(surface.beta_gd))
        return xp.clip(m, 0.2, 3.0)

    # ELR2011 relative T^4, normalised to its area-weighted mean so the
    # disk-integrated flux is preserved (only redistributed).
    th, t4 = _elr_table(round(wr, 4))
    th_full = np.concatenate([th, math.pi - th[::-1]])
    t4_full = np.concatenate([t4, t4[::-1]])
    norm = float(np.trapezoid(t4_full * np.sin(th_full), th_full)
                 / np.trapezoid(np.sin(th_full), th_full))
    t4n = t4_full / max(norm, 1e-30)
    th_arr = B.asarray(th_full)
    t4_arr = B.asarray(t4n)
    th_pix = xp.clip(theta, float(th_full[0]), float(th_full[-1]))
    m = xp.interp(th_pix, th_arr, t4_arr)
    # convective attenuation: blend toward isotropic by beta_gd/0.25
    m = 1.0 + (m - 1.0) * s_attn
    return xp.clip(m, 0.2, 3.0)


def _planck(lam_nm, teff: float):
    from goldilocks.skyview import planck_spectral
    pl = planck_spectral(np.asarray(lam_nm, float), float(teff))
    return pl / np.trapezoid(pl, np.asarray(lam_nm, float) * 1e-9)


# ---------------------------------------------------------------------
# Disk renderer (self-luminous; composites into the bodyview `spec`)
# ---------------------------------------------------------------------
def render_star_disk(spec, surface: StarSurface, star, lam_nm,
                     *, center_px: Tuple[float, float],
                     radius_px: float, W: int, H: int,
                     cam_right, cam_up, cam_fwd,
                     spin_axis_local: np.ndarray,
                     rot_phase: float, t_phase: float = 0.0,
                     flux_scale: float = 1.0,
                     seed: int = 7) -> None:
    """Render `star` as a limb-darkened, granulated, spotted disk with a
    chromospheric limb ring and time-evolving prominence/flare trails.

    `spec` is the (H*W, n_lambda) bodyview radiance buffer.  `t_phase`
    drives granulation advection and eruption trails and must be a pure
    function of time (no feedback buffer) so pool-rendered frames stay
    consistent.
    """
    cc, rr = center_px
    rad = max(float(radius_px), 2.0)
    halo = rad * 1.35
    i0, i1 = max(int(rr - halo - 1), 0), min(int(rr + halo + 2), H)
    j0, j1 = max(int(cc - halo - 1), 0), min(int(cc + halo + 2), W)
    if i1 <= i0 or j1 <= j0:
        return
    jj, ii = xp.meshgrid(xp.arange(j0, j1), xp.arange(i0, i1))
    nx = (jj - cc) / rad
    ny = -(ii - rr) / rad
    rho2 = nx ** 2 + ny ** 2
    disk = rho2 <= 1.0
    mu = xp.sqrt(xp.clip(1.0 - rho2, 0.0, 1.0))  # cos(view angle)

    nlam = len(lam_nm)
    base = B.asarray(_planck(lam_nm, surface.teff))
    spot = B.asarray(_planck(lam_nm, max(surface.teff
                                         + surface.spot_dt_k, 1200.0)))

    # Body-fixed coordinates for granulation/spots (rotate with phase).
    cr = B.asarray(np.asarray(cam_right, float))
    cu = B.asarray(np.asarray(cam_up, float))
    cf = B.asarray(np.asarray(cam_fwd, float))
    nz = mu
    n = nx[..., None] * cr + ny[..., None] * cu - nz[..., None] * cf
    pole = B.asarray(np.asarray(spin_axis_local, float))
    pole = pole / (xp.linalg.norm(pole) + 1e-12)
    e1 = xp.cross(pole, cf)
    if float(xp.linalg.norm(e1)) < 1e-6:
        e1 = xp.cross(pole, cr)
    e1 = e1 / (xp.linalg.norm(e1) + 1e-12)
    e2 = xp.cross(pole, e1)
    blat = xp.arcsin(xp.clip(xp.sum(n * pole, axis=-1), -1.0, 1.0))
    blon = xp.arctan2(xp.sum(n * e2, axis=-1),
                      xp.sum(n * e1, axis=-1)) + float(rot_phase)

    # Granulation: cells at ~ (1 / granule_scale) per radian, advected
    # one step along a divergence-free curl-noise flow (swirling).
    gk = 26.0 / max(surface.granule_scale_rel, 0.15)
    gx = blon * gk
    gy = blat * gk
    uadv, vadv = N.curl_noise_2d(gx * 0.25, gy * 0.25, seed=seed + 3)
    # Granule turnover scales as Ro^-1/2 (research §4.1): low-Rossby
    # (active) stars boil faster.  Normalised to the solar Ro so the
    # slow-rotating Sun's appearance is unchanged.
    _RO_SUN = 2.21
    ro_fac = (max(float(surface.rossby), 1e-3) / _RO_SUN) ** -0.5
    adv = 0.6 * float(t_phase) * ro_fac
    cells = N.value_noise_2d(gx + adv * uadv, gy + adv * vadv,
                             seed=seed + 1)
    superg = N.value_noise_2d(blon * (gk * 0.12),
                              blat * (gk * 0.12), seed=seed + 2)
    gran = 1.0 + 0.16 * cells + 0.06 * superg  # bright/dark
    # Spots (cool) + faculae (bright network toward the limb).
    spotn = N.fbm(blon * 1.7, blat * 1.7, seed=seed + 5, octaves=3)
    thr = 1.0 - 2.0 * surface.spot_coverage
    spot_mask = xp.clip((spotn - thr) / 0.25, 0.0, 1.0)
    fac = xp.clip((spotn - (thr - 0.18)) / 0.2, 0.0, 1.0) \
          * (1.0 - mu) * 0.5

    # Butterfly latitude band: spots confined near +/- spot_lat_deg
    # (research checklist 2.8/5.1).  Symmetric Gaussian in |latitude|.
    blat_deg = blat * (180.0 / math.pi)
    lat_c = float(surface.spot_lat_deg)
    band = xp.exp(-((xp.abs(blat_deg) - lat_c) ** 2) / (2.0 * 14.0 ** 2))
    spot_mask = spot_mask * band
    # Active-longitude modulation (binary, research §4.1): a no-op when
    # active_long_amp == 0 (single stars).
    amp = float(surface.active_long_amp)
    if amp > 0.0:
        spot_mask = spot_mask * xp.clip(
            1.0 + amp * xp.cos(2.0 * (blon - float(surface.phi_sub))),
            0.0, 2.0)

    ld = limb_darkening(mu, surface.ld_u1, surface.ld_u2)
    bright = (ld * gran * (1.0 + 0.25 * fac))[..., None]
    rad_field = (base[None, None, :] * bright * (1.0 - spot_mask[..., None])
                 + spot[None, None, :] * (ld * gran)[..., None]
                 * spot_mask[..., None])
    rad_field = rad_field * float(flux_scale)

    # Gravity darkening (Espinosa-Lara & Rieutord 2011; research §4.2).
    # Bolometric radiance scales by the ELR T_local^4 / <T^4> factor;
    # ~1 for the slowly rotating Sun so the pinned solar disk is
    # unchanged (to <1e-4), equator-darkened for fast rotators.
    gd = gravity_darkening_factor(blat, surface, star)
    rad_field = rad_field * gd[..., None]

    sel = disk
    ridx = (ii[sel] * W + jj[sel]).astype(xp.int64)
    # photospheric disk radiance ~ surface brightness / pi
    B.scatter_add(spec, ridx, rad_field[sel] / math.pi)

    # Chromospheric / coronal limb ring: thin emissive annulus just
    # outside the photosphere, scale-height falloff.
    r_pix = xp.sqrt(rho2) * rad
    ring = (~disk) & (r_pix < halo)
    if bool(xp.any(ring)):
        dr = (r_pix - rad)
        glow = xp.exp(-xp.clip(dr / (0.10 * rad), 0.0, 30.0))
        chr = B.asarray(_planck(lam_nm, max(0.55 * surface.teff, 2500.0)))
        gidx = (ii[ring] * W + jj[ring]).astype(xp.int64)
        B.scatter_add(spec, gidx,
                      chr[None, :] * (0.05 * float(flux_scale)
                                      * glow[ring])[:, None])

    # Prominence / flare eruptions with fading trails (pure in t_phase).
    n_loops = max(1, int(round(6 * surface.prominence_index)))
    rng = np.random.default_rng(seed + 17)
    foot = rng.uniform(0.0, 2.0 * math.pi, n_loops)
    hgt = rng.uniform(0.18, 0.55, n_loops)
    span = rng.uniform(0.25, 0.7, n_loops)
    launch = rng.uniform(0.0, 1.0, n_loops)
    period = 1.6
    prom = B.asarray(_planck(lam_nm, 7500.0))  # Halpha-hot
    ns = 26
    s_arc = np.linspace(0.0, 1.0, ns)
    for k in range(n_loops):
        ph = ((float(t_phase) / period + launch[k]) % 1.0)
        # rise then fade -> trailing tail along the arc
        env = math.sin(math.pi * ph) ** 0.6
        if env < 0.02:
            continue
        a0 = foot[k]
        a1 = foot[k] + span[k]
        # quadratic Bezier: two limb footpoints + apex above the limb
        p0 = np.array([math.cos(a0), math.sin(a0)])
        p2 = np.array([math.cos(a1), math.sin(a1)])
        mid = 0.5 * (p0 + p2)
        apex = mid / (np.linalg.norm(mid) + 1e-9) * (1.0 + hgt[k])
        arc = ((1 - s_arc)[:, None] ** 2 * p0
               + 2 * (1 - s_arc)[:, None] * s_arc[:, None] * apex
               + s_arc[:, None] ** 2 * p2)
        ax = cc + arc[:, 0] * rad
        ay = rr - arc[:, 1] * rad
        # trail: brightest at the rising front, fading back along s
        tail = np.clip(1.2 * ph - s_arc, 0.0, 1.0)
        wgt = env * (0.35 + 0.65 * np.exp(-(tail / 0.45) ** 2))
        for (axk, ayk, wk) in zip(ax, ay, wgt):
            if wk < 0.02:
                continue
            bi0, bi1 = max(int(ayk - 4), 0), min(int(ayk + 5), H)
            bj0, bj1 = max(int(axk - 4), 0), min(int(axk + 5), W)
            if bi1 <= bi0 or bj1 <= bj0:
                continue
            bj, bi = xp.meshgrid(xp.arange(bj0, bj1),
                                 xp.arange(bi0, bi1))
            blob = xp.exp(-(((bj - axk) ** 2 + (bi - ayk) ** 2)
                            / (2.0 * 1.7 ** 2)))
            fidx = (bi.ravel() * W + bj.ravel()).astype(xp.int64)
            B.scatter_add(
                spec, fidx,
                prom[None, :] * (0.18 * float(flux_scale) * float(wk)
                                 * blob.ravel())[:, None])
