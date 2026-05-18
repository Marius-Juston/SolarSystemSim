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
from typing import Optional, Tuple

import numpy as np

from goldilocks import backend as B
from goldilocks import noise as N
from goldilocks.stellar import G_SI, M_SUN_KG, R_SUN_M, T_EFF_SUN_K

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
    summary: str = ""


def star_surface_for(star, rotation_period_h: Optional[float] = None,
                     magnetic_rel: Optional[float] = None
                     ) -> StarSurface:
    """Derive a `StarSurface` from a `Star` (+ optional activity hints).

    `rotation_period_h` / `magnetic_rel` come from the host planet's
    `HabitabilityProfile` when available; both only sharpen the activity
    level and are not required.
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

    # Activity-rotation: cool + fast-rotating + magnetised => spotty.
    cool = float(np.clip((6000.0 - teff) / 3500.0, 0.0, 1.0))
    rot = 1.0
    if rotation_period_h and math.isfinite(rotation_period_h) \
            and rotation_period_h > 0:
        rot = float(np.clip((600.0 / rotation_period_h) ** 0.5,
                            0.3, 3.0))
    mag = float(np.clip(magnetic_rel if magnetic_rel else 1.0, 0.1, 4.0))
    activity = float(np.clip(0.18 * (0.4 + cool) * rot
                             * (0.6 + 0.4 * mag), 0.0, 1.0))
    spot_coverage = float(np.clip(0.02 + 0.30 * activity, 0.0, 0.45))
    flare_rate = activity
    prominence_index = float(np.clip(0.25 + 0.75 * activity, 0.0, 1.0))

    s = StarSurface(
        teff=teff, log_g_cgs=log_g,
        granule_scale_rel=granule_scale_rel, ld_u1=u1, ld_u2=u2,
        spot_coverage=spot_coverage, spot_dt_k=-1500.0,
        flare_rate=flare_rate, prominence_index=prominence_index)
    s.summary = (f"{star.name}: Teff {teff:.0f} K, log g {log_g:.2f}, "
                 f"granule x{granule_scale_rel:.2f} Sun, "
                 f"LD (u1={u1:.2f}, u2={u2:.2f}), "
                 f"spots {spot_coverage:.0%}, "
                 f"prominence {prominence_index:.2f}")
    return s


def limb_darkening(mu, u1: float, u2: float):
    """Quadratic law I(mu)/I(1) = 1 - u1(1-mu) - u2(1-mu)^2."""
    one_m = 1.0 - mu
    return xp.clip(1.0 - u1 * one_m - u2 * one_m ** 2, 0.0, 1.5)


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
    adv = 0.6 * float(t_phase)
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

    ld = limb_darkening(mu, surface.ld_u1, surface.ld_u2)
    bright = (ld * gran * (1.0 + 0.25 * fac))[..., None]
    rad_field = (base[None, None, :] * bright * (1.0 - spot_mask[..., None])
                 + spot[None, None, :] * (ld * gran)[..., None]
                 * spot_mask[..., None])
    rad_field = rad_field * float(flux_scale)

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
