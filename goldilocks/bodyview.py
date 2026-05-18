"""
bodyview.py
-----------
Body-centric view: a fixed, narrow-field camera locked onto any target
(planet / moon / **star**), rendered large and fully detailed, with
siblings + a procedural starfield circling it, plus eclipse / transit
occultation and the physically-correct refracted "blood-moon".

Structurally this is a narrow-FoV sibling of `skyview.render_sky`: it
reuses the observer geometry (`_local_basis`, the oblate observer
placement), `star_sky_list` (illumination spectra + angular sizes),
`sky_bodies` (target/sibling positions, phase, sub-star direction),
`background_starfield`, `_to_screen`, `_overlay`, and the `parallel`
encode path.  The centred body is dispatched to the detailed
`starsurface.render_star_disk` / `moon_surface.render_moon_disk`
shaders; planets use a deliberately simplified banded/cloud shader
(expanded in a later pass).

Occultation
-----------
`occult_fraction(d, r_a, r_b)` is the exact two-disk lens-area overlap.
- *Transit*: an occulter crosses a star as seen from the observer ->
  on-disk illumination dip + a "TRANSIT" banner.
- *Eclipse*: at the **target's** vantage the star is covered by an
  occulter -> penumbra (partial) scales illumination, umbra (full)
  drops it; "PARTIAL / TOTAL ECLIPSE" banner.
- *Refraction*: in a planet's umbra the only light reaching the target
  has skimmed that planet's atmospheric limb -- the host-star spectrum
  attenuated along a grazing Rayleigh path (`atmosphere_for` +
  Chapman grazing air-mass), which removes the blue and floods the
  target deep red: the standard lunar-eclipse model.

Every time dependence is a pure function of `t` so animation frames
rendered in independent pool workers stay consistent.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import numpy as np

from goldilocks import backend as B
from goldilocks.moon_surface import moon_surface_for, render_moon_disk
from goldilocks.skyview import (lambda_grid_nm, planck_spectral,
                                spectrum_to_srgb, cie_xyz_bar,
                                atmosphere_for, star_sky_list, sky_bodies,
                                background_starfield, _local_basis,
                                _to_screen, _planet_pos_msun, _overlay,
                                N_LAMBDA_DEFAULT,
                                H_PLANCK, C_LIGHT, K_BOLTZ)
from goldilocks.starsurface import star_surface_for, render_star_disk
from goldilocks.stellar import AU_M, R_SUN_M

xp = B.xp


# ---------------------------------------------------------------------
# Two-disk angular occultation
# ---------------------------------------------------------------------
def occult_fraction(d: float, r_a: float, r_b: float) -> float:
    """Fraction of disk A (radius r_a) covered by disk B (radius r_b),
    centres separated by angle `d`.  Exact circle-circle lens area."""
    r_a = max(r_a, 1e-12)
    if d >= r_a + r_b:
        return 0.0
    if d <= abs(r_a - r_b):
        # one fully inside the other
        return min(1.0, (r_b / r_a) ** 2) if r_b < r_a else 1.0
    d2, ra2, rb2 = d * d, r_a * r_a, r_b * r_b
    ca = math.acos(np.clip((d2 + ra2 - rb2) / (2 * d * r_a), -1, 1))
    cb = math.acos(np.clip((d2 + rb2 - ra2) / (2 * d * r_b), -1, 1))
    area = (ra2 * (ca - math.sin(2 * ca) / 2.0)
            + rb2 * (cb - math.sin(2 * cb) / 2.0))
    return float(np.clip(area / (math.pi * ra2), 0.0, 1.0))


# ---------------------------------------------------------------------
# Refracted (Rayleigh-stripped) spectrum through a planet's limb
# ---------------------------------------------------------------------
def _refracted_spectrum(planet, star_spec: np.ndarray,
                        lam_nm: np.ndarray) -> np.ndarray:
    """Host-star spectrum filtered along a grazing path through
    `planet`'s atmospheric limb.

    Reuses `skyview.atmosphere_for` for the wavelength-resolved Rayleigh
    coefficient and scale height; the grazing slant column is the
    Chapman tangent air-mass ~ sqrt(2 pi r_p / H), the analytic form of
    the `_light_optical_depth` integral for a limb-tangent ray (no
    per-pixel ray march).  Blue is extinguished -> deep red.
    """
    atmo = atmosphere_for(planet, lam_nm)
    airmass = math.sqrt(max(2.0 * math.pi * atmo.r_planet_m
                            / max(atmo.h_rayleigh_m, 1.0), 1.0))
    tau = (np.asarray(atmo.beta_r, float) * atmo.h_rayleigh_m
           + 1.1 * atmo.beta_m * atmo.h_mie_m) * airmass
    # The eclipse *colour* is set by the *differential* Rayleigh
    # extinction across the visible refracted ring (the absolute grey
    # opacity only fixes the ring altitude).  Work with tau - min(tau)
    # so the least-attenuated (reddest) wavelength is the ring floor and
    # the lambda^-4 slope reddens the rest -- the standard lunar-eclipse
    # treatment.  For a sane column (Earth ~1 bar) this already gives a
    # deep red; only compress the dynamic range if an absurd column
    # (a 10^4 bar giant) would otherwise push the blue end to ~1e6 and
    # leave a single renormalised monochrome sliver.
    d = tau - float(np.min(tau))
    mx = float(np.max(d))
    MAXD = 8.0
    if mx > MAXD:
        d = d * (MAXD / mx)
    trans = np.exp(-np.clip(d, 0.0, 60.0))
    out = np.asarray(star_spec, float) * trans
    s = float(np.trapezoid(out, np.asarray(lam_nm) * 1e-9))
    s0 = float(np.trapezoid(np.asarray(star_spec, float),
                            np.asarray(lam_nm) * 1e-9))
    if s > 0.0:
        out = out / s * s0
    return out


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------
def _find_parent_planet(sys, moon):
    for p in sys.planets:
        for mn in getattr(p, "moons", []):
            if mn is moon or mn.name == getattr(moon, "name", None):
                return p
    return None


def _norm(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-15)


def _planet_world_pos(sys, observer_planet, orbit_phase, obs_pos, sb,
                      planet):
    """World position of `planet` consistent with how `sky_bodies`
    placed the rendered bodies: the observer sits at `orbit_phase`,
    every other planet is taken from the same `sky_bodies` result
    (`obs_pos + dir*dist`) so occulter geometry matches what is drawn.
    """
    if planet is observer_planet:
        p, _ = _planet_pos_msun(sys, planet, orbit_phase)
        return p
    b = sb.get(getattr(planet, "name", None))
    if b is not None and b.kind == "planet":
        return np.asarray(obs_pos, float) + b.dir_inertial * b.dist_au
    p, _ = _planet_pos_msun(sys, planet, 0.0)
    return p


def eclipse_coverage(sys, observer_planet, target, *,
                     t_orbit: float = 0.0, orbit_phase: float = 0.0
                     ) -> Tuple[float, object]:
    """Geometry-only max fraction of any star occulted at the target by
    another planet, plus the refracting planet if it is a full umbra
    (a planet with an atmosphere).  No rendering -- used to scan an
    orbit for the deepest eclipse cheaply.
    """
    sys._update_stellar_positions(t_orbit)
    lam = lambda_grid_nm()
    sb = {b.name: b for b in sky_bodies(sys, observer_planet, lam,
                                        t_orbit, orbit_phase)}
    b = sb.get(getattr(target, "name", None))
    if b is None:
        return 0.0, None
    obs_pos, _ = _planet_pos_msun(sys, observer_planet, orbit_phase)
    tgt_pos = obs_pos + b.dir_inertial * b.dist_au
    best, refr = 0.0, None
    for i, s in enumerate(sys.stars):
        sp_ = np.array(s.position, float)
        to_s = sp_ - tgt_pos
        d_s = float(np.linalg.norm(to_s)) + 1e-12
        u_s = to_s / d_s
        r_s = math.asin(min(float(s.radius or 1.0) * R_SUN_M / AU_M
                            / d_s, 0.999))
        for q in sys.planets:
            if q is target:
                continue
            qpos = _planet_world_pos(sys, observer_planet, orbit_phase,
                                     obs_pos, sb, q)
            to_q = qpos - tgt_pos
            d_q = float(np.linalg.norm(to_q)) + 1e-12
            if d_q >= d_s:
                continue
            sep = math.acos(np.clip(np.dot(_norm(to_q), u_s),
                                    -1.0, 1.0))
            r_q = math.asin(min(float(q.radius_re) * 6.371e6 / AU_M
                                / d_q, 0.999))
            c = occult_fraction(sep, r_s, r_q)
            if c > best:
                best = c
                refr = q if (c > 0.98 and q.habitability is not None) \
                    else None
    return best, refr


def transit_state(sys, observer_planet, target, *,
                  t_orbit: float = 0.0, orbit_phase: float = 0.0
                  ) -> float:
    """Geometry-only: fraction of a star's disk the target covers as
    seen from the observer (a transit light-curve sample).  Used to
    scan an orbit for the deepest transit cheaply."""
    sys._update_stellar_positions(t_orbit)
    lam = lambda_grid_nm()
    sb = {b.name: b for b in sky_bodies(sys, observer_planet, lam,
                                        t_orbit, orbit_phase)}
    b = sb.get(getattr(target, "name", None))
    if b is None:
        return 0.0
    obs_pos, _ = _planet_pos_msun(sys, observer_planet, orbit_phase)
    tdir = b.dir_inertial
    best = 0.0
    for s in sys.stars:
        to_s = np.array(s.position, float) - obs_pos
        d_s = float(np.linalg.norm(to_s)) + 1e-12
        if d_s <= b.dist_au:
            continue  # star nearer than the body -> no transit
        r_s = math.asin(min(float(s.radius or 1.0) * R_SUN_M / AU_M
                            / d_s, 0.999))
        sep = math.acos(np.clip(np.dot(_norm(to_s), tdir), -1.0, 1.0))
        best = max(best, occult_fraction(sep, r_s, b.ang_radius_rad))
    return best


# ---------------------------------------------------------------------
# Core render
# ---------------------------------------------------------------------
def render_body_view(sys, observer_planet, target, *,
                     latitude_deg: float = 0.0, rot_phase: float = 0.0,
                     t_orbit: float = 0.0, orbit_phase: float = 0.0,
                     resolution: Tuple[int, int] = (960, 720),
                     target_fill: float = 0.55,
                     n_lambda: int = N_LAMBDA_DEFAULT,
                     exposure: Optional[float] = None,
                     bg_seed: int = 2026, show_siblings: bool = True
                     ) -> Tuple[np.ndarray, float, Dict]:
    """Render one body-centric frame.

    Returns (rgb HxWx3 uint8, exposure_used, info) where `info` carries
    the target name/altitude and any TRANSIT / ECLIPSE state for the
    overlay.
    """
    W, H = resolution
    lam = lambda_grid_nm(n_lambda)
    lam_m = np.asarray(lam) * 1e-9
    sys._update_stellar_positions(t_orbit)
    stars = star_sky_list(sys, observer_planet, t_orbit, orbit_phase)
    obs_pos, _ = _planet_pos_msun(sys, observer_planet, orbit_phase)
    prof = observer_planet.habitability
    obl = float(getattr(prof, "obliquity_deg", 23.4)) if prof else 23.4
    east, north, zenith = _local_basis(obl, latitude_deg, rot_phase)
    Rml = np.stack([east, north, zenith])  # system -> local rows

    star_pos = [np.array(s.position, float) for s in sys.stars]
    star_L = [float(s.luminosity) for s in sys.stars]
    star_T = [float(s.teff or 5772.0) for s in sys.stars]
    star_R_au = [float(s.radius or 1.0) * R_SUN_M / AU_M
                 for s in sys.stars]

    # --- locate the target + its geometry (reuse skyview lists) ---
    kind = None
    tgt_pos = None
    spin_axis = np.array([0.0, 0.0, 1.0])
    f_obl = 0.0
    star_idx = next((i for i, s in enumerate(sys.stars)
                     if s is target), None)
    if star_idx is not None:
        kind = "star"
        si = star_idx
        tgt_pos = star_pos[si]
        r_t_au = star_R_au[si]
    else:
        sb = {b.name: b for b in sky_bodies(sys, observer_planet, lam,
                                            t_orbit, orbit_phase)}
        b = sb.get(target.name)
        if b is None:
            raise ValueError(f"target {target.name!r} not visible/known")
        kind = b.kind
        tgt_pos = obs_pos + b.dir_inertial * b.dist_au
        r_t_au = math.tan(b.ang_radius_rad) * b.dist_au
        spin_axis = np.asarray(b.spin_axis, float)
        f_obl = b.f_oblate

    vec = np.asarray(tgt_pos, float) - np.asarray(obs_pos, float)
    d_obs = float(np.linalg.norm(vec)) + 1e-12
    ang_r = math.asin(min(r_t_au / d_obs, 0.999))

    # --- fixed camera locked on the target ---
    fwd = _norm(Rml @ _norm(vec))
    up0 = np.array([0.0, 0.0, 1.0])  # local zenith
    right = _norm(np.cross(fwd, up0))
    if not np.all(np.isfinite(right)) or np.linalg.norm(right) < 1e-6:
        right = _norm(np.cross(fwd, np.array([0.0, 1.0, 0.0])))
    up = _norm(np.cross(right, fwd))
    vfov = float(np.clip(ang_r / max(target_fill, 1e-3), 5.0e-7,
                         math.radians(80.0)))
    hfov = vfov * (W / H)
    px_per_rad = (H - 1) / vfov
    rad_px = ang_r * px_per_rad
    cc, rr = (W - 1) / 2.0, (H - 1) / 2.0

    spec = xp.zeros((H * W, n_lambda))

    # --- procedural background starfield behind the body ---
    d0, bt, bf, _pn = background_starfield(bg_seed)
    dl = (B.asarray(d0) @ B.asarray(Rml.T))
    col, row, infront = _to_screen(dl, B.asarray(right), B.asarray(up),
                                   B.asarray(fwd), hfov, W, H,
                                   vfov / 2.0, -vfov / 2.0)
    ci = xp.round(col).astype(xp.int64)
    ri = xp.round(row).astype(xp.int64)
    vis = (infront & (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H))
    if bool(xp.any(vis)):
        flat = (ri[vis] * W + ci[vis]).astype(xp.int64)
        tv = B.asarray(bt)[vis]
        fv = B.asarray(bf)[vis]
        lam_x = B.asarray(lam_m)
        a_pl = 2.0 * H_PLANCK * C_LIGHT ** 2 / lam_x ** 5
        ex = (H_PLANCK * C_LIGHT
              / (lam_x[None, :] * K_BOLTZ * tv[:, None]))
        pl = a_pl[None, :] / xp.expm1(xp.clip(ex, 1e-6, 700.0))
        pl = pl / B.trapezoid(pl, lam_x, axis=1)[:, None]
        B.scatter_add(spec, flat, (3.0e-5 * fv)[:, None] * pl)

    # --- illumination of the target + eclipse / refraction ---
    info: Dict = {"target": target.name, "kind": kind,
                  "banner": "", "altitude_deg": 0.0}
    D_local = Rml @ _norm(vec)
    info["altitude_deg"] = math.degrees(
        math.asin(float(np.clip(D_local[2], -1.0, 1.0))))
    spin_local = _norm(Rml @ _norm(spin_axis))

    if kind == "star":
        surf = star_surface_for(
            target,
            rotation_period_h=getattr(prof, "rotation_period_h", None),
            magnetic_rel=getattr(prof, "magnetic_moment_rel", None))
        flux = star_L[si] / d_obs ** 2
        render_star_disk(
            spec, surf, target, lam, center_px=(cc, rr),
            radius_px=rad_px, W=W, H=H, cam_right=right, cam_up=up,
            cam_fwd=fwd, spin_axis_local=spin_local,
            rot_phase=rot_phase, t_phase=t_orbit, flux_scale=flux,
            seed=(abs(hash(target.name)) % 90000) + 1)
    else:
        # incident starlight at the target (body -> each star)
        sun_dirs, sun_specs = [], []
        illum_scale, illum_override, ecl = 1.0, None, ""
        for i, sp_ in enumerate(star_pos):
            to_s = sp_ - tgt_pos
            d_s = float(np.linalg.norm(to_s)) + 1e-12
            u_s = to_s / d_s
            pl = planck_spectral(lam, star_T[i])
            pl = pl / np.trapezoid(pl, lam_m)
            base = (star_L[i] / d_s ** 2) * pl
            r_s = math.asin(min(star_R_au[i] / d_s, 0.999))
            # occulters between the target and this star
            cov, refr = 0.0, None
            for q in sys.planets:
                if q is target:
                    continue
                qpos = _planet_world_pos(sys, observer_planet,
                                         orbit_phase, obs_pos, sb, q)
                to_q = qpos - tgt_pos
                d_q = float(np.linalg.norm(to_q)) + 1e-12
                if d_q >= d_s:
                    continue  # behind / farther than the star
                sep = math.acos(np.clip(
                    np.dot(_norm(to_q), u_s), -1.0, 1.0))
                r_q = math.asin(min(
                    float(q.radius_re) * 6.371e6 / AU_M / d_q, 0.999))
                c = occult_fraction(sep, r_s, r_q)
                if c > cov:
                    cov = c
                    if c > 0.98 and q.habitability is not None:
                        refr = q
            if cov > 0.0:
                ecl = "TOTAL ECLIPSE" if cov > 0.98 else "PARTIAL ECLIPSE"
            if refr is not None:
                illum_override = _refracted_spectrum(refr, base, lam)
                base = illum_override * 0.05  # deep, dim red
            else:
                base = base * (1.0 - cov)
            sun_dirs.append(Rml @ u_s)
            sun_specs.append(base)
        info["banner"] = ecl

        if kind == "moon":
            parent = _find_parent_planet(sys, target)
            surf = moon_surface_for(target, parent or observer_planet,
                                    sys, np.random.default_rng(
                    abs(hash(target.name)) % 90000))
            render_moon_disk(
                spec, surf, target, lam, center_px=(cc, rr),
                radius_px=rad_px, W=W, H=H, cam_right=right, cam_up=up,
                cam_fwd=fwd, sun_dirs_local=sun_dirs,
                sun_specs=sun_specs, spin_axis_local=spin_local,
                rot_phase=rot_phase)
        else:  # planet -- simplified banded/cloud shader
            _render_planet_disk(spec, target, lam, (cc, rr), rad_px,
                                W, H, right, up, fwd, spin_local,
                                rot_phase, sun_dirs, sun_specs)

    # --- transit banner (occulter crosses a star from the observer) ---
    for i, sd in enumerate(star_pos):
        to_star = _norm(sd - obs_pos)
        sep = math.acos(np.clip(np.dot(_norm(vec), to_star), -1.0, 1.0))
        r_s_obs = math.asin(min(
            star_R_au[i] / (np.linalg.norm(sd - obs_pos) + 1e-9), 0.999))
        if sep < r_s_obs + ang_r and kind != "star":
            info["banner"] = (info["banner"] + "  " if info["banner"]
                              else "") + "TRANSIT"
            break

    # --- a few sibling dots circling the target ---
    if show_siblings:
        for b in sky_bodies(sys, observer_planet, lam, t_orbit,
                            orbit_phase):
            if b.name == target.name:
                continue
            Dl = Rml @ b.dir_inertial
            c1, r1, infr = _to_screen(
                B.asarray(Dl[None, :]), B.asarray(right), B.asarray(up),
                B.asarray(fwd), hfov, W, H, vfov / 2.0, -vfov / 2.0)
            if not bool(B.asnumpy(infr).ravel()[0]):
                continue
            cx = int(round(float(B.asnumpy(c1).ravel()[0])))
            cy = int(round(float(B.asnumpy(r1).ravel()[0])))
            if 0 <= cx < W and 0 <= cy < H:
                B.scatter_add(spec,
                              xp.asarray([cy * W + cx], dtype=xp.int64),
                              B.asarray(b.refl_spec)[None, :]
                              / max(float(np.sum(b.refl_spec)), 1e-9)
                              * 4e-4)

    spec = B.asnumpy(spec).reshape(H, W, n_lambda)
    _, yb, _ = cie_xyz_bar(lam)
    Y = np.tensordot(spec, yb, axes=([-1], [0])) * float(lam[1] - lam[0])
    e_auto = 0.6 / max(np.percentile(Y, 99.5), 1e-12)
    if exposure is None:
        exposure = e_auto
    rgb = spectrum_to_srgb(spec, lam, exposure)
    return rgb, float(exposure), info


# ---------------------------------------------------------------------
# Simplified planet shader (expanded in a later pass)
# ---------------------------------------------------------------------
def _render_planet_disk(spec, planet, lam_nm, center_px, radius_px,
                        W, H, cam_right, cam_up, cam_fwd,
                        spin_local, rot_phase, sun_dirs, sun_specs):
    from goldilocks import noise as Nz
    cc, rr = center_px
    rad = max(float(radius_px), 1.5)
    i0, i1 = max(int(rr - rad - 1), 0), min(int(rr + rad + 2), H)
    j0, j1 = max(int(cc - rad - 1), 0), min(int(cc + rad + 2), W)
    if i1 <= i0 or j1 <= j0:
        return
    jj, ii = xp.meshgrid(xp.arange(j0, j1), xp.arange(i0, i1))
    nx = (jj - cc) / rad
    ny = -(ii - rr) / rad
    rho2 = nx ** 2 + ny ** 2
    inside = rho2 <= 1.0
    if not bool(xp.any(inside)):
        return
    nz = xp.sqrt(xp.clip(1.0 - rho2, 0.0, 1.0))
    cr = B.asarray(np.asarray(cam_right, float))
    cu = B.asarray(np.asarray(cam_up, float))
    cf = B.asarray(np.asarray(cam_fwd, float))
    n = nx[..., None] * cr + ny[..., None] * cu - nz[..., None] * cf
    n = n / (xp.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)
    pole = B.asarray(_norm(spin_local))
    e1 = xp.cross(pole, cf)
    if float(xp.linalg.norm(e1)) < 1e-6:
        e1 = xp.cross(pole, cr)
    e1 = e1 / (xp.linalg.norm(e1) + 1e-12)
    e2 = xp.cross(pole, e1)
    blat = xp.arcsin(xp.clip(xp.sum(n * pole, axis=-1), -1.0, 1.0))
    blon = xp.arctan2(xp.sum(n * e2, axis=-1),
                      xp.sum(n * e1, axis=-1)) + float(rot_phase)
    prof = getattr(planet, "habitability", None)
    hex_c = getattr(prof, "sky_color_hex", "#9AA7B5") if prof else "#9AA7B5"
    try:
        h = hex_c.lstrip("#")
        rgb0 = np.array([int(h[0:2], 16), int(h[2:4], 16),
                         int(h[4:6], 16)], float) / 255.0
    except Exception:
        rgb0 = np.array([0.6, 0.62, 0.66])
    tint = np.interp(np.asarray(lam_nm, float),
                     [380, 470, 550, 640, 740],
                     [rgb0[2], rgb0[2], rgb0[1], rgb0[0], rgb0[0]]) + 1e-3
    tint = tint / float(np.mean(tint))
    # Zonal bands (giant) or soft cloud mottle (terrestrial).
    from goldilocks.planets import is_gas_giant
    if is_gas_giant(planet):
        bands = 0.5 + 0.5 * xp.sin(blat * 9.0
                                   + 1.5 * Nz.fbm(blon * 1.5, blat * 4.0,
                                                  seed=11, octaves=3))
        albedo = 0.35 + 0.35 * bands
    else:
        cloud = Nz.fbm(blon * 2.2, blat * 2.2, seed=13, octaves=4)
        albedo = 0.18 + 0.30 * xp.clip(cloud, 0.0, 1.0)
        cap = xp.clip((xp.abs(blat) - 1.05) / 0.5, 0.0, 1.0)
        albedo = albedo + 0.5 * cap
    nlam = len(lam_nm)
    acc = xp.zeros(n.shape[:-1] + (nlam,))
    for sd, sp_ in zip(sun_dirs, sun_specs):
        sdl = B.asarray(_norm(sd))
        ndl = xp.clip(xp.sum(n * sdl, axis=-1), 0.0, 1.0)
        acc = acc + ndl[..., None] * B.asarray(np.asarray(sp_, float))[None, None, :]
    acc = acc * (albedo[..., None] / math.pi) * B.asarray(tint)[None, None, :]
    sel = inside
    ridx = (ii[sel] * W + jj[sel]).astype(xp.int64)
    B.scatter_add(spec, ridx, acc[sel])

    # Thin atmospheric limb halo: forward-scattered starlight in the
    # ring just outside the disk, coloured by the planet's atmosphere
    # (Rayleigh tint from `atmosphere_for`); height ~ scale-height.
    try:
        atmo = atmosphere_for(planet, lam_nm)
        hfrac = float(np.clip(atmo.h_rayleigh_m
                              / max(atmo.r_planet_m, 1.0) * 8.0,
                              0.02, 0.22))
    except Exception:
        hfrac = 0.05
    rp = xp.sqrt(rho2)
    ring = (~inside) & (rp < 1.0 + hfrac)
    if bool(xp.any(ring)) and sun_specs:
        glow = xp.exp(-xp.clip((rp - 1.0) / (hfrac + 1e-6), 0.0, 30.0))
        src = B.asarray(np.asarray(sun_specs[0], float))
        gidx = (ii[ring] * W + jj[ring]).astype(xp.int64)
        B.scatter_add(spec, gidx,
                      src[None, :] * B.asarray(tint)[None, :]
                      * (0.5 * glow[ring])[:, None])


# ---------------------------------------------------------------------
# Annotated stills + MP4
# ---------------------------------------------------------------------
def _label(sys, info) -> str:
    base = f"{sys.name} / {info['target']} ({info['kind']})"
    line2 = f"alt {info['altitude_deg']:+.1f} deg"
    if info.get("banner"):
        line2 += f"   **{info['banner']}**"
    return base + "\n" + line2


def _bodyview_frame(job: dict) -> np.ndarray:
    rgb, _, info = render_body_view(
        job["sys"], job["observer"], job["target"],
        rot_phase=job["rot_phase"], t_orbit=job["t_orbit"],
        orbit_phase=job["orbit_phase"], resolution=job["resolution"],
        exposure=job["exposure"], **job["kw"])
    return _overlay(rgb, _label(job["sys"], info))


def render_body_still(sys, observer, target, out_path: str, *,
                      resolution: Tuple[int, int] = (960, 720),
                      rot_phase: float = 0.0, t_orbit: float = 0.0,
                      orbit_phase: float = 0.0, **kw) -> str:
    rgb, _, info = render_body_view(
        sys, observer, target, rot_phase=rot_phase, t_orbit=t_orbit,
        orbit_phase=orbit_phase, resolution=resolution, **kw)
    from PIL import Image
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Image.fromarray(np.ascontiguousarray(
        _overlay(rgb, _label(sys, info)), np.uint8)).save(out_path)
    return out_path


def animate_body_view(sys, observer, target, out_path: str, *,
                      n_frames: int = 120, fps: int = 24,
                      duration_t: float = 1.0,
                      rot_turns: float = 1.0,
                      resolution: Tuple[int, int] = (960, 720),
                      orbit_phase: float = 0.0, **kw) -> None:
    """Render an MP4 sweeping `t_orbit` over `duration_t` (years) and the
    target's spin over `rot_turns`.  Frames are fanned across the pool
    and streamed into one ffmpeg (raises if ffmpeg is absent)."""
    from goldilocks import parallel as P
    ts = np.linspace(0.0, duration_t, n_frames, endpoint=False)
    rots = np.linspace(0.0, 2.0 * math.pi * rot_turns, n_frames,
                       endpoint=False)
    _, exp0, _ = render_body_view(
        sys, observer, target, t_orbit=0.0, orbit_phase=orbit_phase,
        resolution=resolution, **kw)
    jobs = [dict(sys=sys, observer=observer, target=target,
                 rot_phase=float(r), t_orbit=float(t),
                 orbit_phase=orbit_phase, resolution=resolution,
                 exposure=exp0, kw=kw)
            for t, r in zip(ts, rots)]
    frames = P.map_ordered(_bodyview_frame, jobs)
    P.encode_frames(out_path, frames, fps)
