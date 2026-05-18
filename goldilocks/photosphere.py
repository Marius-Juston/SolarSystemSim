"""
photosphere.py
--------------
Persistent equirectangular photosphere field (research/sun_render.md
§4.1 / checklist Phase 2).

Unlike `starsurface.render_star_disk` -- which paints a *stateless*
one-step granulation snapshot on the visible disk so independent pool
workers stay consistent -- this module maintains a **stateful** scalar
field on an equirectangular (lon x lat) grid and evolves it with
divergence-free curl-noise **semi-Lagrangian advection**.  It therefore
lives behind a standalone driver (`render_photosphere.py`) and is not
wired into the pool disk renderer.

Two interchangeable step backends sit behind one interface:

* ``warp``      -- a fused NVIDIA-Warp `@wp.kernel` using the built-in
                   divergence-free `wp.curlnoise`.  Warp JIT-compiles the
                   *same* kernel to CPU **or** CUDA, so this is the
                   efficient path on the multi-A6000 box and is still
                   verifiable on a CPU-only machine.  The kernel takes a
                   ``device`` so the research-§5.2 multi-GPU latitude-band
                   partition drops in later with no API change.
* ``reference`` -- a pure NumPy/CuPy path on the `goldilocks.backend`
                   seam using `noise.curl_noise_sphere`.  Dependency-free
                   fallback and the correctness oracle.

Selected by ``GOLDILOCKS_PHOTOSPHERE_BACKEND=auto|warp|reference``
(default ``auto``: warp iff importable, else reference), mirroring
`backend.GOLDILOCKS_BACKEND`.

Temperature -> colour uses the existing CIE pipeline in `skyview`
(`planck_spectral` + `spectrum_to_srgb`); no new black-body table.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np

from goldilocks import backend as B
from goldilocks import noise as N
from goldilocks.stellar import T_EFF_SUN_K
from goldilocks.starsurface import StarSurface, star_surface_for, limb_darkening

xp = B.xp

_RES_PRESETS = {
    "dev": (128, 256),     # (H lat, W lon) -- CPU / tests
    "med": (512, 1024),
    "high": (2048, 4096),  # the A6000 box
}

_EPS_POLE = 1e-4
_UMBRA_T = 4100.0          # K  (research checklist 2.9)
_DT_GRANULE = 500.0        # K  photospheric +/- fluctuation amplitude
# Dravins convective blueshift (research §2.11): a documented,
# physically-anchored *stylisation* -- the ~300 m/s net photospheric
# blueshift is rendered as a small colour-temperature nudge (rising
# bright plasma bluer, sinking lanes redder) rather than a true
# line-shift, consistent with the bolometric CIE pipeline in `skyview`.
_DRAVINS_K = 18.0          # K  per-granule colour-temperature modulation
_DRAVINS_WREF = 0.5        # w-proxy scale for the tanh soft-clip
_DRAVINS_NET = 3.0         # K  disk-mean net blueshift (~300 m/s class)

# Optional Warp backend ------------------------------------------------
_WARP_MODE = os.environ.get(
    "GOLDILOCKS_PHOTOSPHERE_BACKEND", "auto").strip().lower()

try:                                            # pragma: no cover - env
    import warp as wp  # type: ignore

    _HAVE_WARP = True
except Exception:                               # pragma: no cover
    wp = None  # type: ignore
    _HAVE_WARP = False

_WARP_READY = False


def _warp_init() -> bool:
    """Idempotently init Warp + compile the fused step kernel."""
    global _WARP_READY
    if not _HAVE_WARP:
        return False
    if _WARP_READY:
        return True
    wp.init()

    @wp.func
    def _bilin(s: wp.array2d(dtype=wp.float32),
               fi: wp.float32, fj: wp.float32,
               Hh: wp.int32, Ww: wp.int32) -> wp.float32:
        # latitude clamped, longitude wrapped
        if fi < 0.0:
            fi = 0.0
        if fi > wp.float32(Hh - 1):
            fi = wp.float32(Hh - 1)
        i0 = wp.int32(wp.floor(fi))
        j0 = wp.int32(wp.floor(fj))
        ti = fi - wp.float32(i0)
        tj = fj - wp.float32(j0)
        i1 = i0 + 1
        if i1 > Hh - 1:
            i1 = Hh - 1
        j0 = ((j0 % Ww) + Ww) % Ww
        j1 = (j0 + 1) % Ww
        a = s[i0, j0] * (1.0 - tj) + s[i0, j1] * tj
        b = s[i1, j0] * (1.0 - tj) + s[i1, j1] * tj
        return a * (1.0 - ti) + b * ti

    @wp.kernel
    def _step_kernel(s_in: wp.array2d(dtype=wp.float32),
                     s_out: wp.array2d(dtype=wp.float32),
                     Hh: wp.int32, Ww: wp.int32,
                     freq: wp.float32, tx: wp.float32,
                     flow_cells: wp.float32, diff: wp.float32,
                     state: wp.uint32):
        i, j = wp.tid()
        lat = (wp.float32(i) + 0.5) / wp.float32(Hh) * wp.pi - wp.pi * 0.5
        lon = (wp.float32(j) + 0.5) / wp.float32(Ww) * 2.0 * wp.pi
        cl = wp.cos(lat)
        sl = wp.sin(lat)
        clo = wp.cos(lon)
        slo = wp.sin(lon)
        p = wp.vec3(cl * clo, cl * slo, sl)
        g = wp.curlnoise(state, p * freq + wp.vec3(tx, tx, tx))
        # tangent projection (remove radial component)
        rad = wp.dot(g, p)
        vt = g - p * rad
        east = wp.vec3(-slo, clo, 0.0)
        north = wp.vec3(-sl * clo, -sl * slo, cl)
        ve = wp.dot(vt, east)
        vn = wp.dot(vt, north)
        # angular -> grid-cell displacement, clamped for CFL < 0.5
        di = -vn * flow_cells
        dj = -ve / wp.max(cl, wp.float32(_EPS_POLE)) * flow_cells
        if di > 0.45:
            di = 0.45
        if di < -0.45:
            di = -0.45
        if dj > 0.45:
            dj = 0.45
        if dj < -0.45:
            dj = -0.45
        val = _bilin(s_in, wp.float32(i) + di, wp.float32(j) + dj, Hh, Ww)
        # 5% neighbour-average numerical diffusion (research §2.6)
        jm = ((j - 1) % Ww + Ww) % Ww
        jp = (j + 1) % Ww
        im = wp.max(i - 1, 0)
        ip = wp.min(i + 1, Hh - 1)
        avg = (s_in[i, jm] + s_in[i, jp]
               + s_in[im, j] + s_in[ip, j]) * 0.25
        s_out[i, j] = (1.0 - diff) * val + diff * avg

    _warp_init._bilin = _bilin            # keep refs alive
    _warp_init._step_kernel = _step_kernel
    _WARP_READY = True
    return True


# ---------------------------------------------------------------------
class Photosphere:
    """Stateful equirectangular photosphere field."""

    def __init__(self, surface: StarSurface, *, res: str = "dev",
                 seed: int = 0, backend: Optional[str] = None,
                 device: str = "auto", log_mem: bool = False) -> None:
        if res not in _RES_PRESETS:
            raise ValueError(f"res must be one of {list(_RES_PRESETS)}")
        self.surface = surface
        self.res = res
        self.H, self.W = _RES_PRESETS[res]
        self.device = device
        self.seed = int(seed)
        self.t = 0.0
        self.last_cfl = 0.0
        # Physical granule wavenumber (research §2.4): cells per stellar
        # radius ~ R*/1Mm.  Via the angular-cell relation the Sun has
        # ~R_sun/1Mm ~ 696 cells, scaled by the surface's angular cell
        # size; this is then clamped to what the grid can resolve and the
        # cap is logged so the physical-vs-rendered count is explicit.
        gs = max(float(surface.granule_scale_rel), 0.15)
        self.k_phys = 696.0 / gs
        self.freq = float(np.clip(self.k_phys, 6.0,
                                  min(self.W / 8.0, 22.0)))
        self._freq_capped = self.k_phys > self.freq + 1e-9
        # Separate low-frequency active-region channel at 0.05x the
        # granulation wavenumber (research §2.8).
        self.spot_freq = max(0.05 * self.freq, 0.6)
        self.period = 256.0      # lattice period -> bounded time (no drift)
        # Ro^-1/2 turnover (research §4.1), solar-normalised.
        self.flow = 0.9 * (max(float(surface.rossby), 1e-3) / 2.21) ** -0.5
        # Granule *lifetime* also scales with Ro^-1/2 (research §2.4): the
        # noise-time advance rate now tracks the flow timescale instead of
        # a hardcoded constant.  Solar-normalised so the Sun (Ro~2.21)
        # recovers ~0.25 -- i.e. ~ the previous bare t/4 -- while more
        # active (low-Ro) stars boil faster.
        self._time_rate = 0.25 * self.flow / 0.9

        if device == "auto":
            self.device = "cuda" if wp.is_cuda_available() else "cpu"

        want = (backend or _WARP_MODE)
        if want == "auto":
            want = "warp" if _warp_init() else "reference"
        elif want == "warp":
            if not _warp_init():
                raise RuntimeError("warp backend requested but warp-lang "
                                   "is not importable")

        self.backend_name = want

        self._lat = (np.arange(self.H) + 0.5) / self.H * math.pi \
            - math.pi / 2.0
        self._lat = np.clip(self._lat, -math.pi / 2 + _EPS_POLE,
                            math.pi / 2 - _EPS_POLE)
        self._lon = (np.arange(self.W) + 0.5) / self.W * 2.0 * math.pi
        self.reset(self.seed)
        if log_mem:
            print(self.memory_report())

    # --- convenience -------------------------------------------------
    @classmethod
    def for_star_seed(cls, mass_msun: float, seed: int = 7, *,
                      res: str = "dev", age_gyr: float = 4.6,
                      backend: Optional[str] = None) -> "Photosphere":
        from goldilocks.stellar import Star
        star = Star(f"M{mass_msun:g}", mass=float(mass_msun))
        surf = star_surface_for(star, age_gyr=age_gyr)
        return cls(surf, res=res, seed=seed, backend=backend)

    # --- state -------------------------------------------------------
    def reset(self, seed: int) -> None:
        """Deterministic init: low-amplitude granule noise on the sphere.

        Both backends start from this identical field so a same-seed run
        is bit-reproducible on the reference path (pool-safety) and the
        two backends remain statistically comparable."""
        self.seed = int(seed)
        self.t = 0.0
        lo = B.asarray(self._lat)[:, None]
        ln = B.asarray(self._lon)[None, :]
        cl = xp.cos(lo)
        nx = cl * xp.cos(ln)
        ny = cl * xp.sin(ln)
        nz = xp.sin(lo) + 0.0 * ln
        s = N.value_noise_3d(nx * self.freq, ny * self.freq,
                             nz * self.freq, seed=self.seed + 11)
        s = s + 0.5 * N.value_noise_3d(nx * self.freq * 2.0,
                                       ny * self.freq * 2.0,
                                       nz * self.freq * 2.0,
                                       seed=self.seed + 12)
        self._s = xp.asarray(s / 1.5, dtype=xp.float64)
        self._scratch = xp.empty_like(self._s)
        # Dravins convective-blueshift state (research §2.11): vertical
        # velocity proxy from the per-step change in the potential.
        self._w = xp.zeros_like(self._s)
        self._last_dt = 0.1
        self._build_spots()
        self._ws_in = None        # warp persistent buffers (lazy)
        self._ws_out = None

    def _sphere_xyz(self):
        lo = B.asarray(self._lat)[:, None]
        ln = B.asarray(self._lon)[None, :]
        cl = xp.cos(lo)
        return cl * xp.cos(ln), cl * xp.sin(ln), xp.sin(lo) + 0.0 * ln

    # --- sunspots (research checklist 2.8 / 2.9) ---------------------
    def _build_spots(self) -> None:
        s = self.surface
        nx, ny, nz = self._sphere_xyz()
        # low-frequency active-region field
        m = N.value_noise_3d(nx * self.spot_freq, ny * self.spot_freq,
                             nz * self.spot_freq, seed=self.seed + 31)
        m = 0.5 * (m + 1.0)                          # -> [0,1]
        # Ro-dependent threshold (research §4.1).
        ro = max(float(s.rossby), 1e-3)
        thr = 0.78 - 0.30 * float(np.clip(2.0 / ro, 0.0, 1.0))
        # butterfly latitude band + binary active longitude
        lat_deg = B.asarray(self._lat * 180.0 / math.pi)[:, None]
        band = xp.exp(-((xp.abs(lat_deg) - float(s.spot_lat_deg)) ** 2)
                      / (2.0 * 13.0 ** 2))
        amp = float(s.active_long_amp)
        if amp > 0.0:
            ln = B.asarray(self._lon)[None, :]
            band = band * xp.clip(
                1.0 + amp * xp.cos(2.0 * (ln - float(s.phi_sub))),
                0.0, 2.0)
        spotness = xp.clip((m - thr) / 0.16, 0.0, 1.0) * band
        # umbra (deep) / penumbra (shallow) emission + temperature.
        umbra = xp.clip((spotness - 0.55) / 0.45, 0.0, 1.0)
        penum = xp.clip(spotness, 0.0, 1.0) - umbra
        striation = 0.85 + 0.15 * N.value_noise_3d(
            nx * 26.0, ny * 26.0, nz * 26.0, seed=self.seed + 33)
        self._emis = xp.clip(
            1.0 - 0.75 * umbra - 0.30 * penum * striation, 0.05, 1.0)
        self._spot_T = (_UMBRA_T * umbra
                        + (self.surface.teff + self.surface.spot_dt_k * 0.0)
                        * (1.0 - umbra))
        self._umbra = umbra

    # --- evolution ---------------------------------------------------
    def step(self, dt: float = 0.1) -> None:
        prev = self._s
        if self.backend_name == "warp":
            self._step_warp(dt)
        else:
            self._step_reference(dt)
        # vertical-velocity proxy: rising bright granules vs sinking lanes
        self._w = (self._s - prev) / max(float(dt), 1e-6)
        self._last_dt = float(dt)
        self.t += float(dt)

    def _tx(self) -> float:
        # bounded, drift-free time coordinate; advance rate ~ Ro^-1/2 so
        # granule lifetime tracks the flow timescale (research §2.4).
        return float((self.t * self._time_rate) % self.period)

    def _step_reference(self, dt: float) -> None:
        nx, ny, nz = self._sphere_xyz()
        vx, vy, vz = N.curl_noise_sphere(nx, ny, nz, t=self._tx(),
                                         seed=self.seed + 5,
                                         eps=1e-2, freq=self.freq * 0.25)
        lo = B.asarray(self._lat)[:, None]
        ln = B.asarray(self._lon)[None, :]
        sl, cl = xp.sin(lo), xp.cos(lo)
        slo, clo = xp.sin(ln), xp.cos(ln)
        east_x, east_y, east_z = -slo + 0.0 * lo, clo + 0.0 * lo, 0.0 * ln
        north_x = -sl * clo
        north_y = -sl * slo
        north_z = cl + 0.0 * ln
        ve = vx * east_x + vy * east_y + vz * east_z
        vn = vx * north_x + vy * north_y + vz * north_z
        di = -vn * self.H / math.pi
        dj = -ve / xp.clip(cl, _EPS_POLE, None) * self.W / (2.0 * math.pi)
        # Normalise the flow to a fixed Courant budget so advection is
        # steady and CFL is bounded < 0.5 by construction (research §2.6);
        # the Ro^-1/2 turnover sets the budget, dt scales it.
        m = float(xp.maximum(xp.abs(di).max(), xp.abs(dj).max()))
        target = min(0.45, 0.35 * self.flow * (float(dt) / 0.1))
        scale = (target / m) if m > 1e-12 else 0.0
        di = di * scale
        dj = dj * scale
        self.last_cfl = target
        ii = xp.arange(self.H)[:, None] + di
        jj = xp.arange(self.W)[None, :] + dj
        val = self._bilinear(self._s, ii, jj)
        # 5% neighbour-average diffusion (research §2.6)
        avg = 0.25 * (xp.roll(self._s, 1, 1) + xp.roll(self._s, -1, 1)
                      + xp.concatenate([self._s[:1], self._s[:-1]], 0)
                      + xp.concatenate([self._s[1:], self._s[-1:]], 0))
        self._s = 0.95 * val + 0.05 * avg

    def _bilinear(self, s, fi, fj):
        fi = xp.clip(fi, 0.0, self.H - 1.0)
        i0 = xp.floor(fi).astype(xp.int64)
        i1 = xp.clip(i0 + 1, 0, self.H - 1)
        j0 = xp.floor(fj).astype(xp.int64)
        ti = fi - i0
        tj = fj - j0
        j0 = j0 % self.W
        j1 = (j0 + 1) % self.W
        a = s[i0, j0] * (1.0 - tj) + s[i0, j1] * tj
        b = s[i1, j0] * (1.0 - tj) + s[i1, j1] * tj
        return a * (1.0 - ti) + b * ti

    def _step_warp(self, dt: float) -> None:
        k = _warp_init._step_kernel
        H, W = self.H, self.W
        if self._ws_in is None:
            host = B.asnumpy(self._s).astype(np.float32)
            self._ws_in = wp.array(host, dtype=wp.float32,
                                   device=self.device)
            self._ws_out = wp.zeros((H, W), dtype=wp.float32,
                                    device=self.device)
        # wp.curlnoise magnitude is O(1); a fixed cell budget keeps the
        # in-kernel clamp a rare safety net (CFL < 0.5 by construction).
        flow_cells = min(0.45, 0.35 * self.flow * (float(dt) / 0.1))
        st = wp.rand_init(self.seed + 5)
        wp.launch(k, dim=(H, W),
                  inputs=[self._ws_in, self._ws_out, H, W,
                          float(self.freq * 0.25), float(self._tx()),
                          float(flow_cells), 0.05, st],
                  device=self.device)
        wp.synchronize_device(self.device)
        self._ws_in, self._ws_out = self._ws_out, self._ws_in
        arr = self._ws_in.numpy()
        self.last_cfl = flow_cells
        self._s = xp.asarray(arr, dtype=xp.float64)

    # --- readout -----------------------------------------------------
    def scalar(self):
        return self._s

    def temperature(self):
        """Local effective temperature map [K] (granules + spots)."""
        s = xp.clip(self._s, -1.5, 1.5)
        t = self.surface.teff + _DT_GRANULE * s
        # blend toward umbral temperature inside spots
        return t * (1.0 - self._umbra) + self._spot_T * self._umbra

    def emission(self):
        """Relative bolometric emission map (spots + intergranular lanes)."""
        lane = xp.where(self._s < -0.35, 0.80, 1.0)
        return xp.clip(self._emis * lane, 0.02, 1.0)

    def _color_temperature(self):
        """Temperature map with the Dravins blueshift nudge folded in.

        Used only for colour (`to_srgb`/`disk_image`); `temperature()`
        stays the raw physical map so determinism and Warp/reference
        parity tests are unaffected."""
        # net blue bias (granules brighter/larger than lanes) + a
        # per-feature modulation from the vertical-velocity proxy.
        dT = _DRAVINS_NET + _DRAVINS_K * xp.tanh(self._w / _DRAVINS_WREF)
        return self.temperature() + dT

    # --- colour ------------------------------------------------------
    def _srgb_lut(self, lam_nm, n: int = 256):
        from goldilocks.skyview import (planck_spectral, cie_xyz_bar,
                                        spectrum_to_srgb)
        T = self.temperature()
        tmin = max(float(B.asnumpy(T).min()) - 50.0, 1500.0)
        tmax = float(B.asnumpy(T).max()) + 50.0
        temps = np.linspace(tmin, tmax, n)
        specs = np.stack([planck_spectral(lam_nm, tt) for tt in temps])
        _, yb, _ = cie_xyz_bar(lam_nm)
        dl = float(lam_nm[1] - lam_nm[0])
        y_sun = float(np.dot(planck_spectral(lam_nm, self.surface.teff),
                             yb) * dl)
        rgb = spectrum_to_srgb(specs, lam_nm, 0.40 / max(y_sun, 1e-12))
        return temps, rgb.astype(np.float64)

    def to_srgb(self, lam_nm=None):
        """Equirectangular (H, W, 3) uint8 sRGB image."""
        from goldilocks.skyview import lambda_grid_nm
        if lam_nm is None:
            lam_nm = lambda_grid_nm()
        temps, lut = self._srgb_lut(lam_nm)
        Tn = B.asnumpy(self._color_temperature())
        idx = np.interp(Tn, temps, np.arange(len(temps)))
        i0 = np.clip(idx.astype(int), 0, len(temps) - 2)
        f = (idx - i0)[..., None]
        col = lut[i0] * (1.0 - f) + lut[i0 + 1] * f
        col = col * B.asnumpy(self.emission())[..., None]
        return np.clip(col, 0, 255).astype(np.uint8)

    def disk_image(self, size: int = 512, sub_lon: float = 0.0,
                   lam_nm=None) -> np.ndarray:
        """Orthographic limb-darkened disk view of the current field."""
        from goldilocks.skyview import lambda_grid_nm
        if lam_nm is None:
            lam_nm = lambda_grid_nm()
        temps, lut = self._srgb_lut(lam_nm)
        Tn = B.asnumpy(self._color_temperature())
        En = B.asnumpy(self.emission())
        ax = np.linspace(-1.0, 1.0, size)
        X, Y = np.meshgrid(ax, ax)
        rho2 = X ** 2 + Y ** 2
        disk = rho2 <= 1.0
        Z = np.sqrt(np.clip(1.0 - rho2, 0.0, 1.0))
        mu = Z
        lat = np.arcsin(np.clip(Y, -1.0, 1.0))
        lon = (sub_lon + np.arctan2(X, Z)) % (2.0 * math.pi)
        fi = np.clip((lat + math.pi / 2) / math.pi * self.H, 0,
                     self.H - 1).astype(int)
        fj = (lon / (2.0 * math.pi) * self.W).astype(int) % self.W
        T = Tn[fi, fj]
        E = En[fi, fj]
        idx = np.interp(T, temps, np.arange(len(temps)))
        i0 = np.clip(idx.astype(int), 0, len(temps) - 2)
        fr = (idx - i0)[..., None]
        col = lut[i0] * (1.0 - fr) + lut[i0 + 1] * fr
        ld = B.asnumpy(limb_darkening(B.asarray(mu),
                                      self.surface.ld_u1,
                                      self.surface.ld_u2))
        col = col * (E * ld)[..., None]
        out = np.zeros((size, size, 3), np.uint8)
        out[disk] = np.clip(col[disk], 0, 255).astype(np.uint8)
        return out[::-1]            # image-space y-down

    # --- diagnostics -------------------------------------------------
    def memory_report(self) -> str:
        mb = self._s.nbytes / 1e6
        n_buf = 3 + (2 if self.backend_name == "warp" else 0)
        tot = mb * n_buf
        cap = (f"k_phys={self.k_phys:.0f} -> rendered={self.freq:.1f}"
               f"{' (capped to grid)' if self._freq_capped else ''}"
               f", spot_k={self.spot_freq:.2f}")
        return (f"[photosphere] {self.res} {self.H}x{self.W} "
                f"backend={self.backend_name} dev={self.device} "
                f"buffer={mb:.2f} MB x{n_buf} ~ {tot:.1f} MB total; {cap}")
