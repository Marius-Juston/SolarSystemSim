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
from typing import Optional

import numpy as np

from goldilocks import backend as B
from goldilocks import noise as N
from goldilocks.starsurface import StarSurface, star_surface_for, limb_darkening

xp = B.xp

_RES_PRESETS = {
    "dev": (128, 256),  # (H lat, W lon) -- CPU / tests
    "med": (512, 1024),
    "high": (2048, 4096),  # the A6000 box
}

_EPS_POLE = 1e-4
_UMBRA_T = 4100.0  # K  (research checklist 2.9)
_DT_GRANULE = 500.0  # K  photospheric +/- fluctuation amplitude
# Resolution-safety: keep >= this many pixels per granule cell at every
# preset so higher resolution only adds detail (never aliases).
_MIN_PX_PER_CELL = 6.0
_FLOW_K = 6.0  # fixed super-granular advecting-flow wavenumber
_SPOT_K = 16.0  # fixed individual-spot wavenumber
_SPOT_REGION_K = 4.0  # fixed active-region clustering wavenumber
_LANE_CONTRAST = 0.28  # intergranular-lane darkening (network depth)
_EXPOSURE = 0.62  # CIE->sRGB exposure (bright warm solar white)
_EVERSHED_REF_KMS = 4.0  # reference penumbral outflow for cell scaling
_PEN_FIL_N = 38.0  # radial penumbral filament count (DKIST-like)
# Relief: the granulation is a corrugated convective surface, not a flat
# map.  A self-emissive bump model (no external light -- a star is its
# own source): cell tops (flat, high field) brighten, lane shoulders
# (steep gradient) shade -> a 3-D "popcorn" look.
_RELIEF_AMP = 0.20  # cell-top vs slope brightness swing
_RELIEF_SHOULDER = 0.18  # extra ambient-occlusion darkening in lanes
# Granulation continuum RMS intensity contrast (DKIST/Hinode visible
# ~0.15-0.20); the disk tone is scaled so the cell/lane luminance
# spread matches this physical value.
_GRANULE_RMS = 0.18
# Canonical stellar-class apparent sRGB (Mitchell Charity /
# en.wikipedia.org/wiki/Stellar_classification): the disk colour LUT
# must land on this black-body locus so a G star is warm-white, B/A
# blue-white, K/M orange-red -- accurate to spec, not desaturated grey.
STELLAR_CLASS_SRGB = {
    "O": (155, 176, 255), "B": (170, 191, 255), "A": (202, 215, 255),
    "F": (248, 247, 255), "G": (255, 244, 234), "K": (255, 210, 161),
    "M": (255, 204, 111),
}
# Dravins convective blueshift (research §2.11): a documented,
# physically-anchored *stylisation* -- the ~300 m/s net photospheric
# blueshift is rendered as a small colour-temperature nudge (rising
# bright plasma bluer, sinking lanes redder) rather than a true
# line-shift, consistent with the bolometric CIE pipeline in `skyview`.
_DRAVINS_K = 18.0  # K  per-granule colour-temperature modulation
_DRAVINS_WREF = 0.5  # w-proxy scale for the tanh soft-clip
_DRAVINS_NET = 3.0  # K  disk-mean net blueshift (~300 m/s class)

# Optional Warp backend ------------------------------------------------
_WARP_MODE = os.environ.get(
    "GOLDILOCKS_PHOTOSPHERE_BACKEND", "auto").strip().lower()

try:  # pragma: no cover - env
    import warp as wp  # type: ignore

    _HAVE_WARP = True
except Exception:  # pragma: no cover
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

    _warp_init._bilin = _bilin  # keep refs alive
    _warp_init._step_kernel = _step_kernel
    _WARP_READY = True
    return True


def _srgb_oetf(c):
    """Linear -> sRGB opto-electronic transfer (IEC 61966-2-1)."""
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c,
                    1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def blackbody_srgb(teff: float, lam_nm=None):
    """Apparent sRGB (0..1) of a black body at ``teff`` K.

    The standard stellar-classification recipe: Planck spectrum -> CIE
    1931 XYZ -> linear sRGB (D65 primaries) -> clip negatives ->
    normalise to max channel -> sRGB gamma.  Reproduces the
    Charity/Wikipedia O..M apparent colours.  Hue only -- luminance is
    applied by the caller (granulation / limb / relief)."""
    from goldilocks.skyview import (planck_spectral, cie_xyz_bar,
                                    _XYZ_TO_RGB, lambda_grid_nm)
    if lam_nm is None:
        lam_nm = lambda_grid_nm()
    sp = planck_spectral(lam_nm, teff)
    xb, yb, zb = cie_xyz_bar(lam_nm)
    dl = float(lam_nm[1] - lam_nm[0])
    xyz = np.array([np.dot(sp, xb), np.dot(sp, yb), np.dot(sp, zb)]) * dl
    lin = np.clip(_XYZ_TO_RGB @ xyz, 0.0, None)
    lin = lin / (lin.max() + 1e-12)
    return _srgb_oetf(lin)


# ---------------------------------------------------------------------
class Photosphere:
    """Stateful equirectangular photosphere field."""

    def __init__(self, surface: StarSurface, *, res: str = "dev",
                 seed: int = 0, backend: Optional[str] = None,
                 device: str = "cpu", log_mem: bool = False) -> None:
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
        # radius ~ R*/1Mm (the Sun ~ R_sun/1Mm ~ 696, scaled by the
        # surface's angular cell size).  Rendered freq tracks this but is
        # clamped so every grid cell keeps >= _MIN_PX_PER_CELL pixels --
        # i.e. higher resolution shows MORE of the same physical scene
        # (finer granulation), never aliased and never a different scene.
        gs = max(float(surface.granule_scale_rel), 0.15)
        self.k_phys = 696.0 / gs
        self.freq = float(np.clip(self.k_phys, 8.0,
                                  max(16.0, self.W / _MIN_PX_PER_CELL)))
        self._freq_capped = self.k_phys > self.freq + 1e-9
        # Advecting (super-granular) flow + active-region wavenumbers are
        # FIXED physical low frequencies, independent of resolution and of
        # the granule freq, so the boiling motion and sunspot size are the
        # same physical features at dev/med/high (research §2.8: a separate
        # low-frequency channel, well below granulation).
        self.flow_k = _FLOW_K
        self.spot_freq = _SPOT_K
        self.period = 256.0  # lattice period -> bounded time (no drift)
        # Ro^-1/2 turnover (research §4.1), solar-normalised.
        self.flow = 0.9 * (max(float(surface.rossby), 1e-3) / 2.21) ** -0.5
        # Granule *lifetime* also scales with Ro^-1/2 (research §2.4): the
        # noise-time advance rate now tracks the flow timescale instead of
        # a hardcoded constant.  Solar-normalised so the Sun (Ro~2.21)
        # recovers ~0.25 -- i.e. ~ the previous bare t/4 -- while more
        # active (low-Ro) stars boil faster.
        self._time_rate = 0.25 * self.flow / 0.9

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
        # Cellular (Worley) convective granulation -- bright polygonal
        # granule interiors + thin dark intergranular downflow lanes
        # (Worley 1996); the physically-correct basis (value noise gave
        # smooth blobs, not cells).  Same seeding/normalisation contract
        # so temperature() bounds, determinism and CFL are preserved.
        s = N.granulation_field(nx, ny, nz, self.freq, seed=self.seed)
        self._s = xp.asarray(s, dtype=xp.float64)
        self._scratch = xp.empty_like(self._s)
        # Dravins convective-blueshift state (research §2.11): vertical
        # velocity proxy from the per-step change in the potential.
        self._w = xp.zeros_like(self._s)
        self._last_dt = 0.1
        self._build_spots()
        self._ws_in = None  # warp persistent buffers (lazy)
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
        # Active-region field: a fixed *physical* wavenumber (resolution-
        # independent) high enough that individual spots are small and
        # several, not a few giant blobs.
        m = N.value_noise_3d(nx * self.spot_freq, ny * self.spot_freq,
                             nz * self.spot_freq, seed=self.seed + 31)
        m = 0.5 * (m + 1.0)  # -> [0,1]
        # A few discrete active regions (low-frequency, resolution-
        # independent) so spots cluster into groups instead of a uniform
        # belt around the whole star.
        reg = N.value_noise_3d(nx * _SPOT_REGION_K, ny * _SPOT_REGION_K,
                               nz * _SPOT_REGION_K, seed=self.seed + 30)
        reg = 0.5 * (reg + 1.0)
        reg = xp.clip((reg - 0.58) / 0.22, 0.0, 1.0)
        reg = reg * reg * (3.0 - 2.0 * reg)  # ~ few active patches
        # butterfly latitude band + binary active longitude
        lat_deg = B.asarray(self._lat * 180.0 / math.pi)[:, None]
        band = xp.exp(-((xp.abs(lat_deg) - float(s.spot_lat_deg)) ** 2)
                      / (2.0 * 11.0 ** 2))
        amp = float(s.active_long_amp)
        if amp > 0.0:
            ln = B.asarray(self._lon)[None, :]
            band = band * xp.clip(
                1.0 + amp * xp.cos(2.0 * (ln - float(s.phi_sub))),
                0.0, 2.0)
        w = m * band * reg
        # Coverage-driven threshold: pick it so only ~spot_coverage of the
        # surface becomes spotted (area-accurate, auto-scaling with the
        # star's activity).  -> small, sparse spots, not big patches.
        cov = float(np.clip(s.spot_coverage, 0.0, 0.30))
        if cov <= 1e-4:
            zero = xp.zeros_like(w)
            self._emis = xp.ones_like(w)
            self._spot_T = self.surface.teff + 0.0 * w
            self._umbra = zero
            self._ev_w = None  # spotless -> Evershed no-op
            return
        thr = float(xp.quantile(w, 1.0 - cov))
        # crisp core with a small anti-aliased edge
        spot = xp.clip((w - thr) / 0.05, 0.0, 1.0)
        spot = spot * spot * (3.0 - 2.0 * spot)
        # inner umbra vs thin outer penumbra ring
        umbra = xp.clip((spot - 0.55) / 0.45, 0.0, 1.0)
        umbra = umbra * umbra * (3.0 - 2.0 * umbra)
        penum = spot - umbra
        # Spot-field gradient -> outward radial direction (shared by the
        # Evershed flow and the penumbral filaments).
        gj = 0.5 * (xp.roll(spot, -1, 1) - xp.roll(spot, 1, 1))
        gi = 0.5 * (xp.concatenate([spot[1:], spot[-1:]], 0)
                    - xp.concatenate([spot[:1], spot[:-1]], 0))
        gm = xp.sqrt(gi * gi + gj * gj) + 1e-9
        # Penumbral filaments (DKIST): bright-headed radial strands.  The
        # gradient azimuth rotates once around the spot, so cos(N*ang)
        # makes ~N radial filaments; jittered so they are not perfectly
        # periodic.  Evershed-aligned (same outward direction).
        ang = xp.arctan2(gj, gi)
        jit = 0.6 * N.value_noise_3d(nx * 40.0, ny * 40.0, nz * 40.0,
                                     seed=self.seed + 33)
        fstr = 0.5 + 0.5 * xp.cos(_PEN_FIL_N * ang + jit)
        # Umbral dots (DKIST): sparse hot convective intrusions in the
        # umbra, ~photospheric temperature.
        ud = N.value_noise_3d(nx * 95.0, ny * 95.0, nz * 95.0,
                              seed=self.seed + 34)
        ud = xp.clip((0.5 * (ud + 1.0) - 0.62) / 0.12, 0.0, 1.0)
        ud = ud * ud * (3.0 - 2.0 * ud) * umbra  # only in umbra
        # Emission: dark umbra (lifted by umbral dots) + filamentary
        # penumbra (bright heads where fstr high).
        pen_dark = 0.42 - 0.24 * fstr  # 0.18..0.42
        self._emis = xp.clip(
            1.0 - 0.82 * umbra - penum * pen_dark + 0.30 * ud,
            0.05, 1.0)
        self._spot_T = (_UMBRA_T * umbra
                        + self.surface.teff * (1.0 - umbra)
                        + (self.surface.teff - _UMBRA_T) * 0.6 * ud)
        self._umbra = xp.clip(umbra - 0.7 * ud, 0.0, 1.0)
        # Photospheric Evershed (research/checklist 2.10): near-horizontal
        # radial *outflow* through the penumbra, away from the spot.
        self._ev_i = -gi / gm
        self._ev_j = -gj / gm
        self._ev_w = xp.clip(penum, 0.0, 1.0)

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
                                         eps=1e-2, freq=self.flow_k)
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
        # Photospheric Evershed outflow inside the penumbra (2.10):
        # an extra advection, magnitude from surface.evershed_kms,
        # CFL-clamped so the < 0.5 invariant still holds.  Reference
        # path is the verified oracle; the Warp parity is statistical.
        if getattr(self, "_ev_w", None) is not None:
            evk = float(getattr(self.surface, "evershed_kms", 0.0))
            ev_cells = min(0.30, (evk / _EVERSHED_REF_KMS) * 0.30
                           * (float(dt) / 0.1))
            di = di + ev_cells * self._ev_i * self._ev_w
            dj = dj + ev_cells * self._ev_j * self._ev_w
            di = xp.clip(di, -0.49, 0.49)
            dj = xp.clip(dj, -0.49, 0.49)
            self.last_cfl = min(0.49, max(target, ev_cells))
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
                          float(self.flow_k), float(self._tx()),
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
        """Relative bolometric emission (spots + intergranular network).

        The intergranular lanes are the *cell boundaries* -- thin dark
        downflow lanes where the field gradient is steep -- not a hard
        amplitude cut (which produced blobs).  The gradient is
        mean-normalised so the network is identical in appearance and
        thickness at every resolution (resolution-safe)."""
        s = self._s
        # longitude is periodic (axis 1); latitude is clamped (axis 0)
        ds_j = 0.5 * (xp.roll(s, -1, 1) - xp.roll(s, 1, 1))
        ds_i = 0.5 * (xp.concatenate([s[1:], s[-1:]], 0)
                      - xp.concatenate([s[:1], s[:-1]], 0))
        gmag = xp.sqrt(ds_j * ds_j + ds_i * ds_i)
        gn = gmag / (float(gmag.mean()) + 1e-9)  # res/amplitude invariant
        # only well-above-average gradients darken -> a sparse, thin
        # connected lane network (not all-over speckle)
        u = xp.clip((gn - 1.4) / 1.4, 0.0, 1.0)
        network = u * u * (3.0 - 2.0 * u)
        lane = 1.0 - _LANE_CONTRAST * network
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

    def _grav_factor_1d(self):
        """ELR2011 gravity-darkening bolometric multiplier per latitude.

        Colour-only (like the Dravins split): `temperature()`/`scalar()`
        stay raw so determinism and Warp/reference parity are unaffected.
        ~1 for the slowly rotating Sun (omega-tilde~8e-3 -> <1e-4)."""
        from goldilocks.starsurface import gravity_darkening_factor
        g = gravity_darkening_factor(B.asarray(self._lat), self.surface)
        return B.asnumpy(g)  # shape (H,)

    def _color_temperature_grav(self):
        """Colour temperature incl. Dravins + ELR gravity darkening."""
        ct = B.asnumpy(self._color_temperature())
        g = self._grav_factor_1d()  # T_local = T * gd^(1/4)
        return ct * (g[:, None] ** 0.25)

    def _relief(self):
        """Self-emissive bump shading -> a 3-D corrugated convective
        surface (no external light; a star is its own source).

        Bright raised granule domes (positive field, hot upflow), dark
        sunken lanes with shaded shoulders (steep gradient).  Colour/
        disk-only; `temperature()`/`scalar()`/parity unaffected."""
        s = B.asnumpy(self._s).astype(np.float64)
        sn = (s - s.mean()) / (s.std() + 1e-9)
        ds_j = 0.5 * (np.roll(s, -1, 1) - np.roll(s, 1, 1))
        ds_i = 0.5 * (np.concatenate([s[1:], s[-1:]], 0)
                      - np.concatenate([s[:1], s[:-1]], 0))
        gmag = np.sqrt(ds_j ** 2 + ds_i ** 2)
        gn = gmag / (float(gmag.mean()) + 1e-9)
        relief = (1.0 + _RELIEF_AMP * np.tanh(sn)
                  - _RELIEF_SHOULDER * np.clip(gn - 1.0, 0.0, None) * 0.30)
        return np.clip(relief, 0.45, 1.65)

    # --- colour ------------------------------------------------------
    def _srgb_lut(self, lam_nm, n: int = 256):
        """Canonical black-body -> sRGB LUT (photosphere only).

        Reproduces the standard stellar-classification chromaticities
        (Mitchell Charity / Wikipedia "Stellar classification": O
        (155,176,255) ... G (255,244,234) ... M (255,204,111)) -- CIE
        1931 XYZ, sRGB primaries + D65, max-channel normalised, sRGB
        gamma.  A 5772 K G star is therefore a *bright warm white* (the
        physically correct colour, not the desaturated grey the shared
        ACES path produced).  Per-pixel luminance (granulation / limb /
        relief) is applied separately, so the hue stays on the stellar
        locus regardless of brightness."""
        T = self.temperature()
        tmin = max(float(B.asnumpy(T).min()) - 50.0, 1500.0)
        tmax = float(B.asnumpy(T).max()) + 50.0
        temps = np.linspace(tmin, tmax, n)
        lut = np.stack([blackbody_srgb(tt, lam_nm) for tt in temps]) * 255.0
        return temps, lut

    def to_srgb(self, lam_nm=None):
        """Equirectangular (H, W, 3) uint8 sRGB image."""
        from goldilocks.skyview import lambda_grid_nm
        if lam_nm is None:
            lam_nm = lambda_grid_nm()
        temps, lut = self._srgb_lut(lam_nm)
        Tn = self._color_temperature_grav()
        idx = np.interp(Tn, temps, np.arange(len(temps)))
        i0 = np.clip(idx.astype(int), 0, len(temps) - 2)
        f = (idx - i0)[..., None]
        col = lut[i0] * (1.0 - f) + lut[i0 + 1] * f
        col = col * (B.asnumpy(self.emission())
                     * self._relief())[..., None]
        return np.clip(col, 0, 255).astype(np.uint8)

    def disk_image(self, size: int = 512, sub_lon: float = 0.0,
                   lam_nm=None, emission_line=None) -> np.ndarray:
        """Orthographic limb-darkened disk view of the current field.

        ``emission_line`` (None | 'halpha' | 'caiik' | 'heii304' |
        'feix171') composites the physically-grounded chromosphere /
        transition-region overlay (research §4); None keeps the
        bolometric render byte-identical."""
        from goldilocks.skyview import lambda_grid_nm
        if lam_nm is None:
            lam_nm = lambda_grid_nm()
        temps, lut = self._srgb_lut(lam_nm)
        Tn = self._color_temperature_grav()
        En = B.asnumpy(self.emission())
        ax = np.linspace(-1.0, 1.0, size)
        X, Y = np.meshgrid(ax, ax)
        obl = float(getattr(self.surface, "oblateness", 0.0))
        if obl > 1e-3:
            # Oblate Roche spheroid (research §3 / Phase 3): spin axis is
            # screen-up, equatorial radius 1, polar radius q=1/(1+f_obl);
            # per-pixel mu from the ellipsoid surface normal (true
            # foreshortening).  Reduces exactly to the sphere at f->0.
            q = 1.0 / (1.0 + obl)
            v = Y / q
            rho2 = X ** 2 + v ** 2
            disk = rho2 <= 1.0
            Z = np.sqrt(np.clip(1.0 - rho2, 0.0, 1.0))
            nrm = np.sqrt(X ** 2 + (v / q) ** 2 + Z ** 2) + 1e-12
            mu = Z / nrm
            lat = np.arcsin(np.clip(v, -1.0, 1.0))
            lon = (sub_lon + np.arctan2(X, Z)) % (2.0 * math.pi)
        else:
            rho2 = X ** 2 + Y ** 2
            disk = rho2 <= 1.0
            Z = np.sqrt(np.clip(1.0 - rho2, 0.0, 1.0))
            mu = Z
            lat = np.arcsin(np.clip(Y, -1.0, 1.0))
            lon = (sub_lon + np.arctan2(X, Z)) % (2.0 * math.pi)
        # Bilinear sample the equirect maps (lat clamped, lon wrapped) so
        # the disk is smooth at every resolution -- no nearest-neighbour
        # blockiness, and detail tracks the grid resolution.
        fi = np.clip((lat + math.pi / 2) / math.pi * self.H - 0.5,
                     0.0, self.H - 1.0)
        fj = (lon / (2.0 * math.pi) * self.W - 0.5) % self.W
        i0 = np.floor(fi).astype(int)
        i1 = np.clip(i0 + 1, 0, self.H - 1)
        j0 = np.floor(fj).astype(int) % self.W
        j1 = (j0 + 1) % self.W
        ti = fi - i0
        tj = fj - np.floor(fj)

        def _bs(A):
            a = A[i0, j0] * (1.0 - tj) + A[i0, j1] * tj
            b = A[i1, j0] * (1.0 - tj) + A[i1, j1] * tj
            return a * (1.0 - ti) + b * ti

        Rn = self._relief()
        T = _bs(Tn)
        E = _bs(En) * _bs(Rn)
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
        if emission_line:
            from goldilocks.chromosphere import chromosphere_overlay
            out = chromosphere_overlay(out, X, Y, disk, mu, self.surface,
                                       line=emission_line, seed=self.seed)
        return out[::-1]  # image-space y-down

    # --- diagnostics -------------------------------------------------
    def memory_report(self) -> str:
        mb = self._s.nbytes / 1e6
        n_buf = 3 + (2 if self.backend_name == "warp" else 0)
        tot = mb * n_buf
        cap = (f"k_phys={self.k_phys:.0f} -> granule={self.freq:.1f}"
               f"{' (grid-capped)' if self._freq_capped else ''}"
               f" (>= {_MIN_PX_PER_CELL:.0f} px/cell), "
               f"flow_k={self.flow_k:.1f}, spot_k={self.spot_freq:.1f} "
               f"(both resolution-independent)")
        return (f"[photosphere] {self.res} {self.H}x{self.W} "
                f"backend={self.backend_name} dev={self.device} "
                f"buffer={mb:.2f} MB x{n_buf} ~ {tot:.1f} MB total; {cap}")
