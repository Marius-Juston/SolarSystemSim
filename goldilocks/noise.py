"""
noise.py
--------
Vectorised, deterministic procedural noise primitives for the surface
renderers (star granulation, moon regolith / crater mottling).

Pure Layer-0 helper: depends only on the array backend seam
(`goldilocks.backend`).  Every function runs on the active backend
(`B.xp`, NumPy or CuPy), is fully vectorised, and is deterministic for a
given integer `seed` -- the lattice permutation / value tables are
`lru_cache`d and seeded exactly like `skyview.background_starfield` and
`skyview._od_table`, so every frame of an animation (rendered in
independent pool workers) shares one identical field.

Curl noise
----------
`curl_noise_2d` returns the 2-D curl of a scalar noise potential psi,

    v = ( d psi / d y ,  - d psi / d x ),

which is divergence-free by construction (Bridson, Hourihan &
Nordenstam 2007, "Curl-Noise for Procedural Fluid Flow", SIGGRAPH).
Used to advect the granulation brightness field so convective cells
swirl like real photospheric flow without compressible artefacts.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from goldilocks import backend as B

xp = B.xp


# ---------------------------------------------------------------------
# Seeded lattice tables (deterministic, cached -- background_starfield
# pattern)
# ---------------------------------------------------------------------
@lru_cache(maxsize=16)
def _tables(seed: int):
    """Doubled permutation table + matching lattice value table.

    Returns (perm (512,) int32, val (256,) float64 in [-1, 1]) on the
    active backend.  Doubling perm avoids a modulo on the second axis.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(256).astype(np.int32)
    perm2 = np.concatenate([perm, perm])
    val = (rng.random(256) * 2.0 - 1.0)
    return B.asarray(perm2), B.asarray(val)


def _fade(t):
    """Perlin quintic smoothstep 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def value_noise_2d(x, y, seed: int = 1):
    """Lattice value noise in ~[-1, 1] at coordinates (x, y).

    `x`, `y` are broadcastable backend arrays.  Smooth (quintic) bilinear
    interpolation of a seeded per-lattice-point value table.
    """
    perm, val = _tables(int(seed))
    x = xp.asarray(x, dtype=xp.float64)
    y = xp.asarray(y, dtype=xp.float64)
    xi = xp.floor(x).astype(xp.int64)
    yi = xp.floor(y).astype(xp.int64)
    fx = x - xi
    fy = y - yi
    xi = xi & 255
    yi = yi & 255
    u = _fade(fx)
    v = _fade(fy)

    def _h(ix, iy):
        return val[(perm[(perm[ix] + iy) & 255]) & 255]

    n00 = _h(xi, yi)
    n10 = _h((xi + 1) & 255, yi)
    n01 = _h(xi, (yi + 1) & 255)
    n11 = _h((xi + 1) & 255, (yi + 1) & 255)
    nx0 = n00 * (1.0 - u) + n10 * u
    nx1 = n01 * (1.0 - u) + n11 * u
    return nx0 * (1.0 - v) + nx1 * v


def fbm(x, y, seed: int = 1, octaves: int = 4,
        lacunarity: float = 2.0, gain: float = 0.5):
    """Fractal Brownian motion: summed octaves of `value_noise_2d`.

    Normalised so the result stays in roughly [-1, 1].
    """
    total = xp.zeros_like(xp.asarray(x, dtype=xp.float64)
                          + xp.asarray(y, dtype=xp.float64))
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for o in range(int(octaves)):
        total = total + amp * value_noise_2d(x * freq, y * freq,
                                             seed + o)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / max(norm, 1e-9)


def value_noise_3d(x, y, z, seed: int = 1):
    """Lattice value noise in ~[-1, 1] at coordinates (x, y, z).

    Trilinear quintic interpolation of the same seeded lattice value
    table used by `value_noise_2d`, with the permutation hash chain
    extended by a third axis.
    """
    perm, val = _tables(int(seed))
    x = xp.asarray(x, dtype=xp.float64)
    y = xp.asarray(y, dtype=xp.float64)
    z = xp.asarray(z, dtype=xp.float64)
    xi = xp.floor(x).astype(xp.int64)
    yi = xp.floor(y).astype(xp.int64)
    zi = xp.floor(z).astype(xp.int64)
    fx = x - xi
    fy = y - yi
    fz = z - zi
    xi = xi & 255
    yi = yi & 255
    zi = zi & 255
    u = _fade(fx)
    v = _fade(fy)
    w = _fade(fz)

    def _h(ix, iy, iz):
        return val[(perm[(perm[(perm[ix] + iy) & 255] + iz) & 255]) & 255]

    x1 = (xi + 1) & 255
    y1 = (yi + 1) & 255
    z1 = (zi + 1) & 255

    def _lerp(a, b, t):
        return a * (1.0 - t) + b * t

    c00 = _lerp(_h(xi, yi, zi), _h(x1, yi, zi), u)
    c10 = _lerp(_h(xi, y1, zi), _h(x1, y1, zi), u)
    c01 = _lerp(_h(xi, yi, z1), _h(x1, yi, z1), u)
    c11 = _lerp(_h(xi, y1, z1), _h(x1, y1, z1), u)
    c0 = _lerp(c00, c10, v)
    c1 = _lerp(c01, c11, v)
    return _lerp(c0, c1, w)


def value_noise_4d(x, y, z, t, seed: int = 1):
    """Lattice value noise in ~[-1, 1] at coordinates (x, y, z, t).

    Quadrilinear quintic interpolation; the hash chain is extended by a
    fourth (time) axis so an evolving sphere potential stays seamless.
    """
    perm, val = _tables(int(seed))
    x = xp.asarray(x, dtype=xp.float64)
    y = xp.asarray(y, dtype=xp.float64)
    z = xp.asarray(z, dtype=xp.float64)
    t = xp.asarray(t, dtype=xp.float64)
    xi = xp.floor(x).astype(xp.int64)
    yi = xp.floor(y).astype(xp.int64)
    zi = xp.floor(z).astype(xp.int64)
    ti = xp.floor(t).astype(xp.int64)
    fx, fy, fz, ft = x - xi, y - yi, z - zi, t - ti
    xi, yi, zi, ti = xi & 255, yi & 255, zi & 255, ti & 255
    u, v, w, s = _fade(fx), _fade(fy), _fade(fz), _fade(ft)

    def _h(ix, iy, iz, it):
        return val[(perm[(perm[(perm[(perm[ix] + iy) & 255]
                                + iz) & 255] + it) & 255]) & 255]

    x1, y1, z1, t1 = ((xi + 1) & 255, (yi + 1) & 255,
                      (zi + 1) & 255, (ti + 1) & 255)

    def _lerp(a, b, k):
        return a * (1.0 - k) + b * k

    def _cube(it):
        c00 = _lerp(_h(xi, yi, zi, it), _h(x1, yi, zi, it), u)
        c10 = _lerp(_h(xi, y1, zi, it), _h(x1, y1, zi, it), u)
        c01 = _lerp(_h(xi, yi, z1, it), _h(x1, yi, z1, it), u)
        c11 = _lerp(_h(xi, y1, z1, it), _h(x1, y1, z1, it), u)
        return _lerp(_lerp(c00, c10, v), _lerp(c01, c11, v), w)

    return _lerp(_cube(ti), _cube(t1), s)


# ---------------------------------------------------------------------
# Worley / cellular noise (Worley 1996, "A Cellular Texture Basis
# Function").  Physically the right basis for convective granulation:
# jittered feature points = granule centres, F2-F1 = the intergranular
# downflow lane network.  Deterministic (same seeded `_tables` hash).
# ---------------------------------------------------------------------
def _cell_jitter(cx, cy, cz, perm, val, salt: int):
    """Deterministic feature-point offset component in [0, 1) for an
    integer lattice cell, independent per axis via `salt`."""
    h = perm[(perm[(perm[(cx + salt * 131) & 255] + cy) & 255]
              + cz) & 255] & 255
    return 0.5 * (val[h] + 1.0)


def worley_noise_3d(x, y, z, seed: int = 1, return_f2: bool = False):
    """Cellular noise: distance to the nearest (F1) and 2nd-nearest (F2)
    jittered feature point.  ``F2 - F1`` is ~0 on cell boundaries and
    large in cell interiors -- the granule / lane signal."""
    perm, val = _tables(int(seed))
    x = xp.asarray(x, dtype=xp.float64)
    y = xp.asarray(y, dtype=xp.float64)
    z = xp.asarray(z, dtype=xp.float64)
    cx = xp.floor(x).astype(xp.int64)
    cy = xp.floor(y).astype(xp.int64)
    cz = xp.floor(z).astype(xp.int64)
    fx, fy, fz = x - cx, y - cy, z - cz
    big = xp.full(xp.broadcast_arrays(fx, fy, fz)[0].shape, 1.0e30)
    f1 = big
    f2 = big + 1.0
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            for oz in (-1, 0, 1):
                gx = (cx + ox) & 255
                gy = (cy + oy) & 255
                gz = (cz + oz) & 255
                jx = _cell_jitter(gx, gy, gz, perm, val, 0)
                jy = _cell_jitter(gx, gy, gz, perm, val, 1)
                jz = _cell_jitter(gx, gy, gz, perm, val, 2)
                dx = ox + jx - fx
                dy = oy + jy - fy
                dz = oz + jz - fz
                d = xp.sqrt(dx * dx + dy * dy + dz * dz)
                closer = d < f1
                f2 = xp.where(closer, f1, xp.minimum(f2, d))
                f1 = xp.where(closer, d, f1)
    if return_f2:
        return f1, f2
    return f1


# ---------------------------------------------------------------------
# 3-D simplex (gradient) noise -- Perlin/Gustavson skewed-simplex; no
# lattice-axis bias, smoother than value noise.  Deterministic via the
# seeded permutation table.
# ---------------------------------------------------------------------
_SIMPLEX_GRAD3 = np.array(
    [[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
     [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
     [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]], dtype=np.float64)


def simplex_noise_3d(x, y, z, seed: int = 1):
    """3-D simplex gradient noise in ~[-1, 1] (deterministic)."""
    perm, _ = _tables(int(seed))
    g3 = B.asarray(_SIMPLEX_GRAD3)
    x = xp.asarray(x, dtype=xp.float64)
    y = xp.asarray(y, dtype=xp.float64)
    z = xp.asarray(z, dtype=xp.float64)
    F3 = 1.0 / 3.0
    G3 = 1.0 / 6.0
    s = (x + y + z) * F3
    i = xp.floor(x + s)
    j = xp.floor(y + s)
    k = xp.floor(z + s)
    t = (i + j + k) * G3
    x0 = x - (i - t)
    y0 = y - (j - t)
    z0 = z - (k - t)
    # Canonical Gustavson tetra ordering (six cases A..F).
    g = x0 >= y0
    h = y0 >= z0
    p = x0 >= z0
    A = g & h
    Bm = g & (~h) & p
    C = g & (~h) & (~p)
    D = (~g) & (~h)
    E = (~g) & h & (~p)
    Fm = (~g) & h & p
    one = xp.ones_like(x0)
    zero = xp.zeros_like(x0)
    i1 = xp.where(A | Bm, one, zero)
    j1 = xp.where(E | Fm, one, zero)
    k1 = xp.where(C | D, one, zero)
    i2 = xp.where(A | Bm | C | Fm, one, zero)
    j2 = xp.where(A | D | E | Fm, one, zero)
    k2 = xp.where(Bm | C | D | E, one, zero)

    def _grad(ix, iy, iz, dx, dy, dz):
        idx = (perm[(perm[(perm[ix.astype(xp.int64) & 255]
                           + iy.astype(xp.int64)) & 255]
                     + iz.astype(xp.int64)) & 255] % 12)
        gv = g3[idx]
        return gv[..., 0] * dx + gv[..., 1] * dy + gv[..., 2] * dz

    def _corner(dx, dy, dz, ix, iy, iz):
        tt = 0.6 - dx * dx - dy * dy - dz * dz
        tt = xp.where(tt < 0.0, 0.0, tt)
        return tt ** 4 * _grad(ix, iy, iz, dx, dy, dz)

    x1 = x0 - i1 + G3
    y1 = y0 - j1 + G3
    z1 = z0 - k1 + G3
    x2 = x0 - i2 + 2.0 * G3
    y2 = y0 - j2 + 2.0 * G3
    z2 = z0 - k2 + 2.0 * G3
    x3 = x0 - 1.0 + 3.0 * G3
    y3 = y0 - 1.0 + 3.0 * G3
    z3 = z0 - 1.0 + 3.0 * G3
    n = (_corner(x0, y0, z0, i, j, k)
         + _corner(x1, y1, z1, i + i1, j + j1, k + k1)
         + _corner(x2, y2, z2, i + i2, j + j2, k + k2)
         + _corner(x3, y3, z3, i + 1.0, j + 1.0, k + 1.0))
    return xp.clip(32.0 * n, -1.0, 1.0)


def domain_warp3(x, y, z, seed: int = 1, amp: float = 0.5):
    """Organic domain warp p -> p + amp*noise(p) (research 2.3)."""
    wx = simplex_noise_3d(x + 11.3, y, z, seed + 1)
    wy = simplex_noise_3d(x, y + 7.7, z, seed + 2)
    wz = simplex_noise_3d(x, y, z + 3.1, seed + 3)
    return x + amp * wx, y + amp * wy, z + amp * wz


def fbm3(x, y, z, seed: int = 1, octaves: int = 4,
         lacunarity: float = 2.0, gain: float = 0.5,
         kind: str = "value"):
    """3-D fractal Brownian motion (persistence `gain`, `lacunarity`).

    ``kind`` selects the basis: 'value' (lattice) or 'simplex'.
    Normalised to ~[-1, 1]; the octave fall-off gives a 1/f spectrum.
    """
    base = simplex_noise_3d if kind == "simplex" else value_noise_3d
    total = xp.zeros_like(xp.asarray(x, dtype=xp.float64)
                          + xp.asarray(y, dtype=xp.float64)
                          + xp.asarray(z, dtype=xp.float64))
    amp, freq, norm = 1.0, 1.0, 0.0
    for o in range(int(octaves)):
        total = total + amp * base(x * freq, y * freq, z * freq,
                                   seed + o)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / max(norm, 1e-9)


def granulation_field(nx, ny, nz, freq: float = 1.0, seed: int = 1):
    """Cellular convective-granulation scalar in ~[-1, 1].

    Worley ``F2 - F1`` (bright polygonal granule interiors, thin dark
    intergranular downflow lanes) + a faint finer octave, made
    zero-mean / unit-RMS so the downstream ``Teff +/- dT`` mapping and
    amplitude are preserved (resolution-safe: `freq` is the physical
    granule wavenumber)."""
    f1, f2 = worley_noise_3d(nx * freq, ny * freq, nz * freq,
                             seed=seed + 11, return_f2=True)
    g = f2 - f1  # ~0 at lanes
    g = g + 0.18 * value_noise_3d(nx * freq * 2.3, ny * freq * 2.3,
                                  nz * freq * 2.3, seed=seed + 12)
    g = g - g.mean()
    g = g / (float(g.std()) + 1e-9)
    return xp.clip(g, -1.5, 1.5)


def curl_noise_sphere(nx, ny, nz, t=0.0, seed: int = 1,
                      eps: float = 1e-2, freq: float = 1.0):
    """Divergence-free tangent flow on the unit sphere.

    Builds a scalar potential ``psi`` from `value_noise_4d` sampled at
    the (frequency-scaled) unit-sphere point ``n=(nx,ny,nz)`` and time
    ``t``, takes its 3-D gradient by central differences, and returns
    ``v = grad(psi) x n`` with the radial component projected out
    (research §2.5).  ``v`` is tangent to the sphere by construction and
    its surface divergence vanishes to the finite-difference truncation
    error.  Returns the 3-D tangent vector ``(vx, vy, vz)``.
    """
    nx = xp.asarray(nx, dtype=xp.float64)
    ny = xp.asarray(ny, dtype=xp.float64)
    nz = xp.asarray(nz, dtype=xp.float64)

    def psi(px, py, pz):
        return value_noise_4d(px * freq, py * freq, pz * freq, t, seed)

    dpx = (psi(nx + eps, ny, nz) - psi(nx - eps, ny, nz)) / (2.0 * eps)
    dpy = (psi(nx, ny + eps, nz) - psi(nx, ny - eps, nz)) / (2.0 * eps)
    dpz = (psi(nx, ny, nz + eps) - psi(nx, ny, nz - eps)) / (2.0 * eps)
    # v = grad(psi) x n  (automatically tangent: perpendicular to n).
    vx = dpy * nz - dpz * ny
    vy = dpz * nx - dpx * nz
    vz = dpx * ny - dpy * nx
    # Numerically project out any residual radial drift.
    rad = vx * nx + vy * ny + vz * nz
    return vx - rad * nx, vy - rad * ny, vz - rad * nz


def curl_noise_2d(x, y, seed: int = 1, eps: float = 1e-2,
                  octaves: int = 4):
    """Divergence-free 2-D flow = curl of an fbm potential psi.

    Returns (u, v) = (d psi / d y, - d psi / d x), evaluated by central
    finite differences (Bridson et al. 2007).  Divergence
    du/dx + dv/dy is ~0 to the finite-difference truncation error.
    """

    def psi(px, py):
        return fbm(px, py, seed, octaves=octaves)

    dpsi_dy = (psi(x, y + eps) - psi(x, y - eps)) / (2.0 * eps)
    dpsi_dx = (psi(x + eps, y) - psi(x - eps, y)) / (2.0 * eps)
    return dpsi_dy, -dpsi_dx
