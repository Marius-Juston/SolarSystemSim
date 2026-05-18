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
