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
