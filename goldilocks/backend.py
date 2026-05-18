"""
backend.py
----------
Single array-backend seam for the hot numerical kernels.

`xp` is either NumPy (CPU) or CuPy (CUDA GPU).  Everything that wants to
run on the GPU imports `from goldilocks import backend as B` and uses
`B.xp` plus the thin shims here for the handful of ops that differ
between the two libraries (`scatter_add`, `percentile`, `trapezoid`,
`interp`, host transfer).

Selection (env `GOLDILOCKS_BACKEND`):
  * ``auto`` (default) -- CuPy iff it imports and a CUDA device exists,
    else NumPy.
  * ``cpu``            -- force NumPy.
  * ``gpu``            -- force CuPy (raises if unavailable).

The rest of the codebase (physics primitives, ``system.py``) keeps using
plain NumPy; this module only governs the rendering / N-body kernels so
results stay identical on the CPU path and every pinned sanity value is
preserved.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np

_MODE = os.environ.get("GOLDILOCKS_BACKEND", "auto").strip().lower()

xp = np
ON_GPU = False
_cp = None

if _MODE in ("auto", "gpu"):
    try:
        import cupy as _cp_try  # type: ignore

        if _cp_try.cuda.runtime.getDeviceCount() > 0:
            _cp = _cp_try
            xp = _cp_try
            ON_GPU = True
    except Exception:
        if _MODE == "gpu":
            raise
        _cp = None

if _MODE == "gpu" and not ON_GPU:
    raise RuntimeError("GOLDILOCKS_BACKEND=gpu but no working CuPy/CUDA "
                       "device was found.")


def n_gpus() -> int:
    """Number of visible CUDA devices (0 on the CPU backend)."""
    if not ON_GPU:
        return 0
    try:
        return int(_cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def asnumpy(a):
    """Bring an array to host as a NumPy array (no-op on CPU)."""
    if ON_GPU:
        return _cp.asnumpy(a)
    return np.asarray(a)


def asarray(a):
    """Move/keep an array on the active backend device."""
    return xp.asarray(a)


def scatter_add(target, idx, vals):
    """In-place ``target[idx] += vals`` with duplicate indices summed.

    NumPy: ``np.add.at``.  CuPy: ``cupyx.scatter_add``.
    """
    if ON_GPU:
        import cupyx  # type: ignore

        cupyx.scatter_add(target, idx, vals)
    else:
        np.add.at(target, idx, vals)


def percentile(a, q, **kw):
    return xp.percentile(a, q, **kw)


def trapezoid(y, x=None, **kw):
    """Trapezoidal integration (``np.trapezoid`` / ``cp.trapz``)."""
    if ON_GPU:
        return _cp.trapz(y, x, **kw)
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(y, x, **kw)


def interp(x, xp_pts, fp):
    return xp.interp(x, xp_pts, fp)


@contextmanager
def device(i: int):
    """Pin GPU work to device ``i`` (no-op on the CPU backend)."""
    if ON_GPU:
        with _cp.cuda.Device(i):
            yield
    else:
        yield
