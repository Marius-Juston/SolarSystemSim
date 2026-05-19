"""
chromosphere.py
---------------
Offline, physically-grounded chromosphere / transition-region overlay
(research/sun_render.md Phase 4).

The project renders offline (no GL tessellation; the established §2.12
substitution), so the chromosphere is a procedural shell composited onto
the disk image rather than displaced mesh vertices.  Every quantity is
tied to a physical property:

* shell thickness  = `StellarState.chromosphere_thickness_rel`
  (~7 pressure scale heights H = k_B T/(mu m_H g); Sun ~2000 km ~0.3%R).
* limb brightening = optically-thin emission `E ∝ (1-mu)^p`, p≈3, since
  the line-of-sight chord through a thin shell grows ∝ 1/mu toward the
  limb (so the rim glows — the classic chromospheric "ring").
* spicule fringe   = sparse jets gated by the Rossby chromospheric
  activity proxy (Mamajek & Hillenbrand 2008), log-normal heights
  (type I 5–10 Mm / type II 4–5 Mm), per-spicule random phase.
* inverse Evershed = chromospheric *inflow* toward spots, opposite the
  photospheric Evershed outflow (~3–5 km/s; Beck 2018 / ApJ 2018 859).
* emission lines   = H-alpha 656.3 nm, Ca II K 393.4 nm via the existing
  `skyview` CIE pipeline; He II 30.4 nm / Fe IX 17.1 nm as documented
  EUV false colours.

Selected by line key; ``line=None`` returns the image untouched, so the
default bolometric Sun render (and every pinned value) is unchanged.
"""

from __future__ import annotations

import math

import numpy as np

from goldilocks import noise as N

_P_LIMB = 3.0  # optically-thin limb-brightening exponent
_SHELL_VIS = 12.0  # visualisation magnification of the physical
#                        thickness fraction (documented; the true ~0.3%
#                        shell is sub-pixel at disk scale)
_SPIC_K = 90.0  # azimuthal spicule wavenumber (fine fringe)

# Type I / II spicule heights [Mm] -> log-normal (mu, sigma) of ln h.
_SPIC_LOGN = {"typeI": (math.log(7.0), 0.35),
              "typeII": (math.log(4.5), 0.30)}


def emission_line_rgb(line: str) -> np.ndarray:
    """Linear RGB (0..1) chromaticity of a chromospheric emission line."""
    line = line.lower()
    if line in ("heii304", "he304"):
        return np.array([1.00, 0.78, 0.20])  # 30.4 nm EUV false gold
    if line in ("feix171", "fe171"):
        return np.array([0.20, 0.85, 0.80])  # 17.1 nm EUV false teal
    lam0 = {"halpha": 656.28, "ha": 656.28,
            "caiik": 393.37, "cak": 393.37}.get(line)
    if lam0 is None:
        raise ValueError(f"unknown emission line {line!r}")
    from goldilocks.skyview import (lambda_grid_nm, cie_xyz_bar,
                                    _XYZ_TO_RGB)
    lam = lambda_grid_nm()
    spec = np.exp(-0.5 * ((lam - lam0) / 3.0) ** 2)  # narrow line
    xb, yb, zb = cie_xyz_bar(lam)
    dl = float(lam[1] - lam[0])
    xyz = np.array([np.dot(spec, xb), np.dot(spec, yb),
                    np.dot(spec, zb)]) * dl
    rgb = _XYZ_TO_RGB @ xyz
    rgb = np.clip(rgb, 0.0, None)
    return rgb / (rgb.max() + 1e-12)


def inverse_evershed_kms(surface) -> float:
    """Chromospheric inflow speed toward spots (negative = inward).

    Opposite sign and ~0.8x the photospheric Evershed outflow
    (`surface.evershed_kms`), matching observed superpenumbral inflow
    of a few km/s (ApJ 2018, 859, 139)."""
    return -0.8 * float(getattr(surface, "evershed_kms", 4.0))


def chromosphere_overlay(rgb: np.ndarray, X: np.ndarray, Y: np.ndarray,
                         disk: np.ndarray, mu: np.ndarray, surface, *,
                         line: str, seed: int = 7) -> np.ndarray:
    """Composite the chromospheric shell + spicule fringe onto `rgb`.

    `rgb` is (H, W, 3) uint8; returns a new uint8 image.  No-op when
    ``line`` is falsy (default bolometric render stays byte-identical).
    """
    if not line:
        return rgb
    col = emission_line_rgb(line)  # (3,) linear 0..1
    thick = float(np.clip(getattr(surface, "chromo_thickness_rel",
                                  0.003) * _SHELL_VIS, 0.012, 0.10))
    act = float(np.clip(getattr(surface, "chromo_activity", 0.2),
                        0.0, 1.0))

    r = np.sqrt(X ** 2 + Y ** 2)
    az = np.arctan2(Y, X)

    # 1) optically-thin limb brightening on the disk: E ∝ (1-mu)^p
    e_disk = np.where(disk, (1.0 - np.clip(mu, 0.0, 1.0)) ** _P_LIMB, 0.0)

    # 2) off-limb shell: density ~ exp(-(r-1)/thickness) (scale height)
    shell = (~disk) & (r < 1.0 + thick)
    e_ring = np.where(shell, np.exp(-np.clip((r - 1.0) / thick, 0.0, 30.0)),
                      0.0)

    # 3) spicule fringe: sparse jets at the limb, gated by activity,
    #    log-normal heights, per-spicule random phase.  Inverse-Evershed
    #    inflow biases the azimuthal sampling inward (toward the disk).
    rng = np.random.default_rng(seed + 41)
    phase = float(rng.uniform(0.0, 1000.0))
    iev = inverse_evershed_kms(surface)  # < 0 (inflow)
    inflow_bias = 0.04 * (iev / -4.0)  # small inward shift
    s = N.value_noise_2d(az * _SPIC_K + phase,
                         np.full_like(az, 0.37), seed=seed + 41)
    s = 0.5 * (np.asarray(s) + 1.0)  # -> [0,1]
    present = s > (1.0 - 0.85 * act)  # more jets if active
    mu_h, sig_h = _SPIC_LOGN["typeII" if act > 0.5 else "typeI"]
    h_ln = np.exp(mu_h + sig_h * (2.0 * s - 1.0))
    reach = thick * (0.4 + 0.6 * h_ln / math.exp(mu_h + sig_h))
    spic = (~disk) & present & (
            r < 1.0 + reach * (1.0 - inflow_bias))
    e_spic = np.where(spic, 0.7 * np.exp(
        -np.clip((r - 1.0) / (reach + 1e-6), 0.0, 30.0)), 0.0)

    emis = (0.55 * e_disk + e_ring + 0.8 * e_spic) * (0.4 + 0.9 * act)
    add = (emis[..., None] * col[None, None, :] * 255.0)
    out = np.clip(rgb.astype(np.float64) + add, 0, 255).astype(np.uint8)
    return out
