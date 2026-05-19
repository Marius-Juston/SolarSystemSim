"""
diagnostics_photosphere.py
--------------------------
Validation + diagnostics for the procedural photosphere against *real
measured* stellar/solar data (research/sun_render.md Phase 2-4).

For a panel of reference stars whose parameters are taken from the
literature -- the **Sun** (G2V, Teff 5772 K, B-V 0.66, IAU nominal;
white-light granulation RMS contrast ~0.10-0.20, mean granule ~1 Mm,
Hinode/DKIST), a **hot blue star** (B/A, Teff >~1e4 K, B-V ~ 0), and an
**M dwarf** (Teff ~3400 K, B-V ~1.5) -- this driver renders the disk,
zoomed quiet-granulation and sunspot crops, and computes diagnostic
metrics (continuum RMS intensity contrast, granule fill factor, the
granulation power spectrum / characteristic scale, the limb-darkening
profile vs the analytic law, disk-centre chromaticity).  It writes a
multi-panel figure and prints a measured-vs-rendered table with
PASS/FAIL so the foundation can be checked against ground truth.

Usage:  uv run python diagnostics_photosphere.py [res]   (res=dev|med)
Output: figures/photosphere/diagnostics/
"""

from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from goldilocks.photosphere import Photosphere  # noqa: E402
from goldilocks.starsurface import star_surface_for, limb_darkening  # noqa
from goldilocks.stellar import Star  # noqa: E402
import goldilocks.backend as B  # noqa: E402

_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_ROOT, "figures", "photosphere", "diagnostics")
os.makedirs(OUT, exist_ok=True)

# Reference stars: (label, mass, age_gyr, measured refs from literature).
# Sun: Teff 5772 K & B-V 0.656 (IAU 2015 / Cox 2000); granulation
# white-light RMS contrast ~0.10-0.20, granule ~1 Mm (Hinode, DKIST).
# Hot star ~ Vega/Regulus class (Teff 1e4-1.5e4, B-V ~ 0).  M dwarf ~
# M3-4V (Teff ~3300-3500 K, B-V ~1.5).
_STARS = [
    ("Sun (G2V)", 1.0, 4.6, dict(teff=5772.0, bv=0.66, rms=(0.05, 0.25))),
    ("Blue B/A star", 3.0, 0.2, dict(teff=10500.0, bv=0.0, rms=(0.0, 0.30))),
    ("M dwarf (M3V)", 0.3, 5.0, dict(teff=3400.0, bv=1.55, rms=(0.0, 0.35))),
]


def _disk_metrics(p: Photosphere, size: int = 520):
    """Return (disk image, quiet-patch metrics dict)."""
    d = p.disk_image(size)
    n = d.shape[0]
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / (0.5 * n)
    lum = d[..., 0] * 0.30 + d[..., 1] * 0.59 + d[..., 2] * 0.11
    # quiet disk-centre patch (mu ~ 1, exclude dark spots)
    quiet = lum[(rr < 0.32) & (lum > 0.55 * lum.max())]
    rms = float(quiet.std() / max(quiet.mean(), 1e-9))
    fill = float((quiet > quiet.mean()).mean())
    cen = d[c - 28:c + 28, c - 28:c + 28].reshape(-1, 3).mean(0)
    # limb-darkening radial profile (azimuth-averaged), normalised
    prof = []
    rgrid = np.linspace(0.02, 0.98, 40)
    for r0 in rgrid:
        m = (rr > r0 - 0.02) & (rr < r0 + 0.02) & (lum > 0)
        prof.append(lum[m].mean() if m.any() else np.nan)
    prof = np.array(prof)
    prof = prof / np.nanmax(prof)
    return d, dict(rms=rms, fill=fill, cen=cen, rgrid=rgrid, prof=prof)


def _power_spectrum(p: Photosphere):
    """Radially-averaged power spectrum of the granulation scalar; the
    peak wavenumber should track the physical granule freq."""
    s = B.asnumpy(p.scalar())
    s = s - s.mean()
    F = np.abs(np.fft.rfft2(s)) ** 2
    ky = np.fft.fftfreq(s.shape[0]) * s.shape[0]
    kx = np.fft.rfftfreq(s.shape[1]) * s.shape[1]
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    kbin = np.arange(1, min(s.shape) // 2)
    psd = np.array([F[(K >= k - 0.5) & (K < k + 0.5)].mean()
                    for k in kbin])
    peak = float(kbin[1 + np.argmax(psd[1:])])
    return kbin, psd, peak


def main() -> None:
    res = sys.argv[1] if len(sys.argv) > 1 else "med"
    rows = len(_STARS)
    fig, ax = plt.subplots(rows, 4, figsize=(18, 4.4 * rows))
    print(f"\n{'star':16s} {'Teff_meas':>9s} {'Teff_ren':>9s} "
          f"{'BV_meas':>7s} {'BV_ren':>7s} {'RMS':>6s} {'fill':>5s} "
          f"{'peak_k':>7s}/{'freq':>5s}  centreRGB        verdict")
    print("-" * 104)
    for i, (lab, m, age, ref) in enumerate(_STARS):
        star = Star(lab, mass=m)
        surf = star_surface_for(star, age_gyr=age)
        p = Photosphere(surf, res=res, seed=7, backend="reference")
        for _ in range(22):
            p.step(0.1)
        d, mt = _disk_metrics(p)
        kbin, psd, peak = _power_spectrum(p)
        teff_r = surf.teff
        bv_r = None
        # B-V from the L0 state behind the surface (Ballesteros).
        from goldilocks.stellar_state import bv_from_teff
        bv_r = bv_from_teff(teff_r)
        ok_t = abs(teff_r - ref["teff"]) < max(0.06 * ref["teff"], 150)
        ok_c = (ref["rms"][0] <= mt["rms"] <= ref["rms"][1])
        verdict = "PASS" if (ok_t and ok_c) else "CHECK"
        print(f"{lab:16s} {ref['teff']:9.0f} {teff_r:9.0f} "
              f"{ref['bv']:7.2f} {bv_r:7.2f} {mt['rms']:6.3f} "
              f"{mt['fill']:5.2f} {peak:7.1f}/{p.freq:5.1f}  "
              f"({int(mt['cen'][0])},{int(mt['cen'][1])},"
              f"{int(mt['cen'][2])})   {verdict}")

        ax[i, 0].imshow(d)
        ax[i, 0].set_title(f"{lab}\nTeff {teff_r:.0f} K  B-V {bv_r:+.2f}")
        ax[i, 0].axis("off")
        # quiet zoom (centre crop)
        n = d.shape[0]
        z = d[n // 2 - 70:n // 2 + 70, n // 2 - 70:n // 2 + 70]
        ax[i, 1].imshow(z)
        ax[i, 1].set_title(f"quiet zoom  RMS={mt['rms']:.3f} "
                           f"fill={mt['fill']:.2f}")
        ax[i, 1].axis("off")
        lumq = (d[..., 0] * .3 + d[..., 1] * .59 + d[..., 2] * .11)
        ax[i, 2].hist(lumq[lumq > 0].ravel(), bins=60, color="darkorange")
        ax[i, 2].set_title("intensity histogram")
        ax[i, 2].set_xlabel("luminance")
        ax[i, 3].loglog(kbin, psd, "b-")
        ax[i, 3].axvline(p.freq, color="g", ls="--",
                         label=f"granule k={p.freq:.0f}")
        ax[i, 3].axvline(peak, color="r", ls=":",
                         label=f"PSD peak={peak:.0f}")
        ax[i, 3].set_title("granulation power spectrum")
        ax[i, 3].set_xlabel("wavenumber")
        ax[i, 3].legend(fontsize=8)
    fig.tight_layout()
    f1 = os.path.join(OUT, "stellar_validation_panel.png")
    fig.savefig(f1, dpi=85)
    plt.close(fig)

    # Sun: spot vs no-spot zoom + limb-darkening vs analytic law.
    sun = Photosphere.for_star_seed(1.0, 7, res=res, backend="reference")
    for _ in range(22):
        sun.step(0.1)
    dd, ms = _disk_metrics(sun)
    surf_ns = star_surface_for(Star("Sun-nospot", mass=1.0), age_gyr=4.6)
    surf_ns.spot_coverage = 0.0
    nos = Photosphere(surf_ns, res=res, seed=7, backend="reference")
    for _ in range(22):
        nos.step(0.1)
    dn, _ = _disk_metrics(nos)
    fig2, a2 = plt.subplots(1, 3, figsize=(16, 5))
    g = dd.sum(2).astype(float)
    nn = dd.shape[0]
    bs = 150
    best, bv = (0, 0), 9e18
    for y0 in range(0, nn - bs, 30):
        for x0 in range(0, nn - bs, 30):
            bl = g[y0:y0 + bs, x0:x0 + bs]
            if (bl > 0).mean() > 0.9 and (bl < 150).mean() > 0.04:
                if bl.mean() < bv:
                    bv, best = bl.mean(), (y0, x0)
    y0, x0 = best
    a2[0].imshow(dd[y0:y0 + bs, x0:x0 + bs])
    a2[0].set_title("Sun: sunspot zoom\n(umbra + umbral dots + filaments)")
    a2[0].axis("off")
    a2[1].imshow(dn[nn // 2 - 75:nn // 2 + 75, nn // 2 - 75:nn // 2 + 75])
    a2[1].set_title("Sun: spotless quiet granulation")
    a2[1].axis("off")
    mu = np.clip(np.cos(np.arcsin(np.clip(ms["rgrid"], 0, 1))), 0, 1)
    ana = B.asnumpy(limb_darkening(B.asarray(mu),
                                   sun.surface.ld_u1, sun.surface.ld_u2))
    a2[2].plot(ms["rgrid"], ms["prof"], "k-", label="rendered")
    a2[2].plot(ms["rgrid"], ana / np.nanmax(ana), "r--",
               label="quadratic LD law")
    a2[2].set_title("limb-darkening profile vs analytic")
    a2[2].set_xlabel("r / R")
    a2[2].set_ylabel("I / I(centre)")
    a2[2].legend()
    fig2.tight_layout()
    f2 = os.path.join(OUT, "sun_spot_limb_diagnostics.png")
    fig2.savefig(f2, dpi=90)
    plt.close(fig2)

    print(f"\nfigures -> {f1}\n           {f2}")


if __name__ == "__main__":
    main()
