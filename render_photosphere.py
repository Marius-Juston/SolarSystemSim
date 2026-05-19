"""
render_photosphere.py
----------------------
Standalone driver for the stateful equirectangular photosphere
(`goldilocks.photosphere`, research/sun_render.md Phase 2).

Evolves the curl-noise / semi-Lagrangian field for a star, writes an
equirectangular map PNG + an orthographic limb-darkened disk PNG, and
streams the boiling sequence to an MP4 via `parallel.encode_frames`
(skipped cleanly if ffmpeg is absent, like the other drivers).

Usage:  uv run python render_photosphere.py [seed] [mass_msun] [res]

Backend is chosen by GOLDILOCKS_PHOTOSPHERE_BACKEND=auto|warp|reference
(default auto: Warp if importable -- CPU or CUDA -- else NumPy/CuPy).

Output: figures/photosphere/  and  animations/photosphere/  (repo-rel).
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from goldilocks.photosphere import Photosphere  # noqa: E402
from goldilocks.parallel import encode_frames  # noqa: E402

_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_FIG = os.path.join(_ROOT, "figures", "photosphere")
OUT_ANI = os.path.join(_ROOT, "animations", "photosphere")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_ANI, exist_ok=True)


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    mass = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    res = sys.argv[3] if len(sys.argv) > 3 else "dev"
    line = sys.argv[4] if len(sys.argv) > 4 else None  # halpha|caiik|...

    p = Photosphere.for_star_seed(mass, seed, res=res)
    print(p.surface.summary)
    print(p.memory_report())

    # spin up so the field develops structure before the snapshot
    for _ in range(24):
        p.step(0.1)

    eq = p.to_srgb()
    disk = p.disk_image(640, emission_line=line)

    eq_png = os.path.join(OUT_FIG, f"photosphere_{seed}_equirect.png")
    dk_png = os.path.join(OUT_FIG, f"photosphere_{seed}_disk.png")
    plt.imsave(eq_png, eq)
    plt.imsave(dk_png, disk)
    print(f"figures    -> {eq_png}")
    print(f"              {dk_png}")

    # MP4: rotate the sub-observer point while the surface keeps boiling
    def frames():
        n = 90
        for f in range(n):
            p.step(0.1)
            yield p.disk_image(512, sub_lon=2.0 * 3.14159265 * f / n,
                               emission_line=line)

    mp4 = os.path.join(OUT_ANI, f"photosphere_{seed}.mp4")
    try:
        encode_frames(mp4, frames(), fps=30)
        print(f"animation  -> {mp4}")
    except FileNotFoundError:
        print("ffmpeg not found on PATH -- skipped MP4 (PNGs written).")


if __name__ == "__main__":
    main()
