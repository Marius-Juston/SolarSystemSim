"""
parallel.py
-----------
Work distribution + fast video encoding for the rendering drivers.

Two execution shapes:
  * **GPU**  -- one worker process per CUDA device, each pinned to its
    own GPU via ``CUDA_VISIBLE_DEVICES`` (set before CuPy is imported in
    the child, so a ``spawn`` start method is used).
  * **CPU**  -- a process pool sized to the machine's cores.

Frames are always returned in submission order so an MP4 stays correct.
``encode_frames`` streams raw RGB straight into one ``ffmpeg`` process,
replacing the slow serial ``matplotlib.animation.FuncAnimation`` path.

Env:
  * ``GOLDILOCKS_SERIAL=1``      -- force in-process serial map (debug).
  * ``GOLDILOCKS_MAX_WORKERS=N`` -- cap the CPU pool size.
"""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, Iterator, List, Sequence

import numpy as np

from goldilocks import backend as B

_SERIAL = os.environ.get("GOLDILOCKS_SERIAL", "").strip() in ("1", "true")
_CPU_CAP = 64  # plenty of frame parallelism without oversubscription


def n_workers() -> int:
    """Worker count for the active backend."""
    if _SERIAL:
        return 1
    if B.ON_GPU:
        return max(1, B.n_gpus())
    env = os.environ.get("GOLDILOCKS_MAX_WORKERS")
    cap = int(env) if env else _CPU_CAP
    return max(1, min(os.cpu_count() or 1, cap))


def _gpu_init(rank_env: str) -> None:
    # Each worker pins itself to a single GPU.  Imported lazily so the
    # child re-evaluates the backend with exactly one visible device.
    import multiprocessing

    name = multiprocessing.current_process().name
    try:
        rank = int(name.rsplit("-", 1)[-1]) - 1
    except ValueError:
        rank = 0
    ndev = len(rank_env.split(",")) if rank_env else 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % max(ndev, 1))


def map_ordered(fn: Callable, args: Sequence) -> List:
    """Apply ``fn`` to every item of ``args``, results in input order."""
    args = list(args)
    nw = n_workers()
    if nw <= 1 or len(args) <= 1:
        return [fn(a) for a in args]

    if B.ON_GPU:
        import multiprocessing as mp

        vis = os.environ.get("CUDA_VISIBLE_DEVICES",
                             ",".join(str(i) for i in range(B.n_gpus())))
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=nw, mp_context=ctx,
                                 initializer=_gpu_init,
                                 initargs=(vis,)) as ex:
            return list(ex.map(fn, args))

    with ProcessPoolExecutor(max_workers=nw) as ex:
        return list(ex.map(fn, args))


# Back-compat alias used by the drivers.
map_frames = map_ordered


def encode_frames(path: str, frames: Iterable[np.ndarray], fps: int,
                  *, bitrate: int = 2600, codec: str = "libx264") -> None:
    """Stream uint8 HxWx3 RGB frames into one ffmpeg process.

    Raises ``FileNotFoundError`` if ffmpeg is missing -- callers handle
    the skip exactly as the old ``FuncAnimation`` path did.
    """
    it: Iterator[np.ndarray] = iter(frames)
    try:
        first = next(it)
    except StopIteration:
        return
    h, w = first.shape[:2]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", codec, "-b:v", f"{bitrate}k",
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        proc.stdin.write(np.ascontiguousarray(first, np.uint8).tobytes())
        for fr in it:
            proc.stdin.write(np.ascontiguousarray(fr, np.uint8).tobytes())
        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited with code {rc}")
    finally:
        if proc.poll() is None:
            proc.kill()
