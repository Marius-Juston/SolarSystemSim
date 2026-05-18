# Procedural Stellar Surface Simulator — Detailed Implementation Plan

This plan maps the astrophysical models in the source document onto a concrete, parallel, Python-driven architecture
that targets multi-GPU compute, large CPU core counts, and equation-level accuracy.

> **Implementation notes (as-built, Increments 1–2).** Where the realised
> code refines this plan:
> - **L0 (Phase 1)** lives in `goldilocks/stellar_state.py` inside the existing
>   `goldilocks/` package (not a new `stellarsim/`). Mass→L/R/Teff reuse the
>   project's calibrated **Eker 2018** relations (`stellar.py`) instead of the
>   crude piecewise power laws of §3 — smoother and already sanity-pinned.
> - **L2 photosphere (Phase 2)** is `goldilocks/photosphere.py` with **two
>   interchangeable step backends** behind one interface
>   (`GOLDILOCKS_PHOTOSPHERE_BACKEND=auto|warp|reference`): an **NVIDIA Warp**
>   `@wp.kernel` using the built-in divergence-free `wp.curlnoise`, and a
>   dependency-free NumPy/CuPy seam. Key correction to the §2 assumption: Warp
>   JIT-compiles the *same* kernel to **CPU *or* CUDA**, so the GPU path is
>   fully verifiable on a CPU-only machine and is simultaneously the efficient
>   path on the 4×A6000 box. The reference seam uses the project's seeded
>   value-noise lattice, so the backends agree *statistically*, not
>   bit-for-bit (verified in `test_sanity.py` section-9).
> - **Single-GPU now.** The Warp kernel is `device`-parametrised and the
>   texture is authored in latitude bands so the §5.2 **4-GPU strip+halo**
>   partition drops in next with no API change.
> - §2.12's live moderngl viewer is deferred; the standalone driver
>   (`render_photosphere.py`) emits offline equirect+disk PNGs and an MP4 via
>   `parallel.encode_frames` (the established driver pattern).
> - **Increment 3** closed the genuine Phase 1–2 gaps before scaling:
>   `T_to_spectral_class`/`spectral_class`, `StellarState.__post_init__`
>   validation, an explicit pre-100-Myr saturated-rotation plateau, and —
>   correcting a defect where §2.11 was ticked but absent — a real Dravins
>   convective blueshift in the colour path, a dedicated `0.05×` low-freq
>   sunspot channel, and a physically-grounded (`R*/1 Mm`) grid-clamped
>   granule wavenumber with an `Ro^-1/2` granule-lifetime rate. Solar
>   behaviour and every pinned sanity value are preserved.

---

## 1. System Architecture Overview

The system is organized as a four-layer pipeline. Each layer has a different update frequency and a different parallel
decomposition.

| Layer                        | Update Rate                  | Compute Target    | Role                                                                                               |
|------------------------------|------------------------------|-------------------|----------------------------------------------------------------------------------------------------|
| L0 — Global Astrophysics     | ~1 Hz or on parameter change | CPU (NumPy/SciPy) | HR position, $L$, $T_{\text{eff}}$, $\tau_c$, $\text{Ro}$, $\beta$, tidal state                    |
| L1 — Magnetic / Topology     | 1–10 Hz                      | CPU + 1 GPU       | Active-region map, polarity inversion lines (PILs), QSL detection, flux-rope state                 |
| L2 — Fluid / Field Evolution | per-frame (30–120 Hz)        | Multi-GPU         | Curl-noise advection (photosphere), K-S prominence state, coronal density, post-reconnection loops |
| L3 — Rendering               | per-frame                    | Multi-GPU         | Volumetric raymarching, geometry shaders, fragment passes, compositing                             |

The Python process is the **orchestrator**: it owns parameters, schedules GPU work via streams, performs L0/L1
computation, and dispatches L2/L3 kernels. Hot loops never run in pure Python; they live in CUDA / Vulkan / GLSL.

---

## 2. Technology Stack

**Recommended primary stack (NVIDIA-centric, mature, multi-GPU capable):**

- **Python 3.12** as the orchestration language.
- **CuPy ≥ 13** for high-level GPU array math, FFTs (for spectral noise), and reductions. Provides `RawKernel` and
  `RawModule` for hand-written CUDA.
- **NVIDIA Warp** (`warp-lang`) for differentiable, Pythonic GPU kernels with built-in vector/SDF/curl-noise primitives.
  Use Warp for the curl-noise, SDF, and raymarching kernels because its `wp.vec3`/`wp.mat33` types and `wp.noise` family
  map directly onto the equations in §3.2, §5, §6.
- **Numba** (`@njit(parallel=True)`) for CPU-side parallel work on large core counts — magnetic topology
  pre-computation, QSL search, and parameter sweeps.
- **mpi4py + NCCL (via `cupy.cuda.nccl`)** for multi-GPU collectives (halo exchange, allreduce on global energy/mass
  conservation).
- **moderngl** (preferred) or **PyVulkan / vulkan** for the rasterization pass, geometry shaders (post-reconnection
  flare loops), and final compositing. moderngl exposes OpenGL 4.6 including compute, geometry, and tessellation
  shaders.
- **CUDA-GL / CUDA-Vk interop** via `pycuda.gl` or `cuda.cooperative` for zero-copy handoff of compute results into
  rendering textures.
- **Astropy** for unit handling and constants (`astropy.units`, `astropy.constants`) — critical for equation fidelity (
  avoids silent unit errors in $L \propto M^{3.5}$, Skumanich, von Zeipel).
- **xarray + Zarr** for checkpointing simulation state to disk in a chunked, multi-GPU-friendly format.
- **Dask + Dask-CUDA** for distributing the L0/L1 parameter sweeps across a workstation or small cluster.

**Alternative if AMD/Intel GPUs are involved:** replace CuPy/Warp with **Taichi** (`ti.cuda`, `ti.vulkan`, `ti.opengl`
backends). Taichi has weaker multi-GPU support, so use only if single-node single-vendor isn't required.

---

## 3. Data Model and Parameter System (Layer L0)

A single dataclass-like object owns the global stellar state. All downstream layers read from it.

```python
@dataclass
class StellarState:
    M: Quantity  # solar masses
    age: Quantity  # Gyr
    is_binary: bool
    companion_mass: Quantity | None
    orbital_period: Quantity | None
    # Derived (computed in update()):
    L: Quantity  # bolometric luminosity
    T_eff: Quantity  # K
    R: Quantity  # stellar radius
    P_rot: Quantity  # rotation period
    tau_c: Quantity  # convective turnover time
    Ro: float  # Rossby number
    beta: float  # gravity-darkening exponent
    omega_eq: float  # equatorial angular velocity (normalized)
    tidally_locked: bool
```

**Derivation order (CPU, ~microseconds):**

1. **Mass–Luminosity:** $L/L_\odot = (M/M_\odot)^{3.5}$ for $0.43 < M/M_\odot < 2$; piecewise power laws (exponents 2.3,
   4.0, 3.5, 1.0 by mass regime) per Salaris & Cassisi to honor the document's note that the exponent varies.
2. **Mass–Radius:** $R/R_\odot \approx (M/M_\odot)^{0.8}$ (lower MS) or $(M/M_\odot)^{0.57}$ (upper MS).
3. **Effective temperature:** $T_{\text{eff}} = (L / 4\pi R^2 \sigma)^{1/4}$ (Stefan–Boltzmann, exact).
4. **MS lifetime:** $t_{\text{MS}} = 10^{10} (M/M_\odot)^{-2.5}$ yr.
5. **Skumanich spin-down:** $v \sin i \propto t^{-1/2}$; convert to $P_{\text{rot}}$ via $P_{\text{rot}} = 2\pi R / v$.
6. **Convective turnover** (Noyes 1984): $\log \tau_c = 1.362 - 0.166x + 0.025x^2 - 5.323x^3$
   for $x = 1 - (B-V)$, $x > 0$; else $\log \tau_c = 1.362 - 0.14x$. $(B-V)$ from $T_{\text{eff}}$ via Ballesteros'
   formula.
7. **Rossby number:** $\text{Ro} = P_{\text{rot}} / \tau_c$.
8. **Gravity-darkening exponent:** $\beta = 0.25$ (radiative envelope, $M > 1.5\,M_\odot$) or $\beta \approx 0.08$ (
   convective, Lucy's law), with a smooth blend across the transition.
9. **Tidal lock check:** if binary and $t_{\text{lock}} < $ age (Hut 1981 timescale),
   set $P_{\text{rot}} = P_{\text{orb}}$ and flag locked.

Validation hooks: each derived quantity is checked against the Sun's known values
when $M = 1\,M_\odot$, $\text{age} = 4.6$ Gyr (regression test enforcing $|\text{error}| < 1\%$).

---

## 4. Physics Modules and Kernel Design

### 4.1 Photosphere — Granulation via Divergence-Free Curl Noise (L2)

**Mathematics:** On the sphere, generate a scalar potential $\psi(\mathbf{x}, t)$ as 3D Perlin or Simplex noise sampled
on the surface. The surface-tangent curl is

$$\mathbf{v}(\mathbf{x}, t) = \nabla \psi(\mathbf{x}, t) \times \hat{\mathbf{n}}(\mathbf{x})$$

producing a divergence-free flow field. Compute the gradient analytically via finite differences in the kernel:

$$\frac{\partial \psi}{\partial x_i} \approx \frac{\psi(\mathbf{x} + \epsilon \hat{\mathbf{e}}_i) - \psi(\mathbf{x} - \epsilon \hat{\mathbf{e}}_i)}{2\epsilon}$$

**Kernel:** Warp kernel `granulation_advect`, one thread per texel of the equirectangular photosphere texture (typically
8192×4096, ~33 M texels).

- Inputs: previous-frame density/temperature texture, time, $\text{Ro}$.
- Computes $\mathbf{v}$ from `wp.noise` (octaved Perlin, 4 octaves, base frequency tuned so cell size ≈ 1 Mm in stellar
  units).
- Semi-Lagrangian advection: trace back $\mathbf{x}' = \mathbf{x} - \mathbf{v}\,\Delta t$, sample bilinearly.
- Writes new texture (double-buffered ping-pong).
- Granule turnover frequency scaled by $\text{Ro}^{-1/2}$ so high-activity stars boil faster.

**Sunspots overlay:** A separate low-frequency Simplex mask $S(\mathbf{x})$ thresholded against $\text{Ro}$-dependent
value. Where $S > \theta(\text{Ro})$, emission is reduced by
ratio $(T_{\text{umbra}}/T_{\text{phot}})^4 \approx (4100/5778)^4 \approx 0.25$.

**Evershed effect:** Within the penumbra annulus (computed by SDF distance from spot centroid), warp the curl-noise
sample coordinates radially outward by velocity $v_E \approx 4$ km/s, integrated over frame time. Chromospheric
inverse-Evershed flow handled in the chromosphere pass with opposite sign.

**Active longitudes (binary case):** Multiply $S$ by $1 + A\cos(2(\phi - \phi_{\text{sub}}))$ where $\phi_{\text{sub}}$
is the sub-stellar longitude and $A \approx 0.5$.

### 4.2 Gravity Darkening (Vertex / Fragment Pass, L3)

Implemented in the rendering shader. For each surface vertex with colatitude $\theta$:

$$g_{\text{eff}}(\theta) = \sqrt{(g_{\text{grav}} - \omega^2 r \sin^2\theta)^2 + (\omega^2 r \sin\theta \cos\theta)^2}$$

then

$$T_{\text{local}} = T_{\text{eff}} \left(\frac{g_{\text{eff}}}{\bar{g}}\right)^{\beta}, \quad I_{\text{local}} \propto T_{\text{local}}^4$$

Geometric oblateness: vertex displacement along the radial direction by Roche-equipotential surface (Maeder 1999 form).
For non-rotating stars this is a no-op; for fast rotators (Be stars, contact binaries), the equator can bulge by 30%+.

### 4.3 Chromosphere and Transition Region (L3)

- **Spicules:** Tessellation shader subdivides the surface mesh; vertex displacement
  by $h(\mathbf{x}, t) = h_0 \cdot \text{noise}(k\mathbf{x}, t)$ along the local magnetic field direction (from the L1
  topology field). Heights vary 5–10 Mm.
- **Emission:** Fresnel-weighted rim using power-law $F = (1 - \hat{\mathbf{n}} \cdot \hat{\mathbf{v}})^p$. Color from a
  2D LUT keyed by (line, $T$) — H-α (6563 Å) for filament-disk view, Fe IX 171 Å for hot loop view, He II 304 Å for
  transition region.
- **Transition region:** Acts as the emissive boundary for the raymarcher (§4.4). Implemented as the raymarching "near
  plane" at $r = 1.003\,R_\star$.

### 4.4 Corona — Volumetric Raymarching (L3, primary GPU cost)

**Algorithm:** Per output pixel, march from the camera ray's entry to the bounding sphere of $r = 5\,R_\star$ in $N$
steps. At each step $\mathbf{p}_i$:

1. Sample magnetic field $\mathbf{B}(\mathbf{p}_i)$ from a precomputed potential extrapolation (PFSS, §4.6).
2. Compute density $n_e(\mathbf{p}_i)$ as $n_e \propto |\mathbf{B}|^{1.2}$ times a curl-noise modulation aligned
   with $\mathbf{B}$ (so plasma streaks along field lines).
3. Compute temperature $T(\mathbf{p}_i)$ from hydrostatic equilibrium along the field line.
4. Emissivity: $\epsilon = n_e^2 \cdot G(T, \lambda)$ where $G$ is the contribution function for the chosen EUV line (
   precomputed as a 1D texture from CHIANTI).
5. Accumulate: $I_{\text{out}} += \epsilon \cdot \Delta s \cdot \text{transmittance}$.

**Optimizations:**

- **SDF-based step size:** Maintain an SDF for the corona's "high-density" loop systems. Step
  size $\Delta s = \max(\Delta s_{\min}, \text{SDF}(\mathbf{p}))$ so empty regions are skipped.
- **Importance sampling:** Adaptive step density based on $n_e$ gradient — denser sampling near loops, coarse in voids.
- **Half-resolution + temporal reprojection:** Raymarch at half-res, reproject previous frame, refine only where motion
  vectors say the reprojection is stale.
- **Persistent threads:** One persistent CUDA block per tile (16×16 pixels) processing many rays via work-stealing
  queue; reduces launch overhead.

### 4.5 Prominences — K-S Model + 3D Curl Noise (L2/L3)

**Equilibrium structure:** For each prominence, the K-S analytical profile defines:

- $B_x(z) = B_\infty \tanh(z/H)$ (transverse field)
- $\rho(z) = \rho_0 \,\text{sech}^2(z/H)$ (density)

with scale height $H = 2k_B T / (m_H g \tan\phi_0)$ where $\phi_0$ is the magnetic shear angle. These are evaluated *
*once at instantiation** in NumPy and uploaded as a 3D texture per prominence.

**SDF macrostructure:** Each prominence is represented by a thin curved sheet SDF — a swept curve along the PIL with a
thin "hammock" cross-section. Composed via smooth-min into the global coronal SDF.

**Dynamics:** A 3D divergence-free curl noise (vector potential $\mathbf{A}$, $\mathbf{v} = \nabla \times \mathbf{A}$)
advects density inside the sheet. MRT instability is seeded by amplifying noise modes with $k > k_c$ where $k_c$ is the
magnetic-tension cutoff $k_c = \sqrt{g \rho / B^2}$ (linear MRT dispersion). This produces the characteristic
falling-plume morphology without a full MHD solve.

**Ambipolar thickening:** The current-sheet thickness scales as $\delta(t) = \delta_0 (1 + t/\tau_A)^{1/2}$
with $\tau_A$ derived from Cowling resistivity. Implemented as a per-prominence scalar updated on the CPU.

### 4.6 Magnetic Field Extrapolation (L1, CPU + 1 GPU)

The simulation needs a 3D magnetic field for both the corona raymarcher and the prominence/flare topology. Use a *
*PFSS (Potential Field Source Surface) extrapolation** from the surface flux distribution:

- Surface $B_r(\theta, \phi)$ is the L1 active-region map (from §4.1 sunspot mask, signed by polarity from a separate
  noise channel).
- Decompose into spherical harmonics up to $\ell_{\max} = 64$ via SHTns (Python bindings exist).
- Reconstruct $\mathbf{B}$ on a 3D spherical grid out to $r_{\text{ss}} = 2.5\,R_\star$.
- This grid is the lookup texture for the raymarcher.

Recomputed every 1–10 frames depending on how fast spots evolve. SHTns is OpenMP-parallel — utilizes large CPU core
counts.

### 4.7 Flares and Reconnection (L1/L2/L3)

**Trigger detection (CPU, Numba parallel):** Scan the surface magnetic map for QSLs. Compute the squashing factor $Q$ on
a $1024 \times 2048$ grid:

$$Q = \frac{\partial X^+ / \partial x^- \cdot \partial Y^+ / \partial y^- - \partial X^+ / \partial y^- \cdot \partial Y^+ / \partial x^-}{B_n^+ / B_n^-}$$

Trigger flare when $\log Q > 4$ co-located with high magnetic shear gradient. Threshold modulated by $\text{Ro}^{-2}$ to
honor the activity scaling.

**Flare ribbon rendering:** When triggered, the QSL footprint becomes an emission stencil on the photosphere. Convolve
with a moving "reconnection front" (J-shape or circular) using a 2D distance transform. Bloom + chromatic aberration in
post.

**Post-reconnection loops (PRFLs):** Generated by a **geometry shader** in moderngl. Input: pairs of QSL footpoints (
positive/negative). Geometry shader emits a Bezier tube between them with:

- Shear angle $\alpha(t)$ decaying from $\alpha_0 = 70°$ to $\alpha_\infty = 10°$
  via $\alpha(t) = \alpha_\infty + (\alpha_0 - \alpha_\infty) e^{-t/\tau_s}$ — encodes strong-to-weak shear evolution.
- Apex height grows with $\sqrt{t}$ matching standard reconnection rate.
- Texture: 1D scrolling noise representing chromospheric evaporation upflow.

---

## 5. Multi-GPU Strategy

Three parallelization axes are exploited simultaneously.

### 5.1 Volumetric Raymarching — Screen-Space Tile Partition

Split the output image into $N_{\text{GPU}}$ vertical strips with small overlap (2 pixels for filtering). Each GPU
raymarchers its strip independently — **embarrassingly parallel, zero communication** during the pass. Final composite:
each GPU's strip is copied (peer-to-peer via NVLink, or via NCCL `ncclAllGather`) onto GPU 0, which performs the final
tonemap and presents.

This is the dominant cost (~70% of frame time) and scales near-linearly to 4–8 GPUs.

### 5.2 Surface Texture Pass — Patch Partition with Halos

Split the equirectangular surface texture into latitude bands per GPU. Curl-noise advection requires a 2-texel halo (for
the gradient stencil). Each frame:

1. Compute interior on each GPU.
2. Exchange halos via `ncclSendRecv` (small, ~MB-scale).
3. Continue.

Sunspot mask and PFSS lookup are replicated on all GPUs (small, ~100 MB).

### 5.3 PFSS / Magnetic Topology — Replicated Compute

PFSS reconstruction is fast enough (~10 ms on one GPU) that we run it on GPU 0 and broadcast the resulting
3D $\mathbf{B}$ texture to all other GPUs via `ncclBroadcast`. Update every 5–10 frames.

### 5.4 Stream Architecture

Each GPU runs three CUDA streams concurrently:

- **Compute stream:** L2 physics kernels.
- **Copy stream:** P2P / NCCL halo exchanges (overlaps with compute).
- **Graphics stream:** OpenGL/Vulkan rendering (interop'd to compute via shared semaphores).

CUDA events synchronize the streams within a frame; `cudaStreamWaitEvent` ensures the graphics stream doesn't start a
pass until its compute dependencies are complete.

---

## 6. Large-CPU-Core Parallelization

CPU work is non-trivial and benefits from many cores:

| Task                                                   | Module                                                           | Parallelism             |
|--------------------------------------------------------|------------------------------------------------------------------|-------------------------|
| QSL / squashing-factor map                             | `numba.prange` over 2M grid points                               | Scales to 64+ cores     |
| Spherical harmonic transforms (PFSS)                   | SHTns with OpenMP                                                | Scales to socket size   |
| Field-line tracing for flare topology                  | `concurrent.futures.ProcessPoolExecutor` over $10^4$ seed points | Embarrassingly parallel |
| Prominence equilibrium solver                          | NumPy vectorized + Numba                                         | Modest                  |
| Parameter sweep / Monte Carlo over stellar populations | Dask                                                             | Scales to cluster       |
| Data I/O (Zarr checkpoint writes)                      | Async via `asyncio` + Zarr's concurrent.futures store            | Overlapped with compute |

A `ThreadPoolExecutor` with a queue marshals these tasks so the main thread (which owns the GL context) never blocks.

---

## 7. Memory Architecture

**Per-GPU memory budget (assuming 24 GB):**

| Buffer                                       | Size              | Notes                                  |
|----------------------------------------------|-------------------|----------------------------------------|
| Photosphere texture (RGBA16F, ping-pong)     | 2 × 1 GB          | 8192×4096                              |
| Active-region / magnetic surface map         | 256 MB            | signed, $B_r$                          |
| 3D coronal $\mathbf{B}$ grid (256³, RGBA16F) | 512 MB            | PFSS output                            |
| Coronal SDF (256³, R16F)                     | 32 MB             | for raymarching                        |
| Prominence SDFs (up to 32 active)            | 32 × 16 MB        | sparse                                 |
| Framebuffer + G-buffer                       | ~200 MB           | half-res raymarch + full-res composite |
| CUDA workspace / scratch                     | ~2 GB             |                                        |
| **Total**                                    | **~6 GB working** | Leaves headroom for future LOD         |

**Host memory:** ~16 GB for parameter sweeps, checkpointing buffers, and stellar atlas datasets (CHIANTI emission
tables, opacity tables).

**Mapped/pinned memory:** All host-GPU transfers use pinned memory (`cudaMallocHost`) so they overlap with compute.
Astropy unit conversions happen exactly once at parameter-change time.

---

## 8. Per-Frame Pipeline (Synchronization Diagram)

```
Frame N starts
├─ CPU thread: poll parameter changes, recompute StellarState if needed (~50 µs)
├─ GPU 0 compute stream:
│   ├─ [if t % 5 == 0] PFSS reconstruction → 3D B grid
│   ├─ Broadcast B grid to GPUs 1..N via NCCL
│   └─ Trigger ribbon stencil computation if flare event queued
├─ All GPUs compute streams (parallel):
│   ├─ Photosphere curl-noise advection (ping-pong write)
│   ├─ Halo exchange (NCCL sendrecv)
│   ├─ Sunspot mask update (Evershed warp)
│   └─ Prominence MRT noise update
├─ All GPUs graphics streams (parallel, after compute fence):
│   ├─ Surface rasterization with gravity-darkening shader
│   ├─ Chromosphere tessellation + spicule displacement
│   ├─ Volumetric raymarching (corona) on assigned screen strip
│   ├─ PRFL geometry-shader pass
│   └─ Local tonemap
├─ GPU 0: gather strips (NCCL allgather → display), final composite, bloom, present
└─ Frame N ends
```

Target: 33 ms (30 fps) at 4K for high-fidelity mode; 16 ms (60 fps) at 1440p; pure offline batch render mode disables
presentation and writes to disk.

---

## 9. Numerical Accuracy and Validation

To honor the document's emphasis on equation accuracy:

1. **Unit safety:** All physics is done in SI internally using `astropy.units`; conversions happen at kernel boundaries.
   A unit-checking test suite (`pytest`) catches dimensional errors.
2. **Reference comparisons:**
    - HR diagram positioning vs. Pleiades / Hyades / field-star catalogs (Gaia DR3) for the model's mass-luminosity and
      Skumanich predictions.
    - Convective turnover $\tau_c$ vs. Noyes et al. (1984) Table 1.
    - X-ray luminosity ratio $L_X / L_{\text{bol}}$ vs. $\text{Ro}$ trend (Pizzolato et al. 2003).
    - Gravity-darkening profile vs. Espinosa Lara & Rieutord (2011) for fast rotators.
    - K-S density profile vs. analytical $\text{sech}^2$.
3. **Conservation checks:** A debug NCCL allreduce computes total mass and magnetic flux every $N$ frames. Drift > 0.1%
   flags a numerical bug.
4. **Convergence tests:** For the raymarcher, doubling step count must change pixel values by < 1% (built-in optional QA
   pass).
5. **Spectral validation:** A synthetic spectrum is computed from the emergent emission and compared against observed
   solar EUV irradiance (SOHO / SDO references) at fiducial inputs.

---

## 10. Python API Surface

The library exposes a minimal, high-level API:

```python
import stellarsim as ss

star = ss.Star(mass=1.0, age=4.6, binary=False)
# auto-computes L, T_eff, R, P_rot, tau_c, Ro, beta

sim = ss.Simulation(
    star=star,
    resolution=(3840, 2160),
    gpus=[0, 1, 2, 3],
    cpu_threads=64,
    quality='high',  # presets: low/medium/high/research
)

# Add transient events programmatically:
sim.queue_flare(longitude=120, latitude=15, magnitude='X2.5')

# Real-time:
sim.run_interactive()  # opens window, GGUI-style controls

# Offline batch:
sim.render_sequence(
    duration=3600,  # seconds of stellar time
    fps=30,
    output='star.zarr',  # state checkpoints
    video='star.mp4',  # rendered frames
)
```

Under the hood, `Simulation.run_interactive` launches the per-frame pipeline of §8; `render_sequence` disables
presentation and adds asynchronous H.265 encoding via `PyNvVideoCodec` on GPU 0.

---

## 11. Build, Packaging, Deployment

- **Build system:** `scikit-build-core` + `pybind11` for C++/CUDA extensions; `pyproject.toml` for the Python package.
- **CUDA kernels:** compiled via NVRTC at install time so end users don't need nvcc; fallback to AOT-compiled fatbins
  for common SM versions (80, 86, 89, 90).
- **Shader hot-reload:** GLSL shaders watched via `watchdog`; recompiled on save during development. Production builds
  bake them.
- **Containerization:** `Dockerfile` based on `nvcr.io/nvidia/cuda:12.4-devel-ubuntu22.04` with all Python deps
  preinstalled; multi-GPU verified via `nvidia-docker` + `--gpus all`.
- **Testing:** unit tests for each physics module (CPU-only, mocked GPU); integration tests on a 1-GPU CI runner; weekly
  multi-GPU smoke test on a dedicated workstation.

---

## 12. Phased Implementation Roadmap

| Phase                                         | Duration | Deliverable                                                    |
|-----------------------------------------------|----------|----------------------------------------------------------------|
| 1. L0 astrophysics core                       | 1 week   | `StellarState` + validation tests against Sun, Sirius, Proxima |
| 2. Single-GPU photosphere                     | 2 weeks  | Curl-noise granulation + sunspots + Evershed, viewable         |
| 3. Gravity darkening + chromosphere           | 1 week   | Oblate fast rotators, spicule shell                            |
| 4. PFSS + raymarched corona (single GPU)      | 3 weeks  | Full coronal volume rendering                                  |
| 5. K-S prominences                            | 2 weeks  | Static + MRT-dynamic prominences with SDF composition          |
| 6. Flare topology + PRFLs                     | 2 weeks  | QSL detection, ribbons, geometry-shader loops                  |
| 7. Multi-GPU strip raymarcher                 | 2 weeks  | NCCL allgather, P2P, scaling tests                             |
| 8. Multi-GPU surface advection with halos     | 1 week   | Linear scaling on photosphere too                              |
| 9. Python API polish + docs                   | 1 week   | Sphinx docs, example notebooks                                 |
| 10. Validation suite + papers' worth of tests | 2 weeks  | Reproducible benchmarks against published data                 |

Total: ~17 weeks for a single experienced engineer; ~9–10 weeks for a small team (2 engineers + 1 graphics specialist).

---

## 13. Key Risks and Mitigations

- **Volumetric raymarching cost:** Mitigate via SDF empty-space skipping, half-res + temporal reprojection, persistent
  threads. If still too slow, drop to $r_{\max} = 3\,R_\star$ and fade.
- **Multi-GPU diminishing returns from NCCL overhead:** Keep collectives small; overlap with compute streams; use NVLink
  topology if available.
- **PFSS staleness:** If spots evolve faster than 5 frames, raymarched corona will lag. Mitigate by running PFSS on a
  dedicated GPU stream with double-buffered $\mathbf{B}$ grids.
- **Numerical drift in long runs:** Phase-wrap all time-dependent values per §8.1 of the source; recompute K-S profiles
  from analytical form rather than time-integrating.
- **Equation fidelity vs. real-time tradeoff:** Provide `quality='research'` mode that disables aggressive
  optimizations (full-res raymarching, $\ell_{\max} = 128$ PFSS, no temporal reprojection) for offline scientific use.

---

This plan keeps every visual element rooted in an explicit equation from the source document, places the heavy math on
the GPUs in a way that scales across cards, and reserves the orchestration, topology, and validation work for the CPU
cores where Python and Numba can use them effectively. Each module is independently testable against published
astrophysical data, so accuracy can be verified incrementally rather than only at the end.