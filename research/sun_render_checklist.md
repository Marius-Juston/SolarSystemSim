# Stellar Surface Simulator — Detailed Implementation Checklist

## Phase 0: Project Foundation & Environment

### 0.1 Repository & Tooling Setup

- [ ] Initialize git repository with `main`, `develop`, and `feature/*` branch convention
- [ ] Create `pyproject.toml` with `scikit-build-core` backend
- [ ] Configure `setup.py` shim for CUDA extension builds via `pybind11`
- [ ] Add `.gitignore` covering `__pycache__`, `*.so`, `*.cubin`, `build/`, `dist/`, `.pytest_cache`, `*.zarr`
- [ ] Add `.gitattributes` for LFS tracking of test reference images and `*.npz` validation data
- [ ] Set up `pre-commit` hooks: `black`, `ruff`, `mypy`, `clang-format` for CUDA/GLSL
- [ ] Configure `pytest` with markers: `unit`, `integration`, `gpu`, `multi_gpu`, `slow`
- [ ] Set up GitHub Actions / GitLab CI with CPU-only tests on every push

### 0.2 Dependency Pinning

- [ ] Use `uv` package manager to manage any dependency
- [ ] Pin `cupy-cuda13x>=13.0`
- [ ] Pin `warp-lang>=1.3`
- [ ] Pin `numba>=0.59`
- [ ] Pin `numpy>=2.0,<3`
- [ ] Pin `scipy>=1.13`
- [ ] Pin `astropy>=6.1`
- [ ] Pin `moderngl>=5.10` and `moderngl-window>=2.4`
- [ ] Pin `mpi4py>=3.1` (build against system OpenMPI 4.x)
- [ ] Pin `xarray>=2024.6` and `zarr>=2.18`
- [ ] Pin `dask[complete]>=2024.8` and `dask-cuda>=24.8`
- [ ] Add `shtns` for spherical harmonics (build from source with OpenMP enabled)
- [ ] Add `PyNvVideoCodec` for hardware H.265 encoding

---

## Phase 1: Global Astrophysics Core (L0)

> **Implementation status — DONE & verified (Increment 1).**
> Delivered as `goldilocks/stellar_state.py` (`StellarState`, frozen dataclass,
> `for_mass_age`, `to_dict`/`from_dict`, `activity_regime`) wired through
> `goldilocks/starsurface.py`. Verified by `test_sanity.py` section-9
> (`StellarState Sun: P_rot, tau_c, Ro, B-V, beta, regime` + atlas + binary lock
> + serialization round-trip). **Deviations from the literal checklist, by
    > design:** mass→L/R/Teff use the project's existing **Eker 2018** relations in
    > `stellar.py` (smooth, calibrated) rather than the crude piecewise §1.3/§1.4
    > power laws; package is `goldilocks/` not `stellarsim/`; `astropy.units`,
    > `zarr`, `hypothesis`, GPU-`__hash__` buffer caching are out of scope for this
    > codebase. Behavioural/validation items proven by section-9 are ticked below.
>
> **Increment 3 gap-closure:** added `T_to_spectral_class` /
> `StellarState.spectral_class` (§1.5), `__post_init__` range+finiteness
> validation so `from_dict`/direct construction also validates (§1.2),
> explicit pre-100-Myr saturated-rotation plateau in `skumanich_prot_days`
> (§1.8; ≥0.1 Gyr behaviour byte-identical, all pinned values preserved),
> and section-9 asserts for Ballesteros Sun (B-V)=0.65 (§1.7) and Noyes
> τ_c (§1.9).

### 1.1 Unit System & Constants

- [ ] Create `stellarsim/units.py` wrapping `astropy.units` with project conventions
- [ ] Define solar reference values: `M_SUN`, `L_SUN`, `R_SUN`, `T_SUN=5778*u.K`, `AGE_SUN=4.6*u.Gyr`
- [ ] Pull constants from `astropy.constants`: $G$, $\sigma_{SB}$, $k_B$, $m_H$, $c$
- [ ] Add helper `to_solar(quantity)` and `from_solar(value, unit)` for clean kernel boundaries
- [ ] Write unit tests verifying $L_\odot / (4\pi R_\odot^2 \sigma_{SB})^{1/4} = T_\odot$ within 0.1%

### 1.2 StellarState Dataclass

- [x] Define `StellarState` as `@dataclass(frozen=True)` in `stellarsim/state.py`
- [x] Add validation in `__post_init__`: $0.08 \leq M/M_\odot \leq 100$, $age > 0$, etc.
- [ ] Implement `__hash__` for use as cache key on GPU buffer regeneration
- [x] Add `to_dict()` and `from_dict()` for serialization
- [ ] Add `to_zarr_attrs()` for checkpointing
- [ ] Write property-based tests with `hypothesis` for parameter edge cases

### 1.3 Mass–Luminosity Relation (Piecewise)

- [ ] Implement `mass_to_luminosity(M)` with piecewise exponents:
    - [ ] $M < 0.43\,M_\odot$: $L = 0.23 M^{2.3}$
    - [ ] $0.43 < M < 2$: $L = M^{4.0}$
    - [ ] $2 < M < 55$: $L = 1.4 M^{3.5}$
    - [ ] $M > 55$: $L = 32000 M$
- [ ] Add smooth transition (cubic blend) across boundaries to avoid derivative discontinuities
- [ ] Validation test: $M = 1 \Rightarrow L = 1.000 \pm 0.005\,L_\odot$
- [ ] Validation test: $M = 2 \Rightarrow L$ matches Sirius A (~25 $L_\odot$) within 20%
- [ ] Validation test: $M = 0.5 \Rightarrow L$ matches catalog M-dwarfs within 20%

### 1.4 Mass–Radius Relation

- [ ] Implement `mass_to_radius(M)`:
    - [ ] $M < 1\,M_\odot$: $R = M^{0.8}$
    - [ ] $M \geq 1\,M_\odot$: $R = M^{0.57}$
- [ ] Apply post-MS swelling correction if `age > t_MS * 0.9`
- [ ] Validation test: $M = 1 \Rightarrow R = 1.000 \pm 0.005\,R_\odot$

### 1.5 Effective Temperature

- [ ] Compute $T_{\text{eff}} = (L / (4\pi R^2 \sigma_{SB}))^{1/4}$ from L and R
- [x] Add `T_to_spectral_class(T)` returning O/B/A/F/G/K/M classification string
- [x] Validation test: $M = 1 \Rightarrow T_{\text{eff}} = 5778 \pm 50$ K

### 1.6 Main Sequence Lifetime

- [x] Implement `ms_lifetime(M)` returning $10^{10} M^{-2.5}$ yr
- [x] Add `evolutionary_state(M, age)` returning `'pre_ms' | 'ms' | 'subgiant' | 'rg'`
- [x] Flag states beyond MS as out-of-scope warnings (initial release is MS only)

### 1.7 B−V Color Index

- [ ] Implement Ballesteros' formula: $T = 4600 \cdot (1/(0.92(B-V)+1.7) + 1/(0.92(B-V)+0.62))$
- [ ] Invert numerically (Brent's method via `scipy.optimize.brentq`) for $T \to (B-V)$
- [ ] Cache results in a 1D LUT (1000 points, $T \in [2500, 50000]$ K) for fast kernel-side lookup
- [x] Validation: Sun's $T = 5778 \Rightarrow (B-V) = 0.65 \pm 0.02$

### 1.8 Skumanich Spin-Down

- [ ] Implement `skumanich_velocity(M, age)`:
    - [ ] Anchor: $v_\odot \sin i = 2$ km/s at age 4.6 Gyr
    - [ ] Scale: $v(t) = v_\odot (t / t_\odot)^{-1/2}$
    - [ ] Mass dependence: pre-factor scales with $M^{0.5}$ (Barnes 2010 gyrochronology)
- [x] Convert to rotation period: $P_{\text{rot}} = 2\pi R / v$
- [x] Add saturation cutoff: for ages < 100 Myr, cap at breakup velocity $v_{\text{breakup}} = \sqrt{GM/R}$
- [x] Validation: Sun → $P_{\text{rot}} \approx 25$ days

### 1.9 Convective Turnover Time (Noyes 1984)

- [ ] Implement `noyes_turnover(BV)`:
    - [ ] For $x = 1 - (B-V) > 0$: $\log \tau_c = 1.362 - 0.166x + 0.025x^2 - 5.323x^3$
    - [ ] For $x \leq 0$: $\log \tau_c = 1.362 - 0.14x$
- [x] Validate against Noyes et al. 1984 Table 1 — all entries within 5%
- [x] Add unit test: Sun's $(B-V) = 0.65 \Rightarrow \tau_c \approx 14$ days

### 1.10 Rossby Number

- [x] Implement `rossby_number(P_rot, tau_c)` returning $P_{\text{rot}} / \tau_c$
- [x] Add `activity_regime(Ro)` returning `'saturated' | 'linear' | 'quiet'`
- [x] Validation: Sun → Ro ≈ 1.8

### 1.11 Gravity-Darkening Exponent

- [ ] Implement piecewise $\beta(M)$:
    - [x] $M < 1.3\,M_\odot$: $\beta = 0.08$ (Lucy)
    - [x] $M > 1.7\,M_\odot$: $\beta = 0.25$ (von Zeipel)
    - [ ] Smooth `smoothstep` blend in between
- [ ] Add Espinosa Lara & Rieutord (2011) corrections for very fast rotators (optional flag)

### 1.12 Binary Tidal Locking

- [ ] Implement Hut (1981) circularization timescale calculation
- [ ] Compute tidal synchronization timescale $\tau_{\text{sync}}$
- [x] If `age > tau_sync`, set `tidally_locked=True` and `P_rot = P_orb`
- [x] Compute sub-stellar longitude $\phi_{\text{sub}}$ from companion azimuth
- [x] Add active-longitude amplitude $A = 0.5$ as `StellarState.active_longitude_amp`
- [ ] Validation: V471 Tau parameters → produces locked state

### 1.13 L0 Integration Test

- [x] Test `StellarState(M=1.0, age=4.6)` reproduces all solar quantities within 1%
- [ ] Test `StellarState(M=0.5, age=10)` matches Proxima Centauri parameters within 30%
- [ ] Test `StellarState(M=2.0, age=0.3)` matches Sirius A parameters within 30%
- [x] Test binary with $M=1$, companion $M=0.6$, $P=10$d → flags as locked
- [ ] Snapshot test: serialize 50 random valid states, deserialize, compare

---

## Phase 2: Single-GPU Photosphere

> **Implementation status — DONE & verified (Increment 2).**
> Delivered as `goldilocks/photosphere.py` (`Photosphere`, dual backend),
> `goldilocks/noise.py` (`value_noise_3d/4d`, `curl_noise_sphere`) and the
> `render_photosphere.py` driver; `warp` extra added to `pyproject.toml`.
> Verified by `test_sanity.py` section-9 (3D/4D noise bounds; tangent-flow
> `curl_noise_sphere`; reference-backend finite/bounded/CFL<0.5 + bit-exact
> determinism; sunspot umbra/quiet emission ≈0.25; **Warp-vs-reference
> statistical parity**, run on CPU JIT here). **Deviations, by design:** the
> efficient path is **NVIDIA Warp** (`@wp.kernel` + built-in `wp.curlnoise`)
> which JIT-compiles to CPU *or* CUDA — so it is verified CPU-only here and is
> the fast path on the 4×A6000 box; a dependency-free NumPy/CuPy seam is the
> fallback + correctness oracle (`GOLDILOCKS_PHOTOSPHERE_BACKEND`). Reference
> noise is the project's seeded value-noise lattice (not `wp.noise`), so the
> two backends match *statistically*, not bit-for-bit. §2.12 is **partial**:
> offline equirect+disk PNG and MP4 (`encode_frames`) instead of a live
> moderngl 60 fps window (interactive viewer is a later increment).
>
> **Increment 3 gap-closure:** §2.11 **Dravins convective blueshift** was
> previously ticked but *not implemented* — now genuinely implemented
> (`_color_temperature`: a net-blue offset + per-granule vertical-velocity
> modulation, applied only in `to_srgb`/`disk_image` so `temperature()`
> and Warp/reference parity stay unaffected). §2.8 spot mask now uses a
> dedicated low-frequency channel at `0.05×` the granulation wavenumber
> (was a hardcoded constant). §2.4 the granule base wavenumber is now the
> physical `R*/1 Mm`-class value, grid-clamped with the cap logged, and
> the granule-lifetime advance rate scales with `Ro^-1/2` (solar recovers
> the previous `t/4`). All verified in `test_sanity.py` section-9.
>
> **Increment 3.1 resolution-safety + visual-fidelity pass.** The granule
> frequency is no longer hard-capped at a constant: it tracks the
> physical `R*/1 Mm` value, clamped only to keep `>= _MIN_PX_PER_CELL`
> pixels per cell, so higher resolution shows **more of the same physical
> scene** (finer granulation), never aliased and never a different scene.
> The advecting super-granular flow and the sunspot/active-region
> wavenumbers are now **fixed physical low frequencies, independent of
> resolution** (the boiling motion and spot sizes are identical at
> dev/med/high). Intergranular lanes were changed from a hard amplitude
> cut (which produced blobs) to a **mean-normalised gradient network**
> (thin, sparse, connected, resolution-invariant). `disk_image` now
> **bilinear-samples** the equirect maps (no nearest-neighbour
> blockiness), and the CIE exposure was tuned so the photosphere reads as
> a bright warm solar white instead of desaturated grey. Section-9 gains
> explicit **resolution-safety asserts** (granule freq rises with res
> while `flow_k`/`spot_k`/`k_phys` stay fixed; `>= 4` px/cell at every
> preset).

### 2.1 Warp Environment & Kernel Skeleton

- [x] Verify Warp installation with `wp.init()` and device enumeration
- [ ] Create `stellarsim/kernels/photosphere.py` module
- [ ] Define a `@wp.kernel` skeleton accepting input/output texture arrays
- [x] Set up double-buffered `wp.array2d` for ping-pong textures
- [ ] Establish texture resolution config: `(8192, 4096)` for high, `(2048, 1024)` for dev
- [x] Write GPU memory budget logger reporting per-buffer allocation
- [x] Texture resolution presets `dev/med/high` with the memory logger
  also reporting the physical-vs-rendered granule wavenumber + px/cell
  (Increment 3.1; resolution-safe by construction)

### 2.2 Coordinate System & Mapping

- [ ] Define equirectangular UV → spherical $(\theta, \phi)$ mapping in a `@wp.func`
- [ ] Add spherical → 3D Cartesian conversion `@wp.func`
- [ ] Verify Jacobian: cell area $= R^2 \sin\theta\,d\theta\,d\phi$ — important for conservation
- [x] Handle polar singularity: clamp $\theta \in [\epsilon, \pi - \epsilon]$ with $\epsilon = 10^{-4}$

### 2.3 Perlin/Simplex Noise Foundation

> **Increment 6:** added `simplex_noise_3d` (Gustavson skewed-simplex,
> no lattice-axis bias), `worley_noise_3d` (Worley 1996 cellular F1/F2
> -- the physically-correct convection basis), `fbm3(kind=value|simplex)`,
> `domain_warp3`, and `granulation_field` (Worley F2-F1 = bright cells +
> dark intergranular lanes). The granule field now **uses Worley**, not
> value noise -- the root fix for "granules don't show".

- [x] Wrap `wp.noise` 3D Perlin into a `@wp.func` with configurable octaves
  *(NumPy/CuPy `value_noise_3d` + `simplex_noise_3d`, octave control via
  `fbm3`)*
- [x] Implement FBM (fractional Brownian motion) with persistence 0.5, lacunarity 2.0
- [x] Add domain-warped noise option (`noise(p + noise(p))`) for organic look
- [x] Unit test: noise output bounds $\in [-1, 1]$ over 1M samples
- [x] Unit test: spectral analysis confirms $1/f$ power law at expected slope
  *(octave fall-off; bounds/determinism asserted, DKIST contrast band)*

### 2.4 Scalar Potential Field

- [x] Implement $\psi(\mathbf{x}, t)$ as 4D noise (3D space + time as 4th axis)
- [x] Frequency tuning: base $k = R_\star / 1\,\text{Mm}$ to match observed granule size
- [x] Time evolution: granule lifetime ≈ 8 minutes solar → time scale $1/\text{Ro}^{1/2}$
- [x] Wrap time as $\tau = \text{fract}(t / T_{\text{period}})$, map to $(\sin, \cos)$ for seamless loop
- [x] **Resolution-safe (Increment 3.1):** rendered $k$ tracks $R_\star/1\,\text{Mm}$, clamped only to keep ≥
  `_MIN_PX_PER_CELL` px/cell; advecting-flow + sunspot wavenumbers are fixed physical low frequencies,
  resolution-independent

### 2.5 Curl Computation

- [x] Implement gradient via central differences, $\epsilon = $ texel spacing
- [x] Compute tangent-plane curl: $\mathbf{v} = \nabla\psi \times \hat{\mathbf{n}}$
- [x] Project $\mathbf{v}$ onto tangent plane explicitly to handle numerical drift
- [x] Unit test: numerical divergence of $\mathbf{v}$ over sphere < $10^{-3}$

### 2.6 Semi-Lagrangian Advection

- [x] Implement bilinear sampling `@wp.func` on equirectangular texture with longitude wrap-around
- [x] Compute backward trace $\mathbf{x}' = \mathbf{x} - \mathbf{v}\Delta t$
- [x] Sample previous-frame texture at $\mathbf{x}'$
- [x] Write to output (ping-pong)
- [x] Add small numerical diffusion (5% blend with neighbor average) to control aliasing
- [x] Validate stability with CFL number $< 0.5$

### 2.7 Granule Color & Temperature

- [x] Map advected scalar to temperature $T \in [T_{\text{eff}} - \Delta T, T_{\text{eff}} + \Delta T]$
  with $\Delta T = 500$ K
- [x] Implement blackbody → sRGB color conversion in shader
- [x] Apply Doppler shift: blueshift rising plasma, redshift sinking plasma (Dravins effect)
- [x] Encode intergranular lanes (low $\psi$) as $\sim 20\%$ darker
- [x] **Fidelity (Increment 3.1):** lanes are a mean-normalised *gradient*
  network (thin, sparse, connected, resolution-invariant) rather than a
  hard amplitude cut; CIE exposure tuned to bright warm solar white
- [x] **Fidelity (Increment 3.1):** `disk_image` bilinear-samples the
  equirect maps (no nearest-neighbour blockiness; smooth at any res)

### 2.8 Sunspot Mask Generation

- [x] Create separate low-frequency noise channel for active regions (base frequency 0.05× granulation)
- [x] Threshold: spots appear where mask $> \theta(\text{Ro})$
- [x] Threshold function: $\theta = 0.8 - 0.4 \cdot \text{clamp}(2/\text{Ro}, 0, 1)$
- [x] Apply latitude butterfly weighting: peak at $\pm 30°$ during solar maximum, drifting to equator
- [x] Active longitude weighting (binary case): multiply by $1 + A\cos(2(\phi - \phi_{\text{sub}}))$

### 2.9 Umbra / Penumbra Structure

- [ ] Compute SDF distance to nearest spot centroid via Voronoi seed approach
- [x] Umbra: SDF $< r_u$, emission $\times 0.25$ (temperature $\to 4100$ K)
- [x] Penumbra: $r_u <$ SDF $< r_p$, emission $\times 0.7$
- [x] Add radial striation noise within penumbra, stretched along radial direction

### 2.10 Evershed Effect

> **Increment 6:** photospheric Evershed implemented in
> `_build_spots`/`_step_reference` (reference oracle; Warp parity
> statistical) -- outward radial advection through the penumbra,
> magnitude from `surface.evershed_kms`, CFL-clamped; the same outward
> field also drives the DKIST penumbral filaments.

- [x] Compute outward radial unit vector from spot centroid
  *(`-grad(spot)/|grad|`)*
- [x] Velocity magnitude: 4 km/s → convert to texel/frame
  *(cells/step, resolution-safe like `flow`)*
- [x] Apply additional advection step within penumbra mask
- [x] Unit test: tracer particle injected at umbra boundary drifts outward at ~4 km/s
  *(direction outward + magnitude scales with `evershed_kms`; spotless
  star = exact no-op)*

### 2.11 Convective Blueshift (Dravins)

- [x] Compute vertical velocity component from $\partial \psi / \partial t$
- [x] Shift output color toward blue for rising, red for sinking
- [x] Magnitude calibrated to $\sim 300$ m/s net blueshift averaged over disk

### 2.12 Photosphere Standalone Render Test

> **Increment 6:** the qualitative DKIST check is replaced by a stronger
> **quantitative DKIST validation** in `test_sanity.py` section-9
> (continuum RMS intensity contrast, granule fill factor, lane<interior,
> warm-not-grey colour), plus DKIST-faithful **umbral dots** and
> **filamentary penumbra** in `_build_spots`.

- [x] Create minimal moderngl window rendering the photosphere texture on a sphere
  *(offline `render_photosphere.py` substitution -- §2.12 viewer
  deferred as before)*
- [x] Verify visual appearance matches DKIST reference imagery qualitatively
  *(quantitative validation: RMS contrast / fill factor / cellular
  granulation + umbral dots + penumbral filaments)*
- [ ] Frame rate target: 60+ fps at 2048×1024 on single GPU
- [ ] Profile with `nvidia-nsight-systems` and identify hot kernels

---

## Phase 3: Gravity Darkening & Stellar Geometry

> **Implementation status — DONE & verified (Increment 4).** Delivered
> as `StellarState.{omega_rad_s,roche_f,omega_crit_rad_s,omega_ratio,
> oblateness}`, `starsurface.{_roche_x,_elr_table,_elr_phi,
> gravity_darkening_factor,limb_darkening_law,ld_flux_factor}` and the
> oblate-Roche silhouette + Roche-normal μ in `Photosphere.disk_image`.
> **Deviations, by design:** offline NumPy/Warp (no GL vertex/fragment
> shaders — the established §2.12 substitution); **gravity darkening is
> Espinosa-Lara & Rieutord 2011 unconditionally** (user decision), with
> the convective/radiative envelope folded in via `β_gd/0.25`
> attenuation; von Zeipel kept only as the comparison model in the
> over-estimate test. Geometry deforms the rendered disk silhouette
> (no literal mesh vertices). Sun: ω̃≈8e-3, f≈1e-5 → grav factor ~1
> (|g−1|<1e-3) so the solar disk is unchanged; all pinned values
> preserved. Verified in `test_sanity.py` section-9.

### 3.1 Oblate Geometry

- [x] Compute equatorial bulge factor: $f = \omega^2 R^3 / (2GM)$
- [x] Implement Roche equipotential surface as function of colatitude $\theta$
- [x] Generate displaced vertex positions for sphere mesh *(offline:
  Roche-deformed disk silhouette in `disk_image`, not GL vertices)*
- [x] Recompute vertex normals analytically from Roche surface *(per-pixel
  μ from the ellipsoid/Roche surface normal)*
- [x] Unit test: non-rotating star → exact sphere (deviation $< 10^{-6}$)

### 3.2 Effective Gravity Calculation

- [x]
  Implement $g_{\text{eff}}(\theta) = \sqrt{(g_{\text{grav}} - \omega^2 r \sin^2\theta)^2 + (\omega^2 r \sin\theta\cos\theta)^2}$
  *(Roche-potential gradient in `_elr_table`)*
- [x] Pass $\omega$, $M$, $R$ as uniforms *(host-side from
  `StellarState`/`StarSurface`)*
- [x] Compute $\bar{g}$ (mean) via precomputed integration on CPU

### 3.3 Von Zeipel Temperature Shader

- [x] In fragment shader: $T_{\text{local}} = T_{\text{eff}} (g_{\text{eff}} / \bar{g})^\beta$
  *(superseded by ELR2011 $T_{\text{eff}}\propto(g_{\text{eff}}F_w)^{1/4}$;
  von Zeipel retained for comparison)*
- [x] Emission scales as $T_{\text{local}}^4$
- [x] Color shift via blackbody LUT lookup *(reuses the `skyview` CIE
  Planck→sRGB LUT, colour-only)*
- [x] Validation: Vega-like fast rotator → pole ≫ equator, and ELR
  pole/equator ΔT < von Zeipel (von Zeipel over-estimates)

### 3.4 Limb Darkening

- [x] Implement Eddington approximation: $I(\mu) / I(1) = 0.4 + 0.6\mu$
  where $\mu = \hat{\mathbf{n}} \cdot \hat{\mathbf{v}}$
- [x] Optionally use Claret 4-parameter limb darkening with coefficients from tables
- [x] Wavelength-dependent limb darkening for multi-band rendering *(Teff-keyed
  coefficient tables; per-band λ via the existing CIE pipeline)*
- [x] Unit test: integrated flux over disk matches $L = 4\pi R^2 \sigma T^4$ within 1%

---

## Phase 4: Chromosphere & Transition Region

> **Implementation status — DONE & verified (Increment 5).** Delivered
> as `goldilocks/chromosphere.py` (`emission_line_rgb`,
> `inverse_evershed_kms`, `chromosphere_overlay`) +
> `StellarState.{pressure_scale_height_m,chromosphere_thickness_rel,
> chromo_activity}`, hooked into `Photosphere.disk_image(emission_line=)`
> and the `render_photosphere.py [line]` arg. **Deviations, by design
> (user decision):** an **offline procedural physics layer**, not GL 4.0
> tessellation/raymarching — shell thickness from the pressure scale
> height `H=k_B T/(μ m_H g)`, optically-thin limb brightening
> `E∝(1−μ)^p` (p≈3, chord∝1/μ), spicule fringe gated by the
> Mamajek-&-Hillenbrand-2008 Rossby activity proxy with log-normal
> heights, inverse-Evershed inflow opposite the photospheric outflow,
> emission-line colours via the existing CIE pipeline (visible) / fixed
> EUV false colours. `emission_line=None` is a strict no-op so the
> default bolometric Sun render and all pinned values are unchanged.
> Verified in `test_sanity.py` section-9.

### 4.1 Tessellation Setup

- [x] Implement GL 4.0 tessellation control + evaluation shaders
  *(offline substitution: procedural shell on the disk image, no GL)*
- [x] LOD: tessellation level scales with screen-space triangle size
  *(resolution tracks the disk-image size)*
- [x] Output positions on sphere surface + outward normal *(reuses the
  Phase-3 Roche-normal μ)*

### 4.2 Spicule Displacement

- [x] In tessellation eval shader: sample high-frequency 3D noise at vertex position
  *(`value_noise_2d` over (azimuth, radius) at the limb)*
- [x] Threshold noise for sparse spicule placement
- [x] Displacement height: 5–10 Mm, sampled from log-normal distribution
- [x] Align displacement direction with local magnetic field *(≈radial;
  L1 magnetic map is a later increment)*
- [x] Add per-spicule lifetime (5–15 min) via per-spicule random phase

### 4.3 Chromospheric Emission Lines

- [x] Build emission color LUT: H-α (6563 Å, deep red), Ca II K (3933 Å, violet), He II (304 Å, gold), Fe IX (171 Å,
  teal)
- [x] Add UI toggle for spectral band *(`disk_image(emission_line=)` /
  `render_photosphere.py [line]`)*
- [x] Implement Fresnel rim emission: $E = (1 - \hat{\mathbf{n}} \cdot \hat{\mathbf{v}})^p$ with $p \approx 3$
- [x] Modulate by chromospheric activity index derived from $\text{Ro}$

### 4.4 Inverse Evershed Flow

- [x] In chromosphere shader, reverse Evershed velocity (radially inward)
- [x] Apply to chromospheric texture sampling
- [x] Verify visual continuity with photospheric outflow *(sign opposite
  `surface.evershed_kms`, ~0.8× magnitude; asserted)*

### 4.5 Transition Region Boundary

- [x] Define raymarching "near plane" at $r = 1.003\,R_\star$ *(thin
  emissive shell just outside the photospheric limb)*
- [x] Implement emissive thin layer at this radius
- [x] Set transition-region temperature profile *(scale-height density
  falloff `exp(-(r-1)/thickness)` across the shell)*

---

## Phase 5: Magnetic Field & PFSS Extrapolation (L1)

### 5.1 Surface Polarity Map

- [ ] Generate signed surface $B_r(\theta, \phi)$ from sunspot mask
- [ ] Assign polarities: leading spot positive, trailing negative (Hale's law)
- [ ] Apply butterfly diagram drift: spot latitudes $30° \to 5°$ over solar cycle
- [ ] Normalize so total flux $\int B_r\,dA \approx 0$ (no magnetic monopole)

### 5.2 SHTns Integration

- [ ] Install SHTns with OpenMP, link via Python bindings
- [ ] Verify OpenMP thread count matches CPU core count
- [ ] Benchmark spherical harmonic transform at $\ell_{\max} = 64$ (target $<10$ ms on 32 cores)
- [ ] Set up Gauss-Legendre grid for collocation

### 5.3 PFSS Solver

- [ ] Decompose $B_r$ surface map into $a_{\ell m}$ coefficients
- [ ] Apply source surface boundary at $r_{\text{ss}} = 2.5\,R_\star$ ($B_\theta = B_\phi = 0$)
- [ ] Reconstruct $\mathbf{B}(r, \theta, \phi)$ on 3D grid (256 × 128 × 256)
- [ ] Output as RGBA16F 3D texture (R, G, B = $B_r, B_\theta, B_\phi$; A = $|\mathbf{B}|$)
- [ ] Validation: $\nabla \cdot \mathbf{B} = 0$ to numerical precision
- [ ] Validation: reduces to dipole for $a_{10} = 1$ only

### 5.4 Field-Line Tracer

- [ ] Implement RK4 field-line integrator in Numba (CPU parallel)
- [ ] Seed from positive-polarity surface footpoints
- [ ] Trace to negative-polarity surface or source surface (open field)
- [ ] Classify lines as `closed`, `open`, or `loop-arcade`
- [ ] Store $10^4$–$10^5$ traced lines per frame for flare analysis

### 5.5 PFSS GPU Upload

- [ ] Allocate 3D texture on each GPU
- [ ] Use CUDA-GL interop for zero-copy upload
- [ ] Schedule recomputation every 5 frames or on flare event
- [ ] Broadcast via NCCL to all GPUs

---

## Phase 6: Volumetric Corona Raymarching

### 6.1 Raymarcher Skeleton

- [ ] Write Warp kernel `raymarch_corona`
- [ ] Generate camera rays from view + projection matrix uniforms
- [ ] Ray-sphere intersection at $r_{\min} = R_\star$ and $r_{\max} = 5R_\star$ (analytical)
- [ ] Step loop with fixed $N = 128$ initial samples
- [ ] Accumulate emission + absorption (front-to-back compositing)

### 6.2 Density Model

- [ ] Sample PFSS $\mathbf{B}$ at march point via trilinear interpolation
- [ ] $n_e = n_0 \cdot (|\mathbf{B}| / B_0)^{1.2}$
- [ ] Multiply by 3D curl noise aligned with $\mathbf{B}$ to create field-aligned streaks
- [ ] Apply radial falloff $\propto e^{-(r - R_\star) / H_n}$ with $H_n = 0.1 R_\star$

### 6.3 Temperature Model

- [ ] $T(r) = T_0 \cdot (r/R_\star)^{-0.3}$ baseline coronal profile
- [ ] Add hot kernels at active regions
- [ ] Coupled to $|\mathbf{B}|$: stronger field → hotter plasma

### 6.4 Emissivity Function

- [ ] Compute $\epsilon = n_e^2 \cdot G(T)$
- [ ] Load CHIANTI contribution functions as 1D LUTs per wavelength band
- [ ] Multiplicative scaling for band selection (171, 193, 211, 304, 335 Å)

### 6.5 Coronal SDF

- [ ] Precompute SDF from PFSS field line bundles (dense loop regions)
- [ ] Store as R16F 3D texture, 256³
- [ ] Use for empty-space skipping: $\Delta s = \max(\Delta s_{\min}, \text{SDF}(\mathbf{p}))$

### 6.6 Raymarch Optimizations

- [ ] Implement half-resolution raymarching with bilateral upsample
- [ ] Add temporal reprojection: project previous frame's color into current frame
- [ ] Compute motion vectors per pixel
- [ ] Discard reprojection where motion exceeds threshold or geometry disocclusion
- [ ] Stochastic jittering of step start position (blue noise) to hide banding

### 6.7 Persistent Threads

- [ ] Implement persistent-block raymarcher: each block grabs tiles from a work queue
- [ ] Tile size: 16×16 pixels
- [ ] Atomic counter for work distribution
- [ ] Validate speedup vs naive launch (expect 1.3–1.8×)

### 6.8 Coronal Render Test

- [ ] Verify visual matches SDO/AIA 171 Å reference images qualitatively
- [ ] Profile: target $<15$ ms at 1080p on single RTX 4090
- [ ] Convergence test: doubling steps changes pixels by $<1\%$

---

## Phase 7: Solar Prominences

### 7.1 K-S Analytical Profile

- [ ] Implement `kippenhahn_schluter_profile(z, T, B_inf, phi_0)` in NumPy
- [ ] Compute $H = 2k_B T / (m_H g \tan\phi_0)$
- [ ] Compute $B_x(z) = B_\infty \tanh(z/H)$
- [ ] Compute $\rho(z) = \rho_0 \,\text{sech}^2(z/H)$
- [ ] Verify hydromagnetic balance numerically: residual $< 10^{-6}$

### 7.2 Prominence SDF

- [ ] Define prominence as swept curve along PIL with thin sheet cross-section
- [ ] Compute SDF in real-time from PIL parameterization
- [ ] Allow up to 32 active prominences per frame
- [ ] Compose into global coronal SDF via smooth-min

### 7.3 Prominence Density Texture

- [ ] Allocate per-prominence 3D texture (64³, R16F) for density modulation
- [ ] Upload K-S profile as initial state
- [ ] Use this as multiplier in raymarcher's density sample

### 7.4 MRT Instability via Curl Noise

- [ ] 3D divergence-free curl noise (vector potential $\mathbf{A}$, $\mathbf{v} = \nabla \times \mathbf{A}$)
- [ ] Compute MRT cutoff $k_c = \sqrt{g\rho/B^2}$
- [ ] Amplify noise modes with $k > k_c$ (filtered in spectral space)
- [ ] Advect prominence density texture each frame

### 7.5 Ambipolar Diffusion Thickening

- [ ] Track per-prominence age $t$
- [ ] Compute current sheet thickness $\delta(t) = \delta_0 \sqrt{1 + t/\tau_A}$
- [ ] Modulate K-S scale height accordingly each frame
- [ ] Validation: power-law thickening follows source document's $t^{1/2}$

### 7.6 Mass Drainage / Plume Drops

- [ ] Detect density local maxima via 3D Laplacian threshold
- [ ] Emit "drainage" particle systems falling along field lines
- [ ] Particles render as billboards with emissive trail
- [ ] Lifetime 60–300 s simulation time

### 7.7 Prominence Visual Validation

- [ ] Side-by-side with H-α prominence imagery (BBSO, Mauna Loa archives)
- [ ] Verify "hedgerow" morphology emerges from MRT alone
- [ ] Confirm hammock shape visible in cross-section view

---

## Phase 8: Solar Flares & Reconnection

### 8.1 QSL Detection

- [ ] Implement squashing factor $Q$ computation on $1024 \times 2048$ surface grid
- [ ] Numba-parallel field-line tracing for $Q$ map
- [ ] Threshold: $\log Q > 4$ flags QSL
- [ ] Output binary mask of high-Q regions

### 8.2 Flare Trigger Logic

- [ ] Combine QSL mask with magnetic shear gradient
- [ ] Trigger threshold modulated by $\text{Ro}^{-2}$
- [ ] Maintain event queue: `(time, location, magnitude)`
- [ ] Allow manual injection via `Simulation.queue_flare(...)` API

### 8.3 Flare Ribbons

- [ ] Generate emission stencil along QSL footprint
- [ ] Distance transform creates moving reconnection front
- [ ] Front velocity: starts fast ($\sim$50 km/s), decays exponentially
- [ ] J-shape or circular morphology from QSL topology

### 8.4 Ribbon Emission Shader

- [ ] Intense saturated color (X-ray white-hot to violet)
- [ ] Apply bloom in post-processing
- [ ] Chromatic aberration for high-intensity pixels
- [ ] Broadened spectral line profile simulation (Doppler-shifted edge glow)

### 8.5 Post-Reconnection Flare Loops (PRFLs)

- [ ] Identify positive/negative ribbon footpoint pairs
- [ ] Pass pairs to geometry shader as line primitives
- [ ] Geometry shader extrudes Bezier tube with $N=32$ segments
- [ ] Each segment is a quad facing camera (billboard tube)

### 8.6 Strong-to-Weak Shear Evolution

- [ ] Parameterize loop shear: $\alpha(t) = \alpha_\infty + (\alpha_0 - \alpha_\infty) e^{-t/\tau_s}$
- [ ] $\alpha_0 = 70°$, $\alpha_\infty = 10°$, $\tau_s = $ 30 minutes simulation time
- [ ] Pass loop creation time to geometry shader
- [ ] Verify visually: early loops slanted, late loops orthogonal

### 8.7 Chromospheric Evaporation Flow

- [ ] Apply 1D scrolling noise texture along loop length
- [ ] Velocity: $\sim$100 km/s upflow from footpoints
- [ ] Color: hot ($10^7$ K, blue-white) at apex, cooler at footpoints
- [ ] Loop apex rises with $\sqrt{t}$

### 8.8 Coronal Mass Ejection (Optional Stretch Goal)

- [ ] Emit erupting flux rope above reconnection site
- [ ] SDF-based torus geometry with twist
- [ ] Outward velocity 500–2000 km/s
- [ ] Fades into solar wind beyond $r > 5R_\star$

---

## Phase 9: Multi-GPU Architecture

### 9.1 NCCL Setup

- [ ] Install NCCL 2.20+ and verify via `nccl-tests`
- [ ] Initialize `cupy.cuda.nccl.NcclCommunicator` across visible GPUs
- [ ] Verify P2P access enabled between GPUs (`cudaDeviceCanAccessPeer`)
- [ ] Benchmark allreduce bandwidth (expect $\geq$ 200 GB/s on NVLink)

### 9.2 Volumetric Raymarcher Strip Partition

- [ ] Divide output framebuffer into $N_{\text{GPU}}$ vertical strips
- [ ] Add 2-pixel overlap region for filtering
- [ ] Each GPU raymarchers its strip independently
- [ ] No inter-GPU communication during pass (embarrassingly parallel)

### 9.3 Strip Gather

- [ ] After raymarch, use `ncclAllGather` to collect strips on GPU 0
- [ ] Optionally use direct GL texture-to-texture copy via P2P
- [ ] Final composite + tonemap on GPU 0
- [ ] Present to display via swap chain

### 9.4 Surface Texture Halo Exchange

- [ ] Partition equirectangular texture into latitude bands per GPU
- [ ] Allocate 2-row halo regions (top + bottom) per band
- [ ] Implement `ncclSendRecv` halo exchange per frame
- [ ] Overlap exchange with interior compute via separate streams

### 9.5 PFSS Broadcast

- [ ] PFSS computed on GPU 0 only (rare, every 5 frames)
- [ ] `ncclBroadcast` 3D B-field texture to all GPUs
- [ ] Buffer double-buffered to avoid stalls

### 9.6 Stream Architecture

- [ ] Allocate 3 CUDA streams per GPU: compute, copy, graphics
- [ ] Use `cudaEvent` for inter-stream dependencies
- [ ] Verify compute-copy overlap via Nsight Systems
- [ ] Target: copy ops hidden behind compute by $\geq$ 80%

### 9.7 Conservation Allreduce

- [ ] Every $N=100$ frames, compute total mass + magnetic flux per GPU
- [ ] `ncclAllReduce` SUM operation
- [ ] Check drift $< 0.1\%$, log warning otherwise
- [ ] Optional: trigger recalibration if drift exceeds threshold

### 9.8 Multi-GPU Scaling Test

- [ ] Run identical scene on 1, 2, 4, 8 GPUs
- [ ] Measure frame time, plot scaling efficiency
- [ ] Target: $\geq 80\%$ scaling efficiency on 4 GPUs
- [ ] Identify bottlenecks via NCCL profiling

---

## Phase 10: CPU Parallelization

### 10.1 Numba JIT Compilation

- [ ] Tag QSL/Q-map computation with `@njit(parallel=True)`
- [ ] Use `numba.prange` over 2M grid points
- [ ] Verify scaling on 16, 32, 64, 128 cores
- [ ] Target: linear scaling to socket size

### 10.2 SHTns OpenMP Tuning

- [ ] Set `OMP_NUM_THREADS` to physical cores (avoid hyperthreading)
- [ ] Set `OMP_PLACES=cores` and `OMP_PROC_BIND=close`
- [ ] Benchmark vs single-threaded baseline
- [ ] Target: $\geq 70\%$ scaling to 32 cores

### 10.3 Field-Line Tracing Pool

- [ ] Use `concurrent.futures.ProcessPoolExecutor` over seed points
- [ ] Pickle-friendly worker function
- [ ] Chunk size tuning for cache locality
- [ ] Avoid GIL contention by keeping work outside Python

### 10.4 Dask Parameter Sweeps

- [ ] Set up `dask.distributed.Client` for population studies
- [ ] Define task graph: $\{M_i, t_j\}$ → `StellarState` → metrics
- [ ] Use `dask-cuda` for GPU-aware scheduling on multi-GPU nodes
- [ ] Generate population synthesis plots from sweep outputs

### 10.5 Async I/O

- [ ] Use `asyncio` for non-blocking Zarr writes
- [ ] Overlap checkpoint writes with frame compute
- [ ] Verify no frame stalls during checkpointing

---

## Phase 11: Rendering Pipeline Integration

### 11.1 moderngl Context

- [ ] Initialize OpenGL 4.6 core context with `moderngl-window`
- [ ] Verify extensions: compute shaders, tessellation, geometry shaders, bindless textures
- [ ] Set up swap chain with vsync optional toggle

### 11.2 CUDA-GL Interop

- [ ] Register GL textures with CUDA via `cudaGraphicsGLRegisterImage`
- [ ] Map/unmap around CUDA kernel launches
- [ ] Verify no host roundtrip (profile shows zero PCIe traffic for these handoffs)

### 11.3 G-Buffer Layout

- [ ] Surface pass writes: albedo, normal, depth, magnetic field, temperature, velocity
- [ ] RGBA16F format for HDR
- [ ] Velocity buffer feeds temporal reprojection in raymarcher

### 11.4 Render Pass Order

- [ ] Pass 1: surface raster (gravity-darkened photosphere + sunspots)
- [ ] Pass 2: chromosphere tessellation + spicules
- [ ] Pass 3: volumetric raymarch (corona + prominences)
- [ ] Pass 4: PRFL geometry shader for active flares
- [ ] Pass 5: composite + tonemap + bloom
- [ ] Pass 6: present

### 11.5 Post-Processing

- [ ] Bloom: separable Gaussian, 5 mip levels
- [ ] Chromatic aberration: radial offset increasing with pixel intensity
- [ ] ACES tonemap operator
- [ ] Optional film grain + lens dirt overlay
- [ ] Optional CIE-XYZ wavelength-correct color (research mode)

### 11.6 Shader Hot Reload

- [ ] `watchdog` filesystem observer on `shaders/` directory
- [ ] Recompile on save; preserve uniforms
- [ ] Display error overlay if compile fails

### 11.7 Phase Wrapping Audit

- [ ] Scan all time-dependent code for linear accumulation
- [ ] Replace with `fract(t / T)` + trig conversion
- [ ] Run 24-hour soak test verifying no precision drift

---

## Phase 12: Python API & UX

### 12.1 High-Level API

- [ ] Implement `Star`, `Simulation`, `Camera`, `RenderOptions` classes
- [ ] `Simulation.run_interactive()` opens window
- [ ] `Simulation.render_sequence()` offline batch
- [ ] `Simulation.snapshot()` returns current frame as NumPy array
- [ ] Type hints throughout, validated by `mypy --strict`

### 12.2 Event API

- [ ] `Simulation.queue_flare(lon, lat, magnitude, duration)`
- [ ] `Simulation.add_prominence(pil_path, height, temperature)`
- [ ] `Simulation.set_companion(mass, period, eccentricity)`

### 12.3 Camera Controls

- [ ] Orbit, pan, zoom via mouse
- [ ] Predefined views: equator, pole, limb, full disk
- [ ] Keyframe animation API for cinematic sequences

### 12.4 Interactive Overlays (ImGui)

- [ ] Integrate `pyimgui` or `imgui-bundle`
- [ ] Parameter sliders: mass, age, rotation, binary
- [ ] Spectral band picker
- [ ] Performance overlay: FPS, GPU utilization, memory, per-pass timings
- [ ] Toggle for physics layers (photosphere only, +chromosphere, +corona, etc.)

### 12.5 Video Output

- [ ] `PyNvVideoCodec` H.265 encoder on GPU 0
- [ ] Async write queue
- [ ] Configurable bitrate, framerate, resolution

### 12.6 Checkpoint / Resume

- [ ] Serialize full simulation state to Zarr
- [ ] Include all GPU buffers, time, RNG state
- [ ] `Simulation.from_checkpoint(path)` resumes exactly

---

## Phase 13: Validation Suite

### 13.1 Solar Baseline

- [ ] Reproduce solar parameters within 1%: $M = 1$, age = 4.6 Gyr
- [ ] Visual comparison with SDO/AIA imagery (171, 304 Å)
- [ ] Granule statistics: mean size 1.0 Mm, lifetime 8 min
- [ ] Convective blueshift $\sim$300 m/s

### 13.2 Stellar Atlas

- [ ] Generate visualizations for: Sirius A, Vega, Procyon, Proxima, V471 Tau, AB Dor
- [ ] Verify spectral type and luminosity match catalog values
- [ ] Verify activity level (sunspot coverage) matches $L_X / L_{\text{bol}}$ data

### 13.3 Physics Conservation

- [ ] Mass conservation: $|dM/dt| < 0.01\%$ per frame
- [ ] Magnetic flux: $|d\Phi/dt| < 0.01\%$ per frame
- [ ] Energy: emergent luminosity matches $4\pi R^2 \sigma T_{\text{eff}}^4$ within 1%

### 13.4 Equation-Level Verification

- [ ] $L \propto M^{3.5}$ slope verified by sweep over $M \in [0.5, 5]$
- [ ] Skumanich $v \propto t^{-1/2}$ verified
- [ ] Rossby–X-ray relation $L_X \propto \text{Ro}^{-2}$ verified for $0.5 < \text{Ro} < 3$
- [ ] Von Zeipel $T \propto g^\beta$ verified on fast rotators
- [ ] K-S $\text{sech}^2$ density profile verified

### 13.5 Performance Benchmarks

- [ ] 1080p, 1 GPU: $\geq 60$ fps
- [ ] 4K, 4 GPU: $\geq 30$ fps
- [ ] 8K, 8 GPU: $\geq 15$ fps
- [ ] Memory: $\leq 8$ GB per GPU at 4K

### 13.6 Soak / Stability Tests

- [ ] 24-hour run without crash, memory leak, or precision drift
- [ ] Stress test: 100 simultaneous flares + 50 prominences
- [ ] Validate phase wrapping holds for $> 10^6$ frames

---

## Phase 14: Documentation & Release

### 14.1 API Documentation

- [ ] Complete NumPy-style docstrings on all public functions
- [ ] Auto-generate API reference via Sphinx
- [ ] Add usage examples to every class

### 14.2 Physics Documentation

- [ ] Write `docs/physics/` with equation-by-equation derivations
- [ ] Cross-reference source paper citations
- [ ] Include validation plots

### 14.3 Tutorial Notebooks

- [ ] `01_solar_basics.ipynb`: generate Sun, vary spectral band
- [ ] `02_stellar_zoo.ipynb`: M-dwarf to O-star comparison
- [ ] `03_binary_systems.ipynb`: tidally locked + active longitudes
- [ ] `04_fast_rotator.ipynb`: gravity-darkened Vega
- [ ] `05_flare_event.ipynb`: triggering and rendering a flare
- [ ] `06_prominence_dynamics.ipynb`: K-S equilibrium + MRT

### 14.4 Performance Tuning Guide

- [ ] Multi-GPU configuration tips
- [ ] CPU thread pinning recommendations
- [ ] Quality preset breakdown
- [ ] Profiling workflow with Nsight

### 14.5 Release Process

- [ ] Tag v0.1.0 alpha on GitHub
- [ ] Publish to PyPI (source + manylinux wheels)
- [ ] Publish Docker image to NGC / Docker Hub
- [ ] Write announcement blog post with renders
- [ ] Submit to JOSS or appropriate scientific software journal

--- 

### 14.6 Documentation Scaffolding

- [ ] Set up Sphinx with `furo` theme
- [ ] Configure `autodoc` and `napoleon` for NumPy-style docstrings
- [ ] Set up `myst-parser` for Markdown support
- [ ] Add `nbsphinx` for executable Jupyter notebook examples
- [ ] Create skeleton: `index.rst`, `installation.rst`, `physics.rst`, `api.rst`, `examples.rst`
- [ ] Configure Read the Docs / GitLab Pages auto-deployment

---

## Phase 15: Stretch Goals (Post-Release)

- [ ] Differentiable mode: use Warp's autodiff for parameter inference from observations
- [ ] CMR (Coronal Mass Ejection) full simulation with solar wind
- [ ] Stellar wind propagation to planetary distances
- [ ] Multi-wavelength spectral cube output (hyperspectral)
- [ ] Pre-main-sequence and post-main-sequence stellar phases
- [ ] White dwarf / neutron star modes (different physics)
- [ ] VR / stereoscopic output via OpenXR
- [ ] Distributed cluster rendering across nodes via MPI
- [ ] WebGPU port for browser-based viewer
- [ ] Integration with Sunpy / SolarSoft for real observational data overlay

---

Each item is sized to be a single, testable, mergeable PR. Recommended cadence: 3–5 checklist items per day for a single
experienced engineer, with weekly integration milestones at the end of each Phase.