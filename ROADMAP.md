# ROADMAP — Random full solar-system generator

Living status tracker for the random solar-system feature (moons,
habitability profiles, nested N-body stability, visualization). Updated as
each phase lands.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done

## Phases

- [x] **1. `planets.py` (additive)** — `Planet.moons`, `Planet.habitability`,
  `Planet.inclination_deg` fields; `bulk_density_gcc`, `is_gas_giant`. No
  change to existing defaults/sanity values.
- [x] **2. `moons.py`** — `Moon` dataclass; fluid planetary Roche limit;
  Domingos+2006 critical Hill fraction; `generate_moons` (terrestrial 0–2,
  giants ~24–170; Roche + Hill-fraction + mutual-Hill validated; irregulars
  kept inside 0.8·f_crit).
- [x] **3. `habitability.py`** — `HabitabilityProfile` + `profile_for_planet`:
  gravity, insolation/T_eq, rotation, day/night, obliquity/seasons,
  magnetosphere, atmosphere + sky color, winds/storms, biosphere score.
- [x] **4. `nbody_moons.py`** — nested KDK leapfrog (stars analytic, planets +
  moons numeric); vectorized all-pairs; osculating-element `stability_report`
  (secular drift vs bounded libration; ejection/e).
- [x] **5. `solar_system.py`** — Raghavan-2010 PMF (56/33/8/3); 4-star via
  collapsed effective `Star`; snow-line placement; **mutual-Hill spacing
  (Δ=9)**; forced ≥1 PHZ terrestrial; retry loop.
- [x] **6. `viz_solar_system.py`** — overview PNG, per-planet zoom cards,
  long-term trajectory + envelope stability PNG, system MP4.
- [x] **7. `generate_solar_system.py`** — seeded driver, repo-relative output;
  graceful ffmpeg-absent fallback.
- [x] **8. `test_sanity.py`** — additive regression asserts (section 9).

## Skyview v2 — sky bodies, oblate worlds, lit night

Ground-to-sky renderer upgrades (`goldilocks/skyview.py`,
`render_skyview.py`). Physics grounded in: Maclaurin slow-rotation
flattening `f≈(5/4)·ω²R_eq³/GM` (Maclaurin 1742; Murray & Dermott
1999); Lambert phase `Φ(α)=[sin α+(π−α)cos α]/π` (Madhusudhan &
Burrows 2012; Cahoy+ 2010); moonless night budget V≈22 mag/arcsec²,
airglow ≫ zodiacal ≫ integrated starlight (Roach & Gordon 1973;
Hänel+ 2018).

- [x] **S1. Oblateness** — `oblateness_for()` (Maclaurin), exact affine
  `_ellipsoid_quadratic()`; ground + atmosphere-top intersections in
  `render_sky` use the oblate spheroid (Earth-analog f≈0.003 → output
  unchanged; fast rotators visibly flattened).
- [x] **S2. Sibling planets + every moon in the sky** — `sky_bodies()`:
  per-body Kepler phase (`kepler.orbital_elements_to_state`/
  `orbital_period`), reflected Lambert spectrum tinted by
  `sky_color_hex`; phased oblate disks / PSF points composited into
  `render_sky`, horizon-reddened by the existing slant transmittance.
  Faint irregular swarm (`mass_me<1e-5`) skipped for cost.
- [x] **S3. Lit procedural background** — cached
  `background_starfield(seed)` (deterministic; Milky-Way density band +
  diffuse glow), airglow (O I 557.7 nm + van Rhijn) + zodiacal floor;
  capped dark-frame auto-exposure (`max_dark_gain`) so midnight is a
  starlit blue-grey, never pure black, while noon stays faithful.
- [x] **S4. Annotated stills/MP4 + bodies debug** — `_body_line()` adds
  the brightest visible moons/planets + phase to the
  `render_phases`/`animate_day` overlays; `SkyBody` carries its
  on-screen pixel/altitude; `debug_sky_bodies()` writes an annotated
  contact sheet + MP4 that circles & labels every composited sibling
  planet/moon (name, kind, phase, altitude) across a full day, proving
  they render and track across the sky.
- [x] **S5. Driver** — `render_skyview.py`: Moon on the Sol test, two
  random systems, and `sky_colour_gallery()` — a *deterministic
  robustness stress test* that always spans every atmosphere regime
  (blue / deep-blue / CO2 yellow / Titan orange / near-airless black /
  H2-He white / banded giant), a binary two-sun sky, an extreme oblate
  fast rotator, a deep-night multi-colour starfield, and a black-body
  star-colour strip (cool red → hot blue).
- [x] **S6. Sanity** — additive section-9 asserts (oblateness,
  Lambert-phase shape, deterministic starfield, reflected-flux
  scaling); no pinned value touched.

### Staged — next phase (specified, NOT yet implemented)

- [ ] **S7. Body-centric view** — `render_body_view()` /
  `animate_body_view()`: any target (planet/moon/**star**) centred and
  detailed as an oblate body with a single-scattering **atmospheric
  halo / limb ring**; siblings + background circle it in a fixed
  camera. **Eclipse & transit shadow rays** via two-disk
  angular-overlap occultation (penumbra = partial, umbra = full
  illumination scaling) — reuse `sky_bodies` geometry. **Atmospheric
  refraction**: a target entering a *planet's* umbra is lit by that
  planet's Rayleigh-stripped limb-ring spectrum (reuse
  `atmosphere_for` + `_light_optical_depth` on a grazing slant ray) →
  physically-correct "blood-moon". Fully annotated/labelled MP4s
  (body names, alt/az, live "TRANSIT / TOTAL ECLIPSE" banner).

### Deferred — intentionally out of scope

- [ ] **Full procedural universe background.** Replace the lightweight
  `background_starfield` with a real **universe-initialisation
  simulation**: model the host galaxy and its neighbourhood (space
  voids, galaxies, nebulae, dust lanes, realistic stellar
  populations), generate it once and **cache at startup**, then feed
  it to the renderer. This is a separable, expensive concern (a
  galaxy-scale population synthesis + spatial cache), deliberately not
  built here so the sky renderer stays fast and self-contained; the
  procedural starfield is the grounded stand-in until then.

## Notes / decisions

- 4+ star systems: outer node collapsed to one effective `Star`
  (Σmass, ΣL, L-weighted Teff) to keep existing triple dynamics valid.
- Long-term nested N-body horizon is configurable; demo default sized to
  finish in minutes; report records the verified horizon explicitly.
- Core physics primitives untouched → all pinned `test_sanity.py` values
  preserved.

## Verification checklist

- [x] `uv run python test_sanity.py` — old values unchanged, section-9
  asserts pass.
- [x] `uv run python generate_solar_system.py` — ≥1 PHZ planet; giants have
  dozens-to-100+ moons; overview/zoom/stability PNGs written (MP4
  auto-skipped here: ffmpeg not installed in this environment).
- [x] `uv run python render_skyview.py` — Sol night is a starlit
  Milky-Way sky (not black) with the Moon at its correct phase;
  binary/triple show sibling stars; random systems show sibling
  planets + moons; `figures/skyview/sky_colour_gallery.png` contrasts
  the atmosphere regimes; noon Earth-analog visually unchanged.
- [x] Long-term stability verdict separates secular drift from bounded
  Hill-region libration; regular moons capped at 0.33·f_crit and
  irregulars at 0.60·f_crit so generated systems are robustly bound.
  Multi-seed confirmed STABLE, 0 flagged: 2027 (7pl/76mn), 11 (8pl/130mn),
  7 (9pl/145mn), 42 (7pl/224mn), 5 (7pl/108mn).

## Cost note

Per-system nested integration is the expensive step (it integrates
every dynamically-relevant moon under the full field). Driver default:
60 yr / 500 samples / 10 moons-per-planet ≈ ~110 s for one system.
`integrate_solar_system(duration_yr, n_samples,
max_integrated_moons_per_planet, sub_steps_floor)` are all tunable; the
remaining (non-integrated) moons are guaranteed stable by construction
(Roche + Domingos-2006 + mutual-Hill in moons.py).
