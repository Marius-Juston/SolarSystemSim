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

### Skyview v3 — body-centric view, high-detail stars, realistic moons

Done. Centred-body renderer with detailed stars/moons, eclipse/transit
occultation and the physically-correct refracted blood-moon. Physics
grounded in: NASA SVS 11418 "Solar Continuum" (multi-temperature solar
structure); convective granule size ~ pressure scale height
`H_p = kT/(mu m g) ~ T_eff/g` (Nordlund & Stein; Trampedach+2013;
Stagger-grid); quadratic limb darkening `I(mu)/I(1)=1−u1(1−mu)−u2(1−mu)²`
(Claret 2000; Sing 2010); activity–rotation spots (Noyes+1984;
Wright+2011); divergence-free curl noise (Bridson, Hourihan &
Nordenstam 2007); crater production `N(>D)∝D^−2` (Neukum, Ivanov &
Hartmann 2001) + saturation (Gault 1970); simple→complex transition
`D_sc ∝ 1/g` (Pike 1980; Melosh 1989); tidal-heating resurfacing
(Peale, Cassen & Reynolds 1979); Titan dunes / saltation threshold
(Lorenz+2006; Burr+2015, Nature); isostatic relief `h ∝ σ/(ρ g)`
(Melosh; Jeffreys); Jeans atmosphere retention (Catling & Kasting 2017).

- [x] **V1. `noise.py`** — vectorised seeded `value_noise_2d` / `fbm` and
  analytic divergence-free `curl_noise_2d` on the `xp` backend
  (Bridson+2007); cached lattice tables (`background_starfield`
  pattern).
- [x] **V2. `moon_surface.py`** — `MoonSurface` + `moon_surface_for`
  (parallel to `profile_for_planet`; `Moon` gains one additive
  `.surface` field): gravity, ice fraction, tidal-heating index,
  `D_sc`, crater retention, atmosphere (Jeans), dune coverage,
  isostatic relief, surface type. `render_moon_disk` shaded relief from
  a cached equirectangular crater/dune/ice texture (windowed crater
  stamping = disk-bbox trick).
- [x] **V3. `starsurface.py`** — `StarSurface` + `star_surface_for`
  (derived from `Star`, no new fields). `render_star_disk`: quadratic
  limb darkening, curl-advected granulation + supergranulation, cool
  spots / limb faculae, chromospheric limb ring, and analytic
  prominence/flare arcs with fading trails.
- [x] **V4. `bodyview.py`** — `render_body_view` / `render_body_still` /
  `animate_body_view`; `occult_fraction` exact two-disk lens area
  (transit dip + eclipse penumbra/umbra); `_refracted_spectrum`
  (Chapman grazing air-mass through `atmosphere_for`) → blood-moon;
  simplified banded/cloud planet shader; siblings + starfield reuse
  skyview helpers.
- [x] **V5. Driver** — `render_skyview.body_view_showcase()`: detailed
  Sun (granulation/spot/prominence) still + rotation MP4; cratered
  Moon + a scanned **total lunar eclipse** still + eclipse MP4; a
  giant + its moon; graceful ffmpeg-absent skip.
- [x] **V6. Sanity** — additive section-9 asserts: curl-noise
  divergence ≈ 0 & determinism; `D_sc`/relief ∝ 1/g; tidal-heating
  lowers crater retention; ice↔density; limb darkening monotone;
  granule size ∝ T_eff/g; `occult_fraction` exact; refraction is
  red-biased; body view is centred. No pinned value touched.

**Frame-purity constraint:** animation frames render in independent
`parallel` pool workers, so *all* time dependence is a pure function of
`t` (no inter-frame feedback buffer) — eruption "trails" are analytic
along-arc/`t` opacity falloff.

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
