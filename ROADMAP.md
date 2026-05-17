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
