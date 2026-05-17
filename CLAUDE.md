# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Dependencies are managed with `uv` (see `uv.lock`, `pyproject.toml`; requires Python >= 3.13, deps are just numpy + matplotlib).

- Install / sync env: `uv sync`
- Run sanity regression checks: `uv run python test_sanity.py` — this is a **plain script with print + assert statements, not pytest**. It prints computed vs. published reference values; there is no test runner and no single-test selector. Inspect output visually. New solar-system / moon / habitability code adds **additive asserts in "section 9"** — keep older pinned values unchanged.
- 10-system PHZ demo (static PNGs + MP4s): `uv run python demo.py`
- Random full solar-system generator (1-, 2-, 3-star, with moons + habitability + nested N-body stability): `uv run python generate_solar_system.py [seed]`
- Photorealistic ground-to-sky renderer (single / binary / triple sky): `uv run python render_skyview.py [seed]`
- All drivers write **repo-relative** to `figures/` and `animations/` (the old hardcoded `/home/claude/...` path is gone). **MP4 rendering requires `ffmpeg` on PATH**; drivers set `matplotlib.use("Agg")` and gracefully skip MP4s if ffmpeg is absent.
- `main.py` is a stub placeholder — not a real entry point.

## Architecture

Single Python package `goldilocks/` (import as `from goldilocks.<module> import ...`). It computes the **Permanently Habitable Zone (PHZ)** for single/binary/hierarchical-triple star systems, packs planets into the stable zone, generates full random solar systems (moons, habitability), verifies stability with nested N-body, and renders static + animated + photorealistic-sky output. `README.md` has the physics background/formulae/citations; `ROADMAP.md` tracks the random-solar-system feature phases. Read both for algorithm-level work.

### Layering (lower layers have no dependence on higher ones)

1. **Physics primitives** — pure, independently testable:
   - `stellar.py` — `Star` dataclass; Eker 2018 mass→luminosity/radius/T_eff.
   - `kepler.py` — closed-form 2-body Kepler solver, `orbital_period`.
   - `habitable_zone.py` — Kopparapu HZ flux limits, Mueller-Haghighipour multi-star weighted flux.
   - `stability.py` — Holman-Wiegert S/P-type cuts, Mardling-Aarseth triple stability, Hill-radius planet packing.
   - `roche.py` — Eggleton 1983 Roche-lobe geometry.
   - `secular.py` — Heppenheimer / Leung-Lee forced eccentricity, Laplace-Lagrange secular coupling.

2. **Orchestration** — `system.py` (`StarSystem`): keystone class tying primitives together. Constructed via `StarSystem.single/binary/hierarchical_triple`. Places stars on analytical Kepler orbits, computes combined PHZ, applies stability + Roche cuts, packs planets with secular coupling. `summary()` produces the text report.

3. **Bodies & catalogues**:
   - `planets.py` — `Planet` objects (incl. `moons`, `habitability`, `inclination_deg`, `bulk_density_gcc`, `is_gas_giant`) + real-system catalogue (Kepler-16/34/35/38/47, `earth_analog`).
   - `moons.py` — `Moon` dataclass; fluid Roche limit; Domingos+2006 critical Hill fraction; `generate_moons` (terrestrial 0–2, giants dozens–100+; Roche + Hill + mutual-Hill validated).
   - `random_systems.py` — generators for synthetic regimes (star-hopper, Trojan, wide hierarchy, polar planet).
   - `solar_system.py` — `random_solar_system`: Raghavan-2010 multiplicity PMF (56/33/8/3%); 4+-star systems collapse the outer node into one effective `Star` so triple dynamics stay valid; snow-line placement, mutual-Hill spacing, forced ≥1 PHZ terrestrial, retry loop.
   - `habitability.py` — `HabitabilityProfile` + `profile_for_planet`: gravity, insolation/T_eq, rotation, day/night, obliquity/seasons, magnetosphere, atmosphere + sky color, biosphere score. Each field from a published scaling law.

4. **Dynamics**:
   - `nbody.py` — stars on analytical Kepler orbits, planets integrated numerically (leapfrog, adaptive sub-stepping). Source of non-Keplerian planet motion in animations.
   - `nbody_moons.py` — nested KDK leapfrog (stars analytic, planets + moons numeric); vectorized all-pairs; osculating-element stability report separating secular drift from bounded libration.

5. **Output** (depends on everything above; touches no physics primitive, pins no sanity value):
   - `visualization.py` — `plot_system`: multi-panel static figure.
   - `animation.py` — `animate_system`: MP4 driven by `nbody` with moving phase-dependent HZ contours.
   - `viz_solar_system.py` — overview PNG, per-planet zoom cards, long-term stability PNG, system MP4.
   - `skyview.py` — single-scattering atmospheric RT (Nishita 1993); arbitrary stellar Planck spectra, atmosphere from `HabitabilityProfile`, any number of suns, CIE-XYZ→sRGB→ACES tone-map. `render_phases` + `animate_day`.

6. **Drivers** — `demo.py` (10 PHZ systems), `generate_solar_system.py`, `render_skyview.py`, `test_sanity.py`.

### Key conventions
- Units: AU, years, solar masses/luminosities throughout; angles in degrees at API boundaries.
- A `StarSystem` whose binary separation dwarfs any HZ emits a warning and is treated as effectively independent — preserve this when touching `system.py`/`stability.py`. Drivers pass `quiet=True` to suppress that warning where expected.
- 4+-star systems are modelled as a hierarchical triple with a collapsed effective outer `Star` (Σmass, ΣL, L-weighted Teff); the original count is recorded in `system.generation_note`.
- When changing any physics primitive, check the corresponding pinned reference value in `test_sanity.py` (and its section-9 additive asserts) still matches. Core primitives are deliberately kept untouched by the solar-system feature so all pinned values are preserved.
