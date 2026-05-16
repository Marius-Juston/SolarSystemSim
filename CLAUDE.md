# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Dependencies are managed with `uv` (see `uv.lock`, `pyproject.toml`; requires Python >= 3.13, deps are just numpy + matplotlib).

- Install / sync env: `uv sync`
- Run sanity regression checks: `uv run python test_sanity.py` — this is a **plain script with print + assert statements, not pytest**. It prints computed vs. published reference values; there is no test runner and no single-test selector. Inspect output visually.
- Run the full 10-system demo (static PNGs + MP4s): `uv run python demo.py`. **MP4 rendering requires `ffmpeg` on PATH.** `matplotlib.use("Agg")` is set so it runs headless.
- `main.py` is a stub (`uv run` entrypoint placeholder) — not the real entry point.

### Known gotcha
`demo.py` hardcodes output to `/home/claude/goldilocks/figures` and `/home/claude/goldilocks/animations` (a path from the original authoring environment, not this repo). Change `OUT_FIG`/`OUT_ANI` at the top of `demo.py` before running here, or it will write outside the project tree.

## Architecture

This is a flat single-package astrophysics calculator: it computes the **Permanently Habitable Zone (PHZ)** for single/binary/hierarchical-triple star systems, packs Earth-mass planets into the stable zone, and renders static + animated visualizations with real N-body planet dynamics. The README has the full physics background, formulae, and citations — read it for any algorithm-level work.

### Layering (lower layers have no dependence on higher ones)

1. **Physics primitives** — pure functions/dataclasses, independently testable:
   - `stellar.py` — `Star` dataclass; Eker 2018 mass→luminosity/radius/T_eff.
   - `kepler.py` — closed-form 2-body Kepler solver, `orbital_period`.
   - `habitable_zone.py` — Kopparapu HZ flux limits, Mueller-Haghighipour multi-star weighted flux.
   - `stability.py` — Holman-Wiegert S/P-type cuts, Mardling-Aarseth triple stability, Hill-radius planet packing.
   - `roche.py` — Eggleton 1983 Roche-lobe geometry.
   - `secular.py` — Heppenheimer / Leung-Lee forced eccentricity, Laplace-Lagrange secular coupling.

2. **Orchestration** — `system.py` (`StarSystem`): the keystone class that ties all primitives together. Constructed via classmethods `StarSystem.single/binary/hierarchical_triple`. Places stars on analytical Kepler orbits, computes the combined PHZ band, applies stability + Roche cuts, then packs planets with secular coupling (`count_habitable_planets`, `_pack_with_secular`). `summary()` produces the text report.

3. **Dynamics & catalogues**:
   - `planets.py` — `Planet` objects + real-system catalogue (Kepler-16/34/35/38/47, earth_analog).
   - `nbody.py` — N-body integrator: **stars move on analytical Kepler orbits, planets are integrated numerically** (leapfrog, adaptive per-planet sub-stepping). This is what produces non-Keplerian planet motion in animations.
   - `random_systems.py` — generators for the synthetic regimes (star-hopper, Trojan/co-orbital, wide hierarchy, polar planet).

4. **Output**:
   - `visualization.py` — `plot_system`: multi-panel static figure (top-down, side, eccentricity envelope, wide-view panel for hierarchies).
   - `animation.py` — `animate_system`: time-evolution MP4 driven by the `nbody` integrator with moving phase-dependent HZ contours.

5. **Drivers** — `demo.py` (10 example systems via the `make()` helper), `test_sanity.py` (regression vs. published values).

### Key conventions
- Units: AU, years, solar masses/luminosities throughout; angles in degrees at API boundaries.
- A `StarSystem` whose "binary" separation dwarfs any HZ emits a warning and is treated as effectively independent — preserve this behavior when touching `system.py`/`stability.py`.
- When changing any physics primitive, check the corresponding reference value in `test_sanity.py` still matches (values are pinned to published literature, listed in README's sanity-check table).
