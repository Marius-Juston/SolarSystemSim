"""
solar_system.py
----------------
Random *full* solar-system generator.

Star multiplicity is drawn from the Raghavan et al. 2010 (ApJS 190, 1;
arXiv:1007.0414) "A Survey of Stellar Families" result for solar-type
primaries:

    1 star : 56%      2 stars : 33%
    3 stars:  8%      4+ stars:  3%

Stellar dynamics in StarSystem support up to a hierarchical triple, so a
4+ system is modelled as a hierarchical triple whose outer node is an
unresolved tight pair *collapsed into one effective Star* (summed mass,
summed luminosity, luminosity-weighted Teff).  The system still "counts"
as four stars (recorded in `sys.generation_note`) but the closed-form
triple dynamics stay exactly valid.

The generator then:
  * finds the stable Permanently Habitable Zone band (reusing
    StarSystem.count_habitable_planets, unchanged),
  * lays terrestrial planets inside and gas/ice giants beyond the snow
    line a_snow ~ 2.7 sqrt(L) AU (Hayashi 1981),
  * *forces* one Earth-class planet into the PHZ so the system always
    has >= 1 habitable world,
  * attaches a moon system (moons.generate_moons) and a habitability
    profile (habitability.profile_for_planet) to every planet,
  * retries until the PHZ constraint is met.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from goldilocks import stability as stab
from goldilocks.habitability import profile_for_planet
from goldilocks.moons import generate_moons
from goldilocks.planets import Planet, radius_from_mass_me, is_gas_giant
from goldilocks.random_systems import (_random_mass,
                                       _random_binary_separation,
                                       _random_binary_eccentricity)
from goldilocks.stellar import Star
from goldilocks.system import StarSystem

# Smith & Lissauer 2009 long-term (Lagrange/Gyr) packing separation, in
# mutual Hill radii.  >= ~8-10 is required for multi-Gyr stability.
HILL_SPACING_DELTA = 9.0

STAR_COUNT_PMF = {1: 0.56, 2: 0.33, 3: 0.08, 4: 0.03}  # Raghavan+2010


# ---------------------------------------------------------------------
# Stellar layout
# ---------------------------------------------------------------------
def _draw_n_stars(rng: np.random.Generator) -> int:
    ks = list(STAR_COUNT_PMF)
    ps = np.array([STAR_COUNT_PMF[k] for k in ks], dtype=float)
    ps /= ps.sum()
    return int(rng.choice(ks, p=ps))


def _effective_star(name: str, m1: float, m2: float) -> Star:
    """Collapse an unresolved tight pair into one point-mass Star."""
    a = Star("_a", mass=m1)
    b = Star("_b", mass=m2)
    L = a.luminosity + b.luminosity
    teff = (a.teff * a.luminosity + b.teff * b.luminosity) / L
    return Star(name, mass=m1 + m2, luminosity=L, teff=teff)


def _build_stars(rng: np.random.Generator,
                 name: str,
                 n_stars: int | None = None) -> Tuple[StarSystem, str]:
    n = _draw_n_stars(rng) if n_stars is None else n_stars

    if n == 1:
        s = Star("Star A", mass=_random_mass(rng, 0.30, 1.50))
        return StarSystem.single(name, s), "single star"

    if n == 2:
        m1 = _random_mass(rng, 0.5, 1.4)
        m2 = m1 * float(rng.uniform(0.3, 1.0))
        kind = rng.choice(["tight", "wide"])  # circumbinary or S-type
        a = _random_binary_separation(rng, kind=kind)
        e = _random_binary_eccentricity(rng, a)
        A = Star("Star A", mass=m1)
        B = Star("Star B", mass=m2)
        return (StarSystem.binary(name, A, B, separation_au=a,
                                  eccentricity=e, quiet=True),
                f"binary ({kind})")

    # n == 3 or 4 -> hierarchical triple
    m_in_a = _random_mass(rng, 0.7, 1.4)
    m_in_b = m_in_a * float(rng.uniform(0.3, 0.95))
    a_in = float(rng.uniform(0.1, 1.2))
    e_in = float(rng.uniform(0.0, 0.3))
    A = Star("Star A", mass=m_in_a)
    B = Star("Star B", mass=m_in_b)
    if n == 3:
        C = Star("Star C", mass=_random_mass(rng, 0.2, 0.8))
        note = "hierarchical triple"
    else:
        mc1 = _random_mass(rng, 0.3, 0.9)
        mc2 = mc1 * float(rng.uniform(0.3, 0.9))
        C = _effective_star("Star C+D", mc1, mc2)
        note = "quadruple (outer node = unresolved tight pair)"
    # Outer SMA chosen well beyond the Mardling-Aarseth critical ratio.
    a_out = a_in * float(rng.uniform(8.0, 40.0)) + float(rng.uniform(20, 120))
    e_out = float(rng.uniform(0.0, 0.4))
    sysobj = StarSystem.hierarchical_triple(
        name, A, B, C, a_in=a_in, e_in=e_in,
        a_out=a_out, e_out=e_out, quiet=True)
    return sysobj, note


# ---------------------------------------------------------------------
# PHZ band selection (reuses count_habitable_planets unchanged)
# ---------------------------------------------------------------------
def _pick_phz_band(sys: StarSystem
                   ) -> Optional[Tuple[str, Optional[int], Tuple[float, float]]]:
    """Return ('S', star_idx, (in,out)) or ('P', None, (in,out)) for the
    widest stable PHZ band, or None if no stable band exists."""
    res = sys.count_habitable_planets(planet_mass_me=1.0, delta=10.0,
                                      optimistic=False, use_phz=True)
    best = None
    for i, entry in enumerate(res["stars"]):
        b = entry.get("stable_HZ")
        if b and b[1] > b[0]:
            w = b[1] - b[0]
            if best is None or w > best[0]:
                best = (w, ("S", i, (float(b[0]), float(b[1]))))
    cb = res.get("circumbinary")
    if cb and cb.get("stable_HZ"):
        b = cb["stable_HZ"]
        if b[1] > b[0]:
            w = b[1] - b[0]
            if best is None or w > best[0]:
                best = (w, ("P", None, (float(b[0]), float(b[1]))))
    return best[1] if best else None


# ---------------------------------------------------------------------
# Planet placement
# ---------------------------------------------------------------------
def _draw_mass(a: float, snow_line: float,
               rng: np.random.Generator) -> float:
    giant = a > snow_line and rng.random() < 0.75
    if giant:
        return float(10 ** rng.uniform(math.log10(30.0),
                                       math.log10(2000.0)))
    return float(10 ** rng.uniform(math.log10(0.3), math.log10(6.0)))


def _make_planet(name: str, a: float, m: float, host_idx: Optional[int],
                 rng: np.random.Generator) -> Planet:
    e = float(min(0.4, abs(rng.normal(0.0, 0.03))))
    inc = float(abs(rng.normal(0.0, 2.0)))
    return Planet(name=name, mass_me=m, radius_re=radius_from_mass_me(m),
                  semi_major_axis_au=a, eccentricity=e,
                  host_star_index=host_idx, inclination_deg=inc,
                  description="generated", real_planet=False)


def _populate_planets(sys: StarSystem,
                      band_kind: str, host_idx: Optional[int],
                      band: Tuple[float, float],
                      rng: np.random.Generator) -> List[Planet]:
    if band_kind == "S":
        L = sys.stars[host_idx].luminosity
        host_mass = sys.stars[host_idx].mass
        phost: Optional[int] = host_idx
    else:  # circumbinary
        L = sys.total_luminosity()
        host_mass = sum(s.mass for s in sys.stars)
        phost = None
    snow = 2.7 * math.sqrt(max(L, 1e-4))
    b_in, b_out = band
    a_phz = math.sqrt(b_in * b_out)  # guaranteed-in-PHZ slot

    a_lo = max(0.04, 0.35 * b_in)
    a_hi = max(6.0 * snow, 3.0 * b_out)

    # Sequentially place planets enforcing >= HILL_SPACING_DELTA mutual
    # Hill radii between neighbours (Smith & Lissauer 2009), using each
    # planet's *actual* mass -- so a massive giant pushes its neighbours
    # genuinely far out instead of overlapping.
    slots: List[Tuple[float, float]] = []  # (a, mass_me)
    a = a_lo
    guard = 0
    while a <= a_hi and guard < 400:
        guard += 1
        m = _draw_mass(a, snow, rng)
        if slots:
            pa, pm = slots[-1]
            Rh = stab.mutual_hill_radius(pm, m, pa, a, host_mass)
            min_a = pa + HILL_SPACING_DELTA * Rh
            if a < min_a:
                a = min_a
                continue
        slots.append((a, m))
        a *= float(rng.uniform(1.3, 1.8))

    # Force a 1-Me terrestrial into the PHZ.  Replace the nearest slot if
    # one lies close, else insert (its tiny Hill radius rarely conflicts).
    if slots:
        ci = min(range(len(slots)),
                 key=lambda k: abs(math.log(slots[k][0] / a_phz)))
        if abs(math.log(slots[ci][0] / a_phz)) < 0.18:
            slots[ci] = (a_phz, 1.0)
        else:
            slots.append((a_phz, 1.0))
    else:
        slots.append((a_phz, 1.0))
    slots.sort()

    # Reject if the forced terrestrial would sit within the Hill reach of
    # a massive neighbour (let the outer retry loop pick a new system).
    for k, (sa, sm) in enumerate(slots):
        if abs(sa - a_phz) > 1e-12:
            continue
        for j in (k - 1, k + 1):
            if 0 <= j < len(slots):
                na, nm = slots[j]
                Rh = stab.mutual_hill_radius(1.0, nm, sa, na, host_mass)
                if abs(na - sa) < HILL_SPACING_DELTA * Rh:
                    return []

    planets: List[Planet] = []
    for i, (sa, sm) in enumerate(slots):
        if abs(sa - a_phz) < 1e-12 and abs(sm - 1.0) < 1e-12:
            p = Planet(name=f"Planet {_letter(i)}", mass_me=1.0,
                       radius_re=1.0, semi_major_axis_au=sa,
                       eccentricity=float(min(0.03, abs(rng.normal(0, 0.015)))),
                       host_star_index=phost, inclination_deg=2.0,
                       description="forced PHZ terrestrial",
                       real_planet=False)
        else:
            p = _make_planet(f"Planet {_letter(i)}", sa, sm, phost, rng)
        planets.append(p)

    sys.planets = planets
    for p in planets:
        p.moons = generate_moons(p, host_mass, rng)
    return planets


def _in_band(p: Planet, band: Tuple[float, float]) -> bool:
    peri = p.semi_major_axis_au * (1.0 - p.eccentricity)
    apo = p.semi_major_axis_au * (1.0 + p.eccentricity)
    return peri >= band[0] and apo <= band[1]


_LETTERS = "bcdefghijklmnopqrstuvwxyz"


def _letter(i: int) -> str:
    return _LETTERS[i] if i < len(_LETTERS) else f"p{i}"


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------
def random_solar_system(rng: Optional[np.random.Generator] = None,
                        name: Optional[str] = None,
                        max_tries: int = 50,
                        n_stars: int | None = None) -> StarSystem:
    """Generate a random solar system with >= 1 planet in the PHZ."""
    if rng is None:
        rng = np.random.default_rng()
    for attempt in range(max_tries):
        nm = name or f"Random system {rng.integers(1000, 9999)}"
        sys, note = _build_stars(rng, nm, n_stars=n_stars)
        band = _pick_phz_band(sys)
        if band is None:
            continue
        kind, host_idx, bnd = band
        planets = _populate_planets(sys, kind, host_idx, bnd, rng)
        if not planets:
            continue
        # Tag PHZ membership + attach habitability profiles.
        n_phz = 0
        for p in planets:
            inp = _in_band(p, bnd) and not is_gas_giant(p)
            if inp:
                n_phz += 1
            p.habitability = profile_for_planet(p, sys, rng, in_phz=inp)
        if n_phz >= 1:
            sys.generation_note = (
                f"{note}; PHZ {kind}-type band "
                f"[{bnd[0]:.3f}, {bnd[1]:.3f}] AU; "
                f"{len(planets)} planets, {n_phz} in PHZ")
            sys.phz_band = (kind, host_idx, bnd)
            return sys
    raise RuntimeError(
        f"Failed to generate a system with a PHZ planet in {max_tries} tries.")
