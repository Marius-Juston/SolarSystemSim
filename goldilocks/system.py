"""
system.py
---------
The top-level container for a complete N-star, N-planet configuration.

This module ties together
-------------------------
* Closed-form Kepler orbits for single-star and binary stellar
  configurations (kepler.py).
* Hierarchical triples that pass the Mardling-Aarseth stability test
  (stability.py); the inner binary is solved exactly, the outer
  companion is treated on its own Kepler orbit around the inner pair.
* The Permanently Habitable Zone (PHZ) of Eggl et al. 2012 -- a planet
  is *permanently* habitable iff its periastron AND apastron stay
  inside the Kopparapu inner/outer thresholds for every binary phase.
* Holman-Wiegert dynamical-stability cuts for S- and P-type planets.
* Roche-lobe (Eggleton 1983) tidal-disruption cut for S-type orbits
  near close binaries.
* Laplace-Lagrange multi-planet eccentricity coupling for any set of
  planets in the same zone (secular.py).
* The planet catalogue (planets.py) so the same code answers
  "how many Earth-like planets fit?" and "is Kepler-47 c habitable?".

The intent is that for **N = 1 or 2 stars** we use exact closed-form
solutions everywhere; for **N >= 3** we still solve the inner binary
exactly and place the outer companion on a Kepler orbit around the
barycentre, which is accurate as long as Mardling-Aarseth is satisfied
(otherwise we warn the user).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from goldilocks import habitable_zone as hz
from goldilocks import roche
from goldilocks import secular as sec
from goldilocks import stability as stab
from goldilocks.kepler import (kepler_two_body, orbital_period)
from goldilocks.planets import (Planet)
from goldilocks.stellar import Star


# -----------------------------------------------------------------------
# Star system
# -----------------------------------------------------------------------
@dataclass
class StarSystem:
    """N-star, N-planet system with Kepler orbits where possible."""
    name: str
    stars: List[Star]
    # Pairwise stellar orbital edges: (i, j, a, e, omega). j = -1 means
    # 'orbits the centre of mass of all preceding stars' (used for the
    # outer companion of a hierarchical triple).
    stellar_orbits: List[Tuple[int, int, float, float, float]] = \
        field(default_factory=list)
    planets: List[Planet] = field(default_factory=list)
    # Verbose stability warnings.
    quiet: bool = False

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def single(cls, name: str, star: Star,
               planets: Optional[List[Planet]] = None) -> "StarSystem":
        star.position = (0.0, 0.0, 0.0)
        star.velocity = (0.0, 0.0, 0.0)
        sys = cls(name=name, stars=[star], planets=planets or [])
        return sys

    @classmethod
    def binary(cls, name: str,
               primary: Star, secondary: Star,
               separation_au: float, eccentricity: float = 0.0,
               omega: float = 0.0,
               planets: Optional[List[Planet]] = None,
               quiet: bool = False) -> "StarSystem":
        """Two-star system on a Kepler orbit.  Stars are placed at
        periastron (t=0) of the relative orbit.

        Sanity-checks that the binary is close enough to actually
        perturb the HZ of either component: if the secondary lies
        outside ~50 x the optimistic outer HZ of the primary, we warn.
        """
        sys = cls(name=name, stars=[primary, secondary],
                  stellar_orbits=[(0, 1, separation_au, eccentricity, omega)],
                  planets=planets or [], quiet=quiet)
        sys._update_stellar_positions(t=0.0)

        # Sanity check: warn if the binary is "wide" relative to the HZ
        # of either component.
        outer_hz_p = hz.single_star_hz(primary).optimistic_outer
        outer_hz_s = hz.single_star_hz(secondary).optimistic_outer
        bigger_hz = max(outer_hz_p, outer_hz_s)
        if separation_au * (1.0 - eccentricity) > 50.0 * bigger_hz and not quiet:
            warnings.warn(
                f"Binary {name!r}: closest approach "
                f"{separation_au * (1 - eccentricity):.2f} AU is >> 50x the "
                f"HZ extent {bigger_hz:.2f} AU.  Stars are effectively "
                "independent for HZ analysis.")
        return sys

    @classmethod
    def hierarchical_triple(cls, name: str,
                            inner_a: Star, inner_b: Star, outer: Star,
                            a_in: float, e_in: float,
                            a_out: float, e_out: float,
                            i_mut_deg: float = 0.0,
                            planets: Optional[List[Planet]] = None,
                            quiet: bool = False) -> "StarSystem":
        """Hierarchical triple.  Inner binary (stars 0,1) on Kepler orbit
        with SMA a_in, eccentricity e_in.  Outer companion (star 2) on
        a Kepler orbit with SMA a_out, eccentricity e_out around the
        inner-pair barycentre."""
        sys = cls(name=name, stars=[inner_a, inner_b, outer],
                  stellar_orbits=[
                      (0, 1, a_in, e_in, 0.0),
                      (2, -1, a_out, e_out, math.pi / 2),
                  ],
                  planets=planets or [], quiet=quiet)
        sys._update_stellar_positions(t=0.0)

        # Mardling-Aarseth stability check.
        if not stab.mardling_aarseth_stable(a_in, a_out, e_out,
                                            inner_a.mass, inner_b.mass,
                                            outer.mass, i_mut_deg):
            crit = stab.mardling_aarseth_ratio(a_in, e_out,
                                               inner_a.mass, inner_b.mass,
                                               outer.mass, i_mut_deg)
            msg = (f"Triple {name!r} fails Mardling-Aarseth stability:"
                   f"  a_out/a_in = {a_out / a_in:.2f} < critical "
                   f"{crit:.2f}.")
            if not quiet:
                warnings.warn(msg)
            else:
                pass
        return sys

    # ------------------------------------------------------------------
    # Kepler orbit machinery
    # ------------------------------------------------------------------
    def _update_stellar_positions(self, t: float) -> None:
        """Update star positions/velocities from the closed-form Kepler
        solution at time t (years).  Works for N = 1, 2, or hierarchical
        triple; for true N >= 3 this requires the orbits be edge-supplied
        and Mardling-Aarseth-stable."""
        n = len(self.stars)
        if n == 1:
            self.stars[0].position = (0.0, 0.0, 0.0)
            self.stars[0].velocity = (0.0, 0.0, 0.0)
            return

        # Binary or inner-binary: edge (i, j, a, e, omega)
        edge = self.stellar_orbits[0]
        i, j, a, e, omega = edge
        m1, m2 = self.stars[i].mass, self.stars[j].mass
        r1, r2, v1, v2 = kepler_two_body(m1, m2, a, e, t,
                                         omega=omega, t_peri=0.0)
        self.stars[i].position = tuple(r1)
        self.stars[j].position = tuple(r2)
        self.stars[i].velocity = tuple(v1)
        self.stars[j].velocity = tuple(v2)

        # Outer companion (hierarchical triple): edge (2, -1, a_out, e_out, omega)
        if n >= 3 and len(self.stellar_orbits) >= 2:
            edge = self.stellar_orbits[1]
            k, ref, a_o, e_o, omega_o = edge
            m_inner = m1 + m2
            m_outer = self.stars[k].mass
            r_in_bc, r_out_bc, v_in_bc, v_out_bc = kepler_two_body(
                m_inner, m_outer, a_o, e_o, t,
                omega=omega_o, t_peri=0.0)
            # Shift the inner-binary stars by r_in_bc (so the inner
            # barycentre is at r_in_bc, not at origin).
            pi = np.array(self.stars[i].position)
            pj = np.array(self.stars[j].position)
            vi = np.array(self.stars[i].velocity)
            vj = np.array(self.stars[j].velocity)
            self.stars[i].position = tuple(pi + r_in_bc)
            self.stars[j].position = tuple(pj + r_in_bc)
            self.stars[i].velocity = tuple(vi + v_in_bc)
            self.stars[j].velocity = tuple(vj + v_in_bc)
            self.stars[k].position = tuple(r_out_bc)
            self.stars[k].velocity = tuple(v_out_bc)

    def stellar_orbit_period(self, edge_idx: int = 0) -> float:
        """Period (yr) of the orbital edge with the given index."""
        edge = self.stellar_orbits[edge_idx]
        i, j, a, e, _ = edge
        if j == -1:
            m_inner = sum(s.mass for s in self.stars[:-1])
            m_outer = self.stars[-1].mass
            return orbital_period(m_inner, m_outer, a)
        return orbital_period(self.stars[i].mass, self.stars[j].mass, a)

    def total_luminosity(self) -> float:
        return sum(s.luminosity for s in self.stars)

    def barycentre(self) -> np.ndarray:
        total = sum(s.mass for s in self.stars)
        return sum(s.mass * np.array(s.position) for s in self.stars) / total

    # ------------------------------------------------------------------
    # Time-dependent HZ map
    # ------------------------------------------------------------------
    def hz_grid(self, extent_au: float, n: int = 401,
                t: float = 0.0,
                optimistic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Snapshot HZ map in the orbital plane at time t."""
        self._update_stellar_positions(t)
        x = np.linspace(-extent_au, extent_au, n)
        y = np.linspace(-extent_au, extent_au, n)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        grid = np.stack([X, Y, Z], axis=-1)
        mask = hz.hz_mask(grid, self.stars, optimistic=optimistic)
        return X, Y, mask

    # ------------------------------------------------------------------
    # Permanently Habitable Zone (Eggl et al. 2012)
    # ------------------------------------------------------------------
    def _flux_thresholds(self, optimistic: bool) -> Tuple[float, float]:
        inner_lim = "RecentVenus" if optimistic else "RunawayGreenhouse"
        outer_lim = "EarlyMars" if optimistic else "MaxGreenhouse"
        return (float(hz._SEFF_SUN[hz.LIMIT_INDEX[inner_lim]]),
                float(hz._SEFF_SUN[hz.LIMIT_INDEX[outer_lim]]))

    def _stype_phz_for_circular_planet(
            self, host_index: int,
            optimistic: bool,
            n_phase: int = 24,
    ) -> Tuple[float, float]:
        """Find the conservative (inner, outer) HZ around stars[host_index]
        as a function of time: a planet on a circular orbit is permanently
        habitable iff it stays inside the snapshot HZ for EVERY phase of
        the binary's eccentric orbit.

        This is the "PHZ for e_p = 0" version; full Eggl 2012 requires
        a forced-eccentricity correction added later.
        """
        host = self.stars[host_index]
        # Periastron and apastron of the binary give the extreme cases.
        # We sample n_phase mean anomalies between 0 and 2*pi.
        if len(self.stars) == 1:
            bnd = hz.single_star_hz(host)
            return ((bnd.optimistic_inner, bnd.optimistic_outer)
                    if optimistic
                    else (bnd.conservative_inner, bnd.conservative_outer))

        edge_idx = 0
        P = self.stellar_orbit_period(edge_idx)
        s_in_thr, s_out_thr = self._flux_thresholds(optimistic)
        inner_lim = "RecentVenus" if optimistic else "RunawayGreenhouse"
        outer_lim = "EarlyMars" if optimistic else "MaxGreenhouse"

        # The maximum total flux a circular S-type planet receives is
        # bounded above when the secondary is at periastron of the
        # binary AND closest to the planet (planet between the two stars).
        # We sample (binary phase, planet phase) on a grid and find the
        # extreme distances at which a planet is always habitable.
        host_indep_inner = hz.hz_distance(host, inner_lim)
        host_indep_outer = hz.hz_distance(host, outer_lim)

        # Start from the snapshot HZ and contract until habitable at all phases.
        # Iterate over binary phases (the secondary's phase) and over
        # planet phase (theta in [0, 2 pi)).
        n_phase_planet = 48
        thetas = np.linspace(0, 2 * math.pi, n_phase_planet, endpoint=False)
        times = np.linspace(0.0, P, n_phase, endpoint=False)

        # Search the inner edge upward and the outer edge downward from
        # the single-star solution.
        candidate_in = host_indep_inner
        candidate_out = host_indep_outer

        # Scan radii in log space; mark "habitable at all phases" radii.
        radii = np.linspace(0.5 * host_indep_inner,
                            1.5 * host_indep_outer, 200)

        s_in_w = np.array([hz.spectral_weight(s.teff, inner_lim)
                           for s in self.stars])
        s_out_w = np.array([hz.spectral_weight(s.teff, outer_lim)
                            for s in self.stars])

        # Precompute star positions at all sampled times.
        positions_per_time = []
        for t in times:
            self._update_stellar_positions(t)
            positions_per_time.append(
                np.array([s.position for s in self.stars]))
        positions_per_time = np.array(positions_per_time)  # (T, N, 3)

        habitable_radius = np.ones(len(radii), dtype=bool)
        for ri, r in enumerate(radii):
            # For this radius around host, sample planet positions over theta;
            # planet pos = host_pos + r*(cos theta, sin theta, 0)
            host_positions = positions_per_time[:, host_index, :]  # (T, 3)
            ok = True
            for theta in thetas:
                planet_pos = host_positions + r * np.array(
                    [math.cos(theta), math.sin(theta), 0.0])  # broadcasts (T,3)
                # Compute weighted flux from all stars for each time t.
                flux_in = np.zeros(positions_per_time.shape[0])
                flux_out = np.zeros_like(flux_in)
                for si in range(len(self.stars)):
                    star_pos = positions_per_time[:, si, :]
                    diff = planet_pos - star_pos
                    r2 = np.einsum("ij,ij->i", diff, diff)
                    r2 = np.where(r2 > 1e-12, r2, np.nan)
                    flux_in += s_in_w[si] * self.stars[si].luminosity / r2
                    flux_out += s_out_w[si] * self.stars[si].luminosity / r2
                if (np.any(flux_in > s_in_thr) or
                        np.any(flux_out < s_out_thr)):
                    ok = False
                    break
            habitable_radius[ri] = ok

        if not habitable_radius.any():
            return None
        idxs = np.where(habitable_radius)[0]
        groups = np.split(idxs, np.where(np.diff(idxs) > 1)[0] + 1)
        biggest = max(groups, key=len)
        return float(radii[biggest[0]]), float(radii[biggest[-1]])

    def _ptype_phz_for_circular_planet(self,
                                       optimistic: bool,
                                       n_phase: int = 24,
                                       ) -> Tuple[float, float]:
        """Same as above but for circumbinary (P-type) orbits."""
        if len(self.stars) == 1:
            return None
        edge_idx = 0
        P = self.stellar_orbit_period(edge_idx)
        s_in_thr, s_out_thr = self._flux_thresholds(optimistic)
        inner_lim = "RecentVenus" if optimistic else "RunawayGreenhouse"
        outer_lim = "EarlyMars" if optimistic else "MaxGreenhouse"

        # Equivalent single-star scan radii based on total luminosity.
        L_tot = self.total_luminosity()
        r_outer_guess = math.sqrt(L_tot / hz._SEFF_SUN[hz.LIMIT_INDEX[outer_lim]])
        r_inner_guess = math.sqrt(L_tot / hz._SEFF_SUN[hz.LIMIT_INDEX[inner_lim]])

        n_phase_planet = 48
        thetas = np.linspace(0, 2 * math.pi, n_phase_planet, endpoint=False)
        times = np.linspace(0.0, P, n_phase, endpoint=False)
        s_in_w = np.array([hz.spectral_weight(s.teff, inner_lim)
                           for s in self.stars])
        s_out_w = np.array([hz.spectral_weight(s.teff, outer_lim)
                            for s in self.stars])

        positions_per_time = []
        for t in times:
            self._update_stellar_positions(t)
            positions_per_time.append(
                np.array([s.position for s in self.stars]))
        positions_per_time = np.array(positions_per_time)

        radii = np.linspace(0.5 * r_inner_guess, 1.5 * r_outer_guess, 250)
        bary = np.zeros((positions_per_time.shape[0], 3))
        masses = np.array([s.mass for s in self.stars])
        for ti in range(positions_per_time.shape[0]):
            bary[ti] = np.average(positions_per_time[ti], axis=0,
                                  weights=masses)

        habitable = np.ones(len(radii), dtype=bool)
        for ri, r in enumerate(radii):
            ok = True
            for theta in thetas:
                planet_pos = bary + r * np.array(
                    [math.cos(theta), math.sin(theta), 0.0])
                flux_in = np.zeros(positions_per_time.shape[0])
                flux_out = np.zeros_like(flux_in)
                for si in range(len(self.stars)):
                    star_pos = positions_per_time[:, si, :]
                    diff = planet_pos - star_pos
                    r2 = np.einsum("ij,ij->i", diff, diff)
                    r2 = np.where(r2 > 1e-12, r2, np.nan)
                    flux_in += s_in_w[si] * self.stars[si].luminosity / r2
                    flux_out += s_out_w[si] * self.stars[si].luminosity / r2
                if (np.any(flux_in > s_in_thr) or
                        np.any(flux_out < s_out_thr)):
                    ok = False
                    break
            habitable[ri] = ok

        if not habitable.any():
            return None
        idxs = np.where(habitable)[0]
        groups = np.split(idxs, np.where(np.diff(idxs) > 1)[0] + 1)
        biggest = max(groups, key=len)
        return float(radii[biggest[0]]), float(radii[biggest[-1]])

    # ------------------------------------------------------------------
    # Apply dynamical-stability and Roche-lobe cuts
    # ------------------------------------------------------------------
    def _stype_stable_hz(self, host_index: int,
                         phz_band: Tuple[float, float]
                         ) -> Optional[Tuple[float, float]]:
        host = self.stars[host_index]
        in_au, out_au = phz_band

        # Apply Holman-Wiegert outer cut and Roche-lobe inner cut for
        # every other star (treating each as a binary perturber).
        for j, other in enumerate(self.stars):
            if j == host_index:
                continue
            # Try to get an explicit binary orbit (i, j, a, e) for this pair.
            a_b = None
            e_b = 0.0
            for edge in self.stellar_orbits:
                idx_a, idx_b, a, e, _ = edge
                if {idx_a, idx_b} == {host_index, j}:
                    a_b, e_b = a, e
                    break
            if a_b is None:
                # Fall back to current separation.
                a_b = float(np.linalg.norm(np.array(other.position) -
                                           np.array(host.position)))
            mu = other.mass / (host.mass + other.mass)
            a_c = stab.holman_wiegert_stype(mu, e_b) * a_b
            out_au = min(out_au, a_c)

            # Roche-lobe periastron limit (tidal stripping of close orbits).
            r_L = roche.roche_lobe_periastron(host.mass, other.mass, a_b, e_b)
            # No planet survives near r_L; conservatively cut to 0.5 r_L.
            out_au = min(out_au, 0.5 * r_L)

            if out_au <= in_au:
                return None
        return in_au, out_au

    def _ptype_stable_hz(self, phz_band: Tuple[float, float]
                         ) -> Optional[Tuple[float, float]]:
        if len(self.stars) < 2 or not self.stellar_orbits:
            return None
        edge = self.stellar_orbits[0]
        idx_a, idx_b, a_b, e_b, _ = edge
        m1, m2 = self.stars[idx_a].mass, self.stars[idx_b].mass
        mu = min(m1, m2) / (m1 + m2)
        a_c = stab.holman_wiegert_ptype(mu, e_b) * a_b
        in_au, out_au = phz_band
        in_au = max(in_au, a_c)
        if in_au >= out_au:
            return None
        return in_au, out_au

    # ------------------------------------------------------------------
    # Packing planets with Laplace-Lagrange cross-checks
    # ------------------------------------------------------------------
    def count_habitable_planets(self,
                                planet_mass_me: float = 1.0,
                                delta: float = 10.0,
                                optimistic: bool = False,
                                use_phz: bool = True
                                ) -> dict:
        """Compute the maximum number of equal-mass planets that fit in
        each star's (S-type) PHZ and around the whole system (P-type).

        Includes:
          - Heppenheimer forced eccentricity for S-type planets,
          - Leung-Lee forced eccentricity for P-type planets,
          - Laplace-Lagrange mutual planet coupling for the packed planets,
          - rejects a packing if any planet's e_max takes it outside the
            instantaneous HZ at apastron or periastron.
        """
        out = {"system": self.name, "stars": [], "circumbinary": None,
               "use_phz": use_phz}

        # Per-star S-type
        for i, host in enumerate(self.stars):
            if use_phz and len(self.stars) > 1:
                phz = self._stype_phz_for_circular_planet(i, optimistic)
            else:
                bnd = hz.single_star_hz(host)
                phz = ((bnd.optimistic_inner, bnd.optimistic_outer)
                       if optimistic
                       else (bnd.conservative_inner, bnd.conservative_outer))
            if phz is None:
                out["stars"].append({"name": host.name, "stable_HZ": None,
                                     "n_planets": 0, "positions": [],
                                     "e_max": []})
                continue
            stable = self._stype_stable_hz(i, phz)
            if stable is None:
                out["stars"].append({"name": host.name, "stable_HZ": None,
                                     "n_planets": 0, "positions": [],
                                     "e_max": []})
                continue
            entry = self._pack_with_secular(host.mass, stable, planet_mass_me,
                                            delta, i, optimistic, "S")
            entry["name"] = host.name
            out["stars"].append(entry)

        # P-type
        if len(self.stars) >= 2 and self.stellar_orbits:
            if use_phz:
                phz = self._ptype_phz_for_circular_planet(optimistic)
            else:
                # Use the multi-star mask at t=0 to extract a 1-D annulus.
                phz = None
                # Reuse the radial scan from the snapshot.
                self._update_stellar_positions(0.0)
                bary = self.barycentre()
                L_tot = self.total_luminosity()
                rs = np.linspace(0.01, 3.0 * math.sqrt(L_tot / 0.32), 4000)
                pts = np.stack([bary[0] + rs, np.full_like(rs, bary[1]),
                                np.full_like(rs, bary[2])], axis=-1)
                mask = hz.hz_mask(pts, self.stars, optimistic=optimistic)
                if mask.any():
                    idxs = np.where(mask)[0]
                    groups = np.split(idxs, np.where(np.diff(idxs) > 1)[0] + 1)
                    biggest = max(groups, key=len)
                    phz = (float(rs[biggest[0]]), float(rs[biggest[-1]]))
            if phz is not None:
                stable = self._ptype_stable_hz(phz)
                if stable is not None:
                    m_tot = sum(s.mass for s in self.stars)
                    entry = self._pack_with_secular(m_tot, stable,
                                                    planet_mass_me, delta,
                                                    None, optimistic, "P")
                    out["circumbinary"] = entry
        return out

    def _pack_with_secular(self,
                           m_star_msun: float,
                           stable_band: Tuple[float, float],
                           planet_mass_me: float,
                           delta: float,
                           host_index: Optional[int],
                           optimistic: bool,
                           ptype: str) -> dict:
        """Place geometrically-spaced equal-mass planets in `stable_band`
        and verify the configuration is consistent with Laplace-Lagrange
        eccentricity coupling and binary forcing.

        Trims planets from the outside in until every planet's
        peri/apo distance stays inside the snapshot HZ for the host star.
        """
        in_au, out_au = stable_band
        positions = stab.packing_positions(in_au, out_au, m_star_msun,
                                           planet_mass_me, delta)
        if not positions:
            return {"stable_HZ": stable_band, "n_planets": 0,
                    "positions": [], "e_max": []}

        # External binary forcing on each planet
        ext = np.zeros(len(positions))
        if ptype == "S" and host_index is not None and len(self.stars) > 1:
            host = self.stars[host_index]
            for j, other in enumerate(self.stars):
                if j == host_index:
                    continue
                a_b, e_b = None, 0.0
                for edge in self.stellar_orbits:
                    ia, ib, a, e, _ = edge
                    if {ia, ib} == {host_index, j}:
                        a_b, e_b = a, e
                        break
                if a_b is None:
                    a_b = float(np.linalg.norm(np.array(other.position) -
                                               np.array(host.position)))
                for pi, a_p in enumerate(positions):
                    ext[pi] += sec.heppenheimer_e_forced_stype(a_p, a_b, e_b)
        elif ptype == "P":
            edge = self.stellar_orbits[0]
            ia, ib, a_b, e_b, _ = edge
            m1, m2 = self.stars[ia].mass, self.stars[ib].mass
            mu = min(m1, m2) / (m1 + m2)
            for pi, a_p in enumerate(positions):
                if a_p > a_b:
                    ext[pi] = sec.leung_lee_e_forced_ptype(a_p, a_b, e_b, mu)
                else:
                    ext[pi] = 1.0  # forbidden, will be trimmed below

        # Laplace-Lagrange coupling among the packed planets
        masses = [planet_mass_me] * len(positions)
        e_max = sec.secular_max_eccentricities(masses, positions,
                                               m_star_msun,
                                               initial_eccentricities=None,
                                               external_forcing=ext)

        # Now trim: any planet whose periastron/apastron falls outside the
        # stable_band gets removed (outermost first).
        keep_in_au, keep_out_au = stable_band
        positions = list(positions)
        e_max = list(e_max)
        while positions:
            ok = True
            for i_p, a_p in enumerate(positions):
                e_i = e_max[i_p]
                peri = a_p * (1.0 - e_i)
                apo = a_p * (1.0 + e_i)
                if peri < keep_in_au or apo > keep_out_au:
                    ok = False
                    drop = i_p
                    break
            if ok:
                break
            positions.pop(drop)
            e_max.pop(drop)
            if not positions:
                break
            # Recompute coupling for the trimmed set.
            ext_t = np.array([ext[i] for i in range(len(ext))
                              if i != drop])
            ext = ext_t
            masses = [planet_mass_me] * len(positions)
            e_max = list(sec.secular_max_eccentricities(
                masses, positions, m_star_msun,
                initial_eccentricities=None, external_forcing=ext))

        return {"stable_HZ": stable_band,
                "n_planets": len(positions),
                "positions": list(positions),
                "e_max": list(e_max)}

    # ------------------------------------------------------------------
    # Per-planet habitability (for user-supplied planets in self.planets)
    # ------------------------------------------------------------------
    def planet_habitability(self,
                            optimistic: bool = False
                            ) -> List[dict]:
        """For every planet in self.planets, check whether it sits inside
        the (stable) PHZ at periastron and apastron of its own orbit."""
        results = []
        for p in self.planets:
            if p.semi_major_axis_au is None:
                results.append({"planet": p.name,
                                "in_PHZ": None,
                                "note": "No SMA known"})
                continue
            peri = p.semi_major_axis_au * (1.0 - p.eccentricity)
            apo = p.semi_major_axis_au * (1.0 + p.eccentricity)
            if p.is_circumbinary():
                if len(self.stars) < 2:
                    results.append({"planet": p.name, "in_PHZ": False,
                                    "note": "Circumbinary planet but only 1 star"})
                    continue
                phz = self._ptype_phz_for_circular_planet(optimistic)
                stable = self._ptype_stable_hz(phz) if phz else None
                if stable is None:
                    results.append({"planet": p.name, "in_PHZ": False,
                                    "note": "No stable circumbinary PHZ exists"})
                    continue
                in_au, out_au = stable
                in_phz = (peri >= in_au) and (apo <= out_au)
                results.append({"planet": p.name,
                                "in_PHZ": in_phz,
                                "stable_HZ": stable,
                                "periastron": peri, "apastron": apo})
            else:
                i = p.host_star_index
                phz = self._stype_phz_for_circular_planet(i, optimistic)
                stable = self._stype_stable_hz(i, phz) if phz else None
                if stable is None:
                    results.append({"planet": p.name, "in_PHZ": False,
                                    "note": "No stable S-type PHZ exists"})
                    continue
                in_au, out_au = stable
                in_phz = (peri >= in_au) and (apo <= out_au)
                results.append({"planet": p.name,
                                "in_PHZ": in_phz,
                                "stable_HZ": stable,
                                "periastron": peri, "apastron": apo})
        return results

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------
    def summary(self, planet_mass_me: float = 1.0,
                delta: float = 10.0,
                optimistic: bool = False,
                use_phz: bool = True) -> str:
        lines = [f"System: {self.name}",
                 f"  N_stars = {len(self.stars)}",
                 f"  L_tot   = {self.total_luminosity():.3g} Lsun"]
        for s in self.stars:
            lines.append(f"  - {s}")
        if self.stellar_orbits:
            for edge in self.stellar_orbits:
                i, j, a, e, _ = edge
                if j == -1:
                    P = self.stellar_orbit_period(self.stellar_orbits.index(edge))
                    lines.append(f"  Outer companion: a={a:.2f} AU, e={e:.2f}, "
                                 f"P={P:.1f} yr")
                else:
                    P = self.stellar_orbit_period(self.stellar_orbits.index(edge))
                    lines.append(f"  Stellar binary (stars {i},{j}): a={a:.3f} AU, "
                                 f"e={e:.3f}, P={P:.3g} yr")

        res = self.count_habitable_planets(planet_mass_me, delta,
                                           optimistic, use_phz)
        zone_kind = "optimistic" if optimistic else "conservative"
        phz_kind = "PHZ" if use_phz else "classical HZ"
        lines.append(f"  Goldilocks {phz_kind} ({zone_kind}):")
        for entry in res["stars"]:
            if entry["stable_HZ"] is None:
                lines.append(f"    {entry['name']:>14s}: no stable HZ")
            else:
                lo, hi = entry["stable_HZ"]
                emax = entry.get("e_max") or []
                emax_str = "" if not emax else f"  e_max in [{min(emax):.3f}, {max(emax):.3f}]"
                lines.append(f"    {entry['name']:>14s}: r=[{lo:.3f},{hi:.3f}] AU  "
                             f"-> {entry['n_planets']} planet(s){emax_str}")
        if res["circumbinary"]:
            lo, hi = res["circumbinary"]["stable_HZ"]
            emax = res["circumbinary"].get("e_max") or []
            emax_str = "" if not emax else f"  e_max in [{min(emax):.3f}, {max(emax):.3f}]"
            lines.append(f"    circumbinary: r=[{lo:.3f},{hi:.3f}] AU  "
                         f"-> {res['circumbinary']['n_planets']} planet(s){emax_str}")

        # Known-planet check
        if self.planets:
            lines.append("  Known planets:")
            phab = self.planet_habitability(optimistic=optimistic)
            for r in phab:
                if r.get("in_PHZ") is True:
                    msg = f"in PHZ  (r=[{r['stable_HZ'][0]:.2f},{r['stable_HZ'][1]:.2f}] AU)"
                elif r.get("in_PHZ") is False:
                    note = r.get("note") or ""
                    if note:
                        msg = note
                    else:
                        msg = (f"outside PHZ  peri={r['periastron']:.2f} apo={r['apastron']:.2f}, "
                               f"PHZ=[{r['stable_HZ'][0]:.2f},{r['stable_HZ'][1]:.2f}]")
                else:
                    msg = r.get("note", "?")
                lines.append(f"    {r['planet']:>14s}: {msg}")
        return "\n".join(lines)
