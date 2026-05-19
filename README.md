# Goldilocks Zone Calculator for N-Star Systems

A research-grade Python package that computes the **Permanently Habitable
Zone (PHZ)** for any single-star, binary, or hierarchical-triple stellar
configuration, packs Earth-mass planets into that zone with realistic
3-D orbital dynamics, and visualises everything with both static figures
and animated MP4s. All planet motion in the animations comes from a
full **N-body integration**: each planet feels the time-varying
gravitational field of *every* star plus every other planet, producing
genuine non-Keplerian wobble, apse/node precession, and -- for
high-energy "star-hopper" orbits -- bouncing trajectories between binary
companions.

## What it does

Given a list of stars and their orbital configuration, the calculator:

1. Places the stars on **exact closed-form Kepler orbits** (N <= 2) or
   a hierarchical Kepler decomposition (for triples), validated by the
   **Mardling-Aarseth stability criterion**.
2. Computes each star's classical habitable zone using the
   **Kopparapu 2013/2014** polynomial flux limits, then combines them
   across stars via the **Mueller-Haghighipour 2014** spectrally
   weighted-flux scheme.
3. Computes the **Permanently Habitable Zone (PHZ)** of Eggl et al.
   2012: the locus where the planet's instantaneous flux stays inside
   the runaway-greenhouse / max-greenhouse limits at *every* orbital
   phase of the binary.
4. Imposes the **Holman-Wiegert 1999** dynamical stability cut for
   S-type and P-type orbits, and the **Eggleton 1983** Roche-lobe
   geometric cut.
5. Models **multi-planet gravitational coupling**:
    - **Heppenheimer 1978** forced eccentricity for S-type planets,
    - **Leung-Lee 2013** forced eccentricity for circumbinary planets,
    - **Laplace-Lagrange secular coupling** (Murray-Dermott 1999) for
      planets in the same zone.
6. Packs planets in the stable PHZ band with **Smith-Lissauer 2009**
   mutual-Hill-radius spacing (default delta = 10).
7. Integrates every planet as a test particle in the full N-star
   gravitational field using a leapfrog scheme with adaptive
   sub-stepping (more steps for shorter planet periods).
8. Renders 4-to-5-panel static figures and animated MP4s with:
    - top-down (x-y) view recentered on the inner-binary barycentre,
    - **side (x-z) view** showing real orbital inclinations (3-8 deg
      S-type, 1-5 deg P-type),
    - **multi-scale "wide view"** for hierarchical systems where the
      outer companion sits far outside the inner zoom,
    - eccentricity envelope per planet,
    - moving HZ contours that respond to the binary's phase.

## File layout

```
goldilocks/
├── stellar.py            Star dataclass, Eker 2018 mass-luminosity-radius
├── habitable_zone.py     Kopparapu HZ, multi-star weighted flux
├── stability.py          Holman-Wiegert, Mardling-Aarseth, Hill packing
├── roche.py              Eggleton 1983 Roche-lobe geometry
├── kepler.py             Closed-form 2-body Kepler solver
├── secular.py            Heppenheimer, Leung-Lee, Laplace-Lagrange
├── nbody.py              N-body planet integrator (stars analytical, planets numerical)
├── planets.py            Planet objects + catalogue (Kepler 16-47)
├── random_systems.py     Star-hopper, Trojan, wide-hierarchy, polar planet generators
├── system.py             StarSystem: ties everything together
├── visualization.py      Static figure (top + side + e-bar + wide view)
├── animation.py          Time-evolution MP4 with N-body planet dynamics
├── demo.py               End-to-end driver for the 10 example systems
└── test_sanity.py        Regression tests against published values
```

## Sanity-check results

| Test                                  | Value          | Reference        |
|---------------------------------------|----------------|------------------|
| Sun runaway-greenhouse limit          | 0.951 AU       | 0.95 (Kopparapu) |
| Sun max-greenhouse limit              | 1.677 AU       | 1.67 (Kopparapu) |
| Equal-mass Eggleton Roche radius      | 0.379          | 0.380 (textbook) |
| Earth periastron from Kepler solver   | 0.983 AU       | 0.983            |
| Earth speed at periastron             | 6.389 AU/yr    | 6.39             |
| Alpha Cen AB orbital period (a=23.4)  | 79.9 yr        | 79.9             |
| Alpha Cen triple (Mardling-Aarseth)   | stable         | stable           |
| Earth in N-body integrator after 1 yr | <2e-5 AU drift | (machine prec.)  |

## The ten example systems

The demo (`python3 demo.py`) builds and renders ten systems covering the
main physical regimes:

| #  | System                           | Type                      | Highlights                               |
|----|----------------------------------|---------------------------|------------------------------------------|
| 01 | Sun + Earth                      | single                    | Real Earth e=0.0167; 5 planets fit in HZ |
| 02 | Alpha Centauri AB                | close binary              | 3+3 S-type planets, e_max ~0.13          |
| 03 | Kepler-16                        | tight circumbinary        | Real planet just at the PHZ edge         |
| 04 | Kepler-47                        | multi-planet circumbinary | Real Kepler-47 c sits in PHZ             |
| 05 | Alpha Cen + Proxima              | hierarchical triple       | Multi-scale view: 30 AU + 18000 AU       |
| 06 | G+G+M wide hierarchy             | hierarchical triple       | Synthetic stable triple                  |
| 07 | Star-hopper (Moeckel-Veras 2012) | wide binary               | e=0.70 planet bouncing near L1           |
| 08 | Trojan / co-orbital pair         | single                    | Two planets share an orbit, 60 deg apart |
| 09 | Wide hierarchy + M-dwarf HZ      | hierarchical triple       | Earth around the outer M-dwarf           |
| 10 | K-dwarf with polar HZ planet     | single                    | Inclined orbit; HAT-P-7-like geometry    |

Each system produces:

- `figures/<NN>_<name>.png` -- 4-panel diagnostic figure with top-down
  view + PHZ + side view + eccentricity envelope (plus a wide-view
  panel for hierarchical systems).
- `animations/<NN>_<name>.mp4` -- time-evolution movie with real
  N-body planet motion.

## Key formulae

* Distance from flux: r = sqrt(L / S_eff)
* Kopparapu S_eff polynomial in T_eff - 5780 K
* Mueller-Haghighipour weighted flux: Sum_i W_i L_i / r_i^2,
  with W_i = S_eff_sun / S_eff(T_i)
* Heppenheimer S-type: e_f = (5/4) (a_p / a_b) e_b / (1 - e_b^2)
* Leung-Lee P-type: e_f = (5/4) (a_b / a_p) e_b |1 - 2 mu|
* Holman-Wiegert S-type: a_c / a_b = 0.464 - 0.380 mu - 0.631 e + ...
* Holman-Wiegert P-type: a_c / a_b = 1.60 + 5.10 e + ...
* Mardling-Aarseth: (a_out / a_in)_crit = 2.8 [(1 + q)(1 + e) / sqrt(1 - e)]^(2/5)
* Eggleton Roche: r_L / A = 0.49 q^(2/3) / (0.6 q^(2/3) + ln(1 + q^(1/3)))
* Mutual Hill radius: R_H = ((m_1 + m_2) / (3 M*))^(1/3) (a_1 + a_2) / 2
* Laplace-Lagrange matrix (Murray-Dermott eqs 7.128/7.129) with
  numerically evaluated Laplace coefficients b_{3/2}^(j) (alpha)
* N-body planet acceleration:
  d^2 r_i / dt^2 = -G sum_s M_s (r_i - r_s(t)) / |r_i - r_s|^3
    - G sum_{j != i} m_j (r_i - r_j) / |r_i - r_j|^3
      with stars on analytical Kepler orbits and planets integrated with a
      leapfrog symplectic scheme (adaptive sub-stepping per planet period).

## References

* Kopparapu R.K. et al. 2013, ApJ 765, 131 (revised 2014, ApJL 787, L29)
* Mueller T.W.A. & Haghighipour N. 2014, ApJ 782, 26
* Eggl S., Pilat-Lohinger E., Georgakarakos N., Gyergyovits M., Funk B.
  2012, ApJ 752, 74
* Eggl S., Haghighipour N., Pilat-Lohinger E. 2013, ApJ 764, 130
* Heppenheimer T.A. 1978, A&A 65, 421
* Leung G.C.K. & Lee M.H. 2013, ApJ 763, 107 (arXiv:1212.2545)
* Holman M.J. & Wiegert P.A. 1999, AJ 117, 621
* Mardling R.A. & Aarseth S.J. 2001, MNRAS 321, 398
* Eggleton P.P. 1983, ApJ 268, 368
* Murray C.D. & Dermott S.F. 1999, *Solar System Dynamics*, Ch. 7
* Smith A.W. & Lissauer J.J. 2009, Icarus 201, 381
* Raghavan D. et al. 2010, ApJS 190, 1 (arXiv:1007.0414)
* Eker Z. et al. 2018, MNRAS 479, 5491
* Doyle L.R. et al. 2011, Science 333, 1602 (Kepler-16 b)
* Welsh W.F. et al. 2012, Nature 481, 475 (Kepler-34 b, 35 b)
* Orosz J.A. et al. 2012, Science 337, 1511 (Kepler-47 b, c)
* Orosz J.A. et al. 2019, AJ 157, 174 (Kepler-47 d)
* Moeckel N. & Veras D. 2012, MNRAS 422, 831 (arXiv:1201.6582) -- star-hoppers
* Raymond S.N. 2022, planetplanet.net blog -- "Star-hoppers" exposition