"""
secular.py
----------
Secular perturbation theory used to turn the *snapshot* habitable zone
(which assumes a planet on a fixed circular orbit) into the
*Permanently Habitable Zone* (PHZ) of Eggl et al. 2012, which
respects the fact that the planet's eccentricity is excited by both
binary forcing and planet-planet coupling.

Three effects are included
==========================

1.  **S-type binary forcing** (Heppenheimer 1978).
    A planet orbiting one star of an eccentric binary picks up a
    forced eccentricity

        e_f^B = (5/4) (a_p / a_b) e_b / (1 - e_b^2)

    and if it starts circular its eccentricity oscillates between 0
    and 2 e_f^B.

2.  **P-type binary forcing** (Leung & Lee 2013).
    A circumbinary planet picks up a forced eccentricity from the
    binary's m=1 potential moment:

        e_f^B = (5/4) (a_b / a_p) e_b |1 - 2 mu|

    where mu = m2 / (m1+m2).

3.  **Planet-planet Laplace-Lagrange coupling** (Murray & Dermott 1999).
    For N planets mutually perturbing each other around a single star
    the eccentricity vectors evolve as a linear system; the N
    eigenvalues of the secular matrix give precession frequencies and
    the eigenvectors give "free" eccentricity modes.  The upper
    envelope of each planet's eccentricity is

        e_max,i = |e_forced,i| + sum_j |M_ij * c_j|

    where M is the eigenvector matrix and c the initial-condition
    coefficients.

References
----------
Heppenheimer T.A. 1978, A&A 65, 421.
Leung G.C.K. & Lee M.H. 2013, ApJ 763, 107.
Murray C.D. & Dermott S.F. 1999, "Solar System Dynamics", Ch. 7.
Eggl S., Pilat-Lohinger E., Georgakarakos N., Gyergyovits M., Funk B.
   2012, ApJ 752, 74.
"""

from __future__ import annotations
import math
from typing import Sequence, Tuple, Optional
import numpy as np


M_EARTH_OVER_M_SUN = 3.0034893e-6


# -----------------------------------------------------------------------
# Binary forcing
# -----------------------------------------------------------------------
def heppenheimer_e_forced_stype(a_p: float, a_b: float, e_b: float) -> float:
    """S-type forced eccentricity (Heppenheimer 1978)."""
    if e_b >= 1.0:
        raise ValueError("e_b must be < 1")
    return 1.25 * (a_p / a_b) * (e_b / (1.0 - e_b * e_b))


def leung_lee_e_forced_ptype(a_p: float, a_b: float,
                             e_b: float, mu: float) -> float:
    """P-type forced eccentricity (Leung & Lee 2013), leading order."""
    return 1.25 * (a_b / a_p) * e_b * abs(1.0 - 2.0 * mu)


# -----------------------------------------------------------------------
# Laplace coefficients (numerical)
# -----------------------------------------------------------------------
def laplace_coefficient(s: float, j: int, alpha: float,
                        n_panels: int = 2048) -> float:
    """Numerical Laplace coefficient b_s^(j)(alpha)."""
    psi = np.linspace(0.0, 2.0 * math.pi, n_panels, endpoint=False)
    denom = (1.0 - 2.0 * alpha * np.cos(psi) + alpha * alpha) ** s
    return float(np.sum(np.cos(j * psi) / denom) * (2.0 / n_panels))


# -----------------------------------------------------------------------
# Laplace-Lagrange matrix for N coplanar planets around mass M*
# -----------------------------------------------------------------------
def laplace_lagrange_matrix(masses_me: Sequence[float],
                            semi_major_axes_au: Sequence[float],
                            m_star_msun: float
                            ) -> np.ndarray:
    """Laplace-Lagrange secular matrix A (rad/yr) governing the eccentricity
    vector evolution of N planets around a star of mass M*.

    Murray & Dermott (1999), eqs (7.128) / (7.129):

        A_jj  = + n_j/4 * sum_{k != j} m_k/M*  alpha_jk * abar_jk * b_3/2^(1)(alpha_jk)
        A_jk  = - n_j/4 *           m_k/M*     alpha_jk * abar_jk * b_3/2^(2)(alpha_jk)

    where alpha = min(a_j, a_k)/max(a_j, a_k) and abar = alpha if
    a_j < a_k else 1.
    """
    n = len(masses_me)
    A = np.zeros((n, n))
    if n == 0:
        return A
    G_yr = 4.0 * math.pi * math.pi              # AU^3 Msun^-1 yr^-2
    n_j  = np.array([math.sqrt(G_yr * m_star_msun / a**3)
                     for a in semi_major_axes_au])
    m_ratio = np.array([m * M_EARTH_OVER_M_SUN / m_star_msun
                        for m in masses_me])

    for j in range(n):
        for k in range(n):
            if j == k:
                continue
            a_j, a_k = semi_major_axes_au[j], semi_major_axes_au[k]
            alpha = min(a_j, a_k) / max(a_j, a_k)
            abar  = alpha if a_j < a_k else 1.0
            b1 = laplace_coefficient(1.5, 1, alpha)
            b2 = laplace_coefficient(1.5, 2, alpha)
            prefac = 0.25 * n_j[j] * m_ratio[k] * alpha * abar
            A[j, j] += prefac * b1
            A[j, k] -= prefac * b2
    return A


def secular_max_eccentricities(masses_me: Sequence[float],
                               semi_major_axes_au: Sequence[float],
                               m_star_msun: float,
                               initial_eccentricities: Sequence[float] = None,
                               external_forcing: Sequence[float] = None,
                               ) -> np.ndarray:
    """Upper-envelope eccentricity for each planet under Laplace-Lagrange
    coupling combined with binary forcing.

    Decomposition
    -------------
    Two distinct effects act on each planet:

    (a) **Binary forcing** with the planet's own secular response.
        For an S-type planet around a binary, Heppenheimer (1978) gives
        the equilibrium forced eccentricity e_f^B; the planet's eccentricity
        oscillates between 0 and 2*e_f^B if it starts circular.  The same
        is true for P-type (Leung & Lee 2013).  This is captured by
        passing `external_forcing[i] = e_f^B,i`.

    (b) **Mutual planet-planet (Laplace-Lagrange) coupling.** The
        homogeneous secular equation dz/dt = i A z gives a set of
        eigenmodes precessing at the eigenfrequencies of A.  Given
        an initial eccentricity vector z(0), the upper envelope of
        each planet's free eccentricity is

            e_max^LL,i  =  sum_l | v_il * c_l |

        where (lam_l, v_l) are the eigenpairs of A and c are the
        decomposition coefficients of z(0) onto v.

    Because the binary's secular frequency (~ Heppenheimer rate
    g_H ~ (3/4) n_p mu_b (a_p/a_b)^3 (1-e_b^2)^{-3/2}) is generally
    much faster than the LL frequencies for an Alpha-Cen-like binary
    perturbing planets in its primary's HZ, the two contributions are
    well decoupled and we add them by the triangle inequality:

        e_max,i  =  2 * e_f^B,i  +  e_max^LL,i

    The factor of 2 is because, starting from a circular orbit, the
    planet's e oscillates between 0 and 2*e_f^B.

    Parameters
    ----------
    masses_me               : planet masses (M_earth)
    semi_major_axes_au      : SMAs (AU)
    m_star_msun             : host star mass (M_sun)
    initial_eccentricities  : default zero (gives 2*e_f^B contribution)
    external_forcing        : Heppenheimer e_f^B per planet

    Returns
    -------
    e_max : ndarray, shape (N,)
        Worst-case maximum eccentricity envelope.
    """
    n = len(masses_me)
    if n == 0:
        return np.zeros(0)
    if initial_eccentricities is None:
        initial_eccentricities = np.zeros(n)
    if external_forcing is None:
        external_forcing = np.zeros(n)
    init_e = np.asarray(initial_eccentricities, dtype=float)
    ext    = np.asarray(external_forcing, dtype=float)

    # Binary forcing: starting from circular orbit, oscillates between 0 and 2*e_f.
    # If init_e > 0, max is init_e + 2*e_f (worst-case phasing).
    e_max_binary = 2.0 * np.abs(ext) + np.maximum(init_e - np.abs(ext), 0.0)

    if n == 1:
        return e_max_binary

    # LL planet-planet coupling: free precession only (the inhomogeneous
    # forced solution from binary forcing is handled separately).
    A = laplace_lagrange_matrix(masses_me, semi_major_axes_au, m_star_msun)
    eigvals, eigvecs = np.linalg.eig(A)
    try:
        c = np.linalg.solve(eigvecs, init_e.astype(complex))
    except np.linalg.LinAlgError:
        c = np.zeros(n, dtype=complex)
    free_amp = np.abs(eigvecs) @ np.abs(c)
    return e_max_binary + free_amp.real