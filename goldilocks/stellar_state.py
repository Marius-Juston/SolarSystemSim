"""
stellar_state.py
-----------------
L0 global-astrophysics core (research/sun_render.md §3, checklist Phase 1).

`StellarState` is the single object every higher layer of the stellar
simulator reads from: the photosphere granulation rate, sunspot activity,
gravity darkening, and (later) the chromosphere / PFSS corona all derive
their inputs from these scalars.  It is pure CPU physics (microseconds),
so it stays on plain ``math``/``numpy`` -- the NumPy/CuPy backend seam
matters for the per-pixel renderer kernels, not here.

Derivation order (research §3):

1. M -> L, R, Teff      reuse goldilocks.stellar (Eker 2018, six-piece).
                        The research checklist's crude piecewise power
                        laws (1.3/1.4) are *superseded* here -- Eker 2018
                        is the higher-fidelity relation and is already
                        the repo's pinned source of truth.
2. t_MS                 1e10 * (M/Msun)^-2.5 yr.
3. (B-V)                Ballesteros (2012), inverted numerically.
4. P_rot                Skumanich (1972) spin-down v ~ t^-1/2, with a
                        Barnes (2010) M^1/2 pre-factor and a breakup cap.
5. tau_c                Noyes et al. (1984) convective turnover.
6. Ro = P_rot / tau_c   Rossby number (activity proxy).
7. beta                 gravity-darkening exponent: Lucy (0.08, convective)
                        -> von Zeipel (0.25, radiative), smooth blend.
8. tidal lock           Hut (1981)-style synchronization timescale for the
                        binary case.

References
----------
Eker+ 2018 MNRAS 479 5491; Skumanich 1972 ApJ 171 565;
Noyes+ 1984 ApJ 279 763; Barnes 2010 ApJ 722 222;
Ballesteros 2012 EPL 97 34008; Lucy 1967 ZA 65 89;
von Zeipel 1924 MNRAS 84 665; Hut 1981 A&A 99 126.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np

from goldilocks.stellar import (
    G_SI,
    M_SUN_KG,
    R_SUN_M,
    T_EFF_SUN_K,
    luminosity_from_mass,
    radius_from_mass,
    teff_from_l_and_r,
)

# Solar reference anchors -------------------------------------------------
AGE_SUN_GYR = 4.6
VSINI_SUN_KMS = 2.0          # surface rotation speed anchor (Skumanich)
P_ROT_SUN_DAYS = 25.0        # sidereal-ish equatorial value (cross-check)
DAY_S = 86400.0
GYR_S = 1.0e9 * 3.15576e7


# ---------------------------------------------------------------------
# (B-V) <-> Teff  (Ballesteros 2012, inverted with a cached LUT)
# ---------------------------------------------------------------------
def teff_from_bv(bv: float) -> float:
    """Ballesteros (2012) colour-temperature relation, K."""
    a = 0.92 * bv
    return 4600.0 * (1.0 / (a + 1.7) + 1.0 / (a + 0.62))


@lru_cache(maxsize=1)
def _bv_lut():
    """Monotone (T -> B-V) lookup table.

    ``teff_from_bv`` is strictly decreasing in (B-V) over the stellar
    range, so a sorted-by-T table inverts it with ``np.interp`` -- exact
    enough for kernel-side colour and far cheaper than per-call
    bisection.
    """
    bv = np.linspace(-0.40, 2.00, 1024)
    t = teff_from_bv(bv)                 # strictly decreasing
    order = np.argsort(t)               # ascending T for np.interp
    return t[order], bv[order]


def bv_from_teff(teff: float) -> float:
    """Invert Ballesteros for (B-V) from effective temperature."""
    t_tab, bv_tab = _bv_lut()
    return float(np.interp(float(teff), t_tab, bv_tab))


# ---------------------------------------------------------------------
# Harvard spectral classification from Teff (checklist 1.5)
# ---------------------------------------------------------------------
# Standard main-sequence Teff class boundaries [K] (e.g. Habets &
# Heintze 1981 / Pecaut & Mamajek 2013 rounded edges).
_SPECTRAL_EDGES = ((30000.0, "O"), (10000.0, "B"), (7500.0, "A"),
                    (6000.0, "F"), (5200.0, "G"), (3700.0, "K"))


def spectral_class_from_teff(teff: float) -> str:
    """Harvard class letter (O/B/A/F/G/K/M) for an effective temperature."""
    t = float(teff)
    for edge, letter in _SPECTRAL_EDGES:
        if t >= edge:
            return letter
    return "M"


# ---------------------------------------------------------------------
# Convective turnover time (Noyes et al. 1984, Eq. 4)
# ---------------------------------------------------------------------
def noyes_turnover_days(bv: float) -> float:
    """Convective turnover time tau_c [days] from (B-V)."""
    x = 1.0 - float(bv)
    if x > 0.0:
        log_tc = 1.362 - 0.166 * x + 0.025 * x * x - 5.323 * x ** 3
    else:
        log_tc = 1.362 - 0.14 * x
    return 10.0 ** log_tc


# ---------------------------------------------------------------------
# Skumanich spin-down (1972) + Barnes (2010) mass dependence
# ---------------------------------------------------------------------
def skumanich_prot_days(mass_msun: float, age_gyr: float,
                        radius_rsun: float) -> tuple[float, bool]:
    """Rotation period [days]; second value flags the breakup cap."""
    age_gyr = max(float(age_gyr), 1.0e-4)
    # v(t) = v_sun * (M)^0.5 * (t/t_sun)^-0.5   [km/s]
    v_kms = (VSINI_SUN_KMS * math.sqrt(max(mass_msun, 1.0e-3))
             * (age_gyr / AGE_SUN_GYR) ** -0.5)
    v = v_kms * 1.0e3                                   # m/s
    M = mass_msun * M_SUN_KG
    R = radius_rsun * R_SUN_M
    v_break = math.sqrt(G_SI * M / R)
    # Young-star saturation (checklist 1.8): before ~100 Myr the
    # spin-down law is not yet established and rotation sits at the
    # saturated (near-breakup) plateau.  At >= 0.1 Gyr behaviour is
    # byte-identical to the previous breakup-only cap.
    saturated = age_gyr < 0.1
    capped = saturated or v > v_break
    if capped:
        v = v_break
    p_s = 2.0 * math.pi * R / max(v, 1.0e-6)
    return p_s / DAY_S, capped


# ---------------------------------------------------------------------
# Gravity-darkening exponent (Lucy 1967 <-> von Zeipel 1924)
# ---------------------------------------------------------------------
def gravity_darkening_beta(mass_msun: float) -> float:
    """beta(M): 0.08 (convective) -> 0.25 (radiative), smoothstep blend."""
    lo, hi = 1.3, 1.7
    s = min(max((mass_msun - lo) / (hi - lo), 0.0), 1.0)
    s = s * s * (3.0 - 2.0 * s)             # smoothstep
    return 0.08 + (0.25 - 0.08) * s


# ---------------------------------------------------------------------
# Binary tidal synchronization (Hut 1981 equilibrium-tide scaling)
# ---------------------------------------------------------------------
# Equilibrium-tide t_sync ~ q^-2 (a/R)^6.  Calibrated so an equal-mass
# solar binary at P_orb = 12 d (the classic empirical synchronization
# cutoff for solar-type stars) has t_sync == the solar age.
_P_SYNC_REF_DAYS = 12.0


def _semimajor_axis_au(m_tot_msun: float, p_orb_days: float) -> float:
    p_yr = p_orb_days / 365.25
    return (m_tot_msun * p_yr * p_yr) ** (1.0 / 3.0)


def tidal_sync_timescale_gyr(mass_msun: float, companion_msun: float,
                             radius_rsun: float,
                             p_orb_days: float) -> float:
    m_tot = mass_msun + companion_msun
    q = companion_msun / max(mass_msun, 1.0e-6)
    r_au = radius_rsun * (R_SUN_M / 1.495978707e11)
    a = _semimajor_axis_au(m_tot, p_orb_days)
    a_ref = _semimajor_axis_au(2.0, _P_SYNC_REF_DAYS)
    r_ref = R_SUN_M / 1.495978707e11
    ratio6 = (a / max(r_au, 1e-12)) ** 6 / (a_ref / r_ref) ** 6
    return AGE_SUN_GYR * ratio6 / max(q * q, 1.0e-6)


# ---------------------------------------------------------------------
# StellarState
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class StellarState:
    """Global astrophysical state of a single (optionally binary) star.

    Construct via :meth:`for_mass_age` or :meth:`from_star`; all derived
    fields are filled by the research-§3 derivation order.
    """

    mass_msun: float
    age_gyr: float
    # --- derived ---
    luminosity_lsun: float
    radius_rsun: float
    teff_k: float
    bv: float
    p_rot_days: float
    tau_c_days: float
    rossby: float
    beta_gd: float
    t_ms_gyr: float
    # --- binary / activity geometry ---
    is_binary: bool = False
    tidally_locked: bool = False
    phi_sub_rad: float = 0.0
    active_longitude_amp: float = 0.0
    breakup_capped: bool = False

    def __post_init__(self) -> None:
        """Range + finiteness validation (checklist 1.2).

        Runs on every construction -- including ``from_dict`` and direct
        instantiation -- so an invalid state can never silently exist,
        not only when built via :meth:`for_mass_age`.
        """
        if not (0.08 <= self.mass_msun <= 100.0):
            raise ValueError(
                f"mass {self.mass_msun} Msun outside MS range "
                f"[0.08, 100].")
        if not (self.age_gyr > 0.0):
            raise ValueError(f"age {self.age_gyr} Gyr must be > 0.")
        for name in ("luminosity_lsun", "radius_rsun", "teff_k", "bv",
                     "p_rot_days", "tau_c_days", "rossby", "beta_gd",
                     "t_ms_gyr"):
            v = getattr(self, name)
            if not math.isfinite(v):
                raise ValueError(f"derived field {name}={v} is not finite.")

    # --- constructors ---------------------------------------------------
    @classmethod
    def for_mass_age(cls, mass_msun: float, age_gyr: float = AGE_SUN_GYR,
                     *, companion_msun: Optional[float] = None,
                     orbital_period_days: Optional[float] = None,
                     ) -> "StellarState":
        if not (0.08 <= mass_msun <= 100.0):
            raise ValueError(
                f"mass {mass_msun} Msun outside MS range [0.08, 100].")
        if age_gyr <= 0.0:
            raise ValueError(f"age {age_gyr} Gyr must be > 0.")

        L = luminosity_from_mass(mass_msun)
        R = radius_from_mass(mass_msun)
        Teff = teff_from_l_and_r(L, R)
        t_ms_gyr = 10.0 * mass_msun ** -2.5          # 1e10 yr -> Gyr

        if age_gyr > t_ms_gyr:
            warnings.warn(
                f"age {age_gyr:.2f} Gyr exceeds MS lifetime "
                f"{t_ms_gyr:.2f} Gyr for {mass_msun} Msun; "
                f"post-MS physics is out of scope (MS-only model).",
                stacklevel=2)

        bv = bv_from_teff(Teff)
        p_rot, capped = skumanich_prot_days(mass_msun, age_gyr, R)
        tau_c = noyes_turnover_days(bv)
        beta = gravity_darkening_beta(mass_msun)

        is_binary = (companion_msun is not None
                     and orbital_period_days is not None)
        locked = False
        phi_sub = 0.0
        amp = 0.0
        if is_binary:
            t_sync = tidal_sync_timescale_gyr(
                mass_msun, float(companion_msun), R,
                float(orbital_period_days))
            if age_gyr > t_sync:
                locked = True
                p_rot = float(orbital_period_days)
                amp = 0.5                 # active-longitude contrast
                # Sub-stellar point fixed in the corotating frame; use a
                # reference longitude (orbital phase is set downstream).
                phi_sub = 0.0
            rossby = p_rot / max(tau_c, 1.0e-6)
        else:
            rossby = p_rot / max(tau_c, 1.0e-6)

        return cls(
            mass_msun=float(mass_msun), age_gyr=float(age_gyr),
            luminosity_lsun=float(L), radius_rsun=float(R),
            teff_k=float(Teff), bv=float(bv),
            p_rot_days=float(p_rot), tau_c_days=float(tau_c),
            rossby=float(rossby), beta_gd=float(beta),
            t_ms_gyr=float(t_ms_gyr), is_binary=is_binary,
            tidally_locked=locked, phi_sub_rad=float(phi_sub),
            active_longitude_amp=float(amp), breakup_capped=capped)

    @classmethod
    def from_star(cls, star, age_gyr: float = AGE_SUN_GYR,
                  **kw) -> "StellarState":
        """Build from a :class:`goldilocks.stellar.Star`.

        Uses the star's mass when available; otherwise back-solves an
        effective MS mass from luminosity so legacy L/Teff-only stars
        still get a usable state.
        """
        m = getattr(star, "mass", None)
        if m is None:
            lum = getattr(star, "luminosity", None) or 1.0
            m = float(min(max(lum ** (1.0 / 4.0), 0.08), 100.0))
        return cls.for_mass_age(float(m), age_gyr, **kw)

    # --- helpers --------------------------------------------------------
    def activity_regime(self) -> str:
        """Rossby activity regime (Pizzolato+ 2003 / Wright+ 2011)."""
        if self.rossby < 0.13:
            return "saturated"
        if self.rossby < 2.0:
            return "linear"
        return "quiet"

    @property
    def spectral_class(self) -> str:
        """Harvard class letter (O/B/A/F/G/K/M) from ``teff_k``."""
        return spectral_class_from_teff(self.teff_k)

    @property
    def evolutionary_state(self) -> str:
        if self.age_gyr > self.t_ms_gyr:
            return "post_ms"
        if self.age_gyr < 0.05 * self.t_ms_gyr:
            return "pre_ms"
        return "ms"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StellarState":
        return cls(**d)
