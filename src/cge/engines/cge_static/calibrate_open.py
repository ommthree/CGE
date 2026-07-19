"""Calibration for the OPEN-economy static CGE (Phase 5 — Armington/CET).

Extends the closed pilot calibration to a small open economy with separate activity/commodity
accounts, an Armington import composite and a CET export transformation [Hosoe2010, ch. 7]. World
prices are fixed at 1 in foreign currency (small-open-economy assumption); the benchmark exchange
rate is 1, so benchmark domestic = world prices and every calibrated share reads off the balanced
SAM at unit prices.

Structure (per sector ``i``):
- **Activity** produces gross output ``Z_i`` using intermediates (the Armington *composite*
  commodity ``ax[j,i]`` per unit) and value added (Cobb-Douglas over factors).
- **CET**: output ``Z_i`` is transformed into domestic sales ``D_i`` and exports ``E_i`` with
  elasticity ``Ω_i`` (``cet_elast``). ``Z_i = B^T_i [ξ^d_i D_i^ρ + ξ^e_i E_i^ρ]^{1/ρ}``, ρ=(Ω+1)/Ω.
- **Armington**: the composite ``Q_i`` combines domestic ``D_i`` and imports ``M_i`` with elasticity
  ``σ_i`` (``arm_elast``). ``Q_i = A_i [α^d_i D_i^{-η} + α^m_i M_i^{-η}]^{-1/η}``, η=(σ−1)/σ.
- ``Q_i`` is used by intermediates + household; the household is Cobb-Douglas over ``Q``.
- Trade closure: benchmark foreign savings ``Sf`` = Σ imports − Σ exports (here 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cge.contracts.data_objects import SAM

# Default trade elasticities (dimensionless), documented as literature-typical placeholders.
DEFAULT_ARMINGTON_ELAST = 2.0  # σ: domestic↔import substitution
DEFAULT_CET_ELAST = 2.0  # Ω: domestic↔export transformation


def _elast_vector(elast: float | np.ndarray, ns: int, name: str) -> np.ndarray:
    """Coerce an elasticity input to a finite, strictly-positive ``(ns,)`` vector.

    Accepts a scalar (broadcast to every sector) or a length-``ns`` array **only** — a length-1
    array is rejected rather than silently broadcast, and any non-positive/non-finite value raises
    (an elasticity is a strictly positive substitution parameter). This is the single validation
    point for VA, Armington and CET elasticities."""
    a = np.asarray(elast, dtype=float)
    if a.ndim == 0:
        a = np.full(ns, float(a))
    elif a.shape != (ns,):
        raise ValueError(
            f"{name} must be a scalar or a length-{ns} vector (one per sector); got {a.shape}."
        )
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite; got {a.tolist()}.")
    if np.any(a <= 0.0):
        raise ValueError(f"{name} must be strictly positive; got {a.tolist()}.")
    return a


@dataclass(frozen=True)
class OpenCalibratedModel:
    """Benchmark parameters for the open static CGE. All benchmark prices = 1."""

    sectors: list[str]
    factors: list[str]
    ax: (
        np.ndarray
    )  # [j, i] intermediate composite-commodity coefficient (input j per unit output i)
    va_share: np.ndarray  # [i] value added per unit activity output
    beta: np.ndarray  # [f, i] Cobb-Douglas VA factor shares (cols → 1)
    av: np.ndarray  # [i] VA scale so unit VA cost = 1 at benchmark
    va_elast: np.ndarray  # [i] VA substitution elasticity σ_va (1 ⇒ Cobb-Douglas)
    va_ces_share: np.ndarray  # [f, i] CES VA factor share δ (used when σ ≠ 1)
    gamma: np.ndarray  # [i] household budget share over composite commodities (→ 1)
    endowment: np.ndarray  # [f] fixed factor endowment
    # trade parameters
    arm_elast: np.ndarray  # [i] Armington σ
    cet_elast: np.ndarray  # [i] CET Ω
    arm_scale: np.ndarray  # [i] Armington CES scale A
    arm_share_d: np.ndarray  # [i] Armington domestic share α^d
    arm_share_m: np.ndarray  # [i] Armington import share α^m
    cet_scale: np.ndarray  # [i] CET scale B
    cet_share_d: np.ndarray  # [i] CET domestic share ξ^d
    cet_share_e: np.ndarray  # [i] CET export share ξ^e
    foreign_savings: float  # Sf = Σ M − Σ E at benchmark (0 for a balanced trade account)
    # benchmark quantities (= money at unit prices), normalised by GDP
    Z0: np.ndarray  # [i] activity output
    D0: np.ndarray  # [i] domestic sales
    E0: np.ndarray  # [i] exports
    M0: np.ndarray  # [i] imports
    Q0: np.ndarray  # [i] composite supply
    FD0: np.ndarray  # [i] household final demand (on Q)
    F0: np.ndarray  # [f, i] factor demand
    INT0: np.ndarray  # [j, i] intermediate use of composite j by activity i

    @property
    def gdp0(self) -> float:
        return float(self.F0.sum())


def calibrate_open(
    sam: SAM,
    *,
    sectors: list[str],
    factors: list[str],
    arm_elast: float | np.ndarray = DEFAULT_ARMINGTON_ELAST,
    cet_elast: float | np.ndarray = DEFAULT_CET_ELAST,
    va_elast: float | np.ndarray = 1.0,
) -> OpenCalibratedModel:
    """Calibrate the open CGE from a balanced open SAM with ``a_<s>``/``c_<s>`` activity/commodity
    accounts, factors, one household, and a ``ROW`` account. ``va_elast`` is the value-added
    substitution elasticity σ_va (1 ⇒ Cobb-Douglas, the default; ≠ 1 ⇒ CES)."""
    m = sam.matrix
    act = [f"a_{s}" for s in sectors]
    com = [f"c_{s}" for s in sectors]
    hoh = [
        a for a in sam.accounts if a not in act and a not in com and a not in factors and a != "ROW"
    ]
    if len(hoh) != 1:
        raise ValueError(f"open calibration expects exactly one household account, got {hoh}")
    household = hoh[0]
    ns = len(sectors)

    # Benchmark flows (prices = 1 ⇒ money = quantity).
    D0 = np.array([m.loc[act[i], com[i]] for i in range(ns)], dtype=float)  # activity→own commodity
    E0 = np.array([m.loc[act[i], "ROW"] for i in range(ns)], dtype=float)  # exports
    M0 = np.array([m.loc["ROW", com[i]] for i in range(ns)], dtype=float)  # imports
    Z0 = D0 + E0  # activity output
    # INT0[j, i] = composite commodity j used as an intermediate by activity i.
    INT0 = np.array([[m.loc[com[j], act[i]] for i in range(ns)] for j in range(ns)], dtype=float)
    F0 = np.array([[m.loc[f, act[i]] for i in range(ns)] for f in factors], dtype=float)
    FD0 = np.array([m.loc[com[i], household] for i in range(ns)], dtype=float)
    # Composite USE of commodity i = Σ_activities of commodity i (row-sum over activities) + FD_i.
    Q0 = INT0.sum(axis=1) + FD0

    # Normalise by GDP (scale-free; keeps solver residuals meaningful — same as the closed model).
    scale = float(F0.sum())
    if scale <= 0:
        raise ValueError("open SAM has non-positive total value added; cannot calibrate")
    D0, E0, M0, Z0, INT0, F0, FD0, Q0 = (a / scale for a in (D0, E0, M0, Z0, INT0, F0, FD0, Q0))

    if float(Z0.min()) <= 0 or float(Q0.min()) <= 0:
        raise ValueError("open SAM has non-positive activity output or composite supply")
    if float(D0.min()) <= 0:
        raise ValueError("open SAM has non-positive domestic sales; Armington/CET undefined")

    # Value added — activity output value less intermediate composite cost. CD (σ=1) or CES.
    VA0 = Z0 - INT0.sum(axis=0)
    if float(VA0.min()) <= 0:
        raise ValueError("open SAM has non-positive value added for some activity")
    va_share = VA0 / Z0
    ax = INT0 / Z0[None, :]
    beta = F0 / VA0[None, :]
    sigma_va = _elast_vector(va_elast, ns, "va_elast")
    with np.errstate(divide="ignore", invalid="ignore"):
        w_ces = np.where(F0 > 0, np.power(F0, 1.0 / sigma_va[None, :]), 0.0)
    va_ces_share = w_ces / w_ces.sum(axis=0)[None, :]
    av = np.empty(ns)
    for i in range(ns):
        if abs(sigma_va[i] - 1.0) < 1e-12:
            b = beta[:, i]
            av[i] = np.prod(np.power(np.where(b > 0, 1.0 / np.where(b > 0, b, 1.0), 1.0), b))
        else:
            si, d = sigma_va[i], va_ces_share[:, i]
            av[i] = np.power(np.sum(d**si), 1.0 / (1.0 - si))
    gamma = FD0 / FD0.sum()
    endowment = F0.sum(axis=1)

    arm = _elast_vector(arm_elast, ns, "arm_elast")
    cet = _elast_vector(cet_elast, ns, "cet_elast")
    # The Armington/CET aggregators use ρ=(σ−1)/σ and ^{1/ρ}, singular at σ=1 (the Cobb-Douglas
    # special case, not implemented for trade). Require σ ≠ 1 with a clear message.
    if np.any(np.abs(arm - 1.0) < 1e-9) or np.any(np.abs(cet - 1.0) < 1e-9):
        raise ValueError(
            "Armington/CET elasticities must be ≠ 1 (σ=1 is the Cobb-Douglas special case, not "
            "implemented for trade); typical values are 1.5–5."
        )

    # Armington CES: Q = A[δ D^ρ + (1−δ) M^ρ]^{1/ρ}, ρ=(σ−1)/σ. At benchmark (pd=pm=1) the cost-min
    # FOC gives D/M = (δ/(1−δ))^σ, so δ = ratio/(1+ratio) with ratio = (D/M)^{1/σ}; then A from the
    # aggregator. (Forms verified against Shephard's lemma — see the module tests.) A non-traded
    # good (M=0) has δ=1 and A = Q/D.
    rho = (arm - 1.0) / arm
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_dm = np.where(M0 > 0, np.power(D0 / np.where(M0 > 0, M0, 1.0), 1.0 / arm), np.inf)
        # ratio_dm is inf for a non-traded good; inf/(1+inf)=nan there, masked to 1.0 by np.where.
        arm_share_d = np.where(M0 > 0, ratio_dm / (1.0 + ratio_dm), 1.0)
    arm_share_m = 1.0 - arm_share_d
    arm_scale = np.where(
        M0 > 0,
        Q0 / np.power(arm_share_d * D0**rho + arm_share_m * M0**rho, 1.0 / rho),
        Q0 / D0,
    )

    # CET: Z = B[ξ D^κ + (1−ξ) E^κ]^{1/κ}, κ=(Ω+1)/Ω. Revenue-max FOC at benchmark (pd=pe=1) gives
    # ξ/(1−ξ) = (D/E)^{1−κ} = (D/E)^{−1/Ω} (note the NEGATIVE exponent — the transformation frontier
    # is convex, unlike the Armington isoquant). B from the aggregator. (Verified via Hotelling.)
    kappa = (cet + 1.0) / cet
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_de = np.where(E0 > 0, np.power(D0 / np.where(E0 > 0, E0, 1.0), -1.0 / cet), np.inf)
        # ratio_de is inf for a non-exporter; inf/(1+inf)=nan there, masked to 1.0 by np.where.
        cet_share_d = np.where(E0 > 0, ratio_de / (1.0 + ratio_de), 1.0)
    cet_share_e = 1.0 - cet_share_d
    cet_scale = np.where(
        E0 > 0,
        Z0 / np.power(cet_share_d * D0**kappa + cet_share_e * E0**kappa, 1.0 / kappa),
        Z0 / D0,
    )

    # Foreign savings Sf = Σ M − Σ E at benchmark. The pilot's closure fixes Sf at its benchmark
    # level, but the household income identity circulates only factor income + carbon revenue — it
    # does NOT yet carry the ROW transfer / investment flow that a non-zero current account implies.
    # So a non-zero-Sf SAM would not replicate its benchmark (household absorption would be off by
    # exactly the net capital inflow). Rather than return non-replicating numbers, restrict the
    # pilot to a balanced current account and reject non-zero Sf with guidance (review P1). A proper
    # ROW-institution / savings-investment closure is a documented follow-up.
    foreign_savings = float(M0.sum() - E0.sum())
    if abs(foreign_savings) > 1e-9 * max(1.0, float(M0.sum() + E0.sum())):
        raise ValueError(
            "open CGE pilot requires a balanced current account (Σ imports = Σ exports, i.e. zero "
            f"foreign savings); got Sf={foreign_savings:.6g}. A non-zero foreign-savings closure "
            "(ROW transfer / savings-investment account in household income) is a documented "
            "follow-up — the current household income identity would not replicate the benchmark."
        )

    return OpenCalibratedModel(
        sectors=list(sectors),
        factors=list(factors),
        ax=ax,
        va_share=va_share,
        beta=beta,
        av=av,
        va_elast=sigma_va,
        va_ces_share=va_ces_share,
        gamma=gamma,
        endowment=endowment,
        arm_elast=arm,
        cet_elast=cet,
        arm_scale=arm_scale,
        arm_share_d=arm_share_d,
        arm_share_m=arm_share_m,
        cet_scale=cet_scale,
        cet_share_d=cet_share_d,
        cet_share_e=cet_share_e,
        foreign_savings=foreign_savings,
        Z0=Z0,
        D0=D0,
        E0=E0,
        M0=M0,
        Q0=Q0,
        FD0=FD0,
        F0=F0,
        INT0=INT0,
    )
