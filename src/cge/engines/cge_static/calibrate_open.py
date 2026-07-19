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
) -> OpenCalibratedModel:
    """Calibrate the open CGE from a balanced open SAM with ``a_<s>``/``c_<s>`` activity/commodity
    accounts, factors, one household, and a ``ROW`` account."""
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

    # Value added (Cobb-Douglas) — activity output value less intermediate composite cost.
    VA0 = Z0 - INT0.sum(axis=0)
    if float(VA0.min()) <= 0:
        raise ValueError("open SAM has non-positive value added for some activity")
    va_share = VA0 / Z0
    ax = INT0 / Z0[None, :]
    beta = F0 / VA0[None, :]
    av = np.prod(
        np.power(np.where(beta > 0, 1.0 / np.where(beta > 0, beta, 1.0), 1.0), beta), axis=0
    )
    gamma = FD0 / FD0.sum()
    endowment = F0.sum(axis=1)

    arm = np.full(ns, float(arm_elast)) if np.isscalar(arm_elast) else np.asarray(arm_elast, float)
    cet = np.full(ns, float(cet_elast)) if np.isscalar(cet_elast) else np.asarray(cet_elast, float)

    # Armington CES: Q = A[δ D^ρ + (1−δ) M^ρ]^{1/ρ}, ρ=(σ−1)/σ. At benchmark (pd=pm=1) the cost-min
    # FOC gives D/M = (δ/(1−δ))^σ, so δ = ratio/(1+ratio) with ratio = (D/M)^{1/σ}; then A from the
    # aggregator. (Forms verified against Shephard's lemma — see the module tests.) A non-traded
    # good (M=0) has δ=1 and A = Q/D.
    rho = (arm - 1.0) / arm
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_dm = np.where(M0 > 0, np.power(D0 / np.where(M0 > 0, M0, 1.0), 1.0 / arm), np.inf)
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
    cet_share_d = np.where(E0 > 0, ratio_de / (1.0 + ratio_de), 1.0)
    cet_share_e = 1.0 - cet_share_d
    cet_scale = np.where(
        E0 > 0,
        Z0 / np.power(cet_share_d * D0**kappa + cet_share_e * E0**kappa, 1.0 / kappa),
        Z0 / D0,
    )

    foreign_savings = float(M0.sum() - E0.sum())  # 0 for a balanced trade account

    return OpenCalibratedModel(
        sectors=list(sectors),
        factors=list(factors),
        ax=ax,
        va_share=va_share,
        beta=beta,
        av=av,
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
