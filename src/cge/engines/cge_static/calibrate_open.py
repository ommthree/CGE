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
from cge.engines.cge_static.energy_nest import (
    DEFAULT_ENERGY_ELAST,
    DEFAULT_KL_E_ELAST,
    DEFAULT_KLE_M_ELAST,
    calibrate_energy_nest,
)

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
    # Government account (Phase 5d.1; None ⇒ no government declared — pre-5d.1 behaviour, carbon
    # revenue recycles straight to the household). Mirrors the closed model's fields exactly.
    gov_gamma: np.ndarray | None = None  # [i] government CD demand share over composites (→1)
    gov_income0: float = 0.0  # benchmark government income (GDP-normalised)
    GD0: np.ndarray | None = None  # [i] benchmark government demand on Q (GDP-normalised)
    gov_tax_rate0: float = 0.0  # benchmark household→government tax as a rate on factor income
    # Savings-investment account (Phase 5d.2; None ⇒ no account — pre-5d.2 behaviour, er·Sf goes
    # to household income). With the account, foreign savings RE-ROUTE into the investment pool
    # (SAM: ROW↔SAVINV transfer instead of ROW↔household): investment = household savings + er·Sf.
    inv_gamma: np.ndarray | None = None  # [i] investment demand composition over composites (→1)
    INV0: np.ndarray | None = None  # [i] benchmark investment demand on Q (GDP-normalised)
    sav_rate0: float = 0.0  # benchmark household savings rate on disposable income
    # Energy nest (Phase 5d.5; None ⇒ flat Leontief production, bit-identical to pre-5d.5). The
    # intermediates are the Armington COMPOSITE commodities, so the energy nest substitutes over
    # composite prices (imports included) — a carbon price on an energy commodity raises its
    # composite price and shifts substitution within the energy bundle.
    energy_nest: object | None = None  # EnergyNest | None

    @property
    def gdp0(self) -> float:
        return float(self.F0.sum())

    @property
    def has_government(self) -> bool:
        return self.gov_gamma is not None

    @property
    def has_investment(self) -> bool:
        return self.inv_gamma is not None

    @property
    def has_energy_nest(self) -> bool:
        return self.energy_nest is not None


def calibrate_open(
    sam: SAM,
    *,
    sectors: list[str],
    factors: list[str],
    arm_elast: float | np.ndarray = DEFAULT_ARMINGTON_ELAST,
    cet_elast: float | np.ndarray = DEFAULT_CET_ELAST,
    va_elast: float | np.ndarray = 1.0,
    institutions: dict[str, str] | None = None,
    energy_sectors: list[str] | None = None,
    energy_elasticities: dict[str, float] | None = None,
) -> OpenCalibratedModel:
    """Calibrate the open CGE from a balanced open SAM with ``a_<s>``/``c_<s>`` activity/commodity
    accounts, factors, one household, and a ``ROW`` account. ``va_elast`` is the value-added
    substitution elasticity σ_va (1 ⇒ Cobb-Douglas, the default; ≠ 1 ⇒ CES).

    ``institutions`` (Phase 5d.1/5d.2) mirrors the closed calibration exactly: a role→account
    mapping with ``"household"`` required (if the dict is given), ``"government"`` and
    ``"savings_investment"`` optional. Omitted ⇒ exactly one non-activity/commodity/factor/ROW
    account, treated as the household (pre-5d.1 behaviour, unchanged). A government buys
    **composite commodities** (its ``c_<s>`` column cells) and is financed by a
    household→government direct tax, stored as a rate on factor income. A savings-investment
    account buys composites (investment demand) financed by household savings (a rate on
    disposable income) **plus the foreign-savings inflow — the ROW capital transfer must route
    ROW↔savings-investment, not ROW↔household**, the genuine closure change of 5d.2: foreign
    savings finance domestic investment, not consumption. Government trade (GOV↔ROW flows),
    production/factor taxes and institutional transfers not listed above are rejected explicitly
    (documented 5d follow-ups)."""
    m = sam.matrix
    act = [f"a_{s}" for s in sectors]
    com = [f"c_{s}" for s in sectors]
    if institutions is None:
        hoh = [
            a
            for a in sam.accounts
            if a not in act and a not in com and a not in factors and a != "ROW"
        ]
        if len(hoh) != 1:
            raise ValueError(f"open calibration expects exactly one household account, got {hoh}")
        household = hoh[0]
        government = None
        savinv = None
    else:
        if "household" not in institutions:
            raise ValueError("institutions must name a 'household' role")
        household = institutions["household"]
        government = institutions.get("government")
        savinv = institutions.get("savings_investment")
        known = {household, "ROW"} | {x for x in (government, savinv) if x}
        unnamed = [
            a
            for a in sam.accounts
            if a not in act and a not in com and a not in factors and a not in known
        ]
        if unnamed:
            raise ValueError(f"institutions did not account for SAM accounts {unnamed}")
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
    GD0 = None
    gov_tax0 = 0.0
    if government is not None:
        GD0 = np.array([m.loc[com[i], government] for i in range(ns)], dtype=float)
        gov_tax0 = float(m.loc[government, household])
        # Same 5d.1 discipline as the closed model: the ONLY supported benchmark government flows
        # are commodity purchases (the GD0 column) financed by a household direct tax. Anything
        # else would enter without a modelled counterpart and silently break Walras.
        bad_receipts = {
            a: float(m.loc[government, a])
            for a in act + com + list(factors) + ["ROW"] + ([savinv] if savinv else [])
            if abs(float(m.loc[government, a])) > 1e-9
        }
        if bad_receipts:
            raise ValueError(
                f"government account receives from accounts {sorted(bad_receipts)}: "
                "production/factor taxes and government trade are not yet modelled (Phase 5d "
                "follow-up); 5d.1 supports only a household→government benchmark transfer."
            )
        bad_outlays = {
            a: float(m.loc[a, government])
            for a in [household, "ROW", *act, *factors, *((savinv,) if savinv else ())]
            if abs(float(m.loc[a, government])) > 1e-9
        }
        if bad_outlays:
            raise ValueError(
                f"government account pays accounts {sorted(bad_outlays)}: transfers and "
                "non-commodity outlays are not yet modelled (Phase 5d follow-up); the government "
                "may only buy composite commodities (its c_<s> column)."
            )
        if abs(gov_tax0 - float(GD0.sum())) > 1e-6 * max(1.0, gov_tax0):
            raise ValueError(
                f"government account is unbalanced: benchmark receipts {gov_tax0:.6g} ≠ spending "
                f"{float(GD0.sum()):.6g}; check the government column/row."
            )
    INV0 = None
    sav0 = 0.0
    if savinv is not None:
        # Savings-investment account (Phase 5d.2): buys composites, financed by household savings
        # plus the NET foreign-savings inflow (the ROW capital transfer routes here, not to the
        # household — checked below against net trade). Anything else is unsupported.
        INV0 = np.array([m.loc[com[i], savinv] for i in range(ns)], dtype=float)
        sav0 = float(m.loc[savinv, household])
        bad_receipts = {
            a: float(m.loc[savinv, a])
            for a in act + com + list(factors)
            if abs(float(m.loc[savinv, a])) > 1e-9
        }
        if bad_receipts:
            raise ValueError(
                f"savings-investment account receives from accounts {sorted(bad_receipts)}: only "
                "household savings and the ROW capital transfer are modelled (Phase 5d follow-up "
                "for retained earnings)."
            )
        if abs(float(m.loc[household, savinv])) > 1e-9:
            raise ValueError(
                "savings-investment→household transfers are not modelled; the account may only "
                "buy composite commodities (investment demand)."
            )
    # Composite USE of commodity i = Σ_activities (row-sum) + household + government + investment.
    Q0 = (
        INT0.sum(axis=1)
        + FD0
        + (GD0 if GD0 is not None else 0.0)
        + (INV0 if INV0 is not None else 0.0)
    )

    # Normalise by GDP (scale-free; keeps solver residuals meaningful — same as the closed model).
    scale = float(F0.sum())
    if scale <= 0:
        raise ValueError("open SAM has non-positive total value added; cannot calibrate")
    D0, E0, M0, Z0, INT0, F0, FD0, Q0 = (a / scale for a in (D0, E0, M0, Z0, INT0, F0, FD0, Q0))
    if GD0 is not None:
        GD0, gov_tax0 = GD0 / scale, gov_tax0 / scale
    if INV0 is not None:
        INV0, sav0 = INV0 / scale, sav0 / scale

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
    # Safe base for the import term: a non-traded good has M0=0, and with σ<1 (ρ<0) the unused
    # np.where branch would still evaluate 0**ρ = inf and emit a divide-by-zero warning (review P3:
    # structural zeros must be warning-free). Use 1.0 there; the arm_share_m=0 factor zeroes it, and
    # the whole term is discarded by the outer np.where for M0=0 anyway.
    M0_safe = np.where(M0 > 0, M0, 1.0)
    arm_scale = np.where(
        M0 > 0,
        Q0 / np.power(arm_share_d * D0**rho + arm_share_m * M0_safe**rho, 1.0 / rho),
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

    # Foreign savings Sf = Σ M − Σ E at benchmark: the net capital inflow financing a trade deficit
    # (Sf>0 ⇒ imports exceed exports). The rest of the world runs a matching capital account, which
    # the SAM records as a capital transfer in the direction of the flow: **ROW → household** for a
    # deficit (the inflow the household spends), **household → ROW** for a surplus (home lending
    # abroad) — review P1: reading only the ROW→HOH cell rejected every valid exporter SAM. The
    # closure fixes Sf at this benchmark level (a standard small-open-economy closure); household
    # income carries er·Sf (see model_open.derive_open_state) so the model replicates a
    # non-zero current account exactly. We read the NET transfer from the SAM and check it equals
    # Sf, so a mis-specified ROW account (transfer ≠ net trade) is caught rather than silently
    # producing a non-replicating benchmark.
    foreign_savings = float(M0.sum() - E0.sum())
    if savinv is None:
        # Net ROW → HOH capital transfer (normalised): inflow cell minus outflow cell. Pre-5d.2
        # routing: the household spends (or lends) the foreign-savings flow.
        row_transfer = float(m.loc[household, "ROW"] - m.loc["ROW", household]) / scale
        if abs(row_transfer - foreign_savings) > 1e-6 * max(1.0, abs(foreign_savings)):
            raise ValueError(
                f"open SAM's net ROW↔household transfer ({row_transfer:.6g}) must equal net "
                f"foreign savings Σimports−Σexports ({foreign_savings:.6g}); the rest-of-world "
                "capital account does not balance the current account. Check the SAM's ROW "
                "row/column."
            )
    else:
        # Phase 5d.2 routing: foreign savings finance INVESTMENT, not consumption — the ROW
        # capital transfer must run ROW↔savings-investment, and the household must have NO direct
        # ROW capital transfer (mixing both routes is rejected as a probable mis-specification).
        hh_row = float(m.loc[household, "ROW"] - m.loc["ROW", household]) / scale
        if abs(hh_row) > 1e-9:
            raise ValueError(
                "with a savings-investment account, the ROW capital transfer must route "
                "ROW↔savings-investment (foreign savings finance investment, not consumption); "
                f"found a net ROW↔household transfer of {hh_row:.6g}. Move it to the "
                "savings-investment account's row/column."
            )
        si_row = float(m.loc[savinv, "ROW"] - m.loc["ROW", savinv]) / scale
        if abs(si_row - foreign_savings) > 1e-6 * max(1.0, abs(foreign_savings)):
            raise ValueError(
                f"open SAM's net ROW↔savings-investment transfer ({si_row:.6g}) must equal net "
                f"foreign savings Σimports−Σexports ({foreign_savings:.6g}); the capital account "
                "does not balance the current account."
            )
        # And the account itself must balance: household savings + foreign savings = investment.
        if abs(sav0 + foreign_savings - float(INV0.sum())) > 1e-6 * max(1.0, float(INV0.sum())):
            raise ValueError(
                f"savings-investment account is unbalanced: household savings {sav0:.6g} + "
                f"foreign savings {foreign_savings:.6g} ≠ investment {float(INV0.sum()):.6g}."
            )

    # Government parameters (Phase 5d.1 — mirrors the closed calibration): CD demand shares from
    # the benchmark government column (falling back to the household's gamma when zero, so a
    # zero-column government still has a well-defined spending mix for recycled revenue), and the
    # benchmark direct tax as a RATE on factor income (replication + homogeneity both survive).
    gov_gamma = None
    gov_income0 = 0.0
    gov_tax_rate0 = 0.0
    if GD0 is not None:
        gov_income0 = float(GD0.sum())
        gov_gamma = GD0 / gov_income0 if gov_income0 > 0 else gamma.copy()
        gov_tax_rate0 = gov_tax0 / float(endowment.sum())

    # Investment composition + household savings rate (Phase 5d.2 — mirrors the closed
    # calibration; the rate is on DISPOSABLE income, factor income net of the benchmark tax).
    inv_gamma = None
    sav_rate0 = 0.0
    if INV0 is not None:
        inv_total = float(INV0.sum())
        inv_gamma = INV0 / inv_total if inv_total > 0 else gamma.copy()
        sav_rate0 = sav0 / (float(endowment.sum()) - gov_tax0)

    # Energy nest (Phase 5d.5). Opt-in; intermediates are the Armington COMPOSITE commodities, so
    # the nest is calibrated over composite intermediate flows (INT0) exactly like the closed model
    # over its intermediate flows. Reproduces the benchmark (Tier-1 replication).
    energy_nest = None
    if energy_sectors:
        unknown = [e for e in energy_sectors if e not in sectors]
        if unknown:
            raise ValueError(f"energy_sectors {unknown} are not in the sector list {sectors}")
        e_idx = np.array([sectors.index(e) for e in energy_sectors], dtype=int)
        el = energy_elasticities or {}
        energy_nest = calibrate_energy_nest(
            INT0,
            VA0,
            Z0,
            e_idx,
            kle_m_elast=el.get("kle_m", DEFAULT_KLE_M_ELAST),
            kl_e_elast=el.get("kl_e", DEFAULT_KL_E_ELAST),
            energy_elast=el.get("energy", DEFAULT_ENERGY_ELAST),
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
        gov_gamma=gov_gamma,
        gov_income0=gov_income0,
        GD0=GD0,
        gov_tax_rate0=gov_tax_rate0,
        inv_gamma=inv_gamma,
        INV0=INV0,
        sav_rate0=sav_rate0,
        energy_nest=energy_nest,
    )
