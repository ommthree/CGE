"""Calibration: a benchmark SAM → static-CGE parameters (roadmap Phase 5.2b).

Standard CGE calibration [Hosoe2010, ch. 4-6]: read the benchmark flows from the SAM and back out
the structural parameters (Leontief input-output coefficients, Cobb-Douglas value-added and
household share parameters, and scale constants) so that **the model reproduces the benchmark SAM
exactly at the benchmark prices (all prices = 1)**. Calibration is deterministic and unit-tested
against the hand-computed pilot parameters; it is the foundation of the replication test.

Model structure calibrated here (the pilot; see docs/models/cge-static.md):

- **Production** — Leontief in intermediates, Cobb-Douglas in value added (the KLEM-with-Leontief-M
  baseline). For sector ``i``:
    intermediate demand   INT[j,i] = ax[j,i] · X[i]           (Leontief)
    value added           VA[i]    = av[i] · Π_f F[f,i]^β[f,i] (Cobb-Douglas, Σ_f β[f,i] = 1)
    zero profit           px[i]    = Σ_j ax[j,i]·p[j] + pv[i], pv[i] the VA unit cost.
- **Household** — Cobb-Douglas over commodities: budget shares γ[i] from benchmark final demand.
- **Factors** — fixed endowments FF[f] = total benchmark factor income.

At the benchmark all prices are 1, so the coefficients read straight off the SAM money flows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cge.contracts.data_objects import SAM
from cge.engines.cge_static.calibrate_open import _elast_vector


@dataclass(frozen=True)
class CalibratedModel:
    """Benchmark parameters for the static CGE. Vectors/matrices are indexed by the ``sectors`` /
    ``factors`` order. All benchmark prices are 1 by construction."""

    sectors: list[str]
    factors: list[str]
    ax: np.ndarray  # [j, i] Leontief intermediate coefficients (input j per unit output i)
    va_share: np.ndarray  # [i] value added per unit output (Leontief VA requirement)
    beta: np.ndarray  # [f, i] Cobb-Douglas value-added share of factor f in sector i (cols sum→1)
    av: np.ndarray  # [i] value-added scale constant (CD or CES, so unit VA cost = 1 at benchmark)
    gamma: np.ndarray  # [i] household Cobb-Douglas budget share (sums→1)
    endowment: np.ndarray  # [f] fixed factor endowment (benchmark factor income)
    X0: np.ndarray  # [i] benchmark gross output
    F0: np.ndarray  # [f, i] benchmark factor demand
    Z0: np.ndarray  # [j, i] benchmark intermediate flows
    FD0: np.ndarray  # [i] benchmark household final demand
    # Value-added nest elasticity of substitution between factors. σ = 1 ⇒ Cobb-Douglas (uses
    # ``beta``); σ ≠ 1 ⇒ CES (uses ``va_ces_share``). Per sector.
    va_elast: np.ndarray  # [i] VA substitution elasticity σ_va
    va_ces_share: np.ndarray  # [f, i] CES factor share δ_{fi} (cols sum→1); used when σ ≠ 1
    # Government account (Phase 5d.1; optional — None means no government account was declared,
    # and the model behaves exactly as before: 100% of any carbon revenue goes to the household).
    gov_gamma: np.ndarray | None = None  # [i] government Cobb-Douglas demand share (sums→1)
    gov_income0: float = 0.0  # benchmark government income (GDP-normalised, = benchmark tax)
    GD0: np.ndarray | None = None  # [i] benchmark government final demand (GDP-normalised)
    # Benchmark direct-tax rate on factor income: the household→government benchmark transfer as a
    # share of benchmark factor income. The model levies tax = rate·(current factor income), so the
    # benchmark government replicates exactly AND homogeneity survives (a fixed *level* would not
    # scale with the economy; a fixed *rate* does).
    gov_tax_rate0: float = 0.0
    # Savings-investment account (Phase 5d.2; optional — None means no account was declared and
    # the model behaves exactly as before: no savings, no investment demand). Financed by a
    # household savings RATE on disposable income (same rate-not-level logic as the tax).
    inv_gamma: np.ndarray | None = None  # [i] investment demand composition (sums→1)
    INV0: np.ndarray | None = None  # [i] benchmark investment demand (GDP-normalised)
    sav_rate0: float = 0.0  # benchmark household savings rate on disposable income

    @property
    def gdp0(self) -> float:
        """Benchmark GDP = total value added = total final demand."""
        return float(self.F0.sum())

    @property
    def has_government(self) -> bool:
        return self.gov_gamma is not None

    @property
    def has_investment(self) -> bool:
        return self.inv_gamma is not None


def calibrate(
    sam: SAM,
    *,
    sectors: list[str],
    factors: list[str],
    va_elast: float | np.ndarray = 1.0,
    institutions: dict[str, str] | None = None,
) -> CalibratedModel:
    """Calibrate the pilot CGE from a balanced ``sam``. ``sectors`` and ``factors`` name the SAM
    accounts to treat as activities/commodities and factors.

    ``institutions`` (Phase 5d.1/5d.2) maps a role to a SAM account name — ``"household"``
    (required if the dict is given) and, optionally, ``"government"`` and
    ``"savings_investment"``. The savings-investment account's column is investment demand by
    sector; its single supported receipt is household savings, converted to a savings **rate on
    disposable income** (``sav_rate0``). When omitted (the default), every
    non-sector/non-factor account must be exactly one account, treated as the household — this is
    the pre-5d.1 behaviour, preserved unchanged so every existing single-household SAM/fixture
    calibrates identically. When a ``"government"`` account is named, its column (demand for each
    sector) calibrates a Cobb-Douglas government demand vector ``gov_gamma``, and its benchmark
    financing is read off the SAM: the single supported benchmark flow is a household→government
    direct tax (cell ``[government, household]``), converted to a **rate on factor income**
    (``gov_tax_rate0``) so the model's benchmark replicates and homogeneity survives. A zero
    government row/column is the common case (no benchmark fiscal flows — the model's own
    carbon-revenue recycling funds government post-shock; see ``model.py``). Production/factor
    taxes and government→household transfers are rejected explicitly (5d follow-ups).

    ``va_elast`` is the value-added substitution elasticity σ_va (scalar or per-sector). σ = 1 is
    Cobb-Douglas (the default, preserving the pilot); σ ≠ 1 is CES, calibrated so the VA unit cost
    is still 1 at benchmark."""
    m = sam.matrix
    if institutions is None:
        hoh = [a for a in sam.accounts if a not in sectors and a not in factors]
        if len(hoh) != 1:
            raise ValueError(
                f"pilot calibration expects exactly one institution account, got {hoh}"
            )
        household = hoh[0]
        government = None
        savinv = None
    else:
        if "household" not in institutions:
            raise ValueError("institutions must name a 'household' role")
        household = institutions["household"]
        government = institutions.get("government")
        savinv = institutions.get("savings_investment")
        known = {household} | {x for x in (government, savinv) if x}
        unnamed = [
            a for a in sam.accounts if a not in sectors and a not in factors and a not in known
        ]
        if unnamed:
            raise ValueError(f"institutions did not account for SAM accounts {unnamed}")

    # Benchmark flows straight off the SAM (prices = 1 ⇒ money flow = quantity).
    Z0 = np.array([[m.loc[j, i] for i in sectors] for j in sectors], dtype=float)  # [j,i]
    F0 = np.array([[m.loc[f, i] for i in sectors] for f in factors], dtype=float)  # [f,i]
    FD0 = np.array([m.loc[i, household] for i in sectors], dtype=float)  # [i]
    GD0 = None
    gov_tax0 = 0.0
    if government is not None:
        GD0 = np.array([m.loc[i, government] for i in sectors], dtype=float)  # [i] gov demand
        gov_tax0 = float(m.loc[government, household])  # household→government benchmark transfer
        # 5d.1 models exactly ONE government financing flow at benchmark: a direct (lump-sum-like)
        # tax on the household, levied proportionally to factor income in the model. Any other
        # benchmark government receipt/outlay would enter without a modelled counterpart —
        # breaking Walras silently — so reject them explicitly rather than mis-calibrate:
        # production/factor taxes and government→household transfers are documented 5d follow-ups.
        bad_receipts = {
            a: float(m.loc[government, a])
            for a in sectors + factors + ([savinv] if savinv else [])
            if abs(float(m.loc[government, a])) > 1e-9
        }
        if bad_receipts:
            raise ValueError(
                f"government account receives from sectors/factors {sorted(bad_receipts)}: "
                "production/factor taxes are not yet modelled (Phase 5d follow-up); 5d.1 supports "
                "only a household→government benchmark transfer."
            )
        bad_outlays = [
            a
            for a in [household] + ([savinv] if savinv else [])
            if abs(float(m.loc[a, government])) > 1e-9
        ]
        if bad_outlays:
            raise ValueError(
                f"government pays accounts {bad_outlays}: transfers and government savings are "
                "not yet modelled (Phase 5d follow-up; the balanced_budget closure has no surplus "
                "to save); the government account may only buy commodities."
            )
        if abs(gov_tax0 - float(GD0.sum())) > 1e-6 * max(1.0, gov_tax0):
            raise ValueError(
                f"government account is unbalanced: benchmark receipts {gov_tax0:.6g} ≠ spending "
                f"{float(GD0.sum()):.6g}. A balanced SAM with only the supported flows cannot "
                "produce this; check the government column/row."
            )
    INV0 = None
    sav0 = 0.0
    if savinv is not None:
        # Savings-investment account (Phase 5d.2): its column is investment demand by sector; its
        # single supported receipt is household savings (cell [savinv, household]). Anything else
        # (sector/factor receipts, transfers out to institutions) has no modelled counterpart.
        INV0 = np.array([m.loc[i, savinv] for i in sectors], dtype=float)
        sav0 = float(m.loc[savinv, household])
        bad_receipts = {
            a: float(m.loc[savinv, a])
            for a in sectors + factors
            if abs(float(m.loc[savinv, a])) > 1e-9
        }
        if bad_receipts:
            raise ValueError(
                f"savings-investment account receives from sectors/factors "
                f"{sorted(bad_receipts)}: only household savings are modelled (Phase 5d "
                "follow-up for retained earnings / foreign savings in the closed model)."
            )
        if abs(float(m.loc[household, savinv])) > 1e-9:
            raise ValueError(
                "savings-investment→household transfers are not modelled; the account may only "
                "buy commodities (investment demand)."
            )
        if abs(sav0 - float(INV0.sum())) > 1e-6 * max(1.0, sav0):
            raise ValueError(
                f"savings-investment account is unbalanced: savings {sav0:.6g} ≠ investment "
                f"{float(INV0.sum()):.6g}; check the account's row/column."
            )
    # Normalise all levels by GDP so magnitudes are O(1): a CGE is homogeneous of degree zero, so
    # the *level* scale is arbitrary and results are relative changes. Real EXIOBASE flows are
    # ~1e9, where absolute solver residuals never reach a 1e-9 tolerance; unit-scaling fixes that
    # without changing any calibrated ratio or reported change (it cancels in every % result).
    scale = float(F0.sum())  # benchmark GDP (= total value added)
    if scale <= 0:
        raise ValueError("SAM has non-positive total value added; cannot calibrate")
    Z0, F0, FD0 = Z0 / scale, F0 / scale, FD0 / scale
    if GD0 is not None:
        GD0, gov_tax0 = GD0 / scale, gov_tax0 / scale
    if INV0 is not None:
        INV0, sav0 = INV0 / scale, sav0 / scale
    X0 = Z0.sum(axis=0) + F0.sum(axis=0)  # output = Σ intermediate cost + Σ value added

    # Reject degenerate benchmarks before dividing by them (review robustness): a zero-output
    # sector, a sector with no value added (β/av undefined), or non-positive final demand would
    # otherwise produce NaN/inf shares.
    if float(X0.min()) <= 0:
        bad = [s for s, x in zip(sectors, X0, strict=True) if x <= 0]
        raise ValueError(f"SAM has zero/negative gross output for sectors {bad}; cannot calibrate")
    VA0_check = F0.sum(axis=0)
    if float(VA0_check.min()) <= 0:
        bad = [s for s, v in zip(sectors, VA0_check, strict=True) if v <= 0]
        raise ValueError(f"SAM has zero value added for sectors {bad}; Cobb-Douglas VA undefined")
    if float(FD0.min()) < 0 or float(FD0.sum()) <= 0:
        raise ValueError("SAM household final demand must be non-negative with a positive total")

    # Leontief intermediate coefficients: input j per unit output i.
    ax = Z0 / X0[None, :]
    # Value added per unit output (Leontief VA requirement): VA0[i]/X0[i]. With unit VA cost 1 at
    # benchmark, the zero-profit price Σ_j ax[j,i] + va_share[i] = 1 exactly (checked in tests).
    VA0 = F0.sum(axis=0)  # [i]
    va_share = VA0 / X0
    # Cobb-Douglas factor shares within value added (columns sum to 1).
    beta = F0 / VA0[None, :]  # [f,i]
    ns = len(sectors)
    # Validate σ_va: scalar or exactly (ns,), finite, strictly positive — a length-1 vector used to
    # raise a raw IndexError and a non-positive value was silently used (review P2).
    sigma_va = _elast_vector(va_elast, ns, "va_elast")
    # CES factor share δ (used when σ ≠ 1): from the benchmark factor mix, δ_f/δ_g = (F_f/F_g)^{1/σ}
    # at unit factor prices, normalised to sum to 1 per sector. (For σ = 1 δ = β, harmless.)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(F0 > 0, np.power(F0, 1.0 / sigma_va[None, :]), 0.0)
        va_ces_share = weights / weights.sum(axis=0)[None, :]
    # Scale ``av`` so the VA unit cost is exactly 1 at benchmark (w = 1). CD: av = Π (1/β)^β; CES:
    # av = [Σ_f δ_f^σ]^{1/(1-σ)}. Chosen per sector by whether σ = 1.
    av = np.empty(ns)
    for i in range(ns):
        if abs(sigma_va[i] - 1.0) < 1e-12:
            b = beta[:, i]
            av[i] = np.prod(np.power(np.where(b > 0, 1.0 / np.where(b > 0, b, 1.0), 1.0), b))
        else:
            s = sigma_va[i]
            d = va_ces_share[:, i]
            av[i] = np.power(np.sum(d**s), 1.0 / (1.0 - s))
    # Household Cobb-Douglas budget shares (of total final demand).
    gamma = FD0 / FD0.sum()
    # Fixed factor endowments = total benchmark factor income.
    endowment = F0.sum(axis=1)  # [f]

    # Government Cobb-Douglas demand shares (Phase 5d.1), from the SAM's government row. A
    # benchmark with a zero government row (no pre-existing tax/transfer flows — the common case,
    # since the model's own carbon-revenue recycling is what funds government post-shock) has no
    # well-defined *demand composition* to calibrate; fall back to the household's own gamma so a
    # `balanced_budget` closure with no benchmark government spending still has a sensible
    # commodity mix to spend recycled revenue on, rather than an undefined 0/0 shares vector.
    gov_gamma = None
    gov_income0 = 0.0
    gov_tax_rate0 = 0.0
    if GD0 is not None:
        gov_income0 = float(GD0.sum())
        gov_gamma = GD0 / gov_income0 if gov_income0 > 0 else gamma.copy()
        # Benchmark direct-tax RATE on factor income (endowment.sum() is benchmark factor income,
        # = 1 after GDP normalisation). The model levies rate·(current factor income) so the
        # benchmark replicates AND the tax scales with the economy (homogeneity — a fixed level
        # would not survive an endowment rescaling; a fixed rate does).
        gov_tax_rate0 = gov_tax0 / float(endowment.sum())

    # Investment composition + household savings rate (Phase 5d.2). The rate is on DISPOSABLE
    # income (factor income net of the benchmark tax) — the same rate-not-level logic as the tax,
    # so replication and homogeneity both survive. A zero-column account falls back to the
    # household's gamma for the composition (no 0/0), with a zero savings rate.
    inv_gamma = None
    sav_rate0 = 0.0
    if INV0 is not None:
        inv_total = float(INV0.sum())
        inv_gamma = INV0 / inv_total if inv_total > 0 else gamma.copy()
        disposable0 = float(endowment.sum()) - gov_tax0
        sav_rate0 = sav0 / disposable0

    return CalibratedModel(
        sectors=list(sectors),
        factors=list(factors),
        ax=ax,
        va_share=va_share,
        beta=beta,
        av=av,
        gamma=gamma,
        endowment=endowment,
        X0=X0,
        F0=F0,
        Z0=Z0,
        FD0=FD0,
        va_elast=sigma_va,
        va_ces_share=va_ces_share,
        gov_gamma=gov_gamma,
        gov_income0=gov_income0,
        GD0=GD0,
        gov_tax_rate0=gov_tax_rate0,
        inv_gamma=inv_gamma,
        INV0=INV0,
        sav_rate0=sav_rate0,
    )
