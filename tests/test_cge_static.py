"""Unit tests for Engine 3 (static CGE pilot): SAM, calibration, model, engine.

Pins the CGE correctness battery (replication, homogeneity, Walras) and the pilot's economic
behaviour, plus calibration exactness and the engine wrapper. All run on the hand-checkable
2-sector toy SAM via the scipy solver fallback (no IPOPT binary needed).
"""

from dataclasses import replace

import numpy as np
import pytest

from cge.contracts.engine import registry
from cge.contracts.shocks import CarbonPrice
from cge.data.sam import toy_sam
from cge.data.sam.balance import imbalance, is_balanced, ras_balance
from cge.engines.cge_static import model as M
from cge.engines.cge_static.calibrate import calibrate
from cge.engines.cge_static.solver import solve

_SECTORS = ["BRD", "MIL"]
_FACTORS = ["CAP", "LAB"]
_EMISSIONS = np.array([2.0, 0.5])  # BRD dirty, MIL clean


def _cal():
    return calibrate(toy_sam(), sectors=_SECTORS, factors=_FACTORS)


def _solve(cal, carbon_cost=None, drop_factor=0, recycling="lump_sum"):
    cc = np.zeros(len(cal.sectors)) if carbon_cost is None else carbon_cost
    sol = solve(
        lambda z: M.residuals(cal, z, carbon_cost=cc, recycling=recycling, drop_factor=drop_factor),
        M.initial_guess(cal),
        prefer="scipy",
    )
    ns = len(cal.sectors)
    return sol, M.derive_state(cal, sol.x[:ns], sol.x[ns:], carbon_cost=cc, recycling=recycling)


# -- SAM ----------------------------------------------------------------------
def test_toy_sam_is_balanced():
    sam = toy_sam()
    assert is_balanced(sam.matrix)
    assert imbalance(sam.matrix).abs().max() < 1e-9
    # GDP identity: total value added = total final demand = 181
    va = sam.matrix.loc[_FACTORS, _SECTORS].to_numpy().sum()
    fd = sam.matrix.loc[_SECTORS, "HOH"].to_numpy().sum()
    assert np.isclose(va, 181.0) and np.isclose(fd, 181.0)


def test_ras_balances_an_unbalanced_matrix():
    import pandas as pd

    m = pd.DataFrame([[1.0, 2.0], [3.0, 1.0]], index=["a", "b"], columns=["a", "b"])
    targets = pd.Series({"a": 4.0, "b": 4.0})
    balanced = ras_balance(m, targets)
    assert np.allclose(balanced.sum(axis=1), balanced.sum(axis=0), atol=1e-8)
    assert np.allclose(balanced.sum(axis=1), [4.0, 4.0], atol=1e-8)


# -- calibration --------------------------------------------------------------
def test_calibration_reproduces_benchmark_parameters():
    cal = _cal()
    # Levels are GDP-normalised (a CGE is scale-free); ratios and shares are what matter.
    assert np.allclose(cal.X0, [100.0 / 181.0, 120.0 / 181.0])  # scaled by benchmark GDP 181
    assert np.allclose(cal.ax, [[0.0, 0.2], [0.15, 0.0]])  # input j per unit output i (scale-free)
    assert np.allclose(cal.va_share, [0.85, 0.8])
    assert np.allclose(cal.beta, [[0.5, 0.5], [0.5, 0.5]])  # 50/50 CAP/LAB
    assert np.allclose(cal.gamma.sum(), 1.0)
    assert np.allclose(cal.endowment, [90.5 / 181.0, 90.5 / 181.0])  # normalised factor income
    assert np.isclose(cal.gdp0, 1.0)  # GDP normalised to 1


def test_zero_profit_holds_at_benchmark():
    """Σ_j ax[j,i] + va_share[i] = 1 at benchmark prices (the calibration identity)."""
    cal = _cal()
    for i in range(len(cal.sectors)):
        assert np.isclose(cal.ax[:, i].sum() + cal.va_share[i], 1.0)


# -- government account (Phase 5d.1) ------------------------------------------
def _gov_sam():
    """The toy SAM plus a GOV account with a zero benchmark row — the common case (no
    pre-existing tax/transfer flows; government income comes entirely from post-shock carbon
    revenue recycling)."""
    import pandas as pd

    sam = toy_sam()
    acc = list(sam.accounts) + ["GOV"]
    m = pd.DataFrame(0.0, index=acc, columns=acc)
    m.loc[sam.accounts, sam.accounts] = sam.matrix
    return sam.model_copy(update={"accounts": acc, "matrix": m})


def _cal_gov():
    return calibrate(
        _gov_sam(),
        sectors=_SECTORS,
        factors=_FACTORS,
        institutions={"household": "HOH", "government": "GOV"},
    )


def test_institutions_none_matches_pre_5d1_behaviour():
    """Omitting ``institutions`` calibrates identically to naming just the household explicitly —
    the regression-safety net for every pre-5d.1 caller."""
    cal_default = calibrate(toy_sam(), sectors=_SECTORS, factors=_FACTORS)
    cal_named = calibrate(
        toy_sam(), sectors=_SECTORS, factors=_FACTORS, institutions={"household": "HOH"}
    )
    assert not cal_default.has_government
    assert not cal_named.has_government
    assert np.allclose(cal_default.gamma, cal_named.gamma)
    assert np.allclose(cal_default.FD0, cal_named.FD0)


def test_institutions_rejects_unnamed_account():
    with pytest.raises(ValueError, match="did not account for"):
        calibrate(_gov_sam(), sectors=_SECTORS, factors=_FACTORS, institutions={"household": "HOH"})


def test_institutions_requires_household_role():
    with pytest.raises(ValueError, match="must name a 'household' role"):
        calibrate(
            _gov_sam(), sectors=_SECTORS, factors=_FACTORS, institutions={"government": "GOV"}
        )


def test_government_with_zero_benchmark_row_falls_back_to_household_gamma():
    """A GOV account with no benchmark spending has no demand composition of its own to read off
    the SAM; it falls back to the household's gamma rather than an undefined 0/0 share."""
    cal = _cal_gov()
    assert cal.has_government
    assert cal.gov_income0 == 0.0
    assert np.allclose(cal.gov_gamma, cal.gamma)


def test_government_benchmark_replication_holds():
    """Zero shock: with a government account present but zero carbon revenue, the benchmark still
    replicates exactly (government demand is zero, so this reduces to the pre-5d.1 benchmark)."""
    cal = _cal_gov()
    sol = solve(
        lambda z: M.residuals(cal, z, carbon_cost=np.zeros(len(cal.sectors)), recycling="none"),
        M.initial_guess(cal),
        prefer="scipy",
    )
    ns = len(cal.sectors)
    st = M.derive_state(cal, sol.x[:ns], sol.x[ns:], carbon_cost=np.zeros(ns), recycling="none")
    assert np.allclose(sol.x, 1.0, atol=1e-8)
    assert np.allclose(st.X, cal.X0, atol=1e-6)
    assert pytest.approx(0.0) == st.GD
    assert st.fiscal_balance == pytest.approx(0.0)


def test_carbon_revenue_goes_to_government_not_household():
    """With a government account, carbon revenue funds GOVERNMENT spending, not the household's
    own income — the point of 5d.1 (a real institutional transfer, not a same-period pass-through
    to the taxed account itself)."""
    cal = _cal_gov()
    cc = 0.2 * _EMISSIONS
    sol = solve(
        lambda z: M.residuals(cal, z, carbon_cost=cc, recycling="lump_sum"),
        M.initial_guess(cal),
        prefer="scipy",
    )
    ns = len(cal.sectors)
    st = M.derive_state(cal, sol.x[:ns], sol.x[ns:], carbon_cost=cc, recycling="lump_sum")
    assert st.income == pytest.approx(st.factor_income, rel=1e-6)  # no revenue in household income
    assert st.gov_income > 0.0  # government collected the revenue instead
    assert float(st.GD.sum()) > 0.0  # and spent it


def test_fiscal_balance_zero_under_balanced_budget():
    cal = _cal_gov()
    cc = 0.2 * _EMISSIONS
    sol = solve(
        lambda z: M.residuals(cal, z, carbon_cost=cc, recycling="lump_sum"),
        M.initial_guess(cal),
        prefer="scipy",
    )
    ns = len(cal.sectors)
    st = M.derive_state(cal, sol.x[:ns], sol.x[ns:], carbon_cost=cc, recycling="lump_sum")
    assert st.fiscal_balance == pytest.approx(0.0, abs=1e-9)
    assert st.gov_income == pytest.approx(float(cc @ st.X), rel=1e-6)  # income == revenue spent


def test_unsupported_gov_closure_rejected():
    cal = _cal_gov()
    ns = len(cal.sectors)
    with pytest.raises(ValueError, match="unsupported gov_closure"):
        M.derive_state(
            cal,
            np.ones(ns),
            np.ones(len(cal.factors)),
            carbon_cost=0.2 * _EMISSIONS,
            recycling="lump_sum",
            gov_closure="deficit_financed",
        )


def test_walras_law_holds_with_government_account():
    """Tier 1 re-proof (plan §0.1): adding the government account must not introduce a hidden
    money leak — dropping one factor-clearing equation by Walras' law must still hold at the
    solution."""
    cal = _cal_gov()
    cc = 0.2 * _EMISSIONS
    sol = solve(
        lambda z: M.residuals(cal, z, carbon_cost=cc, recycling="lump_sum", drop_factor=0),
        M.initial_guess(cal),
        prefer="scipy",
    )
    ns = len(cal.sectors)
    st = M.derive_state(cal, sol.x[:ns], sol.x[ns:], carbon_cost=cc, recycling="lump_sum")
    excess = float(st.F[0, :].sum()) - cal.endowment[0]
    assert abs(excess) < 1e-6


def test_zero_row_government_is_equivalent_to_household_recycling():
    """THE degenerate-equivalence regression (plan 5d.1 DoD): a zero-benchmark-row GOV account
    spends recycled revenue on the household's own gamma (the calibration fallback), so total
    demand γ·R is identical to handing the household R directly — the equilibrium prices, factor
    prices and outputs must match the no-government lump_sum run exactly; only the institutional
    attribution (income vs gov_income; FD vs FD+GD) differs."""
    cc = 0.2 * _EMISSIONS
    cal_plain = _cal()
    cal_gov = _cal_gov()
    sol_plain, st_plain = _solve(cal_plain, carbon_cost=cc)
    sol_gov = solve(
        lambda z: M.residuals(cal_gov, z, carbon_cost=cc, recycling="lump_sum"),
        M.initial_guess(cal_gov),
        prefer="scipy",
    )
    ns = len(cal_gov.sectors)
    st_gov = M.derive_state(
        cal_gov, sol_gov.x[:ns], sol_gov.x[ns:], carbon_cost=cc, recycling="lump_sum"
    )
    assert np.allclose(sol_plain.x, sol_gov.x, atol=1e-8)  # same equilibrium prices
    assert np.allclose(st_plain.X, st_gov.X, atol=1e-8)  # same outputs
    assert np.allclose(st_plain.FD, st_gov.FD + st_gov.GD, atol=1e-8)  # same total demand
    assert st_gov.income < st_plain.income  # but the household no longer receives the revenue


# -- government account with a BENCHMARK fiscal flow (tax-funded spending) ----
def _gov_funded_sam():
    """The toy SAM rebalanced by hand around a real benchmark fiscal flow: the household pays an
    18.1 direct tax to GOV (10% of factor income 181), which GOV spends on BRD (10) and MIL (8.1).
    Sector totals are unchanged (household demand shrinks by exactly what GOV buys), so the rest
    of the SAM — intermediates, value added, factor income — is untouched and still balanced.
    gov_gamma = (10, 8.1)/18.1 deliberately differs from the household's gamma."""
    import pandas as pd

    sam = toy_sam()
    acc = list(sam.accounts) + ["GOV"]
    m = pd.DataFrame(0.0, index=acc, columns=acc)
    m.loc[sam.accounts, sam.accounts] = sam.matrix
    m.loc["GOV", "HOH"] = 18.1  # household→government direct tax
    m.loc["BRD", "GOV"] = 10.0  # government demand
    m.loc["MIL", "GOV"] = 8.1
    m.loc["BRD", "HOH"] -= 10.0  # household demand shrinks by the same amounts: 76→66
    m.loc["MIL", "HOH"] -= 8.1  # 105→96.9
    return sam.model_copy(update={"accounts": acc, "matrix": m})


def _cal_gov_funded():
    return calibrate(
        _gov_funded_sam(),
        sectors=_SECTORS,
        factors=_FACTORS,
        institutions={"household": "HOH", "government": "GOV"},
    )


def test_benchmark_funded_government_calibrates():
    cal = _cal_gov_funded()
    assert cal.has_government
    assert cal.gov_income0 == pytest.approx(18.1 / 181.0)
    assert cal.gov_tax_rate0 == pytest.approx(0.1)  # 18.1 / 181 factor income
    assert np.allclose(cal.gov_gamma, [10.0 / 18.1, 8.1 / 18.1])
    assert not np.allclose(cal.gov_gamma, cal.gamma)  # composition genuinely differs


def test_benchmark_funded_government_replicates():
    """Tier 1: with a real benchmark fiscal flow (tax-funded government spending), the zero-shock
    benchmark must still reproduce the SAM exactly — prices 1, household demand FD0, government
    demand GD0, and a balanced budget."""
    cal = _cal_gov_funded()
    ns = len(cal.sectors)
    sol = solve(
        lambda z: M.residuals(cal, z, carbon_cost=np.zeros(ns), recycling="none"),
        M.initial_guess(cal),
        prefer="scipy",
    )
    st = M.derive_state(cal, sol.x[:ns], sol.x[ns:], carbon_cost=np.zeros(ns), recycling="none")
    assert np.allclose(sol.x, 1.0, atol=1e-8)
    assert np.allclose(st.FD, cal.FD0, atol=1e-8)
    assert np.allclose(st.GD, cal.GD0, atol=1e-8)
    assert np.allclose(st.X, cal.X0, atol=1e-6)
    assert st.gov_income == pytest.approx(cal.gov_income0, rel=1e-8)
    assert st.income == pytest.approx(1.0 - 18.1 / 181.0, rel=1e-8)  # factor income net of tax


def test_homogeneity_with_benchmark_funded_government():
    """Tier 1: the benchmark tax is a RATE on factor income, so scaling the endowment k× leaves
    prices unchanged and scales all reals — including government demand — by k. A fixed tax LEVEL
    would fail this (government's share would shrink as the economy grows)."""
    cal = _cal_gov_funded()
    ns = len(cal.sectors)

    def _solve_gov(c):
        sol = solve(
            lambda z: M.residuals(c, z, carbon_cost=np.zeros(ns), recycling="none"),
            M.initial_guess(c),
            prefer="scipy",
        )
        return sol, M.derive_state(
            c, sol.x[:ns], sol.x[ns:], carbon_cost=np.zeros(ns), recycling="none"
        )

    sol, st = _solve_gov(cal)
    k = 2.5
    cal_k = replace(cal, endowment=cal.endowment * k, X0=cal.X0 * k, F0=cal.F0 * k, Z0=cal.Z0 * k)
    sol_k, st_k = _solve_gov(cal_k)
    assert np.allclose(sol.x, sol_k.x, atol=1e-8)  # prices unchanged
    assert np.allclose(st_k.X, k * st.X, atol=1e-6)  # reals scale
    assert np.allclose(st_k.GD, k * st.GD, atol=1e-8)  # government demand scales too


def test_walras_law_with_benchmark_funded_government_under_shock():
    cal = _cal_gov_funded()
    cc = 0.2 * _EMISSIONS
    sol = solve(
        lambda z: M.residuals(cal, z, carbon_cost=cc, recycling="lump_sum", drop_factor=0),
        M.initial_guess(cal),
        prefer="scipy",
    )
    ns = len(cal.sectors)
    st = M.derive_state(cal, sol.x[:ns], sol.x[ns:], carbon_cost=cc, recycling="lump_sum")
    excess = float(st.F[0, :].sum()) - cal.endowment[0]
    assert abs(excess) < 1e-6
    # And the budget still balances: gov income (tax + revenue) is exactly exhausted by spending.
    assert st.gov_income == pytest.approx(
        cal.gov_tax_rate0 * st.factor_income + st.carbon_revenue, rel=1e-6
    )


def test_production_tax_in_benchmark_sam_rejected():
    """5d.1 models only a household→government benchmark transfer; a GOV receipt from a sector
    (a production/indirect tax) has no modelled counterpart and must be rejected, not silently
    mis-calibrated."""
    import pandas as pd

    sam = toy_sam()
    acc = list(sam.accounts) + ["GOV"]
    m = pd.DataFrame(0.0, index=acc, columns=acc)
    m.loc[sam.accounts, sam.accounts] = sam.matrix
    m.loc["GOV", "BRD"] = 5.0  # a production tax — unsupported
    bad = sam.model_copy(update={"accounts": acc, "matrix": m})
    with pytest.raises(ValueError, match="production/factor taxes are not yet modelled"):
        calibrate(
            bad,
            sectors=_SECTORS,
            factors=_FACTORS,
            institutions={"household": "HOH", "government": "GOV"},
        )


def test_gov_transfer_to_household_rejected():
    import pandas as pd

    sam = toy_sam()
    acc = list(sam.accounts) + ["GOV"]
    m = pd.DataFrame(0.0, index=acc, columns=acc)
    m.loc[sam.accounts, sam.accounts] = sam.matrix
    m.loc["HOH", "GOV"] = 5.0  # a government→household transfer — unsupported
    bad = sam.model_copy(update={"accounts": acc, "matrix": m})
    with pytest.raises(ValueError, match="transfers .* not yet modelled"):
        calibrate(
            bad,
            sectors=_SECTORS,
            factors=_FACTORS,
            institutions={"household": "HOH", "government": "GOV"},
        )


# -- engine wiring for the government account (Phase 5d.1) --------------------
def test_engine_gov_sam_emits_fiscal_variables():
    """A SAM carrying a GOV account runs through the engine end-to-end: fiscal_balance (≡0 under
    balanced_budget) and gov_spending are emitted, and the manifest records the government
    closure. A no-GOV run must NOT emit them (regression: pre-5d.1 output is unchanged)."""
    eng = registry.get("cge_static")
    data = {"SAM": _gov_funded_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}}
    res = eng.run(data=data, shocks=[CarbonPrice(price=0.15)], years=[2020])
    d = res.data
    fb = d[d["variable"] == "fiscal_balance"]["value"]
    gs = d[d["variable"] == "gov_spending"]["value"]
    assert len(fb) == 1 and abs(float(fb.iloc[0])) < 1e-9  # balanced budget, visibly pinned
    assert len(gs) == 1 and float(gs.iloc[0]) > 18.1 / 181.0  # benchmark tax + carbon revenue
    assert res.manifest.assumptions["government_account"] == "GOV"
    assert res.manifest.assumptions["gov_closure"] == "balanced_budget"
    assert res.manifest.assumptions["gov_benchmark_tax_share_of_factor_income"] == pytest.approx(
        0.1
    )

    plain = eng.run(
        data={"SAM": toy_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}},
        shocks=[CarbonPrice(price=0.15)],
        years=[2020],
    )
    assert not (plain.data["variable"] == "fiscal_balance").any()
    assert not (plain.data["variable"] == "gov_spending").any()
    assert plain.manifest.assumptions["government_account"] == "none"


def test_engine_gov_sam_zero_shock_replicates():
    """The engine's replication gate covers the government column too: a zero-price run on the
    tax-funded GOV SAM reproduces the benchmark (all changes ~0)."""
    eng = registry.get("cge_static")
    data = {"SAM": _gov_funded_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}}
    res = eng.run(data=data, shocks=[CarbonPrice(price=0.0)], years=[2020])
    d = res.data
    for v in ("price_change", "volume_change", "gdp_change_real", "gov_spending"):
        vals = d[d["variable"] == v]["value"].to_numpy()
        assert len(vals) > 0, v
        if v == "gov_spending":
            assert vals[0] == pytest.approx(18.1 / 181.0, rel=1e-6)  # benchmark level, not 0
        else:
            assert np.allclose(vals, 0.0, atol=1e-6), v


# -- CGE correctness battery --------------------------------------------------
def test_benchmark_replication():
    cal = _cal()
    _sol, st = _solve(cal)
    assert np.allclose(st.p, 1.0, atol=1e-8)
    assert np.allclose(st.X, cal.X0, atol=1e-6)
    assert np.allclose(st.FD, cal.FD0, atol=1e-6)
    assert np.allclose(st.F, cal.F0, atol=1e-6)


def test_homogeneity_degree_zero():
    cal = _cal()
    sol, st = _solve(cal)
    k = 2.5
    cal_k = replace(cal, endowment=cal.endowment * k, X0=cal.X0 * k, F0=cal.F0 * k, Z0=cal.Z0 * k)
    sol_k, st_k = _solve(cal_k)
    assert np.allclose(sol.x, sol_k.x, atol=1e-8)  # prices unchanged
    assert np.allclose(st_k.X, k * st.X, atol=1e-6)  # reals scale


def test_walras_law():
    cal = _cal()
    _sol, st = _solve(cal, drop_factor=0)
    excess = float(st.F[0, :].sum()) - cal.endowment[0]
    assert abs(excess) < 1e-8


# -- economic behaviour -------------------------------------------------------
def test_carbon_price_reallocates_dirty_to_clean():
    """With recycling, a carbon price shifts output from the dirty to the clean sector (rather than
    shrinking the economy): dirty output falls, clean output rises, dirty relative price rises."""
    cal = _cal()
    _b, base = _solve(cal)
    _s, st = _solve(cal, carbon_cost=0.2 * _EMISSIONS)
    assert st.X[0] < base.X[0]  # dirty BRD output falls
    assert st.X[1] > base.X[1]  # clean MIL output rises (the reallocation)
    assert st.p[0] / st.p[1] > base.p[0] / base.p[1]  # dirty price rises relative to clean


def test_larger_carbon_price_larger_response():
    """Monotonicity: a bigger carbon price contracts the dirty sector more."""
    cal = _cal()
    _, s1 = _solve(cal, carbon_cost=0.1 * _EMISSIONS)
    _, s2 = _solve(cal, carbon_cost=0.2 * _EMISSIONS)
    _b, base = _solve(cal)
    assert (base.X[0] - s2.X[0]) > (base.X[0] - s1.X[0]) > 0


# -- engine wrapper -----------------------------------------------------------
def test_engine_registered_with_ge_capability():
    import cge.engines  # noqa: F401

    meta = registry.get("cge_static").meta
    caps = [c.value for c in meta.capabilities]
    assert "general_equilibrium" in caps and "prices" in caps and "volumes" in caps


def test_engine_zero_shock_replicates():
    eng = registry.get("cge_static")
    res = eng.run(data={"SAM": toy_sam()}, shocks=[CarbonPrice(price=0.0)], years=[2020])
    assert res.data["value"].abs().max() < 1e-8  # every change is zero


def test_engine_carbon_price_emits_ge_outputs():
    eng = registry.get("cge_static")
    data = {"SAM": toy_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}}
    res = eng.run(data=data, shocks=[CarbonPrice(price=0.1)], years=[2020])
    d = res.data
    for v in ("price_change", "volume_change", "factor_price_change", "gdp_change_real"):
        assert (d["variable"] == v).any(), v
    # dirty sector volume falls
    brd = d[(d["variable"] == "volume_change") & (d["sector"] == "BRD")]["value"].iloc[0]
    assert brd < 0
    assert res.manifest.assumptions["emissions_priced"] is True


def test_engine_cross_check_dirty_sector_falls():
    """Cross-engine consistency: the CGE's dirty-sector volume change is negative, the same sign
    as the partial-equilibrium intuition (a carbon price cuts the emitting sector's output). With
    recycling the clean sector may rise — the GE reallocation Engine 2 cannot show."""
    eng = registry.get("cge_static")
    data = {"SAM": toy_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}}
    res = eng.run(data=data, shocks=[CarbonPrice(price=0.15)], years=[2020])
    d = res.data
    brd = d[(d["variable"] == "volume_change") & (d["sector"] == "BRD")]["value"].iloc[0]
    assert brd < 0  # the dirty sector contracts, same sign as Engine 2


def _welfare_engine(eng, mode):
    data = {"SAM": toy_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}}
    res = eng.run(data=data, shocks=[CarbonPrice(price=0.15, revenue_recycling=mode)], years=[2020])
    d = res.data
    w = float(d[d["variable"] == "welfare_change"]["value"].iloc[0])
    return w, res.manifest.assumptions["recycling_mode"]


def test_recycling_welfare_claims_are_valid():
    """What is actually established (review P2): (1) under a recycled carbon price the CD *utility*
    change is small and negative (the distortion); (2) at those equilibrium prices, adding the
    transfer raises utility vs not. This does NOT use the non-closing `none` equilibrium (which
    violates Walras' law and is not a valid counterfactual)."""
    cal = _cal()
    cc = 0.15 * _EMISSIONS
    _b, base = _solve(cal, recycling="lump_sum")
    _l, lump = _solve(cal, carbon_cost=cc, recycling="lump_sum")

    def cd_u(state):
        return float(np.prod(np.power(state.FD, cal.gamma)))

    # (1) small negative CD-utility change under a recycled carbon price.
    welfare = cd_u(lump) / cd_u(base) - 1.0
    assert -0.05 < welfare < 0.0
    assert lump.carbon_revenue > 0
    # (2) at the recycled prices, the transfer raises utility (income is larger by R).
    factor_income = float(np.dot(lump.w, cal.endowment))
    fd_no_transfer = cal.gamma * factor_income / lump.p
    u_with = float(np.prod(np.power(lump.FD, cal.gamma)))
    u_without = float(np.prod(np.power(fd_no_transfer, cal.gamma)))
    assert u_with > u_without


def test_labour_tax_cut_equivalent_to_lump_sum_in_pilot():
    """With one household, a labour rebate and a lump-sum transfer give the same real allocation
    (documented pilot equivalence)."""
    eng = registry.get("cge_static")
    w_lump, _ = _welfare_engine(eng, "lump_sum")
    w_lab, mode = _welfare_engine(eng, "labour_tax_cut")
    assert np.isclose(w_lump, w_lab, atol=1e-9)
    assert mode == "labour_tax_cut"


def test_carbon_revenue_emitted():
    eng = registry.get("cge_static")
    data = {"SAM": toy_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}}
    res = eng.run(data=data, shocks=[CarbonPrice(price=0.1)], years=[2020])
    rev = res.data[res.data["variable"] == "carbon_revenue"]["value"].iloc[0]
    assert rev > 0  # a positive carbon price on emitting sectors raises revenue


def test_engine_rejects_mixed_recycling_modes():
    eng = registry.get("cge_static")
    with pytest.raises(ValueError, match="single revenue_recycling"):
        eng.run(
            data={"SAM": toy_sam()},
            shocks=[
                CarbonPrice(price=0.1, revenue_recycling="none"),
                CarbonPrice(price=0.1, revenue_recycling="lump_sum"),
            ],
            years=[2020],
        )


# -- review remediation: supplied-SAM gate, units, welfare, gas/coverage, provenance ----------
def _eur_io_sat(intensity_energy=800.0, intensity_mfg=200.0):
    """A small EUR IOSystem + satellite with realistic CO2 emission intensities (t/MEUR)."""
    import pandas as pd

    from cge.contracts.data_objects import (
        Classification,
        IOSystem,
        Provenance,
        SatelliteAccount,
    )

    prov = Provenance(
        source="x", source_version="1", licence="x", reference_year=2020, retrieved="2026-07-18"
    )
    labels = ["R:energy", "R:mfg"]
    io = IOSystem(
        provenance=prov,
        sectors=Classification(name="s", kind="sector", labels=["energy", "mfg"]),
        regions=Classification(name="r", kind="region", labels=["R"]),
        A=pd.DataFrame([[0.1, 0.2], [0.15, 0.1]], index=labels, columns=labels),
        final_demand=pd.DataFrame({"fd": [100.0, 120.0]}, index=labels),
        unit="MEUR",
        currency="EUR",
    )
    sat = SatelliteAccount(
        provenance=prov,
        name="GHG",
        units={"CO2": "t/MEUR"},
        data=pd.DataFrame(
            {"R:energy": [intensity_energy], "R:mfg": [intensity_mfg]}, index=["CO2"]
        ),
    )
    return io, sat


def _run_eur(shock, **sat_kwargs):
    io, sat = _eur_io_sat(**sat_kwargs)
    return registry.get("cge_static").run(
        data={"IOSystem": io, "SatelliteAccount": sat}, shocks=[shock], years=[2020]
    )


def test_carbon_wedge_is_not_a_million_times_too_large():
    """Review P0: the €100/t carbon cost must be finite (the recycling fixed point does not
    diverge), not ~1e6 too large. A realistic-intensity EUR build solves and moves output."""
    res = _run_eur(CarbonPrice(price=100.0))  # would raise 'k ≥ 1' before the 1e-6 fix
    vol = res.data[res.data["variable"] == "volume_change"]
    energy = vol[vol["sector"] == "energy"]["value"].iloc[0]
    assert -0.5 < energy < 0.0  # dirty sector contracts by a plausible amount, not a blowup


def test_gas_and_coverage_change_the_result():
    """Review P1: the CGE must honour gases and coverage (it reuses Engine 1's cost vector)."""
    v_co2 = _run_eur(CarbonPrice(price=100.0, gases=["CO2"])).data
    v_cov = _run_eur(CarbonPrice(price=100.0, gases=["CO2"], coverage_sectors=["energy"])).data

    def energy_vol(d):
        return d[(d["variable"] == "volume_change") & (d["sector"] == "energy")]["value"].iloc[0]

    assert not np.isclose(energy_vol(v_co2), energy_vol(v_cov))  # coverage bites


def test_emissions_change_moves_the_manifest():
    """Review P1: doubling emission intensity changes results AND the substantive manifest."""
    m1 = _run_eur(CarbonPrice(price=100.0), intensity_energy=800.0).manifest
    m2 = _run_eur(CarbonPrice(price=100.0), intensity_energy=1600.0).manifest
    h1 = next(
        i["content_hash"] for i in m1.assumptions["inputs"] if i["name"] == "SatelliteAccount"
    )
    h2 = next(
        i["content_hash"] for i in m2.assumptions["inputs"] if i["name"] == "SatelliteAccount"
    )
    assert h1 != h2  # a changed satellite moves the manifest


def test_non_eur_unit_metadata_rejected():
    """Review P2: emissions in the wrong unit (kg vs t) are rejected before pricing."""
    io, sat = _eur_io_sat()
    sat.units = {"CO2": "kg/MEUR"}  # 1000× off
    with pytest.raises(ValueError, match="expected 't/MEUR'"):
        registry.get("cge_static").run(
            data={"IOSystem": io, "SatelliteAccount": sat},
            shocks=[CarbonPrice(price=100.0)],
            years=[2020],
        )


def test_supplied_unbalanced_sam_rejected():
    """Review P1: a directly-supplied SAM must pass the balance/finiteness/non-negativity gate."""
    sam = toy_sam()
    sam.matrix.loc["BRD", "HOH"] += 50.0  # break the balance by 50 units
    with pytest.raises(ValueError, match="not balanced"):
        registry.get("cge_static").run(
            data={"SAM": sam}, shocks=[CarbonPrice(price=0.1)], years=[2020]
        )


def test_welfare_is_cobb_douglas_utility():
    """Review P1: welfare_change is CD utility U=Π FD_i^γ_i, not the Σ FD sum. Under a carbon
    price with recycling the CD utility falls (the distortion), even if Σ FD is ~unchanged."""
    cal = _cal()
    _b, base = _solve(cal)
    _s, st = _solve(cal, carbon_cost=0.15 * _EMISSIONS, recycling="lump_sum")
    u = float(np.prod(np.power(st.FD, cal.gamma)))
    u_base = float(np.prod(np.power(base.FD, cal.gamma)))
    cd_welfare = u / u_base - 1.0
    sum_fd = st.FD.sum() / base.FD.sum() - 1.0
    assert cd_welfare < -1e-6  # the true CD welfare loss is strictly negative
    assert cd_welfare < sum_fd  # and more negative than the (misleading) sum-of-quantities


# -- second-round review: shock controls, cost-share, axis alignment ----------
def test_io_cge_rejects_unknown_coverage_region():
    """Review P1: the IO-backed CGE must reject a nonexistent coverage region (Engine 1's check),
    not silently produce a zero-impact scenario."""
    io, sat = _eur_io_sat()
    with pytest.raises(ValueError, match="coverage_regions not in the build"):
        registry.get("cge_static").run(
            data={"IOSystem": io, "SatelliteAccount": sat},
            shocks=[CarbonPrice(price=100.0, coverage_regions=["NO_SUCH"])],
            years=[2020],
        )


def test_supplied_sam_cge_rejects_gas_and_coverage():
    """Review P1: the supplied-SAM path cannot honour gas/coverage, so it rejects them."""
    data = {"SAM": toy_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}}
    with pytest.raises(ValueError, match="cannot select gases"):
        registry.get("cge_static").run(
            data=data, shocks=[CarbonPrice(price=0.1, gases=["CH4"])], years=[2020]
        )
    with pytest.raises(ValueError, match="cannot apply sector/region coverage"):
        registry.get("cge_static").run(
            data=data, shocks=[CarbonPrice(price=0.1, coverage_sectors=["BRD"])], years=[2020]
        )


def test_negative_carbon_cost_share_rejected():
    """Review P1: a negative carbon_cost_share is a subsidy, not a price — rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        registry.get("cge_static").run(
            data={"SAM": toy_sam(), "carbon_cost_share": {"BRD": -0.1, "MIL": 0.5}},
            shocks=[CarbonPrice(price=0.1)],
            years=[2020],
        )


def test_carbon_cost_share_unknown_key_rejected():
    with pytest.raises(ValueError, match="keys not in the SAM sectors"):
        registry.get("cge_static").run(
            data={"SAM": toy_sam(), "carbon_cost_share": {"ZZZ": 0.5}},
            shocks=[CarbonPrice(price=0.1)],
            years=[2020],
        )


def test_supplied_sam_renamed_axis_gives_clean_error():
    """Review P2: a balanced matrix with axes not matching the accounts raises a clean ValueError,
    not a raw KeyError during calibration."""
    sam = toy_sam()
    m = sam.matrix.copy()
    idx = list(m.index)
    idx[0] = "XXX"  # rename BRD on both axes; accounts list still says BRD
    m.index = idx
    m.columns = idx
    sam.matrix = m
    with pytest.raises(ValueError, match="axes must equal the declared accounts"):
        registry.get("cge_static").run(
            data={"SAM": sam, "sectors": ["BRD", "MIL"]},
            shocks=[CarbonPrice(price=0.1)],
            years=[2020],
        )


def test_solver_residual_norm_recorded():
    """Review P2: the manifest records the max solver residual norm (convergence evidence)."""
    res = registry.get("cge_static").run(
        data={"SAM": toy_sam()}, shocks=[CarbonPrice(price=0.0)], years=[2020]
    )
    rn = res.manifest.assumptions["solver_max_residual_norm"]
    assert isinstance(rn, float) and rn == rn and rn < 1e-6  # finite (not NaN), converged


# -- third-round review: missing emissions, extra accounts, numeraire, IO alignment ----------
def test_positive_carbon_price_requires_emissions_input():
    """Review P1: a positive carbon price with no emissions input must be rejected, not silently
    zero-impact."""
    io, _sat = _eur_io_sat()
    with pytest.raises(ValueError, match="requires a 'SatelliteAccount'"):
        registry.get("cge_static").run(
            data={"IOSystem": io}, shocks=[CarbonPrice(price=100.0)], years=[2020]
        )
    with pytest.raises(ValueError, match="requires a\n?.*carbon_cost_share"):
        registry.get("cge_static").run(
            data={"SAM": toy_sam()}, shocks=[CarbonPrice(price=100.0)], years=[2020]
        )


def test_zero_carbon_price_baseline_needs_no_emissions_input():
    """A genuine zero-price baseline still runs without an emissions input (replication)."""
    res = registry.get("cge_static").run(
        data={"SAM": toy_sam()}, shocks=[CarbonPrice(price=0.0)], years=[2020]
    )
    assert res.data["value"].abs().max() < 1e-8


def test_cge_reports_no_deflator():
    """Review P1: the CPI is the numéraire, so there is no spurious (AM-GM-bound) deflator."""
    res = registry.get("cge_static").run(
        data={"SAM": toy_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}},
        shocks=[CarbonPrice(price=0.15)],
        years=[2020],
    )
    assert not (res.data["variable"] == "deflator").any()
    assert (res.data["variable"] == "gdp_change_real").any()


def test_supplied_sam_extra_account_rejected():
    """Review P2: an extra balanced account (not in a required role) is rejected, not silently
    ignored while the manifest claims the SAM is aligned."""
    sam = toy_sam()
    m = sam.matrix.copy()
    m["EXTRA"] = 0.0
    m.loc["EXTRA"] = 0.0  # extra all-zero (balanced) account
    sam.matrix = m
    with pytest.raises(ValueError, match="must equal the declared accounts"):
        registry.get("cge_static").run(
            data={"SAM": sam, "sectors": ["BRD", "MIL"]},
            shocks=[CarbonPrice(price=0.1)],
            years=[2020],
        )


# -- CES value-added nest (non-unitary factor substitution) -------------------
def _asymmetric_sam():
    """A SAM where BRD is capital-intensive and MIL labour-intensive, so a change in the relative
    factor price exercises the CES value-added nest."""
    import pandas as pd

    from cge.contracts.data_objects import SAM, Provenance

    acc = ["BRD", "MIL", "CAP", "LAB", "HOH"]
    m = pd.DataFrame(0.0, index=acc, columns=acc)
    m.loc["BRD", "MIL"] = 24.0
    m.loc["MIL", "BRD"] = 15.0
    m.loc["CAP", "BRD"] = 60.0  # BRD capital-heavy
    m.loc["LAB", "BRD"] = 25.0
    m.loc["CAP", "MIL"] = 30.0  # MIL labour-heavy
    m.loc["LAB", "MIL"] = 66.0
    m.loc["BRD", "HOH"] = 76.0
    m.loc["MIL", "HOH"] = 105.0
    m.loc["HOH", "CAP"] = 90.0
    m.loc["HOH", "LAB"] = 91.0
    prov = Provenance(
        source="t", source_version="1", licence="x", reference_year=0, retrieved="2026-07-19"
    )
    return SAM(provenance=prov, accounts=acc, matrix=m)


def test_ces_va_replicates_for_all_elasticities():
    """The CES value-added nest replicates the benchmark exactly for σ = 1 (CD), 0.5, 2."""
    sam = _asymmetric_sam()
    for sigma in (1.0, 0.5, 2.0):
        cal = calibrate(sam, sectors=_SECTORS, factors=_FACTORS, va_elast=sigma)
        sol = solve(lambda z, c=cal: M.residuals(c, z), M.initial_guess(cal) * 1.05, prefer="scipy")
        st = M.derive_state(cal, sol.x[:2], sol.x[2:])
        assert np.allclose(st.X, cal.X0, atol=1e-6), f"σ={sigma} failed replication"


def test_ces_elasticity_governs_factor_price_swing():
    """With asymmetric factor intensities, a carbon price shifts the relative factor price, and a
    LOWER VA elasticity (harder to substitute) gives a LARGER swing — the CES bites."""
    sam = _asymmetric_sam()
    e = np.array([2.0, 0.5])

    def rel_factor_price(sigma):
        cal = calibrate(sam, sectors=_SECTORS, factors=_FACTORS, va_elast=sigma)
        sol = solve(
            lambda z: M.residuals(cal, z, carbon_cost=0.15 * e, recycling="lump_sum"),
            M.initial_guess(cal),
            prefer="scipy",
        )
        st = M.derive_state(cal, sol.x[:2], sol.x[2:], carbon_cost=0.15 * e, recycling="lump_sum")
        return abs(st.w[0] / st.w[1] - 1.0)  # |wCAP/wLAB − 1|

    swing_low = rel_factor_price(0.3)
    swing_high = rel_factor_price(3.0)
    assert swing_low > swing_high > 0  # lower elasticity → larger relative-price swing


@pytest.mark.parametrize("bad", [-1.0, 0.0, np.array([0.8])])
def test_closed_va_elast_validated(bad):
    """P2: the closed model rejects non-positive and wrong-length VA elasticities (a length-1 vector
    used to raise a raw IndexError; a negative value was silently used)."""
    sam = _asymmetric_sam()
    with pytest.raises(ValueError):
        calibrate(sam, sectors=_SECTORS, factors=_FACTORS, va_elast=bad)


def test_closed_manifest_records_va_elast():
    """P1: va_elast is in the closed run's manifest — two runs differing only in it are
    distinguishable (previously only incidental solver residuals told them apart)."""
    eng = registry.get("cge_static")
    data = {"SAM": _asymmetric_sam(), "carbon_cost_share": {"BRD": 0.1, "MIL": 0.02}}
    a1 = eng.run(data={**data, "va_elast": 0.5}, shocks=[CarbonPrice(price=1.0)], years=[2020])
    a2 = eng.run(data={**data, "va_elast": 2.0}, shocks=[CarbonPrice(price=1.0)], years=[2020])
    assert a1.manifest.assumptions["va_elast"] == [0.5, 0.5]
    assert a1.manifest.assumptions != a2.manifest.assumptions


def test_closed_manifest_nest_description_reflects_calibrated_nest():
    """P2 (round 2): the manifest describes the nest that was actually calibrated — CES for a
    non-unitary va_elast, Cobb-Douglas for σ=1 — not a hardcoded 'Cobb-Douglas'."""
    eng = registry.get("cge_static")
    sam = _asymmetric_sam()
    ces = eng.run(data={"SAM": sam, "va_elast": 0.5}, shocks=[], years=[2020]).manifest.assumptions
    cd = eng.run(data={"SAM": sam}, shocks=[], years=[2020]).manifest.assumptions
    assert "CES" in ces["value_added_nest"] and "CES" in ces["model"]
    assert "Cobb-Douglas" in cd["value_added_nest"]
    assert "double" not in ces["value_added_nest"].lower()  # no double-dividend overclaim


def test_closed_replication_gate_rejects_unsupported_topology():
    """P1 (round 2): a balanced closed SAM with an offsetting household↔sector loop passes the
    structural validators but does not replicate; the post-calibration gate rejects it."""
    eng = registry.get("cge_static")
    sam = toy_sam()
    sam.matrix.loc["BRD", "HOH"] += 10.0
    sam.matrix.loc["HOH", "BRD"] += 10.0
    assert is_balanced(sam.matrix, tol=1e-9)
    with pytest.raises(ValueError, match="does not replicate"):
        eng.run(data={"SAM": sam}, shocks=[CarbonPrice(price=0.0)], years=[2020])


def test_closed_recycling_solver_survives_benchmark_start_on_k_ridge():
    """P2 (round 2): a closed shocked solve whose benchmark start has k≈1 still converges to the
    valid equilibrium — the smooth clamp + multi-start rescue it instead of the guard aborting."""
    eng = registry.get("cge_static")
    res = eng.run(
        data={"SAM": toy_sam(), "carbon_cost_share": {"BRD": 1.4, "MIL": 0.35}},
        shocks=[CarbonPrice(price=1.0)],
        years=[2020],
    )
    assert res.manifest.assumptions["solver_max_residual_norm"] < 1e-9


def test_open_ces_va_replicates():
    """The open model's CES value-added nest also replicates the benchmark for σ ≠ 1."""
    from cge.data.sam import toy_open_sam
    from cge.engines.cge_static import model_open as MO
    from cge.engines.cge_static.calibrate_open import calibrate_open

    for sigma in (0.5, 2.0):
        cal = calibrate_open(toy_open_sam(), sectors=_SECTORS, factors=_FACTORS, va_elast=sigma)
        sol = solve(
            lambda z, c=cal: MO.residuals(c, z, recycling="lump_sum"),
            MO.initial_guess(cal) * 1.03,
            prefer="scipy",
        )
        st = MO.derive_open_state(
            cal, sol.x[:2], sol.x[2:4], sol.x[4:6], float(sol.x[6]), recycling="lump_sum"
        )
        assert np.allclose(st.Z, cal.Z0, atol=1e-6), f"open σ={sigma} failed replication"


# -- savings-investment account (Phase 5d.2) ----------------------------------
def _inv_sam():
    """The toy SAM plus a SAVINV account: the household saves 27.15 (15% of income 181), which
    finances investment purchases of BRD (20) and MIL (7.15). Household demand shrinks by exactly
    what SAVINV buys, so sector totals and the balance are untouched. inv_gamma is deliberately
    BRD-heavy (capital-goods-like), different from the household's gamma."""
    import pandas as pd

    sam = toy_sam()
    acc = list(sam.accounts) + ["SAVINV"]
    m = pd.DataFrame(0.0, index=acc, columns=acc)
    m.loc[sam.accounts, sam.accounts] = sam.matrix
    m.loc["SAVINV", "HOH"] = 27.15
    m.loc["BRD", "SAVINV"] = 20.0
    m.loc["MIL", "SAVINV"] = 7.15
    m.loc["BRD", "HOH"] -= 20.0  # 76 → 56
    m.loc["MIL", "HOH"] -= 7.15  # 105 → 97.85
    return sam.model_copy(update={"accounts": acc, "matrix": m})


def _gov_inv_sam():
    """Toy SAM + GOV + SAVINV together: an 18.1 tax funds government purchases (BRD 10, MIL 8.1);
    the household then saves 16.29 (10% of its 162.9 disposable income), financing investment
    (BRD 9, MIL 7.29). Both institutional accounts balance; the SAM stays balanced."""
    import pandas as pd

    sam = toy_sam()
    acc = list(sam.accounts) + ["GOV", "SAVINV"]
    m = pd.DataFrame(0.0, index=acc, columns=acc)
    m.loc[sam.accounts, sam.accounts] = sam.matrix
    m.loc["GOV", "HOH"] = 18.1
    m.loc["BRD", "GOV"] = 10.0
    m.loc["MIL", "GOV"] = 8.1
    m.loc["SAVINV", "HOH"] = 16.29
    m.loc["BRD", "SAVINV"] = 9.0
    m.loc["MIL", "SAVINV"] = 7.29
    m.loc["BRD", "HOH"] -= 19.0  # 76 → 57
    m.loc["MIL", "HOH"] -= 15.39  # 105 → 89.61
    return sam.model_copy(update={"accounts": acc, "matrix": m})


def _cal_inv():
    return calibrate(
        _inv_sam(),
        sectors=_SECTORS,
        factors=_FACTORS,
        institutions={"household": "HOH", "savings_investment": "SAVINV"},
    )


def _cal_gov_inv():
    return calibrate(
        _gov_inv_sam(),
        sectors=_SECTORS,
        factors=_FACTORS,
        institutions={
            "household": "HOH",
            "government": "GOV",
            "savings_investment": "SAVINV",
        },
    )


def _solve_state(cal, cc=None, recycling="lump_sum", inv_closure="savings_driven", drop_factor=0):
    ns = len(cal.sectors)
    cc = np.zeros(ns) if cc is None else cc
    sol = solve(
        lambda z: M.residuals(
            cal,
            z,
            carbon_cost=cc,
            recycling=recycling,
            inv_closure=inv_closure,
            drop_factor=drop_factor,
        ),
        M.initial_guess(cal),
        prefer="scipy",
    )
    st = M.derive_state(
        cal,
        sol.x[:ns],
        sol.x[ns:],
        carbon_cost=cc,
        recycling=recycling,
        strict=True,
        inv_closure=inv_closure,
    )
    return sol, st


def test_savinv_calibrates():
    cal = _cal_inv()
    assert cal.has_investment and not cal.has_government
    assert cal.sav_rate0 == pytest.approx(0.15)  # 27.15 / 181 disposable
    assert np.allclose(cal.inv_gamma, [20.0 / 27.15, 7.15 / 27.15])
    assert not np.allclose(cal.inv_gamma, cal.gamma)  # composition genuinely differs


def test_savinv_with_government_calibrates():
    cal = _cal_gov_inv()
    assert cal.has_investment and cal.has_government
    assert cal.gov_tax_rate0 == pytest.approx(0.1)
    assert cal.sav_rate0 == pytest.approx(0.1)  # 16.29 / 162.9 disposable (net of tax)


def test_savinv_benchmark_replicates_under_both_closures():
    """Tier 1: the benchmark reproduces the SAM exactly — prices 1, FD0 and INV0 — under BOTH
    investment closures (they coincide at the benchmark by construction)."""
    for closure in ("savings_driven", "fixed_real"):
        for cal in (_cal_inv(), _cal_gov_inv()):
            sol, st = _solve_state(cal, recycling="none", inv_closure=closure)
            assert np.allclose(sol.x, 1.0, atol=1e-8), closure
            assert np.allclose(st.FD, cal.FD0, atol=1e-6), closure
            assert np.allclose(st.ID, cal.INV0, atol=1e-6), closure
            assert np.allclose(st.X, cal.X0, atol=1e-6), closure


def test_savings_investment_identity_under_shock():
    """Tier 2 (the 5d.2 identity): under savings_driven, nominal investment == household savings
    to 1e-9, at a shocked equilibrium, with and without a government."""
    cc = 0.15 * _EMISSIONS
    for cal in (_cal_inv(), _cal_gov_inv()):
        _sol, st = _solve_state(cal, cc=cc)
        assert float(np.dot(st.p, st.ID)) == pytest.approx(st.savings, rel=1e-9)
        assert st.savings == pytest.approx(cal.sav_rate0 * st.income, rel=1e-9)


def test_fixed_real_closure_holds_investment_quantity():
    """Under fixed_real the investment QUANTITY vector stays at INV0 under a shock; household
    savings adjust residually (= p·INV0, which moves with prices). Under savings_driven the
    quantities move — the closures genuinely differ post-shock."""
    cal = _cal_inv()
    cc = 0.15 * _EMISSIONS
    _s1, fixed = _solve_state(cal, cc=cc, inv_closure="fixed_real")
    _s2, driven = _solve_state(cal, cc=cc, inv_closure="savings_driven")
    assert np.allclose(fixed.ID, cal.INV0, atol=1e-9)  # quantities pinned
    assert fixed.savings == pytest.approx(float(np.dot(fixed.p, fixed.ID)), rel=1e-9)
    assert not np.allclose(driven.ID, cal.INV0, atol=1e-6)  # savings_driven moved


def test_savinv_homogeneity_savings_driven():
    """Tier 1: the savings rate is a RATE, so scaling the endowment k× scales investment demand
    by k with prices unchanged (savings_driven; fixed_real is intentionally NOT homogeneous in
    the endowment alone — its INV0 quantity anchor stays fixed by design)."""
    cal = _cal_inv()
    _sol, st = _solve_state(cal, recycling="none")
    k = 2.5
    cal_k = replace(cal, endowment=cal.endowment * k, X0=cal.X0 * k, F0=cal.F0 * k, Z0=cal.Z0 * k)
    _sol_k, st_k = _solve_state(cal_k, recycling="none")
    assert np.allclose(_sol.x, _sol_k.x, atol=1e-8)
    assert np.allclose(st_k.ID, k * st.ID, atol=1e-8)


def test_savinv_walras_under_shock():
    """Tier 1 re-proof: the dropped factor market still clears with investment (and government)
    active under a carbon shock, under both closures."""
    cc = 0.15 * _EMISSIONS
    for closure in ("savings_driven", "fixed_real"):
        for cal in (_cal_inv(), _cal_gov_inv()):
            _sol, st = _solve_state(cal, cc=cc, inv_closure=closure, drop_factor=0)
            excess = float(st.F[0, :].sum()) - cal.endowment[0]
            assert abs(excess) < 1e-6, closure


def test_savinv_unbalanced_rejected():
    import pandas as pd

    sam = toy_sam()
    acc = list(sam.accounts) + ["SAVINV"]
    m = pd.DataFrame(0.0, index=acc, columns=acc)
    m.loc[sam.accounts, sam.accounts] = sam.matrix
    m.loc["SAVINV", "HOH"] = 10.0
    m.loc["BRD", "SAVINV"] = 7.0  # spends less than it collects
    bad = sam.model_copy(update={"accounts": acc, "matrix": m})
    with pytest.raises(ValueError, match="savings-investment account is unbalanced"):
        calibrate(
            bad,
            sectors=_SECTORS,
            factors=_FACTORS,
            institutions={"household": "HOH", "savings_investment": "SAVINV"},
        )


def test_unsupported_inv_closure_rejected():
    cal = _cal_inv()
    with pytest.raises(ValueError, match="unsupported inv_closure"):
        M.derive_state(
            cal,
            np.ones(len(cal.sectors)),
            np.ones(len(cal.factors)),
            inv_closure="golden_rule",
        )


def test_engine_savinv_emits_investment_and_manifest():
    """End-to-end: a SAM carrying SAVINV emits investment + savings (equal under savings_driven),
    records the closure + savings rate in the manifest, and a no-SAVINV run emits neither."""
    eng = registry.get("cge_static")
    data = {"SAM": _inv_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}}
    res = eng.run(data=data, shocks=[CarbonPrice(price=0.15)], years=[2020])
    d = res.data
    inv = d[d["variable"] == "investment"]["value"]
    sav = d[d["variable"] == "savings"]["value"]
    assert len(inv) == 1 and len(sav) == 1
    assert float(inv.iloc[0]) == pytest.approx(float(sav.iloc[0]), rel=1e-9)  # S = I visible
    assert res.manifest.assumptions["savings_investment_account"] == "SAVINV"
    assert res.manifest.assumptions["inv_closure"] == "savings_driven"
    assert res.manifest.assumptions["benchmark_savings_rate_of_disposable_income"] == pytest.approx(
        0.15
    )

    plain = eng.run(
        data={"SAM": toy_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}},
        shocks=[CarbonPrice(price=0.15)],
        years=[2020],
    )
    assert not (plain.data["variable"] == "investment").any()
    assert plain.manifest.assumptions["savings_investment_account"] == "none"


def test_engine_inv_closure_switch_changes_results():
    """The two closures are switchable by config and genuinely differ under a shock (Tier 3:
    closures switchable, each solvable). fixed_real without a SAVINV account is rejected."""
    eng = registry.get("cge_static")
    data = {"SAM": _inv_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}}
    driven = eng.run(data=data, shocks=[CarbonPrice(price=0.15)], years=[2020])
    fixed = eng.run(
        data={**data, "inv_closure": "fixed_real"},
        shocks=[CarbonPrice(price=0.15)],
        years=[2020],
    )
    assert fixed.manifest.assumptions["inv_closure"] == "fixed_real"
    inv_d = float(driven.data[driven.data["variable"] == "investment"]["value"].iloc[0])
    inv_f = float(fixed.data[fixed.data["variable"] == "investment"]["value"].iloc[0])
    assert inv_d != pytest.approx(inv_f, rel=1e-6)  # the closure genuinely matters

    with pytest.raises(ValueError, match="needs a 'SAVINV' account"):
        eng.run(
            data={
                "SAM": toy_sam(),
                "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5},
                "inv_closure": "fixed_real",
            },
            shocks=[CarbonPrice(price=0.15)],
            years=[2020],
        )


# -- labour market: wage-floor closure (Phase 5d.4) ---------------------------
def _floor_solve(cal, cc, labour_floor, recycling="lump_sum", drop_factor=0):
    ns = len(cal.sectors)
    sol = solve(
        lambda z: M.residuals(
            cal,
            z,
            carbon_cost=cc,
            recycling=recycling,
            labour_floor=labour_floor,
            drop_factor=drop_factor,
        ),
        M.initial_guess(cal),
        prefer="scipy",
    )
    st = M.derive_state(
        cal, sol.x[:ns], sol.x[ns:], carbon_cost=cc, recycling=recycling, labour_floor=labour_floor
    )
    return sol, st


def test_no_floor_is_full_employment_and_pre_5d4_identical():
    """Default closure: unemployment is exactly 0, and derive_state without a floor is unchanged."""
    cal = _cal()
    _sol, st = _solve(cal, carbon_cost=0.3 * _EMISSIONS)
    assert st.unemployment == pytest.approx(0.0, abs=1e-12)


def test_floor_primitive_pins_wage_unconditionally():
    """The model-level ``labour_floor`` is a PRIMITIVE that always pins w[LAB] = floor; deciding
    WHETHER to impose it (only when it genuinely binds) is the engine's regime-switch, not this
    residual's job. Here we confirm the primitive does what it says: the wage lands on the floor."""
    cal = _cal()
    cc = 0.3 * _EMISSIONS
    lab = cal.factors.index("LAB")
    _s0, st0 = _solve(cal, carbon_cost=cc)
    floor = 0.5 * (st0.w[lab] + 1.0)  # a genuinely binding floor (above the eq wage)
    _s1, st1 = _floor_solve(cal, cc, labour_floor=floor)
    assert st1.w[lab] == pytest.approx(floor, abs=1e-8)


def test_binding_floor_pins_wage_and_produces_unemployment():
    """A floor ABOVE the market-clearing wage binds: the LAB wage sits exactly at the floor,
    labour demand falls short of supply, and the shortfall is reported as unemployment."""
    cal = _cal()
    cc = 0.3 * _EMISSIONS
    lab = cal.factors.index("LAB")
    _s0, st0 = _solve(cal, carbon_cost=cc)
    w_eq = st0.w[lab]
    floor = 0.5 * (w_eq + 1.0)  # between the depressed eq wage and the benchmark 1.0
    _s1, st1 = _floor_solve(cal, cc, labour_floor=floor)
    assert st1.w[lab] == pytest.approx(floor, abs=1e-8)  # wage pinned at the floor
    employed = float(st1.F[lab, :].sum())
    assert employed < cal.endowment[lab]  # demand short of supply
    assert st1.unemployment == pytest.approx(cal.endowment[lab] - employed, abs=1e-9)
    assert st1.unemployment > 0.0


def test_walras_holds_under_binding_floor():
    """THE Tier-1 re-proof (plan §5d.4): when the labour market clears on the pinned wage (not on
    quantity), the OTHER factor market (capital) must still clear exactly — the regime switch
    must not open a hidden imbalance. Drop the capital row by Walras and confirm it holds."""
    cal = _cal()
    cc = 0.3 * _EMISSIONS
    lab = cal.factors.index("LAB")
    cap = cal.factors.index("CAP")
    _s0, st0 = _solve(cal, carbon_cost=cc)
    floor = 0.5 * (st0.w[lab] + 1.0)
    # Drop the CAP market (drop_factor=cap); the floor replaces the LAB clearing row. If the
    # system is genuinely square and consistent, CAP still clears at the solution.
    _s1, st1 = _floor_solve(cal, cc, labour_floor=floor, drop_factor=cap)
    cap_excess = float(st1.F[cap, :].sum()) - cal.endowment[cap]
    assert abs(cap_excess) < 1e-7


def test_binding_floor_lowers_welfare_vs_full_employment():
    """Economic-sense: forcing a wage floor that creates unemployment makes the household worse
    off than the full-employment outcome (it earns less labour income)."""
    cal = _cal()
    cc = 0.3 * _EMISSIONS
    lab = cal.factors.index("LAB")
    _s0, st0 = _solve(cal, carbon_cost=cc)
    floor = 0.5 * (st0.w[lab] + 1.0)
    _s1, st1 = _floor_solve(cal, cc, labour_floor=floor)

    def cd_u(state):
        return float(np.prod(np.power(state.FD, cal.gamma)))

    assert cd_u(st1) < cd_u(st0)  # unemployment lowers household utility


def test_higher_floor_more_unemployment():
    """Monotonicity: a higher wage floor binds harder — more unemployment."""
    cal = _cal()
    cc = 0.3 * _EMISSIONS
    lab = cal.factors.index("LAB")
    _s0, st0 = _solve(cal, carbon_cost=cc)
    w_eq = st0.w[lab]
    _lo, st_lo = _floor_solve(cal, cc, labour_floor=w_eq + 0.15 * (1.0 - w_eq))
    _hi, st_hi = _floor_solve(cal, cc, labour_floor=w_eq + 0.5 * (1.0 - w_eq))
    assert st_hi.unemployment > st_lo.unemployment > 0.0


def test_labour_income_fixed_point_makes_household_poorer():
    """The employed-labour income fixed point: under a binding floor the household's factor income
    counts only EMPLOYED labour, so it is strictly below the full-employment factor income at the
    same prices (this is the mechanism that couples unemployment back into demand)."""
    cal = _cal()
    cc = 0.3 * _EMISSIONS
    lab = cal.factors.index("LAB")
    _s0, st0 = _solve(cal, carbon_cost=cc)
    floor = 0.5 * (st0.w[lab] + 1.0)
    _s1, st1 = _floor_solve(cal, cc, labour_floor=floor)
    # factor_income under the floor = w·(capital endowment) + w_L·(employed labour) — less than
    # w·(full endowment) would give at the same wages.
    full_emp_income = float(np.dot(st1.w, cal.endowment))
    assert st1.factor_income < full_emp_income


# -- engine wiring for the wage floor (Phase 5d.4) ----------------------------
def test_engine_labour_floor_rejected_at_or_above_benchmark_wage():
    """A floor ≥ the benchmark wage (1.0) is nonsensical (it would bind at the full-employment
    benchmark); rejected up front, not at the replication gate."""
    eng = registry.get("cge_static")
    with pytest.raises(ValueError, match="below the benchmark wage"):
        eng.run(
            data={
                "SAM": toy_sam(),
                "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5},
                "labour_floor": 1.0,
            },
            shocks=[CarbonPrice(price=0.15)],
            years=[2020],
        )


def test_engine_binding_floor_emits_unemployment_and_manifest():
    """End-to-end: a floor that binds post-shock emits an `unemployment` rate and records the
    wage-floor closure in the manifest; a no-floor run emits neither and reports full employment."""
    eng = registry.get("cge_static")
    data = {"SAM": toy_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}, "labour_floor": 0.75}
    res = eng.run(data=data, shocks=[CarbonPrice(price=0.3)], years=[2020])
    u = res.data[res.data["variable"] == "unemployment"]
    assert len(u) == 1 and float(u["value"].iloc[0]) > 0.0
    assert res.manifest.assumptions["labour_closure"] == "wage_floor"
    assert res.manifest.assumptions["labour_floor"] == pytest.approx(0.75)
    assert res.manifest.assumptions["labour_floor_bound"] is True
    lab_ch = float(
        res.data[(res.data["variable"] == "factor_price_change") & (res.data["sector"] == "LAB")][
            "value"
        ].iloc[0]
    )
    assert (1.0 + lab_ch) == pytest.approx(0.75, abs=1e-6)  # wage pinned at the floor

    plain = eng.run(
        data={"SAM": toy_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}},
        shocks=[CarbonPrice(price=0.3)],
        years=[2020],
    )
    assert not (plain.data["variable"] == "unemployment").any()
    assert plain.manifest.assumptions["labour_closure"] == "flexible_wage_full_employment"


def test_engine_slack_floor_is_byte_identical_to_no_floor():
    """A configured-but-slack floor (below the post-shock wage) leaves results byte-identical to
    the full-employment run — only the manifest records that a floor was configured."""
    eng = registry.get("cge_static")
    shk = [CarbonPrice(price=0.3)]
    plain = eng.run(
        data={"SAM": toy_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}},
        shocks=shk,
        years=[2020],
    )
    slack = eng.run(
        data={"SAM": toy_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}, "labour_floor": 0.4},
        shocks=shk,
        years=[2020],
    )
    merged = plain.data.merge(
        slack.data, on=["variable", "sector", "region", "year", "scenario"], suffixes=("_p", "_s")
    )
    assert len(merged) == len(plain.data) == len(slack.data)
    assert np.allclose(merged["value_p"], merged["value_s"], atol=1e-12)
    assert slack.manifest.assumptions["labour_floor_bound"] is False
