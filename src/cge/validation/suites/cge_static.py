"""Validation suite for Engine 3 (static CGE pilot) — the standard CGE correctness battery.

These are the non-negotiable tests every CGE must pass (docs/phase-5-plan.md §7, Tier 1) plus the
economic-sense and cross-engine checks (Tier 2). They run on the hand-checkable 2-sector toy SAM
via the scipy solver fallback, so they pass in CI with no IPOPT binary.

- **Benchmark replication** — zero shock ⇒ the calibrated model reproduces the SAM (all changes 0).
- **Homogeneity** — scaling nominal size (endowments) leaves prices unchanged, reals scale.
- **Walras' law** — the dropped market clears residually at the solution.
- **Carbon-price direction** — the dirty sector's output falls; real GDP falls.
- **Cross-engine sign** — CGE carbon-price volume changes are same-sign as the intuitive fall.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from cge.data.sam import toy_sam
from cge.engines.cge_static import model as M
from cge.engines.cge_static.calibrate import calibrate
from cge.engines.cge_static.solver import solve
from cge.validation.framework import check

SUITE = "cge_static"

_SECTORS = ["BRD", "MIL"]
_FACTORS = ["CAP", "LAB"]
# Emission intensity per unit output for the carbon-price checks (BRD is the dirty sector).
_EMISSIONS = np.array([2.0, 0.5])


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


@check(SUITE, "benchmark_replication")
def _replication():
    """THE CGE correctness test: with zero shock the calibrated model returns the benchmark SAM
    to machine precision (prices = 1; X, FD, F = benchmark)."""
    cal = _cal()
    _sol, st = _solve(cal)
    err = max(
        float(np.max(np.abs(st.p - 1.0))),
        float(np.max(np.abs(st.X - cal.X0))),
        float(np.max(np.abs(st.F - cal.F0))),
    )
    return err < 1e-6, f"max|benchmark − replicated| = {err:.2e}", err, 1e-6


@check(SUITE, "homogeneity_degree_zero")
def _homogeneity():
    """Scaling nominal size (all endowments ×k) leaves prices unchanged and scales real
    quantities by k — the model has no money illusion."""
    cal = _cal()
    sol, _ = _solve(cal)
    k = 3.0
    cal_k = replace(
        cal,
        endowment=cal.endowment * k,
        X0=cal.X0 * k,
        F0=cal.F0 * k,
        Z0=cal.Z0 * k,
        FD0=cal.FD0 * k,
    )
    sol_k, st_k = _solve(cal_k)
    _, st = _solve(cal)
    price_err = float(np.max(np.abs(sol.x - sol_k.x)))
    real_err = float(np.max(np.abs(st_k.X - k * st.X)))
    err = max(price_err, real_err)
    return err < 1e-6, f"×{k}: max(price drift, real-scale error) = {err:.2e}", err, 1e-6


@check(SUITE, "walras_law")
def _walras():
    """Dropping one factor market (CAP) by Walras' law, that market still clears at the solution —
    confirming the square-model count and the accounting closure."""
    cal = _cal()
    _sol, st = _solve(cal, drop_factor=0)
    excess = float(st.F[0, :].sum()) - cal.endowment[0]
    return (
        abs(excess) < 1e-6,
        f"dropped-market (CAP) excess demand = {excess:.2e}",
        abs(excess),
        1e-6,
    )


@check(SUITE, "walras_holds_under_carbon_price_with_recycling")
def _walras_recycled():
    """Under a carbon price WITH revenue recycling, the dropped factor market still clears — the
    revenue circulates so the closed economy remains balanced (a pure-loss `none` would not)."""
    cal = _cal()
    _sol, st = _solve(cal, carbon_cost=0.15 * _EMISSIONS, drop_factor=0, recycling="lump_sum")
    excess = float(st.F[0, :].sum()) - cal.endowment[0]
    return (
        abs(excess) < 1e-6,
        f"dropped-market excess under recycled carbon price = {excess:.2e}",
        abs(excess),
        1e-6,
    )


def _cd_utility(cal, state):
    """Cobb-Douglas household utility U = Π FD_i^{γ_i} — the correct welfare measure for the CD
    household (the emitted ``welfare_change``); Σ FD (quantities) is NOT utility (review P1)."""
    return float(np.prod(np.power(state.FD, cal.gamma)))


@check(SUITE, "recycled_carbon_price_welfare_is_small_and_negative")
def _recycling_effect():
    """Validate the **Cobb-Douglas welfare** the engine emits (not a Σ-FD sum): under a carbon
    price WITH lump-sum recycling, CD utility falls only slightly — the revenue is returned, so the
    remaining loss is just the relative-price distortion. (No comparison to the non-closing `none`
    model, which violates Walras and is not a valid equilibrium counterfactual — review P1.)"""
    cal = _cal()
    _b, base = _solve(cal)
    _r, st = _solve(cal, carbon_cost=0.15 * _EMISSIONS, recycling="lump_sum")
    welfare = _cd_utility(cal, st) / _cd_utility(cal, base) - 1.0
    revenue = st.carbon_revenue
    # Recycled: a small NEGATIVE CD-welfare change (the distortion), and revenue is collected.
    ok = revenue > 0 and -0.05 < welfare < 0.0
    return (
        ok,
        f"carbon revenue={revenue:.4f}, recycled CD welfare change={welfare:+.5f}",
        None,
        None,
    )


@check(SUITE, "recycling_improves_welfare_over_no_recycling")
def _recycling_beats_none():
    """A *valid* recycling comparison at fixed prices: at the recycled equilibrium prices, the
    household's CD utility is higher WITH the revenue transfer than WITHOUT it (income is strictly
    larger by the transfer). This isolates the recycling benefit without invoking the non-closing
    `none` equilibrium."""
    cal = _cal()
    _r, st = _solve(cal, carbon_cost=0.15 * _EMISSIONS, recycling="lump_sum")
    # Same prices, but strip the recycled revenue from income → lower demand, lower utility.
    factor_income = float(np.dot(st.w, cal.endowment))
    fd_no_transfer = cal.gamma * factor_income / st.p
    u_with = float(np.prod(np.power(st.FD, cal.gamma)))
    u_without = float(np.prod(np.power(fd_no_transfer, cal.gamma)))
    ok = u_with > u_without
    return ok, f"CD utility with transfer {u_with:.5f} > without {u_without:.5f} = {ok}", None, None


@check(SUITE, "carbon_price_reallocates_dirty_to_clean")
def _carbon_direction():
    """With revenue recycling, a carbon price **reallocates** output from the dirty sector to the
    clean one (rather than simply shrinking the economy) — the GE substitution signal. The dirty
    sector's output falls and the clean sector's rises."""
    cal = _cal()
    _b, base = _solve(cal)
    _s, st = _solve(cal, carbon_cost=0.15 * _EMISSIONS)
    dirty_falls = st.X[0] < base.X[0] - 1e-9  # BRD, the dirty sector
    clean_rises = st.X[1] > base.X[1] + 1e-9  # MIL, the clean sector
    ok = dirty_falls and clean_rises
    return ok, f"dirty output falls={dirty_falls}, clean output rises={clean_rises}", None, None


@check(SUITE, "carbon_price_raises_dirty_relative_price")
def _relative_price():
    """The dirty good's price rises relative to the clean good's under a carbon price (the
    substitution signal), confirmed against the CPI-numéraire equilibrium."""
    cal = _cal()
    _b, base = _solve(cal)
    _s, st = _solve(cal, carbon_cost=0.15 * _EMISSIONS)
    rel_base = base.p[0] / base.p[1]
    rel_shock = st.p[0] / st.p[1]
    return (
        rel_shock > rel_base + 1e-9,
        f"p_dirty/p_clean {rel_base:.4f} → {rel_shock:.4f} (should rise)",
        None,
        None,
    )


@check(SUITE, "replicates_on_built_sam")
def _real_sam_replication():
    """The 5.1b gate: build a SAM from an EXIOBASE-shaped build (the offline pymrio **test** MRIO,
    not live EXIOBASE — see the honest-status note), quality-gate it, and confirm the CGE
    calibrates and replicates its benchmark to machine precision — proving the SAM→calibrate→solve
    pipeline works on structured multi-region data, not only the hand-built toy."""
    import tempfile

    from cge.data.build import build_test
    from cge.data.sam import build_sam
    from cge.data.store import DataStore

    store = DataStore(tempfile.mkdtemp())
    build_test(store=store)  # offline pymrio test MRIO (NOT live EXIOBASE)
    bid = next(b for b in store.build_ids() if b != "exiobase-test")
    io = store.load(bid)["IOSystem"]
    sam, report, sectors = build_sam(io)
    if not report.passed:
        return False, "SAM quality gate failed on the built SAM", None, None
    cal = calibrate(sam, sectors=sectors, factors=["CAP", "LAB"])
    sol = solve(lambda z: M.residuals(cal, z), M.initial_guess(cal) * 1.05, prefer="scipy")
    ns = len(sectors)
    st = M.derive_state(cal, sol.x[:ns], sol.x[ns:])
    err = float(np.max(np.abs(st.X - cal.X0)))
    return err < 1e-6, f"real-SAM benchmark replication error = {err:.2e}", err, 1e-6


# -- open economy (Armington/CET) ---------------------------------------------
_OPEN_EMISSIONS = np.array([2.0, 0.5])


def _open_cal():
    from cge.data.sam import toy_open_sam
    from cge.engines.cge_static.calibrate_open import calibrate_open

    return calibrate_open(toy_open_sam(), sectors=["BRD", "MIL"], factors=["CAP", "LAB"])


def _open_solve(cal, carbon_cost=None):
    from cge.engines.cge_static import model_open as MO

    ns, nf = len(cal.sectors), len(cal.factors)
    cc = np.zeros(ns) if carbon_cost is None else carbon_cost
    sol = solve(
        lambda z: MO.residuals(cal, z, carbon_cost=cc, recycling="lump_sum"),
        MO.initial_guess(cal) * 1.03,
        prefer="scipy",
    )
    st = MO.derive_open_state(
        cal,
        sol.x[:ns],
        sol.x[ns : 2 * ns],
        sol.x[2 * ns : 2 * ns + nf],
        float(sol.x[-1]),
        carbon_cost=cc,
        recycling="lump_sum",
    )
    return sol, st


@check(SUITE, "open_benchmark_replication")
def _open_replication():
    """The open Armington/CET model replicates its benchmark SAM to machine precision (activity
    output, domestic sales, imports, exports)."""
    cal = _open_cal()
    _s, st = _open_solve(cal)
    err = max(
        float(np.max(np.abs(st.Z - cal.Z0))),
        float(np.max(np.abs(st.M - cal.M0))),
        float(np.max(np.abs(st.E - cal.E0))),
    )
    return err < 1e-6, f"open benchmark replication error = {err:.2e}", err, 1e-6


@check(SUITE, "open_carbon_price_causes_leakage")
def _open_leakage():
    """A carbon price on the dirty sector causes **carbon leakage**: its domestic output falls, its
    imports rise (substitution to foreign supply) and its exports fall (lost competitiveness) — the
    open-economy response Engines 1–2 and the closed CGE cannot show."""
    cal = _open_cal()
    _b, base = _open_solve(cal)
    _s, st = _open_solve(cal, carbon_cost=0.15 * _OPEN_EMISSIONS)
    out_falls = st.Z[0] < base.Z[0] - 1e-9
    imports_rise = st.M[0] > base.M[0] + 1e-9
    exports_fall = st.E[0] < base.E[0] - 1e-9
    ok = out_falls and imports_rise and exports_fall
    return (
        ok,
        f"dirty: output↓={out_falls}, imports↑={imports_rise}, exports↓={exports_fall}",
        None,
        None,
    )


@check(SUITE, "ces_value_added_replicates")
def _ces_va_replication():
    """The CES value-added nest (σ_va ≠ 1) replicates the benchmark to machine precision — the
    Cobb-Douglas pilot is the σ = 1 special case; a non-unitary elasticity must still calibrate to
    reproduce the base year."""
    cal = calibrate(toy_sam(), sectors=_SECTORS, factors=_FACTORS, va_elast=0.6)
    sol = solve(lambda z: M.residuals(cal, z), M.initial_guess(cal) * 1.05, prefer="scipy")
    ns = len(cal.sectors)
    st = M.derive_state(cal, sol.x[:ns], sol.x[ns:])
    err = float(np.max(np.abs(st.X - cal.X0)))
    return err < 1e-6, f"CES (σ=0.6) benchmark replication error = {err:.2e}", err, 1e-6


def _balanced_nonzero_sf_sam():
    """A **genuinely balanced** open SAM with a non-zero current account (review round-2 P2: the old
    fixture omitted the ROW capital transfer, so it was actually unbalanced). Imports (40) exceed
    exports (30) ⇒ Sf = 10; the ROW account balances via a **ROW→HOH capital transfer of 10** (the
    net capital inflow that finances the trade deficit). The household spends that inflow, so its
    budget also balances. This is the exact balanced-Sf≠0 case the pilot must reject at calibration
    (its income identity does not yet carry the ROW transfer)."""
    from datetime import date

    import pandas as pd

    from cge.contracts.data_objects import SAM, Provenance

    accts = ["a_BRD", "a_MIL", "c_BRD", "c_MIL", "CAP", "LAB", "HOH", "ROW"]
    exp = {"BRD": 20.0, "MIL": 10.0}  # Σ = 30
    imp = {"BRD": 22.0, "MIL": 18.0}  # Σ = 40  ⇒ Sf = 10
    dom = {"BRD": 80.0, "MIL": 110.0}
    inter = {("c_MIL", "a_BRD"): 24.0, ("c_BRD", "a_MIL"): 15.0}
    transfer = imp["BRD"] + imp["MIL"] - exp["BRD"] - exp["MIL"]  # ROW → HOH capital inflow = 10
    m = pd.DataFrame(0.0, index=accts, columns=accts)
    for s in ("BRD", "MIL"):
        m.loc[f"a_{s}", f"c_{s}"] = dom[s]
        m.loc[f"a_{s}", "ROW"] = exp[s]
        m.loc["ROW", f"c_{s}"] = imp[s]
    for (com, act), v in inter.items():
        m.loc[com, act] = v
    for s in ("BRD", "MIL"):
        va = dom[s] + exp[s] - sum(m.loc[c, f"a_{s}"] for c in ("c_BRD", "c_MIL"))
        m.loc["CAP", f"a_{s}"] = m.loc["LAB", f"a_{s}"] = va / 2.0
    m.loc["HOH", "ROW"] = transfer  # ROW pays HOH the net capital inflow (balances the ROW account)
    for s in ("BRD", "MIL"):
        m.loc[f"c_{s}", "HOH"] = m[f"c_{s}"].sum() - m.loc[f"c_{s}"].sum()
    m.loc["HOH", "CAP"] = m.loc["CAP", ["a_BRD", "a_MIL"]].sum()
    m.loc["HOH", "LAB"] = m.loc["LAB", ["a_BRD", "a_MIL"]].sum()
    prov = Provenance(
        source="validation",
        source_version="v",
        licence="n/a",
        reference_year=0,
        retrieved=date.today().isoformat(),
        notes="balanced non-zero-Sf open SAM",
    )
    return SAM(provenance=prov, accounts=accts, matrix=m)


@check(SUITE, "open_nonzero_foreign_savings_rejected")
def _open_nonzero_sf_rejected():
    """The open pilot's closure (income = factor income + recycled revenue) only replicates a
    balanced current account, so a **genuinely balanced** SAM with non-zero foreign savings is
    **rejected at calibration** rather than silently returning a non-replicating benchmark (2026-07
    review P1). A proper ROW-transfer closure is a documented follow-up."""
    from cge.data.sam.balance import is_balanced
    from cge.engines.cge_static.calibrate_open import calibrate_open

    sam = _balanced_nonzero_sf_sam()
    if not is_balanced(sam.matrix, tol=1e-9):
        return False, "fixture SAM is not actually balanced", None, None
    try:
        calibrate_open(sam, sectors=["BRD", "MIL"], factors=["CAP", "LAB"])
        return False, "non-zero foreign savings was NOT rejected", None, None
    except ValueError as e:
        ok = "balanced current account" in str(e)
        return ok, f"balanced Sf≠0 SAM rejected as expected: {str(e)[:50]}", None, None


@check(SUITE, "open_homogeneity")
def _open_homogeneity():
    """Scaling factor endowments by κ scales all real quantities by κ and leaves prices + the
    exchange rate unchanged — the open model has no money illusion (standard CGE property)."""
    from dataclasses import replace

    cal = _open_cal()
    _s, st = _open_solve(cal)
    cal_k = replace(cal, endowment=cal.endowment * 1.5)
    _sk, st_k = _open_solve(cal_k)
    price_err = float(np.max(np.abs(_sk.x - _s.x)))  # prices + er unchanged
    scale_err = float(np.max(np.abs(st_k.Z - 1.5 * st.Z)))  # output scales by κ
    err = max(price_err, scale_err)
    return err < 1e-6, f"open homogeneity error = {err:.2e}", err, 1e-6


@check(SUITE, "open_walras_and_trade_balance")
def _open_walras():
    """Walras' law + trade balance at the open equilibrium under a carbon shock: the dropped factor
    market clears (though its equation was omitted by Walras), and the value trade balance closes
    (Σ pm·M = Σ pe·E at zero foreign savings). If either failed, the 'square system + one dropped
    market' construction would be unsound."""
    cal = _open_cal()
    _s, st = _open_solve(cal, carbon_cost=0.15 * _OPEN_EMISSIONS)
    # The dropped factor (index 0) must still clear at the solution.
    dropped_gap = float(abs(st.F[0, :].sum() - cal.endowment[0]))
    pm = st.er * np.ones(len(cal.sectors))
    trade_gap = float(abs(pm @ st.M - pm @ st.E - st.er * cal.foreign_savings))
    err = max(dropped_gap, trade_gap)
    return (
        err < 1e-7,
        f"dropped-factor gap={dropped_gap:.2e}, trade-balance gap={trade_gap:.2e}",
        err,
        1e-7,
    )


@check(SUITE, "open_income_identity")
def _open_income_identity():
    """The household budget identity holds exactly at the open equilibrium: income = factor income
    + recycled carbon revenue (review round-1 P2 fixed the closed-form; this pins it in the standing
    suite under a carbon shock)."""
    cal = _open_cal()
    _s, st = _open_solve(cal, carbon_cost=0.15 * _OPEN_EMISSIONS)
    factor_income = float(np.dot(st.w, cal.endowment))
    gap = float(abs(st.income - (factor_income + st.carbon_revenue)))
    return gap < 1e-9, f"open income-identity gap = {gap:.2e}", gap, 1e-9
