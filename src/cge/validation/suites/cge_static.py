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
    # The IO build now carries GOV/SAVINV (institution split, review P1 round 13); name them so the
    # macro closures calibrate — this gate now also proves the closures run on a REAL built SAM.
    institutions = {"household": "HOH"}
    if "GOV" in sam.accounts:
        institutions["government"] = "GOV"
    if "SAVINV" in sam.accounts:
        institutions["savings_investment"] = "SAVINV"
    cal = calibrate(sam, sectors=sectors, factors=["CAP", "LAB"], institutions=institutions)
    sol = solve(lambda z: M.residuals(cal, z), M.initial_guess(cal) * 1.05, prefer="scipy")
    ns = len(sectors)
    st = M.derive_state(cal, sol.x[:ns], sol.x[ns:])
    err = float(np.max(np.abs(st.X - cal.X0)))
    return err < 1e-6, f"real-SAM benchmark replication error = {err:.2e}", err, 1e-6


@check(SUITE, "open_replicates_on_built_sam")
def _open_real_sam_replication():
    """The open analogue of the 5.1b gate: build an **open** SAM (home region + rest-of-world) from
    an EXIOBASE-shaped build, quality-gate it, and confirm the open CGE calibrates and replicates
    its benchmark to machine precision — proving the IOSystem→open-SAM→calibrate→solve pipeline
    works on structured multi-region data (offline pymrio test MRIO, not live EXIOBASE)."""
    import tempfile

    from cge.data.build import build_test
    from cge.data.sam import build_open_sam
    from cge.data.store import DataStore
    from cge.engines.cge_static import model_open as MO
    from cge.engines.cge_static.calibrate_open import calibrate_open

    store = DataStore(tempfile.mkdtemp())
    build_test(store=store)
    bid = next(b for b in store.build_ids() if b != "exiobase-test")
    io = store.load(bid)["IOSystem"]
    home = list(io.regions.labels)[0]
    sam, report, sectors = build_open_sam(io, home_region=home)
    if not report.passed:
        return False, "open SAM quality gate failed on the built SAM", None, None
    cal = calibrate_open(sam, sectors=sectors, factors=["CAP", "LAB"])
    ns = len(sectors)
    sol = solve(
        lambda z: MO.residuals(cal, z, recycling="lump_sum"),
        MO.initial_guess(cal) * 1.03,
        prefer="scipy",
    )
    st = MO.derive_open_state(
        cal,
        sol.x[:ns],
        sol.x[ns : 2 * ns],
        sol.x[2 * ns : 2 * ns + 2],
        float(sol.x[-1]),
        recycling="lump_sum",
        strict=True,
    )
    err = max(float(np.max(np.abs(st.Z - cal.Z0))), float(np.max(np.abs(st.M - cal.M0))))
    return err < 1e-6, f"open real-SAM replication error = {err:.2e}", err, 1e-6


@check(SUITE, "multi_region_live_replicates_on_built_sam")
def _multi_real_sam_replication():
    """The Phase 5.1b DoD gate: build a genuine **multi-region** SAM (bilateral trade among ALL
    build regions) from an EXIOBASE-shaped IOSystem via ``build_multi_sam``, quality- and
    topology-gate it, then confirm the multi-region CGE calibrates and replicates its benchmark to
    machine precision — proving the IOSystem→multi-SAM→calibrate→solve pipeline works on structured
    multi-region data (offline pymrio test MRIO, not live EXIOBASE), and that the built SAM's
    trade-materiality/connectivity survives calibration (active_routes ≠ ∅, one connected
    component).

    **Scope (review P1 round 14):** this gate validates the COARSE 2-region × 3-sector aggregated
    build — whose cross-region trade is above the materiality threshold. The FULL 6-region ×
    8-sector
    ``exiobase-test`` fixture has entirely-dust inter-region trade at the sector granularity, so
    ``build_multi_sam`` correctly rejects it and ``aggregate_dust_regions`` reports that no genuine
    multi-region structure survives (both behaviours are unit-tested in ``test_sam.py``). A
    live-EXIOBASE multi-region build with real cross-region trade is the remaining data step."""
    import tempfile

    from cge.data.build import build_test
    from cge.data.sam import build_multi_sam
    from cge.data.store import DataStore
    from cge.engines.cge_static import model_multi as MM
    from cge.engines.cge_static.calibrate_multi import calibrate_multi

    store = DataStore(tempfile.mkdtemp())
    build_test(store=store)
    # The aggregated small build is the 2-region × 3-sector multi-region system (coarse; the full
    # 6×8 fixture's cross-region trade is dust — see the docstring and the aggregate_dust_regions
    # workflow tests).
    bid = next(b for b in store.build_ids() if b != "exiobase-test")
    io = store.load(bid)["IOSystem"]
    sam, report, regions, sectors = build_multi_sam(io)
    if not report.passed:
        return False, "multi-region SAM quality gate failed on the built SAM", None, None
    cal = calibrate_multi(sam, regions=regions, sectors=sectors, factors=["CAP", "LAB"])
    if not cal.active_routes:
        return False, "built multi-region SAM has no active trade routes", None, None
    if len(cal.connected_components) != 1:
        return False, "built multi-region SAM is disconnected", None, None
    sol = solve(
        lambda z: MM.residuals(cal, z, recycling="lump_sum"),
        MM.initial_guess(cal) * 1.03,
        prefer="scipy",
    )
    st = MM.unpack_state(cal, sol.x, recycling="lump_sum", strict=True)
    err = max(float(np.max(np.abs(st.Z - cal.Z0))), float(np.max(np.abs(st.M - cal.M0))))
    return err < 1e-6, f"multi-region live-SAM replication error = {err:.2e}", err, 1e-6


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


def _multi_cal():
    from cge.data.sam.toy_multi import REGIONS, SECTORS, toy_multi_sam
    from cge.engines.cge_static.calibrate_multi import calibrate_multi

    return calibrate_multi(
        toy_multi_sam(), regions=REGIONS, sectors=SECTORS, factors=["CAP", "LAB"]
    )


def _multi_solve(cal, carbon_cost=None):
    from cge.engines.cge_static import model_multi as MM

    sol = solve(
        lambda z: MM.residuals(cal, z, carbon_cost=carbon_cost, recycling="lump_sum"),
        MM.initial_guess(cal) * 1.03,
        prefer="scipy",
    )
    st = MM.unpack_state(cal, sol.x, carbon_cost=carbon_cost, recycling="lump_sum")
    return sol, st


@check(SUITE, "multi_region_benchmark_replication")
def _multi_replication():
    """The multi-region model (R regions, bilateral Armington/CET) replicates its benchmark to
    machine precision — activity output and every bilateral import/export return to the SAM
    values at unit prices, proving the region-indexed calibration + model are sound."""
    cal = _multi_cal()
    _s, st = _multi_solve(cal)
    err = max(
        float(np.max(np.abs(st.Z - cal.Z0))),
        float(np.max(np.abs(st.M - cal.M0))),
        float(np.max(np.abs(st.EX - cal.EX0))),
    )
    return err < 1e-6, f"multi-region benchmark replication error = {err:.2e}", err, 1e-6


@check(SUITE, "multi_region_cross_region_leakage")
def _multi_leakage():
    """The signature multi-region result: a carbon price on ONE region's dirty sector cuts that
    region's output, RAISES its imports of the good from the partner region (cross-region leakage),
    and RAISES the partner's output — carbon pricing relocates production across build regions."""
    import numpy as _np

    cal = _multi_cal()
    _b, base = _multi_solve(cal)
    cc = _np.zeros((cal.nr, cal.ns))
    cc[0, 0] = 0.3  # region 0, sector 0 (the dirty sector)
    _s, st = _multi_solve(cal, carbon_cost=cc)
    out_falls = st.Z[0, 0] < base.Z[0, 0] - 1e-9
    imports_rise = st.M[0, 0, 1] > base.M[0, 0, 1] + 1e-12
    partner_rises = st.Z[1, 0] > base.Z[1, 0] + 1e-9
    ok = out_falls and imports_rise and partner_rises
    return (
        ok,
        f"taxed↓={out_falls}, imports↑={imports_rise}, partner output↑={partner_rises}",
        None,
        None,
    )


@check(SUITE, "multi_region_markets_clear_under_shock")
def _multi_clearing():
    """The equilibrium correctness gate the earlier design failed: at the SHOCKED equilibrium every
    **bilateral goods market** clears (import demand M[d,s,o] = export supply EX[o,s,d]) and every
    **regional factor market** clears (Walras). A machine-zero solver residual with unbalanced trade
    is NOT an equilibrium; this pins that the redesigned bilateral-price system genuinely clears."""
    import numpy as _np

    cal = _multi_cal()
    cc = _np.zeros((cal.nr, cal.ns))
    cc[0, 0] = 0.3
    _s, st = _multi_solve(cal, carbon_cost=cc)
    trade_disc = 0.0
    for d in range(cal.nr):
        for o in range(cal.nr):
            if d != o:
                trade_disc = max(
                    trade_disc, float(_np.max(_np.abs(st.M[d, :, o] - st.EX[o, :, d])))
                )
    factor_gap = 0.0
    for fi in range(cal.nf):
        for ri in range(cal.nr):
            factor_gap = max(factor_gap, abs(float(st.F[fi, ri, :].sum()) - cal.endowment[fi, ri]))
    err = max(trade_disc, factor_gap)
    return err < 1e-8, f"max bilateral-trade + factor-market discrepancy = {err:.2e}", err, 1e-8


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


@check(SUITE, "open_nonzero_foreign_savings_replicates")
def _open_nonzero_sf_replicates():
    """A **genuinely balanced** open SAM with a non-zero current account (imports ≠ exports, closed
    by a ROW→household capital transfer of Sf) calibrates and **replicates its benchmark to machine
    precision**. Foreign savings enter household income as er·Sf, so the model runs a non-zero
    current account exactly (Phase 5 deferred: the ROW closure that lifted the balanced-CA
    restriction)."""
    from cge.data.sam.balance import is_balanced
    from cge.engines.cge_static.calibrate_open import calibrate_open

    sam = _balanced_nonzero_sf_sam()
    if not is_balanced(sam.matrix, tol=1e-9):
        return False, "fixture SAM is not actually balanced", None, None
    cal = calibrate_open(sam, sectors=["BRD", "MIL"], factors=["CAP", "LAB"])
    _s, st = _open_solve(cal)
    err = max(
        float(np.max(np.abs(st.Z - cal.Z0))),
        float(np.max(np.abs(st.M - cal.M0))),
        float(np.max(np.abs(st.FD - cal.FD0))),
    )
    return (
        err < 1e-6 and cal.foreign_savings > 0,
        f"non-zero-Sf (Sf={cal.foreign_savings:.4g}) replication error = {err:.2e}",
        err,
        1e-6,
    )


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


# -- interaction gates (review P2, 2026-07-27) — the standing suite must exercise the interactions
# the round-12 review found the P1 defects in, not just each mechanism in isolation. -----------
def _gov_inv_cal():
    """A closed SAM with BOTH a government (GOV) and a savings-investment (SAVINV) account, for the
    deficit-closure × carbon-recipient and deficit × adaptation interaction gates."""
    import pandas as pd

    from cge.contracts.data_objects import SAM, Provenance

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
    m.loc["BRD", "HOH"] -= 19.0
    m.loc["MIL", "HOH"] -= 15.39
    prov = Provenance(
        source="t", source_version="1", licence="x", reference_year=0, retrieved="2026-07-27"
    )
    return calibrate(
        SAM(provenance=prov, accounts=acc, matrix=m),
        sectors=_SECTORS,
        factors=_FACTORS,
        institutions={"household": "HOH", "government": "GOV", "savings_investment": "SAVINV"},
    )


def _solve_gov(cal, recipient, cc, adapt_amount=0.0, adapt_gamma=None):
    ns = len(cal.sectors)
    sol = solve(
        lambda z: M.residuals(
            cal,
            z,
            carbon_cost=cc,
            recycling="lump_sum",
            gov_closure="deficit_financed",
            carbon_revenue_recipient=recipient,
            adapt_amount=adapt_amount,
            adapt_gamma=adapt_gamma,
        ),
        M.initial_guess(cal),
        prefer="scipy",
    )
    st = M.derive_state(
        cal,
        sol.x[:ns],
        sol.x[ns:],
        carbon_cost=cc,
        recycling="lump_sum",
        strict=True,
        gov_closure="deficit_financed",
        carbon_revenue_recipient=recipient,
        adapt_amount=adapt_amount,
        adapt_gamma=adapt_gamma,
    )
    return st


@check(SUITE, "deficit_carbon_recipient_controls_ownership")
def _deficit_carbon_recipient():
    """Deficit closure × carbon-revenue recipient: 'government' gives the government tax + carbon
    revenue (its income exceeds the tax alone); 'household' gives it the tax only. The recipient —
    not the closure — controls ownership (review P1)."""
    cal = _gov_inv_cal()
    cc = 0.3 * _EMISSIONS
    gov = _solve_gov(cal, "government", cc)
    hh = _solve_gov(cal, "household", cc)
    tax = float(cal.gov_tax_rate0 * gov.factor_income)
    gov_gets_revenue = gov.gov_income > tax + 1e-4
    hh_gov_is_tax_only = abs(hh.gov_income - tax) < 1e-6
    ok = gov_gets_revenue and hh_gov_is_tax_only
    return (
        ok,
        f"gov_inc(gov)={gov.gov_income:.4f} vs tax={tax:.4f}; gov_inc(hh)={hh.gov_income:.4f}",
        None,
        None,
    )


@check(SUITE, "deficit_investment_nonnegative_componentwise")
def _deficit_investment_nonnegative():
    """Deficit closure × adaptation: a feasible run keeps every sector's investment demand
    non-negative, and an over-earmark (adaptation > net investment budget) is rejected — never a
    numerically-exact but negative-investment equilibrium (review P1)."""
    cal = _gov_inv_cal()
    cc = 0.5 * _EMISSIONS
    st = _solve_gov(cal, "household", cc)  # a genuine deficit that crowds out investment
    min_id = float(st.ID.min())
    # And the over-earmark case raises.
    rejected = False
    try:
        _solve_gov(cal, "household", cc, adapt_amount=0.02, adapt_gamma=cal.inv_gamma * 0 + 0.5)
    except ValueError:
        rejected = True
    ok = min_id >= -1e-12 and rejected
    return ok, f"min ID={min_id:.3e}; over-earmark rejected={rejected}", min_id, -1e-12


@check(SUITE, "every_closure_combination_clears_every_market")
def _every_closure_clears_every_market():
    """THE false-equilibrium gate (review P1 round 14): for EVERY supported closure combination
    (gov_closure × inv_closure × carbon_revenue_recipient) the ACCEPTED equilibrium must clear
    EVERY factor market — including the Walras-dropped one, whose residual the solver never sees.
    The earlier fixed_real + household-recipient case returned solver success while carrying a
    ~0.577% capital-market gap (its revenue fixed point used the savings-driven demand deriv.)."""
    cal = _gov_inv_cal()
    cc = 0.3 * _EMISSIONS
    ns, nf = len(cal.sectors), len(cal.factors)
    worst = 0.0
    worst_combo = ""
    for gov in ("balanced_budget", "deficit_financed"):
        for inv in ("savings_driven", "fixed_real"):
            for recip in ("government", "household"):
                if gov == "deficit_financed" and inv == "fixed_real":
                    continue  # deficit_financed is savings_driven-only (rejected loudly elsewhere)
                kw = dict(
                    carbon_cost=cc,
                    recycling="lump_sum",
                    gov_closure=gov,
                    inv_closure=inv,
                    carbon_revenue_recipient=recip,
                )
                sol = solve(
                    lambda z, kw=kw: M.residuals(cal, z, **kw), M.initial_guess(cal), prefer="scipy"
                )
                st = M.derive_state(cal, sol.x[:ns], sol.x[ns:], strict=True, **kw)
                gap = max(abs(float(st.F[f, :].sum()) - cal.endowment[f]) for f in range(nf))
                if gap > worst:
                    worst, worst_combo = gap, f"{gov}/{inv}/{recip}"
    ok = worst < 1e-9
    return ok, f"worst factor-market gap {worst:.2e} at {worst_combo or 'n/a'}", worst, 1e-9


@check(SUITE, "multi_dust_route_rejected")
def _multi_dust_rejected():
    """A supplied multi-region SAM with a sub-threshold dust route is rejected at calibration (not
    silently calibrated into a 35%-imbalance equilibrium), and the toy SAM (clean) calibrates —
    every calibrated trade share has a clearing route (review P1)."""

    from cge.contracts.data_objects import SAM, Provenance
    from cge.data.sam.toy_multi import REGIONS as MR
    from cge.data.sam.toy_multi import SECTORS as MS
    from cge.data.sam.toy_multi import toy_multi_sam
    from cge.engines.cge_static.calibrate_multi import calibrate_multi

    # Clean toy calibrates and every share has a route.
    cal = calibrate_multi(toy_multi_sam(), regions=MR, sectors=MS, factors=_FACTORS)
    active = set(cal.active_routes)
    aligned = all(
        (cal.arm_share_m[d, s, o] > 0.0) == ((o, s, d) in active) or cal.arm_share_m[d, s, o] == 0.0
        for o in range(cal.nr)
        for d in range(cal.nr)
        if o != d
        for s in range(cal.ns)
    )
    # A dust route is rejected.
    m = toy_multi_sam().matrix.copy()
    accounts = list(m.index)
    m.loc["a_N_MIL", "c_S_MIL"] = m.loc["a_N_MIL", "c_S_MIL"]  # existing real route unchanged
    # Inject a truly tiny extra route cell that is sub-threshold and unbalanced → dust gate fires.
    m2 = m.copy()
    m2.loc["a_N_BRD", "c_S_BRD"] = 1e-9
    prov = Provenance(
        source="t", source_version="1", licence="x", reference_year=0, retrieved="2026-07-27"
    )
    rejected = False
    try:
        calibrate_multi(
            SAM(provenance=prov, accounts=accounts, matrix=m2),
            regions=MR,
            sectors=MS,
            factors=_FACTORS,
        )
    except ValueError as e:
        rejected = "dust" in str(e)
    ok = aligned and rejected
    return ok, f"shares aligned with routes={aligned}; dust rejected={rejected}", None, None


@check(SUITE, "capital_bridge_reports_implied_growth")
def _capital_bridge_growth():
    """The capital bridge does not assume a steady state: it converts capital INCOME to a stock and
    reports the implied growth rate from the ACTUAL INV0, which is nonzero on the fixtures (review
    P2). g = INV0/K0 − δ matches the real INV0, not a fabricated δ·K."""
    from cge.engines.cge_static.capital import (
        DEFAULT_DEPRECIATION_RATE,
        benchmark_capital,
        implied_growth_rate,
    )

    cal = _gov_inv_cal()  # has SAVINV, so INV0 exists
    k0 = benchmark_capital(cal)
    inv0 = float(cal.INV0.sum())
    g = implied_growth_rate(cal)
    expected = inv0 / k0[0] - DEFAULT_DEPRECIATION_RATE
    gap = abs(float(g[0]) - expected)
    return gap < 1e-9, f"implied g={float(g[0]):+.4f} from real INV0 (gap {gap:.1e})", gap, 1e-9


@check(SUITE, "standard_output_schema_all_variants")
def _standard_output_schema():
    """Every variant emits the standard named result schema (review P2): real GDP, GVA, wage,
    capital return, employment, emissions, welfare, deflator — including multi-region real GDP."""
    from cge.contracts.engine import registry
    from cge.contracts.shocks import CarbonPrice
    from cge.data.sam import toy_open_sam, toy_sam
    from cge.data.sam.toy_multi import toy_multi_sam

    eng = registry.get("cge_static")
    core = {
        "gdp_change_real",
        "gva_change",
        "wage_change",
        "capital_return_change",
        "employment_change",
        "covered_emissions_change",
        "welfare_change",
        "gdp_deflator_change",
    }
    runs = {
        "closed": eng.run(
            data={"SAM": toy_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}},
            shocks=[CarbonPrice(price=0.1)],
            years=[2020],
        ),
        "open": eng.run(
            data={"SAM": toy_open_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}},
            shocks=[CarbonPrice(price=0.1)],
            years=[2020],
        ),
        "multi": eng.run(
            data={"SAM": toy_multi_sam(), "carbon_cost_share": {"N": {"BRD": 0.3}}},
            shocks=[CarbonPrice(price=0.3)],
            years=[2020],
        ),
    }
    missing = {k: sorted(core - set(r.data["variable"].unique())) for k, r in runs.items()}
    bad = {k: v for k, v in missing.items() if v}
    return (
        not bad,
        f"variants missing schema vars: {bad}" if bad else "all variants emit the schema",
        None,
        None,
    )
