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


def _solve(cal, carbon_cost=None, drop_factor=0):
    cc = np.zeros(len(cal.sectors)) if carbon_cost is None else carbon_cost
    sol = solve(
        lambda z: M.residuals(cal, z, carbon_cost=cc, drop_factor=drop_factor),
        M.initial_guess(cal),
        prefer="scipy",
    )
    ns = len(cal.sectors)
    return sol, M.derive_state(cal, sol.x[:ns], sol.x[ns:])


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


@check(SUITE, "carbon_price_direction")
def _carbon_direction():
    """A carbon price makes the dirty sector's output fall and real GDP fall (the GE carbon-price
    response Engines 1–2 approximate)."""
    cal = _cal()
    _b, base = _solve(cal)
    _s, st = _solve(cal, carbon_cost=0.15 * _EMISSIONS)
    dirty_falls = st.X[0] < base.X[0] - 1e-9  # BRD, the dirty sector
    real_gdp_falls = st.FD.sum() < base.FD.sum() - 1e-9
    ok = dirty_falls and real_gdp_falls
    return ok, f"dirty output falls={dirty_falls}, real GDP falls={real_gdp_falls}", None, None


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


@check(SUITE, "replicates_on_real_exiobase_sam")
def _real_sam_replication():
    """The 5.1b gate: build a SAM from a real (offline) EXIOBASE build, quality-gate it, and
    confirm the CGE calibrates and replicates its benchmark to machine precision — proving the
    model works on real balanced data, not only the toy."""
    import tempfile

    from cge.data.build import build_test
    from cge.data.sam import build_sam
    from cge.data.store import DataStore

    store = DataStore(tempfile.mkdtemp())
    build_test(store=store)
    bid = next(b for b in store.build_ids() if b != "exiobase-test")
    io = store.load(bid)["IOSystem"]
    sam, report, sectors = build_sam(io)
    if not report.passed:
        return False, "SAM quality gate failed on the real build", None, None
    cal = calibrate(sam, sectors=sectors, factors=["CAP", "LAB"])
    sol = solve(lambda z: M.residuals(cal, z), M.initial_guess(cal) * 1.05, prefer="scipy")
    ns = len(sectors)
    st = M.derive_state(cal, sol.x[:ns], sol.x[ns:])
    err = float(np.max(np.abs(st.X - cal.X0)))
    return err < 1e-6, f"real-SAM benchmark replication error = {err:.2e}", err, 1e-6
