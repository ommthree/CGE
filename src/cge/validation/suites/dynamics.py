"""Validation suite for the recursive-dynamic wrapper (Phase 7.1).

Standing model-correctness checks tied to docs/models/recursive-dynamics.md — the load-bearing
invariants of the capital-carry loop, so a regression re-surfaces in the battery, not just pytest:

- the accumulation identity K_{t+1} = (1−δ)(1−r)K_t + INV_t holds at every step;
- the recursive path STARTS from the benchmark (year 0, no shock ⇒ real GDP change 0);
- premature retirement (stranded assets) lowers the closing stock;
- the multi-region CGE carries a genuinely PER-REGION capital path (each region's identity holds
  independently), on the hand-checkable ``toy_cge_multi_gov`` SAM.

The single-region checks run on the hand-checkable ``toy_cge_gov`` SAM, exactly like a user's
recursive run.
"""

from __future__ import annotations

from cge.validation.framework import check

SUITE = "dynamics"


def _base_path():
    from cge.dynamics import run_recursive
    from cge.scenarios.loader import Scenario

    sc = Scenario(name="v", engine="cge_static", years=[2025, 2030, 2035], shocks=[])
    return run_recursive(sc, data_source="toy_cge_gov")


@check(SUITE, "accumulation_identity_holds_across_path")
def _identity():
    from cge.engines.cge_static.capital import capital_next

    path = _base_path()
    # benchmark_capital_stock is a per-region list; the closed/gov suite SAM has one aggregate K.
    k_prev = path.result.manifest.assumptions["recursive_dynamics"]["benchmark_capital_stock"][0]
    worst = 0.0
    for year in (2025, 2030, 2035):
        expected = float(capital_next(k_prev, path.investment[year], depreciation=0.05))
        worst = max(worst, abs(expected - path.capital_stock[year]))
        k_prev = path.capital_stock[year]
    return worst < 1e-10, "K_{t+1}=(1−δ)K_t+INV_t at every step", worst, 1e-10


@check(SUITE, "path_starts_from_benchmark")
def _starts_at_benchmark():
    path = _base_path()
    d = path.result.data
    g = d[(d["variable"] == "gdp_change_real") & (d["scenario"] == "central") & (d["year"] == 2025)]
    val = float(g["value"].iloc[0])
    return abs(val) < 1e-9, "year 0 (no shock) is the benchmark: real GDP change 0", val, 1e-9


@check(SUITE, "retirement_lowers_the_stock")
def _retirement():
    from cge.dynamics import DynamicConfig, run_recursive
    from cge.scenarios.loader import Scenario

    sc = Scenario(name="v", engine="cge_static", years=[2025, 2030], shocks=[])
    base = run_recursive(sc, data_source="toy_cge_gov")
    stranded = run_recursive(
        sc, config=DynamicConfig(retirement={2030: 0.2}), data_source="toy_cge_gov"
    )
    drop = base.capital_stock[2030] - stranded.capital_stock[2030]
    return drop > 0, "premature retirement writes down the closing stock", drop


@check(SUITE, "multi_region_per_region_capital_path")
def _multi_region():
    import numpy as np

    from cge.dynamics import run_recursive
    from cge.engines.cge_static.capital import capital_next
    from cge.scenarios.loader import Scenario

    sc = Scenario(name="v", engine="cge_static", years=[2025, 2030, 2035], shocks=[])
    path = run_recursive(sc, data_source="toy_cge_multi_gov")
    rd = path.result.manifest.assumptions["recursive_dynamics"]
    k_prev = np.asarray(rd["benchmark_capital_stock"], dtype=float)
    if k_prev.shape != (2,):
        return False, "multi CGE reports a per-region capital vector (2 regions)", k_prev.shape
    worst = 0.0
    for year in (2025, 2030, 2035):
        inv = np.asarray(path.investment[year], dtype=float)
        expected = capital_next(k_prev, inv, depreciation=0.05)
        worst = max(worst, float(np.max(np.abs(expected - np.asarray(path.capital_stock[year])))))
        k_prev = np.asarray(path.capital_stock[year])
    return worst < 1e-10, "per-region K_{t+1}=(1−δ)K_t+INV_t at every step", worst, 1e-10
