"""Tests for the recursive-dynamic wrapper (Phase 7.1).

Covers the capital-carry loop, the accumulation identity across the path, exogenous labour/TFP
trends, premature retirement (stranded assets), the result path + manifest provenance, and the
guard rails (needs a capital + savings-investment SAM; single-region only).
"""

from __future__ import annotations

import numpy as np
import pytest

from cge.dynamics import DynamicConfig, DynamicPath, run_recursive
from cge.engines.cge_static.capital import capital_next
from cge.scenarios.loader import Scenario


def _scenario(years):
    return Scenario(name="dyn", engine="cge_static", years=years, shocks=[])


def test_first_year_is_the_benchmark():
    """Year 0 solves at the benchmark stock (scale 1.0), so with no shock its real GDP change is 0 —
    the recursive path starts from the benchmark, exactly like a static run."""
    path = run_recursive(_scenario([2025, 2030]), data_source="toy_cge_gov")
    d = path.result.data
    g = d[(d["variable"] == "gdp_change_real") & (d["scenario"] == "central") & (d["year"] == 2025)]
    assert float(g["value"].iloc[0]) == pytest.approx(0.0, abs=1e-9)


def test_capital_accumulation_identity_holds_across_the_path():
    """Each step must satisfy K_{t+1} = (1−δ)·K_t + INV_t exactly (Phase 5d.3 identity)."""
    cfg = DynamicConfig(depreciation=0.05)
    path = run_recursive(_scenario([2025, 2030, 2035]), config=cfg, data_source="toy_cge_gov")
    k0 = path.result.manifest.assumptions["recursive_dynamics"]["benchmark_capital_stock"]
    k_prev = k0
    for year in (2025, 2030, 2035):
        expected = float(capital_next(k_prev, path.investment[year], depreciation=0.05))
        assert path.capital_stock[year] == pytest.approx(expected, rel=1e-12)
        k_prev = path.capital_stock[year]


def test_premature_retirement_reduces_the_stock():
    """A stranded-asset write-off in a year drops that year's closing stock below the no-retirement
    path (Phase 5d.3 retirement fraction)."""
    base = run_recursive(_scenario([2025, 2030]), data_source="toy_cge_gov")
    stranded = run_recursive(
        _scenario([2025, 2030]),
        config=DynamicConfig(retirement={2030: 0.2}),
        data_source="toy_cge_gov",
    )
    assert stranded.capital_stock[2030] < base.capital_stock[2030]


def test_labour_and_productivity_trends_raise_output():
    """With positive labour + TFP growth, later years' real GDP is higher than with flat trends
    (more effective factor supply), holding the capital dynamics comparable."""
    flat = run_recursive(_scenario([2025, 2040]), data_source="toy_cge_gov")
    grown = run_recursive(
        _scenario([2025, 2040]),
        config=DynamicConfig(labour_growth=0.02, productivity_growth=0.02),
        data_source="toy_cge_gov",
    )

    def gdp(path, year):
        d = path.result.data
        r = d[
            (d["variable"] == "gdp_change_real")
            & (d["scenario"] == "central")
            & (d["year"] == year)
        ]
        return float(r["value"].iloc[0])

    assert gdp(grown, 2040) > gdp(flat, 2040)


def test_result_has_capital_path_rows_and_manifest():
    path = run_recursive(_scenario([2025, 2030]), data_source="toy_cge_gov")
    assert isinstance(path, DynamicPath)
    path.result.validate_schema()
    variables = set(path.result.data["variable"].unique())
    assert {"capital_stock", "capital_growth"} <= variables
    rd = path.result.manifest.assumptions["recursive_dynamics"]
    assert "no perfect foresight" in rd["mode"]
    assert rd["horizon_years"] == [2025, 2030]
    assert rd["depreciation_rate"] == 0.05


def test_zero_trend_growth_matches_the_pure_capital_step():
    """With flat trends the only between-year change is capital; the growth rows equal the implied
    K path (a self-consistency check that the endowment scale and the identity agree)."""
    path = run_recursive(_scenario([2025, 2030, 2035]), data_source="toy_cge_gov")
    k0 = path.result.manifest.assumptions["recursive_dynamics"]["benchmark_capital_stock"]
    k_prev = k0
    for year in (2025, 2030, 2035):
        assert path.growth[year] == pytest.approx(
            path.capital_stock[year] / k_prev - 1.0, rel=1e-12
        )
        k_prev = path.capital_stock[year]


def test_carbon_shock_runs_through_the_dynamic_path():
    """A carbon price applies per year within the recursive path (shocks and dynamics compose)."""
    from cge.contracts.shocks import CarbonPrice

    sc = Scenario(
        name="dyn-carbon",
        engine="cge_static",
        years=[2025, 2030],
        shocks=[CarbonPrice(price=50.0)],
    )
    path = run_recursive(sc, data_source="toy_cge_gov")
    d = path.result.data
    # The dirty sector contracts under the carbon price in at least one year.
    vol = d[(d["variable"] == "volume_change") & (d["scenario"] == "central")]
    assert (vol["value"] < 0.0).any()


def test_requires_capital_and_investment_sam():
    """A SAM with no savings-investment account cannot accumulate — the wrapper raises a clear
    error rather than silently doing nothing."""
    with pytest.raises(ValueError, match="savings-investment|capital"):
        run_recursive(_scenario([2025, 2030]), data_source="toy_cge")  # no SAVINV account


def test_config_validation():
    with pytest.raises(ValueError, match="depreciation"):
        DynamicConfig(depreciation=1.5)
    with pytest.raises(ValueError, match="retirement"):
        DynamicConfig(retirement={2030: 1.5})
    assert np.isfinite(DynamicConfig().depreciation)
