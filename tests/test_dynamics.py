"""Tests for the recursive-dynamic wrapper (Phase 7.1).

Covers the capital-carry loop, the accumulation identity across the path, exogenous labour/TFP
trends, premature retirement (stranded assets), the result path + manifest provenance, and the
guard rails (needs a capital + savings-investment SAM). All three CGE variants: the closed/gov SAM
and the open economy (one aggregate stock), and the multi-region CGE (a per-region capital path).
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
    # benchmark_capital_stock is a per-region list (one entry for the single aggregate stock here).
    k_prev = path.result.manifest.assumptions["recursive_dynamics"]["benchmark_capital_stock"][0]
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
    k_prev = path.result.manifest.assumptions["recursive_dynamics"]["benchmark_capital_stock"][0]
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


# --- Open-economy variant (Phase 7.1 follow-up: recursive dynamics on the open CGE) ---------------
# The open CGE has one aggregate capital stock (like closed/gov), so the same scalar capital carry
# applies; these mirror the closed tests on the dynamic-capable ``toy_cge_open_gov`` SAM and confirm
# the open engine (Armington/CET + rest-of-world) really ran.


def test_open_first_year_is_the_benchmark():
    """Year 0 solves at the benchmark stock, so with no shock the open economy's real GDP change is
    0 — the open recursive path starts from the benchmark, exactly like the closed one."""
    path = run_recursive(_scenario([2025, 2030]), data_source="toy_cge_open_gov")
    d = path.result.data
    g = d[(d["variable"] == "gdp_change_real") & (d["scenario"] == "central") & (d["year"] == 2025)]
    assert float(g["value"].iloc[0]) == pytest.approx(0.0, abs=1e-9)
    # The open engine really ran: exchange-rate / trade variables are present.
    assert {"exchange_rate_change", "export_change"} <= set(d["variable"].unique())


def test_open_capital_accumulation_identity_holds():
    """Each step satisfies K_{t+1} = (1−δ)·K_t + INV_t exactly on the open variant too."""
    cfg = DynamicConfig(depreciation=0.05)
    path = run_recursive(_scenario([2025, 2030, 2035]), config=cfg, data_source="toy_cge_open_gov")
    k_prev = path.result.manifest.assumptions["recursive_dynamics"]["benchmark_capital_stock"][0]
    for year in (2025, 2030, 2035):
        expected = float(capital_next(k_prev, path.investment[year], depreciation=0.05))
        assert path.capital_stock[year] == pytest.approx(expected, rel=1e-12)
        k_prev = path.capital_stock[year]


def test_open_carbon_shock_runs_through_the_dynamic_path():
    """A carbon price composes with the open recursive path (shocks + dynamics + trade)."""
    from cge.contracts.shocks import CarbonPrice

    sc = Scenario(
        name="dyn-open-carbon",
        engine="cge_static",
        years=[2025, 2030],
        shocks=[CarbonPrice(price=50.0)],
    )
    path = run_recursive(sc, data_source="toy_cge_open_gov")
    d = path.result.data
    vol = d[(d["variable"] == "volume_change") & (d["scenario"] == "central")]
    assert (vol["value"] < 0.0).any()


def test_open_result_has_capital_path_rows_and_manifest():
    path = run_recursive(_scenario([2025, 2030]), data_source="toy_cge_open_gov")
    assert isinstance(path, DynamicPath)
    path.result.validate_schema()
    assert {"capital_stock", "capital_growth"} <= set(path.result.data["variable"].unique())
    rd = path.result.manifest.assumptions["recursive_dynamics"]
    assert rd["horizon_years"] == [2025, 2030]


# --- Multi-region variant (Phase 7.1 follow-up: a per-region capital path) -----------------------
# The multi-region CGE carries one capital stock per region; each steps by its own investment. These
# run on the dynamic-capable ``toy_cge_multi_gov`` SAM (per-region SAVINV accounts).


def test_multi_first_year_is_the_benchmark_per_region():
    """Year 0 solves at the benchmark stock in every region, so with no shock each region's real GDP
    change is 0 — the multi recursive path starts from the benchmark."""
    path = run_recursive(_scenario([2025, 2030]), data_source="toy_cge_multi_gov")
    d = path.result.data
    regions = path.result.manifest.assumptions["recursive_dynamics"]["capital_regions"]
    assert set(regions) == {"N", "S"}
    for region in regions:
        g = d[
            (d["variable"] == "gdp_change_real")
            & (d["scenario"] == "central")
            & (d["year"] == 2025)
            & (d["region"] == region)
        ]
        assert float(g["value"].iloc[0]) == pytest.approx(0.0, abs=1e-9)


def test_multi_capital_path_is_per_region():
    """The path dicts carry one capital entry per region, and the accumulation identity
    K_{t+1,r}=(1−δ)K_{t,r}+INV_{t,r} holds independently for each region."""
    cfg = DynamicConfig(depreciation=0.05)
    path = run_recursive(_scenario([2025, 2030, 2035]), config=cfg, data_source="toy_cge_multi_gov")
    rd = path.result.manifest.assumptions["recursive_dynamics"]
    k0 = np.asarray(rd["benchmark_capital_stock"], dtype=float)
    assert k0.shape == (2,)  # two regions
    k_prev = k0
    for year in (2025, 2030, 2035):
        inv = np.asarray(path.investment[year], dtype=float)
        expected = capital_next(k_prev, inv, depreciation=0.05)
        assert np.allclose(np.asarray(path.capital_stock[year]), expected, rtol=1e-12)
        k_prev = np.asarray(path.capital_stock[year])


def test_multi_result_has_per_region_capital_rows():
    """The result carries a capital_stock / capital_growth row for EACH region per year."""
    path = run_recursive(_scenario([2025, 2030]), data_source="toy_cge_multi_gov")
    path.result.validate_schema()
    d = path.result.data
    cap = d[(d["variable"] == "capital_stock") & (d["year"] == 2030)]
    assert set(cap["region"]) == {"N", "S"}


def test_multi_region_specific_capital_scaling_diverges():
    """Because each region carries its own stock, regions with different benchmark growth diverge:
    the capital path is genuinely per-region, not a shared aggregate applied everywhere."""
    path = run_recursive(_scenario([2025, 2035]), data_source="toy_cge_multi_gov")
    # N and S have different implied benchmark growth, so their year-2035 stock RATIOS to benchmark
    # differ — a shared scalar carry would move them identically.
    k0 = np.asarray(
        path.result.manifest.assumptions["recursive_dynamics"]["benchmark_capital_stock"]
    )
    k_end = np.asarray(path.capital_stock[2035])
    ratios = k_end / k0
    assert not np.isclose(ratios[0], ratios[1])
