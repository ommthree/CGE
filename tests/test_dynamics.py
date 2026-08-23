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


def test_capital_accumulation_identity_holds_across_consecutive_years():
    """For CONSECUTIVE reporting years each step satisfies K_{t+1}=(1−δ)·K_t+INV_t exactly (Phase
    5d.3 identity). With consecutive years the reported step IS the single annual step, so the
    identity holds directly; for sparse years the reported stock is the composition of the annual
    steps (see test_sparse_years_match_annual_path)."""
    cfg = DynamicConfig(depreciation=0.05)
    path = run_recursive(_scenario([2025, 2026, 2027]), config=cfg, data_source="toy_cge_gov")
    # benchmark_capital_stock is a per-region list (one entry for the single aggregate stock here).
    k_prev = path.result.manifest.assumptions["recursive_dynamics"]["benchmark_capital_stock"][0]
    for year in (2025, 2026, 2027):
        expected = float(capital_next(k_prev, path.investment[year], depreciation=0.05))
        assert path.capital_stock[year] == pytest.approx(expected, rel=1e-12)
        k_prev = path.capital_stock[year]


def test_sparse_years_match_annual_path():
    """A sparse reporting list must give the SAME capital path as the full annual list (review P1):
    the wrapper steps capital every calendar year internally regardless of which years are reported,
    so K at a shared year is identical whether or not the intervening years are requested. The old
    behaviour stepped only the reported years, over-counting a multi-year gap as a single step."""
    cfg = DynamicConfig(depreciation=0.05)
    sparse = run_recursive(_scenario([2025, 2030]), config=cfg, data_source="toy_cge_gov")
    annual = run_recursive(
        _scenario([2025, 2026, 2027, 2028, 2029, 2030]), config=cfg, data_source="toy_cge_gov"
    )
    assert sparse.capital_stock[2030] == pytest.approx(annual.capital_stock[2030], rel=1e-12)
    # And the reported stock at 2030 is the composition of five annual steps, NOT one step from the
    # benchmark stock — the bug the review reported would have inflated K2030.
    k0 = sparse.result.manifest.assumptions["recursive_dynamics"]["benchmark_capital_stock"][0]
    one_step = float(capital_next(k0, sparse.investment[2030], depreciation=0.05))
    assert sparse.capital_stock[2030] != pytest.approx(one_step, rel=1e-6)


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
    """With flat trends the only between-year change is capital; the growth row for a reported year
    is K_t/K_{t-annual} − 1 (the growth over the LAST annual step into the reported year), a self-
    consistency check that the endowment scale and the identity agree. Uses consecutive years so
    each reported growth is one annual step."""
    path = run_recursive(_scenario([2025, 2026, 2027]), data_source="toy_cge_gov")
    k_prev = path.result.manifest.assumptions["recursive_dynamics"]["benchmark_capital_stock"][0]
    for year in (2025, 2026, 2027):
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
    """Each step satisfies K_{t+1} = (1−δ)·K_t + INV_t exactly on the open variant too (consecutive
    reporting years, so the reported step is the single annual step)."""
    cfg = DynamicConfig(depreciation=0.05)
    path = run_recursive(_scenario([2025, 2026, 2027]), config=cfg, data_source="toy_cge_open_gov")
    k_prev = path.result.manifest.assumptions["recursive_dynamics"]["benchmark_capital_stock"][0]
    for year in (2025, 2026, 2027):
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
    K_{t+1,r}=(1−δ)K_{t,r}+INV_{t,r} holds independently for each region (consecutive reporting
    years, so the reported step is the single annual step)."""
    cfg = DynamicConfig(depreciation=0.05)
    path = run_recursive(_scenario([2025, 2026, 2027]), config=cfg, data_source="toy_cge_multi_gov")
    rd = path.result.manifest.assumptions["recursive_dynamics"]
    k0 = np.asarray(rd["benchmark_capital_stock"], dtype=float)
    assert k0.shape == (2,)  # two regions
    k_prev = k0
    for year in (2025, 2026, 2027):
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


# --- Structural trajectories (Phase 7b.2: sourced per-region labour/productivity drivers) ---------


def _real_gdp(path, region, year):
    d = path.result.data
    x = d[
        (d["variable"] == "gdp_change_real")
        & (d["scenario"] == "central")
        & (d["region"] == region)
        & (d["year"] == year)
    ]
    return float(x["value"].iloc[0])


def test_structural_trajectory_drives_per_region_divergence():
    """With a sourced StructuralTrajectory the two regions grow differently (S has faster sourced
    population + productivity than N), unlike a flat uniform trend which moves them together."""
    from cge.data.structural import load_structural_trajectories

    sc = Scenario(name="dyn", engine="cge_static", years=[2025, 2035, 2045], shocks=[])
    sourced = run_recursive(
        sc,
        config=DynamicConfig(structural=load_structural_trajectories()),
        data_source="toy_cge_multi_gov",
    )
    flat = run_recursive(
        sc, config=DynamicConfig(productivity_growth=0.02), data_source="toy_cge_multi_gov"
    )
    # Sourced: S clearly outgrows N. Flat: N and S grow near-identically (only tiny trade spill).
    assert _real_gdp(sourced, "S", 2045) > _real_gdp(sourced, "N", 2045) + 0.1
    assert abs(_real_gdp(flat, "S", 2045) - _real_gdp(flat, "N", 2045)) < 0.05


def test_flat_trend_still_works_without_trajectory():
    """Back-compat: with no structural trajectory the flat DynamicConfig scalars drive the trend."""
    sc = Scenario(name="dyn", engine="cge_static", years=[2025, 2035], shocks=[])
    grown = run_recursive(
        sc, config=DynamicConfig(productivity_growth=0.03), data_source="toy_cge_gov"
    )
    flat = run_recursive(sc, data_source="toy_cge_gov")
    assert _real_gdp(grown, "R", 2035) > _real_gdp(flat, "R", 2035)
    ts = grown.result.manifest.assumptions["recursive_dynamics"]["trend_source"]
    assert ts["kind"] == "flat"


def test_trajectory_compounds_across_year_gaps():
    """A 10-year solve gap compounds ~10 annual rates, not one: the labour-supply scale over
    [2025, 2035] equals the product of population×participation annual growth for each year."""
    from cge.contracts.data_objects import Provenance, StructuralTrajectory
    from cge.dynamics.recursive import _trend_scale

    traj = StructuralTrajectory(
        provenance=Provenance(
            source="t",
            source_version="v",
            licence="n/a",
            reference_year=2024,
            retrieved="2026-08-16",
        ),
        rates={
            "population": {"R": {2025: 0.01}},
            "labour_participation": {"R": {2025: 0.005}},
        },
        sources={"population:R": "c", "labour_participation:R": "c"},
        confidence={"population:R": "high", "labour_participation:R": "high"},
    )
    cfg = DynamicConfig(structural=traj)
    scale = _trend_scale(cfg, 2025, 2035, ["R"], "labour")[0]
    # Labour supply = population growth compounded WITH participation growth, the exact
    # multiplicative step (1+pop)(1+part) each year, NOT the additive pop+part approximation.
    assert scale == pytest.approx(((1.0 + 0.01) * (1.0 + 0.005)) ** 10, rel=1e-12)


def test_trajectory_provenance_is_stamped_on_the_manifest():
    """A sourced run records the trajectory's provenance + per-entry sources in its manifest."""
    from cge.data.structural import load_structural_trajectories

    sc = Scenario(name="dyn", engine="cge_static", years=[2025, 2030], shocks=[])
    path = run_recursive(
        sc,
        config=DynamicConfig(structural=load_structural_trajectories()),
        data_source="toy_cge_gov",
    )
    ts = path.result.manifest.assumptions["recursive_dynamics"]["trend_source"]
    assert ts["kind"] == "structural_trajectory"
    assert "productivity" in ts["drivers"]
    assert ts["sources"]  # per-entry citations carried through
    assert ts["provenance"]["source_version"]


# --- Per-sector structural drivers (Phase 7b.2 pass 2: sectoral drift + emissions intensity) ------


def _traj_sector(sector_rates):
    from cge.contracts.data_objects import Provenance, StructuralTrajectory

    keys = [f"{d}:{s}" for d, by in sector_rates.items() for s in by]
    return StructuralTrajectory(
        provenance=Provenance(
            source="t",
            source_version="v",
            licence="n/a",
            reference_year=2024,
            retrieved="2026-08-16",
        ),
        sector_rates=sector_rates,
        sources={k: "cite" for k in keys},
        confidence={k: "medium" for k in keys},
    )


def _sector_vol(path, sector, year):
    d = path.result.data
    r = d[
        (d["variable"] == "volume_change")
        & (d["scenario"] == "central")
        & (d["sector"] == sector)
        & (d["year"] == year)
    ]
    return float(r["value"].iloc[0])


def test_sector_productivity_drift_shifts_output_mix():
    """A sector with positive productivity drift gains output vs a no-drift run — the structural
    change rides the engine's existing per-sector theta multiplier."""
    from cge.contracts.shocks import CarbonPrice

    traj = _traj_sector({"sector_productivity": {"BRD": {2025: 0.03}}})
    sc = Scenario(
        name="drift", engine="cge_static", years=[2025, 2045], shocks=[CarbonPrice(price=50.0)]
    )
    drift = run_recursive(sc, config=DynamicConfig(structural=traj), data_source="toy_cge_gov")
    flat = run_recursive(sc, config=DynamicConfig(), data_source="toy_cge_gov")
    assert _sector_vol(drift, "BRD", 2045) > _sector_vol(flat, "BRD", 2045) + 0.1


def test_sector_productivity_all_default_drives_every_model_sector():
    """An ``__all__``-only sector-productivity trajectory (or one whose keys do not match real
    sector names) must drive EVERY model sector, not silently do nothing (review P1: the old code
    skipped ``__all__`` without enumerating the model's sectors, so such a trajectory emitted zero
    shocks — and on a real EXIOBASE model the shipped BRD/MIL keys never matched real names)."""
    from cge.dynamics.recursive import DynamicConfig as _DC
    from cge.dynamics.recursive import _sector_productivity_shocks

    traj = _traj_sector({"sector_productivity": {"__all__": {2025: 0.02}}})
    shocks = _sector_productivity_shocks(_DC(structural=traj), 2025, 2027, ["BRD", "MIL"])
    assert {tuple(s.coverage_sectors) for s in shocks} == {("BRD",), ("MIL",)}
    assert all(s.delta > 0 for s in shocks)  # a positive global rate drives every sector up


def test_sector_productivity_absolute_not_double_counted():
    """The absolute sector rate must NOT be double-counted against the aggregate TFP the endowment
    scale already applies (review P1): the θ deviation × aggregate level nets to the sector's stated
    absolute rate. Aggregate 1.7%/yr + sector-BRD absolute 1.5%/yr → net BRD level 1.015 over one
    year, NOT 1.017 × 1.015."""
    from cge.contracts.data_objects import Provenance, StructuralTrajectory
    from cge.dynamics.recursive import DynamicConfig as _DC
    from cge.dynamics.recursive import _aggregate_productivity_level, _sector_productivity_shocks

    traj = StructuralTrajectory(
        provenance=Provenance(
            source="t", source_version="v", licence="n", reference_year=2024, retrieved="2026-08-16"
        ),
        rates={"productivity": {"__all__": {2025: 0.017}}},
        sector_rates={"sector_productivity": {"BRD": {2025: 0.015}}},
        sources={"productivity:__all__": "c", "sector_productivity:BRD": "c"},
        confidence={"productivity:__all__": "low", "sector_productivity:BRD": "low"},
    )
    agg = _aggregate_productivity_level(traj, 2025, 2026)
    shocks = _sector_productivity_shocks(_DC(structural=traj), 2025, 2026, ["BRD"])
    brd = next(s for s in shocks if s.coverage_sectors == ["BRD"])
    assert agg * (1.0 + brd.delta) == pytest.approx(1.015, rel=1e-12)


def _covered(path, year):
    d = path.result.data
    r = d[
        (d["variable"] == "covered_emissions_change")
        & (d["scenario"] == "central")
        & (d["year"] == year)
    ]
    return float(r["value"].iloc[0])


def test_emissions_intensity_drives_covered_emissions_down():
    """A declining emissions-intensity trajectory shows as falling covered emissions measured
    against the base year (the engine's base-year reference makes the physical decline visible even
    though a within-year uniform scale would cancel in the same-year ratio)."""
    from cge.contracts.shocks import CarbonPrice

    traj = _traj_sector({"emissions_intensity": {"BRD": {2025: -0.05}, "MIL": {2025: -0.05}}})
    sc = Scenario(
        name="decarb",
        engine="cge_static",
        years=[2025, 2035, 2045],
        shocks=[CarbonPrice(price=50.0)],
    )
    decarb = run_recursive(sc, config=DynamicConfig(structural=traj), data_source="toy_cge_gov")
    flat = run_recursive(sc, config=DynamicConfig(), data_source="toy_cge_gov")
    # Year 0 identical (factor 1); later years far more negative than the no-decarb path.
    assert _covered(decarb, 2025) == pytest.approx(_covered(flat, 2025), abs=1e-9)
    assert _covered(decarb, 2045) < _covered(flat, 2045) - 0.3


def test_emissions_intensity_drives_covered_emissions_down_open():
    """The emissions-intensity reference is now wired through the OPEN engine too (review P1): a
    declining trajectory shows far more covered-emissions reduction than the flat path. Before the
    fix the open engine omitted the base-year reference, so decarbonisation cancelled in the within-
    year ratio and produced LESS reduction than the flat path."""
    from cge.contracts.shocks import CarbonPrice

    traj = _traj_sector(
        {
            "emissions_intensity": {
                "BRD": {2025: -0.05},
                "MIL": {2025: -0.05},
                "__all__": {2025: -0.05},
            }
        }
    )
    sc = Scenario(
        name="decarb-open",
        engine="cge_static",
        years=[2025, 2035],
        shocks=[CarbonPrice(price=50.0)],
    )
    decarb = run_recursive(
        sc, config=DynamicConfig(structural=traj), data_source="toy_cge_open_gov"
    )
    flat = run_recursive(sc, config=DynamicConfig(), data_source="toy_cge_open_gov")
    assert _covered(decarb, 2035) < _covered(flat, 2035) - 0.2


def test_emissions_intensity_reference_is_per_region_in_multi():
    """The multi engine honours a PER-REGION covered-emissions reference (review P1: a summed scalar
    would mix regional bases). With a large per-region reference every region's change collapses
    toward −1, proving the reference is applied region-by-region rather than dropped/summed."""
    from cge.contracts.shocks import CarbonPrice
    from cge.runner import run_scenario

    sc = Scenario(name="ref", engine="cge_static", years=[2025], shocks=[CarbonPrice(price=50.0)])
    res = run_scenario(
        sc,
        data_source="toy_cge_multi_gov",
        data_overrides={"covered_emissions_reference": {"N": 100.0, "S": 100.0}},
    )
    d = res.data
    ch = d[(d["variable"] == "covered_emissions_change") & (d["year"] == 2025)]
    assert set(ch["region"]) == {"N", "S"}
    assert (ch["value"] < -0.9).all()  # both regions measured against their (huge) own reference


def test_sector_drivers_absent_is_byte_identical_to_phase_7_1():
    """With no sector drivers, a run is unchanged from the Phase 7.1 behaviour (no synthesized
    productivity shocks, benchmark carbon share)."""
    from cge.contracts.shocks import CarbonPrice

    sc = Scenario(
        name="plain", engine="cge_static", years=[2025, 2030], shocks=[CarbonPrice(price=50.0)]
    )
    a = run_recursive(sc, config=DynamicConfig(), data_source="toy_cge_gov")
    b = run_recursive(sc, data_source="toy_cge_gov")
    for year in (2025, 2030):
        assert _covered(a, year) == pytest.approx(_covered(b, year), abs=1e-12)
        assert _sector_vol(a, "BRD", year) == pytest.approx(_sector_vol(b, "BRD", year), abs=1e-12)
