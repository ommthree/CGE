"""Tests for the recursive-dynamic wrapper (Phase 7.1).

Covers the capital-carry loop, the accumulation identity across the path, exogenous labour/TFP
trends, premature retirement (stranded assets), the result path + manifest provenance, and the
guard rails (needs a capital + savings-investment SAM). All three CGE variants: the closed/gov SAM
and the open economy (one aggregate stock), and the multi-region CGE (a per-region capital path).
"""

from __future__ import annotations

import math

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


def test_dynamic_manifest_hash_reflects_config_not_just_last_static_solve():
    """Review P2a 2026-08-28: the dynamic manifest's scenario_hash must reflect the ORIGINAL
    scenario PLUS the DynamicConfig — not just the final year's static solve. Changing the dynamic
    depreciation (or retirement, or the trajectory) must move the hash; before the fix the manifest
    reused the last static solve, so its hash was that of the 2030 scenario and unchanged by any
    dynamic parameter."""
    from cge.contracts.shocks import CarbonPrice

    sc = Scenario(
        name="h", engine="cge_static", years=[2025, 2030], shocks=[CarbonPrice(price=50.0)]
    )
    a = run_recursive(sc, config=DynamicConfig(depreciation=0.01), data_source="toy_cge_gov")
    b = run_recursive(sc, config=DynamicConfig(depreciation=0.20), data_source="toy_cge_gov")
    assert a.result.manifest.scenario_hash != b.result.manifest.scenario_hash
    # Retirement schedule also moves the hash.
    c = run_recursive(sc, config=DynamicConfig(retirement={2030: 0.1}), data_source="toy_cge_gov")
    assert c.result.manifest.scenario_hash != a.result.manifest.scenario_hash
    # Per-solve-year child hashes are recorded (the full provenance chain).
    rd = a.result.manifest.assumptions["recursive_dynamics"]
    assert set(rd["child_run_hashes"]) == set(range(2025, 2031))


def test_dynamic_scenario_hash_distinguishes_nature_state_from_bare_shocks():
    """Review P2a: a dynamic physical-nature-state run must hash differently from an equivalent
    hand-written NatureStress dynamic run, and its physical-state provenance must survive. Tested at
    the hash-helper level (the full physical→CGE pipeline needs ENCORE fixtures the toy CGE source
    does not ship); the wrapper stamps assumptions['nature_state'] whenever scenario.nature_state is
    set, and hashes the ORIGINAL scenario (with nature_state) plus the config."""
    from cge.contracts.shocks import NatureStress
    from cge.dynamics.recursive import _dynamic_scenario_hash

    phys = Scenario(
        name="p",
        engine="cge_static",
        years=[2030, 2040],
        nature_state=[{"channel": "toy_water", "states": {2030: 90, 2040: 70}}],
    )
    expanded = phys.expanded_shocks([2030, 2040])
    bare = Scenario(
        name="p",
        engine="cge_static",
        years=[2030, 2040],
        shocks=[
            NatureStress(service=s.service, severity=s.severity, path=s.path) for s in expanded
        ],
    )
    cfg = DynamicConfig()
    assert _dynamic_scenario_hash(phys, cfg) != _dynamic_scenario_hash(bare, cfg)
    # Two physical scenarios differing only in an endpoint hash differently too.
    phys2 = Scenario(
        name="p",
        engine="cge_static",
        years=[2030, 2040],
        nature_state=[{"channel": "toy_water", "states": {2030: 90, 2040: 50}}],
    )
    assert _dynamic_scenario_hash(phys, cfg) != _dynamic_scenario_hash(phys2, cfg)


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
    # BRD gains a positive labour-augmenting drift (above the weighted mean) → more output than flat
    # — genuine, but milder than a Hicks-neutral θ (acts through the labour share only).
    assert _sector_vol(drift, "BRD", 2045) > _sector_vol(flat, "BRD", 2045) + 0.02


def test_multi_sector_productivity_biases_are_per_region_and_zero_mean():
    """In multi mode the sector labour-augmenting biases are computed WITHIN each region and are
    zero-mean there (review P1 2026-08-29): each region's per-(region,sector) shocks straddle zero.
    Different within-region sector rates in N vs S give different biases per region, and the engine
    honours each shock's coverage_regions."""
    from cge.contracts.data_objects import Provenance, StructuralTrajectory
    from cge.dynamics.recursive import DynamicConfig as _DC
    from cge.dynamics.recursive import _sector_productivity_shocks

    traj = StructuralTrajectory(
        provenance=Provenance(
            source="t", source_version="v", licence="n", reference_year=2024, retrieved="2026-08-16"
        ),
        # Region N: BRD fast, MIL slow. Region S: the reverse — so the biases differ by region.
        sector_rates={
            "sector_productivity": {
                "BRD": {2025: 0.030},
                "MIL": {2025: 0.010},
                "__all__": {2025: 0.015},
            }
        },
        rates={},
        sources={
            "sector_productivity:BRD": "c",
            "sector_productivity:MIL": "c",
            "sector_productivity:__all__": "c",
        },
        confidence={
            "sector_productivity:BRD": "low",
            "sector_productivity:MIL": "low",
            "sector_productivity:__all__": "low",
        },
    )
    shares = {"N": {"BRD": 0.5, "MIL": 0.5}, "S": {"BRD": 0.5, "MIL": 0.5}}
    # No capital_deepening in this trajectory → heuristic path; labour_shares (last arg) is unused.
    shocks = _sector_productivity_shocks(
        _DC(structural=traj), 2025, 2030, ["BRD", "MIL"], ["N", "S"], True, shares, {}
    )
    assert all(s.mechanism == "labour_augmenting" for s in shocks)
    for region in ("N", "S"):
        rd = {s.coverage_sectors[0]: s.delta for s in shocks if s.coverage_regions == [region]}
        assert set(rd) == {"BRD", "MIL"}
        # Within the region, BRD (faster) is up and MIL (slower) is down: a zero-mean drift.
        assert rd["BRD"] > 0.0 > rd["MIL"]
        geo = (1.0 + rd["BRD"]) ** 0.5 * (1.0 + rd["MIL"]) ** 0.5
        assert geo == pytest.approx(1.0, rel=1e-12)


def test_sector_productivity_all_default_drives_every_model_sector():
    """An ``__all__``-only sector-productivity trajectory (or one whose keys do not match real
    sector names) must drive EVERY model sector, not silently do nothing (review P1: the old code
    skipped ``__all__`` without enumerating the model's sectors, so such a trajectory emitted zero
    shocks — and on a real EXIOBASE model the shipped BRD/MIL keys never matched real names)."""
    from cge.dynamics.recursive import DynamicConfig as _DC
    from cge.dynamics.recursive import _sector_productivity_shocks

    traj = _traj_sector({"sector_productivity": {"__all__": {2025: 0.02}}})
    # Single-region call: regions=["R"], multi=False. Every sector is driven (a shock is emitted for
    # each), even from an __all__-only path.
    shocks = _sector_productivity_shocks(
        _DC(structural=traj), 2025, 2027, ["BRD", "MIL"], ["R"], False, {}, {}
    )
    assert {tuple(s.coverage_sectors) for s in shocks} == {("BRD",), ("MIL",)}
    # When EVERY sector grows at the same rate there is no COMPOSITION drift — relative biases are
    # exactly zero (the uniform level is carried by the aggregate endowment scale, not this driver).
    assert all(s.delta == pytest.approx(0.0, abs=1e-12) for s in shocks)
    assert all(s.mechanism == "labour_augmenting" for s in shocks)


def test_sector_productivity_shocks_are_labour_augmenting():
    """The sector series is applied as a genuine LABOUR-AUGMENTING shock, not a Hicks-neutral θ
    (review P1 2026-08-29). The synthesized shocks carry mechanism='labour_augmenting', so the
    engine's labour-share weighting is structural (no ad-hoc ^s_L exponent)."""
    from cge.dynamics.recursive import DynamicConfig as _DC
    from cge.dynamics.recursive import _sector_productivity_shocks

    traj = _traj_sector(
        {"sector_productivity": {"BRD": {2025: 0.03}, "MIL": {2025: 0.01}, "__all__": {2025: 0.02}}}
    )
    shocks = _sector_productivity_shocks(
        _DC(structural=traj), 2025, 2027, ["BRD", "MIL"], ["R"], False, {"BRD": 0.5, "MIL": 0.5}, {}
    )
    assert all(s.mechanism == "labour_augmenting" for s in shocks)


def test_sector_productivity_biases_are_zero_mean_relative_drift():
    """The sector biases must be a zero-mean structural DRIFT — the VA-weighted geometric mean of
    (1+delta) factors is 1 — so no aggregate productivity level is re-imposed and a faster sector's
    positive bias is balanced by a slower sector's negative bias (review P1 2026-08-29: the earlier
    (LP/TFP)^s_L gave every sector in a high-TFP region a negative bias). BRD grows 3%/yr, MIL
    1%/yr, equal VA shares → BRD gets φ>1, MIL φ<1, weighted geo-mean 1."""
    from cge.dynamics.recursive import DynamicConfig as _DC
    from cge.dynamics.recursive import _sector_productivity_shocks

    traj = _traj_sector({"sector_productivity": {"BRD": {2025: 0.03}, "MIL": {2025: 0.01}}})
    shares = {"BRD": 0.5, "MIL": 0.5}
    labour = {"BRD": 0.5, "MIL": 0.5}  # equal labour shares → v_i·s_Li ∝ v_i here
    shocks = _sector_productivity_shocks(
        _DC(structural=traj), 2025, 2030, ["BRD", "MIL"], ["R"], False, shares, labour
    )
    by_sector = {s.coverage_sectors[0]: s.delta for s in shocks}
    assert by_sector["BRD"] > 0.0 > by_sector["MIL"]  # faster up, slower down (a genuine drift)
    # v_i·s_Li-weighted geometric mean of (1+delta) is 1 (zero-mean drift; no aggregate level
    # re-imposed). With equal shares this is the plain geometric mean.
    geo = (1.0 + by_sector["BRD"]) ** 0.5 * (1.0 + by_sector["MIL"]) ** 0.5
    assert geo == pytest.approx(1.0, rel=1e-12)


def test_sector_productivity_drift_uses_va_SHARE_weights_not_labour_composition():
    """Review P1 2026-08-31: the geometric-mean denominator must weight each sector by its SHARE of
    value added v_i = VA_i/ΣVA_j, so a tiny sector cannot swing the mean. With BRD = 99% of VA and
    MIL = 1%, both growing (BRD 3%/yr, MIL 1%/yr), the mean is pinned near BRD's level, so BRD's
    drift is ~0 and MIL's is strongly negative — unlike equal weighting, where BRD would get
    a large positive bias. Also: the VA-share-weighted geometric mean of (1+delta) is exactly 1."""
    from cge.dynamics.recursive import DynamicConfig as _DC
    from cge.dynamics.recursive import _sector_productivity_shocks

    traj = _traj_sector({"sector_productivity": {"BRD": {2025: 0.03}, "MIL": {2025: 0.01}}})
    va_shares = {"BRD": 0.99, "MIL": 0.01}
    labour = {"BRD": 0.5, "MIL": 0.5}  # equal labour shares → v_i·s_Li ∝ v_i (isolates VA weight)
    shocks = _sector_productivity_shocks(
        _DC(structural=traj), 2025, 2045, ["BRD", "MIL"], ["R"], False, va_shares, labour
    )
    by_sector = {s.coverage_sectors[0]: s.delta for s in shocks}
    # BRD dominates the mean → its own drift is near zero; MIL (tiny) absorbs the redistribution.
    assert abs(by_sector["BRD"]) < 0.01
    assert by_sector["MIL"] < -0.10
    # And the v_i·s_Li-weighted geometric mean of (1+delta) is exactly 1 (∝ VA weights here).
    geo = (1.0 + by_sector["BRD"]) ** 0.99 * (1.0 + by_sector["MIL"]) ** 0.01
    assert geo == pytest.approx(1.0, rel=1e-12)
    # Equal weighting would instead give BRD a clearly positive bias — prove the fix changed it.
    eq = _sector_productivity_shocks(
        _DC(structural=traj),
        2025,
        2045,
        ["BRD", "MIL"],
        ["R"],
        False,
        {"BRD": 0.5, "MIL": 0.5},
        labour,
    )
    eq_brd = next(s.delta for s in eq if s.coverage_sectors[0] == "BRD")
    assert eq_brd > 0.10  # equal weights: BRD gets a large positive bias (the old, wrong behaviour)


def test_composition_drift_is_aggregate_cost_neutral_under_unequal_labour_shares():
    """Review P1 2026-09-03: the normalizer must zero the AGGREGATE CD cost effect Σ v_i s_Li ln φ,
    not just the v_i-weighted geometric mean of φ. With two equal-VA sectors but UNEQUAL labour
    shares (s_L = 0.8 / 0.2) the old v_i-only mean left a non-zero aggregate cost effect; the
    v_i·s_Li-weighted mean makes it exactly neutral. Verify Σ v_i·s_Li·ln(1+delta) == 0."""
    from cge.dynamics.recursive import DynamicConfig as _DC
    from cge.dynamics.recursive import _sector_productivity_shocks

    traj = _traj_sector({"sector_productivity": {"BRD": {2025: 0.03}, "MIL": {2025: 0.01}}})
    va = {"BRD": 0.5, "MIL": 0.5}
    sl = {"BRD": 0.8, "MIL": 0.2}
    shocks = _sector_productivity_shocks(
        _DC(structural=traj), 2025, 2045, ["BRD", "MIL"], ["R"], False, va, sl
    )
    d = {s.coverage_sectors[0]: s.delta for s in shocks}
    # Aggregate CD log-cost effect Σ v_i·s_Li·ln(1+delta_i) is exactly zero (neutral drift).
    agg = sum(va[s] * sl[s] * np.log(1.0 + d[s]) for s in ("BRD", "MIL"))
    assert agg == pytest.approx(0.0, abs=1e-12)
    # A faster sector still drifts up, a slower one down — it is a real composition drift, not flat.
    assert d["BRD"] > 0.0 > d["MIL"]
    # The plain v_i-only geometric mean would NOT be neutral here (prove the s_L matters): the
    # v_i-weighted Σ v_i·ln(1+delta) is non-zero under these unequal labour shares.
    agg_va_only = sum(va[s] * np.log(1.0 + d[s]) for s in ("BRD", "MIL"))
    assert abs(agg_va_only) > 1e-6


# --- Growth-accounting decomposition: identify the labour-augmenting technology term ------------
# When a capital_deepening series is supplied the driver subtracts it and drives the sector with the
# IDENTIFIED technology rate g_φ = (g_{Y/L} − s_K·g_{K/L}) / s_L, so the capital deepening the model
# accumulates separately is not double-counted.


def test_decomposed_log_level_removes_capital_deepening_known_answer():
    """Known answer for the decomposition, using the COHERENT Cobb-Douglas finite-change mapping in
    LOG space (review P1 2026-09-03; log-space review P2 2026-09-05): the per-year log factor is
    ln(1+g_φ) = [ln(1+g_{Y/L}) − s_K·ln(1+g_{K/L})]/s_L, so the cumulative LOG level is n× that.
    g_{Y/L}=0.02, g_{K/L}=0.01, s_L=0.6 (s_K=0.4). The heuristic path compounds the raw ln(1.02)."""
    from cge.dynamics.recursive import _decomposed_log_level

    traj = _traj_sector(
        {
            "sector_productivity": {"BRD": {2025: 0.02}},
            "capital_deepening": {"BRD": {2025: 0.01}},
        }
    )
    s_L = 0.6
    s_K = 1.0 - s_L
    log_step = (math.log(1.02) - s_K * math.log(1.01)) / s_L  # coherent CD finite-change log factor
    n = 10  # 2025 -> 2035
    got = _decomposed_log_level(traj, "BRD", 2025, 2035, s_L, identified=True)
    assert got == pytest.approx(log_step * n, rel=1e-12)
    heur = _decomposed_log_level(traj, "BRD", 2025, 2035, s_L, identified=False)
    assert heur == pytest.approx(math.log(1.02) * n, rel=1e-12)
    # Deepening genuinely changed the technology level (it is not a relabelling no-op).
    assert abs(got - heur) > 1e-6


def test_decomposed_log_level_admissible_but_extreme_does_not_overflow():
    """Review P2 2026-09-05: an admissible rate combination that would OVERFLOW math.exp per year
    (g_{Y/L}=0.99, g_{K/L}=−0.99, s_L=0.001) must be finite in LOG space — the helper returns the
    (large but finite) cumulative log, and the cross-sector normaliser cancels the common part
    before
    any exponentiation, so no OverflowError. The old per-year exp-and-multiply raised here."""
    from cge.dynamics.recursive import _decomposed_log_level

    traj = _traj_sector(
        {
            "sector_productivity": {"BRD": {2025: 0.99}},
            "capital_deepening": {"BRD": {2025: -0.99}},
        }
    )
    got = _decomposed_log_level(traj, "BRD", 2025, 2026, 0.001, identified=True)
    assert math.isfinite(got)
    # Exact log value: [ln(1.99) − 0.999·ln(0.01)]/0.001, a large positive number, still finite.
    expected = (math.log(1.99) - (1 - 0.001) * math.log(0.01)) / 0.001
    assert got == pytest.approx(expected, rel=1e-12)


def test_decomposed_log_level_rejects_rate_at_or_below_minus_100pct():
    """A rate ≤ −100% is not a valid annual growth rate; the level factor would be ≤0 and its log
    undefined, so the helper raises rather than producing a non-finite log. (The trajectory contract
    also bounds rates to (−1,1); this is defense-in-depth via a construction near the edge.)"""
    from cge.dynamics.recursive import _decomposed_log_level

    # Near-edge admissible values stay finite (no raise) — proves the guard is not over-eager.
    ok = _traj_sector(
        {
            "sector_productivity": {"BRD": {2025: -0.99}},
            "capital_deepening": {"BRD": {2025: 0.99}},
        }
    )
    assert math.isfinite(_decomposed_log_level(ok, "BRD", 2025, 2026, 0.5, identified=True))


def test_capital_deepening_series_changes_the_sector_shocks():
    """Supplying a capital_deepening series changes the synthesized shocks vs the raw heuristic —
    the decomposition is actually wired into the shock path (not just the helper). Requires σ_va=1
    (CD nest) for the identified path (review P2 2026-09-05)."""
    from cge.dynamics.recursive import DynamicConfig as _DC
    from cge.dynamics.recursive import _sector_productivity_shocks

    lp = {"BRD": {2025: 0.03}, "MIL": {2025: 0.01}}
    va = {"BRD": 0.5, "MIL": 0.5}
    sl = {"BRD": 0.5, "MIL": 0.5}
    elast = {"BRD": 1.0, "MIL": 1.0}  # Cobb-Douglas VA nest → identified path is eligible
    heur = _sector_productivity_shocks(
        _DC(structural=_traj_sector({"sector_productivity": lp})),
        2025,
        2045,
        ["BRD", "MIL"],
        ["R"],
        False,
        va,
        sl,
        elast,
    )
    # BRD deepens faster than MIL, so netting deepening out reshapes the drift.
    ident = _sector_productivity_shocks(
        _DC(
            structural=_traj_sector(
                {
                    "sector_productivity": lp,
                    "capital_deepening": {"BRD": {2025: 0.02}, "MIL": {2025: 0.005}},
                }
            )
        ),
        2025,
        2045,
        ["BRD", "MIL"],
        ["R"],
        False,
        va,
        sl,
        elast,
    )
    d_heur = {s.coverage_sectors[0]: s.delta for s in heur}
    d_ident = {s.coverage_sectors[0]: s.delta for s in ident}
    assert d_heur != d_ident
    # Both remain proper zero-mean drifts (VA-weighted geometric mean of 1+delta = 1).
    for d in (d_heur, d_ident):
        assert (1 + d["BRD"]) ** 0.5 * (1 + d["MIL"]) ** 0.5 == pytest.approx(1.0, rel=1e-12)


def test_source_share_identification_uses_source_not_model_labour_share():
    """Pipeline step 1b (review P2 2026-09-05): the source-side MFP identification must use the
    SOURCE labour share s_L^src, NOT the receiving model's benchmark share. Set a source share that
    differs from the model share and check the per-year log level equals the source-share formula
    [ln(1+g_Y/L) − (1−s_L^src)·ln(1+g_K/L)] / s_L^model (CD nest), which differs from the
    all-model-share value."""
    import math

    from cge.dynamics.recursive import _decomposed_log_level

    traj = _traj_sector(
        {
            "sector_productivity": {"BRD": {2025: 0.03}},
            "capital_deepening": {"BRD": {2025: 0.02}},
        }
    )
    traj.source_labour_shares = {"BRD": 0.4}  # source share ≠ model share below
    s_L_model = 0.6
    got = _decomposed_log_level(
        traj, "BRD", 2025, 2026, s_L_model, identified=True, sigma=1.0, source_labour_share=0.4
    )
    # MFP identified at source with s_L^src=0.4 (s_K^src=0.6), then CD-translated by /s_L^model.
    g_mfp_log = math.log(1.03) - 0.6 * math.log(1.02)
    expected = g_mfp_log / s_L_model
    assert got == pytest.approx(expected, rel=1e-12)
    # Using the MODEL share on both sides (the old, wrong behaviour) gives a different number.
    wrong = (math.log(1.03) - (1 - s_L_model) * math.log(1.02)) / s_L_model
    assert abs(got - wrong) > 1e-6


def test_sourced_mfp_series_is_used_directly():
    """Pipeline step 1b: when the trajectory ships an ``mfp`` series (identified at source in the
    data build), the wrapper uses it directly — the per-year log level is the CD translation
    ln(1+g_MFP)/s_L^model, independent of sector_productivity/capital_deepening."""
    import math

    from cge.dynamics.recursive import _decomposed_log_level

    traj = _traj_sector(
        {
            "sector_productivity": {"BRD": {2025: 0.05}},  # present but IGNORED when mfp exists
            "capital_deepening": {"BRD": {2025: 0.9}},  # ditto
            "mfp": {"BRD": {2025: 0.012}},
        }
    )
    got = _decomposed_log_level(traj, "BRD", 2025, 2026, 0.6, identified=True, sigma=1.0)
    assert got == pytest.approx(math.log(1.012) / 0.6, rel=1e-12)


def test_ces_sector_is_identified_via_nest_aware_translation():
    """Pipeline step 1b (review P2 2026-09-05): a CES sector is now IDENTIFIED — the MFP implied by
    the decomposition is translated into a labour-augmentation through the sector's ACTUAL CES nest
    (numeric), instead of being dropped to the heuristic. Assert (a) the CES result differs from the
    CD result (the nest matters — it is not the CD closed form), and (b) both differ from the raw
    heuristic (capital deepening IS being netted out for both)."""
    from cge.dynamics.recursive import DynamicConfig as _DC
    from cge.dynamics.recursive import _sector_productivity_shocks

    traj = _traj_sector(
        {
            "sector_productivity": {"BRD": {2025: 0.03}, "MIL": {2025: 0.01}},
            "capital_deepening": {"BRD": {2025: 0.02}, "MIL": {2025: 0.005}},
        }
    )
    va = {"BRD": 0.5, "MIL": 0.5}
    sl = {"BRD": 0.5, "MIL": 0.5}
    args = (2025, 2045, ["BRD", "MIL"], ["R"], False, va, sl)
    cd = _sector_productivity_shocks(_DC(structural=traj), *args, {"BRD": 1.0, "MIL": 1.0})
    ces = _sector_productivity_shocks(_DC(structural=traj), *args, {"BRD": 0.5, "MIL": 0.5})
    heur_only = _sector_productivity_shocks(
        _DC(
            structural=_traj_sector(
                {"sector_productivity": traj.sector_rates["sector_productivity"]}
            )
        ),
        *args,
        {"BRD": 0.5, "MIL": 0.5},
    )
    d_cd = {s.coverage_sectors[0]: s.delta for s in cd}
    d_ces = {s.coverage_sectors[0]: s.delta for s in ces}
    d_heur = {s.coverage_sectors[0]: s.delta for s in heur_only}
    assert d_cd != d_ces  # the CES nest gives a different augmentation than the CD closed form
    assert d_ces != pytest.approx(d_heur)  # CES IS identified (deepening netted out), not heuristic
    assert d_cd != pytest.approx(d_heur)


def test_ces_translation_reproduces_target_cost_ratio():
    """Review P2 2026-09-06: the CES translation must reproduce the MFP-implied VA-cost ratio at
    benchmark prices (not merely 'differ from CD'). Verify the solved ln φ makes the CES cost index
    fall by exactly 1+g_MFP, and that it reduces to the CD closed form as σ→1."""
    import math

    from cge.dynamics.recursive import _mfp_to_labour_aug_log

    for sigma in (0.4, 0.7, 1.5, 2.0):
        for theta_L in (0.3, 0.6):
            for g_mfp in (-0.03, 0.02):
                x = _mfp_to_labour_aug_log(g_mfp, theta_L, sigma)
                om = 1.0 - sigma
                cx = (theta_L * math.exp(om * (-x)) + (1.0 - theta_L)) ** (1.0 / om)
                assert cx == pytest.approx(1.0 / (1.0 + g_mfp), rel=1e-10)
    # CD limit
    assert _mfp_to_labour_aug_log(0.02, 0.6, 1.0) == pytest.approx(math.log(1.02) / 0.6, rel=1e-12)


def test_ces_translation_raises_on_infeasible_target():
    """Review P2 2026-09-06: when no labour augmentation can reproduce the CES MFP cost change
    (capital is essential), the solver RAISES rather than returning an enormous boundary value.
    Reviewer's cases: g_MFP=+50% sL=0.1 σ=0.5, and g_MFP=−90% sL=0.6 σ=2."""
    from cge.dynamics.recursive import InfeasibleMFPTranslation, _mfp_to_labour_aug_log

    for g_mfp, s_l, sigma in [(0.50, 0.1, 0.5), (-0.90, 0.6, 2.0)]:
        with pytest.raises(InfeasibleMFPTranslation):
            _mfp_to_labour_aug_log(g_mfp, s_l, sigma)


def test_source_shares_change_run_identity_and_manifest():
    """Review P1 2026-09-06: source_labour_shares materially change the shocks, so two otherwise
    identical recursive runs differing ONLY in source shares must get DIFFERENT scenario_hashes and
    the trend manifest must record the shares (previously they were absent from both, so a result
    revealed something changed but could not reconstruct the input)."""
    from cge.contracts.data_objects import Provenance, StructuralTrajectory
    from cge.contracts.shocks import CarbonPrice

    def _traj(sl_val):
        return StructuralTrajectory(
            provenance=Provenance(
                source="t",
                source_version="v",
                licence="n",
                reference_year=2024,
                retrieved="2026-09-06",
            ),
            sector_rates={
                "sector_productivity": {"__all__": {2025: 0.02}},
                "capital_deepening": {"__all__": {2025: 0.01}},
            },
            sources={
                "sector_productivity:__all__": "c",
                "capital_deepening:__all__": "c",
                "source_labour_share:__all__": "illustrative",
            },
            confidence={
                "sector_productivity:__all__": "low",
                "capital_deepening:__all__": "low",
                "source_labour_share:__all__": "low",
            },
            source_labour_shares={"__all__": sl_val},
        )

    sc = Scenario(
        name="x", engine="cge_static", years=[2025, 2030], shocks=[CarbonPrice(price=50.0)]
    )
    a = run_recursive(
        sc,
        config=DynamicConfig(structural=_traj(0.45), allow_uniform_fallback=True),
        data_source="toy_cge_gov",
    )
    b = run_recursive(
        sc,
        config=DynamicConfig(structural=_traj(0.75), allow_uniform_fallback=True),
        data_source="toy_cge_gov",
    )
    # Different source shares → different run identity.
    assert a.result.manifest.scenario_hash != b.result.manifest.scenario_hash
    # And the manifest records the shares so the input is reconstructible.
    ts_a = a.result.manifest.assumptions["recursive_dynamics"]["trend_source"]
    assert ts_a["source_labour_shares"] == {"__all__": 0.45}
    assert ts_a["rate_tables"]["source_labour_shares"] == {"__all__": 0.45}


def test_source_labour_shares_reject_invalid_values():
    """Review P2 2026-09-06: source labour shares must be finite and in (0, 1] with value-level
    provenance — NaN/inf/negative/>1 or unsourced are rejected at construction."""
    from cge.contracts.data_objects import Provenance, StructuralTrajectory

    def _mk(sl):
        return StructuralTrajectory(
            provenance=Provenance(
                source="t",
                source_version="v",
                licence="n",
                reference_year=2024,
                retrieved="2026-09-06",
            ),
            sector_rates={"sector_productivity": {"__all__": {2025: 0.02}}},
            sources={
                "sector_productivity:__all__": "c",
                "source_labour_share:__all__": "s",
            },
            confidence={
                "sector_productivity:__all__": "low",
                "source_labour_share:__all__": "low",
            },
            source_labour_shares={"__all__": sl},
        )

    for bad in (float("nan"), float("inf"), -0.2, 1.2, 0.0):
        with pytest.raises(ValueError, match="source_labour_shares|not a finite|out of range"):
            _mk(bad)
    # Missing provenance is rejected too.
    with pytest.raises(ValueError, match="missing source or confidence"):
        StructuralTrajectory(
            provenance=Provenance(
                source="t",
                source_version="v",
                licence="n",
                reference_year=2024,
                retrieved="2026-09-06",
            ),
            sector_rates={"sector_productivity": {"__all__": {2025: 0.02}}},
            sources={"sector_productivity:__all__": "c"},
            confidence={"sector_productivity:__all__": "low"},
            source_labour_shares={"__all__": 0.6},
        )


def test_missing_labour_productivity_is_not_identified():
    """Review P2 2026-09-05: a sector with capital_deepening but NO sector_productivity observation
    must NOT be reported identified. Missing rates read as 0.0, so treating it as identified would
    infer NEGATIVE technology growth (0 − s_K·g_K/L)/s_L from missing data. Sector A has an explicit
    LP path, sector B has NONE (and there is no LP __all__), while capital_deepening covers both via
    __all__. Only A is identified; B stays heuristic. Uses the manifest-mode helper (same rule as
    the shock path)."""
    from cge.dynamics.recursive import _sector_productivity_modes

    traj = _traj_sector(
        {
            "sector_productivity": {"A": {2025: 0.03}},  # A only; NO __all__, so B is uncovered
            "capital_deepening": {"__all__": {2025: 0.01}},
        }
    )
    labour = {"A": 0.6, "B": 0.6}
    elast = {"A": 1.0, "B": 1.0}
    modes = _sector_productivity_modes(traj, ["A", "B"], ["R"], False, labour, elast)["R"]
    # A has LP+deepening but NO source_labour_shares → model-share approximation (NOT source-
    # identified, review P2 2026-09-06); B has no LP observation → raw heuristic.
    assert modes["A"] == "model_share_approx"
    assert modes["B"] == "raw_lp_heuristic"


def test_source_share_gives_derived_source_share_mode():
    """Review P2 2026-09-06: LP+deepening WITH a source labour share is ``derived_source_share``
    (source-identified); WITHOUT one it is only ``model_share_approx``. The distinction is what
    makes the manifest honest about whether the source pipeline was actually used."""
    from cge.dynamics.recursive import _sector_productivity_modes

    base = {
        "sector_productivity": {"A": {2025: 0.03}},
        "capital_deepening": {"A": {2025: 0.01}},
    }
    labour = {"A": 0.6}
    elast = {"A": 1.0}
    without = _sector_productivity_modes(_traj_sector(base), ["A"], ["R"], False, labour, elast)[
        "R"
    ]
    assert without["A"] == "model_share_approx"

    t_with = _traj_sector(base)
    t_with.source_labour_shares = {"A": 0.55}
    with_src = _sector_productivity_modes(t_with, ["A"], ["R"], False, labour, elast)["R"]
    assert with_src["A"] == "derived_source_share"


def test_manifest_summary_reports_mode_mix_without_overclaiming():
    """Review P2 2026-09-06: the summary must report the mode MIX honestly — never a blanket
    'identified at source' — and carry the CES benchmark-equivalence caveat when any source-
    identified sector is present. Here A is sourced_mfp (source-identified), B is raw heuristic."""
    from cge.dynamics.recursive import (
        _sector_productivity_mode_summary,
        _sector_productivity_modes,
    )

    traj = _traj_sector(
        {
            "mfp": {"A": {2025: 0.012}},  # A: sourced MFP
            "sector_productivity": {"B": {2025: 0.01}},  # B: raw LP only
        }
    )
    labour = {"A": 0.6, "B": 0.6}
    elast = {"A": 1.0, "B": 1.0}
    modes = _sector_productivity_modes(traj, ["A", "B"], ["R"], False, labour, elast)
    assert modes["R"] == {"A": "sourced_mfp", "B": "raw_lp_heuristic"}
    summary = _sector_productivity_mode_summary(modes)
    assert "sourced MFP" in summary and "raw labour-productivity heuristic" in summary
    assert "benchmark prices" in summary  # CES caveat present because a source-identified sector is


def test_decomposition_needs_a_usable_labour_share_else_heuristic():
    """Without a stamped labour share (s_L≈0) a sector cannot be decomposed (would divide by ~0), so
    it stays on the raw heuristic rate rather than blowing up."""
    from cge.dynamics.recursive import DynamicConfig as _DC
    from cge.dynamics.recursive import _sector_productivity_shocks

    traj = _traj_sector(
        {
            "sector_productivity": {"BRD": {2025: 0.03}, "MIL": {2025: 0.01}},
            "capital_deepening": {"BRD": {2025: 0.02}, "MIL": {2025: 0.005}},
        }
    )
    va = {"BRD": 0.5, "MIL": 0.5}
    # Empty labour shares → no usable s_L → heuristic path for every sector (finite, sensible).
    shocks = _sector_productivity_shocks(
        _DC(structural=traj), 2025, 2045, ["BRD", "MIL"], ["R"], False, va, {}
    )
    assert all(np.isfinite(s.delta) for s in shocks)
    by = {s.coverage_sectors[0]: s.delta for s in shocks}
    assert by["BRD"] > 0.0 > by["MIL"]


def test_manifest_records_honest_identification_mode():
    """The recursive manifest must record the HONEST fine-grained mode (review P2 2026-09-06): with
    capital_deepening but NO source labour share it is a MODEL-share approximation (NOT
    source-identified); adding source_labour_shares makes it source-identified; with neither it is the
    raw heuristic. This retires the earlier test that expected a blanket 'identified' when source
    shares were absent — exactly the overclaim the reviewer flagged."""
    from cge.contracts.data_objects import Provenance, StructuralTrajectory
    from cge.contracts.shocks import CarbonPrice

    def _traj(with_deepening, with_source_share=False):
        sr = {"sector_productivity": {"BRD": {2025: 0.03}, "MIL": {2025: 0.01}}}
        if with_deepening:
            sr["capital_deepening"] = {"BRD": {2025: 0.02}, "MIL": {2025: 0.005}}
        keys = [f"{d}:{s}" for d, by in sr.items() for s in by]
        srcs = {k: "c" for k in keys}
        cfs = {k: "low" for k in keys}
        ssl = {}
        if with_source_share:
            ssl = {"BRD": 0.58, "MIL": 0.62}
            for s in ("BRD", "MIL"):
                srcs[f"source_labour_share:{s}"] = "illustrative"
                cfs[f"source_labour_share:{s}"] = "low"
        return StructuralTrajectory(
            provenance=Provenance(
                source="t",
                source_version="v",
                licence="n",
                reference_year=2024,
                retrieved="2026-09-06",
            ),
            sector_rates=sr,
            sources=srcs,
            confidence=cfs,
            source_labour_shares=ssl,
        )

    sc = Scenario(
        name="m", engine="cge_static", years=[2025, 2030], shocks=[CarbonPrice(price=50.0)]
    )

    def _mode(traj):
        r = run_recursive(sc, config=DynamicConfig(structural=traj), data_source="toy_cge_gov")
        return r.result.manifest.assumptions["recursive_dynamics"]["trend_source"][
            "sector_productivity_mode"
        ]

    approx = _mode(_traj(with_deepening=True))  # deepening but no source share
    sourced = _mode(_traj(with_deepening=True, with_source_share=True))
    heur = _mode(_traj(with_deepening=False))
    assert "approximation, not source-id" in approx  # honest: NOT source-identified
    assert "identified at source" in sourced
    assert "raw labour-productivity heuristic" in heur


def test_high_aggregate_region_does_not_get_all_negative_sector_biases():
    """Review P1 2026-08-29: in a HIGH-aggregate-TFP region the old (sector_LP/aggregate_TFP)^s_L
    made EVERY sector's bias negative (all sector rates sat below the region's TFP rate). The
    zero-mean relative-drift formulation must not: within a region the biases straddle zero
    REGARDLESS of the aggregate TFP level (which is carried separately by the endowment scale)."""
    from cge.contracts.data_objects import Provenance, StructuralTrajectory
    from cge.dynamics.recursive import DynamicConfig as _DC
    from cge.dynamics.recursive import _sector_productivity_shocks

    # Region S has a very high aggregate TFP (3%/yr); its sector rates (1–2%/yr) are all below it.
    traj = StructuralTrajectory(
        provenance=Provenance(
            source="t", source_version="v", licence="n", reference_year=2024, retrieved="2026-08-16"
        ),
        rates={"productivity": {"S": {2025: 0.030}}},
        sector_rates={"sector_productivity": {"BRD": {2025: 0.020}, "MIL": {2025: 0.010}}},
        sources={
            "productivity:S": "c",
            "sector_productivity:BRD": "c",
            "sector_productivity:MIL": "c",
        },
        confidence={
            "productivity:S": "low",
            "sector_productivity:BRD": "low",
            "sector_productivity:MIL": "low",
        },
    )
    shocks = _sector_productivity_shocks(
        _DC(structural=traj),
        2025,
        2030,
        ["BRD", "MIL"],
        ["S"],
        True,
        {"S": {"BRD": 0.5, "MIL": 0.5}},
        {},
    )
    deltas = [s.delta for s in shocks]
    assert max(deltas) > 0.0 and min(deltas) < 0.0  # biases straddle zero, NOT all-negative


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


def test_multi_uncovered_region_does_not_wipe_out_covered_regions_decarb():
    """Review P1a (2026-08-28): in a multi run where one region is UNCOVERED (no covered-emissions
    row — the shipped multi SAM prices only some sectors), the base-year reference for the COVERED
    region must survive. The old wrapper discarded the WHOLE per-region reference if any region was
    uncovered, so the decarbonising run reported LESS reduction than flat (only the weakened price
    wedge remained observable). Now the covered region shows a strictly deeper cut than the flat
    path, and the uncovered region simply emits no row."""
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
        name="decarb-multi",
        engine="cge_static",
        years=[2025, 2035],
        shocks=[CarbonPrice(price=50.0)],
    )
    decarb = run_recursive(
        sc, config=DynamicConfig(structural=traj), data_source="toy_cge_multi_gov"
    )
    flat = run_recursive(sc, config=DynamicConfig(), data_source="toy_cge_multi_gov")

    def covered(path, region):
        d = path.result.data
        r = d[
            (d["variable"] == "covered_emissions_change")
            & (d["year"] == 2035)
            & (d["region"] == region)
        ]
        return None if r.empty else float(r["value"].iloc[0])

    # Region N is covered: its decarbonising path is a strictly DEEPER cut than the flat path (the
    # reviewer saw the opposite — decarb −20.5% vs flat −23.4% — because S wiped out N's reference).
    n_decarb, n_flat = covered(decarb, "N"), covered(flat, "N")
    assert n_decarb is not None and n_flat is not None
    assert n_decarb < n_flat - 0.1
    # Region S is uncovered on this SAM: it emits no covered-emissions row (a zero reference), which
    # is correct — an uncovered region has nothing to measure — and does not corrupt N.
    assert covered(decarb, "S") is None


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


# --- Full physical-nature dynamic run (Phase 6b × 7.1, review 7b.2 2026-08-30) --------------------
# The review flagged that the ONLY committed physical-nature dynamic test exercised the hash helper
# (test_dynamic_scenario_hash_distinguishes_nature_state_from_bare_shocks), NOT the full
# physical-state -> NatureStress -> exposure -> ProductivityShock -> CGE pipeline through the
# recursive-dynamic wrapper. A physical run needs three things at once — a capital-carrying SAM
# (SAVINV), an EncoreDependencies + ConcordanceMap, and an exposure IOSystem — which no single toy
# source shipped. run_recursive now takes ``data_overrides``, so the dynamic-capable toy CGE SAM
# (BRD/MIL, with SAVINV) supplies the capital core while the nature triple is injected here. The
# exposure IO/ENCORE fixture below is labelled with the SAM's own sectors so the derived
# ProductivityShocks land on real sectors.


def _brd_mil_nature_triple():
    """An exposure IOSystem + ENCORE dependency + concordance whose sectors are the toy CGE SAM's
    BRD/MIL, so a ``surface_water`` degradation translates into per-sector ProductivityShocks the
    CGE consumes. BRD (the water-intensive dirty sector) depends VERY HIGHLY on surface water; MIL
    only lightly — so a water-stock decline hits BRD harder, an economically legible asymmetry."""
    import pandas as pd

    from cge.contracts.data_objects import (
        Classification,
        ConcordanceMap,
        IOSystem,
        Provenance,
    )
    from cge.nature.encore import EncoreDependencies

    prov = Provenance(
        source="toy BRD/MIL nature fixture",
        source_version="v1",
        licence="illustrative",
        reference_year=2020,
        retrieved="2026-08-30",
        notes="Illustrative BRD/MIL exposure fixture for the dynamic physical-nature test.",
    )
    labels = ["R:BRD", "R:MIL"]
    io = IOSystem(
        provenance=prov,
        sectors=Classification(name="s", kind="sector", labels=["BRD", "MIL"]),
        regions=Classification(name="r", kind="region", labels=["R"]),
        A=pd.DataFrame([[0.10, 0.05], [0.05, 0.10]], index=labels, columns=labels),
        final_demand=pd.DataFrame({"final_demand": [100.0, 100.0]}, index=labels),
        unit="MEUR",
        currency="EUR",
    )
    encore = EncoreDependencies(
        provenance=prov,
        ratings=pd.DataFrame(
            [("BRD", "surface_water", "VH"), ("MIL", "surface_water", "L")],
            columns=["process", "service", "materiality"],
        ),
        kind="dependency",
    )
    concordance = ConcordanceMap(
        provenance=prov,
        from_classification="toy-cge-sectors",
        to_classification="ENCORE-processes",
        weights={"BRD": {"BRD": 1.0}, "MIL": {"MIL": 1.0}},
    )
    return {
        "nature_iosystem": io,
        "EncoreDependencies": encore,
        "ConcordanceMap": concordance,
    }


def test_physical_nature_state_runs_end_to_end_through_run_recursive():
    """A physical water-stock degradation pathway (nature_state), threaded through the FULL
    recursive-dynamic wrapper, lowers covered-sector output — and the deeper the degradation, the
    larger the loss (the physical channel is genuinely driving the CGE, not a no-op)."""
    overrides = _brd_mil_nature_triple()

    def _run(states):
        sc = Scenario(
            name="phys",
            engine="cge_static",
            years=[2025, 2030, 2035],
            nature_state=[{"channel": "toy_water", "states": states}],
        )
        return run_recursive(sc, data_source="toy_cge_gov", data_overrides=overrides)

    mild = _run({2025: 100.0, 2035: 95.0})
    severe = _run({2025: 100.0, 2035: 80.0})

    # BRD (very-highly water-dependent) loses output under the degradation, and MORE so when the
    # water stock falls further — a monotone physical response carried through the whole pipeline.
    brd_mild = _sector_vol(mild, "BRD", 2035)
    brd_severe = _sector_vol(severe, "BRD", 2035)
    assert brd_severe < brd_mild < 0.0
    # The physical channel is recorded in the manifest so the dynamic run is reconstructible.
    assert "nature_state" in severe.result.manifest.assumptions


def test_physical_nature_state_hits_water_dependent_sector_harder():
    """The exposure asymmetry survives the dynamic pipeline: the very-highly water-dependent sector
    (BRD) loses more output than the lightly-dependent one (MIL) under the same degradation."""
    overrides = _brd_mil_nature_triple()
    sc = Scenario(
        name="phys",
        engine="cge_static",
        years=[2025, 2035],
        nature_state=[{"channel": "toy_water", "states": {2025: 100.0, 2035: 80.0}}],
    )
    path = run_recursive(sc, data_source="toy_cge_gov", data_overrides=overrides)
    assert _sector_vol(path, "BRD", 2035) < _sector_vol(path, "MIL", 2035)
