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
    shocks = _sector_productivity_shocks(
        _DC(structural=traj), 2025, 2030, ["BRD", "MIL"], ["N", "S"], True, shares
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
        _DC(structural=traj), 2025, 2027, ["BRD", "MIL"], ["R"], False, {}
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
        _DC(structural=traj), 2025, 2027, ["BRD", "MIL"], ["R"], False, {"BRD": 0.5, "MIL": 0.5}
    )
    assert all(s.mechanism == "labour_augmenting" for s in shocks)


def test_sector_productivity_biases_are_zero_mean_relative_drift():
    """The sector biases must be a zero-mean structural DRIFT — the VA-weighted geometric mean of
    (1+delta) factors is 1 — so no aggregate productivity level is re-imposed and a faster sector's
    positive bias is balanced by a slower sector's negative bias (review P1 2026-08-29: the earlier
    (LP/TFP)^s_L gave every sector in a high-TFP region a negative bias). BRD grows 3%/yr, MIL
    1%/yr, equal labour shares → BRD gets φ>1, MIL φ<1, weighted geo-mean 1."""
    from cge.dynamics.recursive import DynamicConfig as _DC
    from cge.dynamics.recursive import _sector_productivity_shocks

    traj = _traj_sector({"sector_productivity": {"BRD": {2025: 0.03}, "MIL": {2025: 0.01}}})
    shares = {"BRD": 0.5, "MIL": 0.5}
    shocks = _sector_productivity_shocks(
        _DC(structural=traj), 2025, 2030, ["BRD", "MIL"], ["R"], False, shares
    )
    by_sector = {s.coverage_sectors[0]: s.delta for s in shocks}
    assert by_sector["BRD"] > 0.0 > by_sector["MIL"]  # faster up, slower down (a genuine drift)
    # VA-weighted geometric mean of (1+delta) is 1 (zero-mean drift; no aggregate level re-imposed).
    geo = (1.0 + by_sector["BRD"]) ** 0.5 * (1.0 + by_sector["MIL"]) ** 0.5
    assert geo == pytest.approx(1.0, rel=1e-12)


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
