"""Unit tests for Engine 1 (io_price). These gate CI on code changes; the model-level
validation suite (tests/test_validation.py, scripts/validate.py) shares the same checks."""

import numpy as np
import pytest

from cge.contracts.engine import registry
from cge.contracts.shocks import CarbonPrice, EnergyPrice, NatureStress
from cge.engines.io_price.engine import decompose, energy_price_pins, price_change
from cge.runner import run_scenario
from cge.scenarios.loader import Scenario
from cge.validation import toy_economy


def _toy():
    io, sat = toy_economy()
    labels = list(io.A.columns)
    A = io.A.to_numpy(dtype=float)
    e = sat.data.loc["CO2"].reindex(labels).to_numpy(dtype=float)
    return labels, A, e


def test_engine_registered():
    assert "io_price" in registry.names()


def test_matches_explicit_inverse():
    _, A, e = _toy()
    c = 100.0 * e
    dp = price_change(A, c)
    dp_ref = np.linalg.inv(np.eye(A.shape[0]) - A.T) @ c
    assert np.allclose(dp, dp_ref, atol=1e-9)


def test_zero_shock_zero_change():
    _, A, e = _toy()
    assert np.max(np.abs(price_change(A, 0.0 * e))) == 0.0


def test_linearity():
    _, A, e = _toy()
    assert np.allclose(price_change(A, 200.0 * e), 2.0 * price_change(A, 100.0 * e), atol=1e-9)


def test_pass_through_adds_cost():
    _, A, e = _toy()
    c = 100.0 * e
    assert np.all(price_change(A, c) >= c - 1e-9)


def test_decomposition_sums_to_total():
    _, A, e = _toy()
    c = 100.0 * e
    parts = decompose(A, c, tiers=3)
    assert np.allclose(sum(parts.values()), price_change(A, c), atol=1e-9)


def test_non_productive_economy_rejected():
    _, A, e = _toy()
    with pytest.raises(ValueError, match="not productive"):
        price_change(A * 10.0, e)


def test_negative_coefficient_rejected():
    """A negative A entry breaks non-negative pass-through (positive tax could lower a price);
    the admissibility guard must reject it (review counterexample)."""
    A = np.array([[0.0, -0.5], [0.0, 0.0]])
    with pytest.raises(ValueError, match="negative entries"):
        price_change(A, np.array([1.0, 0.0]))


def _ghg_sat():
    import pandas as pd

    from cge.contracts.data_objects import Provenance, SatelliteAccount

    prov = Provenance(
        source="t", source_version="1", licence="x", reference_year=2020, retrieved="2026-07-17"
    )
    return SatelliteAccount(
        provenance=prov,
        name="GHG",
        units={"CO2": "t/MEUR", "CH4": "t/MEUR"},
        data=pd.DataFrame({"A:x": [100.0, 10.0]}, index=["CO2", "CH4"]),
    )


def test_gas_selection_distinct_and_additive():
    """gases=[CO2] and gases=[CH4] give different intensities; combined is GWP-additive."""
    from cge.engines.io_price.engine import _gas_intensity

    sat = _ghg_sat()
    co2 = _gas_intensity(sat, ["A:x"], ["CO2"])
    ch4 = _gas_intensity(sat, ["A:x"], ["CH4"])
    both = _gas_intensity(sat, ["A:x"], ["CO2", "CH4"])
    assert not np.allclose(co2, ch4)  # 100 vs 280 (10×28)
    assert np.allclose(both, co2 + ch4)


def test_unknown_gas_rejected_not_silently_aggregated():
    """An unknown or partially-unavailable gas must raise, never fall back to CO2e (review)."""
    from cge.engines.io_price.engine import _gas_intensity

    sat = _ghg_sat()
    with pytest.raises(ValueError, match="not in satellite"):
        _gas_intensity(sat, ["A:x"], ["NOT_A_GAS"])
    with pytest.raises(ValueError, match="not in satellite"):
        _gas_intensity(sat, ["A:x"], ["CO2", "TYPO"])


def test_multi_gas_shocks_do_not_cross_multiply():
    """Two shocks, each pricing its own gas, contribute price×own-gas independently — not
    union-gases-and-sum-prices (the review's cross-multiplication counterexample)."""
    from cge.engines.io_price.engine import MEUR_TO_EUR, carbon_cost_vector

    sat = _ghg_sat()
    shocks = [CarbonPrice(price=100.0, gases=["CO2"]), CarbonPrice(price=10.0, gases=["CH4"])]
    cost, _ = carbon_cost_vector(shocks, sat, ["A:x"], 2020)
    expected = (100.0 * 100.0 + 10.0 * 280.0) * MEUR_TO_EUR  # CO2@100·100 + CH4@10·280
    assert np.isclose(cost[0], expected)


def test_missing_satellite_label_rejected():
    """A product missing from the satellite is an alignment error, not zero emissions."""
    from cge.engines.io_price.engine import carbon_cost_vector

    sat = _ghg_sat()
    with pytest.raises(ValueError, match="missing"):
        carbon_cost_vector([CarbonPrice(price=100.0)], sat, ["A:x", "A:y"], 2020)


def test_negative_path_rejected_at_construction():
    with pytest.raises(ValueError):
        CarbonPrice(price=100.0, path={2020: -20.0})


def test_full_build_rejected_dense_only():
    """The dense engine refuses builds above the product cap (small-build-only enforcement)."""
    from cge.engines.io_price.engine import MAX_DENSE_PRODUCTS, IOPriceEngine
    from cge.validation import toy_economy

    io, sat = toy_economy()
    # Fake a too-large build by padding the label count check via a monkeyish wrapper.
    big_labels = [f"R{i}:s" for i in range(MAX_DENSE_PRODUCTS + 1)]
    io.A = io.A.reindex(index=big_labels, columns=big_labels).fillna(0.0)
    with pytest.raises(ValueError, match="dense-only"):
        IOPriceEngine().run(
            data={"IOSystem": io, "SatelliteAccount": sat},
            shocks=[CarbonPrice(price=100.0)],
            years=[2020],
        )


def test_wrong_monetary_unit_rejected():
    """The 1e-6 scaling is only valid for a MEUR base; a different unit must be rejected."""
    from cge.engines.io_price.engine import IOPriceEngine
    from cge.validation import toy_economy

    io, sat = toy_economy()
    io.unit = "EUR"  # not MEUR
    with pytest.raises(ValueError, match="MEUR monetary base"):
        IOPriceEngine().run(
            data={"IOSystem": io, "SatelliteAccount": sat},
            shocks=[CarbonPrice(price=100.0)],
            years=[2020],
        )


def _run_toy(io, sat):
    from cge.engines.io_price.engine import IOPriceEngine

    return IOPriceEngine().run(
        data={"IOSystem": io, "SatelliteAccount": sat},
        shocks=[CarbonPrice(price=100.0)],
        years=[2020],
    )


def test_kg_per_meur_intensity_rejected():
    """kg/MEUR passes a '/MEUR' suffix but is 1000× wrong; exact-unit check must reject it."""
    from cge.validation import toy_economy

    io, sat = toy_economy()
    sat.units = {"CO2": "kg/MEUR"}
    with pytest.raises(ValueError, match="expected 't/MEUR'"):
        _run_toy(io, sat)


def test_missing_intensity_units_rejected():
    from cge.validation import toy_economy

    io, sat = toy_economy()
    sat.units = {}
    with pytest.raises(ValueError, match="no unit metadata"):
        _run_toy(io, sat)


def test_non_eur_currency_rejected():
    from cge.validation import toy_economy

    io, sat = toy_economy()
    io.currency = "USD"
    with pytest.raises(ValueError, match="currency"):
        _run_toy(io, sat)


def test_manifest_identifies_satellite_not_just_io():
    """Same IO system, different satellite (doubled emissions + different generation) moves the
    prices, so the manifest must change too — it must identify the satellite, not only the IO
    system (review P1)."""
    from cge.validation import toy_economy

    io, sat = toy_economy()
    prov1 = sat.provenance.model_copy(update={"build_id": "sat", "generation": "g1"})
    sat1 = sat.model_copy(update={"provenance": prov1})
    prov2 = sat.provenance.model_copy(update={"build_id": "sat", "generation": "g2"})
    sat2 = sat.model_copy(update={"data": sat.data * 2.0, "provenance": prov2})

    m1 = _run_toy(io, sat1).manifest
    m2 = _run_toy(io, sat2).manifest
    s1 = next(i for i in m1.assumptions["inputs"] if i["name"] == "SatelliteAccount")
    s2 = next(i for i in m2.assumptions["inputs"] if i["name"] == "SatelliteAccount")
    assert s1["generation"] == "g1" and s2["generation"] == "g2"
    assert s1["content_hash"] != s2["content_hash"]  # doubled emissions → different content hash


def _run_shocks(io, sat, shocks):
    from cge.engines.io_price.engine import IOPriceEngine

    return IOPriceEngine().run(
        data={"IOSystem": io, "SatelliteAccount": sat}, shocks=shocks, years=[2020]
    )


def _price_series(result):
    d = result.data
    return d[d["variable"] == "price_change"].set_index(["region", "sector"])["value"].sort_index()


def test_energy_price_pins_carrier_exactly_and_propagates():
    """A carrier output-price rise pins the carrier's own Δp to EXACTLY the change (a boundary
    condition, review P2), and propagates to energy users; energy-intensive sectors move more."""
    io, sat = toy_economy()
    dp = _price_series(_run_shocks(io, sat, [EnergyPrice(carrier="energy", change=0.5)]))
    for region in ("A", "B"):
        assert dp.loc[(region, "energy")] == pytest.approx(0.5)  # EXACT, not +0.5 plus own upstream
        # manufacturing uses more energy than agriculture in the toy economy → larger price rise.
        assert dp.loc[(region, "manufacturing")] > dp.loc[(region, "agriculture")] > 0.0


def test_energy_price_linear_in_change():
    """Doubling the carrier price change doubles every price response (the system is linear)."""
    io, sat = toy_economy()
    d1 = _price_series(_run_shocks(io, sat, [EnergyPrice(carrier="energy", change=0.2)]))
    d2 = _price_series(_run_shocks(io, sat, [EnergyPrice(carrier="energy", change=0.4)]))
    assert np.allclose(d2.to_numpy(), 2.0 * d1.to_numpy(), atol=1e-12)


def test_carbon_and_energy_combine_with_pinned_carrier():
    """A combined CarbonPrice + EnergyPrice run pins the carrier's price to the energy change (a
    boundary condition wins over the carbon cost on the carrier), while non-carrier sectors reflect
    BOTH the carbon cost and the propagated pinned energy price. The carrier is not additive
    (that was the cost-wedge bug); free sectors are higher than under carbon alone."""
    io, sat = toy_economy()
    carbon = [CarbonPrice(price=100.0)]
    energy = [EnergyPrice(carrier="energy", change=0.3)]
    c = _price_series(_run_shocks(io, sat, carbon))
    both = _price_series(_run_shocks(io, sat, carbon + energy))
    # The carrier is pinned to exactly 0.3, regardless of the carbon price.
    assert both.loc[("A", "energy")] == pytest.approx(0.3)
    # A non-carrier sector reflects both effects → strictly above the carbon-only response.
    assert both.loc[("A", "manufacturing")] > c.loc[("A", "manufacturing")]


def test_energy_price_coverage_restricts_direct_hit():
    """A region-restricted energy price hits that region's carrier directly; another region moves
    only through cross-region supply-chain use, so by strictly less on its own carrier."""
    io, sat = toy_economy()
    dp = _price_series(
        _run_shocks(io, sat, [EnergyPrice(carrier="energy", change=0.5, coverage_regions=["A"])])
    )
    assert dp.loc[("A", "energy")] == pytest.approx(0.5)  # pinned in the covered region
    assert dp.loc[("B", "energy")] < dp.loc[("A", "energy")]  # only indirect in B


def test_unknown_carrier_rejected():
    """A carrier that is not a sector in the build is rejected, not silently zero-impact."""
    io, sat = toy_economy()
    with pytest.raises(ValueError, match="carrier"):
        _run_shocks(io, sat, [EnergyPrice(carrier="not_a_sector", change=0.5)])


def test_energy_price_pins_are_on_carrier_labels_only():
    """Unit-level: the price pins target exactly the carrier's labels, set to the change value."""
    io, _ = toy_economy()
    labels = list(io.A.columns)
    pins, _desc = energy_price_pins([EnergyPrice(carrier="energy", change=0.3)], labels, 2020)
    energy_idx = {i for i, lab in enumerate(labels) if lab.split(":", 1)[1] == "energy"}
    assert set(pins) == energy_idx
    assert all(v == pytest.approx(0.3) for v in pins.values())


def test_energy_price_time_path():
    """An EnergyPrice path is read per year (piecewise-linear), like every shock."""
    io, sat = toy_economy()
    from cge.engines.io_price.engine import IOPriceEngine

    ep = EnergyPrice(carrier="energy", change=0.0, path={2020: 0.0, 2030: 0.4})
    d = (
        IOPriceEngine()
        .run(data={"IOSystem": io, "SatelliteAccount": sat}, shocks=[ep], years=[2020, 2025, 2030])
        .data
    )
    energy = d[(d["variable"] == "price_change") & (d["sector"] == "energy") & (d["region"] == "A")]
    by_year = energy.set_index("year")["value"]
    assert by_year[2020] == pytest.approx(0.0)
    assert by_year[2020] < by_year[2025] < by_year[2030]  # ramps up along the path


def test_size_cap_runs_before_dense_ops():
    """The dense-size cap must fire before any eigvals/solve (else the guard is pointless)."""
    import numpy as np

    from cge.engines.io_price.engine import MAX_DENSE_PRODUCTS
    from cge.validation import toy_economy

    io, sat = toy_economy()
    big = [f"R{i}:s" for i in range(MAX_DENSE_PRODUCTS + 1)]
    io.A = io.A.reindex(index=big, columns=big).fillna(0.0)

    called = {"eigvals": False}
    orig = np.linalg.eigvals
    np.linalg.eigvals = lambda A: called.__setitem__("eigvals", True) or orig(A)
    try:
        with pytest.raises(ValueError, match="dense-only"):
            _run_toy(io, sat)
    finally:
        np.linalg.eigvals = orig
    assert called["eigvals"] is False  # cap fired before the dense eigenvalue computation


def test_malformed_gas_selections_rejected():
    from cge.engines.io_price.engine import _validate_gases

    with pytest.raises(ValueError, match="non-empty"):
        _validate_gases([])
    with pytest.raises(ValueError, match="duplicates"):
        _validate_gases(["CO2", "CO2"])
    with pytest.raises(ValueError, match="mix"):
        _validate_gases(["CO2e", "CO2"])


def test_empty_or_duplicate_gases_rejected_at_construction():
    with pytest.raises(ValueError):
        CarbonPrice(price=100.0, gases=[])
    with pytest.raises(ValueError):
        CarbonPrice(price=100.0, gases=["CO2", "CO2"])


def test_nan_path_rejected():
    with pytest.raises(ValueError):
        CarbonPrice(price=100.0, path={2020: float("nan")})


def test_engine_version_is_current():
    from cge.engines.io_price.engine import IOPriceEngine

    assert IOPriceEngine().meta.version == "0.6.0"


def test_gas_without_gwp_factor_rejected():
    """A gas present in the data but absent from the GWP table must be rejected, not given
    GWP=1 (review: SF6 with GWP~23500 ran as if it were CO2)."""
    import pandas as pd

    from cge.contracts.data_objects import Provenance, SatelliteAccount
    from cge.engines.io_price.engine import _gas_intensity

    prov = Provenance(
        source="t", source_version="1", licence="x", reference_year=2020, retrieved="2026-07-18"
    )
    sat = SatelliteAccount(
        provenance=prov,
        name="GHG",
        units={"SF6": "t/MEUR"},
        data=pd.DataFrame({"A:x": [100.0]}, index=["SF6"]),
    )
    with pytest.raises(ValueError, match="No GWP-100 factor"):
        _gas_intensity(sat, ["A:x"], ["SF6"])


def test_infinite_carbon_price_rejected():
    with pytest.raises(ValueError):
        CarbonPrice(price=float("inf"))


def test_co2e_mixed_with_component_rejected_at_construction():
    with pytest.raises(ValueError, match="cannot mix"):
        CarbonPrice(price=100.0, gases=["CO2e", "CO2"])


def test_coverage_typo_rejected():
    """A coverage label absent from the build must raise, not silently zero the scenario."""
    scenario = Scenario(
        name="t",
        engine="io_price",
        years=[2020],
        shocks=[CarbonPrice(price=100.0, coverage_regions=["NOT_A_REGION"])],
    )
    with pytest.raises(ValueError, match="coverage_regions not in the build"):
        run_scenario(scenario, data_source="toy")


def test_negative_intensity_rejected():
    """A negative selected intensity would make a positive tax lower a price; reject it."""
    from cge.validation import toy_economy

    io, sat = toy_economy()
    sat.data.loc["CO2"] = -100.0  # all-negative intensities
    with pytest.raises(ValueError, match="negative emission intensities"):
        _run_toy(io, sat)


def test_negative_gas_cannot_cancel_against_positive(tmp_path):
    """A negative CO2 intensity must be rejected even when a positive gas outweighs it in the
    sum (review: sum-then-check missed it)."""
    import pandas as pd

    from cge.contracts.data_objects import Provenance, SatelliteAccount
    from cge.engines.io_price.engine import carbon_cost_vector

    prov = Provenance(
        source="t", source_version="1", licence="x", reference_year=2020, retrieved="2026-07-18"
    )
    sat = SatelliteAccount(
        provenance=prov,
        name="GHG",
        units={"CO2": "t/MEUR", "CH4": "t/MEUR"},
        data=pd.DataFrame({"A:x": [-100.0, 10.0]}, index=["CO2", "CH4"]),
    )
    shocks = [CarbonPrice(price=100.0, gases=["CO2"]), CarbonPrice(price=100.0, gases=["CH4"])]
    with pytest.raises(ValueError, match="negative emission intensities"):
        carbon_cost_vector(shocks, sat, ["A:x"], 2020)


def _ghg_two_gas(co2_vals, ch4_vals, labels):
    import pandas as pd

    from cge.contracts.data_objects import Provenance, SatelliteAccount

    prov = Provenance(
        source="t", source_version="1", licence="x", reference_year=2020, retrieved="2026-07-18"
    )
    return SatelliteAccount(
        provenance=prov,
        name="GHG",
        units={"CO2": "t/MEUR", "CH4": "t/MEUR"},
        data=pd.DataFrame(
            {lab: [co2_vals[i], ch4_vals[i]] for i, lab in enumerate(labels)}, index=["CO2", "CH4"]
        ),
    )


def test_single_multigas_shock_negative_gas_rejected():
    """A negative CO2 inside a single gases=[CO2,CH4] shock must be rejected before GWP
    aggregation — a positive CH4 must not hide it (review counterexample)."""
    from cge.engines.io_price.engine import carbon_cost_vector

    sat = _ghg_two_gas([-100.0], [10.0], ["A:x"])
    with pytest.raises(ValueError, match="negative emission intensities"):
        carbon_cost_vector([CarbonPrice(price=100.0, gases=["CO2", "CH4"])], sat, ["A:x"], 2020)


def test_uncovered_negative_row_excluded_by_coverage():
    """An uncovered negative row is NOT rejected — excluding it via coverage works (review)."""
    from cge.engines.io_price.engine import carbon_cost_vector

    sat = _ghg_two_gas([-100.0, 50.0], [0.0, 0.0], ["A:x", "B:y"])
    cost, _ = carbon_cost_vector(
        [CarbonPrice(price=100.0, gases=["CO2"], coverage_regions=["B"])], sat, ["A:x", "B:y"], 2020
    )
    assert cost[0] == 0.0 and cost[1] > 0.0  # A excluded, B priced


def test_multi_year_manifest_records_each_year():
    """A time-path run records every year's contributions, not only the last (review)."""
    scenario = Scenario(
        name="p",
        engine="io_price",
        years=[2020, 2030],
        shocks=[CarbonPrice(price=0.0, path={2020: 50.0, 2030: 200.0})],
    )
    result = run_scenario(scenario, data_source="toy")
    by_year = result.manifest.assumptions["shock_contributions_by_year"]
    assert set(by_year) == {"2020", "2030"}
    assert "50" in str(by_year["2020"]) and "200" in str(by_year["2030"])


def test_time_path_varies_by_year():
    scenario = Scenario(
        name="p",
        engine="io_price",
        years=[2020, 2030],
        shocks=[CarbonPrice(price=0.0, path={2020: 0.0, 2030: 200.0})],
    )
    df = run_scenario(scenario, data_source="toy").data
    by_year = df[df["variable"] == "price_change"].groupby("year")["value"].sum()
    assert by_year[2020] < by_year[2030]


def test_revenue_recycling_rejected():
    scenario = Scenario(
        name="r",
        engine="io_price",
        years=[2020],
        shocks=[CarbonPrice(price=100.0, revenue_recycling="lump_sum")],
    )
    with pytest.raises(ValueError, match="revenue recycling"):
        run_scenario(scenario, data_source="toy")


def test_negative_carbon_price_rejected_at_construction():
    with pytest.raises(ValueError):
        CarbonPrice(price=-10.0)


def test_end_to_end_via_runner():
    scenario = Scenario(
        name="t", engine="io_price", years=[2020], shocks=[CarbonPrice(price=100.0)]
    )
    result = run_scenario(scenario, data_source="toy")
    df = result.data
    # main variable present for all 6 labels
    assert (df["variable"] == "price_change").sum() == 6
    # decomposition variables present
    assert (df["variable"] == "price_change_direct").sum() == 6
    # assumptions carry the interpretation caveat (no volume effect; over-states vs substitution)
    assert "volume effect" in result.manifest.assumptions["interpretation"]


def test_engine_rejects_unsupported_shock():
    scenario = Scenario(
        name="t",
        engine="io_price",
        years=[2020],
        shocks=[NatureStress(service="pollination", severity=0.3)],
    )
    with pytest.raises(ValueError, match="does not support"):
        run_scenario(scenario)


def test_coverage_restricts_direct_cost():
    """Region-restricted carbon price: region B has zero direct cost (upstream may leak)."""
    scenario = Scenario(
        name="t",
        engine="io_price",
        years=[2020],
        shocks=[CarbonPrice(price=100.0, coverage_regions=["A"])],
    )
    df = run_scenario(scenario, data_source="toy").data
    b_direct = df[(df["variable"] == "price_change_direct") & (df["region"] == "B")]
    assert np.allclose(b_direct["value"].to_numpy(), 0.0)


def test_conflicting_energy_pins_rejected():
    """Review P2: two EnergyPrice shocks pinning the same carrier product to different values are
    rejected (the outcome would otherwise be silently order-dependent)."""
    from cge.engines.io_price.engine import IOPriceEngine

    io, sat = toy_economy()
    with pytest.raises(ValueError, match="conflicting energy-price pins"):
        IOPriceEngine().run(
            data={"IOSystem": io, "SatelliteAccount": sat},
            shocks=[
                EnergyPrice(carrier="energy", change=0.2),
                EnergyPrice(carrier="energy", change=0.5),
            ],
            years=[2020],
        )


def test_scenario_year_list_validated():
    """Review P2: Scenario rejects empty and duplicate year lists at construction."""
    from cge.scenarios.loader import Scenario

    with pytest.raises(ValueError, match="non-empty"):
        Scenario(name="y", engine="io_price", years=[], shocks=[CarbonPrice(price=1.0)])
    with pytest.raises(ValueError, match="duplicates"):
        Scenario(name="y", engine="io_price", years=[2020, 2020], shocks=[CarbonPrice(price=1.0)])
    # a valid unsorted list is accepted and returned sorted
    sc = Scenario(name="y", engine="io_price", years=[2030, 2020], shocks=[CarbonPrice(price=1.0)])
    assert sc.years == [2020, 2030]
