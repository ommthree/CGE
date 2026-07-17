"""Unit tests for Engine 1 (io_price). These gate CI on code changes; the model-level
validation suite (tests/test_validation.py, scripts/validate.py) shares the same checks."""

import numpy as np
import pytest

from cge.contracts.engine import registry
from cge.contracts.shocks import CarbonPrice, NatureStress
from cge.engines.io_price.engine import decompose, price_change
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


def test_gas_selection_distinct_and_additive():
    """gases=[CO2] and gases=[CH4] give different intensities; combined is GWP-additive."""
    import pandas as pd

    from cge.contracts.data_objects import Provenance, SatelliteAccount
    from cge.engines.io_price.engine import _intensity_for_gases

    prov = Provenance(
        source="t", source_version="1", licence="x", reference_year=2020, retrieved="2026-07-17"
    )
    sat = SatelliteAccount(
        provenance=prov,
        name="GHG",
        units={"CO2": "t/MEUR", "CH4": "t/MEUR"},
        data=pd.DataFrame({"L0": [100.0, 10.0]}, index=["CO2", "CH4"]),
    )
    co2, _ = _intensity_for_gases(sat, ["L0"], ["CO2"])
    ch4, _ = _intensity_for_gases(sat, ["L0"], ["CH4"])
    both, _ = _intensity_for_gases(sat, ["L0"], ["CO2", "CH4"])
    assert not np.allclose(co2, ch4)  # 100 vs 280 (10×28)
    assert np.allclose(both, co2 + ch4)


def test_missing_satellite_label_rejected():
    """A product missing from the satellite is an alignment error, not zero emissions."""
    import pandas as pd

    from cge.contracts.data_objects import Provenance, SatelliteAccount
    from cge.engines.io_price.engine import _intensity_for_gases

    prov = Provenance(
        source="t", source_version="1", licence="x", reference_year=2020, retrieved="2026-07-17"
    )
    sat = SatelliteAccount(
        provenance=prov,
        name="GHG",
        units={"CO2": "t/MEUR"},
        data=pd.DataFrame({"L0": [100.0]}, index=["CO2"]),
    )
    with pytest.raises(ValueError, match="missing"):
        _intensity_for_gases(sat, ["L0", "L1"], ["CO2"])


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
    # assumptions carry the interpretation caveat
    assert "UPPER BOUND" in result.manifest.assumptions["interpretation"]


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
