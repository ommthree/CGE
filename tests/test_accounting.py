"""Unit tests for the macro-aggregate accounting layer (Phase 4b, PE tier).

Pins the accounting identities and the real/nominal semantics: base-year GDP = ΣVA = Σfinal
demand; a price-only engine yields inflation but zero real GDP; a volume-bearing engine yields a
negative real GDP; GDP aggregates sector GVA; and bands propagate to the aggregates.
"""

import numpy as np

from cge.accounting import (
    ECONOMY_SECTOR,
    augment_with_macro_aggregates,
    base_year_value_added,
    macro_records,
)
from cge.contracts.shocks import CarbonPrice
from cge.runner import run_scenario
from cge.scenarios.loader import Scenario
from cge.validation import toy_economy


def _run(engine, price=100.0):
    sc = Scenario(name="m", engine=engine, years=[2020], shocks=[CarbonPrice(price=price)])
    return run_scenario(sc, data_source="toy")


def _econ(df, region, variable, scenario="central"):
    sub = df[
        (df["region"] == region)
        & (df["variable"] == variable)
        & (df["scenario"] == scenario)
        & (df["sector"] == ECONOMY_SECTOR)
    ]
    return float(sub["value"].iloc[0])


def test_base_year_value_added_matches_final_demand():
    """Production GDP (ΣVA) equals expenditure GDP (Σ final demand) at the base year."""
    io, _ = toy_economy()
    va = base_year_value_added(io)
    labels = list(io.A.columns)
    fd = io.final_demand.sum(axis=1).reindex(labels).fillna(0.0).sum()
    assert np.isclose(va.sum(), fd, rtol=1e-9)
    assert (va > 0).all()  # every toy sector has positive value added


def test_macro_variables_present_after_run():
    df = _run("io_price").data
    for v in ("gva_change", "gva_change_real", "gdp_change", "gdp_change_real", "deflator"):
        assert (df["variable"] == v).any(), v


def test_price_only_engine_zero_real_gdp():
    """Engine 1 has no volume response → inflation but zero real GDP change."""
    df = _run("io_price").data
    assert _econ(df, "A", "deflator") > 1e-6
    assert abs(_econ(df, "A", "gdp_change_real")) < 1e-12
    # nominal GDP change equals the deflator when there is no real movement
    assert np.isclose(_econ(df, "A", "gdp_change"), _econ(df, "A", "deflator"))


def test_volume_engine_negative_real_gdp():
    """Engine 2: a carbon price lowers real GDP in every region and band."""
    df = _run("partial_eq").data
    real = df[df["variable"] == "gdp_change_real"]["value"]
    assert (real < 0).all()


def test_real_equals_nominal_deflated():
    """The emitted real GDP equals (1+nominal)/(1+deflator)−1 exactly."""
    df = _run("partial_eq").data
    for region in ("A", "B"):
        for band in ("low", "central", "high"):
            dfl = _econ(df, region, "deflator", band)
            nom = _econ(df, region, "gdp_change", band)
            real = _econ(df, region, "gdp_change_real", band)
            assert np.isclose(real, (1 + nom) / (1 + dfl) - 1, atol=1e-12)


def test_gdp_aggregates_sector_gva():
    """Region GDP change = value-added-weighted mean of sector GVA changes."""
    io, _ = toy_economy()
    va = base_year_value_added(io)
    df = _run("partial_eq").data
    for region in ("A", "B"):
        sub = df[(df["region"] == region) & (df["scenario"] == "central")]
        gva = sub[(sub["variable"] == "gva_change") & (sub["sector"] != ECONOMY_SECTOR)]
        num = sum(float(va[f"{region}:{r.sector}"]) * float(r.value) for r in gva.itertuples())
        den = sum(float(va[f"{region}:{r.sector}"]) for r in gva.itertuples())
        assert np.isclose(_econ(df, region, "gdp_change"), num / den, atol=1e-9)


def test_bands_propagate_to_real_gdp():
    """Elasticity bands carry through to real GDP: low (most elastic) is the most negative."""
    df = _run("partial_eq").data
    lo = _econ(df, "A", "gdp_change_real", "low")
    ce = _econ(df, "A", "gdp_change_real", "central")
    hi = _econ(df, "A", "gdp_change_real", "high")
    assert lo <= ce <= hi < 0.0


def test_augment_is_idempotent():
    """Augmenting an already-augmented result does not double-add rows."""
    result = _run("io_price")
    io, _ = toy_economy()
    again = augment_with_macro_aggregates(result, io)
    assert len(again.data) == len(result.data)


def test_results_view_macro_helpers():
    """The GUI reshaping helpers expose the aggregates: a GDP table per region and a GVA table
    per sector, both with nominal and real columns."""
    from cge.gui import results_view as rv

    result = _run("partial_eq")
    assert rv.has_macro(result)
    gdp = rv.macro_gdp_table(result)
    assert {"GDP Δ (nominal)", "GDP Δ (real)", "deflator (inflation)"} <= set(gdp.columns)
    assert set(gdp["region"]) == {"A", "B"}
    gva = rv.macro_gva_table(result)
    assert {"GVA Δ (nominal)", "GVA Δ (real)"} <= set(gva.columns)
    assert ECONOMY_SECTOR not in set(gva["sector"])  # economy sentinel excluded from GVA table


def test_no_price_response_is_noop():
    """A result with no price rows yields no macro records (nothing to roll up)."""
    result = _run("io_price")
    io, _ = toy_economy()
    empty = result.data[result.data["variable"] == "__none__"]  # empty frame, same columns
    from cge.contracts.results import ResultSet

    stub = ResultSet(data=empty, manifest=result.manifest)
    assert macro_records(stub, io) == []
