"""Unit tests for Engine 2 (partial_eq): first-order volume response Δq/q = ε·Δp."""

import numpy as np
import pytest

from cge.contracts.engine import registry
from cge.contracts.shocks import CarbonPrice, NatureStress
from cge.runner import run_scenario
from cge.scenarios.loader import Scenario


def _run(price=100.0, engine="partial_eq"):
    sc = Scenario(name="v", engine=engine, years=[2020], shocks=[CarbonPrice(price=price)])
    return run_scenario(sc, data_source="toy")


def test_engine_registered_with_volume_capability():
    assert "partial_eq" in registry.names()
    caps = [c.value for c in registry.get("partial_eq").meta.capabilities]
    assert "volumes" in caps and "prices" in caps


def test_volume_sign_negative():
    df = _run().data
    vol = df[(df["variable"] == "volume_change") & (df["scenario"] == "central")]["value"]
    assert (vol <= 1e-12).all()  # a carbon price reduces volumes


def test_zero_shock_zero_volume():
    df = _run(price=0.0).data
    vol = df[df["variable"] == "volume_change"]["value"]
    assert np.allclose(vol, 0.0)


def test_proportional_to_price():
    def central(df):
        v = df[(df["variable"] == "volume_change") & (df["scenario"] == "central")]
        return v.set_index(["region", "sector"])["value"]

    v1, v2 = central(_run(100.0).data), central(_run(200.0).data)
    assert np.allclose(v2.to_numpy(), 2.0 * v1.to_numpy(), atol=1e-9)


def test_bands_bracket_central():
    df = _run().data
    vol = df[df["variable"] == "volume_change"]
    for _, g in vol.groupby(["region", "sector"]):
        by = {r.scenario: r.value for r in g.itertuples()}
        assert by["low"] <= by["central"] <= by["high"] <= 1e-12


def test_prices_passed_through_from_engine1():
    def prices(df):
        return (
            df[df["variable"] == "price_change"]
            .set_index(["region", "sector"])["value"]
            .sort_index()
        )

    pe = prices(_run(engine="partial_eq").data)
    io = prices(_run(engine="io_price").data)
    assert np.allclose(pe.to_numpy(), io.to_numpy(), atol=1e-12)


def test_default_elasticity_flagged():
    result = _run()
    # toy 'energy'/'manufacturing' aren't coarse keys → default used and counted.
    assert result.manifest.assumptions["n_goods_using_default_elasticity"] >= 1


def test_matches_analytic_elasticity():
    """Δq/q equals ε·Δp with ε the elasticity actually used for that good."""
    from cge.data.elasticities import default_demand_set
    from cge.engines.partial_eq.engine import _elasticity_for

    df = _run().data
    eset = default_demand_set()
    for r in df[(df["variable"] == "volume_change") & (df["scenario"] == "central")].itertuples():
        dp = df[
            (df["variable"] == "price_change")
            & (df["region"] == r.region)
            & (df["sector"] == r.sector)
        ]["value"].iloc[0]
        (_, ce, _), _ = _elasticity_for(r.sector, eset)
        assert np.isclose(r.value, ce * dp, atol=1e-12)


def test_rejects_unsupported_shock():
    sc = Scenario(
        name="v",
        engine="partial_eq",
        years=[2020],
        shocks=[NatureStress(service="pollination", severity=0.3)],
    )
    with pytest.raises(ValueError, match="does not support"):
        run_scenario(sc, data_source="toy")
