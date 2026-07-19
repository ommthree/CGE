"""Unit tests for Engine 2 (partial_eq): production-volume response.

Production follows the Leontief quantity model x=(I−A)⁻¹y: elasticities move final demand
(finite-change (1+Δp)^ε−1), which propagates to gross-output (production) volume. These tests
pin the *correct* semantics (review found the old ε·Δp-per-good version wasn't production
volume and could go below −100%)."""

import numpy as np
import pandas as pd
import pytest

from cge.contracts.data_objects import (
    Classification,
    ElasticitySet,
    IOSystem,
    Provenance,
    SatelliteAccount,
)
from cge.contracts.engine import registry
from cge.contracts.shocks import CarbonPrice, NatureStress
from cge.engines.partial_eq.engine import PartialEqEngine, _finite_demand_response
from cge.runner import run_scenario
from cge.scenarios.loader import Scenario


def _run(price=100.0, engine="partial_eq"):
    sc = Scenario(name="v", engine=engine, years=[2020], shocks=[CarbonPrice(price=price)])
    return run_scenario(sc, data_source="toy")


def _prov():
    return Provenance(
        source="t", source_version="1", licence="x", reference_year=2020, retrieved="2026-07-18"
    )


def _two_sector_network(eps=-1.0):
    """fin (final good) uses 0.5 of up per unit; final demand only for fin; only fin emits."""
    prov = _prov()
    A = pd.DataFrame([[0.0, 0.5], [0.0, 0.0]], index=["R:up", "R:fin"], columns=["R:up", "R:fin"])
    fd = pd.DataFrame({"final_demand": [0.0, 100.0]}, index=["R:up", "R:fin"])
    io = IOSystem(
        provenance=prov,
        sectors=Classification(name="s", kind="sector", labels=["up", "fin"]),
        regions=Classification(name="r", kind="region", labels=["R"]),
        A=A,
        final_demand=fd,
        unit="MEUR",
        currency="EUR",
    )
    sat = SatelliteAccount(
        provenance=prov,
        name="GHG",
        units={"CO2": "t/MEUR", "CO2e": "tCO2e/MEUR"},
        data=pd.DataFrame({"R:up": [0.0, 0.0], "R:fin": [1000.0, 1000.0]}, index=["CO2", "CO2e"]),
    )
    eset = ElasticitySet(
        provenance=prov,
        kind="demand",
        classification="s",
        values={"up": (eps, eps, eps), "fin": (eps, eps, eps)},
        sources={"up": "t", "fin": "t"},
        confidence={"up": "high", "fin": "high"},
    )
    return {"IOSystem": io, "SatelliteAccount": sat, "ElasticitySet": eset}


def test_engine_registered_with_volume_capability():
    assert "partial_eq" in registry.names()
    caps = [c.value for c in registry.get("partial_eq").meta.capabilities]
    assert "volumes" in caps and "prices" in caps


def test_production_propagates_upstream():
    """THE key fix: a fall in final-good demand pulls its upstream supplier's PRODUCTION down
    too (via the Leontief inverse). The old engine reported the upstream good at 0%."""
    data = _two_sector_network(eps=-1.0)
    df = PartialEqEngine().run(data=data, shocks=[CarbonPrice(price=100.0)], years=[2020]).data
    vol = df[(df["variable"] == "volume_change") & (df["scenario"] == "central")]
    by = {r.sector: r.value for r in vol.itertuples()}
    # fin's price rose 10% → demand (1.1)^-1-1 = -9.09%; up supplies fin so its output falls too.
    assert np.isclose(by["fin"], (1.1) ** -1 - 1, atol=1e-9)
    assert by["up"] < -0.01  # NOT zero — this is the whole point
    assert np.isclose(by["up"], by["fin"], atol=1e-9)  # 1:1 here (up used only by fin)


def test_large_price_stays_above_minus_100pct():
    """Finite-change form keeps demand response > −100% even at live-scale price changes
    (review: linear ε·Δp gave an impossible −142% on real data)."""
    dp = np.array([2.369])  # the live +236.9% price change
    r = _finite_demand_response(dp, np.array([-0.6]))
    assert -1.0 < r[0] < 0.0
    # and via the engine end to end, no volume below −100%
    df = _run(price=500.0).data
    vol = df[df["variable"] == "volume_change"]["value"]
    assert (vol > -1.0).all()


def test_volume_and_final_demand_variables_present():
    df = _run().data
    for v in ("price_change", "final_demand_change", "volume_change", "elasticity_used"):
        assert (df["variable"] == v).any(), v


def test_volume_sign_negative():
    df = _run().data
    vol = df[(df["variable"] == "volume_change") & (df["scenario"] == "central")]["value"]
    assert (vol <= 1e-9).all()  # a carbon price reduces production volumes


def test_zero_shock_zero_volume():
    df = _run(price=0.0).data
    vol = df[df["variable"] == "volume_change"]["value"]
    assert np.allclose(vol, 0.0)


def test_bands_bracket_central():
    df = _run().data
    vol = df[df["variable"] == "volume_change"]
    for _key, g in vol.groupby(["region", "sector"]):
        by = {r.scenario: r.value for r in g.itertuples()}
        assert by["low"] <= by["central"] <= by["high"] <= 1e-9


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


def test_manifest_sensitive_to_elasticity_values():
    """Two elasticity sets with identical provenance but different values must produce
    different manifests (review: manifests were identical → not reproducible)."""
    base = _two_sector_network(eps=-0.5)
    other = _two_sector_network(eps=-1.5)
    m1 = PartialEqEngine().run(data=base, shocks=[CarbonPrice(price=100.0)], years=[2020]).manifest
    m2 = PartialEqEngine().run(data=other, shocks=[CarbonPrice(price=100.0)], years=[2020]).manifest
    h1 = m1.assumptions["elasticity_set"]["content_hash"]
    h2 = m2.assumptions["elasticity_set"]["content_hash"]
    assert h1 != h2  # different values → different content hash


def test_manifest_distinguishes_build_generations():
    """Two runs on the *same* build_id but different generations (a rewrite) must produce
    distinguishable manifests, so results stay reproducible (review P1c: generation protected
    recovery but not the run manifest's data-source identity)."""
    data_a = _two_sector_network(eps=-0.5)
    data_b = _two_sector_network(eps=-0.5)
    for d, gen in ((data_a, "genA"), (data_b, "genB")):
        prov = d["IOSystem"].provenance.model_copy(update={"build_id": "b1", "generation": gen})
        d["IOSystem"] = d["IOSystem"].model_copy(update={"provenance": prov})
    shocks = [CarbonPrice(price=100.0)]
    ma = PartialEqEngine().run(data=data_a, shocks=shocks, years=[2020]).manifest
    mb = PartialEqEngine().run(data=data_b, shocks=shocks, years=[2020]).manifest
    assert ma.data_source == "b1@genA" and mb.data_source == "b1@genB"
    assert ma.data_source != mb.data_source
    assert ma.assumptions["data_generation"] == "genA"


def test_manifest_carries_per_good_elasticity_provenance():
    data = _two_sector_network(eps=-0.5)
    m = PartialEqEngine().run(data=data, shocks=[CarbonPrice(price=100.0)], years=[2020]).manifest
    per_good = m.assumptions["elasticity_per_good"]
    assert set(per_good) == {"up", "fin"}
    assert (
        "source" in per_good["up"]
        and "confidence" in per_good["up"]
        and "default" in per_good["up"]
    )


def test_default_flag_counts_distinct_sectors_not_observations():
    """n_sectors_using_default counts distinct SECTORS, not year×region observations (review:
    it grew 2/4/6 with more years/regions for the same goods)."""
    m1 = run_scenario(
        Scenario(name="a", engine="partial_eq", years=[2020], shocks=[CarbonPrice(price=100.0)]),
        data_source="toy",
    ).manifest
    m2 = run_scenario(
        Scenario(
            name="b", engine="partial_eq", years=[2020, 2030], shocks=[CarbonPrice(price=100.0)]
        ),
        data_source="toy",
    ).manifest
    # Same toy sectors, different #years → the default-sector COUNT must be identical.
    assert m1.assumptions["n_sectors_using_default"] == m2.assumptions["n_sectors_using_default"]


def test_rejects_positive_demand_elasticity():
    """Positive demand elasticity is rejected at ElasticitySet construction (contract-level)."""
    with pytest.raises(ValueError, match="positive"):
        ElasticitySet(
            provenance=_prov(),
            kind="demand",
            classification="s",
            values={"fin": (0.1, 0.2, 0.3)},  # positive → invalid
            sources={"fin": "t"},
            confidence={"fin": "high"},
        )


def test_rejects_unordered_bands():
    with pytest.raises(ValueError, match="not ordered"):
        ElasticitySet(
            provenance=_prov(),
            kind="demand",
            classification="s",
            values={"fin": (-0.2, -0.5, -0.1)},  # central < low
            sources={"fin": "t"},
            confidence={"fin": "high"},
        )


def test_rejects_missing_metadata():
    with pytest.raises(ValueError, match="missing source"):
        ElasticitySet(
            provenance=_prov(),
            kind="demand",
            classification="s",
            values={"fin": (-1.0, -1.0, -1.0)},
            sources={},
            confidence={},  # no source/confidence
        )


def test_engine_rejects_wrong_kind_elasticity_set():
    """A non-demand ElasticitySet is rejected by the engine (kind check)."""
    data = _two_sector_network()
    data["ElasticitySet"] = ElasticitySet(
        provenance=_prov(),
        kind="armington",
        classification="s",
        values={"up": (0.5, 1.0, 1.5), "fin": (0.5, 1.0, 1.5)},
        sources={"up": "t", "fin": "t"},
        confidence={"up": "high", "fin": "high"},
    )
    with pytest.raises(ValueError, match="demand ElasticitySet"):
        PartialEqEngine().run(data=data, shocks=[CarbonPrice(price=100.0)], years=[2020])


def test_manifest_sensitive_to_fallback_band(monkeypatch):
    """Changing the FALLBACK elasticity band (applied to goods with no explicit value) must move
    the manifest, even though no explicit value changed (review P2: the hash covered only explicit
    values, so a fallback change left the substantive manifest identical)."""
    import cge.data.elasticities.library as lib
    import cge.engines.partial_eq.engine as pe

    # A network whose goods have NO explicit elasticity → both fall back to the default triple.
    def _net():
        d = _two_sector_network()
        d["ElasticitySet"] = ElasticitySet(
            provenance=_prov(),
            kind="demand",
            classification="s",
            values={},  # empty → every good uses the fallback
            sources={},
            confidence={},
        )
        return d

    shocks = [CarbonPrice(price=100.0)]
    monkeypatch.setattr(lib, "DEFAULT_DEMAND_ELASTICITY", (-0.8, -0.5, -0.2))
    monkeypatch.setattr(pe, "DEFAULT_DEMAND_ELASTICITY", (-0.8, -0.5, -0.2))
    m1 = PartialEqEngine().run(data=_net(), shocks=shocks, years=[2020]).manifest

    monkeypatch.setattr(lib, "DEFAULT_DEMAND_ELASTICITY", (-1.6, -0.5, -0.1))
    monkeypatch.setattr(pe, "DEFAULT_DEMAND_ELASTICITY", (-1.6, -0.5, -0.1))
    m2 = PartialEqEngine().run(data=_net(), shocks=shocks, years=[2020]).manifest

    e1 = m1.assumptions["elasticity_set"]["effective_content_hash"]
    e2 = m2.assumptions["elasticity_set"]["effective_content_hash"]
    assert e1 != e2  # different fallback band → different effective hash
    # And the full triple is recorded per good, not just the central value.
    per_good = m1.assumptions["elasticity_per_good"]["up"]
    assert per_good["low"] == -0.8 and per_good["central"] == -0.5 and per_good["high"] == -0.2


def test_rejects_incompatible_classification():
    """An elasticity set on an unrelated classification is rejected, not matched by coincidental
    sector names (review P2)."""
    data = _two_sector_network()
    data["ElasticitySet"] = ElasticitySet(
        provenance=_prov(),
        kind="demand",
        classification="completely-unrelated",
        values={"up": (-0.5, -0.5, -0.5), "fin": (-0.5, -0.5, -0.5)},
        sources={"up": "t", "fin": "t"},
        confidence={"up": "high", "fin": "high"},
    )
    with pytest.raises(ValueError, match="not compatible"):
        PartialEqEngine().run(data=data, shocks=[CarbonPrice(price=100.0)], years=[2020])


def test_manifest_distinguishes_satellite_generations():
    """Same IO system, different SATELLITE (generation and doubled emissions) → the prices change,
    so the manifest must change too (review P1: manifests recorded only the IO system)."""
    base = _two_sector_network()
    other = _two_sector_network()
    # Double the satellite emissions and stamp a different provenance generation.
    sat = other["SatelliteAccount"]
    prov2 = sat.provenance.model_copy(update={"build_id": "sat", "generation": "g2"})
    other["SatelliteAccount"] = sat.model_copy(update={"data": sat.data * 2.0, "provenance": prov2})
    prov1 = base["SatelliteAccount"].provenance.model_copy(
        update={"build_id": "sat", "generation": "g1"}
    )
    base["SatelliteAccount"] = base["SatelliteAccount"].model_copy(update={"provenance": prov1})

    shocks = [CarbonPrice(price=100.0)]
    m1 = PartialEqEngine().run(data=base, shocks=shocks, years=[2020]).manifest
    m2 = PartialEqEngine().run(data=other, shocks=shocks, years=[2020]).manifest
    sat1 = next(i for i in m1.assumptions["inputs"] if i["name"] == "SatelliteAccount")
    sat2 = next(i for i in m2.assumptions["inputs"] if i["name"] == "SatelliteAccount")
    assert sat1["generation"] != sat2["generation"]
    assert sat1["content_hash"] != sat2["content_hash"]  # doubled emissions → different hash


def test_manifest_sensitive_to_final_demand():
    """Review P1: final demand sets the baseline y0 that drives the volume response, so a changed
    final demand must move the IO input's content hash (it fingerprints A AND final demand)."""
    base = _two_sector_network()
    other = _two_sector_network()
    io = other["IOSystem"]
    fd = io.final_demand.copy()
    fd.iloc[1, 0] *= 1.5  # change final demand for the final good
    other["IOSystem"] = io.model_copy(update={"final_demand": fd})
    shocks = [CarbonPrice(price=100.0)]
    m1 = PartialEqEngine().run(data=base, shocks=shocks, years=[2020]).manifest
    m2 = PartialEqEngine().run(data=other, shocks=shocks, years=[2020]).manifest
    io1 = next(i for i in m1.assumptions["inputs"] if i["name"] == "IOSystem")
    io2 = next(i for i in m2.assumptions["inputs"] if i["name"] == "IOSystem")
    assert io1["content_hash"] != io2["content_hash"]  # changed final demand → different hash


def test_energy_price_drives_volume_response():
    """An EnergyPrice flows through Engine 1's prices into the volume response: a carrier price
    rise reduces production volumes, with no positive central responses."""
    from cge.contracts.shocks import EnergyPrice

    sc = Scenario(
        name="e",
        engine="partial_eq",
        years=[2020],
        shocks=[EnergyPrice(carrier="energy", change=0.5)],
    )
    df = run_scenario(sc, data_source="toy").data
    vol = df[(df["variable"] == "volume_change") & (df["scenario"] == "central")]["value"]
    assert (vol <= 1e-9).all() and (vol < -1e-6).any()  # volumes fall, some materially
    assert (vol > -1.0).all()  # bounded above −100%


def test_carbon_plus_energy_supported_by_partial_eq():
    """partial_eq accepts a combined carbon + energy scenario (both are supported shocks)."""
    from cge.contracts.shocks import EnergyPrice

    sc = Scenario(
        name="ce",
        engine="partial_eq",
        years=[2020],
        shocks=[CarbonPrice(price=50.0), EnergyPrice(carrier="energy", change=0.2)],
    )
    df = run_scenario(sc, data_source="toy").data
    assert (df["variable"] == "volume_change").any()


def test_rejects_unsupported_shock():
    sc = Scenario(
        name="v",
        engine="partial_eq",
        years=[2020],
        shocks=[NatureStress(service="pollination", severity=0.3)],
    )
    with pytest.raises(ValueError, match="does not support"):
        run_scenario(sc, data_source="toy")


def test_energy_price_minus_100pct_rejected():
    """Review P1: EnergyPrice(change=-1) pins the carrier to a zero price; with a negative demand
    elasticity 0^ε diverges. partial_eq must reject the non-positive price ratio, not emit inf."""
    from cge.contracts.shocks import EnergyPrice

    sc = Scenario(
        name="e",
        engine="partial_eq",
        years=[2020],
        shocks=[EnergyPrice(carrier="energy", change=-1.0)],
    )
    with pytest.raises(ValueError, match="positive price ratio"):
        run_scenario(sc, data_source="toy")
