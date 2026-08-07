"""Phase 6 (nature/ENCORE) — ingestion, materiality scale, concordance, and the exposure engine."""

import numpy as np
import pandas as pd
import pytest

from cge.contracts.data_objects import ConcordanceMap, Provenance
from cge.nature.concord import broadcast_to_goods, sector_scores
from cge.nature.encore import (
    MATERIALITY_SCALE,
    EncoreDependencies,
    materiality_to_score,
)
from cge.nature.exposure import compute_exposure
from cge.nature.fixture import encore_fixture, toy_encore_concordance
from cge.validation.toy import toy_economy


def _prov():
    return Provenance(
        source="x", source_version="1", licence="x", reference_year=2020, retrieved="2026-08-01"
    )


# -- 6.1 materiality scale + ingestion --------------------------------------------------------
def test_materiality_scale_is_monotone_and_bounded():
    """The documented VH..VL scale is strictly decreasing and lands in (0, 1] — it drives every
    downstream score, so its ordering is the load-bearing property."""
    order = ["VH", "H", "M", "L", "VL"]
    vals = [MATERIALITY_SCALE[c] for c in order]
    assert vals == sorted(vals, reverse=True)  # strictly higher class → higher score
    assert all(0.0 < v <= 1.0 for v in vals)
    assert materiality_to_score("vh") == 1.0  # case-insensitive
    with pytest.raises(ValueError, match="unknown ENCORE materiality"):
        materiality_to_score("EXTREME")


def test_encore_dependencies_validates_and_scores():
    dep = encore_fixture()
    assert dep.kind == "dependency"
    m = dep.score_matrix()  # process × service, in [0, 1]
    assert (m.values >= 0).all() and (m.values <= 1).all()
    # A published direction: agriculture very-highly depends on surface water (VH → 1.0).
    assert m.loc["agriculture", "surface_water"] == 1.0
    # Manufacturing's DIRECT water dependency is low (its exposure is upstream — tested below).
    assert m.loc["manufacturing", "surface_water"] == pytest.approx(MATERIALITY_SCALE["L"])


def test_encore_rejects_unknown_class_and_duplicate_pairs():
    with pytest.raises(ValueError, match="unknown materiality class"):
        EncoreDependencies(
            provenance=_prov(),
            ratings=pd.DataFrame(
                [("ag", "water", "SUPER")], columns=["process", "service", "materiality"]
            ),
        )
    with pytest.raises(ValueError, match="duplicate .*process, service"):
        EncoreDependencies(
            provenance=_prov(),
            ratings=pd.DataFrame(
                [("ag", "water", "VH"), ("ag", "water", "L")],
                columns=["process", "service", "materiality"],
            ),
        )


def test_encore_csv_roundtrips(tmp_path):
    from cge.nature.encore import load_encore_csv

    csv = tmp_path / "encore.csv"
    pd.DataFrame(
        [("ag", "water", "VH"), ("mfg", "water", "L")],
        columns=["process", "service", "materiality"],
    ).to_csv(csv, index=False)
    dep = load_encore_csv(str(csv), provenance=_prov())
    assert dep.processes == ["ag", "mfg"]
    assert dep.score_matrix().loc["ag", "water"] == 1.0


# -- 6.2 concordance --------------------------------------------------------------------------
def test_sector_scores_maps_processes_to_sectors():
    dep = encore_fixture()
    cmap = toy_encore_concordance()
    io, _ = toy_economy()
    s = sector_scores(dep, cmap, io.sectors.labels)
    assert set(s.index) == set(io.sectors.labels)
    assert s.loc["agriculture", "surface_water"] == 1.0  # one-to-one mapping carries the score


def test_concordance_rejects_uncovered_sector():
    dep = encore_fixture()
    # A concordance that omits 'manufacturing' must be rejected, not silently zero it.
    cmap = ConcordanceMap(
        provenance=_prov(),
        from_classification="toy",
        to_classification="encore",
        weights={"agriculture": {"agriculture": 1.0}, "energy": {"energy": 1.0}},
    )
    with pytest.raises(ValueError, match="does not cover economy sector"):
        sector_scores(dep, cmap, ["agriculture", "energy", "manufacturing"])


def test_broadcast_gives_every_region_the_same_sector_dependency():
    dep = encore_fixture()
    io, _ = toy_economy()
    s = sector_scores(dep, toy_encore_concordance(), io.sectors.labels)
    goods = list(io.A.columns)
    direct = broadcast_to_goods(s, goods)
    # A:agriculture and B:agriculture carry identical (region-independent) dependency intensity.
    assert direct.loc["A:agriculture"].equals(direct.loc["B:agriculture"])


# -- 6.3 exposure engine ----------------------------------------------------------------------
def _fixture_direct():
    io, _ = toy_economy()
    s = sector_scores(encore_fixture(), toy_encore_concordance(), io.sectors.labels)
    return io, broadcast_to_goods(s, list(io.A.columns))


def test_exposure_total_is_at_least_direct_and_bounded_both_rules():
    """The exposure invariant: a good is at least as exposed as its own direct dependency (upstream
    only ADDS exposure), and every score stays in [0, 1] — for both aggregation rules."""
    io, direct = _fixture_direct()
    for rule in ("weighted_mean", "max"):
        total, d = compute_exposure(io.A, direct, rule=rule)
        assert (total.values >= d.values - 1e-9).all(), f"{rule}: total < direct"
        assert (total.values >= -1e-9).all() and (total.values <= 1.0 + 1e-9).all()


def test_exposure_propagates_upstream_dependency():
    """Manufacturing has LOW direct water dependency but buys agricultural/energy inputs that are
    highly water-dependent, so its TOTAL exposure exceeds its direct — the supply-chain channel."""
    io, direct = _fixture_direct()
    g = "A:manufacturing"
    total_wm, d = compute_exposure(io.A, direct, rule="weighted_mean")
    total_max, _ = compute_exposure(io.A, direct, rule="max")
    assert total_wm.loc[g, "surface_water"] > d.loc[g, "surface_water"]  # strictly inherited
    # The conservative 'max' rule lifts it to the worst thing in its chain (agriculture, VH = 1.0).
    assert total_max.loc[g, "surface_water"] == pytest.approx(1.0)


def test_exposure_max_is_conservative_relative_to_direct():
    """Under 'max', every good reaching a highly-dependent input inherits that input's full score —
    a conservative screen, so max-rule totals dominate the direct scores by a wide margin here."""
    io, direct = _fixture_direct()
    total_max, d = compute_exposure(io.A, direct, rule="max")
    # In the toy economy every good's supply chain reaches agriculture (VH water), so every good's
    # surface-water exposure screens to 1.0 under the conservative rule.
    assert np.allclose(total_max["surface_water"].to_numpy(), 1.0)


def test_exposure_rejects_non_productive_economy():
    """If a good's intermediate inputs sum to ≥ 1 (no value added) the propagation would not damp;
    reject rather than diverge."""
    goods = ["g1", "g2"]
    A = pd.DataFrame([[0.6, 0.7], [0.6, 0.7]], index=goods, columns=goods)  # columns sum to 1.2
    direct = pd.DataFrame({"water": [0.5, 0.5]}, index=goods)
    with pytest.raises(ValueError, match="not productive"):
        compute_exposure(A, direct)


def test_exposure_unknown_rule_rejected():
    io, direct = _fixture_direct()
    with pytest.raises(ValueError, match="unknown aggregation rule"):
        compute_exposure(io.A, direct, rule="median")  # type: ignore[arg-type]


# -- 6.4 NatureStress → ProductivityShock translation + engine consumption --------------------
def test_nature_stress_translates_to_productivity_scaled_by_exposure():
    """A NatureStress on a service degrades each good's productivity in proportion to its exposure:
    a fully-exposed good (E=1) loses the full severity; a less-exposed good loses proportionally
    less. Delta = Π(1 − σ·E) − 1 ≤ 0, targeting the right good."""
    from cge.contracts.shocks import NatureStress
    from cge.nature.translate import nature_to_productivity

    io, direct = _fixture_direct()
    exposure, _ = compute_exposure(io.A, direct, rule="weighted_mean")
    ns = NatureStress(service="surface_water", severity=0.5)
    shocks = nature_to_productivity([ns], exposure)
    by_good = {(s.coverage_regions[0], s.coverage_sectors[0]): s.delta for s in shocks}
    # Agriculture is fully water-dependent (E=1) → loses exactly 50%.
    assert by_good[("A", "agriculture")] == pytest.approx(-0.5)
    # Manufacturing's water exposure < 1 → loses less than 50% (but still bites via upstream).
    assert -0.5 < by_good[("A", "manufacturing")] < 0.0


def test_nature_translation_rejects_unknown_service():
    from cge.contracts.shocks import NatureStress
    from cge.nature.translate import nature_to_productivity

    io, direct = _fixture_direct()
    exposure, _ = compute_exposure(io.A, direct)
    with pytest.raises(ValueError, match="not in the exposure matrix"):
        nature_to_productivity([NatureStress(service="unicorns", severity=0.3)], exposure)


def test_build_nature_shocks_runs_the_full_chain():
    """The end-to-end convenience: ENCORE + concordance + IO → NatureStress → ProductivityShocks."""
    from cge.contracts.shocks import NatureStress
    from cge.nature.translate import build_nature_shocks

    io, _ = toy_economy()
    shocks = build_nature_shocks(
        [NatureStress(service="surface_water", severity=0.5)],
        io,
        encore_fixture(),
        toy_encore_concordance(),
    )
    assert shocks and all(s.delta <= 0 for s in shocks)


def test_partial_eq_consumes_productivity_shock_as_supply_hit():
    """Engine 2 now consumes a ProductivityShock: a −20% productivity hit on one good cuts that
    good's output ~20% and emits a productivity_change row, while an unshocked good is unchanged
    (and a run with no productivity shock is byte-identical to before — no productivity_change)."""
    import cge.engines  # noqa: F401  (register engines)
    from cge.contracts.engine import registry
    from cge.contracts.shocks import ProductivityShock

    io, sat = toy_economy()
    eng = registry.get("partial_eq")
    ps = ProductivityShock(delta=-0.2, coverage_sectors=["agriculture"], coverage_regions=["A"])
    res = eng.run(data={"IOSystem": io, "SatelliteAccount": sat}, shocks=[ps], years=[2020])
    d = res.data
    vol = d[(d["variable"] == "volume_change") & (d["scenario"] == "central")]
    v = {(r.region, r.sector): r.value for r in vol.itertuples()}
    assert v[("A", "agriculture")] == pytest.approx(-0.2, abs=1e-9)  # direct supply hit
    assert v[("B", "agriculture")] == pytest.approx(0.0, abs=1e-9)  # region-scoped: B unaffected
    prod = d[d["variable"] == "productivity_change"]
    assert len(prod) == 1 and prod.iloc[0]["value"] == pytest.approx(-0.2)

    # No productivity shock → no productivity_change rows (byte-identical channel).
    from cge.contracts.shocks import CarbonPrice

    res2 = eng.run(
        data={"IOSystem": io, "SatelliteAccount": sat},
        shocks=[CarbonPrice(price=0.0)],
        years=[2020],
    )
    assert (res2.data["variable"] == "productivity_change").sum() == 0


def test_nature_scenario_end_to_end_through_engine():
    """The Phase-6 DoD path: a NatureStress scenario runs end-to-end through an economic engine and
    produces a schema-valid ResultSet whose volume response reflects exposure (agriculture hit
    hardest, manufacturing least)."""
    import cge.engines  # noqa: F401
    from cge.contracts.engine import registry
    from cge.contracts.shocks import NatureStress
    from cge.nature.translate import build_nature_shocks

    io, sat = toy_economy()
    shocks = build_nature_shocks(
        [NatureStress(service="surface_water", severity=0.4)],
        io,
        encore_fixture(),
        toy_encore_concordance(),
    )
    res = registry.get("partial_eq").run(
        data={"IOSystem": io, "SatelliteAccount": sat}, shocks=shocks, years=[2020]
    )
    res.validate_schema()  # schema-valid ResultSet (DoD)
    vol = res.data[(res.data["variable"] == "volume_change") & (res.data["scenario"] == "central")]
    v = {(r.region, r.sector): r.value for r in vol.itertuples()}
    # Every good loses output; agriculture (fully exposed) most, manufacturing least.
    assert v[("A", "agriculture")] < v[("A", "manufacturing")] < 0.0
