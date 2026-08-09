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
    with pytest.raises(ValueError, match="unrecognised materiality value"):
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


# -- Exposure precondition guards (review P1 2026-08-07) --------------------------------------
def test_exposure_rejects_nan_in_A():
    io, direct = _fixture_direct()
    A = io.A.copy()
    A.iloc[0, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        compute_exposure(A, direct)


def test_exposure_rejects_negative_coefficient():
    """A negative coefficient would let upstream SUBTRACT exposure, violating total ≥ direct — it is
    rejected so the caller must apply an explicit negative-coefficient policy (EXIOBASE has small
    negatives) rather than silently breaking the risk invariant."""
    io, direct = _fixture_direct()
    A = io.A.copy()
    A.iloc[0, 1] = -0.1
    with pytest.raises(ValueError, match="negative coefficient"):
        compute_exposure(A, direct)


def test_exposure_rejects_out_of_range_direct_scores():
    io, direct = _fixture_direct()
    bad = direct.copy()
    bad.iloc[0, 0] = 1.5
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        compute_exposure(io.A, bad)


def test_exposure_raises_on_non_convergence():
    """A run that fails to converge within max_iter is raised, never silently returned as an
    unconverged iterate. A 3-good chain needs several passes; max_iter=1 cannot converge."""
    idx = ["x", "y", "z"]
    A = pd.DataFrame([[0.0, 0.9, 0.0], [0.0, 0.0, 0.9], [0.0, 0.0, 0.0]], index=idx, columns=idx)
    d = pd.DataFrame([[0.8], [0.0], [0.0]], index=idx, columns=["w"])
    with pytest.raises(ValueError, match="did not converge"):
        compute_exposure(A, d, max_iter=1)
    # enough iterations converges cleanly
    total, _ = compute_exposure(A, d, max_iter=1000)
    assert total.loc["z", "w"] > 0.0  # exposure propagated through the full chain


def test_exposure_max_rule_materiality_threshold_screens_negligible_links():
    """The `max` rule's materiality threshold stops a negligible input coefficient from propagating
    the global maximum across the economy (review P1: an unthresholded dense MRIO collapses to the
    global max)."""
    idx = ["dirty", "clean"]
    # clean uses only a tiny 1e-6 amount of dirty as input.
    A = pd.DataFrame([[0.0, 1e-6], [0.0, 0.0]], index=idx, columns=idx)
    d = pd.DataFrame([[1.0], [0.0]], index=idx, columns=["w"])
    no_thresh, _ = compute_exposure(A, d, rule="max", max_link_threshold=0.0)
    thresh, _ = compute_exposure(A, d, rule="max", max_link_threshold=1e-3)
    assert no_thresh.loc["clean", "w"] == 1.0  # negligible link still propagates the max
    assert thresh.loc["clean", "w"] == 0.0  # screened out above the threshold


def test_sector_scores_rejects_impact_kind_object():
    """Review P1 (2026-08-07): ENCORE impacts (the economy's pressure ON nature) must NOT feed into
    the dependency→productivity channel — that inverts the causality. sector_scores rejects a
    kind='impact' object rather than silently converting impacts into productivity losses."""
    from cge.nature.concord import sector_scores

    impact = encore_fixture().model_copy(update={"kind": "impact"})
    with pytest.raises(ValueError, match="dependency-kind"):
        sector_scores(impact, toy_encore_concordance(), ["agriculture", "energy", "manufacturing"])


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


def test_build_nature_shocks_incidence_direct_vs_total():
    """Review P1 (2026-08-07): incidence='direct' shocks each good only for its OWN dependency (the
    engine transmits upstream), while 'total' carries the full direct+upstream exposure. A good with
    upstream-inherited exposure (manufacturing) is shocked LESS under direct; a fully directly-
    dependent good (agriculture, E=1) is identical under both."""
    from cge.contracts.shocks import NatureStress
    from cge.nature.translate import build_nature_shocks

    io, _ = toy_economy()
    ns = [NatureStress(service="surface_water", severity=0.5)]
    total = build_nature_shocks(
        ns, io, encore_fixture(), toy_encore_concordance(), incidence="total"
    )
    direct = build_nature_shocks(
        ns, io, encore_fixture(), toy_encore_concordance(), incidence="direct"
    )

    def _by_good(shocks):
        return {(s.coverage_regions[0], s.coverage_sectors[0]): s.delta for s in shocks}

    t, d = _by_good(total), _by_good(direct)
    # Agriculture is fully water-dependent directly (E=1) → identical under both.
    assert t[("A", "agriculture")] == pytest.approx(d[("A", "agriculture")])
    # Manufacturing's water exposure is largely upstream-inherited → smaller (less negative) shock
    # under direct incidence than total.
    assert d[("A", "manufacturing")] > t[("A", "manufacturing")]


def test_incidence_by_engine_defaults():
    """The CGE (endogenous supply transmission) defaults to direct incidence; partial_eq (no
    transmission) to total — the map the runner/GUI use to avoid double-counting upstream."""
    from cge.nature import INCIDENCE_BY_ENGINE

    assert INCIDENCE_BY_ENGINE["cge_static"] == "direct"
    assert INCIDENCE_BY_ENGINE["io_price"] == "direct"
    assert INCIDENCE_BY_ENGINE["partial_eq"] == "total"


def test_nature_stress_runs_through_standard_runner_with_manifest_provenance():
    """Review P1 (2026-08-07): a NatureStress scenario runs through the STANDARD runner (not a
    GUI-only path), and the manifest records the nature inputs — ENCORE + concordance hashes,
    materiality scale, exposure rule and incidence — so a nature run is reconstructible from its
    manifest (the 'explicit and auditable' promise)."""
    from cge.contracts.shocks import NatureStress
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    sc = Scenario(
        name="nat",
        engine="partial_eq",
        years=[2020],
        shocks=[NatureStress(service="surface_water", severity=0.4)],
    )
    res = run_scenario(sc, data_source="toy")
    assert (res.data["variable"] == "volume_change").any()
    nat = res.manifest.assumptions.get("nature")
    assert nat is not None
    for key in (
        "encore_content_hash",
        "concordance_content_hash",
        "materiality_scale",
        "exposure_rule",
        "incidence",
        "stresses",
    ):
        assert key in nat, f"manifest nature provenance missing {key!r}"
    assert nat["incidence"] == "total"  # partial_eq default


def test_nature_stress_cge_uses_direct_incidence_in_manifest():
    from cge.contracts.shocks import NatureStress
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    sc = Scenario(
        name="nat",
        engine="cge_static",
        years=[2020],
        shocks=[NatureStress(service="surface_water", severity=0.4)],
    )
    res = run_scenario(sc, data_source="toy")
    assert res.manifest.assumptions["nature"]["incidence"] == "direct"


# -- Second review (2026-08-07 round 2): the four P1 counterexamples -----------------------------
def test_nature_time_path_flows_through_per_year():
    """Review P1 round 2: a NatureStress time path must produce a per-YEAR productivity shock, not
    collapse to the scalar severity. A path of 10%→50% (2020→2030) gives −10% then −50%, and two
    scenarios differing only in the 2030 endpoint get DIFFERENT scenario hashes (no provenance
    collision)."""
    from cge.contracts.shocks import NatureStress
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    ns1 = NatureStress(service="surface_water", severity=0.1, path={2020: 0.1, 2030: 0.5})
    ns2 = NatureStress(service="surface_water", severity=0.1, path={2020: 0.1, 2030: 0.9})
    sc1 = Scenario(name="p", engine="partial_eq", years=[2020, 2030], shocks=[ns1])
    sc2 = Scenario(name="p", engine="partial_eq", years=[2020, 2030], shocks=[ns2])
    r1 = run_scenario(sc1, data_source="toy")
    r2 = run_scenario(sc2, data_source="toy")
    vol = r1.data[
        (r1.data["variable"] == "volume_change")
        & (r1.data["sector"] == "agriculture")
        & (r1.data["scenario"] == "central")
    ]
    by_year = {int(r.year): r.value for r in vol.itertuples()}
    assert by_year[2020] == pytest.approx(-0.1)  # agriculture E=1 → loss = severity
    assert by_year[2030] == pytest.approx(-0.5)  # the PATH value, not the scalar 0.1
    assert r1.manifest.scenario_hash != r2.manifest.scenario_hash  # no collision


def test_region_scoped_shock_is_not_economy_wide_in_collapsed_cge():
    """Review P1 round 2: in the collapsed single-region CGE, a shock on ONE region must not behave
    like an economy-wide shock. Unshocked regions contribute θ=1 to the average, so a −20% hit on
    region A alone gives θ_agri = mean(0.8, 1.0) = 0.9, distinct from a both-regions shock (0.8)."""
    from cge.contracts.shocks import ProductivityShock
    from cge.engines.cge_static.engine import _productivity_by_sector

    sectors = ["agriculture", "energy"]
    # Both regions present (via energy shocks); agriculture degraded only in A.
    a_only = [
        ProductivityShock(delta=-0.2, coverage_sectors=["agriculture"], coverage_regions=["A"]),
        ProductivityShock(delta=-0.05, coverage_sectors=["energy"], coverage_regions=["A"]),
        ProductivityShock(delta=-0.05, coverage_sectors=["energy"], coverage_regions=["B"]),
    ]
    both = a_only + [
        ProductivityShock(delta=-0.2, coverage_sectors=["agriculture"], coverage_regions=["B"])
    ]
    theta_a = _productivity_by_sector(a_only, sectors, 2020)[0]
    theta_both = _productivity_by_sector(both, sectors, 2020)[0]
    assert theta_a == pytest.approx(0.9)  # A shocked, B at θ=1 → mean
    assert theta_both == pytest.approx(0.8)  # both shocked
    assert theta_a != theta_both  # region scope actually matters


def test_nature_pipeline_runs_with_encore_injected_via_overrides():
    """Review P1 round 2: the standard pipeline must not require the data source to carry ENCORE —
    a caller can inject EncoreDependencies + ConcordanceMap via data_overrides (so a real stored
    build, which does not persist nature data, can still run a nature scenario)."""
    from cge.contracts.shocks import NatureStress
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    sc = Scenario(
        name="n",
        engine="partial_eq",
        years=[2020],
        shocks=[NatureStress(service="surface_water", severity=0.4)],
    )
    res = run_scenario(
        sc,
        data_source="toy",
        data_overrides={
            "EncoreDependencies": encore_fixture(),
            "ConcordanceMap": toy_encore_concordance(),
        },
    )
    assert "nature" in res.manifest.assumptions


def test_exposure_rejects_explicit_nan_direct_but_fills_absent_goods():
    """Review P1 round 2: an EXPLICIT NaN in supplied direct data is rejected (not silently zeroed),
    while a good simply ABSENT from the input still fills to 0 direct dependency."""
    idx = ["x", "y"]
    A = pd.DataFrame([[0.0, 0.3], [0.2, 0.0]], index=idx, columns=idx)
    with pytest.raises(ValueError, match="non-finite"):
        compute_exposure(A, pd.DataFrame([[np.nan], [0.1]], index=idx, columns=["w"]))
    # 'y' absent from the supplied direct scores → filled to 0, not rejected.
    total, d = compute_exposure(A, pd.DataFrame([[0.5]], index=["x"], columns=["w"]))
    assert d.loc["y", "w"] == 0.0


def test_incidence_invalid_value_rejected():
    """Review P2: an invalid incidence string must not silently fall through to 'total'."""
    from cge.contracts.shocks import NatureStress
    from cge.nature.translate import build_nature_shocks

    io, _ = toy_economy()
    with pytest.raises(ValueError, match="unknown incidence"):
        build_nature_shocks(
            [NatureStress(service="surface_water", severity=0.4)],
            io,
            encore_fixture(),
            toy_encore_concordance(),
            incidence="typo",  # type: ignore[arg-type]
        )


def test_duplicate_service_stresses_rejected():
    """Review P2: two stresses on the SAME service compound as if independent — reject rather than
    silently double-count (composition is across DISTINCT services)."""
    from cge.contracts.shocks import NatureStress
    from cge.nature.translate import build_nature_shocks

    io, _ = toy_economy()
    with pytest.raises(ValueError, match="duplicate NatureStress service"):
        build_nature_shocks(
            [
                NatureStress(service="surface_water", severity=0.3),
                NatureStress(service="surface_water", severity=0.2),
            ],
            io,
            encore_fixture(),
            toy_encore_concordance(),
        )


def test_shock_validation_rejects_invalid_values():
    """Review P2: ProductivityShock rejects NaN and delta < −1; NatureStress rejects a blank service
    and out-of-range path levels."""
    from cge.contracts.shocks import NatureStress, ProductivityShock

    with pytest.raises(ValueError, match="≥ −1"):
        ProductivityShock(delta=-2.0)
    with pytest.raises(ValueError, match="finite"):
        ProductivityShock(delta=float("nan"))
    with pytest.raises(ValueError, match="non-empty"):
        NatureStress(service="   ", severity=0.3)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        NatureStress(service="surface_water", severity=0.3, path={2020: 1.5})


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


# -- Real ENCORE knowledge base ingestion (2026-08-09) — golden-file tests ----------------------
# These run against the real CC BY-SA 4.0 ENCORE data vendored under data/encore/. Skip cleanly if
# that directory is absent (a checkout that excluded the data), so the suite still passes.
import pytest as _pytest  # noqa: E402

from cge.nature.real import encore_data_available  # noqa: E402

_needs_encore = _pytest.mark.skipif(
    not encore_data_available(), reason="vendored ENCORE data (data/encore/) not present"
)


@_needs_encore
def test_real_encore_dependency_ingestion_shape_and_values():
    """The real May-2026 dependency table ingests to the expected shape, real vocabulary, and known
    cell values — a golden-file guard so a re-vendored file or a broken melt is caught."""
    from cge.nature.real import real_encore_dependencies

    dep = real_encore_dependencies()
    assert dep.kind == "dependency"
    assert len(dep.processes) == 271
    assert len(dep.services) == 25
    # Real ENCORE service vocabulary (not our toy labels).
    for svc in ("Pollination", "Global climate regulation", "Water supply"):
        assert svc in dep.services
    sm = dep.score_matrix()
    # A_1_14_141 (raising cattle) is VH on Biomass provisioning → 1.0.
    assert sm.loc["A_1_14_141", "Biomass provisioning"] == 1.0


@_needs_encore
def test_real_encore_nd_kept_distinct_not_zeroed():
    """ND (No Data) cells are tracked distinctly (nd_mask) and NOT silently 0 — the review's core
    concern. data_coverage reports the rated fraction (<1 because real ND cells exist)."""
    from cge.nature.real import real_encore_dependencies

    dep = real_encore_dependencies()
    nd = dep.nd_mask()
    assert nd.values.sum() > 0  # the real file genuinely has ND cells
    assert 0.0 < dep.data_coverage() < 1.0  # some rated, some ND
    # An ND cell scores 0 in the matrix BUT is flagged in nd_mask (known-unknown, not a zero).
    nd_cells = [(p, s) for p in dep.processes for s in dep.services if nd.loc[p, s]]
    p, s = nd_cells[0]
    assert dep.score_matrix().loc[p, s] == 0.0 and nd.loc[p, s]


@_needs_encore
def test_real_encore_process_ids_unique_and_energy_split():
    """The ISIC code alone is not unique (ENCORE splits some codes into finer production processes);
    the adapter disambiguates by the finest ISIC name, so process ids are unique and the energy
    split (D_35_351 → fossil/nuclear/hydro/solar/wind…) is preserved."""
    from cge.nature.real import real_encore_dependencies

    dep = real_encore_dependencies()
    assert len(dep.processes) == len(set(dep.processes))  # unique
    energy = [p for p in dep.processes if p.startswith("D_35_351")]
    assert len(energy) >= 8  # split into distinct generation processes
    assert any("Fossil" in p for p in energy) and any("Solar" in p for p in energy)


@_needs_encore
def test_real_encore_pressures_typed_impact():
    """The pressure/impact table ingests separately, typed kind='impact' — so it is NOT mistaken for
    a dependency (feeding impacts into the dependency→productivity channel inverts causality)."""
    from cge.nature.concord import sector_scores
    from cge.nature.real import real_encore_pressures

    pr = real_encore_pressures()
    assert pr.kind == "impact"
    assert len(pr.services) >= 10  # pressure/impact drivers
    # And the dependency pipeline refuses it (kind guard from the earlier review).
    with _pytest.raises(ValueError, match="dependency-kind"):
        sector_scores(pr, toy_encore_concordance(), ["agriculture"])


def test_encore_ratings_wide_drops_na_keeps_nd(tmp_path):
    """Unit test of the wide-melt N/A-vs-ND rule on a tiny synthetic wide file (no real data):
    a rating is kept, ND is kept as ND, and N/A / blank are dropped (→ absent = 0)."""
    import pandas as pd

    from cge.contracts.data_objects import Provenance
    from cge.nature.encore import load_encore_ratings_wide

    wide = pd.DataFrame(
        {
            "ISIC Unique code": ["X_1", "X_2"],
            "ISIC Section": ["s", "s"],
            "ISIC Division": ["d", "d"],
            "ISIC Group": ["g1", "g2"],
            "ISIC Class": ["", ""],
            "Pollination": ["VH", "ND"],
            "Water supply": ["N/A", ""],
        }
    )
    csv = tmp_path / "wide.csv"
    wide.to_csv(csv, index=False)
    prov = Provenance(
        source="t", source_version="1", licence="x", reference_year=2026, retrieved="2026-08-09"
    )
    dep = load_encore_ratings_wide(str(csv), provenance=prov)
    rows = {(r.process, r.service): r.materiality for r in dep.ratings.itertuples()}
    assert rows[("X_1", "Pollination")] == "VH"  # rating kept
    assert rows[("X_2", "Pollination")] == "ND"  # ND kept distinct
    assert ("X_1", "Water supply") not in rows  # N/A dropped
    assert ("X_2", "Water supply") not in rows  # blank dropped


@_needs_encore
def test_real_exiobase_encore_concordance_builds_and_covers_all_sectors():
    """The real EXIOBASE→ENCORE concordance builds from the vendored crosswalk: all 162 EXIOBASE
    sectors resolve to ≥1 ENCORE process (ISIC-level rollback), weights sum to 1, and every mapped
    process is a real ENCORE process id."""
    from cge.nature.real import real_encore_concordance, real_encore_dependencies

    dep = real_encore_dependencies()
    cmap, audit = real_encore_concordance()
    assert audit.unresolved_sectors == []  # every EXIOBASE sector resolved
    assert audit.n_exiobase_sectors == len(cmap.weights) == 162
    valid = set(dep.processes)
    for sector, w in cmap.weights.items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, sector  # normalised
        assert set(w) <= valid  # maps only to real ENCORE process ids
    assert audit.multi_process_sectors  # some sectors genuinely map to several processes


@_needs_encore
def test_real_encore_dependency_scores_for_real_exiobase_sector():
    """End-to-end on REAL data: a real EXIOBASE agricultural sector, mapped through the real
    concordance to real ENCORE processes, comes out highly water/biomass-dependent (a sanity check
    that the whole ISIC→ENCORE→score chain carries meaning, not just runs)."""
    from cge.nature.concord import sector_scores
    from cge.nature.real import real_encore_concordance, real_encore_dependencies

    dep = real_encore_dependencies()
    cmap, _ = real_encore_concordance()
    sector = "Cultivation of cereal grains nec"
    assert sector in cmap.weights
    scores = sector_scores(dep, cmap, [sector])
    row = scores.loc[sector]
    # Agriculture depends highly on water-related services (real ENCORE ratings, not toy).
    assert row["Water purification"] > 0.8
    assert row["Biomass provisioning"] > 0.5
    # A financial sector, by contrast, is only weakly dependent.
    fin = next((s for s in cmap.weights if "financial intermediation" in s.lower()), None)
    if fin:
        assert sector_scores(dep, cmap, [fin]).loc[fin].max() < row.max()
