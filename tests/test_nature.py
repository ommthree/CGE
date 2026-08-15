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


def tmp_json(obj) -> str:
    """Write ``obj`` to a temp JSON file and return its path (for artifact-validation tests)."""
    import json
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".json")
    with open(fd, "w") as fh:
        json.dump(obj, fh)
    return path


def _read_json(path: str):
    import json

    with open(path) as fh:
        return json.load(fh)


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


def test_exposure_nonzero_threshold_rejected_under_weighted_mean():
    """Review P2 (2026-08-10): the max-link threshold applies only to 'max'; a nonzero value
    under 'weighted_mean' is a no-op and must be REJECTED as an inapplicable control, not silently
    recorded (0 and 0.999 previously gave identical results)."""
    idx = ["x", "y"]
    A = pd.DataFrame([[0.0, 0.3], [0.2, 0.0]], index=idx, columns=idx)
    d = pd.DataFrame([[0.5], [0.1]], index=idx, columns=["w"])
    with pytest.raises(ValueError, match="no effect under the 'weighted_mean'"):
        compute_exposure(A, d, rule="weighted_mean", max_link_threshold=0.5)
    # A zero threshold is always fine (the default no-op).
    compute_exposure(A, d, rule="weighted_mean", max_link_threshold=0.0)


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


def test_collapsed_cge_rejects_region_scoped_shock_isolated():
    """Review P1 round 3 (2026-08-09): the earlier "average across regions" fix was masked by a test
    that added unrelated shocks to populate the region universe. The real defect: with an ISOLATED
    region-A-only shock the universe was {A}, so it came out identical to an economy-wide shock. The
    correct contract is to REJECT a region-scoped shock on the single-region collapsed model —
    here with an isolated shock (no other shocks to mask it)."""
    from cge.contracts.shocks import ProductivityShock
    from cge.engines.cge_static.engine import _assert_no_region_scoped_productivity

    a_only = [
        ProductivityShock(delta=-0.2, coverage_sectors=["agriculture"], coverage_regions=["A"])
    ]
    with pytest.raises(ValueError, match="single-region"):
        _assert_no_region_scoped_productivity(a_only, "closed")
    # An economy-wide shock (no region coverage) is accepted.
    _assert_no_region_scoped_productivity(
        [ProductivityShock(delta=-0.2, coverage_sectors=["agriculture"])], "closed"
    )


def test_typoed_stress_coverage_rejected_not_silent_baseline():
    """Review P1 round 5 (2026-08-13): a misspelled coverage_region/sector matches no good and
    silently produces zero shocks — a "successful" baseline run with no sign it was invalid. Both
    are now rejected before translation (the service name was validated; coverage was not)."""
    from cge.contracts.shocks import NatureStress
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    for cov in ({"coverage_regions": ["TYPO"]}, {"coverage_sectors": ["TYPO"]}):
        sc = Scenario(
            name="t",
            engine="partial_eq",
            years=[2020],
            shocks=[NatureStress(service="surface_water", severity=0.4, **cov)],
        )
        with pytest.raises(ValueError, match="unknown coverage"):
            run_scenario(sc, data_source="toy")
    # Valid coverage (toy region A, sector agriculture) still runs.
    ok = Scenario(
        name="ok",
        engine="partial_eq",
        years=[2020],
        shocks=[NatureStress(service="surface_water", severity=0.4, coverage_regions=["A"])],
    )
    assert run_scenario(ok, data_source="toy").manifest.assumptions["nature"][
        "derived_productivity_shocks"
    ]


def test_direct_incidence_skips_total_and_rejects_inapplicable_controls():
    """Review P1 round 5: under incidence='direct' the shock uses only each good's dependency, so
    the upstream propagation (and its rule/threshold) is not computed — a non-default rule/threshold
    is then an inapplicable control and is rejected, and the direct result is unchanged."""
    from cge.contracts.shocks import NatureStress
    from cge.nature.fixture import encore_fixture, toy_encore_concordance
    from cge.nature.translate import build_nature_shocks

    io, _ = toy_economy()
    ns = [NatureStress(service="surface_water", severity=0.5)]
    # direct with defaults works; a nonzero threshold / non-default rule is rejected.
    assert build_nature_shocks(
        ns, io, encore_fixture(), toy_encore_concordance(), incidence="direct"
    )
    with pytest.raises(ValueError, match="NO effect under incidence='direct'"):
        build_nature_shocks(
            ns, io, encore_fixture(), toy_encore_concordance(), incidence="direct", rule="max"
        )


def test_runner_rejects_region_scoped_nature_stress_end_to_end():
    """Review P1 round 4 (2026-08-10): the round-3 engine guard was BYPASSED because the runner
    stripped region coverage (collapse) BEFORE the engine saw it — so a region-scoped NatureStress
    ran identically to an economy-wide one END TO END. This tests the REAL boundary (run_scenario):
    a region-scoped NatureStress against the single-region CGE is rejected; economy-wide runs;
    and the two are no longer identical (the defect was that they were)."""
    from cge.contracts.shocks import NatureStress
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    region_scoped = Scenario(
        name="a",
        engine="cge_static",
        years=[2020],
        shocks=[NatureStress(service="surface_water", severity=0.4, coverage_regions=["A"])],
    )
    with pytest.raises(ValueError, match="region-scoped NatureStress"):
        run_scenario(region_scoped, data_source="toy")
    # Economy-wide (no region coverage) runs fine.
    economy_wide = Scenario(
        name="g",
        engine="cge_static",
        years=[2020],
        shocks=[NatureStress(service="surface_water", severity=0.4)],
    )
    res = run_scenario(economy_wide, data_source="toy")
    assert (res.data["variable"] == "volume_change").any()


def test_runner_multi_region_flag_from_overrides_not_collapsed():
    """Review P1 round 4: multi_region arrives via data_overrides (merged after preprocessing), so
    reading only data.get('multi_region') wrongly collapsed a real multi run. The runner must read
    the flag from overrides too — so a region-scoped NatureStress is NOT rejected when multi_region
    is requested through data_overrides."""
    from cge.contracts.shocks import NatureStress
    from cge.runner import _preprocess_nature
    from cge.scenarios.loader import Scenario
    from cge.validation.toy import toy_economy

    io, _ = toy_economy()
    sc = Scenario(
        name="m",
        engine="cge_static",
        years=[2020],
        shocks=[NatureStress(service="surface_water", severity=0.4, coverage_regions=["A"])],
    )
    # With multi_region via overrides, the region-scoped stress must NOT be rejected (it collapses
    # only when single-region). It should preprocess without raising the single-region error.
    from cge.contracts.engine import registry

    shocks, stamp = _preprocess_nature(
        sc,
        {
            "IOSystem": io,
            "EncoreDependencies": encore_fixture(),
            "ConcordanceMap": toy_encore_concordance(),
        },
        registry.get("cge_static"),
        {"multi_region": True},
    )
    assert stamp is not None and stamp["collapse_regions"] is False


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


def test_duplicate_service_same_good_rejected_but_disjoint_allowed():
    """Review P2 (2026-08-07, round 5): two stresses on the SAME service that both
    reach the SAME good compound as if independent — rejected. But DISJOINT regional coverage on one
    service (the natural way to express heterogeneous regional degradation) is ALLOWED."""
    from cge.contracts.shocks import NatureStress
    from cge.nature.translate import build_nature_shocks

    io, _ = toy_economy()
    # Both economy-wide → both hit every good → conflict.
    with pytest.raises(ValueError, match="both cover good"):
        build_nature_shocks(
            [
                NatureStress(service="surface_water", severity=0.3),
                NatureStress(service="surface_water", severity=0.2),
            ],
            io,
            encore_fixture(),
            toy_encore_concordance(),
        )
    # Disjoint regions (A vs B) on the same service → no shared good → allowed.
    assert build_nature_shocks(
        [
            NatureStress(service="surface_water", severity=0.2, coverage_regions=["A"]),
            NatureStress(service="surface_water", severity=0.6, coverage_regions=["B"]),
        ],
        io,
        encore_fixture(),
        toy_encore_concordance(),
    )


def test_water_supply_overlap_rejected_by_default_and_opt_out():
    """Review P1 round 3 (2026-08-09): 'Water supply' is a COMBINED ENCORE service that duplicates
    Water purification / Water flow regulation (ENCORE Explanatory note #1). Stressing it together
    with a component double-counts, so it is rejected by default — with an explicit opt-out."""
    import pandas as pd

    from cge.contracts.shocks import NatureStress
    from cge.nature.translate import nature_to_productivity

    exp = pd.DataFrame(
        0.5, index=["R:s"], columns=["Water supply", "Water purification", "Water flow regulation"]
    )
    combo = [
        NatureStress(service="Water supply", severity=0.3),
        NatureStress(service="Water purification", severity=0.3),
    ]
    with pytest.raises(ValueError, match="Water supply.*COMBINED|overlap"):
        nature_to_productivity(combo, exp)
    # Opt-out allows it; the combined service alone is always fine.
    assert nature_to_productivity(combo, exp, allow_water_overlap=True)
    assert nature_to_productivity([NatureStress(service="Water supply", severity=0.3)], exp)
    # DISJOINT coverage (combined in A, component in B) — no good gets both — is allowed (round 5).
    exp2 = pd.DataFrame(
        0.5,
        index=["A:s", "B:s"],
        columns=["Water supply", "Water purification", "Water flow regulation"],
    )
    assert nature_to_productivity(
        [
            NatureStress(service="Water supply", severity=0.3, coverage_regions=["A"]),
            NatureStress(service="Water purification", severity=0.3, coverage_regions=["B"]),
        ],
        exp2,
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


def _supply_shares_present() -> bool:
    from cge.nature.real import supply_shares_available

    return supply_shares_available()


_needs_supply_shares = _pytest.mark.skipif(
    not _supply_shares_present(),
    reason="vendored EXIOBASE supply-share artifact (data/exiobase/) not present",
)


@_needs_encore
@_needs_supply_shares
def test_product_bridge_uses_observed_supply_shares_not_prefix_guess():
    """Review P1-methodology round 7 (2026-08-14): the pxp product bridge must weight producing
    industries by the OBSERVED EXIOBASE MRSUT supply shares, not a code-prefix guess. The six
    products the reviewer flagged (which previously all received byte-identical prefix-inferred
    chemical weights) must now resolve by supply share to their real dominant producer AND be
    mutually distinct."""
    import json

    from cge.nature.real import real_encore_concordance_products

    cmap, uncovered, audit = real_encore_concordance_products(with_audit=True)
    assert not uncovered and len(cmap.weights) == 200
    # Dominant producing industry per flagged product, from the observed supply matrix.
    expected_dominant = {
        "Motor Gasoline": "Petroleum Refinery",
        "Natural Gas Liquids": "Extraction of natural gas and services related to natural gas "
        "extraction, excluding surveying",
        "Biodiesels": "Chemicals nec",
        "Biogasoline": "Chemicals nec",
        "Charcoal": "Chemicals nec",
        "Additives/Blending Components": "Chemicals nec",
        "Other Liquid Biofuels": "Chemicals nec",
        "Electricity by coal": "Production of electricity by coal",
        "Electricity nec": "Production of electricity nec",
    }
    for product, dominant in expected_dominant.items():
        entry = audit.entries[product]
        assert entry.method == "supply-share", f"{product} did not use observed supply shares"
        top = max(entry.industry_weights.items(), key=lambda kv: kv[1])[0]
        assert top == dominant, f"{product}: dominant industry {top!r} != expected {dominant!r}"

    # The five biofuel/chemical products that were byte-identical under the prefix method must now
    # have DISTINCT ENCORE weight vectors (the reviewer's core complaint).
    biofuels = [
        "Biodiesels",
        "Biogasoline",
        "Charcoal",
        "Additives/Blending Components",
        "Other Liquid Biofuels",
    ]
    hashes = {json.dumps(cmap.weights[p], sort_keys=True) for p in biofuels}
    assert len(hashes) == len(biofuels), "biofuel products still collapse to identical weights"


@_needs_encore
@_needs_supply_shares
def test_zero_supply_products_fall_back_and_are_audited():
    """Products with NO market supply in the MRSUT (recycling/treatment residuals, extra-territorial
    bodies) legitimately can't use supply shares — they must fall back to the code-prefix method and
    say so in the audit, not silently drop out (review P1-methodology round 7)."""
    from cge.nature.real import real_encore_concordance_products

    _cmap, _uncovered, audit = real_encore_concordance_products(with_audit=True)
    # There are exactly the 16 supply-less products; all present and flagged as fallbacks.
    fallbacks = {p for p, e in audit.entries.items() if e.method == "code-prefix-fallback"}
    assert len(fallbacks) == 16
    manure = "Manure (conventional treatment)"
    assert manure in fallbacks
    assert "no market supply" in audit.entries[manure].fallback_reason
    assert audit.n_supply_share == 184  # the rest use observed shares


@_needs_supply_shares
def test_supply_share_artifact_validation_rejects_incomplete():
    """A malformed/incomplete artifact must FAIL on load, not silently degrade the bridge back to
    the prefix method (review P2 round 8 2026-08-14). Removing a product without declaring it
    zero-supply — the reviewer's reproduction — is rejected."""
    from cge.nature.real import SupplyShareValidationError, load_supply_shares

    art = _read_json("data/exiobase/supply_shares_2019.json")
    # (1) product missing from BOTH shares and zero_supply → not the 200 pxp set → rejected.
    del art["shares"]["Motor Gasoline"]
    with pytest.raises(SupplyShareValidationError, match="do not match pymrio|missing"):
        load_supply_shares(path=tmp_json(art))
    # (2) a FAKE product name preserving the COUNT is still rejected — identity, not count (P2
    # round 9): the reviewer swapped Motor Gasoline for a fake name and it slipped through before.
    art_fake = _read_json("data/exiobase/supply_shares_2019.json")
    art_fake["shares"]["FAKE PRODUCT XYZ"] = art_fake["shares"].pop("Motor Gasoline")
    with pytest.raises(SupplyShareValidationError, match="unknown name|do not match pymrio"):
        load_supply_shares(path=tmp_json(art_fake))
    # (3) weights that don't sum to 1 → rejected.
    art2 = _read_json("data/exiobase/supply_shares_2019.json")
    k = next(iter(art2["shares"]))
    art2["shares"][k] = {next(iter(art2["shares"][k])): 0.5}
    with pytest.raises(SupplyShareValidationError, match="sum to"):
        load_supply_shares(path=tmp_json(art2))


@_needs_supply_shares
def test_supply_shares_sut_year_identity_is_enforced():
    """A file whose declared sut_year disagrees with the requested year is rejected — the file's
    content and its year binding must agree (review P2 round 9 2026-08-15)."""
    from cge.nature.real import SupplyShareValidationError, load_supply_shares

    art = _read_json("data/exiobase/supply_shares_2019.json")
    art["provenance"]["sut_year"] = 1995  # content says 1995…
    with pytest.raises(SupplyShareValidationError, match="sut_year"):
        load_supply_shares(2019, path=tmp_json(art))  # …but we asked for 2019


@_needs_supply_shares
def test_supply_shares_year_binding_falls_back_visibly():
    """A build year without its own artifact falls back to the default year, but the mismatch is
    recorded in provenance — never silent (review P2 round 8 2026-08-14)."""
    from cge.nature.real import load_supply_shares

    _shares, prov = load_supply_shares(2020)  # no 2020 artifact → falls back to 2019
    assert "year_fallback" in prov and "2020" in prov["year_fallback"]
    # The default year loads with NO fallback note.
    _s19, prov19 = load_supply_shares(2019)
    assert "year_fallback" not in prov19


@_needs_encore
@_needs_supply_shares
def test_nature_manifest_carries_concordance_version_and_year_fallback(tmp_path):
    """The run manifest's nature stamp must record the concordance version — which carries the MRSUT
    supply-share version AND any year-fallback disclosure — not just a source label + hash (review
    P2 round 9 2026-08-15: 'full nature provenance')."""
    import numpy as np

    from cge.contracts.data_objects import Classification, IOSystem, SatelliteAccount
    from cge.contracts.shocks import NatureStress
    from cge.data.metadata import BuildMeta
    from cge.data.store import DataStore
    from cge.nature.real import real_encore_concordance_products, real_encore_dependencies
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    dep = real_encore_dependencies()
    cmap, _unc = real_encore_concordance_products(year=2020)  # 2020 → year fallback to 2019
    secs = list(cmap.weights)[:2]
    labels = [f"R:{s}" for s in secs]
    io = IOSystem(
        provenance=_prov(),
        sectors=Classification(name="e", kind="sector", labels=secs),
        regions=Classification(name="r", kind="region", labels=["R"]),
        A=pd.DataFrame(np.full((2, 2), 0.05), index=labels, columns=labels),
        final_demand=pd.DataFrame({"final_demand": [100.0, 100.0]}, index=labels),
        unit="MEUR",
        currency="EUR",
    )
    sat = SatelliteAccount(
        provenance=_prov(),
        name="GHG",
        units={"CO2": "t/MEUR", "CO2e": "tCO2e/MEUR"},
        data=pd.DataFrame(
            {labels[0]: [1000.0, 1000.0], labels[1]: [1000.0, 1000.0]}, index=["CO2", "CO2e"]
        ),
    )
    store = DataStore(tmp_path)
    store.save(
        meta=BuildMeta(
            build_id="b",
            source="s",
            source_version="v",
            reference_year=2020,
            licence="x",
            currency="EUR",
            monetary_unit="MEUR",
            retrieved="2026-08-15",
        ),
        io=io,
        satellites=[sat],
        encore=dep,
        concordance=cmap,
    )
    sc = Scenario(
        name="r",
        engine="partial_eq",
        years=[2020],
        shocks=[NatureStress(service="Water purification", severity=0.4)],
    )
    res = run_scenario(sc, data_source="b", store=store)
    nat = res.manifest.assumptions["nature"]
    assert "concordance_version" in nat
    assert "MRSUT" in nat["concordance_version"]  # supply-share version survived
    assert "2020" in nat["concordance_version"] and "FALLBACK" in nat["concordance_version"]


@_needs_encore
def test_ixi_build_unaffected_by_damaged_product_artifact(monkeypatch):
    """An ixi build whose labels all match the industry concordance must NOT construct the product
    bridge, so a damaged pxp supply-share artifact can't block it (review P3 round 9 2026-08-15)."""
    import cge.nature.real as R
    from cge.data.build import _nature_for_sectors

    ind, _ = R.real_encore_concordance_industries()
    labels = list(ind.weights)  # all 163 industry labels

    def _boom(*a, **k):
        raise AssertionError("product bridge should not be built for a fully-covered ixi build")

    monkeypatch.setattr(R, "real_encore_concordance_products", _boom)
    enc, conc = _nature_for_sectors(labels, policy="required", reference_year=2019)
    assert enc is not None and set(conc.weights) == set(labels)


@_needs_encore
@_needs_supply_shares
def test_persisted_concordance_provenance_carries_supply_share_version():
    """The PERSISTED product-bridge concordance provenance must record the MRSUT version (review P2
    round 8: previously only the discarded audit carried it, so a stored concordance concealed which
    SUT year produced its weights)."""
    from cge.nature.real import real_encore_concordance_products

    cmap, _uncovered = real_encore_concordance_products(year=2019)
    assert "MRSUT" in cmap.provenance.source_version
    assert "supply-share" in cmap.provenance.source_version
    # A year fallback surfaces on the concordance provenance too.
    cmap20, _ = real_encore_concordance_products(year=2020)
    assert "2020" in cmap20.provenance.source_version


@_needs_encore
def test_product_bridge_falls_back_cleanly_without_supply_shares():
    """With no supply-share artifact (MRSUT-absent checkout), the bridge must still cover all 200
    products via the code-prefix fallback — nature runs without the multi-GB download (round-6
    behaviour preserved)."""
    from cge.nature.concordance_build import bridge_to_products, pxp_to_ixi_industries
    from cge.nature.real import real_encore_concordance

    industry_conc, _ = real_encore_concordance()
    cmap, uncovered, audit = bridge_to_products(
        industry_conc, pxp_to_ixi_industries(), supply_shares=None
    )
    assert not uncovered and len(cmap.weights) == 200
    assert audit.n_supply_share == 0 and audit.n_fallback == 200


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


@_needs_encore
def test_sector_nd_mask_flags_all_nd_cells_not_as_zero():
    """Review P1 round 3 (2026-08-09): an EXIOBASE sector whose dependency on a service is entirely
    ND (all contributing ENCORE processes are No-Data) scores 0 numerically but must be FLAGGED as
    unknown, not read as 'no dependency'. Wholesale trade × Water purification is such a cell."""
    from cge.nature.concord import sector_nd_mask, sector_scores
    from cge.nature.real import real_encore_concordance, real_encore_dependencies

    dep = real_encore_dependencies()
    cmap, _ = real_encore_concordance()
    whole = "Wholesale trade and commission trade, except of motor vehicles and motorcycles (51)"
    svc = "Water purification"
    assert sector_scores(dep, cmap, [whole]).loc[whole, svc] == 0.0  # numerically zero
    assert bool(sector_nd_mask(dep, cmap, [whole]).loc[whole, svc])  # but flagged unknown
    # A sector with real ratings on the service is NOT flagged.
    agri = "Cultivation of cereal grains nec"
    assert not bool(sector_nd_mask(dep, cmap, [agri]).loc[agri, svc])


@_needs_encore
def test_sector_nd_share_surfaces_partial_unknowns():
    """Review P1 round 4 (2026-08-10): partially-ND cells (e.g. 90% of a sector's concordance weight
    is No-Data) score ≈0 and were hidden by the all-or-nothing mask. sector_nd_share returns the
    WEIGHTED unknown fraction in [0, 1], so a partially-unknown cell is visible; a fully-unknown one
    is 1.0 and a fully-rated cell is 0.0."""
    from cge.nature.concord import sector_nd_share
    from cge.nature.real import real_encore_concordance, real_encore_dependencies

    dep = real_encore_dependencies()
    cmap, _ = real_encore_concordance()
    share = sector_nd_share(dep, cmap, list(cmap.weights)[:80])
    vals = share.to_numpy()
    assert (vals >= -1e-9).all() and (vals <= 1.0 + 1e-9).all()  # a fraction
    # There exist genuinely partial cells (0 < share < 1), not only 0/1.
    assert ((vals > 1e-6) & (vals < 1 - 1e-6)).any()
    # The fully-unknown wholesale/Water-purification cell is 1.0.
    whole = "Wholesale trade and commission trade, except of motor vehicles and motorcycles (51)"
    w = sector_nd_share(dep, cmap, [whole])
    assert w.loc[whole, "Water purification"] == pytest.approx(1.0)


@_needs_encore
def test_sector_nd_share_rejects_uncovered_sector():
    """Review P3 round 6 (2026-08-14): a sector absent from the concordance previously got an
    all-zero (0% unknown = fully KNOWN) row — the opposite of the truth. It must be rejected, the
    same way sector_scores does."""
    from cge.nature.concord import sector_nd_share
    from cge.nature.real import real_encore_concordance, real_encore_dependencies

    dep = real_encore_dependencies()
    cmap, _ = real_encore_concordance()
    with pytest.raises(ValueError, match="does not cover"):
        sector_nd_share(dep, cmap, ["NO_SUCH_SECTOR"])


def test_nature_to_productivity_rejects_typoed_coverage():
    """Review P2 round 6 (2026-08-14): the exported translator must reject a typo'd
    coverage_regions/coverage_sectors instead of silently baselining (matching no good → no shocks →
    a clean-looking zero-response run). Vocabulary is derived from the exposure index."""
    from cge.contracts.shocks import NatureStress
    from cge.nature.translate import nature_to_productivity

    exposure = pd.DataFrame(
        {"Water purification": [0.5, 0.5]},
        index=["RegA:farming", "RegB:farming"],
    )
    # A real region/sector works.
    ok = nature_to_productivity(
        [NatureStress(service="Water purification", severity=0.4, coverage_regions=["RegA"])],
        exposure,
    )
    assert any(s.delta < 0 for s in ok)
    # A typo'd region is rejected (was: silent empty result).
    with pytest.raises(ValueError, match="coverage_regions"):
        nature_to_productivity(
            [NatureStress(service="Water purification", severity=0.4, coverage_regions=["TYPO"])],
            exposure,
        )
    # A typo'd sector is rejected too.
    with pytest.raises(ValueError, match="coverage_sectors"):
        nature_to_productivity(
            [NatureStress(service="Water purification", severity=0.4, coverage_sectors=["TYPO"])],
            exposure,
        )


@_needs_encore
def test_manifest_records_weighted_nd_share():
    """A real nature run records the weighted ND share per stressed service in the manifest, so a
    partially-unknown sector/service is visible — not only the all-ND ones."""
    import numpy as np

    from cge.contracts.data_objects import Classification, IOSystem
    from cge.contracts.engine import registry
    from cge.contracts.shocks import NatureStress
    from cge.nature.real import real_encore_concordance, real_encore_dependencies
    from cge.runner import _preprocess_nature
    from cge.scenarios.loader import Scenario

    dep = real_encore_dependencies()
    cmap, _ = real_encore_concordance()
    secs = ["Cultivation of cereal grains nec", "Animal products nec"]
    labels = [f"R:{s}" for s in secs]
    io = IOSystem(
        provenance=_prov(),
        sectors=Classification(name="e", kind="sector", labels=secs),
        regions=Classification(name="r", kind="region", labels=["R"]),
        A=pd.DataFrame(np.full((2, 2), 0.05), index=labels, columns=labels),
        final_demand=pd.DataFrame({"final_demand": [100.0, 100.0]}, index=labels),
        unit="MEUR",
        currency="EUR",
    )
    sc = Scenario(
        name="n",
        engine="partial_eq",
        years=[2020],
        shocks=[NatureStress(service="Global climate regulation", severity=0.4)],
    )
    _shocks, stamp = _preprocess_nature(
        sc,
        {"IOSystem": io},
        registry.get("partial_eq"),
        {"IOSystem": io, "EncoreDependencies": dep, "ConcordanceMap": cmap},
    )
    assert "nd_weighted_share" in stamp
    share = stamp["nd_weighted_share"].get("Global climate regulation", {})
    # Animal products nec is partially unknown on this service (a real partial-ND cell).
    assert any(0.0 < v < 1.0 for v in share.values())


@_needs_encore
def test_nature_run_records_nd_unknown_sectors_in_manifest():
    """A real nature run whose stressed service is entirely unknown for some sectors records those
    sectors in the manifest's nd_unknown_sectors — so the zero shock reads as 'no data', not 'no
    dependency'."""
    import numpy as np

    from cge.contracts.data_objects import Classification, IOSystem
    from cge.contracts.engine import registry
    from cge.contracts.shocks import NatureStress
    from cge.nature.real import real_encore_concordance, real_encore_dependencies
    from cge.runner import _preprocess_nature
    from cge.scenarios.loader import Scenario

    dep = real_encore_dependencies()
    cmap, _ = real_encore_concordance()
    secs = [
        "Cultivation of cereal grains nec",
        "Wholesale trade and commission trade, except of motor vehicles and motorcycles (51)",
    ]
    labels = [f"R:{s}" for s in secs]
    A = pd.DataFrame(np.full((2, 2), 0.05), index=labels, columns=labels)
    io = IOSystem(
        provenance=_prov(),
        sectors=Classification(name="e", kind="sector", labels=secs),
        regions=Classification(name="r", kind="region", labels=["R"]),
        A=A,
        final_demand=pd.DataFrame({"final_demand": [100.0, 100.0]}, index=labels),
        unit="MEUR",
        currency="EUR",
    )
    sc = Scenario(
        name="n",
        engine="partial_eq",
        years=[2020],
        shocks=[NatureStress(service="Water purification", severity=0.4)],
    )
    _shocks, stamp = _preprocess_nature(
        sc,
        {"IOSystem": io},
        registry.get("partial_eq"),
        {"IOSystem": io, "EncoreDependencies": dep, "ConcordanceMap": cmap},
    )
    assert "Water purification" in stamp["nd_unknown_sectors"]
    assert any("Wholesale" in s for s in stamp["nd_unknown_sectors"]["Water purification"])


@_needs_encore
def test_real_nature_scenario_runs_end_to_end_through_runner_and_engine():
    """Review P1 round 3 (2026-08-09): a genuine end-to-end run on REAL data — real ENCORE ratings +
    real EXIOBASE↔ENCORE concordance + a real-EXIOBASE-labelled economy → the standard runner →
    Engine 2 → a schema-valid ResultSet with a real volume response. (Earlier the "end-to-end" test
    stopped at sector_scores; this drives the whole pipeline.)"""
    import numpy as np

    from cge.contracts.data_objects import (
        Classification,
        IOSystem,
        SatelliteAccount,
    )
    from cge.contracts.shocks import NatureStress
    from cge.nature.real import real_encore_concordance, real_encore_dependencies
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    dep = real_encore_dependencies()
    cmap, _ = real_encore_concordance()
    # Real EXIOBASE sector labels the concordance covers.
    secs = ["Cultivation of cereal grains nec", "Cultivation of crops nec"]
    labels = [f"R:{s}" for s in secs]
    A = pd.DataFrame(np.full((2, 2), 0.05), index=labels, columns=labels)
    io = IOSystem(
        provenance=_prov(),
        sectors=Classification(name="e", kind="sector", labels=secs),
        regions=Classification(name="r", kind="region", labels=["R"]),
        A=A,
        final_demand=pd.DataFrame({"final_demand": [100.0, 100.0]}, index=labels),
        unit="MEUR",
        currency="EUR",
    )
    sat = SatelliteAccount(
        provenance=_prov(),
        name="GHG",
        units={"CO2": "t/MEUR", "CO2e": "tCO2e/MEUR"},
        data=pd.DataFrame(
            {labels[0]: [1000.0, 1000.0], labels[1]: [1000.0, 1000.0]}, index=["CO2", "CO2e"]
        ),
    )
    sc = Scenario(
        name="real",
        engine="partial_eq",
        years=[2020],
        shocks=[NatureStress(service="Water purification", severity=0.4)],
    )
    res = run_scenario(
        sc,
        data_source="toy",
        data_overrides={
            "IOSystem": io,
            "SatelliteAccount": sat,
            "EncoreDependencies": dep,
            "ConcordanceMap": cmap,
        },
    )
    res.validate_schema()
    # A real volume response: cereal cultivation is highly water-dependent in the real ratings.
    vol = res.data[(res.data["variable"] == "volume_change") & (res.data["scenario"] == "central")]
    v = {r.sector: r.value for r in vol.itertuples()}
    assert v["Cultivation of cereal grains nec"] < 0.0  # real degradation cuts output
    # Full nature provenance in the manifest (reconstructible).
    nat = res.manifest.assumptions["nature"]
    assert nat["encore_source"] and nat["shock_coverage"]


@_needs_encore
def test_real_nature_runs_from_a_persisted_store_build(tmp_path):
    """Review P1 round 4 (2026-08-11): prove a nature scenario runs from a genuinely PERSISTED build
    — real ENCORE + real concordance saved through DataStore, then a NatureStress run via
    run_scenario(data_source=build_id) (the store→runner→engine path), not an override injection.
    The economy uses real EXIOBASE labels the real concordance covers. (This is a real-labelled
    build; a normal AGGREGATED EXIOBASE build still needs an aggregation-aware concordance — a
    documented follow-up.)"""
    import numpy as np

    from cge.contracts.data_objects import (
        Classification,
        IOSystem,
        SatelliteAccount,
    )
    from cge.contracts.shocks import NatureStress
    from cge.data.metadata import BuildMeta
    from cge.data.store import DataStore
    from cge.nature.real import real_encore_concordance, real_encore_dependencies
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    dep = real_encore_dependencies()
    cmap, _ = real_encore_concordance()
    secs = ["Cultivation of cereal grains nec", "Cultivation of crops nec"]
    labels = [f"R:{s}" for s in secs]
    io = IOSystem(
        provenance=_prov(),
        sectors=Classification(name="e", kind="sector", labels=secs),
        regions=Classification(name="r", kind="region", labels=["R"]),
        A=pd.DataFrame(np.full((2, 2), 0.05), index=labels, columns=labels),
        final_demand=pd.DataFrame({"final_demand": [100.0, 100.0]}, index=labels),
        unit="MEUR",
        currency="EUR",
    )
    sat = SatelliteAccount(
        provenance=_prov(),
        name="GHG",
        units={"CO2": "t/MEUR", "CO2e": "tCO2e/MEUR"},
        data=pd.DataFrame(
            {labels[0]: [1000.0, 1000.0], labels[1]: [1000.0, 1000.0]}, index=["CO2", "CO2e"]
        ),
    )
    store = DataStore(tmp_path)
    meta = BuildMeta(
        build_id="real_nat",
        source="real-labelled",
        source_version="v",
        reference_year=2026,
        licence="x",
        currency="EUR",
        monetary_unit="MEUR",
        retrieved="2026-08-11",
    )
    store.save(meta=meta, io=io, satellites=[sat], encore=dep, concordance=cmap)

    # Nature data must have round-tripped, so the scenario runs from the build id alone.
    loaded = store.load("real_nat")
    assert "EncoreDependencies" in loaded and "ConcordanceMap" in loaded

    sc = Scenario(
        name="real",
        engine="partial_eq",
        years=[2020],
        shocks=[NatureStress(service="Water purification", severity=0.4)],
    )
    res = run_scenario(sc, data_source="real_nat", store=store)
    res.validate_schema()
    vol = res.data[(res.data["variable"] == "volume_change") & (res.data["scenario"] == "central")]
    v = {r.sector: r.value for r in vol.itertuples()}
    assert v["Cultivation of cereal grains nec"] < 0.0  # real degradation from the persisted build
    assert res.manifest.assumptions["nature"]["encore_source"]


@_needs_encore
def test_aggregation_aware_concordance_runs_on_grouped_sectors(tmp_path):
    """Review P1 round 5 (2026-08-13): an AGGREGATED build (coarse sector groups, not real EXIOBASE
    labels) can run nature via an aggregation-aware concordance — from the real concordance with
    the sector-aggregation map. Persist it through DataStore and run from the build id."""
    import numpy as np

    from cge.contracts.data_objects import (
        Classification,
        IOSystem,
        SatelliteAccount,
    )
    from cge.contracts.shocks import NatureStress
    from cge.data.metadata import BuildMeta
    from cge.data.store import DataStore
    from cge.nature.concordance_build import aggregate_concordance
    from cge.nature.real import real_encore_concordance, real_encore_dependencies
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    dep = real_encore_dependencies()
    full_cmap, _ = real_encore_concordance()
    # Map several real EXIOBASE sectors into two coarse groups (the build's sectors).
    fine = list(full_cmap.weights)[:6]
    sector_map = {fine[i]: ("primary" if i < 3 else "services") for i in range(6)}
    agg = aggregate_concordance(full_cmap, sector_map)
    groups = ["primary", "services"]
    assert set(agg.weights) == set(groups)  # both groups covered

    labels = [f"R:{g}" for g in groups]
    io = IOSystem(
        provenance=_prov(),
        sectors=Classification(name="e", kind="sector", labels=groups),
        regions=Classification(name="r", kind="region", labels=["R"]),
        A=pd.DataFrame(np.full((2, 2), 0.05), index=labels, columns=labels),
        final_demand=pd.DataFrame({"final_demand": [100.0, 100.0]}, index=labels),
        unit="MEUR",
        currency="EUR",
    )
    sat = SatelliteAccount(
        provenance=_prov(),
        name="GHG",
        units={"CO2": "t/MEUR", "CO2e": "tCO2e/MEUR"},
        data=pd.DataFrame(
            {labels[0]: [1000.0, 1000.0], labels[1]: [1000.0, 1000.0]}, index=["CO2", "CO2e"]
        ),
    )
    store = DataStore(tmp_path)
    meta = BuildMeta(
        build_id="agg_nat",
        source="aggregated",
        source_version="v",
        reference_year=2026,
        licence="x",
        currency="EUR",
        monetary_unit="MEUR",
        retrieved="2026-08-13",
    )
    store.save(meta=meta, io=io, satellites=[sat], encore=dep, concordance=agg)

    sc = Scenario(
        name="agg",
        engine="partial_eq",
        years=[2020],
        shocks=[NatureStress(service="Water purification", severity=0.4)],
    )
    res = run_scenario(sc, data_source="agg_nat", store=store)
    res.validate_schema()
    assert (
        res.data["variable"] == "volume_change"
    ).any()  # aggregated build runs nature end-to-end


def _pymrio_system_for_labels(labels, regions=("RegA", "RegB")):
    """A minimal pymrio IOSystem over the given REAL EXIOBASE sector ``labels`` — a small dense A +
    household FD + a trivial GHG extension, so the adapter and quality gates accept it without a
    multi-GB download. Used to exercise the real orchestration over the actual classifications."""
    import numpy as np
    import pymrio

    idx = pd.MultiIndex.from_product([list(regions), list(labels)], names=["region", "sector"])
    n = len(idx)
    A = pd.DataFrame(np.full((n, n), 0.001), index=idx, columns=idx)
    y_cols = pd.MultiIndex.from_product(
        [list(regions), ["Household final consumption"]], names=["region", "category"]
    )
    Y = pd.DataFrame(np.full((n, len(y_cols)), 1.0), index=idx, columns=y_cols)
    pio = pymrio.IOSystem(A=A, Y=Y)
    pio.x = pymrio.calc_x_from_L(pymrio.calc_L(A), Y.sum(axis=1))
    # A trivial GHG extension so the adapter emits a SatelliteAccount (partial_eq requires one). The
    # stressor is aliased to CO2 at build time; values are uniform — we only need the account there.
    F = pd.DataFrame(np.full((1, n), 1.0), index=["emission_type1"], columns=idx)
    ext = pymrio.Extension(name="emissions", F=F)
    ext.unit = pd.DataFrame({"unit": ["t"]}, index=["emission_type1"])
    pio.emissions = ext
    return pio


def _pxp_pymrio_system(regions=("RegA", "RegB")):
    """A minimal pymrio IOSystem whose sector labels are the REAL 200 EXIOBASE **product** (pxp)
    labels from pymrio's ``exio3_pxp`` classification — the actual default-build classification, not
    a hand-picked subset (review P1 round 6 2026-08-14)."""
    import pymrio

    products = [str(x).strip() for x in pymrio.get_classification("exio3_pxp").sectors["ExioName"]]
    return _pymrio_system_for_labels(products, regions=regions), products


@_needs_encore
def test_build_from_pymrio_attaches_nature_for_real_pxp_products(tmp_path):
    """The DEFAULT (pxp) live-build classification runs nature end to end (review P1 round 6).

    Drives the REAL orchestration — build_from_pymrio over the actual 200 EXIOBASE product labels,
    not a hand-picked subset — under the STRICT ``required`` policy, so the product→industry→ENCORE
    bridge must cover every one of the 200 products or the build fails. Then a NatureStress scenario
    runs from the persisted build id (store→runner→engine), proving the previously-broken live pxp
    path is genuinely closed."""
    from cge.contracts.shocks import NatureStress
    from cge.data.build import build_from_pymrio
    from cge.data.store import DataStore
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    pio, products = _pxp_pymrio_system()
    store = DataStore(tmp_path)
    written = build_from_pymrio(
        pio,
        source="EXIOBASE",
        source_version="3-pxp-test",
        reference_year=2019,
        build_id="exio-pxp",
        store=store,
        make_small=False,
        gas_aliases={"emission_type1": "CO2"},
        attach_nature="required",  # every one of the 200 products must be covered, or raise
    )
    loaded = store.load(written["full"])
    assert "EncoreDependencies" in loaded and "ConcordanceMap" in loaded
    assert set(loaded["ConcordanceMap"].weights) == set(products)  # complete coverage, all 200

    sc = Scenario(
        name="pxp-nature",
        engine="partial_eq",
        years=[2020],
        shocks=[NatureStress(service="Water purification", severity=0.4)],
    )
    res = run_scenario(sc, data_source="exio-pxp", store=store)
    res.validate_schema()
    vol = res.data[(res.data["variable"] == "volume_change") & (res.data["scenario"] == "central")]
    assert (vol["value"] < 0.0).any()  # real degradation propagates through the persisted pxp build
    assert res.manifest.assumptions["nature"]["encore_source"]


@_needs_encore
def test_build_from_pymrio_attaches_nature_for_real_ixi_industries(tmp_path):
    """The supported ``system="ixi"`` classification also runs nature end to end (P2 round 7).

    Drives build_from_pymrio over the actual 163 EXIOBASE INDUSTRY labels under ``required``. The
    crosswalk covers 162; the residual ``Production of electricity nec`` is filled by the shared
    NACE-sibling fallback (``complete_industry_concordance``), so the direct ixi path attaches over
    the WHOLE classification — previously it failed 162/163. Then a NatureStress runs from the
    persisted build."""
    import pymrio

    from cge.contracts.shocks import NatureStress
    from cge.data.build import build_from_pymrio
    from cge.data.store import DataStore
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    ixi = pymrio.get_classification("exio3_ixi").sectors
    industries = [str(x).strip() for x in ixi["ExioName"]]
    assert len(industries) == 163
    pio = _pymrio_system_for_labels(industries)
    store = DataStore(tmp_path)
    written = build_from_pymrio(
        pio,
        source="EXIOBASE",
        source_version="3-ixi-test",
        reference_year=2019,
        build_id="exio-ixi",
        store=store,
        make_small=False,
        gas_aliases={"emission_type1": "CO2"},
        attach_nature="required",  # every one of the 163 industries must be covered, or raise
    )
    loaded = store.load(written["full"])
    assert set(loaded["ConcordanceMap"].weights) == set(industries)  # all 163, incl. the residual
    assert "Production of electricity nec" in loaded["ConcordanceMap"].weights

    sc = Scenario(
        name="ixi-nature",
        engine="partial_eq",
        years=[2020],
        shocks=[NatureStress(service="Water purification", severity=0.4)],
    )
    res = run_scenario(sc, data_source="exio-ixi", store=store)
    res.validate_schema()
    vol = res.data[(res.data["variable"] == "volume_change") & (res.data["scenario"] == "central")]
    assert (vol["value"] < 0.0).any()


@_needs_encore
def test_build_partial_coverage_is_a_hard_error_not_a_silent_subset(tmp_path):
    """A build whose sectors are only PARTIALLY covered must fail, not persist a covered subset
    (review P1 round 6 — the old logic persisted a 13/200 concordance then failed at run time)."""
    from cge.data.build import NatureAttachError, _nature_for_sectors
    from cge.nature.real import real_encore_concordance

    covered = list(real_encore_concordance()[0].weights)[:2]
    labels = [*covered, "NOT_AN_EXIOBASE_SECTOR"]
    # 'auto' still rejects a partial match (it is a real defect, not optional-data absence).
    for policy in ("auto", "required"):
        with pytest.raises(NatureAttachError, match="covers only"):
            _nature_for_sectors(labels, policy=policy)


def test_build_test_skips_nature_cleanly_when_labels_not_exiobase(tmp_path):
    """The offline test MRIO's labels are NOT EXIOBASE, so the default 'auto' policy skips nature
    cleanly and the build still succeeds — the 'optional, skipped when inapplicable' contract."""
    from cge.data.build import build_test
    from cge.data.store import DataStore

    store = DataStore(tmp_path)
    written = build_test(store=store)
    assert "EncoreDependencies" not in store.load(written["small"])
