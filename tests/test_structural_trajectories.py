"""Tests for the Phase 7b.2 structural-trajectory contract + vendored loader.

Covers the StructuralTrajectory validation (finite rates in a plausible band, known drivers,
per-entry provenance), the piecewise-constant rate lookup with __all__ fallback, and that the
vendored real-sourced artifact loads and carries a citation + confidence for every path.
"""

from __future__ import annotations

import pytest

from cge.contracts.data_objects import Provenance, StructuralTrajectory
from cge.data.structural import load_structural_trajectories


def _prov() -> Provenance:
    return Provenance(
        source="test",
        source_version="v1",
        licence="n/a",
        reference_year=2024,
        retrieved="2026-08-16",
    )


def _traj(rates, sources=None, confidence=None) -> StructuralTrajectory:
    # Auto-fill source/confidence for every (driver, region) key unless the test overrides them.
    keys = [f"{d}:{r}" for d, by in rates.items() for r in by]
    return StructuralTrajectory(
        provenance=_prov(),
        rates=rates,
        sources=sources if sources is not None else {k: "cite" for k in keys},
        confidence=confidence if confidence is not None else {k: "medium" for k in keys},
    )


def test_rate_is_piecewise_constant_between_knots():
    t = _traj({"productivity": {"N": {2025: 0.02, 2040: 0.01}}})
    assert t.rate("productivity", "N", 2025) == pytest.approx(0.02)
    assert t.rate("productivity", "N", 2030) == pytest.approx(0.02)  # holds the 2025 knot
    assert t.rate("productivity", "N", 2040) == pytest.approx(0.01)
    assert t.rate("productivity", "N", 2050) == pytest.approx(0.01)  # holds the last knot
    assert t.rate("productivity", "N", 2000) == pytest.approx(0.02)  # predates → earliest knot


def test_all_region_fallback_and_explicit_override():
    t = _traj({"population": {"__all__": {2025: 0.005}, "S": {2025: 0.012}}})
    assert t.rate("population", "S", 2025) == pytest.approx(0.012)  # explicit wins
    assert t.rate("population", "ZZ", 2025) == pytest.approx(0.005)  # unknown → __all__
    assert t.rate("labour_participation", "N", 2025) == 0.0  # absent driver → 0


def test_unknown_driver_rejected():
    with pytest.raises(ValueError, match="unknown region structural driver"):
        _traj({"gdp_share": {"N": {2025: 0.01}}})


def test_implausible_rate_rejected():
    with pytest.raises(ValueError, match="plausible band"):
        _traj({"productivity": {"N": {2025: 1.5}}})  # +150%/yr
    with pytest.raises(ValueError, match="plausible band"):
        _traj({"population": {"N": {2025: -1.0}}})  # −100%/yr drives endowment to 0


def test_missing_source_or_confidence_rejected():
    with pytest.raises(ValueError, match="missing source or confidence"):
        StructuralTrajectory(
            provenance=_prov(),
            rates={"productivity": {"N": {2025: 0.02}}},
            sources={},  # no source for productivity:N
            confidence={"productivity:N": "medium"},
        )


def test_empty_path_rejected():
    with pytest.raises(ValueError, match="no dated rates"):
        _traj({"productivity": {"N": {}}})


def test_sector_rate_lookup_and_all_fallback():
    t = _traj(
        {"productivity": {"N": {2025: 0.01}}},
    )
    # Rebuild with sector_rates too (the _traj helper only fills region-axis keys).
    t = StructuralTrajectory(
        provenance=_prov(),
        rates={"productivity": {"N": {2025: 0.01}}},
        sector_rates={
            "sector_productivity": {"BRD": {2025: 0.03}, "__all__": {2025: 0.005}},
            "emissions_intensity": {"BRD": {2025: -0.05}},
        },
        sources={
            "productivity:N": "c",
            "sector_productivity:BRD": "c",
            "sector_productivity:__all__": "c",
            "emissions_intensity:BRD": "c",
        },
        confidence={
            "productivity:N": "high",
            "sector_productivity:BRD": "medium",
            "sector_productivity:__all__": "low",
            "emissions_intensity:BRD": "medium",
        },
    )
    assert t.sector_rate("sector_productivity", "BRD", 2030) == pytest.approx(0.03)
    assert t.sector_rate("sector_productivity", "MIL", 2030) == pytest.approx(0.005)  # __all__
    assert t.sector_rate("emissions_intensity", "BRD", 2025) == pytest.approx(-0.05)
    assert t.sector_rate("emissions_intensity", "MIL", 2025) == 0.0  # absent → 0


def test_sector_driver_on_region_axis_rejected():
    with pytest.raises(ValueError, match="unknown region structural driver"):
        StructuralTrajectory(
            provenance=_prov(),
            rates={"emissions_intensity": {"N": {2025: -0.05}}},
            sources={"emissions_intensity:N": "c"},
            confidence={"emissions_intensity:N": "low"},
        )


def test_region_driver_on_sector_axis_rejected():
    with pytest.raises(ValueError, match="unknown sector structural driver"):
        StructuralTrajectory(
            provenance=_prov(),
            sector_rates={"population": {"BRD": {2025: 0.01}}},
            sources={"population:BRD": "c"},
            confidence={"population:BRD": "high"},
        )


def test_vendored_artifact_loads_with_full_provenance():
    """The real vendored artifact loads through the contract and carries a citation + confidence for
    every (driver, region) path — nothing enters a run unsourced."""
    t = load_structural_trajectories()
    assert set(t.rates) == {"population", "labour_participation", "productivity"}
    for driver, by_region in t.rates.items():
        for region in by_region:
            key = f"{driver}:{region}"
            assert t.sources.get(key), f"{key} has no source"
            assert t.confidence.get(key), f"{key} has no confidence"
    # Sanity on the sourced figures: emerging-proxy S has faster population + productivity than N.
    assert t.rate("population", "S", 2025) > t.rate("population", "N", 2025)
    assert t.rate("productivity", "S", 2025) > t.rate("productivity", "N", 2025)


def test_concordance_maps_real_build_labels_to_differentiated_trajectories():
    """Review P1c 2026-08-28: the concordance binds a REAL EXIOBASE build's coarse-v3 labels to the
    trajectory's archetypes, so a real build carries DIFFERENTIATED country/sector trajectories —
    advanced regions (US/DE) grow TFP slower than emerging (CN/IN), goods sectors (manufacturing)
    have a different productivity drift than services — instead of every label collapsing to
    __all__."""
    from cge.data.structural import structural_trajectories_for_build

    regions = ["US", "DE", "CN", "IN", "RoW_Africa"]
    sectors = ["manufacturing", "metals", "services", "transport", "electricity"]
    t = structural_trajectories_for_build(regions, sectors)
    # Every real label is now an explicit key (not __all__), and provenance carries through.
    for r in regions:
        assert r in t.rates["productivity"]
        assert t.sources.get(f"productivity:{r}")
    # Advanced vs emerging differentiation exists at the REAL labels.
    assert t.rate("productivity", "US", 2025) < t.rate("productivity", "CN", 2025)
    # Goods vs services sectoral differentiation exists at the REAL labels.
    assert t.sector_rate("sector_productivity", "manufacturing", 2025) != t.sector_rate(
        "sector_productivity", "services", 2025
    )


def test_concordance_rejects_unmapped_build_labels():
    """An unmapped build label must fail loudly (review P1c): otherwise every real label would
    silently take the global __all__ trajectory and the run would falsely claim country/sector
    differentiation. The gate is the check the reviewer asked for."""
    from cge.data.structural import structural_trajectories_for_build
    from cge.data.structural.library import UnmappedStructuralLabels

    with pytest.raises(UnmappedStructuralLabels):
        structural_trajectories_for_build(["US", "ZZ_not_a_region"], ["manufacturing"])
    with pytest.raises(UnmappedStructuralLabels):
        structural_trajectories_for_build(["US"], ["not_a_sector"])
    # Opting out of the gate is allowed and explicit (the label then takes __all__).
    t = structural_trajectories_for_build(["US"], ["not_a_sector"], require_full_coverage=False)
    assert t.sector_rate("sector_productivity", "not_a_sector", 2025) == t.sector_rate(
        "sector_productivity", "__all__", 2025
    )


def test_concordance_coverage_diagnostic_flags_all_fallthrough():
    """The recursive manifest's structural coverage diagnostic must distinguish a differentiated run
    (real labels explicitly keyed) from an all-__all__ fallthrough (review P1c)."""
    from cge.data.structural import structural_trajectories_for_build
    from cge.dynamics.recursive import _structural_coverage

    # A concordance-mapped trajectory on real labels → differentiated (not all-fallthrough).
    mapped = structural_trajectories_for_build(["US", "CN"], ["manufacturing", "services"])
    cov = _structural_coverage(mapped, ["US", "CN"], ["manufacturing", "services"])
    assert cov["all_fallthrough"] is False
    assert cov["explicit_region_matches"] > 0 and cov["explicit_sector_matches"] > 0

    # The bare archetype trajectory on real labels → every label falls through to __all__.
    archetype = load_structural_trajectories()
    cov2 = _structural_coverage(archetype, ["US", "CN"], ["manufacturing", "services"])
    assert cov2["all_fallthrough"] is True


def test_concordance_v2_blends_mixed_blocks_with_gdp_weights():
    """Review P1 2026-08-29: a MIXED coarse block (RoW_Asia contains advanced AU/KR/TW and emerging
    ID/WA) must get a GDP-WEIGHTED BLEND of its members' archetype paths, NOT one archetype. So its
    productivity rate lies strictly between the N and S archetype rates, while a pure-N block (US)
    equals N and a pure-S block (RoW_Africa) equals S."""
    from cge.data.structural import load_structural_trajectories, structural_trajectories_for_build

    arch = load_structural_trajectories()
    n_rate = arch.rate("productivity", "N", 2025)
    s_rate = arch.rate("productivity", "S", 2025)
    t = structural_trajectories_for_build(
        ["US", "RoW_Asia", "RoW_Europe", "RoW_Africa"], ["manufacturing"]
    )
    assert t.rate("productivity", "US", 2025) == pytest.approx(n_rate)
    assert t.rate("productivity", "RoW_Africa", 2025) == pytest.approx(s_rate)
    # Mixed blocks are genuine blends strictly between the two archetypes.
    assert n_rate < t.rate("productivity", "RoW_Asia", 2025) < s_rate
    # RoW_Europe is mostly advanced (+ some emerging TR/Eastern EU) → closer to N than RoW_Asia is.
    assert t.rate("productivity", "RoW_Europe", 2025) < t.rate("productivity", "RoW_Asia", 2025)


def test_concordance_v2_carries_composite_provenance():
    """Review P2 2026-08-29: the mapped trajectory must record BOTH the concordance and archetype
    identities, not just the bare structural-trajectories-v1 provenance."""
    from cge.data.structural import structural_trajectories_for_build

    t = structural_trajectories_for_build(["US", "CN"], ["manufacturing", "services"])
    assert "concordance" in t.provenance.source.lower()
    assert "structural-concordance-v2" in t.provenance.source_version
    assert "structural-trajectories-v1" in t.provenance.source_version


def test_concordance_validation_rejects_bad_archetype_and_weights(tmp_path):
    """Review P2 2026-08-29: load_structural_concordance validates through the Provenance contract,
    rejects a country archetype that isn't a known trajectory path (the reviewer's US→TYPO), and
    rejects block weights that don't sum to 1."""
    import json
    from pathlib import Path

    from cge.data.structural.library import load_structural_concordance

    raw = json.loads(Path("data/structural/concordance_v2.json").read_text())

    bad_arch = json.loads(json.dumps(raw))
    bad_arch["country_archetype"]["US"] = "TYPO"
    p1 = tmp_path / "bad_arch.json"
    p1.write_text(json.dumps(bad_arch))
    with pytest.raises(ValueError, match="not among the trajectory"):
        load_structural_concordance(p1)

    bad_w = json.loads(json.dumps(raw))
    bad_w["block_membership"]["RoW_Asia"] = {"AU": 0.5, "KR": 0.4}  # sums to 0.9
    p2 = tmp_path / "bad_w.json"
    p2.write_text(json.dumps(bad_w))
    with pytest.raises(ValueError, match="sum to"):
        load_structural_concordance(p2)


def test_run_recursive_rejects_all_fallthrough_structural_run():
    """Review P2 2026-08-29: 'unmapped label fails loudly' must hold at the RUN boundary — an
    all-__all__ structural trajectory on a real build is rejected unless the caller opts into the
    uniform fallback."""
    from cge.contracts.data_objects import Provenance, StructuralTrajectory
    from cge.contracts.shocks import CarbonPrice
    from cge.dynamics import DynamicConfig, run_recursive
    from cge.scenarios.loader import Scenario

    allonly = StructuralTrajectory(
        provenance=Provenance(
            source="t", source_version="v", licence="n", reference_year=2024, retrieved="2026-08-16"
        ),
        rates={"productivity": {"__all__": {2025: 0.02}}},
        sector_rates={"sector_productivity": {"__all__": {2025: 0.01}}},
        sources={"productivity:__all__": "c", "sector_productivity:__all__": "c"},
        confidence={"productivity:__all__": "low", "sector_productivity:__all__": "low"},
    )
    sc = Scenario(
        name="x", engine="cge_static", years=[2025, 2030], shocks=[CarbonPrice(price=50.0)]
    )
    with pytest.raises(ValueError, match="does not differentiate"):
        run_recursive(sc, config=DynamicConfig(structural=allonly), data_source="toy_cge_gov")
    # Opt-in runs it deliberately.
    path = run_recursive(
        sc,
        config=DynamicConfig(structural=allonly, allow_uniform_fallback=True),
        data_source="toy_cge_gov",
    )
    assert path.result is not None
