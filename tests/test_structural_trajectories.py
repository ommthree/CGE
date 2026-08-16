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
