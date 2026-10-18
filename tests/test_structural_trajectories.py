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


def test_emissions_intensity_routes_each_sector_to_its_ngfs_channel():
    """review-11 P1: emissions_intensity is mapped by sector_emissions_channel, so electricity gets
    the ELECTRICITY-GENERATION path (not industry combustion), transport the transport path, metals
    the process path — while sector_productivity still uses the goods/services archetypes. The
    channels give materially different rates, and electricity ≠ manufacturing (the old bug mapped
    both to the industry bundle)."""
    from cge.data.structural import (
        load_structural_trajectories,
        structural_trajectories_for_build,
    )

    arch = load_structural_trajectories()
    # The archetype trajectory carries per-channel emissions keys, not BRD/MIL.
    assert {"electricity", "industry", "process", "transport", "buildings"} <= set(
        arch.sector_rates["emissions_intensity"]
    )
    t = structural_trajectories_for_build(
        ["US"], ["electricity", "manufacturing", "metals", "transport", "services"]
    )
    # electricity uses the generation channel; manufacturing the industry channel → they DIFFER
    # (previously both inherited the single industry bundle).
    e_elec = t.sector_rate("emissions_intensity", "electricity", 2040)
    e_manu = t.sector_rate("emissions_intensity", "manufacturing", 2040)
    e_metals = t.sector_rate("emissions_intensity", "metals", 2040)
    e_transport = t.sector_rate("emissions_intensity", "transport", 2040)
    assert e_elec != e_manu  # electricity no longer inherits the industry path
    assert e_metals != e_manu  # metals routes to the process channel, not industry
    assert e_transport != e_manu  # transport its own channel
    # electricity (grid greening) decarbonises fastest.
    assert e_elec < e_transport


def test_emissions_intensity_blends_cumulative_levels_not_annual_rates(tmp_path):
    """review-13 P1#2: a sector's emissions path is a WEIGHTED BLEND of channels (mixed sources),
    the weights are BASELINE SOURCE SHARES, so the sector's intensity LEVEL is Σ w_c F_c(t) with
    F_c(t)=∏_{τ<t}(1+r_{c,τ}) — NOT a per-year average of the channels' annual rates (which silently
    rebalances the source shares every year). Multi-knot channels where the two formulas DIVERGE:
    process = {2025:+0.10, 2035:0}, industry = {2025:−0.30, 2035:0}, metals={process:0.6,
    industry:0.4}. The emitted metals rate, compounded 2025→2035, must reproduce the LEVEL blend
    0.6·1.10^10 + 0.4·0.70^10, NOT ∏(1+0.6·0.10+0.4·−0.30)=1.0^10=1."""
    import json

    from cge.data.structural import structural_trajectories_for_build

    traj = tmp_path / "traj.json"
    traj.write_text(
        json.dumps(
            {
                "provenance": {
                    "source": "t",
                    "source_version": "t",
                    "licence": "t",
                    "reference_year": 2024,
                    "retrieved": "2026-10-14",
                },
                "rates": {"productivity": {"__all__": {"2025": 0.0}}},
                "sector_rates": {
                    "emissions_intensity": {
                        "process": {"2025": 0.10, "2035": 0.0},
                        "industry": {"2025": -0.30, "2035": 0.0},
                        "__all__": {"2025": -0.03, "2035": 0.0},
                    }
                },
                "sources": {
                    "productivity:__all__": "x",
                    "emissions_intensity:process": "x",
                    "emissions_intensity:industry": "x",
                    "emissions_intensity:__all__": "x",
                },
                "confidence": {
                    "productivity:__all__": "low",
                    "emissions_intensity:process": "low",
                    "emissions_intensity:industry": "low",
                    "emissions_intensity:__all__": "low",
                },
            }
        )
    )
    conc = {
        "provenance": {
            "source": "c",
            "source_version": "concordance-x",
            "licence": "x",
            "reference_year": 2024,
            "retrieved": "2026-10-14",
        },
        "region_archetype": {"US": "__all__"},
        "sector_archetype": {"metals": "__all__"},
        "sector_emissions_channel": {"metals": {"process": 0.6, "industry": 0.4}},
        "sources": {"x": "x"},
    }
    p = tmp_path / "conc.json"
    p.write_text(json.dumps(conc))
    t = structural_trajectories_for_build(
        ["US"],
        ["metals"],
        trajectory_path=str(traj),
        concordance_path=str(p),
        require_full_coverage=False,
    )
    # Compound the emitted metals rate over the interval and compare to the LEVEL blend.
    factor = 1.0
    for y in range(2025, 2035):
        factor *= 1.0 + t.sector_rate("emissions_intensity", "metals", y)
    level_blend = 0.6 * (1.10**10) + 0.4 * (0.70**10)
    assert factor == pytest.approx(level_blend, rel=1e-9)
    # And it must NOT equal the buggy annual-rate average (which here would be exactly 1.0).
    annual_rate_avg = (1.0 + (0.6 * 0.10 + 0.4 * -0.30)) ** 10
    assert abs(factor - annual_rate_avg) > 0.05

    # review-14 P2#3: exact at EVERY year, not just the knots (a weighted sum of differently-growing
    # channel levels is NOT geometric between knots). Check an intermediate year 2030.
    def F_s(y):  # level blend from the shared t0=2025
        return 0.6 * (1.10 ** (y - 2025)) + 0.4 * (0.70 ** (y - 2025))

    mid = 1.0
    for y in range(2025, 2030):
        mid *= 1.0 + t.sector_rate("emissions_intensity", "metals", y)
    assert mid == pytest.approx(F_s(2030), rel=1e-9)  # exact mid-interval, not just at 2035

    # review-14 P1#1: both channels hold their final rate flat (0.0) from 2035, so the mapped sector
    # must ALSO be flat after 2035 — the level stops changing. The old code repeated the previous
    # interval's (non-zero) rate past the last knot, so a truncated-flat channel kept decaying.
    assert t.sector_rate("emissions_intensity", "metals", 2035) == pytest.approx(0.0, abs=1e-12)
    assert t.sector_rate("emissions_intensity", "metals", 2060) == pytest.approx(0.0, abs=1e-12)
    tail = 1.0
    for y in range(2035, 2100):
        tail *= 1.0 + t.sector_rate("emissions_intensity", "metals", y)
    assert tail == pytest.approx(1.0, rel=1e-9)  # dead flat past the last channel knot

    # provenance names both channels with weights.
    src = t.sources["emissions_intensity:metals"]
    assert "channel-blended" in src and "process (0.60)" in src and "industry (0.40)" in src


def test_emissions_blend_holds_final_knot_flat_when_a_channel_is_truncated(tmp_path):
    """review-14 P1#1: the shipped electricity channel is intentionally truncated to 0.0 after its
    net-negative tail. A sector on {electricity:1.0} must inherit that FLAT tail exactly — the last
    mapped knot must be the channel's held final rate (0.0), NOT a repeat of the prior decline. With
    a single channel the mapped path equals the source path, so electricity at 2080+ must be 0.0."""
    from cge.data.structural import structural_trajectories_for_build

    t = structural_trajectories_for_build(["US"], ["electricity"], require_full_coverage=False)
    # The real concordance maps electricity → {electricity: 1.0}; the source channel is 0.0 from
    # 2080 (truncated). The mapped rate must be exactly flat there, not the ~−0.30 prior interval.
    assert t.sector_rate("emissions_intensity", "electricity", 2080) == pytest.approx(0.0, abs=1e-9)
    assert t.sector_rate("emissions_intensity", "electricity", 2100) == pytest.approx(0.0, abs=1e-9)


def test_emissions_channel_coverage_enforced_even_for_all_fallback_only_map(tmp_path):
    """review-14 P2#4: the complete-coverage check must fire INDEPENDENTLY of whether any NAMED
    channel exists. If every present sector maps to {"__all__": 1.0}, an OMITTED sector previously
    slipped through (the check was gated on a non-empty named-channel set) and silently took the
    global __all__ rate. Now a missing sector raises regardless."""
    import json

    from cge.data.structural import load_structural_concordance, load_structural_trajectories

    traj = tmp_path / "traj.json"
    traj.write_text(
        json.dumps(
            {
                "provenance": {
                    "source": "t",
                    "source_version": "t",
                    "licence": "t",
                    "reference_year": 2024,
                    "retrieved": "2026-10-18",
                },
                "rates": {"productivity": {"__all__": {"2025": 0.0}}},
                "sector_rates": {"emissions_intensity": {"__all__": {"2025": -0.03}}},
                "sources": {
                    "productivity:__all__": "x",
                    "emissions_intensity:__all__": "x",
                },
                "confidence": {
                    "productivity:__all__": "low",
                    "emissions_intensity:__all__": "low",
                },
            }
        )
    )
    trajectory = load_structural_trajectories(str(traj))
    conc = {
        "provenance": {
            "source": "c",
            "source_version": "concordance-x",
            "licence": "x",
            "reference_year": 2024,
            "retrieved": "2026-10-18",
        },
        "region_archetype": {"US": "__all__"},
        # TWO sectors present; both would be __all__-only, but 'coal' is OMITTED from the map.
        "sector_archetype": {"metals": "__all__", "coal": "__all__"},
        "sector_emissions_channel": {"metals": {"__all__": 1.0}},  # coal missing
        "sources": {"x": "x"},
    }
    p = tmp_path / "conc.json"
    p.write_text(json.dumps(conc))
    with pytest.raises(ValueError, match="missing sector"):
        load_structural_concordance(str(p), trajectory=trajectory)


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


def test_concordance_v3_per_driver_and_time_varying_weights(tmp_path):
    """Review-9 1c: v3 blends each driver with its OWN weight class, and population weights are
    year-indexed. Synthetic block BLK = {N-country CA, S-country IN}. output weights (productivity)
    favour CA 0.9/0.1; population weights favour IN and DRIFT: 2025 CA 0.5 / IN 0.5 → 2050 CA 0.2 /
    IN 0.8. So productivity(BLK) sits near N; population(BLK) moves toward S as years advance."""
    import json

    from cge.data.structural import structural_trajectories_for_build

    traj = tmp_path / "traj.json"
    traj.write_text(
        json.dumps(
            {
                "provenance": {
                    "source": "t",
                    "source_version": "t",
                    "licence": "t",
                    "reference_year": 2024,
                    "retrieved": "2026-09-16",
                },
                "rates": {
                    "productivity": {"N": {"2025": 0.02}, "S": {"2025": 0.06}},
                    "population": {
                        "N": {"2025": 0.005, "2050": 0.005},
                        "S": {"2025": 0.02, "2050": 0.02},
                    },
                },
                "sources": {
                    "productivity:N": "x",
                    "productivity:S": "x",
                    "population:N": "x",
                    "population:S": "x",
                },
                "confidence": {
                    "productivity:N": "low",
                    "productivity:S": "low",
                    "population:N": "low",
                    "population:S": "low",
                },
            }
        )
    )
    conc = tmp_path / "conc.json"
    conc.write_text(
        json.dumps(
            {
                "provenance": {
                    "source": "concordance c",
                    "source_version": "concordance-x",
                    "licence": "x",
                    "reference_year": 2024,
                    "retrieved": "2026-09-16",
                },
                "country_archetype": {"CA": "N", "IN": "S"},
                "block_membership": {"BLK": {"CA": 0.5, "IN": 0.5}},
                "sector_archetype": {"manufacturing": "BRD"},
                "block_weights": {
                    "output": {"BLK": {"CA": 0.9, "IN": 0.1}},
                    "population": {
                        "BLK": {"2025": {"CA": 0.5, "IN": 0.5}, "2050": {"CA": 0.2, "IN": 0.8}}
                    },
                },
                "driver_weight_class": {"population": "population", "productivity": "output"},
                "sources": {"x": "x"},
            }
        )
    )
    t = structural_trajectories_for_build(
        ["BLK"],
        ["manufacturing"],
        trajectory_path=str(traj),
        concordance_path=str(conc),
        require_full_coverage=False,
    )
    # productivity uses OUTPUT weights (CA 0.9): 0.9·0.02 + 0.1·0.06 = 0.024.
    assert t.rate("productivity", "BLK", 2025) == pytest.approx(0.9 * 0.02 + 0.1 * 0.06)
    # population uses POPULATION weights, which DRIFT: 2025 (0.5/0.5) vs 2050 (0.2/0.8).
    assert t.rate("population", "BLK", 2025) == pytest.approx(0.5 * 0.005 + 0.5 * 0.02)
    assert t.rate("population", "BLK", 2050) == pytest.approx(0.2 * 0.005 + 0.8 * 0.02)
    # the drift moves the blended population rate UP toward S between 2025 and 2050.
    assert t.rate("population", "BLK", 2050) > t.rate("population", "BLK", 2025)


def _v3_pair(tmp_path):
    """A minimal VALID (trajectory, concordance-v3) pair for the coverage-validation tests."""
    import json

    traj = tmp_path / "traj.json"
    traj.write_text(
        json.dumps(
            {
                "provenance": {
                    "source": "t",
                    "source_version": "t",
                    "licence": "t",
                    "reference_year": 2024,
                    "retrieved": "2026-09-16",
                },
                "rates": {
                    "productivity": {"N": {"2025": 0.02}, "S": {"2025": 0.06}},
                    "population": {"N": {"2025": 0.005}, "S": {"2025": 0.02}},
                },
                "sources": {
                    "productivity:N": "x",
                    "productivity:S": "x",
                    "population:N": "x",
                    "population:S": "x",
                },
                "confidence": {
                    "productivity:N": "low",
                    "productivity:S": "low",
                    "population:N": "low",
                    "population:S": "low",
                },
            }
        )
    )
    conc = {
        "provenance": {
            "source": "c",
            "source_version": "concordance-x",
            "licence": "x",
            "reference_year": 2024,
            "retrieved": "2026-09-16",
        },
        "country_archetype": {"CA": "N", "IN": "S"},
        "block_membership": {"BLK": {"CA": 0.5, "IN": 0.5}},
        "sector_archetype": {"manufacturing": "BRD"},
        "block_weights": {
            "output": {"BLK": {"CA": 0.9, "IN": 0.1}},
            "population": {"BLK": {"2025": {"CA": 0.5, "IN": 0.5}}},
        },
        "driver_weight_class": {"population": "population", "productivity": "output"},
        "sources": {"x": "x"},
    }
    return traj, conc


def test_concordance_v3_rejects_incomplete_coverage(tmp_path):
    """Review-10 P2#4: v3 must NOT silently fall back to v2 static weights. A weight class missing a
    block, and a trajectory driver with no driver_weight_class entry, both fail loudly."""
    import json

    from cge.data.structural import load_structural_concordance, load_structural_trajectories

    traj_path, conc = _v3_pair(tmp_path)
    trajectory = load_structural_trajectories(str(traj_path))

    # baseline: the valid pair loads.
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps(conc))
    load_structural_concordance(str(ok), trajectory=trajectory)

    # (a) output class missing the BLK block → reject.
    import copy

    miss_block = copy.deepcopy(conc)
    miss_block["block_weights"]["output"] = {}  # no BLK
    p1 = tmp_path / "miss_block.json"
    p1.write_text(json.dumps(miss_block))
    with pytest.raises(ValueError, match="missing block"):
        load_structural_concordance(str(p1), trajectory=trajectory)

    # (b) a trajectory driver (population) with no driver_weight_class entry → reject.
    miss_driver = copy.deepcopy(conc)
    del miss_driver["driver_weight_class"]["population"]
    p2 = tmp_path / "miss_driver.json"
    p2.write_text(json.dumps(miss_driver))
    with pytest.raises(ValueError, match="missing region driver"):
        load_structural_concordance(str(p2), trajectory=trajectory)

    # (c) review-11 P2#3: a weight for a country from a DIFFERENT block (BLK members are CA+IN; swap
    # IN→a mapped-but-non-member country) must be rejected even though it is in country_archetype.
    wrong_block = copy.deepcopy(conc)
    wrong_block["country_archetype"]["ZZ"] = "S"  # mapped, but not a BLK member
    wrong_block["block_weights"]["output"]["BLK"] = {"CA": 0.9, "ZZ": 0.1}
    p3 = tmp_path / "wrong_block.json"
    p3.write_text(json.dumps(wrong_block))
    with pytest.raises(ValueError, match="do not match block_membership"):
        load_structural_concordance(str(p3), trajectory=trajectory)


def test_concordance_rejects_unknown_emissions_channel(tmp_path):
    """review-11 P1: a sector_emissions_channel target that the emissions_intensity trajectory does
    not carry (a typo) must fail loudly, not silently fall through to __all__."""
    import json

    from cge.data.structural import load_structural_concordance, load_structural_trajectories

    # A trajectory carrying per-channel emissions keys.
    traj = tmp_path / "traj.json"
    traj.write_text(
        json.dumps(
            {
                "provenance": {
                    "source": "t",
                    "source_version": "t",
                    "licence": "t",
                    "reference_year": 2024,
                    "retrieved": "2026-09-19",
                },
                "rates": {"productivity": {"__all__": {"2025": 0.01}}},
                "sector_rates": {
                    "emissions_intensity": {
                        "electricity": {"2025": -0.05},
                        "__all__": {"2025": -0.02},
                    }
                },
                "sources": {
                    "productivity:__all__": "x",
                    "emissions_intensity:electricity": "x",
                    "emissions_intensity:__all__": "x",
                },
                "confidence": {
                    "productivity:__all__": "low",
                    "emissions_intensity:electricity": "low",
                    "emissions_intensity:__all__": "low",
                },
            }
        )
    )
    trajectory = load_structural_trajectories(str(traj))
    conc = {
        "provenance": {
            "source": "c",
            "source_version": "concordance-x",
            "licence": "x",
            "reference_year": 2024,
            "retrieved": "2026-09-19",
        },
        "region_archetype": {"US": "__all__"},
        "sector_archetype": {"electricity": "__all__"},
        # blend form {sector: {channel: weight}} with a typo'd channel → not a trajectory channel.
        "sector_emissions_channel": {"electricity": {"ELECTRICTY_TYPO": 1.0}},
        "sources": {"x": "x"},
    }
    p = tmp_path / "conc.json"
    p.write_text(json.dumps(conc))
    with pytest.raises(ValueError, match="sector_emissions_channel channel"):
        load_structural_concordance(str(p), trajectory=trajectory)

    # review-12 P2#4: a sector present in sector_archetype but MISSING from sector_emissions_channel
    # must raise (no silent __all__ fallback). Add a second sector with no channel entry.
    conc2 = dict(conc)
    conc2["sector_archetype"] = {"electricity": "__all__", "manufacturing": "__all__"}
    conc2["sector_emissions_channel"] = {
        "electricity": {"electricity": 1.0}
    }  # manufacturing absent
    p2 = tmp_path / "conc2.json"
    p2.write_text(json.dumps(conc2))
    with pytest.raises(ValueError, match="missing sector"):
        load_structural_concordance(str(p2), trajectory=trajectory)

    # review-13 P2: channel weights must be a non-empty CONVEX COMBINATION — non-numeric,
    # negative, non-finite, or non-sum-to-1 weights must all raise (a non-convex blend like
    # {electricity: -0.2, __all__: 1.2} was previously accepted).
    def _conc_with(blend):
        c = dict(conc)
        c["sector_archetype"] = {"electricity": "__all__"}
        c["sector_emissions_channel"] = {"electricity": blend}
        pth = tmp_path / f"conc_w_{abs(hash(str(blend)))}.json"
        pth.write_text(json.dumps(c))
        return str(pth)

    with pytest.raises(ValueError, match="non-negative"):
        load_structural_concordance(
            _conc_with({"electricity": -0.2, "__all__": 1.2}), trajectory=trajectory
        )
    with pytest.raises(ValueError, match="sum to 1"):
        load_structural_concordance(_conc_with({"electricity": 0.5}), trajectory=trajectory)
    with pytest.raises(ValueError, match="non-numeric"):
        load_structural_concordance(_conc_with({"electricity": "lots"}), trajectory=trajectory)
    with pytest.raises(ValueError, match="non-empty"):
        load_structural_concordance(_conc_with({}), trajectory=trajectory)


def test_sensitivity_cases_adverse_central_favourable():
    """Review-10 P1#2: structural_trajectory_variants returns adverse/central/favourable
    DETERMINISTIC sensitivity cases. For a growth driver the INPUT rates order adverse ≤ central ≤
    favourable (the labels order the drivers by favourability). The siblings carry the sensitivity
    provenance and are explicitly NOT called intervals/bounds."""
    from cge.data.structural import structural_trajectory_variants

    v = structural_trajectory_variants()  # raw archetype cases
    assert set(v) == {"adverse", "central", "favourable"}
    ad = v["adverse"].rate("productivity", "N", 2019)
    ce = v["central"].rate("productivity", "N", 2019)
    fa = v["favourable"].rate("productivity", "N", 2019)
    assert ad <= ce <= fa and ad < fa  # input drivers ordered by favourability
    assert "adverse" in v["adverse"].provenance.source_version
    notes = (v["adverse"].provenance.notes or "").lower()
    assert (
        "sensitivity" in notes and "not a statistical" in notes
    )  # honest framing, not an interval
    # emissions-intensity: adverse = SLOWER decarbonisation (less negative) than central.
    ad_e = v["adverse"].sector_rate("emissions_intensity", "__all__", 2025)
    ce_e = v["central"].sector_rate("emissions_intensity", "__all__", 2025)
    assert ad_e > ce_e  # less negative = slower decarbonisation


def test_sensitivity_cases_map_onto_build_labels():
    """The cases can be mapped onto real build labels for a sweep, sharing the concordance."""
    from cge.data.structural import structural_trajectory_variants

    v = structural_trajectory_variants(["US", "RoW_Asia"], ["manufacturing"])
    assert set(v) == {"adverse", "central", "favourable"}
    for label in ("adverse", "central", "favourable"):
        assert v[label].rate("productivity", "US", 2019) is not None


def test_variants_require_both_regions_and_sectors_or_neither():
    """Review-10 P3: passing exactly one of regions/sectors raises rather than silently returning
    archetype cases."""
    from cge.data.structural import structural_trajectory_variants

    with pytest.raises(ValueError, match="BOTH regions and sectors"):
        structural_trajectory_variants(["US"], None)
    with pytest.raises(ValueError, match="BOTH regions and sectors"):
        structural_trajectory_variants(None, ["manufacturing"])


def test_structural_sensitivity_bounds_take_realised_per_output_minmax():
    """Review-10 P1#2: output bounds are the REALISED per-output min/max across cases, not the
    adverse/favourable labels — a nonlinear output need not be monotone in input favourability."""
    from cge.data.structural import structural_sensitivity_bounds

    # adverse GDP happens to exceed favourable for this output (nonlinearity) — bounds must still be
    # the true min/max, not label-ordered.
    results = {
        "adverse": {"gdp": 1.02, "emissions": 0.90},
        "central": {"gdp": 1.05, "emissions": 0.80},
        "favourable": {"gdp": 1.01, "emissions": 0.70},
    }
    bounds = structural_sensitivity_bounds(results)
    assert bounds["gdp"] == (1.01, 1.05)  # realised min/max, NOT (adverse, favourable)
    assert bounds["emissions"] == (0.70, 0.90)


def test_structural_sensitivity_bounds_rejects_incomplete_or_nonfinite(tmp_path):
    """Review-11 P2#5: the reporting helper must not silently produce a range from one case or from
    NaNs — it validates non-empty, exact case set, matching output keys, and finite values."""
    import math

    from cge.data.structural import structural_sensitivity_bounds

    with pytest.raises(ValueError, match="empty"):
        structural_sensitivity_bounds({})
    # missing a case
    with pytest.raises(ValueError, match="!= expected"):
        structural_sensitivity_bounds({"adverse": {"gdp": 1.0}, "central": {"gdp": 1.0}})
    # mismatched output keys across cases (emissions only in one)
    with pytest.raises(ValueError, match="different output keys"):
        structural_sensitivity_bounds(
            {
                "adverse": {"gdp": 1.0},
                "central": {"gdp": 1.0, "emissions": 2.0},
                "favourable": {"gdp": 1.0},
            }
        )
    # non-finite value
    with pytest.raises(ValueError, match="non-finite"):
        structural_sensitivity_bounds(
            {
                "adverse": {"gdp": math.nan},
                "central": {"gdp": 1.0},
                "favourable": {"gdp": 1.0},
            }
        )


def test_concordance_v2_carries_composite_provenance():
    """Review P2 2026-08-29: the mapped trajectory must record BOTH the concordance and archetype
    identities, not just the bare structural-trajectories-v1 provenance."""
    from cge.data.structural import structural_trajectories_for_build

    t = structural_trajectories_for_build(["US", "CN"], ["manufacturing", "services"])
    assert "concordance" in t.provenance.source.lower()
    # concordance v4 (review-12): per-sector emissions-channel blend + working_age class.
    # trajectories v5 (review-12): per-channel NGFS emissions intensities + 15+ labour population.
    assert "structural-concordance-v5" in t.provenance.source_version
    assert "structural-trajectories-v5" in t.provenance.source_version


def test_mapped_trajectory_stamps_the_weighting_note():
    """Review P1 2026-08-31 + review-9 1c: the weighting basis + residual-block limitation must be
    recorded in the mapped trajectory's provenance, auditable-in-place. With the default v3
    concordance the note describes the PER-DRIVER, TIME-VARYING weights and the residual-W*
    proxy — not the old single-static-GDP caveat."""
    from cge.data.structural import structural_trajectories_for_build

    t = structural_trajectories_for_build(["US", "CN", "RoW_Asia"], ["manufacturing", "services"])
    notes = t.provenance.notes
    assert "PER-DRIVER, TIME-VARYING" in notes  # v3 weights
    assert "population" in notes.lower() and "output shares" in notes.lower()
    assert "RoW_MiddleEast" in notes  # the residual-W* proxy is still disclosed


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


def test_concordance_rejects_disjoint_archetype_coverage_across_drivers(tmp_path):
    """Review P2 2026-09-05: per-driver validation must REJECT a trajectory where the axis's drivers
    cover DISJOINT archetypes, so a block mapped to both cannot resolve one driver each. Here a
    mixed block AB maps to members with archetypes N and S, but ``population`` carries only N and
    ``productivity`` only S — at blend time population would silently drop the S member (→100% N)
    and productivity the N member (→100% S), making the SAME block diverge per driver. The old
    union-of-archetypes check passed this ({N,S} existed across the union); the per-driver check
    fails it."""
    import json

    from cge.contracts.data_objects import Provenance, StructuralTrajectory
    from cge.data.structural.library import load_structural_concordance

    prov = Provenance(
        source="t", source_version="v", licence="n/a", reference_year=2024, retrieved="2026-09-05"
    )
    # population resolves ONLY N; productivity resolves ONLY S — neither has an __all__ fallback.
    traj = StructuralTrajectory(
        provenance=prov,
        rates={
            "population": {"N": {2025: 0.01}},
            "productivity": {"S": {2025: 0.02}},
        },
        sources={"population:N": "c", "productivity:S": "c"},
        confidence={"population:N": "medium", "productivity:S": "medium"},
    )
    conc = {
        "provenance": {
            "source": "t",
            "source_version": "v",
            "licence": "n/a",
            "reference_year": 2024,
            "retrieved": "2026-09-05",
        },
        "sector_archetype": {"anySector": "__all__"},  # sector axis absent in traj → not checked
        "country_archetype": {"CA": "N", "CB": "S"},
        "block_membership": {"AB": {"CA": 0.5, "CB": 0.5}},
    }
    p = tmp_path / "disjoint.json"
    p.write_text(json.dumps(conc))
    with pytest.raises(ValueError, match="cannot resolve concordance archetype"):
        load_structural_concordance(p, trajectory=traj)


def test_concordance_accepts_all_only_uniform_driver(tmp_path):
    """Review P2 2026-09-06: an intentionally UNIFORM trajectory whose only region path is
    ``population:__all__`` must NOT be rejected against an N/S concordance — ``__all__`` resolves
    every archetype, so the deliberate uniform fallback is legitimate. (The earlier 'real archetype'
    check rejected it because no driver supplied an EXPLICIT N or S.)"""
    import json

    from cge.contracts.data_objects import Provenance, StructuralTrajectory
    from cge.data.structural.library import load_structural_concordance

    prov = Provenance(
        source="t", source_version="v", licence="n/a", reference_year=2024, retrieved="2026-09-06"
    )
    traj = StructuralTrajectory(
        provenance=prov,
        rates={"population": {"__all__": {2025: 0.01}}},  # uniform: no explicit N/S
        sources={"population:__all__": "c"},
        confidence={"population:__all__": "low"},
    )
    conc = {
        "provenance": {
            "source": "t",
            "source_version": "v",
            "licence": "n/a",
            "reference_year": 2024,
            "retrieved": "2026-09-06",
        },
        "sector_archetype": {"anySector": "__all__"},
        "country_archetype": {"CA": "N", "CB": "S"},
        "block_membership": {"AB": {"CA": 0.5, "CB": 0.5}},
    }
    p = tmp_path / "uniform.json"
    p.write_text(json.dumps(conc))
    # Must NOT raise — the uniform population path resolves N and S via __all__.
    assert load_structural_concordance(p, trajectory=traj) is not None


def test_lock_is_hash_pinned_and_platform_marked():
    """Review P2 2026-09-05: requirements.lock must be a real deterministic lock — hash-pinned
    (``--hash=sha256:`` per package) and platform-marked so a non-Darwin install pulls Streamlit's
    watchdog dependency (the reviewer's specific omission under a plain pip freeze)."""
    from pathlib import Path

    lock = Path("requirements.lock")
    assert lock.exists(), "requirements.lock must be committed (generated by scripts/gen_lock.py)"
    text = lock.read_text()
    assert "--hash=sha256:" in text, "lock must be hash-pinned (--generate-hashes)"
    # watchdog is a Streamlit runtime dep on non-Darwin platforms; the universal lock carries it
    # with a platform marker so a Linux deployment installs it.
    assert "watchdog==" in text and "sys_platform" in text


def test_gen_lock_verify_is_deterministic_and_covers_declared_deps():
    """review-14 P2#5: --verify is the DETERMINISTIC CI gate — it must pass on the committed lock
    WITHOUT re-resolving against the package index (no network, no uv), and it must flag a lock that
    omits a declared direct dependency (a real input change). We assert both the committed-lock pass
    and the declared-vs-locked coverage logic."""
    from scripts.gen_lock import _declared_top_level, _locked_names, _verify_frozen

    assert _verify_frozen() == 0  # committed lock is frozen & complete — offline, deterministic
    declared = _declared_top_level()
    locked = _locked_names()
    assert declared, "pyproject must declare direct dependencies"
    # Every DIRECT dependency the project declares is pinned in the lock (the coverage invariant).
    assert declared <= locked, f"lock missing declared deps: {sorted(declared - locked)}"
    # A couple of known direct deps are present (sanity that parsing found real names).
    assert {"numpy", "pydantic"} <= locked


def test_vendored_artifact_reproduces_from_source_digest():
    """Pipeline step 1a (2026-09-06): the committed trajectories_v1.json must be REPRODUCIBLE from
    the committed source digest — the build --check passes — so the artifact can never drift
    from its documented inputs. Also asserts the assembled content matches the loaded trajectory."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    from cge.data.structural import load_structural_trajectories

    root = Path(__file__).resolve().parent.parent
    r = subprocess.run(
        [sys.executable, "scripts/build_structural_trajectories.py", "--check"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"artifact drifted from digest:\n{r.stdout}\n{r.stderr}"

    # The digest's source_labour_shares reach the loaded trajectory (proves the 1b field round-trips
    # through the reproducible build path, not just a hand-edited artifact). The digest carries
    # value-level provenance ({key: {value, source, confidence}}); the loaded field is the flat
    # {key: value}.
    digest = json.loads((root / "data/structural/sources/inputs.json").read_text())
    traj = load_structural_trajectories()
    assert traj.source_labour_shares == {
        k: float(spec["value"]) for k, spec in digest["source_labour_shares"].items()
    }


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


def test_non_co2_carbon_price_rejected_with_co2_emissions_trajectory():
    """review-12 P1#1: the structural emissions_intensity path is CO2-derived, so a CH4/CO2e
    CarbonPrice combined with it must RAISE (not silently inherit the CO2 decarbonisation path). A
    CO2-only price is fine; a trajectory WITHOUT an emissions driver imposes no restriction."""
    from cge.contracts.shocks import CarbonPrice
    from cge.dynamics import DynamicConfig, run_recursive
    from cge.scenarios.loader import Scenario

    emis = StructuralTrajectory(
        provenance=Provenance(
            source="t", source_version="v", licence="n", reference_year=2024, retrieved="2026-10-14"
        ),
        rates={"productivity": {"__all__": {2025: 0.0}}},
        sector_rates={"emissions_intensity": {"__all__": {2025: -0.05}}},
        sources={"productivity:__all__": "c", "emissions_intensity:__all__": "c"},
        confidence={"productivity:__all__": "low", "emissions_intensity:__all__": "low"},
    )
    cfg = DynamicConfig(structural=emis, allow_uniform_fallback=True)
    ch4 = Scenario(
        name="x",
        engine="cge_static",
        years=[2025, 2030],
        shocks=[CarbonPrice(price=50.0, gases=["CH4"])],
    )
    with pytest.raises(ValueError, match="CO2-only"):
        run_recursive(ch4, config=cfg, data_source="toy_cge_gov")
    # CO2-only price is accepted (reaches the solve).
    co2 = Scenario(
        name="x", engine="cge_static", years=[2025, 2030], shocks=[CarbonPrice(price=50.0)]
    )
    assert run_recursive(co2, config=cfg, data_source="toy_cge_gov").result is not None


def test_final_manifest_preserves_concordance_hashes_and_weighting_caveat():
    """Review P2 2026-09-03: a concordance-mapped trajectory records the concordance/archetype
    CONTENT HASHES and the weighting caveat in provenance.notes, and these must survive the
    run-manifest transformation into the FINAL ResultSet (previously _trend_provenance projected
    only source/version/licence/retrieved and DROPPED them). Assert they reach result.manifest
    end-to-end, both as the full provenance object and as the explicit audit fields."""
    from cge.contracts.shocks import CarbonPrice
    from cge.data.structural import structural_trajectories_for_build
    from cge.dynamics import DynamicConfig, run_recursive
    from cge.scenarios.loader import Scenario

    mapped = structural_trajectories_for_build(["US", "CN"], ["manufacturing", "services"])
    # Sanity: the mapped trajectory really carries the hashes + weighting note (source of truth).
    assert "concordance_content_hash=" in mapped.provenance.notes
    assert "PER-DRIVER, TIME-VARYING" in mapped.provenance.notes  # v3 weighting note

    sc = Scenario(
        name="x", engine="cge_static", years=[2025, 2030], shocks=[CarbonPrice(price=50.0)]
    )
    # The mapped trajectory keys real EXIOBASE labels, not the toy build's, so it falls through to
    # __all__ here — opt into the uniform fallback so the run proceeds; the provenance flow is what
    # we are testing, independent of label coverage.
    path = run_recursive(
        sc,
        config=DynamicConfig(structural=mapped, allow_uniform_fallback=True),
        data_source="toy_cge_gov",
    )
    trend = path.result.manifest.assumptions["recursive_dynamics"]["trend_source"]
    # Full provenance object preserved (notes carried through, not just source/version/licence).
    assert "PER-DRIVER, TIME-VARYING" in trend["provenance"]["notes"]
    assert "concordance_content_hash=" in trend["provenance"]["notes"]
    # Explicit audit fields surfaced at the top level (no prose-parsing needed downstream).
    assert trend["concordance_content_hash"] == _hash_in(mapped.provenance.notes, "concordance")
    assert trend["archetype_content_hash"] == _hash_in(mapped.provenance.notes, "archetype")
    assert "PER-DRIVER, TIME-VARYING" in trend["weighting_note"]


def _hash_in(notes: str, which: str) -> str:
    import re

    return re.search(rf"{which}_content_hash=([0-9a-fA-F]+)", notes).group(1)
