"""Tests for Phase 6b — physical ecosystem-service state modelling.

Covers the state→severity response (6b.1/6b.4), degradation/restoration pathways with recovery
hysteresis (6b.2/6b.4), translation into the Phase-6.4 NatureStress vocabulary (6b.3), the
double-counting reconciliation (6b.5), and the end-to-end path through the real ENCORE pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cge.contracts.data_objects import Classification, IOSystem, Provenance, SatelliteAccount
from cge.nature.state import (
    DoubleCountError,
    ServiceStateChannel,
    StatePathway,
    StateResponse,
    build_state_scenario,
    check_double_counting,
    get_channel,
    nature_mechanisms_of,
    shipped_channels,
    state_severity_path,
    state_to_nature_stresses,
)

try:
    from cge.nature.real import encore_data_available

    _HAS_ENCORE = encore_data_available()
except Exception:  # pragma: no cover
    _HAS_ENCORE = False

_needs_encore = pytest.mark.skipif(not _HAS_ENCORE, reason="vendored ENCORE data not present")


def _prov():
    return Provenance(
        source="x", source_version="1", licence="x", reference_year=2020, retrieved="2026-08-15"
    )


# --- 6b.1 / 6b.4: the state -> severity response ----------------------------------------------


def test_linear_response_is_proportional_by_default():
    """The default response is linear in the fractional shortfall: a 30% state shortfall at unit
    sensitivity is a 30% severity, and state at/above baseline is severity 0."""
    r = StateResponse(baseline=100.0)  # sensitivity defaults to 1.0
    assert r.severity(100.0) == 0.0
    assert r.severity(110.0) == 0.0  # surplus, no shortfall
    assert r.severity(70.0) == pytest.approx(0.30)
    assert r.severity(0.0) == pytest.approx(1.0)


def test_sensitivity_scales_the_response():
    r = StateResponse(baseline=100.0, sensitivity=0.6)
    assert r.severity(50.0) == pytest.approx(0.30)  # 50% shortfall * 0.6


def test_threshold_adds_convex_penalty_only_below_x_crit():
    """The opt-in threshold accelerates degradation below x_crit but leaves the response linear
    above it — and never exceeds the linear value above the threshold."""
    r = StateResponse(baseline=100.0, sensitivity=1.0, threshold=0.5, threshold_exponent=2.0)
    assert r.severity(60.0) == pytest.approx(0.40)  # above x_crit: pure linear
    below = r.severity(40.0)
    assert below > 0.60  # below x_crit: convex boost above the linear 0.60
    assert r.severity(10.0) > r.severity(40.0)  # deeper collapse, larger severity
    for state in (0, 25, 50, 75, 100, 150):
        assert 0.0 <= r.severity(state) <= 1.0


def test_response_rejects_bad_threshold():
    with pytest.raises(ValueError, match="threshold"):
        StateResponse(baseline=100.0, threshold=1.5)


def test_shipped_channels_are_well_formed():
    """Every shipped channel drives at least one ENCORE service, carries provenance + a source note,
    and its severities stay in [0, 1]."""
    chs = shipped_channels()
    assert {
        "water_availability",
        "pollination",
        "soil_quality",
        "forestry_stock",
        "fisheries_stock",
    } <= set(chs)
    for ch in chs.values():
        assert ch.services and all(s.strip() for s in ch.services)
        assert ch.source_note and ch.provenance.source
        for state in (0, 50, 100, 200):
            assert 0.0 <= ch.severity(state) <= 1.0


@_needs_encore
def test_shipped_channel_services_are_real_encore_labels():
    """Each shipped channel maps to actual ENCORE service labels (not typos) — else the downstream
    translate layer would reject them."""
    from cge.nature.real import real_encore_dependencies

    services = set(real_encore_dependencies().services)
    for ch in shipped_channels().values():
        for svc in ch.services:
            assert svc in services, f"{ch.channel_id} -> unknown service {svc!r}"


# --- 6b.2 / 6b.4: pathways + recovery hysteresis ----------------------------------------------


def test_pathway_explicit_states_interpolate_piecewise_linear():
    p = StatePathway(channel_id="water_availability", states={2030: 80.0, 2040: 60.0})
    path = p.state_path([2025, 2030, 2035, 2040, 2045])
    assert path[2025] == 100.0  # baseline before the first point
    assert path[2030] == 80.0
    assert path[2035] == pytest.approx(70.0)  # midpoint
    assert path[2045] == 60.0  # flat-extrapolated


def test_pathway_rate_form_degrades_from_start():
    p = StatePathway(channel_id="soil_quality", degradation_rate=2.0)
    path = p.state_path([2025, 2030, 2035])
    assert path[2025] == 100.0
    assert path[2030] == pytest.approx(90.0)  # 2 pts/yr * 5 yr
    assert path[2035] == pytest.approx(80.0)


def test_recovery_hysteresis_lags_restoration_not_degradation():
    """Degradation is prompt; a restored physical state lets the effective state recover by at most
    recovery_rate per year (6b.4)."""
    p = StatePathway(
        channel_id="water_availability",
        states={2025: 100.0, 2030: 60.0, 2031: 100.0},
        recovery_rate=5.0,
    )
    path = p.state_path([2025, 2030, 2031, 2032, 2040])
    assert path[2030] == 60.0  # degradation felt promptly
    assert path[2031] == pytest.approx(65.0)  # physical jumped to 100, effective capped at +5
    assert path[2032] == pytest.approx(70.0)
    assert path[2040] == pytest.approx(100.0)  # eventually catches up


def test_pathway_requires_exactly_one_form():
    with pytest.raises(ValueError, match="exactly one"):
        StatePathway(channel_id="x", states={2030: 80.0}, degradation_rate=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        StatePathway(channel_id="x")


# --- 6b.3: translation to NatureStress --------------------------------------------------------


def test_state_translation_emits_one_stress_per_service_with_severity_path():
    w = get_channel("water_availability")
    p = StatePathway(channel_id="water_availability", states={2030: 90.0, 2040: 70.0})
    years = [2025, 2030, 2035, 2040]
    sev = state_severity_path(w, p, years)
    assert sev[2025] == 0.0 and sev[2040] == pytest.approx(0.30)
    stresses = state_to_nature_stresses(w, p, years, coverage_regions=["DE"])
    assert {s.service for s in stresses} == set(w.services)
    for s in stresses:
        assert s.coverage_regions == ["DE"]
        assert s.path[2040] == pytest.approx(0.30)
        assert s.severity == pytest.approx(max(sev.values()))  # scalar = peak


def test_flat_pathway_is_a_noop():
    w = get_channel("water_availability")
    flat = StatePathway(channel_id="water_availability", states={2030: 100.0, 2040: 100.0})
    assert state_to_nature_stresses(w, flat, [2030, 2040]) == []


def test_translation_rejects_mismatched_channel_and_pathway():
    w = get_channel("water_availability")
    wrong = StatePathway(channel_id="pollination", degradation_rate=1.0)
    with pytest.raises(ValueError, match="matching pair"):
        state_to_nature_stresses(w, wrong, [2030])


def test_build_state_scenario_composes_channels():
    w = get_channel("water_availability")
    poll = get_channel("pollination")
    items = [
        (w, StatePathway(channel_id="water_availability", states={2030: 80.0})),
        (poll, StatePathway(channel_id="pollination", degradation_rate=1.0)),
    ]
    stresses = build_state_scenario(items, [2025, 2030])
    services = {s.service for s in stresses}
    assert "Pollination" in services
    assert "Water flow regulation" in services


# --- 6b.5: double-counting reconciliation -----------------------------------------------------


def test_double_counting_flags_shared_mechanism():
    report = check_double_counting(["water_availability"], ["water_availability"])
    assert not report.ok
    with pytest.raises(DoubleCountError, match="double-count"):
        report.raise_if_conflict()


def test_double_counting_allows_disjoint_mechanisms():
    report = check_double_counting(["pollination"], ["water_availability"])
    assert report.ok
    report.raise_if_conflict()  # does not raise


def test_nature_mechanisms_of_helper():
    chs = [get_channel("water_availability"), get_channel("soil_quality")]
    assert nature_mechanisms_of(chs) == {"water_availability", "soil_quality"}


# --- end-to-end through the real ENCORE pipeline ----------------------------------------------


@_needs_encore
def test_state_pathway_runs_end_to_end_through_build_nature_shocks():
    """The Phase-6b DoD: a physical degradation pathway propagates through the Phase-6.4 translation
    into ProductivityShocks, with no bare severity number in the scenario."""
    from cge.nature.real import real_encore_concordance, real_encore_dependencies
    from cge.nature.translate import build_nature_shocks

    dep = real_encore_dependencies()
    cmap, _ = real_encore_concordance()
    secs = list(cmap.weights)[:3]
    labels = [f"DE:{s}" for s in secs]
    io = IOSystem(
        provenance=_prov(),
        sectors=Classification(name="e", kind="sector", labels=secs),
        regions=Classification(name="r", kind="region", labels=["DE"]),
        A=pd.DataFrame(np.full((3, 3), 0.05), index=labels, columns=labels),
        final_demand=pd.DataFrame({"final_demand": [100.0, 100.0, 100.0]}, index=labels),
        unit="MEUR",
        currency="EUR",
    )
    w = get_channel("water_availability")
    pathway = StatePathway(channel_id="water_availability", states={2030: 90.0, 2040: 70.0})
    years = [2030, 2040]
    stresses = build_state_scenario([(w, pathway)], years)
    assert stresses  # a physical pathway produced NatureStress objects
    shocks = build_nature_shocks(stresses, io, dep, cmap, years=years)
    assert shocks  # …which the existing pipeline turned into ProductivityShocks


@_needs_encore
def test_state_scenario_runs_from_run_scenario(tmp_path):
    """Full orchestration: a 6b physical-pathway-derived NatureStress list runs via run_scenario
    (store → runner → engine → ResultSet) exactly like a hand-written Phase-6 nature scenario."""
    from cge.data.metadata import BuildMeta
    from cge.data.store import DataStore
    from cge.nature.real import real_encore_concordance, real_encore_dependencies
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    dep = real_encore_dependencies()
    cmap, _ = real_encore_concordance()
    secs = list(cmap.weights)[:2]
    labels = [f"DE:{s}" for s in secs]
    io = IOSystem(
        provenance=_prov(),
        sectors=Classification(name="e", kind="sector", labels=secs),
        regions=Classification(name="r", kind="region", labels=["DE"]),
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
    w = get_channel("water_availability")
    stresses = build_state_scenario(
        [(w, StatePathway(channel_id="water_availability", states={2030: 70.0}))], [2030]
    )
    sc = Scenario(name="nature-state", engine="partial_eq", years=[2030], shocks=stresses)
    res = run_scenario(sc, data_source="b", store=store)
    res.validate_schema()
    d = res.data
    vol = d[(d["variable"] == "volume_change") & (d["scenario"] == "central")]
    assert (vol["value"] < 0.0).any()  # the modelled physical degradation hit output


def test_scenario_nature_state_expands_to_shocks():
    """A scenario file can express a PHYSICAL pathway (nature_state) that expands into NatureStress
    shocks — no bare severity number in the scenario (Phase 6b.3, YAML wiring)."""
    from cge.scenarios.loader import Scenario

    sc = Scenario(
        name="water-decline",
        engine="partial_eq",
        years=[2030, 2040],
        nature_state=[
            {"channel": "toy_water", "states": {2030: 90, 2040: 70}, "coverage_regions": ["reg1"]}
        ],
    )
    expanded = sc.expanded_shocks([2030, 2040])
    assert expanded  # the physical pathway produced shocks
    assert all(s.type == "nature_stress" for s in expanded)
    assert {s.service for s in expanded} == {"surface_water"}
    assert all(s.coverage_regions == ["reg1"] for s in expanded)
    assert expanded[0].path[2040] == pytest.approx(0.30)


def test_example_nature_state_scenario_runs_on_toy():
    """The shipped example scenario file runs end-to-end on the offline toy source — the tutorial's
    runnable Phase-6b example."""
    from cge.runner import run_scenario
    from cge.scenarios.loader import load_scenario

    sc = load_scenario("examples/nature_state_water.yaml")
    res = run_scenario(sc, data_source="toy")
    res.validate_schema()
    d = res.data
    vol = d[(d["variable"] == "volume_change") & (d["scenario"] == "central")]
    assert (vol["value"] < 0.0).any()  # the modelled physical degradation hit output
    assert "nature" in res.manifest.assumptions  # full nature provenance recorded


def test_custom_channel_can_be_defined():
    """A user can define their own ServiceStateChannel (the model is open, not a closed set)."""
    ch = ServiceStateChannel(
        channel_id="my_channel",
        mechanism="water_availability",
        services=("Water purification",),
        state_variable="my state",
        unit="index (baseline = 100)",
        response=StateResponse(baseline=100.0, sensitivity=0.5),
        provenance=_prov(),
        source_note="my documented assumption",
    )
    assert ch.severity(50.0) == pytest.approx(0.25)
