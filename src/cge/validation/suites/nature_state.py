"""Validation suite for the physical nature-state layer (Phase 6b).

Standing model-correctness checks tied to docs/models/nature-state.md — the load-bearing invariants
of the state→severity→NatureStress chain, so a regression re-surfaces in the battery, not only in
pytest:

- the linear response is proportional and bounded in [0, 1];
- the opt-in threshold adds a convex penalty ONLY below x_crit (and never below the linear value);
- recovery hysteresis lags restoration but not degradation;
- a shipped channel's severity path translates into a NatureStress carrying a per-year path;
- the 6b.5 double-counting check flags a shared mechanism and clears disjoint ones.

These run on the shipped channels + a synthetic pathway (no ENCORE data required), exactly like the
6b unit path.
"""

from __future__ import annotations

from cge.validation.framework import check

SUITE = "nature_state"


@check(SUITE, "linear_response_proportional_and_bounded")
def _linear_response():
    from cge.nature.state import StateResponse

    r = StateResponse(baseline=100.0)  # sensitivity 1.0
    ok = (
        r.severity(100.0) == 0.0
        and abs(r.severity(70.0) - 0.30) < 1e-12
        and r.severity(110.0) == 0.0  # surplus → no shortfall
        and all(0.0 <= r.severity(s) <= 1.0 for s in (0, 25, 50, 75, 100, 200))
    )
    return ok, "linear severity = shortfall, clamped to [0,1]; surplus → 0", r.severity(70.0)


@check(SUITE, "threshold_convex_only_below_x_crit")
def _threshold():
    from cge.nature.state import StateResponse

    r = StateResponse(baseline=100.0, sensitivity=1.0, threshold=0.5, threshold_exponent=2.0)
    above_is_linear = abs(r.severity(60.0) - 0.40) < 1e-12  # above x_crit: pure linear
    below_is_convex = r.severity(40.0) > 0.60  # below x_crit: convex boost over linear
    deeper_is_larger = r.severity(10.0) > r.severity(40.0)
    ok = above_is_linear and below_is_convex and deeper_is_larger
    return ok, "threshold response is convex below x_crit, linear above", r.severity(40.0)


@check(SUITE, "recovery_hysteresis_lags_restoration")
def _recovery():
    from cge.nature.state import StatePathway

    p = StatePathway(
        channel_id="water_availability",
        states={2025: 100.0, 2030: 60.0, 2031: 100.0},
        recovery_rate=5.0,
    )
    path = p.state_path([2030, 2031, 2032])
    # Degradation prompt (60 at 2030); recovery capped at +5/yr despite the physical jump to 100.
    ok = path[2030] == 60.0 and abs(path[2031] - 65.0) < 1e-9 and abs(path[2032] - 70.0) < 1e-9
    return ok, "effective state falls freely, recovers ≤ recovery_rate/yr", path[2031]


@check(SUITE, "state_translates_to_nature_stress_path")
def _translate():
    from cge.nature.state import StatePathway, build_state_scenario, get_channel

    w = get_channel("water_availability")
    years = [2025, 2030, 2035, 2040]
    stresses = build_state_scenario(
        [(w, StatePathway(channel_id="water_availability", states={2030: 90.0, 2040: 70.0}))],
        years,
    )
    # One stress per water COMPONENT service, each with a per-year severity path peaking at ~0.30.
    services = {s.service for s in stresses}
    ok = (
        services == set(w.services)
        and all(s.path is not None for s in stresses)
        and all(abs(s.path[2040] - 0.30) < 1e-9 for s in stresses)
    )
    return ok, "physical pathway → NatureStress per service with a per-year path", len(stresses)


@check(SUITE, "double_counting_flags_shared_mechanism")
def _double_count():
    from cge.nature.state import check_double_counting

    conflict = check_double_counting(["water_availability"], ["water_availability"])
    disjoint = check_double_counting(["pollination"], ["water_availability"])
    ok = (not conflict.ok) and disjoint.ok
    return ok, "shared mechanism flagged, disjoint mechanisms cleared (6b.5)", None
