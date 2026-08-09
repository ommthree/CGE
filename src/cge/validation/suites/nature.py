"""Validation suite for the nature/ENCORE extension (Phase 6).

Standing model-correctness checks tied to docs/models/nature-encore.md — the counterpart to the
per-engine suites, and the suite the second review (2026-08-07) noted was missing. These assert the
load-bearing invariants AND the specific defects the two reviews found, so a regression re-surfaces
in the battery, not just in pytest:

- the exposure risk invariant ``total ≥ direct`` and the [0, 1] bound;
- the emissions/revenue contract under a combined carbon + productivity shock (revenue = Σ cc·X,
  the wedge is physical and NOT scaled by θ);
- a NatureStress **time path** produces a per-year shock (not the collapsed scalar);
- a **region-scoped** shock is not economy-wide in the collapsed CGE;
- direct vs. total **incidence** genuinely differ (the CGE default avoids double-counting upstream);
- a nature run through the standard runner carries full **provenance** in its manifest.

All run on the toy economy + the illustrative (synthetic) ENCORE fixture, exactly like a nature GUI
run — so the battery exercises the same path a user does.
"""

from __future__ import annotations

import numpy as np

from cge.validation.framework import check

SUITE = "nature"


def _exposure():
    from cge.nature.concord import broadcast_to_goods, sector_scores
    from cge.nature.exposure import compute_exposure
    from cge.nature.fixture import encore_fixture, toy_encore_concordance
    from cge.validation.toy import toy_economy

    io, _ = toy_economy()
    direct = broadcast_to_goods(
        sector_scores(encore_fixture(), toy_encore_concordance(), io.sectors.labels),
        list(io.A.columns),
    )
    total, aligned = compute_exposure(io.A, direct)
    return total, aligned


@check(SUITE, "exposure_total_ge_direct_bounded")
def _exposure_invariant():
    """Exposure is a risk measure: upstream only ADDS, so total ≥ direct everywhere, and every
    score stays in [0, 1]."""
    total, direct = _exposure()
    min_slack = float((total.to_numpy() - direct.to_numpy()).min())
    in_bounds = bool((total.to_numpy() >= -1e-9).all() and (total.to_numpy() <= 1 + 1e-9).all())
    ok = min_slack >= -1e-9 and in_bounds
    return ok, f"min(total−direct)={min_slack:.3e} (≥0), all in [0,1]={in_bounds}", min_slack, 0.0


@check(SUITE, "carbon_revenue_equals_emissions_under_productivity")
def _revenue_identity():
    """A combined carbon + productivity shock keeps revenue = Σ cc·X: the carbon wedge is a physical
    per-output quantity, NOT scaled by θ (review P1: scaling it broke the emissions/revenue contract
    by 1/θ)."""
    from cge.data.sam import toy_sam
    from cge.engines.cge_static import model as M
    from cge.engines.cge_static.calibrate import calibrate
    from cge.engines.cge_static.solver import solve

    cal = calibrate(toy_sam(), sectors=["BRD", "MIL"], factors=["CAP", "LAB"])
    ns = len(cal.sectors)
    cc = np.array([0.10, 0.02])
    theta = np.array([0.8, 1.0])
    sol = solve(
        lambda z: M.residuals(cal, z, carbon_cost=cc, recycling="lump_sum", productivity=theta),
        M.initial_guess(cal),
        prefer="scipy",
    )
    st = M.derive_state(
        cal,
        sol.x[:ns],
        sol.x[ns:],
        carbon_cost=cc,
        recycling="lump_sum",
        strict=True,
        productivity=theta,
    )
    ratio = float(st.carbon_revenue) / float(cc @ st.X)
    return abs(ratio - 1.0) < 1e-9, f"revenue / (cc·X) = {ratio:.9f} (should be 1)", ratio, 1.0


@check(SUITE, "nature_time_path_flows_per_year")
def _time_path():
    """A NatureStress time path (10%→50%, 2020→2030) produces a per-YEAR shock — agriculture (E=1)
    loses 10% then 50% — not the collapsed scalar (review P1 round 2)."""
    from cge.contracts.shocks import NatureStress
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    ns = NatureStress(service="surface_water", severity=0.1, path={2020: 0.1, 2030: 0.5})
    sc = Scenario(name="p", engine="partial_eq", years=[2020, 2030], shocks=[ns])
    d = run_scenario(sc, data_source="toy").data
    vol = d[
        (d["variable"] == "volume_change")
        & (d["sector"] == "agriculture")
        & (d["scenario"] == "central")
    ]
    by_year = {int(r.year): r.value for r in vol.itertuples()}
    gap = abs(by_year[2030] - (-0.5)) + abs(by_year[2020] - (-0.1))
    return gap < 1e-6, f"agri Δ: 2020={by_year[2020]:.3f} 2030={by_year[2030]:.3f}", gap, 0.0


@check(SUITE, "region_scoped_shock_rejected_on_collapsed_cge")
def _region_scope():
    """The collapsed single-region CGE has no region dimension, so a region-scoped productivity
    shock is ill-posed and must be REJECTED — NOT silently applied economy-wide (review P1 round 3
    2026-08-09). Tested with an ISOLATED region-A shock (no unrelated shocks to mask the region
    universe — the exact masking flaw the previous version of this check had)."""
    from cge.contracts.shocks import ProductivityShock
    from cge.engines.cge_static.engine import _assert_no_region_scoped_productivity

    isolated = [
        ProductivityShock(delta=-0.2, coverage_sectors=["agriculture"], coverage_regions=["A"])
    ]
    try:
        _assert_no_region_scoped_productivity(isolated, "closed")
        rejected = False
    except ValueError:
        rejected = True
    # An economy-wide shock (no region coverage) must still be accepted.
    _assert_no_region_scoped_productivity(
        [ProductivityShock(delta=-0.2, coverage_sectors=["agriculture"])], "closed"
    )
    return rejected, f"isolated region-scoped shock rejected on collapsed CGE = {rejected}", None


@check(SUITE, "incidence_direct_vs_total_differ")
def _incidence():
    """direct vs total incidence genuinely differ: a good with upstream-inherited exposure
    (manufacturing) is shocked less under direct incidence (the CGE default — the network transmits
    the upstream effect, so total would double-count)."""
    from cge.contracts.shocks import NatureStress
    from cge.nature.fixture import encore_fixture, toy_encore_concordance
    from cge.nature.translate import build_nature_shocks
    from cge.validation.toy import toy_economy

    io, _ = toy_economy()
    ns = [NatureStress(service="surface_water", severity=0.5)]
    kw = dict(io=io, encore=encore_fixture(), concordance=toy_encore_concordance())
    t = {
        (s.coverage_regions[0], s.coverage_sectors[0]): s.delta
        for s in build_nature_shocks(ns, incidence="total", **kw)
    }
    d = {
        (s.coverage_regions[0], s.coverage_sectors[0]): s.delta
        for s in build_nature_shocks(ns, incidence="direct", **kw)
    }
    gap = d[("A", "manufacturing")] - t[("A", "manufacturing")]
    return gap > 1e-6, f"mfg delta: direct−total = {gap:.4f} (direct less negative)", gap, 0.0


@check(SUITE, "nature_run_records_provenance")
def _provenance():
    """A NatureStress run through the standard runner records full provenance (ENCORE + concordance
    hashes, materiality scale, rule, incidence, translation version) so it is reconstructible."""
    from cge.contracts.shocks import NatureStress
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    sc = Scenario(
        name="n",
        engine="partial_eq",
        years=[2020],
        shocks=[NatureStress(service="surface_water", severity=0.4)],
    )
    nat = run_scenario(sc, data_source="toy").manifest.assumptions.get("nature", {})
    required = {
        "translation_version",
        "encore_content_hash",
        "concordance_content_hash",
        "materiality_scale",
        "exposure_rule",
        "incidence",
    }
    present = required <= set(nat)
    missing = sorted(required - set(nat))
    return present, f"manifest nature keys present={present}, missing={missing}", None
