"""Translate ``NatureStress`` into ``ProductivityShock``s via exposure scores (Phase 6.4).

This is the seam the roadmap describes: a nature *degradation scenario* — one or more
``NatureStress(service, severity)`` shocks — becomes per-good ``ProductivityShock``s scaled by the
ENCORE exposure scores (from Phase 6.3), which the economic engines then consume through the
standard shock vocabulary. Keeping it a distinct, explicit step means a scenario reads in *nature*
terms (a service degrading), and the nature→economics conversion is auditable rather than hidden in
an engine.

**The translation, stated plainly.** A ``NatureStress`` on service *k* with severity *σ* (fraction
of the service lost, 0..1) reduces good *j*'s output capacity in proportion to how much *j*
**depends on** *k* — its exposure score ``E[j, k] ∈ [0, 1]`` from the exposure engine:

    productivity loss of good j from service k  =  σ_k · E[j, k]

A good fully dependent on a fully-degraded service (E = 1, σ = 1) loses all of its productivity; a
half-exposed good loses half as much. Multiple degraded services **compose multiplicatively** (each
is an independent proportional hit), so good *j*'s surviving productivity is
``Π_k (1 − σ_k · E[j, k])`` and its ``ProductivityShock.delta`` is that product minus one (≤ 0).

The severity → loss mapping is deliberately **linear in the exposure score** — the exposure score
already encodes materiality (via the documented ``MATERIALITY_SCALE``) and supply-chain propagation,
so no second non-linearity is introduced here. A convex severity response (small degradations
tolerated, large ones biting hard) would be a documented modelling choice layered on top; it is not
assumed by default.

Outputs are ``ProductivityShock``s carrying the good's ``region``/``sector`` in their coverage, so
an engine applies each to exactly the right good. See ``docs/models/nature-encore.md`` §7.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from cge.contracts.data_objects import ConcordanceMap, IOSystem
from cge.contracts.shocks import NatureStress, ProductivityShock
from cge.nature.concord import broadcast_to_goods, sector_scores
from cge.nature.encore import EncoreDependencies
from cge.nature.exposure import AggregationRule, compute_exposure

# Per-engine default shock incidence (review P1 2026-08-07). An engine that endogenously propagates
# a sector shock through the supply chain (the CGE, and the IO price engine) must receive DIRECT
# incidence — total exposure would double-count upstream. A partial-equilibrium engine with no such
# transmission receives the reduced-form TOTAL. Callers that don't specify incidence use this.
INCIDENCE_BY_ENGINE: dict[str, str] = {
    "cge_static": "direct",
    "io_price": "direct",
    "partial_eq": "total",
}
DEFAULT_INCIDENCE = "total"  # for an unknown engine: the conservative reduced-form total

# Version of the nature translation itself (exposure propagation + severity→productivity mapping +
# incidence handling). Recorded in a nature run's manifest ALONGSIDE the engine version, because the
# runner-level translation can change results without the engine version moving (review P2
# 2026-08-07). Bump on any change to the translation math or the incidence/aggregation rules.
NATURE_TRANSLATION_VERSION = "0.3.0"

Incidence = Literal["direct", "total"]

# Known ENCORE ecosystem-service OVERLAP (Explanatory note #1, verified in the May-2026 KB): "Water
# supply" is a COMBINED final service in the SEEA-EA categorisation that duplicates the component
# water services below — ENCORE advises users to consider EXCLUDING it to avoid double-counting.
# Stressing the combined service together with its components multiplies overlapping degradation as
# if independent, overstating the hit (review P1 round 3 2026-08-09). Matched case-insensitively so
# it works on ENCORE's real vocabulary ("Water supply") — a scenario using a different service name
# is unaffected.
_WATER_SUPPLY_COMBINED = "water supply"
_WATER_SUPPLY_COMPONENTS = frozenset({"water purification", "water flow regulation"})


def nature_to_productivity(
    stresses: list[NatureStress],
    exposure: pd.DataFrame,
    *,
    min_delta: float = 1e-9,
    years: list[int] | None = None,
    collapse_regions: bool = False,
    allow_water_overlap: bool = False,
) -> list[ProductivityShock]:
    """Translate ``NatureStress`` shocks into per-good ``ProductivityShock``s using an exposure
    matrix (goods × service, in [0, 1], from ``nature.exposure.compute_exposure``).

    Each returned shock targets one good (``coverage_sectors``/``coverage_regions`` set to that
    good's sector/region) with ``delta = Π_k (1 − σ_k · E[j, k]) − 1 ≤ 0``. A ``NatureStress``'s
    own ``coverage`` narrows which goods it can touch (empty = every good). Goods whose net
    productivity loss is below ``min_delta`` get no shock (byte-identical to an unstressed run).
    A stress naming a service not in the exposure matrix is rejected — it would silently do nothing.

    **Time paths (review P1 2026-08-07).** A ``NatureStress`` may carry a ``path`` (year→severity).
    Translation happens once, before the engine's year loop, so it must produce a per-YEAR
    productivity **path** — otherwise the whole path collapses to the scalar ``severity`` and every
    year gets the same shock. When ``years`` is given and any stress has a path, each good's derived
    ``ProductivityShock`` carries ``path={year: delta_at(year)}``; with no path anywhere, a scalar
    ``delta`` is emitted (byte-identical to a path-free run). The severity at a ``year`` is the
    own piecewise-linear path level (flat-extrapolated), so the shock varies exactly as the scenario
    author specified."""
    goods = list(exposure.index)
    services = set(exposure.columns)
    for st in stresses:
        if st.service not in services:
            raise ValueError(
                f"NatureStress service {st.service!r} is not in the exposure matrix "
                f"(services: {sorted(services)}); it would have no effect. Check the service name "
                "or the ENCORE data."
            )

    # The overlap guards below test conflicts at the level of the actual TARGET GOODS, not just the
    # service names (review P2 round 5 2026-08-13): DISJOINT regional stresses (surface_water 20% in
    # A, 60% in B) are the natural way to express heterogeneous regional degradation and must be
    # ALLOWED; only two stresses that can hit the SAME good conflict.
    def _goods_of(st: NatureStress) -> set[str]:
        out = set()
        for g in goods:
            region, sector = (g.split(":", 1) + [""])[:2] if ":" in g else (g, g)
            if st.applies_to(sector, region):
                out.add(g)
        return out

    # Composition is across DISTINCT services (multiplicative). Two stresses on the SAME
    # service that both reach the SAME good would compound as if independent — reject that; disjoint
    # coverage on one service is fine.
    for i, a in enumerate(stresses):
        for b in stresses[i + 1 :]:
            if a.service == b.service and (_goods_of(a) & _goods_of(b)):
                clash = sorted(_goods_of(a) & _goods_of(b))[:3]
                raise ValueError(
                    f"two NatureStress on service {a.service!r} both cover good(s) {clash}: they "
                    "would compound as if independent. Combine them (or a time path), or give them "
                    "disjoint coverage. (Disjoint regional/sector stresses on one service ARE "
                    "allowed.)"
                )
    # ENCORE water-service overlap (review P1 round 3, round 5): "Water supply" is a combined
    # service duplicating its components; a good that gets BOTH double-counts. Reject only where the
    # combined and a component reach the SAME good — "Water supply in A + Water purification in
    # B" is fine (no good gets both).
    if not allow_water_overlap:
        supply = [s for s in stresses if s.service.lower() == _WATER_SUPPLY_COMBINED]
        components = [s for s in stresses if s.service.lower() in _WATER_SUPPLY_COMPONENTS]
        for a in supply:
            for b in components:
                if _goods_of(a) & _goods_of(b):
                    raise ValueError(
                        f"NatureStress overlap: 'Water supply' (a COMBINED ENCORE service) and "
                        f"{b.service!r} both cover a good — this double-counts (ENCORE Explanatory "
                        "note #1 advises excluding the combined service). Drop one, give them "
                        "disjoint coverage, or pass allow_water_overlap=True to keep the overlap."
                    )

    # A path is time-varying if some stress carries one. If the caller didn't pass ``years``, derive
    # them from the union of the stresses' own path years, so a DIRECT call to this helper does not
    # silently collapse a time path to its scalar (review P2 2026-08-09 — previously only the
    # which passes years, was safe). An explicit ``years`` still takes precedence.
    any_path = any(st.path for st in stresses)
    # An EMPTY ``years`` with a path present would silently produce no shocks (review P3 round 5
    # 2026-08-13): treat ``years is None`` OR ``years == []`` the same — derive the years from the
    # union of the stresses' own path years, so a direct helper call is never silently empty.
    if not years and any_path:
        years = sorted({y for st in stresses if st.path for y in st.path})
    has_path = any_path and bool(years)
    path_years = list(years) if years else []

    def _delta_at(g: str, sector: str, region: str, severity_of) -> float:
        """Surviving-productivity − 1 for good g, given a per-stress severity accessor."""
        surviving = 1.0
        for st in stresses:
            if not st.applies_to(sector, region):
                continue
            loss = float(severity_of(st)) * float(exposure.loc[g, st.service])
            surviving *= 1.0 - loss
        return surviving - 1.0  # ≤ 0

    # When collapsing regions, aggregate the FULL per-sector delta (over ALL regions of that sector,
    # INCLUDING zero/sub-threshold regions) — otherwise the equal-region mean is taken only
    # over the shocked regions and a zero-exposure region is silently excluded, biasing the mean up
    # (review P2 round 5 2026-08-13). So for the collapse path we build one economy-wide shock per
    # sector from the mean across the sector's regions here, not from the filtered per-good shocks.
    if collapse_regions:
        return _collapse_full(
            goods, _delta_at, path_years if has_path else None, min_delta, stresses
        )

    out: list[ProductivityShock] = []
    for g in goods:
        region, sector = (g.split(":", 1) + [""])[:2] if ":" in g else (g, g)
        if has_path:
            # Per-year delta path: the severity at each year is the stress's own path level there.
            path = {
                y: _delta_at(g, sector, region, lambda st, y=y: st._path_level_at(y, st.severity))
                for y in path_years
            }
            # Skip a good only if below threshold in EVERY year (some years may bite, others not).
            if all(-dv < min_delta for dv in path.values()):
                continue
            # The scalar ``delta`` fallback is the first year's level (used only if a consumer skips
            # the path); the path is what the engines read per year.
            scalar = path[path_years[0]]
            shock_path = path
        else:
            scalar = _delta_at(g, sector, region, lambda st: st.severity)
            if -scalar < min_delta:
                continue
            shock_path = None
        out.append(
            ProductivityShock(
                delta=scalar,
                coverage_sectors=[sector] if ":" in g else [g],
                coverage_regions=[region] if ":" in g else [],
                path=shock_path,
            )
        )
    return out


def _collapse_full(
    goods: list[str],
    delta_at,
    path_years: list[int] | None,
    min_delta: float,
    stresses: list[NatureStress],
) -> list[ProductivityShock]:
    """Aggregate to ONE economy-wide (region-less) shock per sector for a single-region target
    (review P1 round 3; zeros fix round 5 2026-08-13). Each sector's economy-wide delta is the mean
    over **ALL regions of that sector, including regions with zero/sub-threshold loss** — computed
    from the full delta function, not from the already-filtered per-good shocks, so a zero-exposure
    region is included in the mean (equal region weights, the documented assumption). A sector whose
    mean loss is below ``min_delta`` in every year gets no shock."""
    # Group the economy's goods by their (bare) sector, keeping every region of each sector.
    by_sector: dict[str, list[tuple[str, str]]] = {}  # sector -> [(good, region)]
    for g in goods:
        region, sector = (g.split(":", 1) + [""])[:2] if ":" in g else (g, g)
        by_sector.setdefault(sector, []).append((g, region))

    out: list[ProductivityShock] = []
    for sector, members in by_sector.items():
        if path_years:
            path = {
                y: float(
                    np.mean(
                        [
                            delta_at(
                                g, sector, region, lambda st, y=y: st._path_level_at(y, st.severity)
                            )
                            for g, region in members
                        ]
                    )
                )
                for y in path_years
            }
            if all(-dv < min_delta for dv in path.values()):
                continue
            out.append(
                ProductivityShock(
                    delta=path[path_years[0]],
                    coverage_sectors=[sector] if sector else [],
                    coverage_regions=[],
                    path=path,
                )
            )
        else:
            mean_delta = float(
                np.mean(
                    [delta_at(g, sector, region, lambda st: st.severity) for g, region in members]
                )
            )
            if -mean_delta < min_delta:
                continue
            out.append(
                ProductivityShock(
                    delta=mean_delta,
                    coverage_sectors=[sector] if sector else [],
                    coverage_regions=[],
                )
            )
    return out


def build_nature_shocks(
    stresses: list[NatureStress],
    io: IOSystem,
    encore: EncoreDependencies,
    concordance: ConcordanceMap,
    *,
    rule: AggregationRule = "weighted_mean",
    incidence: Incidence = "total",
    max_link_threshold: float = 0.0,
    years: list[int] | None = None,
    collapse_regions: bool = False,
    allow_water_overlap: bool = False,
) -> list[ProductivityShock]:
    """End-to-end Phase-6 convenience: ENCORE ratings + concordance + IO structure →
    ``NatureStress`` → per-good ``ProductivityShock``s, ready for an engine.

    Runs the whole 6.2→6.3→6.4 chain: map ENCORE process scores onto the economy's sectors
    (``sector_scores``), broadcast to goods and propagate upstream (``compute_exposure`` under
    ``rule``), then translate the stresses (``nature_to_productivity``). ``io.assert_integrity`` is
    re-run first so a mutated payload is rejected at this boundary too.

    **Incidence (review P1 2026-08-07 — avoids double-counting upstream).** The exposure engine
    embeds upstream dependence into each good's TOTAL exposure. Whether the shock should carry that
    upstream part depends on the consuming engine:

    - ``"direct"`` — each good is shocked only for its OWN direct dependency; upstream propagation
      is left to the engine's own supply-chain transmission. **Correct for the CGE / IO price**,
      which already propagates a shocked sector's price and input requirements through the network —
      applying total exposure there would count the upstream channel twice.
    - ``"total"`` (default, back-compat) — each good is shocked for its full direct + upstream
      exposure. Defensible for a **partial-equilibrium** engine with NO endogenous supply
      transmission (Engine 2 ``partial_eq``): the reduced-form total is the only way upstream
      dependence reaches the good there.

    The caller (runner/GUI) selects the mode by engine; ``nature.INCIDENCE_BY_ENGINE`` records the
    per-engine default. ``years`` is passed through to the translation so a ``NatureStress`` time
    path becomes a per-year productivity path (review P1 2026-08-07).

    ``collapse_regions`` (review P1 round 3 2026-08-09): emit ONE economy-wide (region-less) shock
    per sector — each sector's delta the mean across regions — not one shock per good. Set this
    when the target engine is **single-region** (the closed/open CGE, which rejects region-scoped
    shocks): region aggregation then happens explicitly here, not silently in the engine. A
    multi-region engine (multi CGE) leaves it False and keeps per-region shocks."""
    if incidence not in ("direct", "total"):
        raise ValueError(
            f"unknown incidence {incidence!r}; use 'direct' (engine transmits upstream) or "
            "'total' (reduced-form direct+upstream)."
        )
    io.assert_integrity()
    # Validate each stress's COVERAGE against the economy's labels (review P1 round 5 2026-08-13): a
    # misspelled coverage_region/sector matches no good, so translation silently makes ZERO shocks
    # — a "successful" baseline run with no sign the scenario was invalid. Reject unknown labels
    # up front (the service name is checked in nature_to_productivity; coverage was not).
    _validate_stress_coverage(stresses, io)

    ssc = sector_scores(encore, concordance, io.sectors.labels)
    direct_scores = broadcast_to_goods(ssc, list(io.A.columns))
    # Under DIRECT incidence the shock uses only each good's OWN dependency, so the upstream
    # propagation is not needed — skip the (expensive, dense) total-exposure fixed point entirely
    # (review P1 round 5). ``rule``/``max_link_threshold`` then do nothing under direct, so
    # a non-default value there is an inapplicable control and is rejected rather than silently
    # ignored.
    if incidence == "direct":
        if rule != "weighted_mean" or max_link_threshold != 0.0:
            raise ValueError(
                f"exposure rule/threshold (rule={rule!r}, max_link_threshold={max_link_threshold}) "
                "have NO effect under incidence='direct' (each good is shocked for its own direct "
                "dependency; there is no upstream propagation to tune). Use incidence='total' to "
                "apply them, or leave them at their defaults."
            )
        exposure = direct_scores.reindex(index=list(io.A.columns)).fillna(0.0)
    else:
        total, _direct_aligned = compute_exposure(
            io.A, direct_scores, rule=rule, max_link_threshold=max_link_threshold
        )
        exposure = total
    return nature_to_productivity(
        stresses,
        exposure,
        years=years,
        collapse_regions=collapse_regions,
        allow_water_overlap=allow_water_overlap,
    )


def _validate_stress_coverage(stresses: list[NatureStress], io: IOSystem) -> None:
    """Reject a ``NatureStress`` whose ``coverage_regions``/``coverage_sectors`` name a label the
    economy does not have — a typo that would silently match no good and produce a zero-response
    baseline run (review P1 round 5 2026-08-13)."""
    regions = set(io.regions.labels)
    sectors = set(io.sectors.labels)
    for st in stresses:
        bad_r = [r for r in st.coverage_regions if r not in regions]
        bad_s = [s for s in st.coverage_sectors if s not in sectors]
        if bad_r or bad_s:
            problems = []
            if bad_r:
                problems.append(f"region(s) {bad_r} (have: {sorted(regions)})")
            if bad_s:
                problems.append(f"sector(s) {bad_s} (have: {sorted(sectors)})")
            raise ValueError(
                f"NatureStress on {st.service!r} names unknown coverage {'; '.join(problems)}. A "
                "coverage label matching no good would silently produce zero shocks (a baseline "
                "run). Fix the label or drop the coverage to apply the stress economy-wide."
            )
