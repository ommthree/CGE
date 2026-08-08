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
NATURE_TRANSLATION_VERSION = "0.2.0"

Incidence = Literal["direct", "total"]


def nature_to_productivity(
    stresses: list[NatureStress],
    exposure: pd.DataFrame,
    *,
    min_delta: float = 1e-9,
    years: list[int] | None = None,
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
    # Composition is across DISTINCT services (multiplicative, independent). Two stresses on the
    # SAME service would compound as if independent — almost certainly a scenario mistake, and not
    # what the model documents — so reject a duplicated service (review P2 2026-08-07).
    seen = [st.service for st in stresses]
    dupes = sorted({s for s in seen if seen.count(s) > 1})
    if dupes:
        raise ValueError(
            f"duplicate NatureStress service(s) {dupes}: the model composes across DISTINCT "
            "services; two stresses on one service would compound as if independent. Combine them "
            "into a single stress (or a time path) instead."
        )

    # A path is time-varying only if some stress carries one AND we were told which years to build.
    has_path = years is not None and any(st.path for st in stresses)
    path_years = list(years) if years is not None else []

    def _delta_at(g: str, sector: str, region: str, severity_of) -> float:
        """Surviving-productivity − 1 for good g, given a per-stress severity accessor."""
        surviving = 1.0
        for st in stresses:
            if not st.applies_to(sector, region):
                continue
            loss = float(severity_of(st)) * float(exposure.loc[g, st.service])
            surviving *= 1.0 - loss
        return surviving - 1.0  # ≤ 0

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
    path becomes a per-year productivity path (review P1 2026-08-07)."""
    if incidence not in ("direct", "total"):
        raise ValueError(
            f"unknown incidence {incidence!r}; use 'direct' (engine transmits upstream) or "
            "'total' (reduced-form direct+upstream)."
        )
    io.assert_integrity()
    ssc = sector_scores(encore, concordance, io.sectors.labels)
    direct_scores = broadcast_to_goods(ssc, list(io.A.columns))
    total, direct_aligned = compute_exposure(
        io.A, direct_scores, rule=rule, max_link_threshold=max_link_threshold
    )
    exposure = direct_aligned if incidence == "direct" else total
    return nature_to_productivity(stresses, exposure, years=years)
