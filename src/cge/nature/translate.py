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

import pandas as pd

from cge.contracts.data_objects import ConcordanceMap, IOSystem
from cge.contracts.shocks import NatureStress, ProductivityShock
from cge.nature.concord import broadcast_to_goods, sector_scores
from cge.nature.encore import EncoreDependencies
from cge.nature.exposure import AggregationRule, compute_exposure


def nature_to_productivity(
    stresses: list[NatureStress],
    exposure: pd.DataFrame,
    *,
    min_delta: float = 1e-9,
) -> list[ProductivityShock]:
    """Translate ``NatureStress`` shocks into per-good ``ProductivityShock``s using an exposure
    matrix (goods × service, in [0, 1], from ``nature.exposure.compute_exposure``).

    Each returned shock targets one good (``coverage_sectors``/``coverage_regions`` set to that
    good's sector/region) with ``delta = Π_k (1 − σ_k · E[j, k]) − 1 ≤ 0``. A ``NatureStress``'s
    own ``coverage`` narrows which goods it can touch (empty = every good). Goods whose net
    productivity loss is below ``min_delta`` get no shock (byte-identical to an unstressed run).
    A stress naming a service not in the exposure matrix is rejected — it would silently do nothing.
    """
    goods = list(exposure.index)
    services = set(exposure.columns)
    for st in stresses:
        if st.service not in services:
            raise ValueError(
                f"NatureStress service {st.service!r} is not in the exposure matrix "
                f"(services: {sorted(services)}); it would have no effect. Check the service name "
                "or the ENCORE data."
            )

    out: list[ProductivityShock] = []
    for g in goods:
        region, sector = (g.split(":", 1) + [""])[:2] if ":" in g else (g, g)
        surviving = 1.0
        for st in stresses:
            if not st.applies_to(sector, region):
                continue
            loss = float(st.severity) * float(exposure.loc[g, st.service])
            surviving *= 1.0 - loss
        delta = surviving - 1.0  # ≤ 0
        if -delta >= min_delta:
            out.append(
                ProductivityShock(
                    delta=delta,
                    coverage_sectors=[sector] if ":" in g else [g],
                    coverage_regions=[region] if ":" in g else [],
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
) -> list[ProductivityShock]:
    """End-to-end Phase-6 convenience: ENCORE ratings + concordance + IO structure →
    ``NatureStress`` → per-good ``ProductivityShock``s, ready for an engine.

    Runs the whole 6.2→6.3→6.4 chain: map ENCORE process scores onto the economy's sectors
    (``sector_scores``), broadcast to goods and propagate upstream (``compute_exposure`` under
    ``rule``), then translate the stresses (``nature_to_productivity``). ``io.assert_integrity`` is
    re-run first so a mutated payload is rejected at this boundary too."""
    io.assert_integrity()
    ssc = sector_scores(encore, concordance, io.sectors.labels)
    direct = broadcast_to_goods(ssc, list(io.A.columns))
    exposure, _ = compute_exposure(io.A, direct, rule=rule)
    return nature_to_productivity(stresses, exposure)
