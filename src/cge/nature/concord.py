"""ENCORE ↔ economy concordance (Phase 6.2).

ENCORE rates *production processes* (its own taxonomy); the economic model works in EXIOBASE-shaped
*sectors*, labelled ``<region>:<sector>``. This module maps an ENCORE dependency score matrix
(process × service) onto the economy's goods (``region:sector`` × service) using the Phase-1
``ConcordanceMap`` framework, so the mapping is **reviewable data with a cited source**, not a code
constant (the roadmap's single biggest credibility surface — every weight a documented judgement).

A concordance here maps each **economy sector** to the ENCORE process(es) that represent it, with
weights summing to 1 (a sector blended from several processes is a weighted average of its scores).
Dependency is a per-unit-of-output *intensity*, so it is broadcast **identically across regions** (a
region-specific ecosystem-service dependency would need region-specific ENCORE data — a documented
follow-up); the concordance is over the sector axis only.

See ``docs/models/nature-encore.md`` for the sourcing of the shipped fixture concordance.
"""

from __future__ import annotations

import pandas as pd

from cge.contracts.data_objects import ConcordanceMap
from cge.nature.encore import EncoreDependencies


def sector_scores(
    dep: EncoreDependencies,
    concordance: ConcordanceMap,
    sectors: list[str],
) -> pd.DataFrame:
    """Map ENCORE process scores onto the economy's SECTORS via ``concordance``.

    Returns a sectors × service score matrix in [0, 1]. ``concordance`` maps each economy sector
    (source) to ENCORE process(es) (target) with weights summing to 1. Every sector must be covered
    (an unmapped sector would silently get zero dependency — reject so the gap is explicit).

    Only a ``kind="dependency"`` object is accepted (review P1 2026-08-07). ENCORE distinguishes
    **dependencies** (how much a process relies on an ecosystem service — the channel that becomes a
    productivity loss when the service degrades) from **impacts/pressures** (how much a process
    *degrades* nature). Feeding an ``impact`` object here would invert the causality — treating the
    economy's pressure ON nature as nature's effect on the economy — so it is rejected.
    """
    if getattr(dep, "kind", "dependency") != "dependency":
        raise ValueError(
            f"sector_scores requires a dependency-kind ENCORE object, got kind={dep.kind!r}. "
            "Impact/pressure ratings measure the economy's effect ON nature, not a dependency that "
            "becomes a productivity loss — converting them into productivity shocks inverts the "
            "causality. Use a dependency object (or an impact→pressure channel, a documented "
            "follow-up)."
        )
    scores = dep.score_matrix()  # process × service
    missing_sectors = [s for s in sectors if s not in concordance.weights]
    if missing_sectors:
        raise ValueError(
            f"ENCORE concordance does not cover economy sector(s) {missing_sectors}; every sector "
            "must map to at least one ENCORE process (an unmapped sector would get silent-zero "
            "dependency). Add the mapping (it is reviewable data) or drop the sector."
        )
    out = pd.DataFrame(0.0, index=sectors, columns=list(scores.columns))
    for s in sectors:
        for process, w in concordance.weights[s].items():
            if process not in scores.index:
                raise ValueError(
                    f"concordance maps sector {s!r} to ENCORE process {process!r}, which is not in "
                    f"the dependency data (processes: {list(scores.index)})"
                )
            out.loc[s] += w * scores.loc[process]
    return out


def sector_nd_share(
    dep: EncoreDependencies,
    concordance: ConcordanceMap,
    sectors: list[str],
) -> pd.DataFrame:
    """Sectors × service matrix of the **weighted ND share**: the fraction of a sector's concordance
    weight whose contributing ENCORE process is ``ND`` (No Data) on that service (review P1 round 4
    2026-08-10).

    ``sector_scores`` averages process scores by concordance weight and scores an ND cell as 0, so a
    cell where 90% of the weight is unknown looks almost the same as a genuine near-zero dependency.
    This returns that hidden uncertainty as a number in [0, 1]: 0 = fully rated, 1 = fully unknown,
    0.9 = 90% of the sector's weight for that service is No-Data. A run can then surface a partially
    unknown cell (not only the all-ND cells the earlier boolean mask caught), matching ENCORE's
    N/A-vs-ND semantics. Weights are the concordance's (which sum to 1 per sector).

    A concordance process that is NOT in the ENCORE data is REJECTED (review P3 round 5 2026-08-13):
    silently dropping it and renormalising over the remainder would understate the unknown share and
    mislead a direct caller. (The standard runner is protected because ``sector_scores`` validates
    process coverage first, but this helper must not be silently wrong on its own.)"""
    nd = dep.nd_mask()  # ENCORE process × service (True = ND)
    out = pd.DataFrame(0.0, index=sectors, columns=list(nd.columns))
    for s in sectors:
        wmap = concordance.weights.get(s, {})
        unknown_procs = [p for p in wmap if p not in nd.index]
        if unknown_procs:
            raise ValueError(
                f"sector {s!r} maps to ENCORE process(es) {unknown_procs} not in the dependency "
                "data; cannot compute the ND share without them (dropping them understates the "
                "unknown fraction). Fix the concordance or the ENCORE object."
            )
        total_w = sum(wmap.values())
        if total_w <= 0:
            continue
        # Weighted fraction of the sector's mapped weight that is ND on each service.
        share = sum(w * nd.loc[p].astype(float) for p, w in wmap.items()) / total_w
        out.loc[s] = share
    return out


def sector_nd_mask(
    dep: EncoreDependencies,
    concordance: ConcordanceMap,
    sectors: list[str],
    *,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """Boolean sectors × service mask, True where the **weighted ND share** (``sector_nd_share``) is
    at or above ``threshold``. The default ``threshold=1.0`` flags only ENTIRELY-unknown cells (the
    original all-ND behaviour, back-compatible); a lower threshold (e.g. 0.5) flags cells that are
    mostly No-Data. See ``sector_nd_share`` for the underlying fraction."""
    import math

    if not (isinstance(threshold, int | float) and math.isfinite(threshold)) or not (
        0.0 <= threshold <= 1.0
    ):
        # A NaN threshold would flag nothing, a negative one flag everything — either silently wrong
        # (review P3 round 5 2026-08-13). The share is in [0, 1], so a threshold must be too.
        raise ValueError(
            f"sector_nd_mask threshold must be a finite value in [0, 1]; got {threshold!r}"
        )
    share = sector_nd_share(dep, concordance, sectors)
    return share >= threshold - 1e-12


def broadcast_to_goods(sector_score: pd.DataFrame, goods: list[str]) -> pd.DataFrame:
    """Broadcast a sectors × service score matrix onto the economy's GOODS (``region:sector``),
    giving every region the same per-sector dependency (dependency is a per-unit intensity). Goods
    whose sector is absent from ``sector_score`` get 0 (no rated dependency)."""
    services = list(sector_score.columns)
    out = pd.DataFrame(0.0, index=goods, columns=services)
    for g in goods:
        sector = g.split(":", 1)[1] if ":" in g else g
        if sector in sector_score.index:
            out.loc[g] = sector_score.loc[sector]
    return out
