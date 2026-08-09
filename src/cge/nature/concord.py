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


def sector_nd_mask(
    dep: EncoreDependencies,
    concordance: ConcordanceMap,
    sectors: list[str],
) -> pd.DataFrame:
    """Boolean sectors × service mask, True where a sector's dependency on a service is **entirely
    unknown** — i.e. EVERY ENCORE process the sector maps to is ``ND`` (No Data) on that service
    (review P1 round 3 2026-08-09).

    The numeric ``sector_scores`` scores an all-ND cell as 0, indistinguishable from a genuine
    "no dependency". This mask preserves the distinction so a run can FLAG the gap (e.g. wholesale
    trade's water-purification dependency is unknown, not zero) rather than silently reporting no
    risk — matching ENCORE's own N/A-vs-ND semantics. A cell is *not* masked if any contributing
    process has a real rating (a partial gap still yields a usable, if under-estimated, score)."""
    nd = dep.nd_mask()  # ENCORE process × service (True = ND)
    out = pd.DataFrame(False, index=sectors, columns=list(nd.columns))
    for s in sectors:
        procs = [p for p in concordance.weights.get(s, {}) if p in nd.index]
        if not procs:
            continue
        # A sector/service is unknown iff ALL its contributing processes are ND there.
        out.loc[s] = nd.loc[procs].all(axis=0)
    return out


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
