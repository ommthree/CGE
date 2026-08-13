"""Derive the **EXIOBASE → ENCORE** concordance from the vendored crosswalk (Phase 6, 2026-08-09).

The real ENCORE dependency ratings are keyed by ISIC production process; the economy's sectors are
EXIOBASE labels. This module builds the ``ConcordanceMap`` bridging them, so ``sector_scores`` can
compute real per-EXIOBASE-sector dependencies. It is **generated, reviewable data**, not opaque:
every weighting rule is explicit and an audit of each mapping decision is produced alongside.

**The bridge.** ENCORE processes and the ``EXIOBASE - NACE - ISIC`` crosswalk share the ISIC Rev.4
code space. For each EXIOBASE sector, the crosswalk lists its ISIC Rev.4 codes; each is resolved to
the ENCORE process that rates it via **ISIC-level rollback** — try the full Class code, then its
Group prefix, then Division, then Section — because ENCORE rates most processes at Group level, not
Class. The 9 finer ENCORE splits (all electricity generation, ``D_35_351 — …``) are matched on the
crosswalk's ISIC-Class *name* so, e.g., "Production of electricity by coal" maps to "Fossil fuels
energy production", not the generic electricity process.

**Weighting (a stated assumption, review 2026-08-09).** When an EXIOBASE sector resolves to several
distinct ENCORE processes, weight is split **equally** across them (renormalised to sum to 1, as
``ConcordanceMap`` requires). Equal weighting is a transparent v1 default, NOT an output-weighted or
otherwise calibrated split — the audit records every multi-process sector for review/override.

**Scope honesty.** This removes the "can't run real ENCORE against a real economy" blocker, but the
equal-weight assumption and ENCORE's own "potential, not calibrated" caveat mean results
stay illustrative-of-method. See ``docs/models/nature-encore.md`` and ``data/encore/NOTICE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from cge.contracts.data_objects import ConcordanceMap, Provenance
from cge.nature.encore import EncoreDependencies

_DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "encore"
_CROSSWALK = (
    _DEFAULT_ROOT / "Crosswalk tables" / "EXIOBASE - NACE Rev. 2 - ISIC Rev. 4 - ISIC Rev. 5.csv"
)
_EXIOBASE_COL = "EXIOBASE"
_ISIC4_CODE_COL = "ISIC Rev. 4 Unique Code"
_ISIC4_CLASS_COL = "ISIC Rev 4. Class"


@dataclass
class ConcordanceAudit:
    """A record of how the concordance was built, for review."""

    n_exiobase_sectors: int
    n_encore_processes_used: int
    multi_process_sectors: dict[str, list[str]] = field(default_factory=dict)  # sector -> processes
    unresolved_sectors: list[str] = field(default_factory=list)
    # ISIC code -> the resolved ENCORE process id(s). A list because a code that ENCORE split into
    # finer named processes (e.g. the electricity code → fossil/nuclear/solar…) resolves to SEVERAL
    # process ids across the crosswalk's rows; a plain str would overwrite all but one (review P2).
    rollback_used: dict[str, list[str]] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.n_exiobase_sectors} EXIOBASE sectors → {self.n_encore_processes_used} ENCORE "
            f"processes; {len(self.multi_process_sectors)} sectors map to >1 (equal-weighted);"
            f" {len(self.unresolved_sectors)} unresolved."
        )


def _build_process_index(encore: EncoreDependencies) -> tuple[dict[str, str], dict[str, str]]:
    """Return two lookups over ENCORE process ids:
    - ``by_code``: bare ISIC code → process id, for codes that are NOT split (unique).
    - ``by_name``: ISIC-class name (lowercased) → process id, for split processes (``code — name``).
    """
    by_code: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for pid in encore.processes:
        if " — " in pid:
            code, name = pid.split(" — ", 1)
            by_name[name.strip().lower()] = pid
        else:
            by_code[pid.strip()] = pid
    return by_code, by_name


def _resolve(
    isic_code: str,
    isic_class_name: str,
    by_code: dict[str, str],
    by_name: dict[str, str],
) -> str | None:
    """Resolve one crosswalk row (ISIC Rev.4 code + class name) to an ENCORE process id.

    First try the class NAME (handles ENCORE's finer splits, e.g. energy generation). Then roll the
    ISIC code up (Class → Group → Division → Section) to the first prefix ENCORE rated.
    """
    name_key = str(isic_class_name).strip().lower()
    if name_key and name_key in by_name:
        return by_name[name_key]
    parts = str(isic_code).strip().split("_")
    for k in range(len(parts), 0, -1):
        cand = "_".join(parts[:k])
        if cand in by_code:
            return cand
    return None


def build_exiobase_encore_concordance(
    encore: EncoreDependencies,
    *,
    crosswalk_path: str | Path | None = None,
) -> tuple[ConcordanceMap, ConcordanceAudit]:
    """Build the EXIOBASE→ENCORE ``ConcordanceMap`` (+ audit) from the vendored crosswalk.

    ``encore`` supplies the target process ids (so map keys match ``EncoreDependencies.processes``
    exactly). Weights are equal across the distinct ENCORE processes an EXIOBASE sector resolves to,
    renormalised to sum to 1. An EXIOBASE sector resolving to no ENCORE process is recorded in the
    audit's ``unresolved_sectors`` and omitted (``sector_scores`` then flags it as an
    uncovered sector rather than silently zeroing it)."""
    path = Path(crosswalk_path) if crosswalk_path else _CROSSWALK
    if not path.exists():
        raise FileNotFoundError(f"EXIOBASE↔ISIC crosswalk not found at {path}")
    cw = pd.read_csv(path, encoding="latin-1")
    cw.columns = [str(c).strip() for c in cw.columns]
    for col in (_EXIOBASE_COL, _ISIC4_CODE_COL, _ISIC4_CLASS_COL):
        if col not in cw.columns:
            raise ValueError(f"crosswalk missing column {col!r}; has {list(cw.columns)}")

    by_code, by_name = _build_process_index(encore)
    audit = ConcordanceAudit(n_exiobase_sectors=0, n_encore_processes_used=0)

    weights: dict[str, dict[str, float]] = {}
    sector_to_procs: dict[str, list[str]] = {}
    # Iterate by explicit column selection (robust to spaces/dots in headers, unlike itertuples).
    for sector_raw, code, cls in zip(
        cw[_EXIOBASE_COL], cw[_ISIC4_CODE_COL], cw[_ISIC4_CLASS_COL], strict=True
    ):
        sector = str(sector_raw).strip()
        pid = _resolve(code, cls, by_code, by_name)
        if pid is None:
            continue
        # Record ISIC-code → resolved-process (the audit trail): note where rollback fired, i.e. the
        # full crosswalk code did NOT equal its resolved ENCORE process id (ENCORE rated a coarser
        # level or a named split). Accumulate a list so a code split into several named processes
        # keeps all of them (not just the last one seen).
        code_str = str(code).strip()
        if code_str != pid:
            resolved = audit.rollback_used.setdefault(code_str, [])
            if pid not in resolved:
                resolved.append(pid)
        procs = sector_to_procs.setdefault(sector, [])
        if pid not in procs:  # distinct processes only; the crosswalk repeats codes across rows
            procs.append(pid)

    all_sectors = sorted(set(cw[_EXIOBASE_COL].astype(str).str.strip()))
    used_processes: set[str] = set()
    for sector in all_sectors:
        procs = sector_to_procs.get(sector, [])
        if not procs:
            audit.unresolved_sectors.append(sector)
            continue
        w = 1.0 / len(procs)  # equal weight (documented assumption), sums to 1
        weights[sector] = {p: w for p in procs}
        used_processes.update(procs)
        if len(procs) > 1:
            audit.multi_process_sectors[sector] = procs

    audit.n_exiobase_sectors = len(all_sectors)
    audit.n_encore_processes_used = len(used_processes)

    prov = Provenance(
        source="EXIOBASE→ENCORE concordance (derived from ENCORE crosswalk, CC BY-SA 4.0)",
        source_version="ENCORE May 2026 crosswalk; equal-weight v1",
        licence="CC BY-SA 4.0 (derived; see data/encore/NOTICE.md)",
        reference_year=2026,
        retrieved="2026-08-09",
        notes=(
            "Generated by cge.nature.concordance_build from the EXIOBASE↔ISIC crosswalk via "
            "ISIC-level rollback. Multi-process sectors are EQUAL-WEIGHTED (a stated v1 choice), "
            "not calibrated; see the audit and docs/models/nature-encore.md."
        ),
    )
    cmap = ConcordanceMap(
        provenance=prov,
        from_classification="EXIOBASE",
        to_classification="ENCORE production process (ISIC)",
        weights=weights,
    )
    return cmap, audit


def aggregate_concordance(
    fine: ConcordanceMap,
    sector_map: dict[str, str],
    *,
    to_classification: str = "aggregated-sectors",
) -> ConcordanceMap:
    """Compose a fine EXIOBASE→ENCORE ``ConcordanceMap`` with a **sector-aggregation map**
    (fine EXIOBASE sector → coarse group) to give a coarse ``group → ENCORE process`` concordance
    (review P1 round 5 2026-08-13). This makes an AGGREGATED build (whose sectors are the coarse
    groups) runnable against ENCORE without a bespoke concordance.

    Each group's weight over ENCORE processes is the **equal-weighted average of its member sectors'
    weight vectors**, renormalised to sum to 1. Equal member weights are the same documented v1
    assumption as the fine concordance (output weights unavailable here). A group with no covered
    member is omitted (``sector_scores`` then flags it, rather than silently zeroing)."""
    # group -> accumulated {process: weight}
    grouped: dict[str, dict[str, float]] = {}
    members: dict[str, int] = {}
    for fine_sector, group in sector_map.items():
        w = fine.weights.get(fine_sector)
        if not w:
            continue
        acc = grouped.setdefault(group, {})
        for proc, weight in w.items():
            acc[proc] = acc.get(proc, 0.0) + weight  # member vectors summed (equal member weight)
        members[group] = members.get(group, 0) + 1

    weights: dict[str, dict[str, float]] = {}
    for group, acc in grouped.items():
        total = sum(acc.values())
        if total <= 0:
            continue
        weights[group] = {p: w / total for p, w in acc.items()}  # renormalise to sum to 1

    prov = Provenance(
        source=f"{fine.provenance.source} → aggregated",
        source_version=f"{fine.provenance.source_version}; aggregation-aware",
        licence=fine.provenance.licence,
        reference_year=fine.provenance.reference_year,
        retrieved=fine.provenance.retrieved,
        notes=(
            "Aggregation-aware concordance: fine EXIOBASE→ENCORE weights averaged (equal member "
            "weight) into the build's coarse sector groups; equal-weighted v1, not calibrated."
        ),
    )
    return ConcordanceMap(
        provenance=prov,
        from_classification=to_classification,
        to_classification=fine.to_classification,
        weights=weights,
    )
