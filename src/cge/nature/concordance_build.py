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

import re
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


@dataclass
class ProductBridgeEntry:
    """How ONE EXIOBASE product resolved to its producing industry weights, for the audit."""

    product: str
    method: str  # 'supply-share' (observed MRSUT) | 'code-prefix-fallback'
    industry_weights: dict[str, float]  # producing industry -> weight (sums to 1)
    fallback_reason: str = ""  # populated only for code-prefix-fallback


@dataclass
class ProductBridgeAudit:
    """A reviewable record of the pxp product→industry bridge (review P1-methodology round 7).

    Records, per product, the candidate producing industries, how they were weighted (observed
    EXIOBASE MRSUT **supply shares** vs the **code-prefix fallback**), and — for fallbacks — why the
    observed shares were unavailable. This is the load-bearing nature-methodology surface, so it is
    generated, reviewable data, not opaque."""

    entries: dict[str, ProductBridgeEntry] = field(default_factory=dict)
    supply_shares_version: str = ""  # provenance of the observed shares, if any

    @property
    def n_supply_share(self) -> int:
        return sum(1 for e in self.entries.values() if e.method == "supply-share")

    @property
    def n_fallback(self) -> int:
        return sum(1 for e in self.entries.values() if e.method == "code-prefix-fallback")

    def summary(self) -> str:
        return (
            f"{len(self.entries)} products bridged: {self.n_supply_share} by observed MRSUT "
            f"supply share, {self.n_fallback} by code-prefix fallback."
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


def _numeric_nace_key(exio_code: str) -> str:
    """Normalise an EXIOBASE ``ExioCode`` (``p23.20.a`` / ``i23.2``) to its numeric-NACE prefix.

    Drops the leading ``p``/``i`` classification marker, keeps only the leading dotted-numeric run
    (so the alphabetic product/industry sub-suffix ``.a``/``.b`` is discarded), and normalises the
    EXIOBASE trailing-zero notation (``23.20`` == ISIC/NACE ``23.2``) by stripping a single trailing
    zero from any multi-digit segment. This is what lets a pxp *product* code line up with its ixi
    *industry* code when they differ only by that notation."""
    bare = str(exio_code).strip()[1:]  # drop the p/i marker
    m = re.match(r"^(\d+(?:\.\d+)*)", bare)
    num = m.group(1) if m else bare
    segs = [(s.rstrip("0") if len(s) > 1 and s.endswith("0") else s) for s in num.split(".")]
    return ".".join(segs)


def pxp_to_ixi_industries(
    pxp_sectors: pd.DataFrame | None = None,
    ixi_sectors: pd.DataFrame | None = None,
) -> dict[str, list[str]]:
    """Map each EXIOBASE **product** (pxp) label to the **industry** (ixi) label(s) that produce it.

    The ENCORE crosswalk is keyed by industry-style EXIOBASE labels, but the default live build is
    ``system="pxp"`` (200 product labels), so a product→industry bridge is required before a pxp
    build can carry nature (review P1 round 6 2026-08-14). Resolution, per product:

    1. **Exact base-code match** — a product whose ``ExioCode`` base equals an industry's (e.g.
       ``p40.11.a`` ↔ ``i40.11.a``: *Electricity by coal* ↔ *Production of electricity by coal*)
       maps 1:1, preserving the fine generation split.
    2. Else **longest numeric-NACE-prefix rollback** — a product-only split (the 18 refinery
       products ``p23.20.*`` → *Petroleum Refinery* ``i23.2``; the biofuels ``p24.*`` → the
       chemicals industries ``i24.*``) maps to every industry sharing that NACE prefix.

    Passing the pymrio classification frames is optional; they default to the bundled EXIOBASE
    classifications. Returns ``{product_label: [industry_label, …]}`` for all 200 products."""
    # Default each frame INDEPENDENTLY — passing only one custom frame must not silently revert both
    # to the bundled defaults (review P3 round 7 2026-08-14).
    if pxp_sectors is None:
        import pymrio

        pxp_sectors = pymrio.get_classification("exio3_pxp").sectors
    if ixi_sectors is None:
        import pymrio

        ixi_sectors = pymrio.get_classification("exio3_ixi").sectors

    def _base(code: str) -> str:
        return str(code).strip()[1:]

    ixi_by_base: dict[str, str] = {}
    ixi_by_num: dict[str, list[str]] = {}
    for row in ixi_sectors.itertuples():
        name = str(row.ExioName).strip()
        ixi_by_base[_base(row.ExioCode)] = name
        ixi_by_num.setdefault(_numeric_nace_key(row.ExioCode), []).append(name)

    mapping: dict[str, list[str]] = {}
    for row in pxp_sectors.itertuples():
        product = str(row.ExioName).strip()
        base = _base(row.ExioCode)
        if base in ixi_by_base:  # (1) exact code — keep the 1:1 split
            mapping[product] = [ixi_by_base[base]]
            continue
        segs = _numeric_nace_key(row.ExioCode).split(".")  # (2) longest numeric-prefix rollback
        for j in range(len(segs), 0, -1):
            cand = ".".join(segs[:j])
            if cand in ixi_by_num:
                mapping[product] = list(ixi_by_num[cand])
                break
    return mapping


def complete_industry_concordance(
    industry_conc: ConcordanceMap,
    *,
    ixi_sectors: pd.DataFrame | None = None,
) -> tuple[ConcordanceMap, dict[str, list[str]]]:
    """Fill any ixi **industry** the crosswalk omits, via a **NACE-sibling fallback**, so the result
    covers the FULL 163-industry EXIOBASE classification (review P2 round 7 2026-08-14).

    The vendored crosswalk omits exactly one ixi industry — ``Production of electricity nec``, a
    residual generation category sharing NACE 35.11 with the covered generation industries. An
    uncovered industry inherits the equal-weighted mean of the covered industries sharing its
    numeric-NACE key. This is the SINGLE place that fallback lives, so **both** a direct ixi build
    and the product bridge attach nature over the whole classification (previously the fallback
    lived only inside ``bridge_to_products``, so a direct ``system="ixi"`` build failed on it).

    Returns ``(completed_concordance, filled)`` where ``filled`` maps each newly-covered industry to
    the covered NACE siblings it was averaged from (for the audit). An industry with NO covered
    sibling is left uncovered (there is nothing to inherit from). Pass ``ixi_sectors`` to override
    the bundled classification used to compute NACE keys."""
    if ixi_sectors is None:
        import pymrio

        ixi_sectors = pymrio.get_classification("exio3_ixi").sectors
    # numeric-NACE key -> covered industries (those the concordance actually rates), for fallback
    nace_covered: dict[str, list[str]] = {}
    ind_nace: dict[str, str] = {}
    all_industries: list[str] = []
    for row in ixi_sectors.itertuples():
        name = str(row.ExioName).strip()
        key = _numeric_nace_key(row.ExioCode)
        ind_nace[name] = key
        all_industries.append(name)
        if industry_conc.weights.get(name):
            nace_covered.setdefault(key, []).append(name)

    weights = dict(industry_conc.weights)  # start from the crosswalk-covered industries
    filled: dict[str, list[str]] = {}
    for ind in all_industries:
        if industry_conc.weights.get(ind):
            continue  # already covered by the crosswalk
        siblings = nace_covered.get(ind_nace.get(ind, ""), [])
        if not siblings:
            continue  # nothing to inherit from → leave uncovered (caller's coverage gate decides)
        acc: dict[str, float] = {}
        for sib in siblings:  # equal-weighted mean of covered NACE siblings
            for proc, w in industry_conc.weights[sib].items():
                acc[proc] = acc.get(proc, 0.0) + w
        total = sum(acc.values())
        if total > 0:
            weights[ind] = {p: w / total for p, w in acc.items()}
            filled[ind] = list(siblings)

    fill_note = ""
    if filled:
        fill_note = (
            f" NACE-sibling fallback filled {len(filled)} crosswalk-missing industry(ies): "
            f"{dict(sorted(filled.items()))}."
        )
    prov = Provenance(
        source=industry_conc.provenance.source,
        source_version=f"{industry_conc.provenance.source_version}; complete-industry",
        licence=industry_conc.provenance.licence,
        reference_year=industry_conc.provenance.reference_year,
        retrieved=industry_conc.provenance.retrieved,
        notes=(industry_conc.provenance.notes or "") + fill_note,
    )
    completed = ConcordanceMap(
        provenance=prov,
        from_classification=industry_conc.from_classification,
        to_classification=industry_conc.to_classification,
        weights=weights,
    )
    return completed, filled


def bridge_to_products(
    industry_conc: ConcordanceMap,
    product_to_industries: dict[str, list[str]],
    *,
    from_classification: str = "EXIOBASE-pxp",
    ixi_sectors: pd.DataFrame | None = None,
    supply_shares: dict[str, dict[str, float]] | None = None,
    supply_shares_version: str = "",
) -> tuple[ConcordanceMap, list[str], ProductBridgeAudit]:
    """Re-key an **industry**-keyed EXIOBASE→ENCORE concordance onto **product** labels.

    Each product's ENCORE-process weight vector is a weighted average of its producing industries'
    vectors (renormalised to sum to 1). The producing-industry weights come from:

    1. **Observed EXIOBASE MRSUT supply shares** (``supply_shares[product]``) when available — the
       fraction of the product's monetary supply produced by each industry, from the supply-use
       table (review P1-method round 7 2026-08-14). This replaces the code-prefix guess with the
       real product→industry relationship, so, e.g., the biofuels no longer receive byte-identical
       weights. Only industries present in the (completed) concordance contribute; the observed
       shares over those are renormalised.
    2. **Code-prefix fallback** — equal weight across the ``product_to_industries`` candidate set
       (``pxp_to_ixi_industries``) when the product has no observed supply (recycling/treatment
       residuals) or ``supply_shares`` is absent (no MRSUT download). This is the round-6 behaviour.

    The industry concordance is first completed via ``complete_industry_concordance`` (the shared
    NACE-sibling fallback), so the residual ``Production of electricity nec`` is covered here as for
    a direct ixi build. A product whose producing industries are ALL uncovered is omitted and
    returned in ``uncovered_products`` so the caller's complete-coverage gate can act on it. Also
    returns a ``ProductBridgeAudit`` recording each product's method, industry weights, and fallback
    reason. Pass ``ixi_sectors`` to override the bundled classification."""
    completed, _filled = complete_industry_concordance(industry_conc, ixi_sectors=ixi_sectors)
    audit = ProductBridgeAudit(supply_shares_version=supply_shares_version)

    weights: dict[str, dict[str, float]] = {}
    uncovered: list[str] = []
    for product, industries in product_to_industries.items():
        # Choose the producing-industry weights: observed supply shares (over concordance-covered
        # industries) if present, else equal weight across the prefix candidate set.
        obs = (supply_shares or {}).get(product)
        method = "code-prefix-fallback"
        reason = ""
        ind_weights: dict[str, float] = {}
        if obs:
            covered_obs = {i: s for i, s in obs.items() if completed.weights.get(i)}
            if covered_obs:
                tot = sum(covered_obs.values())
                ind_weights = {i: s / tot for i, s in covered_obs.items()}
                method = "supply-share"
            else:
                reason = "observed supply industries are all uncovered by the concordance"
        elif supply_shares is None:
            reason = "no supply-share artifact loaded (MRSUT not available)"
        else:
            reason = "product has no market supply in the MRSUT (recycling/treatment residual)"
        if not ind_weights:  # fallback: equal weight over the covered prefix candidates
            covered = [i for i in industries if completed.weights.get(i)]
            if covered:
                ind_weights = {i: 1.0 / len(covered) for i in covered}

        if not ind_weights:
            uncovered.append(product)
            continue

        acc: dict[str, float] = {}
        for ind, iw in ind_weights.items():
            for proc, w in completed.weights[ind].items():
                acc[proc] = acc.get(proc, 0.0) + iw * w
        total = sum(acc.values())
        weights[product] = {p: w / total for p, w in acc.items()}
        audit.entries[product] = ProductBridgeEntry(
            product=product,
            method=method,
            industry_weights=ind_weights,
            fallback_reason=reason,
        )

    # Embed the supply-share artifact version in the PERSISTED provenance (review P2 round 8
    # 2026-08-14): previously only the discarded audit carried it, so a stored concordance concealed
    # which MRSUT year/version produced its weights (and any year fallback).
    if supply_shares is not None and audit.n_supply_share > 0:
        ver = supply_shares_version or "unversioned MRSUT supply shares"
        src_note = f"observed EXIOBASE MRSUT supply shares ({ver}); code-prefix fallback otherwise"
        version_tag = f"product bridge; supply-share weighting [{ver}]"
    else:
        src_note = "equal-weighted code-prefix fallback (no MRSUT supply shares loaded)"
        version_tag = "product bridge; code-prefix fallback"
    prov = Provenance(
        source=f"{industry_conc.provenance.source} → pxp products",
        source_version=f"{industry_conc.provenance.source_version}; {version_tag}",
        licence=industry_conc.provenance.licence,
        reference_year=industry_conc.provenance.reference_year,
        retrieved=industry_conc.provenance.retrieved,
        notes=(
            "EXIOBASE product→industry bridge: each product's ENCORE weights = supply-share-"
            f"weighted mean of its producing industries', renormalised. Weighting: {src_note}. "
            "ENCORE ratings remain indicators of potential significance, not calibrated."
        ),
    )
    cmap = ConcordanceMap(
        provenance=prov,
        from_classification=from_classification,
        to_classification=industry_conc.to_classification,
        weights=weights,
    )
    return cmap, uncovered, audit


def aggregate_concordance(
    fine: ConcordanceMap,
    sector_map: dict[str, str],
    *,
    to_classification: str = "aggregated-sectors",
    require_complete: bool = False,
) -> ConcordanceMap:
    """Compose a fine EXIOBASE→ENCORE ``ConcordanceMap`` with a **sector-aggregation map**
    (fine EXIOBASE sector → coarse group) to give a coarse ``group → ENCORE process`` concordance
    (review P1 round 5 2026-08-13). This makes an AGGREGATED build (whose sectors are the coarse
    groups) runnable against ENCORE without a bespoke concordance.

    Each group's weight over ENCORE processes is the **equal-weighted average of its member sectors'
    weight vectors**, renormalised to sum to 1. Equal member weights are the same documented v1
    assumption as the fine concordance (output weights unavailable here).

    Omitted-member auditing (review P1 round 6 2026-08-14): a fine member the concordance does NOT
    cover was previously skipped silently, so a group could be built from a covered *subset* and
    renormalised — masking partial coverage. Every omitted member is now recorded on the returned
    map's provenance notes. With ``require_complete=True`` any group that loses a member raises
    ``ValueError`` (the build's complete-coverage gate) rather than quietly renormalising the rest.
    A group with NO covered member is always omitted (``sector_scores`` then flags it)."""
    # group -> accumulated {process: weight}; also track which members were covered vs omitted.
    grouped: dict[str, dict[str, float]] = {}
    group_members: dict[str, list[str]] = {}
    omitted: dict[str, list[str]] = {}
    for fine_sector, group in sector_map.items():
        group_members.setdefault(group, []).append(fine_sector)
        w = fine.weights.get(fine_sector)
        if not w:
            omitted.setdefault(group, []).append(fine_sector)
            continue
        acc = grouped.setdefault(group, {})
        for proc, weight in w.items():
            acc[proc] = acc.get(proc, 0.0) + weight  # member vectors summed (equal member weight)

    if require_complete and omitted:
        detail = "; ".join(f"{g}: {sorted(m)}" for g, m in sorted(omitted.items()))
        raise ValueError(
            "aggregate_concordance: incomplete coverage — the fine concordance does not cover "
            f"every member of {len(omitted)} aggregated group(s): {detail}. Either extend the "
            "concordance or pass require_complete=False to accept a covered-subset average."
        )

    weights: dict[str, dict[str, float]] = {}
    for group, acc in grouped.items():
        total = sum(acc.values())
        if total <= 0:
            continue
        weights[group] = {p: w / total for p, w in acc.items()}  # renormalise to sum to 1

    audit_note = ""
    if omitted:
        n_om = sum(len(m) for m in omitted.values())
        audit_note = (
            f" WARNING: {n_om} fine member(s) across {len(omitted)} group(s) were uncovered and "
            f"excluded from the average: {dict(sorted(omitted.items()))}."
        )
    prov = Provenance(
        source=f"{fine.provenance.source} → aggregated",
        source_version=f"{fine.provenance.source_version}; aggregation-aware",
        licence=fine.provenance.licence,
        reference_year=fine.provenance.reference_year,
        retrieved=fine.provenance.retrieved,
        notes=(
            "Aggregation-aware concordance: fine EXIOBASE→ENCORE weights averaged (equal member "
            "weight) into the build's coarse sector groups; equal-weighted v1, not calibrated."
            + audit_note
        ),
    )
    return ConcordanceMap(
        provenance=prov,
        from_classification=to_classification,
        to_classification=fine.to_classification,
        weights=weights,
    )
