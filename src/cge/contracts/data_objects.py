"""Contract 1 — harmonised data objects.

Every data source is an *adapter* that maps raw downloads into these objects
(see ADR-0001, ADR-0003). Engines never see raw source formats, so a second data
source (FIGARO, ICIO, …) is a new adapter, not a refactor.

Phase 0 keeps the numeric payloads as plain pandas structures referenced by these
models, with the models themselves carrying the *metadata* that makes the payload
interpretable: classifications, units, price basis, reference year, provenance.
The heavy numerics (sparse matrices, chunked parquet) arrive in Phase 1; the shapes
here are the stable envelope around them.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Classification(BaseModel):
    """A named list of labels (sectors or regions) — the axis metadata that makes
    an otherwise anonymous matrix interpretable and concordable."""

    name: str = Field(description="e.g. 'EXIOBASE-pxp-200' or 'small-build-45'")
    kind: Literal["sector", "region"]
    labels: list[str]

    @model_validator(mode="after")
    def _labels_unique(self) -> Classification:
        if len(set(self.labels)) != len(self.labels):
            dupes = sorted({x for x in self.labels if self.labels.count(x) > 1})
            raise ValueError(f"Classification {self.name!r} has duplicate labels: {dupes}")
        return self

    def __len__(self) -> int:
        return len(self.labels)


class Provenance(BaseModel):
    """Where a data object came from — attached to every object so results can
    record exactly which build produced them (see provenance.py)."""

    source: str = Field(description="e.g. 'EXIOBASE 3.8.2'")
    source_version: str
    licence: str
    reference_year: int
    retrieved: str = Field(description="ISO date the raw data was retrieved")
    build_id: str | None = Field(
        default=None, description="exact data build this object came from (set on load)"
    )
    aggregation: str | None = Field(
        default=None, description="aggregation name/hash, e.g. 'full' or 'small'"
    )
    generation: str | None = Field(
        default=None,
        description="per-save generation id of the build (distinguishes rewrites of one id)",
    )
    notes: str = ""


class _DataObject(BaseModel):
    """Shared base: pydantic config plus provenance. Allows pandas/numpy payloads
    to live on subclasses without pydantic trying to validate their internals."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    provenance: Provenance


class IOSystem(_DataObject):
    """A (multi-regional) input–output system.

    Envelope only in Phase 0; Phase 1 fills the frames from EXIOBASE. The technical
    coefficients ``A`` and final demand ``final_demand`` are indexed by the sector ×
    region product of ``sectors`` and ``regions``.

    ``final_demand`` rows are producing labels; its columns are either a single aggregate
    column (legacy builds) or **one column per consuming region** (review P1: the open-SAM
    builder needs to know WHO consumes each product to attribute home final demand exactly
    rather than imputing it). ``final_demand_kind`` is an **explicit** discriminator between the
    two shapes — set once by whoever constructs the object (the adapter, the aggregator, or the
    store on load) — rather than inferred from the column set (review P2: inferring by
    ``set(columns) == set(regions.labels)`` silently misclassified an *incomplete* by-region frame
    — e.g. one region's column dropped by a bug — as a legacy aggregate, since the set comparison
    just failed and fell through; the caller then imputed on data that was actually corrupted,
    with no signal anything was wrong). Every consumer that only needs totals keeps using
    ``final_demand.sum(axis=1)``; ``fd_by_region()`` exposes the per-consumer split when present,
    and validates completeness/uniqueness whenever ``final_demand_kind == "by_region"``.
    """

    sectors: Classification
    regions: Classification
    price_basis: Literal["basic", "purchaser"] = "basic"
    currency: str = "EUR"
    unit: str = "MEUR"
    # Payloads (empty-friendly at Phase 0). Index/columns follow sectors×regions.
    A: pd.DataFrame = Field(default_factory=pd.DataFrame, description="technical coefficients")
    final_demand: pd.DataFrame = Field(default_factory=pd.DataFrame)
    value_added: pd.DataFrame = Field(default_factory=pd.DataFrame)
    final_demand_kind: Literal["aggregate", "by_region"] = Field(
        default="aggregate",
        description=(
            "Explicit discriminator for final_demand's column shape: a single aggregate column "
            "(legacy builds) or one column per consuming region. Set by the constructing code, "
            "never inferred from the column set."
        ),
    )
    # Final demand split by consuming region AND institution (Phase 5.1b+, review P1 round 13): the
    # producing-label-indexed table whose columns are "<region>|<institution>" with institution ∈
    # {household, government, investment}. Present when the source (EXIOBASE Y categories) lets us
    # separate household consumption, government consumption and gross capital formation — so the
    # SAM builders can route them to HOH / GOV / SAVINV instead of lumping everything into HOH.
    # Empty when the source carries only an aggregate/by-region column (the older behaviour).
    final_demand_by_institution: pd.DataFrame = Field(default_factory=pd.DataFrame)

    @model_validator(mode="after")
    def _payloads_aligned(self) -> IOSystem:
        """When populated, the ``A`` matrix must be square with **identical, unique** row and
        column labels, and ``final_demand`` must share that index. This makes label alignment a
        contract invariant enforced at **every** entry path (not just the build pipeline), so a
        permuted-axis matrix supplied directly to ``Engine.run()`` is rejected rather than silently
        producing wrong results (review P2). Empty frames (the Phase-0 envelope) are exempt.

        When ``final_demand_kind == "by_region"``, the columns must be COMPLETE and UNIQUE against
        ``regions`` — a dropped or duplicated region column is a real data-integrity error, not a
        silent fallback to the aggregate interpretation (review P2)."""
        a = self.A
        if a.empty:
            return self
        if a.shape[0] != a.shape[1]:
            raise ValueError(f"IOSystem.A must be square; got {a.shape}")
        rows, cols = list(a.index), list(a.columns)
        if len(set(cols)) != len(cols):
            raise ValueError("IOSystem.A has duplicate column labels")
        if rows != cols:
            raise ValueError(
                "IOSystem.A row and column labels must be identical and identically ordered "
                "(a permuted axis silently changes results)"
            )
        if not self.final_demand.empty and list(self.final_demand.index) != cols:
            raise ValueError("IOSystem.final_demand index is not aligned with A's labels")
        if self.final_demand_kind == "by_region" and not self.final_demand.empty:
            fd_cols = list(self.final_demand.columns)
            if len(set(fd_cols)) != len(fd_cols):
                raise ValueError(
                    "IOSystem.final_demand has duplicate consuming-region columns; expected "
                    "one column per region"
                )
            if set(fd_cols) != set(self.regions.labels):
                raise ValueError(
                    f"IOSystem.final_demand_kind='by_region' but its columns {sorted(fd_cols)} "
                    f"do not exactly match the region labels {sorted(self.regions.labels)} — "
                    "a region's final-demand column is missing or an extra column is present"
                )
        # Symmetric enforcement for "aggregate" (review P1, round 10): a genuinely multi-column
        # (by-region-shaped) table mislabelled "aggregate" was previously accepted without
        # complaint — the validator only ever checked the by_region shape — and then silently
        # routed through the imputation path via fd_by_region() returning None. "aggregate" now
        # means exactly one column, enforced the same way "by_region" enforces completeness.
        if self.final_demand_kind == "aggregate" and not self.final_demand.empty:
            fd_cols = list(self.final_demand.columns)
            if len(fd_cols) != 1:
                raise ValueError(
                    f"IOSystem.final_demand_kind='aggregate' requires exactly one column; got "
                    f"{len(fd_cols)} ({sorted(fd_cols)}) — if this is genuinely a per-region "
                    "final-demand table, set final_demand_kind='by_region' instead"
                )
        # Institution split (Phase 5.1b+, hardened review P1 round 14). Beyond column-name syntax
        # and index alignment, the split must be a TRUSTWORTHY decomposition of final demand: finite
        # complete/unique region×institution columns, and its per-region totals must AGREE with the
        # by-region final_demand within a tight tolerance. A large discrepancy is corruption (the
        # review supplied a 0.4 split against 200 total FD that the builder silently amplified 500×
        # into a plausible SAM) — reject, do not silently rescale.
        fbi = self.final_demand_by_institution
        if not fbi.empty:
            if list(fbi.index) != cols:
                raise ValueError(
                    "IOSystem.final_demand_by_institution index is not aligned with A's labels"
                )
            if not np.isfinite(fbi.to_numpy(dtype=float)).all():
                raise ValueError("IOSystem.final_demand_by_institution has non-finite values")
            for c in fbi.columns:
                parts = str(c).split(self.INSTITUTION_SEP)
                if (
                    len(parts) != 2
                    or parts[0] not in self.regions.labels
                    or (parts[1] not in self.INSTITUTIONS)
                ):
                    raise ValueError(
                        f"IOSystem.final_demand_by_institution column {c!r} must be "
                        f"'<region>|<institution>' with region in {self.regions.labels} and "
                        f"institution in {self.INSTITUTIONS}"
                    )
            # Complete AND unique: exactly one column per (region, institution).
            expected = {
                f"{r}{self.INSTITUTION_SEP}{i}"
                for r in self.regions.labels
                for i in self.INSTITUTIONS
            }
            got = list(fbi.columns)
            if len(got) != len(set(got)):
                raise ValueError("IOSystem.final_demand_by_institution has duplicate columns")
            if set(got) != expected:
                raise ValueError(
                    "IOSystem.final_demand_by_institution columns must be exactly one per "
                    f"(region, institution): missing {sorted(expected - set(got))[:4]}, extra "
                    f"{sorted(set(got) - expected)[:4]}"
                )
            # Totals are CONSISTENT with the by-region final demand, PER (region, sector). The
            # institution split covers DOMESTIC final demand (household + government + investment)
            # and EXCLUDES exports, while ``final_demand`` (from the adapter) includes exports — so
            # for EACH sector the split total must be ≥ 0 and NOT EXCEED that sector's final demand
            # (bar float tolerance). Checking per-sector (not just the region-wide total) is what
            # catches the review-P1-round-15 case: a split whose SECTOR distribution is [1, 199]
            # against a final demand of [120, 80] has the SAME region total (200) so a region-total
            # check passes, yet it silently rewrites a sector by ~11,900% downstream. A region-total
            # check is necessary but not sufficient; the per-sector check subsumes it. (The
            # round-14 aggregate "0.4 vs 200" corruption is also caught, per sector.) Only when a
            # by-region final_demand is present to compare against.
            if self.final_demand_kind == "by_region" and not self.final_demand.empty:
                region_set = set(self.regions.labels)
                for a, lbl in enumerate(cols):  # A's labels: '<region>:<sector>'
                    r = str(lbl).split(":", 1)[0]  # region prefix of this producing label
                    if r not in region_set or r not in self.final_demand:
                        continue
                    inst_cell = sum(
                        float(fbi.iloc[a][f"{r}{self.INSTITUTION_SEP}{i}"])
                        for i in self.INSTITUTIONS
                    )
                    fd_cell = float(self.final_demand[r].iloc[a])
                    scale = max(abs(fd_cell), abs(float(self.final_demand[r].mean())), 1.0)
                    if inst_cell < -1e-9 * scale:
                        raise ValueError(
                            f"IOSystem.final_demand_by_institution for {lbl!r} has a negative "
                            f"total {inst_cell:.6g} (an explicit negative-inventory policy is a "
                            "documented follow-up; reject rather than guess)"
                        )
                    if inst_cell > fd_cell + 1e-6 * scale:
                        raise ValueError(
                            f"IOSystem.final_demand_by_institution for {lbl!r} totals "
                            f"{inst_cell:.6g}, EXCEEDING that sector's final_demand {fd_cell:.6g}: "
                            "the domestic institution split cannot exceed final demand SECTOR-BY-"
                            "SECTOR (an aggregate/region match can still hide a large per-sector "
                            "rewrite — review P1 round 15). Reject rather than silently rescale."
                        )
        return self

    @property
    def n(self) -> int:
        return len(self.sectors) * len(self.regions)

    def fd_by_region(self) -> pd.DataFrame | None:
        """The per-consuming-region final-demand split, or ``None`` when this build only carries
        an aggregate column. Reads the explicit ``final_demand_kind`` discriminator (review P2) —
        completeness/uniqueness against ``regions`` is already enforced by the validator above, so
        this never needs to re-derive the shape from the column set."""
        if self.final_demand_kind == "by_region" and not self.final_demand.empty:
            return self.final_demand
        return None

    # The institution split's column separator ("A|household"), chosen not to collide with region
    # or sector names (which use "A:energy").
    INSTITUTION_SEP: ClassVar[str] = "|"
    INSTITUTIONS: ClassVar[tuple[str, ...]] = ("household", "government", "investment")

    def fd_by_institution(self) -> pd.DataFrame | None:
        """The final demand split by consuming region AND institution, or ``None`` if the source
        did not carry the category detail. Columns are ``"<region>|<institution>"`` (see
        ``final_demand_by_institution``); institutions are ``INSTITUTIONS``. Lets the SAM builders
        route household consumption / government consumption / gross capital formation to
        HOH / GOV / SAVINV rather than lumping all final demand into the household (review P1
        round 13)."""
        if self.final_demand_by_institution.empty:
            return None
        return self.final_demand_by_institution


class SatelliteAccount(_DataObject):
    """Environmental extensions (emissions by gas, energy use, …) per sector×region.

    Kept separate from ``IOSystem`` because sources, units and update cadences differ,
    and because nature/ENCORE work adds satellite accounts without touching the IO core.
    """

    name: str = Field(description="e.g. 'GHG' or 'energy'")
    units: dict[str, str] = Field(default_factory=dict, description="row label -> unit")
    data: pd.DataFrame = Field(default_factory=pd.DataFrame, description="stressor × sector×region")


class SAM(_DataObject):
    """Social Accounting Matrix — the CGE calibration target (Phase 5).

    A square, balanced matrix over named accounts (activities, commodities, factors,
    household, government, savings/investment, rest-of-world). Present as a contract
    from Phase 0 so the CGE has a stable target to build toward.
    """

    accounts: list[str]
    matrix: pd.DataFrame = Field(default_factory=pd.DataFrame)

    @model_validator(mode="after")
    def _square_if_populated(self) -> SAM:
        m = self.matrix
        if not m.empty and m.shape[0] != m.shape[1]:
            raise ValueError(f"SAM matrix must be square; got {m.shape}")
        return self


class ElasticitySet(_DataObject):
    """A collection of behavioural elasticities with per-value sourcing.

    Uncertainty is first-class (low/central/high) because volume results are sensitive
    to these (see roadmap P4). Values carry their native classification so they can be
    concorded onto whatever build an engine runs on.
    """

    kind: Literal["demand", "armington", "substitution", "supply"]
    classification: str = Field(description="native classification of the entries")
    # entry key -> (low, central, high)
    values: dict[str, tuple[float, float, float]] = Field(default_factory=dict)
    sources: dict[str, str] = Field(default_factory=dict, description="entry key -> citation")
    confidence: dict[str, Literal["high", "medium", "low", "default"]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _values_valid(self) -> ElasticitySet:
        """Reject economically-invalid or incomplete elasticities (review P2). Every value must
        be finite and band-ordered (low ≤ central ≤ high) with per-value source + confidence;
        **demand** elasticities must additionally be ≤ 0 (a price rise must not raise demand)."""
        import math

        for key, triple in self.values.items():
            if len(triple) != 3 or not all(math.isfinite(v) for v in triple):
                raise ValueError(f"elasticity {key!r} must be 3 finite values, got {triple}")
            lo, ce, hi = triple
            if not (lo <= ce <= hi):
                raise ValueError(f"elasticity {key!r} bands not ordered low≤central≤high: {triple}")
            if self.kind == "demand" and hi > 0:
                raise ValueError(
                    f"demand elasticity {key!r} has a positive value {hi}; demand elasticities "
                    f"must be ≤ 0 (a price rise cannot raise demand)"
                )
            if key not in self.sources or key not in self.confidence:
                raise ValueError(f"elasticity {key!r} is missing source or confidence metadata")
        return self


class ConcordanceMap(_DataObject):
    """A many-to-many, weighted map between two classifications.

    First-class data (not code): reused for aggregation (P1), ENCORE↔EXIOBASE (P6),
    and any second data source (P7). Weights out of each source label must sum to 1.
    """

    from_classification: str
    to_classification: str
    # source label -> {target label: weight}
    weights: dict[str, dict[str, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _weights_valid(self) -> ConcordanceMap:
        import math

        for src, targets in self.weights.items():
            # Weights are shares — each must be finite (NaN would corrupt any downstream sum;
            # review) and non-negative (negatives that happen to sum to 1 are not a valid
            # split), and they must sum to 1.
            nonfinite = {t: w for t, w in targets.items() if not math.isfinite(w)}
            if nonfinite:
                raise ValueError(f"Concordance weights for {src!r} are not finite: {nonfinite}")
            negatives = {t: w for t, w in targets.items() if w < 0}
            if negatives:
                raise ValueError(f"Concordance weights for {src!r} include negatives: {negatives}")
            total = sum(targets.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Concordance weights for {src!r} sum to {total}, expected 1.0")
        return self
