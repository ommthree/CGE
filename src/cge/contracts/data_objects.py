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

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Classification(BaseModel):
    """A named list of labels (sectors or regions) — the axis metadata that makes
    an otherwise anonymous matrix interpretable and concordable."""

    name: str = Field(description="e.g. 'EXIOBASE-pxp-200' or 'small-build-45'")
    kind: Literal["sector", "region"]
    labels: list[str]

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

    @property
    def n(self) -> int:
        return len(self.sectors) * len(self.regions)


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
    def _weights_sum_to_one(self) -> ConcordanceMap:
        for src, targets in self.weights.items():
            total = sum(targets.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Concordance weights for {src!r} sum to {total}, expected 1.0")
        return self
