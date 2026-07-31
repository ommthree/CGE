"""ENCORE ingestion and the materiality → numeric scale (Phase 6.1).

ENCORE (Exploring Natural Capital Opportunities, Risks and Exposure) rates, for each production
process, how much it **depends on** each ecosystem service (pollination, surface water, climate
regulation, …) and how much it **impacts** each natural-capital asset via impact drivers. Ratings
are ordinal **materiality classes**: Very High / High / Medium / Low / Very Low (VH/H/M/L/VL).

This module turns that ordinal knowledge base into the numeric, provenance-carrying data objects the
exposure engine (6.3) and the nature→shock translation (6.4) consume. Two deliberate design points
the roadmap flags:

- **The materiality → numeric scale is documented and explicit** (``MATERIALITY_SCALE`` below), not
  buried — it drives every downstream number, so it is a named, cited choice a reviewer can change.
- **Ratings are DATA, not code.** An ``EncoreDependencies`` object carries its own provenance
  (source, version, retrieved date) so a run records exactly which ENCORE snapshot produced it. The
  shipped fixture is a small, explicitly-labelled illustrative subset seeded from published
  central-bank mappings — the full ENCORE export drops in via the same contract, no code change.

See ``docs/models/nature-encore.md`` for the equations and sourcing.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import pandas as pd
from pydantic import Field, model_validator

from cge.contracts.data_objects import Provenance, _DataObject

# The five ENCORE materiality classes, most-to-least material.
MaterialityClass = Literal["VH", "H", "M", "L", "VL"]

# Materiality → numeric scale (review-visible, cited). A linear 0.2-step ramp VL..VH, the mapping
# used by DNB "Indebted to nature" (van Toor et al. 2020) and subsequent central-bank studies to
# turn ENCORE's ordinal classes into a [0, 1] dependency weight. Documented here because it drives
# every propagated exposure score; swap it for a convex ramp (e.g. VH=1.0, H=0.5, M=0.25, …) to make
# only the highest classes bite — that is a modelling choice, not a code detail.
MATERIALITY_SCALE: dict[str, float] = {"VH": 1.0, "H": 0.8, "M": 0.6, "L": 0.4, "VL": 0.2}


def materiality_to_score(cls: str) -> float:
    """Map an ENCORE materiality class to its numeric [0, 1] score (``MATERIALITY_SCALE``)."""
    key = str(cls).strip().upper()
    if key not in MATERIALITY_SCALE:
        raise ValueError(
            f"unknown ENCORE materiality class {cls!r}; expected one of {sorted(MATERIALITY_SCALE)}"
        )
    return MATERIALITY_SCALE[key]


class EncoreDependencies(_DataObject):
    """ENCORE dependency ratings as first-class, provenance-carrying data.

    ``ratings`` is a long table with columns ``process``, ``service``, ``materiality`` (a class in
    {VH,H,M,L,VL}). Each (process, service) pair appears at most once. ``services`` is the sorted
    list of distinct ecosystem services; ``processes`` the sorted list of distinct production
    processes. The ``score_matrix`` property applies ``MATERIALITY_SCALE`` to give a numeric
    process × service dependency matrix in [0, 1] — the input to the exposure engine.
    """

    _COLUMNS: ClassVar[tuple[str, ...]] = ("process", "service", "materiality")

    ratings: pd.DataFrame = Field(description="long table: process, service, materiality")
    kind: Literal["dependency", "impact"] = "dependency"

    @model_validator(mode="after")
    def _validate(self) -> EncoreDependencies:
        df = self.ratings
        missing = [c for c in self._COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"EncoreDependencies.ratings is missing columns {missing}")
        # Every materiality value must be a known class (so the numeric scale is total).
        bad = sorted(
            {str(m) for m in df["materiality"] if str(m).strip().upper() not in MATERIALITY_SCALE}
        )
        if bad:
            raise ValueError(
                f"EncoreDependencies.ratings has unknown materiality class(es) {bad}; "
                f"expected {sorted(MATERIALITY_SCALE)}"
            )
        # (process, service) must be unique — a duplicated pair would double-count or silently
        # override, so reject rather than pick one.
        dupes = df.duplicated(subset=["process", "service"])
        if dupes.any():
            example = df[dupes][["process", "service"]].head(3).to_dict("records")
            raise ValueError(
                f"EncoreDependencies.ratings has duplicate (process, service) pairs, e.g. {example}"
            )
        return self

    @property
    def processes(self) -> list[str]:
        return sorted(self.ratings["process"].unique())

    @property
    def services(self) -> list[str]:
        return sorted(self.ratings["service"].unique())

    def score_matrix(self) -> pd.DataFrame:
        """A dense process × service numeric dependency matrix in [0, 1] (missing pairs = 0, i.e.
        no rated dependency). Applies ``MATERIALITY_SCALE`` to each rated class."""
        m = pd.DataFrame(0.0, index=self.processes, columns=self.services)
        for row in self.ratings.itertuples(index=False):
            m.loc[row.process, row.service] = materiality_to_score(row.materiality)
        return m


def load_encore_csv(
    path: str,
    *,
    provenance: Provenance,
    kind: Literal["dependency", "impact"] = "dependency",
) -> EncoreDependencies:
    """Ingest an ENCORE ratings CSV (columns ``process``, ``service``, ``materiality``) into an
    ``EncoreDependencies`` object. This is the real ingestion path the full ENCORE export uses;
    the shipped fixture (``fixture.py``) constructs the same object in-memory for offline tests.
    """
    df = pd.read_csv(path)
    return EncoreDependencies(provenance=provenance, ratings=df, kind=kind)
