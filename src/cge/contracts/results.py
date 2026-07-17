"""Contract 4 — the result schema.

Every engine emits a ``ResultSet``: a single long-format table (variable × sector ×
region × year) plus provenance. Long format (see ADR-0004) makes comparison across
engines, scenarios and data sources a query rather than a feature, and serialises
cleanly to parquet.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from cge.contracts.provenance import RunManifest

# The canonical long-format columns. 'variable' is e.g. 'price_change',
# 'volume_change', 'temperature'. 'scenario' in {low, central, high} carries
# uncertainty bands as data rather than as separate result objects.
RESULT_COLUMNS = ["variable", "sector", "region", "year", "scenario", "value"]


class ResultSet(BaseModel):
    """Engine output. ``data`` holds RESULT_COLUMNS; ``manifest`` records exactly what
    produced it (data version, engine version, scenario hash, assumption dump)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pd.DataFrame = Field(default_factory=lambda: pd.DataFrame(columns=RESULT_COLUMNS))
    manifest: RunManifest

    def validate_schema(self) -> ResultSet:
        """Raise if ``data`` is malformed. Called by the runner so no bad result reaches the
        store or GUI. Checks: required columns present, no unexpected columns, ``value`` is
        numeric and finite (no NaN/inf), and ``scenario`` uses known band labels."""
        cols = set(self.data.columns)
        missing = set(RESULT_COLUMNS) - cols
        if missing:
            raise ValueError(f"ResultSet missing columns: {sorted(missing)}")
        extra = cols - set(RESULT_COLUMNS)
        if extra:
            raise ValueError(f"ResultSet has unexpected columns: {sorted(extra)}")
        if self.data.empty:
            return self
        values = pd.to_numeric(self.data["value"], errors="coerce")
        if not values.notna().all() or not pd.Series(values).map(lambda v: v == v).all():
            raise ValueError("ResultSet 'value' contains non-numeric or NaN entries")
        import numpy as np

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("ResultSet 'value' contains non-finite (inf) entries")
        bad_bands = set(self.data["scenario"].unique()) - {"low", "central", "high"}
        if bad_bands:
            raise ValueError(f"ResultSet 'scenario' has invalid band labels: {sorted(bad_bands)}")
        return self

    @classmethod
    def from_records(cls, records: list[dict], manifest: RunManifest) -> ResultSet:
        df = pd.DataFrame.from_records(records, columns=RESULT_COLUMNS)
        return cls(data=df, manifest=manifest).validate_schema()
