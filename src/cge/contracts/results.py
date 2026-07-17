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
        """Raise if ``data`` doesn't match the canonical columns. Called by the runner
        so no malformed result ever reaches the store or GUI."""
        missing = set(RESULT_COLUMNS) - set(self.data.columns)
        if missing:
            raise ValueError(f"ResultSet missing columns: {sorted(missing)}")
        return self

    @classmethod
    def from_records(cls, records: list[dict], manifest: RunManifest) -> ResultSet:
        df = pd.DataFrame.from_records(records, columns=RESULT_COLUMNS)
        return cls(data=df, manifest=manifest).validate_schema()
