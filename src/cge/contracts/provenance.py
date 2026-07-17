"""Provenance & config.

A ``RunManifest`` is attached to every ``ResultSet`` so that no result exists without
the information needed to reproduce and trust it: which data build, which engine
version, which scenario (by content hash), and the full assumption dump the engine
declared. This is the single highest-leverage credibility feature of the platform
(see roadmap §5).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from cge.contracts import CONTRACTS_VERSION


def content_hash(obj: Any) -> str:
    """Stable short hash of any JSON-serialisable object (used for scenario hashes)."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


class RunManifest(BaseModel):
    """The reproducibility record for one engine run."""

    contracts_version: str = CONTRACTS_VERSION
    engine_name: str
    engine_version: str
    data_source: str = Field(description="e.g. 'EXIOBASE 3.8.2 / small-build-45'")
    scenario_hash: str
    # Free-form but *mandatory* dump of the assumptions behind the numbers. The GUI
    # prints this on every run page. Engines are expected to populate it richly.
    assumptions: dict[str, Any] = Field(default_factory=dict)
    created: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @classmethod
    def build(
        cls,
        *,
        engine_name: str,
        engine_version: str,
        data_source: str,
        scenario: dict[str, Any],
        assumptions: dict[str, Any],
    ) -> RunManifest:
        return cls(
            engine_name=engine_name,
            engine_version=engine_version,
            data_source=data_source,
            scenario_hash=content_hash(scenario),
            assumptions=assumptions,
        )
