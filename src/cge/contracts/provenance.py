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

from pydantic import BaseModel, Field, field_validator

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
    # *Mandatory* dump of the assumptions behind the numbers (printed on every run page).
    # Required with no default so a manifest cannot be constructed — directly or via build —
    # without it; a non-empty check enforces it is actually populated (review).
    assumptions: dict[str, Any]
    created: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @field_validator("assumptions")
    @classmethod
    def _assumptions_nonempty(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("RunManifest.assumptions is mandatory and must be non-empty")
        return v

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
        # Assumptions are the credibility surface (printed on every result); an engine that
        # produces numbers must declare them. Empty here is a programming error.
        if not assumptions:
            raise ValueError(
                f"engine {engine_name!r} produced a result with empty assumptions; the "
                f"assumptions dump is mandatory (it is printed on every result)."
            )
        return cls(
            engine_name=engine_name,
            engine_version=engine_version,
            data_source=data_source,
            scenario_hash=content_hash(scenario),
            assumptions=assumptions,
        )
