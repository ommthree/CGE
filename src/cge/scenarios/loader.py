"""Declarative scenario files.

A scenario names an engine, lists typed shocks, and gives the years to run. Shocks
deserialise via the discriminated union so YAML restores concrete subclasses. The
whole thing is hashable (via its ``dict``) for the run manifest's scenario hash.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from cge.contracts.shocks import AnyShock


class Scenario(BaseModel):
    """A named, declarative scenario."""

    name: str
    description: str = ""
    engine: str = Field(description="engine name to run this scenario against")
    years: list[int] = Field(default_factory=lambda: [2020])
    shocks: list[AnyShock] = Field(default_factory=list)

    def to_hashable(self) -> dict:
        """Deterministic dict for content hashing (see provenance.content_hash)."""
        return self.model_dump(mode="json")


def load_scenario(path: str | Path) -> Scenario:
    raw = yaml.safe_load(Path(path).read_text())
    return Scenario.model_validate(raw)
