"""Declarative scenario files.

A scenario names an engine, lists typed shocks, and gives the years to run. Shocks
deserialise via the discriminated union so YAML restores concrete subclasses. The
whole thing is hashable (via its ``dict``) for the run manifest's scenario hash.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from cge.contracts.shocks import AnyShock


class NatureStatePathwaySpec(BaseModel):
    """A Phase-6b physical ecosystem-service **state pathway** in a scenario file.

    Names a shipped state channel (``channel``) and how its physical state evolves — either an
    explicit ``states`` (year→index) path or a ``degradation_rate`` (index pts/yr) — plus optional
    recovery hysteresis and coverage. The scenario expands each spec into ``NatureStress`` shocks
    (one per ENCORE service the channel drives) at run time, so a scenario expresses a *physical*
    trajectory rather than a bare severity number (Phase 6b.3). See docs/models/nature-state.md."""

    channel: str = Field(description="shipped channel id, e.g. 'water_availability'")
    states: dict[int, float] | None = Field(
        default=None, description="explicit year -> physical state index (baseline = 100)"
    )
    degradation_rate: float | None = Field(
        default=None, description="index points of baseline lost per year (negative = restoration)"
    )
    start_year: int | None = Field(default=None, description="year degradation_rate begins")
    recovery_rate: float | None = Field(
        default=None, description="opt-in recovery hysteresis: max index points recovered per year"
    )
    coverage_sectors: list[str] = Field(default_factory=list)
    coverage_regions: list[str] = Field(default_factory=list)


class Scenario(BaseModel):
    """A named, declarative scenario."""

    name: str
    description: str = ""
    engine: str = Field(description="engine name to run this scenario against")
    years: list[int] = Field(default_factory=lambda: [2020])
    shocks: list[AnyShock] = Field(default_factory=list)
    nature_state: list[NatureStatePathwaySpec] = Field(
        default_factory=list,
        description="Phase-6b physical state pathways, expanded into NatureStress at run time",
    )

    @field_validator("years")
    @classmethod
    def _years_valid(cls, v: list[int]) -> list[int]:
        """Years must be a **non-empty, unique** list — reject at construction rather than let an
        empty list produce an empty/`IndexError` run or duplicates produce duplicate result rows
        that fail schema validation late (review P2). Returned sorted for deterministic output."""
        if not v:
            raise ValueError("Scenario.years must be non-empty")
        if len(set(v)) != len(v):
            dupes = sorted({y for y in v if v.count(y) > 1})
            raise ValueError(f"Scenario.years has duplicates: {dupes}")
        return sorted(v)

    def to_hashable(self) -> dict:
        """Deterministic dict for content hashing (see provenance.content_hash)."""
        return self.model_dump(mode="json")

    def expanded_shocks(self, years: list[int] | None = None) -> list[AnyShock]:
        """The scenario's explicit ``shocks`` PLUS the ``NatureStress`` shocks derived from any
        Phase-6b ``nature_state`` pathways (Phase 6b.3). Called by the runner so a scenario file can
        express a physical degradation pathway instead of a bare severity number. Pure/​side-effect
        free; the physical→severity translation is documented in docs/models/nature-state.md."""
        out: list[AnyShock] = list(self.shocks)
        if not self.nature_state:
            return out
        # Lazy import so the (pymrio-free) scenario model doesn't hard-depend on the nature deps.
        from cge.nature.state import StatePathway, get_channel, state_to_nature_stresses

        run_years = years if years is not None else self.years
        for spec in self.nature_state:
            channel = get_channel(spec.channel)
            pathway = StatePathway(
                channel_id=spec.channel,
                states=spec.states,
                degradation_rate=spec.degradation_rate,
                start_year=spec.start_year,
                recovery_rate=spec.recovery_rate,
            )
            out.extend(
                state_to_nature_stresses(
                    channel,
                    pathway,
                    run_years,
                    coverage_sectors=spec.coverage_sectors,
                    coverage_regions=spec.coverage_regions,
                )
            )
        return out


def load_scenario(path: str | Path) -> Scenario:
    raw = yaml.safe_load(Path(path).read_text())
    return Scenario.model_validate(raw)
