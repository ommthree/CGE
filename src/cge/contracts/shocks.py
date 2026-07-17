"""Contract 2 — the typed shock vocabulary.

Scenarios are declarative and composed of typed shocks. This is the key seam
(see ADR-0002): the nature module, the NGFS reader and the damage module all *emit*
shocks in this vocabulary rather than talking to engines directly, so any future
stress type is a new ``Shock`` subclass plus zero engine changes.

Each shock optionally carries a time path. Static engines take a year-slice; dynamic
engines (the recursive wrapper, P7) consume the whole path. The discriminated union
``AnyShock`` lets scenarios round-trip through YAML with the right subclass restored.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Shock(BaseModel):
    """Base for all shocks. ``type`` is the discriminator used for (de)serialisation.

    ``coverage`` narrows where the shock applies; empty means 'everywhere'. A shock
    with a ``path`` (year -> multiplier-or-level, shock-specific) is a time path;
    otherwise ``at(year)`` returns the shock unchanged.
    """

    type: str
    coverage_sectors: list[str] = Field(default_factory=list)
    coverage_regions: list[str] = Field(default_factory=list)
    path: dict[int, float] | None = Field(
        default=None, description="optional year -> scalar time path"
    )

    def applies_to(self, sector: str, region: str) -> bool:
        ok_s = not self.coverage_sectors or sector in self.coverage_sectors
        ok_r = not self.coverage_regions or region in self.coverage_regions
        return ok_s and ok_r


class CarbonPrice(Shock):
    """A carbon price (currency per tonne CO2e)."""

    type: Literal["carbon_price"] = "carbon_price"
    price: float = Field(description="currency per tCO2e")
    gases: list[str] = Field(default_factory=lambda: ["CO2"])
    revenue_recycling: Literal["none", "lump_sum", "labour_tax_cut"] = "none"


class ProductivityShock(Shock):
    """A proportional change in total-factor or sectoral productivity.

    The lingua franca shock: nature degradation and climate damages both land here.
    """

    type: Literal["productivity"] = "productivity"
    delta: float = Field(description="fractional change, e.g. -0.1 for -10%")


class DemandShift(Shock):
    """A proportional shift in final demand for a commodity."""

    type: Literal["demand_shift"] = "demand_shift"
    delta: float


class TradeCost(Shock):
    """A proportional change in trade/transport cost (an iceberg-style cost)."""

    type: Literal["trade_cost"] = "trade_cost"
    delta: float


class NatureStress(Shock):
    """An ecosystem-service degradation, tagged by service.

    Emitted by the nature module (P6). Engines don't interpret ``service`` directly;
    the nature module translates a NatureStress into ``ProductivityShock``s scaled by
    dependency scores. Kept as a distinct type so scenarios read in nature terms and
    so the translation step is explicit and auditable.
    """

    type: Literal["nature_stress"] = "nature_stress"
    service: str = Field(description="e.g. 'pollination', 'surface_water'")
    severity: float = Field(description="fractional degradation of the service, 0..1")


AnyShock = Annotated[
    CarbonPrice | ProductivityShock | DemandShift | TradeCost | NatureStress,
    Field(discriminator="type"),
]
"""Discriminated union of concrete shocks — use this in scenario models so YAML
round-trips restore the correct subclass."""
