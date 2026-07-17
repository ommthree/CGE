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

from pydantic import BaseModel, Field, model_validator


class Shock(BaseModel):
    """Base for all shocks. ``type`` is the discriminator used for (de)serialisation.

    ``coverage`` narrows where the shock applies; empty means 'everywhere'. A shock with a
    ``path`` (year -> scalar level, shock-specific) is a time path; ``level_at(year)`` reads
    it (piecewise-linear between given years, flat outside the range), falling back to the
    shock's own scalar when no path is set.
    """

    type: str
    coverage_sectors: list[str] = Field(default_factory=list)
    coverage_regions: list[str] = Field(default_factory=list)
    path: dict[int, float] | None = Field(
        default=None, description="optional year -> scalar level time path"
    )

    def applies_to(self, sector: str, region: str) -> bool:
        ok_s = not self.coverage_sectors or sector in self.coverage_sectors
        ok_r = not self.coverage_regions or region in self.coverage_regions
        return ok_s and ok_r

    def _path_level_at(self, year: int, default: float) -> float:
        """Piecewise-linear interpolation of ``path`` at ``year``; flat-extrapolate ends.
        Returns ``default`` when there is no path."""
        if not self.path:
            return default
        years = sorted(self.path)
        if year <= years[0]:
            return float(self.path[years[0]])
        if year >= years[-1]:
            return float(self.path[years[-1]])
        for lo, hi in zip(years, years[1:], strict=False):
            if lo <= year <= hi:
                frac = (year - lo) / (hi - lo)
                return float(self.path[lo] + frac * (self.path[hi] - self.path[lo]))
        return default  # unreachable given the bracketing above


class CarbonPrice(Shock):
    """A carbon price (currency per tonne CO2e).

    ``gases`` selects which greenhouse gases the price applies to (must exist in the data's
    GHG account). ``path`` (if set) gives the price *level* per year, overriding ``price``
    for those years; ``price`` is the level when no path applies.
    """

    type: Literal["carbon_price"] = "carbon_price"
    price: float = Field(description="currency per tCO2e", ge=0.0)
    gases: list[str] = Field(default_factory=lambda: ["CO2"])
    revenue_recycling: Literal["none", "lump_sum", "labour_tax_cut"] = "none"

    @model_validator(mode="after")
    def _validate(self) -> CarbonPrice:
        import math

        # Price must be finite (ge=0 alone accepts +inf, which fails only later at result
        # validation — review).
        if not math.isfinite(self.price):
            raise ValueError(f"CarbonPrice.price must be finite, got {self.price}")
        # ``gases`` must be a non-empty, unique list (an explicit [] is an error, not CO2).
        if not self.gases:
            raise ValueError("CarbonPrice.gases must be non-empty")
        if len(set(self.gases)) != len(self.gases):
            raise ValueError(f"CarbonPrice.gases has duplicates: {self.gases}")
        # Cannot mix the aggregate CO2e row with component gases (double-count) — reject at
        # construction, matching the engine (review: the doc claimed this but only the engine
        # enforced it).
        if "CO2e" in self.gases and len(self.gases) > 1:
            raise ValueError(
                f"CarbonPrice.gases cannot mix 'CO2e' with component gases: {self.gases}"
            )
        # Path values must be finite and non-negative (a carbon price is ≥ 0; NaN bypasses <0).
        if self.path:
            for yr, v in self.path.items():
                if not math.isfinite(v):
                    raise ValueError(f"CarbonPrice.path[{yr}] is not finite: {v}")
                if v < 0:
                    raise ValueError(
                        f"CarbonPrice.path[{yr}]={v} < 0 (a carbon price is non-negative)"
                    )
        return self

    def price_at(self, year: int) -> float:
        """The carbon price level in ``year`` (reads ``path`` if present, else ``price``)."""
        return self._path_level_at(year, self.price)


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
    severity: float = Field(
        description="fractional degradation of the service, 0..1", ge=0.0, le=1.0
    )


AnyShock = Annotated[
    CarbonPrice | ProductivityShock | DemandShift | TradeCost | NatureStress,
    Field(discriminator="type"),
]
"""Discriminated union of concrete shocks — use this in scenario models so YAML
round-trips restore the correct subclass."""
