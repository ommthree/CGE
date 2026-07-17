"""Task 1.5 — metadata registry.

Central definitions of classifications, units, price basis and reference-year handling,
plus a ``BuildMeta`` that identifies a data build. Keeping these in one place means the
adapter, aggregation and quality modules agree on names and units without importing each
other, and results can record exactly which build produced them.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# EXIOBASE 3 conventions (see [Stadler2018]). Product-by-product (pxp) tables recommended
# in the roadmap; classification sizes recorded for sanity checks.
EXIOBASE = {
    "n_products": 200,  # product-by-product
    "n_regions": 49,  # 44 countries + 5 rest-of-world regions
    "price_basis": "basic",
    "currency": "EUR",
    "monetary_unit": "MEUR",
}

# GWP-100 (AR5) factors live in cge.constants (shared with engines); re-exported here for
# the adapter that lifts gases into the GHG SatelliteAccount.
from cge.constants import GWP100_AR5  # noqa: E402, F401


class BuildMeta(BaseModel):
    """Identifies a data build and carries the metadata every downstream object needs."""

    build_id: str = Field(description="e.g. 'exiobase-3.8.2-2019-pxp-full' or '...-small45'")
    source: str
    source_version: str
    reference_year: int
    licence: str
    price_basis: str = "basic"
    currency: str = "EUR"
    monetary_unit: str = "MEUR"
    aggregation: str = Field(default="full", description="'full' or a named aggregation")
    retrieved: str = Field(description="ISO date the raw data was retrieved/derived")
    notes: str = ""

    def derived(self, *, build_id: str, aggregation: str, notes: str = "") -> BuildMeta:
        """Return a copy describing a build derived from this one (e.g. an aggregation)."""
        return self.model_copy(
            update={"build_id": build_id, "aggregation": aggregation, "notes": notes}
        )
