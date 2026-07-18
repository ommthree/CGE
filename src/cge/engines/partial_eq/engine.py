"""Engine 2 — partial-equilibrium volume response.

Implements ``docs/models/partial-equilibrium.md``: Δq/q = ε·Δp per good (equation 1),
evaluated across low/central/high demand-elasticity bands so the volume answer carries an
uncertainty envelope. Prices Δp come from Engine 1 (reused, not reimplemented), so this engine
adds only the demand response.

Emitted result variables (long-format ``ResultSet``):
- ``price_change`` — passed through from Engine 1 (band ``central``);
- ``volume_change`` — Δq/q per good, one row per band (``low``/``central``/``high``);
- ``elasticity_used`` — the ε applied (band ``central``), for provenance.
"""

from __future__ import annotations

from cge.contracts.data_objects import ElasticitySet, IOSystem
from cge.contracts.engine import Capability, EngineMeta, registry
from cge.contracts.provenance import RunManifest
from cge.contracts.results import ResultSet
from cge.contracts.shocks import Shock
from cge.data.elasticities import DEFAULT_DEMAND_ELASTICITY, default_demand_set
from cge.engines.io_price.engine import ASSUMPTIONS as IO_ASSUMPTIONS
from cge.engines.io_price.engine import IOPriceEngine

VERSION = "0.1.0"

ASSUMPTIONS = {
    "model": "first-order partial-equilibrium demand response Δq/q = ε·Δp on Engine-1 prices",
    "scope": "own-price demand only; no income effect, no factor markets, no GE feedback",
    "prices_from": "Engine 1 (io_price), taken as exogenous",
    "uncertainty": "low/central/high demand-elasticity bands → volume envelope",
    "elasticity_default": (
        f"goods without an assembled elasticity use {DEFAULT_DEMAND_ELASTICITY} (tagged 'default')"
    ),
    "interpretation": (
        "INDICATIVE volume response, not precise: magnitudes depend on assembled elasticities "
        "(no clean open database). Use for screening/stress; cross-check with the CGE (Phase 5)."
    ),
    "reference": "Armington (1969) for the optional substitution nest; demand theory for (1)",
}


def _elasticity_for(sector: str, eset: ElasticitySet) -> tuple[tuple[float, float, float], str]:
    """Return ((low, central, high), confidence) for a sector, falling back to the tagged
    default. The default's presence is why the engine can always run and always flags it."""
    if sector in eset.values:
        return eset.values[sector], eset.confidence.get(sector, "medium")
    return DEFAULT_DEMAND_ELASTICITY, "default"


class PartialEqEngine:
    """Partial-equilibrium volume response. Satisfies the ``Engine`` protocol."""

    meta = EngineMeta(
        name="partial_eq",
        version=VERSION,
        description="Partial-equilibrium volume response: Δq/q = ε·Δp with an uncertainty band.",
        capabilities=[Capability.PRICES, Capability.VOLUMES],
        supported_shocks=["carbon_price"],
        required_data=["IOSystem", "SatelliteAccount"],
    )

    def run(self, *, data: dict, shocks: list[Shock], years: list[int]) -> ResultSet:
        io: IOSystem = data["IOSystem"]

        # 1. Prices from Engine 1 (single source of truth; reuse, don't reimplement).
        price_result = IOPriceEngine().run(data=data, shocks=shocks, years=years)
        pdf = price_result.data
        prices = pdf[pdf["variable"] == "price_change"]

        # 2. Demand elasticities. An explicit ElasticitySet in ``data`` wins; else the default.
        eset: ElasticitySet = data.get("ElasticitySet") or default_demand_set()
        if eset.kind != "demand":
            raise ValueError(f"partial_eq needs a demand ElasticitySet, got kind={eset.kind!r}")

        records = []
        n_default = 0
        for row in prices.itertuples():
            sector, region, year, dp = row.sector, row.region, row.year, float(row.value)
            (lo, ce, hi), conf = _elasticity_for(sector, eset)
            if conf == "default":
                n_default += 1
            # 3. Δq/q = ε·Δp per band. Pass the price through, emit the elasticity used, and a
            # volume-change row per band.
            records.append(_rec("price_change", sector, region, year, "central", dp))
            records.append(_rec("elasticity_used", sector, region, year, "central", ce))
            for band, eps in (("low", lo), ("central", ce), ("high", hi)):
                records.append(_rec("volume_change", sector, region, year, band, eps * dp))

        manifest = RunManifest.build(
            engine_name=self.meta.name,
            engine_version=self.meta.version,
            data_source=io.provenance.build_id
            or f"{io.provenance.source} {io.provenance.source_version}",
            scenario={"shocks": [s.model_dump(mode="json") for s in shocks], "years": years},
            assumptions={
                **ASSUMPTIONS,
                # Carry Engine 1's assumptions too — its prices are the input, so its caveats apply.
                "price_model": IO_ASSUMPTIONS["model"],
                "price_interpretation": IO_ASSUMPTIONS["interpretation"],
                "elasticity_source": f"{eset.provenance.source} {eset.provenance.source_version}",
                "n_goods_using_default_elasticity": n_default,
                "data_build_id": io.provenance.build_id,
            },
        )
        return ResultSet.from_records(records, manifest)


def _rec(variable: str, sector: str, region: str, year: int, scenario: str, value: float) -> dict:
    return {
        "variable": variable,
        "sector": sector,
        "region": region,
        "year": year,
        "scenario": scenario,
        "value": float(value),
    }


registry.register(PartialEqEngine())
