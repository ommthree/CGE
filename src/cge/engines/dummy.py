"""The Phase 0 dummy engine: no economics, all plumbing.

Its only job is to prove the seams — consume harmonised data + typed shocks, honour the
engine protocol, and emit a schema-valid ``ResultSet`` with a full provenance manifest.
It applies a transparent placeholder rule (a flat carbon 'cost' proportional to emission
intensity) so the end-to-end path produces *some* interpretable number. Engine 1 (P2)
replaces the rule with real Leontief price pass-through.
"""

from __future__ import annotations

from cge.contracts.data_objects import IOSystem, SatelliteAccount
from cge.contracts.engine import Capability, EngineMeta, registry
from cge.contracts.provenance import RunManifest
from cge.contracts.results import ResultSet
from cge.contracts.shocks import CarbonPrice, Shock

VERSION = "0.0.1"


class DummyEngine:
    """Satisfies the ``Engine`` protocol; does placeholder arithmetic only."""

    meta = EngineMeta(
        name="dummy",
        version=VERSION,
        description="Phase 0 plumbing check — placeholder carbon cost, not real economics.",
        capabilities=[Capability.PRICES],
        supported_shocks=["carbon_price"],
        required_data=["IOSystem", "SatelliteAccount"],
    )

    def run(self, *, data: dict, shocks: list[Shock], years: list[int]) -> ResultSet:
        io: IOSystem = data["IOSystem"]
        sat: SatelliteAccount = data["SatelliteAccount"]
        intensities = sat.data.loc["CO2"]  # per sector×region label

        carbon_prices = [s for s in shocks if isinstance(s, CarbonPrice)]
        price = sum(s.price for s in carbon_prices)  # naive: additive over carbon shocks

        labels = list(io.A.columns)
        records = []
        for year in years:
            for label in labels:
                region, sector = label.split(":", 1)
                # Placeholder: direct carbon cost only (NO supply-chain pass-through).
                # This is explicitly the wrong model — it exists to move a number through
                # the pipeline. Engine 1 does (I - A^T)^-1 pass-through instead.
                applies = not carbon_prices or any(
                    s.applies_to(sector, region) for s in carbon_prices
                )
                delta = float(intensities[label]) * price if applies else 0.0
                records.append(
                    {
                        "variable": "price_change_direct_only",
                        "sector": sector,
                        "region": region,
                        "year": year,
                        "scenario": "central",
                        "value": delta,
                    }
                )

        manifest = RunManifest.build(
            engine_name=self.meta.name,
            engine_version=self.meta.version,
            data_source=f"{io.provenance.source} {io.provenance.source_version}",
            scenario={"shocks": [s.model_dump(mode="json") for s in shocks], "years": years},
            assumptions={
                "model": "PLACEHOLDER — direct carbon cost only, no pass-through",
                "carbon_price_total": price,
                "warning": "Phase 0 dummy engine; numbers are not economically meaningful.",
            },
        )
        return ResultSet.from_records(records, manifest)


# Register at import time.
registry.register(DummyEngine())
