"""The five contracts every module talks through.

These are the load-bearing interfaces of the whole system (see ADR-0002). Modules
depend on these schemas, never on each other's implementations. Everything here is
versioned via ``CONTRACTS_VERSION``; breaking changes bump the major.

1. data objects   -> data_objects.py
2. shocks         -> shocks.py
3. engine protocol-> engine.py
4. result schema  -> results.py
5. module slots   -> modules.py

Provenance/config live alongside in provenance.py.
"""

# Semver for the contracts as a set. Data builds, engines and results record the
# contracts version they were produced against so mismatches are detectable.
CONTRACTS_VERSION = "0.1.0"

from cge.contracts.data_objects import (  # noqa: E402
    SAM,
    ConcordanceMap,
    ElasticitySet,
    IOSystem,
    SatelliteAccount,
)
from cge.contracts.engine import Capability, Engine, EngineMeta  # noqa: E402
from cge.contracts.modules import ClimateModule, DamageModule  # noqa: E402
from cge.contracts.provenance import RunManifest  # noqa: E402
from cge.contracts.quality import QualityCheck, QualityReport, Severity  # noqa: E402
from cge.contracts.results import ResultSet  # noqa: E402
from cge.contracts.shocks import (  # noqa: E402
    CarbonPrice,
    DemandShift,
    NatureStress,
    ProductivityShock,
    Shock,
    TradeCost,
)

__all__ = [
    "CONTRACTS_VERSION",
    # data objects
    "IOSystem",
    "SAM",
    "SatelliteAccount",
    "ElasticitySet",
    "ConcordanceMap",
    # shocks
    "Shock",
    "CarbonPrice",
    "ProductivityShock",
    "DemandShift",
    "TradeCost",
    "NatureStress",
    # engine
    "Engine",
    "EngineMeta",
    "Capability",
    # results
    "ResultSet",
    # quality
    "QualityReport",
    "QualityCheck",
    "Severity",
    # modules
    "ClimateModule",
    "DamageModule",
    # provenance
    "RunManifest",
]
