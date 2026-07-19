"""The five contracts every module talks through.

These are the load-bearing interfaces of the whole system (see ADR-0002). Modules
depend on these schemas, never on each other's implementations. Everything here is
versioned via ``CONTRACTS_VERSION`` (semver). **Pre-1.0 convention:** while the major is 0,
a breaking change bumps the *minor* (0.1→0.2) and additive changes bump the patch; once we
reach 1.0, breaking changes bump the major. (Standard semver 0.x semantics.)

1. data objects   -> data_objects.py
2. shocks         -> shocks.py
3. engine protocol-> engine.py
4. result schema  -> results.py
5. module slots   -> modules.py

Provenance/config live alongside in provenance.py.
"""

# Semver for the contracts as a set. Data builds, engines and results record the
# contracts version they were produced against so mismatches are detectable.
# 0.2.0: post-review hardening tightened validation (manifests, shocks, results,
# classifications, concordances now reject inputs 0.1.0 accepted) — a breaking change.
# 0.2.1: additive — Provenance.generation (per-save id) and per-input identity/content-hash
# records in run manifests (review: additive changes bump the patch).
# 0.2.2: additive — EnergyPrice shock (carrier output-price change) added to the vocabulary.
# 0.3.0: BREAKING — EnergyPrice semantics changed from a cost wedge to an exogenous price pin (the
# carrier's Δp equals the request exactly and can override its carbon-induced price). Under the
# pre-1.0 policy a breaking change bumps the minor (review P2: the earlier 0.2.3 patch was wrong).
CONTRACTS_VERSION = "0.3.0"

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
    EnergyPrice,
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
    "EnergyPrice",
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
