"""The runner: the one place that ties data + scenario + engine + result together.

Everything the GUI and CLI do goes through ``run_scenario`` so provenance, schema
validation and shock-support checks happen in exactly one place.
"""

from __future__ import annotations

import cge.engines  # noqa: F401  (import side effect registers engines)
from cge.contracts.engine import registry
from cge.contracts.results import ResultSet
from cge.scenarios.loader import Scenario
from cge.validation import toy_economy


def load_data(source: str) -> dict:
    """Return harmonised data objects keyed by type name.

    ``'toy'`` returns the built-in fixture; any other value is a **build id** looked up in
    the data store (Phase 1). The keys ('IOSystem', 'SatelliteAccount', …) match what
    engines declare in ``meta.required_data``.
    """
    if source == "toy":
        io, sat = toy_economy()
        return {"IOSystem": io, "SatelliteAccount": sat}

    from cge.data.store import default_store

    store = default_store()
    if not store.has(source):
        available = ", ".join(store.build_ids()) or "none"
        raise ValueError(
            f"Unknown data source {source!r}. Use 'toy' or a build id. "
            f"Available builds: {available}."
        )
    return store.load(source)


def run_scenario(scenario: Scenario, *, data_source: str = "toy") -> ResultSet:
    engine = registry.get(scenario.engine)

    unsupported = [s.type for s in scenario.shocks if not engine.meta.supports(s)]
    if unsupported:
        raise ValueError(
            f"Engine {engine.meta.name!r} does not support shock types: {sorted(set(unsupported))}"
        )

    data = load_data(data_source)
    missing = [d for d in engine.meta.required_data if d not in data]
    if missing:
        raise ValueError(f"Data source {data_source!r} is missing required objects: {missing}")

    result = engine.run(data=data, shocks=list(scenario.shocks), years=scenario.years)
    return result.validate_schema()
