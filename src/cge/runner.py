"""The runner: the one place that ties data + scenario + engine + result together.

Everything the GUI and CLI do goes through ``run_scenario`` so provenance, schema
validation and shock-support checks happen in exactly one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cge.engines  # noqa: F401  (import side effect registers engines)
from cge.contracts.engine import registry
from cge.contracts.results import ResultSet
from cge.scenarios.loader import Scenario
from cge.validation import toy_economy

if TYPE_CHECKING:
    from cge.data.store import DataStore


# Dirty/clean per-€ carbon cost shares for the toy CGE SAMs. The effective cost wedge is
# ``price × share``; these are sized so a realistic price (tens to low hundreds of €/t) gives a
# meaningful-but-well-posed wedge (a share of ~0.2 would make €50 a 1000% wedge and not converge).
_TOY_DIRTY_SHARE = 0.004
_TOY_CLEAN_SHARE = 0.001


def _toy_cge_closed() -> dict:
    from cge.data.sam import toy_sam

    return {
        "SAM": toy_sam(),
        "carbon_cost_share": {"BRD": _TOY_DIRTY_SHARE, "MIL": _TOY_CLEAN_SHARE},
    }


def _toy_cge_open() -> dict:
    from cge.data.sam import toy_open_sam

    return {
        "SAM": toy_open_sam(),
        "carbon_cost_share": {"BRD": _TOY_DIRTY_SHARE, "MIL": _TOY_CLEAN_SHARE},
    }


def _toy_cge_multi() -> dict:
    from cge.data.sam import toy_multi_sam

    # Carbon cost on the North region's dirty sector (so a price shows cross-region leakage).
    return {
        "SAM": toy_multi_sam(),
        "carbon_cost_share": {"N": {"BRD": _TOY_DIRTY_SHARE}, "S": {"BRD": 0.0}},
    }


# Built-in CGE toy SAMs selectable as a data source (closed / open / multi-region). Each returns the
# data dict the CGE engine consumes; the engine dispatches on SAM structure.
_CGE_TOY_SAMS = {
    "toy_cge": _toy_cge_closed,
    "toy_cge_open": _toy_cge_open,
    "toy_cge_multi": _toy_cge_multi,
}


def load_data(source: str, *, store: DataStore | None = None) -> dict:
    """Return harmonised data objects keyed by type name.

    ``'toy'`` returns the built-in fixture; any other value is a **build id** looked up in
    ``store`` (defaults to the process store). The keys ('IOSystem', 'SatelliteAccount', …)
    match what engines declare in ``meta.required_data``.
    """
    if source == "toy":
        io, sat = toy_economy()
        return {"IOSystem": io, "SatelliteAccount": sat}

    # Built-in CGE toy SAMs — the hand-checkable calibration targets, so the CGE (and the GUI) can
    # run the closed / open / multi-region variants without a data build. Each ships a default
    # per-sector carbon_cost_share so a carbon price produces a visible response out of the box
    # (the dirty sector BRD carries the cost). See docs/user-guide.md.
    if source in _CGE_TOY_SAMS:
        return _CGE_TOY_SAMS[source]()

    if store is None:
        from cge.data.store import default_store

        store = default_store()
    if not store.has(source):
        available = ", ".join(store.build_ids()) or "none"
        raise ValueError(
            f"Unknown data source {source!r}. Use 'toy' or a build id. "
            f"Available builds: {available}."
        )
    return store.load(source)


def run_scenario(
    scenario: Scenario,
    *,
    data_source: str = "toy",
    store: DataStore | None = None,
    data_overrides: dict | None = None,
) -> ResultSet:
    engine = registry.get(scenario.engine)

    unsupported = [s.type for s in scenario.shocks if not engine.meta.supports(s)]
    if unsupported:
        raise ValueError(
            f"Engine {engine.meta.name!r} does not support shock types: {sorted(set(unsupported))}"
        )

    data = load_data(data_source, store=store)
    # Optional engine parameters supplied by the caller (e.g. the GUI's CGE elasticity controls:
    # armington_elast / cet_elast / va_elast / open_home_region). Merged into the data dict the
    # engine consumes; unknown keys are simply ignored by engines that don't read them.
    if data_overrides:
        data = {**data, **data_overrides}
    missing = [d for d in engine.meta.required_data if d not in data]
    if missing:
        raise ValueError(f"Data source {data_source!r} is missing required objects: {missing}")

    result = engine.run(data=data, shocks=list(scenario.shocks), years=scenario.years)
    result = result.validate_schema()

    # Macro-aggregate accounting (roadmap Phase 4b, PE tier): roll the per-good price/volume
    # responses up into GVA/GDP/deflator (nominal + real). Engine-agnostic post-step so every
    # price-bearing engine (present and future) gains the aggregates; a no-op for engines that
    # emit no price response or that already provide them natively (the CGE, later).
    io = data.get("IOSystem")
    if io is not None:
        from cge.accounting import augment_with_macro_aggregates

        result = augment_with_macro_aggregates(result, io)
    return result
