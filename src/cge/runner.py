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


def _toy_cge_open_gov() -> dict:
    from cge.data.sam import toy_open_gov_sam

    # The open SAM plus a government + savings-investment account, so the open CGE has the benchmark
    # stock-flow bridge the recursive-dynamic wrapper (Phase 7.1) steps forward.
    return {
        "SAM": toy_open_gov_sam(),
        "carbon_cost_share": {"BRD": _TOY_DIRTY_SHARE, "MIL": _TOY_CLEAN_SHARE},
    }


def _toy_cge_multi() -> dict:
    from cge.data.sam import toy_multi_sam

    # Carbon cost on the North region's dirty sector (so a price shows cross-region leakage).
    return {
        "SAM": toy_multi_sam(),
        "carbon_cost_share": {"N": {"BRD": _TOY_DIRTY_SHARE}, "S": {"BRD": 0.0}},
    }


def _toy_cge_multi_gov() -> dict:
    from cge.data.sam import toy_multi_gov_sam

    # The multi SAM plus per-region government + savings-investment accounts, so the multi CGE has a
    # per-region benchmark stock-flow bridge for the recursive-dynamic wrapper (Phase 7.1).
    return {
        "SAM": toy_multi_gov_sam(),
        "carbon_cost_share": {"N": {"BRD": _TOY_DIRTY_SHARE}, "S": {"BRD": 0.0}},
    }


def _toy_cge_gov() -> dict:
    from cge.data.sam.toy_5d import toy_gov_sam

    # The closed SAM plus a government + savings-investment account (Phase 5d fiscal/investment
    # closures). The carbon wedge is on the dirty sector, as for the plain closed SAM.
    return {
        "SAM": toy_gov_sam(),
        "carbon_cost_share": {"BRD": _TOY_DIRTY_SHARE, "MIL": _TOY_CLEAN_SHARE},
    }


def _toy_cge_energy() -> dict:
    from cge.data.sam.toy_5d import toy_energy_sam

    # A 3-sector SAM with two energy commodities (DIRTY fossil / CLEAN electricity). Declaring
    # ``energy_sectors`` activates the KL-E-M nest so a fossil carbon price substitutes within the
    # energy bundle. The wedge sits on the fossil commodity.
    return {
        "SAM": toy_energy_sam(),
        "carbon_cost_share": {"DIRTY": _TOY_DIRTY_SHARE, "CLEAN": 0.0, "MFG": 0.0},
        "energy_sectors": ["DIRTY", "CLEAN"],
    }


# Built-in CGE toy SAMs selectable as a data source (closed / open / multi-region, plus the Phase
# 5d government and energy-nest variants). Each returns the data dict the CGE engine consumes; the
# engine dispatches on SAM structure.
_CGE_TOY_SAMS = {
    "toy_cge": _toy_cge_closed,
    "toy_cge_open": _toy_cge_open,
    "toy_cge_open_gov": _toy_cge_open_gov,
    "toy_cge_multi": _toy_cge_multi,
    "toy_cge_multi_gov": _toy_cge_multi_gov,
    "toy_cge_gov": _toy_cge_gov,
    "toy_cge_energy": _toy_cge_energy,
}


def load_data(source: str, *, store: DataStore | None = None) -> dict:
    """Return harmonised data objects keyed by type name.

    ``'toy'`` returns the built-in fixture; any other value is a **build id** looked up in
    ``store`` (defaults to the process store). The keys ('IOSystem', 'SatelliteAccount', …)
    match what engines declare in ``meta.required_data``.
    """
    if source == "toy":
        io, sat = toy_economy()
        # Attach the illustrative ENCORE fixture + concordance so a NatureStress scenario runs
        # through the standard pipeline on the toy source (review P1 2026-08-07). A real build
        # supplies these from the store; without them, a NatureStress run raises a clear error.
        from cge.nature.fixture import encore_fixture, toy_encore_concordance

        return {
            "IOSystem": io,
            "SatelliteAccount": sat,
            "EncoreDependencies": encore_fixture(),
            "ConcordanceMap": toy_encore_concordance(),
        }

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


# Nature scenario controls a caller may set via ``data_overrides`` (review P2 2026-08-07 — the rule,
# incidence and max-link threshold were previously hardcoded and unreachable from YAML/CLI/GUI).
# are extracted BEFORE nature preprocessing (unlike other overrides, which merge afterwards) and
# never reach the engine.
_NATURE_CONTROL_KEYS = (
    "nature_rule",
    "nature_incidence",
    "nature_max_link_threshold",
    "nature_allow_water_overlap",
    "EncoreDependencies",
    "ConcordanceMap",
)


def _preprocess_nature(
    scenario: Scenario, data: dict, engine, overrides: dict
) -> tuple[list, dict | None]:
    """Translate any ``NatureStress`` in the scenario into ``ProductivityShock``s BEFORE the engine
    runs (review P1 2026-08-07 — nature runs through the standard pipeline, not a GUI-only path).

    Returns ``(shocks, nature_stamp)``: ``shocks`` is the scenario's shocks with every
    ``NatureStress`` replaced by its derived per-good ``ProductivityShock``s (other shocks kept),
    and ``nature_stamp`` is the provenance dict to append to the run manifest — or ``None`` when the
    scenario has no NatureStress.

    ``EncoreDependencies``/``ConcordanceMap`` are read from the data source OR injected via
    ``overrides`` (so a run can supply nature data even when the stored build does not carry it —
    review P1 2026-08-07), then **removed** so the engine (which may strictly reject unknown keys)
    never sees them. The exposure rule, incidence and max-link threshold are selectable via
    ``overrides`` (``nature_rule``/``nature_incidence``/``nature_max_link_threshold``); with none
    set, the rule defaults to ``weighted_mean`` and incidence to the engine's default."""
    from cge.contracts.shocks import NatureStress

    nature = [s for s in scenario.shocks if isinstance(s, NatureStress)]
    # Injected/override nature data takes precedence over what the source carries.
    encore = overrides.get("EncoreDependencies", data.pop("EncoreDependencies", None))
    concordance = overrides.get("ConcordanceMap", data.pop("ConcordanceMap", None))
    if not nature:
        return list(scenario.shocks), None

    from cge.nature import DEFAULT_INCIDENCE, INCIDENCE_BY_ENGINE
    from cge.nature.encore import MATERIALITY_SCALE
    from cge.nature.translate import NATURE_TRANSLATION_VERSION, build_nature_shocks

    io = overrides.get("IOSystem", data.get("IOSystem"))
    missing = [
        name
        for name, obj in (
            ("IOSystem", io),
            ("EncoreDependencies", encore),
            ("ConcordanceMap", concordance),
        )
        if obj is None
    ]
    if missing:
        raise ValueError(
            f"a NatureStress scenario needs an IOSystem, an EncoreDependencies object and a "
            f"ConcordanceMap; got neither from the data source nor from data_overrides: {missing}. "
            "Use the 'toy' source (which ships the illustrative fixture), a build carrying ENCORE "
            "data, or inject them via data_overrides (EncoreDependencies + ConcordanceMap)."
        )

    rule = overrides.get("nature_rule", "weighted_mean")
    incidence = overrides.get(
        "nature_incidence", INCIDENCE_BY_ENGINE.get(scenario.engine, DEFAULT_INCIDENCE)
    )
    threshold = float(overrides.get("nature_max_link_threshold", 0.0))
    # Single-region target? The CGE runs single-region unless multi_region is explicitly requested —
    # which arrives via data_overrides, so check BOTH (review P1 round 4 2026-08-10: reading only
    # data.get('multi_region') wrongly collapsed a real multi run). partial_eq keeps a region
    # dimension, so it is never single-region here.
    multi_region = bool(data.get("multi_region") or overrides.get("multi_region"))
    single_region_target = scenario.engine == "cge_static" and not multi_region
    # A region-scoped NatureStress against a single-region model is ILL-POSED: there is no region
    # dimension to target, and silently collapse-averaging it made a region-A stress identical to an
    # economy-wide one END TO END (review P1 round 4 — the earlier engine-only guard was bypassed
    # because the runner stripped the region coverage first). So REJECT it here, at the runner
    # boundary, rather than collapse it. An economy-wide NatureStress (no region coverage) is fine;
    # is emitted as one economy-wide shock per sector.
    if single_region_target:
        scoped = sorted({r for s in nature for r in s.coverage_regions})
        if scoped:
            raise ValueError(
                f"a region-scoped NatureStress (coverage_regions={scoped}) cannot run against the "
                f"single-region {scenario.engine!r} model — it has no region dimension to target. "
                "Use the multi-region CGE (multi_region=True) for region-specific stress, or "
                "drop the region coverage to apply it economy-wide."
            )
    allow_water_overlap = bool(overrides.get("nature_allow_water_overlap", False))
    derived = build_nature_shocks(
        nature,
        io,
        encore,
        concordance,
        rule=rule,
        incidence=incidence,
        max_link_threshold=threshold,
        years=list(scenario.years),  # so a NatureStress time path becomes a per-year shock path
        collapse_regions=single_region_target,  # economy-wide NatureStress → one shock per sector
        allow_water_overlap=allow_water_overlap,
    )
    # Keep any non-nature shocks (e.g. a carbon price alongside the degradation), then append the
    # derived productivity shocks.
    shocks = [s for s in scenario.shocks if not isinstance(s, NatureStress)] + derived

    from cge.contracts.provenance import content_hash

    # ND (No-Data) coverage for the stressed services (review P1 rounds 3–4). sector_scores scores
    # an ND cell as 0, so a mostly-unknown cell looks like a near-zero dependency. Record BOTH the
    # entirely-unknown sectors AND — per sector, for each stressed service — the WEIGHTED ND SHARE
    # (fraction of the sector's concordance weight that is No-Data), so a PARTIALLY unknown cell
    # is visible too, not just the all-ND ones (round-4 P1: 262 partial-ND cells were hidden).
    from cge.nature.concord import sector_nd_share

    stressed_services = sorted({s.service for s in nature})
    econ_sectors = sorted({lab.split(":", 1)[-1] for lab in io.A.columns})
    nd_flags: dict[str, list[str]] = {}
    nd_share: dict[str, dict[str, float]] = {}
    try:
        share = sector_nd_share(encore, concordance, econ_sectors)
        for svc in stressed_services:
            if svc not in share.columns:
                continue
            fully = [sec for sec in econ_sectors if float(share.loc[sec, svc]) >= 1.0 - 1e-9]
            if fully:
                nd_flags[svc] = fully
            # Any sector with a non-trivial unknown share (>0) for this stressed service.
            partial = {
                sec: round(float(share.loc[sec, svc]), 4)
                for sec in econ_sectors
                if float(share.loc[sec, svc]) > 1e-9
            }
            if partial:
                nd_share[svc] = partial
    except (ValueError, KeyError):
        # A concordance that doesn't cover these sectors is already reported elsewhere; don't let a
        # coverage-diagnostic failure break the run.
        nd_flags, nd_share = {}, {}

    stamp = {
        "translation_version": NATURE_TRANSLATION_VERSION,
        "stresses": [
            {
                "service": s.service,
                "severity": s.severity,
                "path": s.path,
                # The ORIGINAL NatureStress coverage (sectors/regions it named), so a run is fully
                # reconstructible from the manifest — not just the derived shocks (review P2).
                "coverage_sectors": list(s.coverage_sectors),
                "coverage_regions": list(s.coverage_regions),
            }
            for s in nature
        ],
        "encore_source": encore.provenance.source,
        "encore_version": encore.provenance.source_version,
        "encore_content_hash": content_hash(encore.ratings.to_dict(orient="records")),
        "concordance_source": concordance.provenance.source,
        # The concordance's source_version carries the MRSUT supply-share version AND any
        # year-fallback disclosure (e.g. "2019 fallback for 2020"); record it so the run manifest is
        # full nature provenance, not just a source label + hash (review P2 round 9 2026-08-15).
        "concordance_version": concordance.provenance.source_version,
        "concordance_content_hash": content_hash(concordance.weights),
        "materiality_scale": dict(MATERIALITY_SCALE),
        "exposure_rule": rule,
        "incidence": incidence,
        "max_link_threshold": threshold,
        "collapse_regions": single_region_target,
        "allow_water_overlap": allow_water_overlap,
        "derived_productivity_shocks": len(derived),
        # Shock coverage: the (region, sector) pairs the derived productivity shocks actually touch,
        # so the manifest fully reconstructs WHICH goods were shocked — not just how many (review P2
        # 2026-08-09). Region-less (collapsed) shocks report an empty region.
        "shock_coverage": sorted(
            f"{r}:{sec}" if r else sec
            for s in derived
            for sec in (s.coverage_sectors or [""])
            for r in (s.coverage_regions or [""])
        ),
        # Data-coverage: per stressed service, sectors whose dependency is entirely ND (unknown).
        # A NON-EMPTY entry means those sectors' zero shock is "no data", NOT "no dependency".
        "nd_unknown_sectors": nd_flags,
        # Weighted ND share: per stressed service, {sector: fraction of concordance weight that is
        # No-Data}. Surfaces PARTIALLY-unknown cells (0<share<1), not just the all-ND ones — so a
        # sector whose dependency is, say, 90% unknown is visible in the manifest.
        "nd_weighted_share": nd_share,
    }
    return shocks, stamp


def run_scenario(
    scenario: Scenario,
    *,
    data_source: str = "toy",
    store: DataStore | None = None,
    data_overrides: dict | None = None,
) -> ResultSet:
    engine = registry.get(scenario.engine)

    # Expand any Phase-6b physical state pathways (scenario.nature_state) into NatureStress shocks,
    # so a scenario file can express a physical degradation trajectory rather than a bare severity
    # number (Phase 6b.3). The runner then treats them exactly like hand-written NatureStress.
    if scenario.nature_state:
        scenario = scenario.model_copy(
            update={"shocks": scenario.expanded_shocks(scenario.years), "nature_state": []}
        )

    # NatureStress is not consumed by engines directly — it is translated to ProductivityShocks here
    # (the auditable step), so support is checked on the TRANSLATED shocks below, not the raw ones.
    from cge.contracts.shocks import NatureStress

    unsupported = [
        s.type
        for s in scenario.shocks
        if not isinstance(s, NatureStress) and not engine.meta.supports(s)
    ]
    if unsupported:
        raise ValueError(
            f"Engine {engine.meta.name!r} does not support shock types: {sorted(set(unsupported))}"
        )

    overrides = dict(data_overrides or {})
    data = load_data(data_source, store=store)
    shocks, nature_stamp = _preprocess_nature(scenario, data, engine, overrides)
    # A NatureStress that produced productivity shocks needs an engine that consumes them.
    if nature_stamp is not None and not engine.meta.supports_type("productivity"):
        raise ValueError(
            f"Engine {engine.meta.name!r} does not consume 'productivity' shocks, so a "
            "NatureStress scenario cannot run against it. Use 'partial_eq' or 'cge_static'."
        )
    # Optional engine parameters supplied by the caller (e.g. the GUI's CGE elasticity controls:
    # armington_elast / cet_elast / va_elast / open_home_region). Merged into the data dict the
    # engine consumes. Engines that don't read a key ignore it; the CGE engine is deliberately
    # STRICT — it rejects unknown/other-variant keys and reserved engine-internal ``_`` keys (so a
    # data_override cannot forge IO-backed state or mislabel provenance — review P1 round 15). The
    # nature-control keys are consumed by _preprocess_nature and stripped here so they never reach
    # (and are never rejected by) the engine.
    engine_overrides = {k: v for k, v in overrides.items() if k not in _NATURE_CONTROL_KEYS}
    if engine_overrides:
        data = {**data, **engine_overrides}
    missing = [d for d in engine.meta.required_data if d not in data]
    if missing:
        raise ValueError(f"Data source {data_source!r} is missing required objects: {missing}")

    result = engine.run(data=data, shocks=shocks, years=scenario.years)
    result = result.validate_schema()

    # Stamp the nature translation's provenance into the manifest (review P1 2026-08-07): the engine
    # only saw the derived ProductivityShocks, so record the ENCORE snapshot, concordance,
    # materiality scale, rule and incidence here — a nature run is reconstructible from it. Also
    # OVERWRITE the scenario_hash with the ORIGINAL scenario's hash (the engine hashed only the
    # derived shocks, so two nature scenarios differing only in a path endpoint would otherwise
    # collide — review P1 2026-08-07).
    if nature_stamp is not None:
        from cge.contracts.provenance import content_hash

        manifest = result.manifest.model_copy(
            update={
                "assumptions": {**result.manifest.assumptions, "nature": nature_stamp},
                "scenario_hash": content_hash(scenario.model_dump(mode="json")),
            }
        )
        result = ResultSet(data=result.data, manifest=manifest).validate_schema()

    # Macro-aggregate accounting (roadmap Phase 4b, PE tier): roll the per-good price/volume
    # responses up into GVA/GDP/deflator (nominal + real). Engine-agnostic post-step so every
    # price-bearing engine (present and future) gains the aggregates; a no-op for engines that
    # emit no price response or that already provide them natively (the CGE, later).
    io = data.get("IOSystem")
    if io is not None:
        from cge.accounting import augment_with_macro_aggregates

        result = augment_with_macro_aggregates(result, io)
    return result
