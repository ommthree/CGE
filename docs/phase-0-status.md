# Phase 0 — status

Phase 0 ("Foundations & contracts") from `roadmap.md` is complete. This records what was
built against each task and the definition of done.

## Tasks (roadmap §Phase 0)

| # | Task | Status | Where |
|---|---|---|---|
| 0.1 | Repo, env, pyproject, ruff/pre-commit, pytest, CI | ✅ | `pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml` |
| 0.2 | Package skeleton per layout | ✅ | `src/cge/**` |
| 0.3 | Contracts v0 (5 contracts) | ✅ | `src/cge/contracts/` |
| 0.4 | Provenance & config (run manifest, YAML loader) | ✅ | `contracts/provenance.py`, `scenarios/loader.py` |
| 0.5 | Toy fixtures with known structure | ✅ | `src/cge/validation/toy.py` |
| 0.6 | ADR template + first ADRs | ✅ | `docs/adr/` |
| 0.7 | Documentation standard + model-doc template + first worked example | ✅ | `docs/documentation-standard.md`, `docs/models/`, `docs/references.md` |

## Definition of done

> `pip install -e .` works; a dummy engine registers itself, runs end-to-end on the toy
> economy via a YAML scenario, and emits a schema-valid `ResultSet` with provenance; CI green.

All satisfied:
- Package installs editable (`pip install -e ".[dev]"`).
- `cge engines` lists the self-registered `dummy` engine.
- `cge run --scenario examples/carbon_price_toy.yaml` runs end-to-end and prints a
  schema-valid `ResultSet` with a provenance manifest (scenario hash + assumption dump).
- `pytest` — 10 tests pass (contract invariants + end-to-end + a toy sanity check).
- `ruff check` and `ruff format --check` clean.

## The five contracts (what to build engines against)

| Contract | Module | Key types |
|---|---|---|
| Data objects | `contracts/data_objects.py` | `IOSystem`, `SAM`, `SatelliteAccount`, `ElasticitySet`, `ConcordanceMap` |
| Shock vocabulary | `contracts/shocks.py` | `Shock` + `CarbonPrice`, `ProductivityShock`, `DemandShift`, `TradeCost`, `NatureStress`; `AnyShock` union |
| Engine protocol | `contracts/engine.py` | `Engine` (Protocol), `EngineMeta`, `Capability`, `registry` |
| Result schema | `contracts/results.py` | `ResultSet`, `RESULT_COLUMNS` |
| Module slots | `contracts/modules.py` | `ClimateModule`, `DamageModule` (Protocols) |

Versioned together as `CONTRACTS_VERSION` (`0.1.0`).

## Decisions forced (resolved, see ADRs)

- **pydantic vs dataclasses** → pydantic (validation is the point of contracts).
- **canonical tabular format** → long-format results, wide matrices inside data objects
  ([ADR-0001](adr/0001-canonical-data-format.md)).
- **base classes vs protocols for engines** → Protocols ([ADR-0002](adr/0002-contracts-and-registry.md)).
- **language/performance** → Python + a targeted performance strategy
  ([ADR-0003](adr/0003-language-and-performance.md)).

## Notes for Phase 1

- `runner.load_data("toy")` is the only data source; Phase 1 makes this dispatch to the
  data store (build id → loaded objects). The seam is ready.
- `DummyEngine` is the template for a real engine: implement `meta` + `run`, register in
  `cge.engines.__init__`. Replace its placeholder rule with the Leontief price model (P2).
- The concordance/quality/store packages are stubs awaiting Phase 1.
- **Every engine/module DoD now requires an equation-level model doc with citations**
  (see `docs/documentation-standard.md`). `docs/models/io-price-model.md` is the worked
  example and doubles as the Phase 2 spec — implement Engine 1 against it.
