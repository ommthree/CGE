# ADR-0002: Modules talk through five versioned contracts; engines self-register

- **Status:** accepted
- **Date:** 2026-07-17
- **Phase:** 0

## Context

The project's whole value proposition is extensibility: new engines, new data sources, a
nature module, a pathway stack — added over months by (mostly) one person, without each
addition rippling through the rest. Left implicit, "modularity" degrades into modules
importing each other's internals until nothing can move independently.

## Decision

Define five explicit contracts, versioned as a set via `CONTRACTS_VERSION`, that modules
depend on **instead of** depending on each other:

1. **Data objects** (`IOSystem`, `SAM`, `SatelliteAccount`, `ElasticitySet`,
   `ConcordanceMap`) — every data source is an adapter into these.
2. **Shock vocabulary** — declarative, typed, optionally time-pathed. The nature/NGFS/
   damage modules *emit* shocks rather than calling engines.
3. **Engine protocol** — a `runtime_checkable` Protocol plus an `EngineMeta` descriptor;
   engines self-register into a process-wide `registry` at import time.
4. **Result schema** — a single `ResultSet` (long-format data + mandatory `RunManifest`).
5. **Module slots** — `ClimateModule`, `DamageModule` Protocols; swappable and omittable.

The GUI and CLI enumerate engines purely through `registry.all_meta()`, so adding an
engine adds a run option with no GUI code.

## Consequences

- A second data source is a new adapter; a new stress type is a new `Shock` subclass; a
  new engine is a new module that registers itself. None touch existing modules.
- The Protocol-based engine contract means engines don't import a base class — they only
  satisfy a shape — which keeps coupling minimal and testing trivial.
- Self-registration via import side effect is a mild "action at a distance"; mitigated by
  confining it to `cge.engines.__init__` and documenting it.
- Contracts are the one place we accept up-front design cost. Risk: over-engineering
  before real engines exist — mitigated by keeping each contract to one file and evolving
  under the version tag as phases land (the dummy engine is the only consumer at P0).

## Alternatives considered

- **Base classes instead of Protocols:** more familiar, but forces an inheritance
  dependency and is heavier to mock in tests. Protocols give the same guarantees here.
- **Entry-point plugin discovery** (packaging metadata): overkill for a single repo/single
  maintainer; revisit only if engines ever ship as separate distributions.
- **No registry (explicit wiring):** simplest, but then the GUI must hard-code the engine
  list — exactly the coupling we're trying to avoid.
