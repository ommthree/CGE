# CGE — Open, Modular CGE/IAM Platform with Nature Extension

An open-data economic model to answer questions like *"what happens to the cost and
production volume of good X under a carbon price?"*, extensible to nature-related risk
via ecosystem-service dependencies (ENCORE), with a web GUI for data inspection, data
quality, model builds, and model runs.

See [`roadmap.md`](roadmap.md) for the full plan, [`docs/adr/`](docs/adr/) for design
decisions, and [`docs/documentation-standard.md`](docs/documentation-standard.md) for the
rule that every engine/module ships an **equation-level model doc with citations** (worked
example: [`docs/models/io-price-model.md`](docs/models/io-price-model.md)).

## Status

**Phase 1 — Data layer complete.** Builds from the live EXIOBASE source (with an offline
test path) into harmonised data objects at full + small resolution, with correctness and
consistency checks enforced along the pipeline and a stored quality report per build. The
five inter-module contracts (Phase 0) are in place and a dummy engine proves the seams
end-to-end. No economics engines yet — Engine 1 is next (Phase 2).

```bash
cge build --test            # offline build from pymrio's test MRIO (no download)
cge build --exiobase        # live EXIOBASE build from Zenodo (large download)
cge data                    # list builds in the store
cge quality <build_id>      # show a build's quality + consistency report
```

## The five contracts

Modules talk through versioned schemas, never by importing each other. See
[`src/cge/contracts/`](src/cge/contracts/):

1. **Data objects** — `IOSystem`, `SAM`, `SatelliteAccount`, `ElasticitySet`, `ConcordanceMap`.
2. **Shock vocabulary** — typed, optionally time-pathed shocks (`CarbonPrice`, …).
3. **Engine protocol** — engines declare inputs, supported shocks, capabilities; a registry lists them.
4. **Result schema** — a common `ResultSet` (variable × sector × region × year) with provenance.
5. **Module slots** — `ClimateModule`, `DamageModule` interfaces for the pathway stack.

## Quickstart

```bash
# Environment — uv if you have it, else stdlib venv works identically
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
#   or:
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

# Prove the seams: run the dummy engine on the toy economy
cge engines                                        # list registered engines
cge run --scenario examples/carbon_price_toy.yaml  # run end-to-end

# Tests + lint
pytest
ruff check src tests && ruff format --check src tests
```

> The core install is intentionally light (pydantic, pandas, numpy, pyyaml). The heavier
> scientific deps are optional extras pulled in by the phase that needs them:
> `.[data]` (pymrio/duckdb), `.[cge]` (pyomo/scipy), `.[gui]` (streamlit).

## Layout

```
src/cge/
├── contracts/   # the five contracts (Phase 0)
├── data/        # ingestion, concordance, quality, store (Phase 1+)
├── engines/     # io_price, partial_eq, cge_static + registry
├── scenarios/   # shock loading, scenario library, NGFS reader
├── nature/      # ENCORE ingestion, exposure, nature→shock
├── modules/     # climate (FaIR), damages, dynamics
├── gui/         # Streamlit app
└── validation/  # toy economies, identity & known-answer tests
```
