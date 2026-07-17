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

**Phase 3 — Web GUI complete.** A Streamlit app ([`docs/gui.md`](docs/gui.md)) now covers the
everyday workflow: browse data builds, explore any build's matrices like a spreadsheet, check
data quality, build datasets, run carbon-price scenarios, and explore results with a
supply-chain decomposition waterfall and assumptions printout. Built on the Phase 2 engine
(Leontief carbon-cost price model), the Phase 1 data layer (live EXIOBASE + quality/consistency
checks), and the Phase 0 contracts, with a standing model-validation suite
([`docs/validation.md`](docs/validation.md)). Next: partial-equilibrium volume response (Phase 4).

```bash
cge gui                                             # launch the web GUI
cge build --test                                    # offline data build (no download)
cge build --exiobase                                # live EXIOBASE build from Zenodo
cge data                                            # list builds in the store
cge quality <build_id>                              # build quality + consistency report
cge run --scenario examples/carbon_price_io.yaml    # Engine 1: carbon-cost price impacts
cge validate                                        # run the model-validation suite
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
# Environment — uv if you have it, else stdlib venv works identically.
# Install dev + data (+ gui) extras: the test suite exercises the data layer and GUI.
uv venv && source .venv/bin/activate && uv pip install -e ".[dev,data,gui]"
#   or:
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev,data,gui]"

# Prove the seams: run the dummy engine on the toy economy
cge engines                                        # list registered engines
cge run --scenario examples/carbon_price_io.yaml   # run Engine 1 end-to-end

# Tests + lint + model validation
pytest
ruff check src tests scripts && ruff format --check src tests scripts
cge validate --strict
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
