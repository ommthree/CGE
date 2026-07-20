# CGE — Open, Modular CGE/IAM Platform with Nature Extension

An open-data economic model to answer questions like *"what happens to the cost and
production volume of good X under a carbon price?"*, extensible to nature-related risk
via ecosystem-service dependencies (ENCORE), with a web GUI for data inspection, data
quality, model builds, and model runs.

> **New here? Start with [`docs/overview.md`](docs/overview.md)** — the executive summary of
> what the platform does, how the components fit together, and where everything is documented.
> Then follow [`docs/user-guide.md`](docs/user-guide.md) — a hands-on walkthrough from the
> simplest toy run up through every current feature.

See also [`roadmap.md`](roadmap.md) for the full plan, [`docs/adr/`](docs/adr/) for design
decisions, and [`docs/documentation-standard.md`](docs/documentation-standard.md) for the
rule that every engine/module ships an **equation-level model doc with citations** (worked
example: [`docs/models/io-price-model.md`](docs/models/io-price-model.md)).

## Status

**Phase 4 — volume response complete.** The platform now answers both halves of the original
question — the change in the **cost** of a good under a carbon price (Engine 1, Leontief price
model, validated on live EXIOBASE) and the change in its **production volume** (Engine 2,
partial-equilibrium demand response with a low/central/high uncertainty band). Built on the
Phase 3 GUI, Phase 1 data layer (live EXIOBASE + quality/consistency checks), and Phase 0
contracts, with a standing model-validation suite ([`docs/validation.md`](docs/validation.md)).
Also: **macro aggregates** (GVA/GDP/deflators, real vs nominal — Phase 4b) and a **static CGE**
(Engine 3, Phase 5) with **carbon-tax revenue recycling** — a pilot that calibrates on an
EXIOBASE-shaped SAM, passes the standard CGE battery (replication/homogeneity/Walras), and shows
the revenue-recycling effect and dirty→clean reallocation. It also runs as an **open economy**
(Armington imports + CET exports + a rest-of-world account, CES value added, elasticity sweeps),
where a carbon price causes textbook **carbon leakage**, and as a **true multi-region economy**
with bilateral trade between build regions (destination-specific route prices, explicit bilateral
market clearing, cross-region leakage; an IOSystem-driven multi-region SAM build is the remaining
sub-phase). Volume magnitudes are indicative; cost answers are validated. See
[`docs/models/cge-static.md`](docs/models/cge-static.md).

```bash
cge gui                                                # launch the web GUI
cge build --test                                       # offline data build (no download)
cge build --exiobase                                   # live EXIOBASE build from Zenodo
cge data                                               # list builds in the store
cge quality <build_id>                                 # build quality + consistency report
cge run --scenario examples/carbon_price_io.yaml       # Engine 1: carbon-cost price impacts
cge run --scenario examples/carbon_price_volume.yaml   # Engine 2: volume response (with bands)
cge validate                                           # run the model-validation suite
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
>
> The CGE (Phase 5) currently solves with a pure-Python **scipy** root-find (the model residual is
> numeric-only, so IPOPT is not yet wired for it — a symbolic-residual follow-up). The solver
> abstraction supports **IPOPT via pyomo** for symbolic problems when its binary is present
> (install once with `idaes get-extensions`, or a conda-forge `ipopt`), and falls back to a
> pure-Python **scipy** solver otherwise — so the engine and its tests run anywhere without a
> solver binary. `cge validate` gates IPOPT-only checks on availability.

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
