# CGE/IAM Platform — Overview

*The entry point. Read this first, then follow the links into the detailed docs.*

---

## 1. What we are trying to achieve

A **modular, open-data economic model** that answers, for any good in any region:

> *"What happens to its **cost** — and its **production volume** — under a carbon price?"*

…and extends to **nature-related risk** (how dependent a good is on ecosystem services, and
what a degradation scenario does to it). It is built to be a **screening and stress-testing
tool**: transparent about its assumptions, defensible to reviewers, and usable by a non-modeller
through a web GUI.

**The design philosophy in one line:** *precise about costs, indicative about volumes,
transparent about assumptions.* We ship a useful answer early with the simplest defensible
model, then deepen the economics behind a stable interface — every stage is usable on its own.

**Deliberate non-goal:** this is **not** a GTAP-precision CGE or a process-based IAM
(GCAM/REMIND-class). Those are multi-year, multi-team efforts. This is a "toy but honest"
platform whose value is transparency, extensibility, and directionally-sound answers — the kind
of first-pass carbon-price and nature-exposure analysis several central banks and financial
institutions actually use.

---

## 2. How it fits together (the architecture)

```
┌─────────────────────────── Web GUI (Streamlit) ───────────────────────────┐
│  Data catalogue │ Data explorer │ Data quality │ Build │ Run │ Results     │
└──────────┬────────────────────────────┬───────────────────────┬──────────┘
           │                            │                       │
┌──────────▼──────────┐   ┌─────────────▼───────────┐   ┌───────▼──────────┐
│  Data layer         │   │  Engines (pluggable)    │   │  Scenario layer  │
│  live EXIOBASE →    │──▶│  1. IO price (cost)     │◀──│  typed shocks:   │
│  harmonised objects,│   │  2. partial-eq (volume) │   │  carbon price,   │
│  quality, aggregation│  │  3. CGE (planned)       │   │  nature stress … │
│  (parquet + DuckDB) │   │  + nature/ENCORE (plan) │   └──────────────────┘
└─────────────────────┘   └───────────┬─────────────┘
                          ┌───────────▼─────────────┐
                          │  Module slots (planned):│
                          │  climate (FaIR),        │
                          │  damages, dynamics      │
                          └─────────────────────────┘
```

The whole thing hangs off **five versioned contracts** — schemas that modules talk through
instead of importing each other. This is *the* load-bearing design decision: it's why a new
engine, data source, or shock type plugs in without rippling through the rest, and why the GUI
picked up Engines 2 with **zero GUI changes**.

| Contract | What it is |
|---|---|
| **Data objects** | `IOSystem`, `SAM`, `SatelliteAccount`, `ElasticitySet`, `ConcordanceMap` — every data source is an *adapter* into these |
| **Shock vocabulary** | typed, optionally time-pathed shocks (`CarbonPrice`, `NatureStress`, …); nature/climate modules *emit* shocks rather than calling engines |
| **Engine protocol** | engines declare capabilities (`prices`/`volumes`/`general_equilibrium`/`dynamic`) + supported shocks; a registry lists them, and the GUI renders from that metadata |
| **Result schema** | one long-format `ResultSet` with a mandatory provenance manifest (data version, scenario hash, assumption dump) |
| **Module slots** | `ClimateModule`, `DamageModule` interfaces for the future pathway stack |

→ Design rationale in the **Architecture Decision Records** ([`docs/adr/`](adr/)):
[canonical data format](adr/0001-canonical-data-format.md),
[contracts & registry](adr/0002-contracts-and-registry.md),
[language & performance](adr/0003-language-and-performance.md).

---

## 3. The components, at a high level

### Data layer — *turning raw EXIOBASE into something engines can use*
Ingests **EXIOBASE 3** (the best fully-open, environmentally-extended multi-regional
input–output database — 200 products × 49 regions, with emissions accounts) from its live
Zenodo source, harmonises it into the contract data objects, aggregates it to a runnable
"small build," and gates every step with **correctness and consistency checks** (balance
identities, unit normalisation, aggregation conservation) enforced *in the pipeline*, not just
in tests. Stored as parquet + a DuckDB catalogue.
→ [`docs/models/data-layer.md`](models/data-layer.md) (equation-level).

### Engine 1 — IO carbon-cost price model — *the cost answer*
The **Leontief price model**: a carbon price becomes a cost shock that propagates through the
full supply chain via `Δp = (I − Aᵀ)⁻¹ · τ · e`, giving the change in the price of every good,
decomposed into direct vs upstream tiers. Fixed technology, full pass-through — an
over-statement relative to a model with substitution, and no volume effect. **Validated on
live EXIOBASE 3** (reproduces the global CO₂ total, plausible fractional price changes, energy
sectors most exposed). This is the most defensible number in the platform — it has no fitted
parameters.
→ [`docs/models/io-price-model.md`](models/io-price-model.md) (the worked reference example
for the documentation standard).

### Engine 2 — partial-equilibrium volume response — *the volume answer*
Applies demand elasticities to Engine 1's prices: `Δq/q = ε·Δp`, across low/central/high
elasticity **bands** so the answer is an uncertainty envelope, not a point. Reuses Engine 1 for
prices (single source of truth). Volume magnitudes are **indicative** — there is no clean open
elasticity database, so the default set is assembled, ranged, and flags where it falls back to
a default.
→ [`docs/models/partial-equilibrium.md`](models/partial-equilibrium.md) (equation-level).

### Engine 3 — static CGE — *the general-equilibrium answer (planned)*
A small static computable general equilibrium model: nested-CES production, Armington trade, a
household and government with **carbon-tax revenue recycling** — the thing Engines 1–2
structurally cannot do. "Toy but honest": validated to replicate its benchmark and satisfy the
standard homogeneity/Walras tests, and to *bracket* Engines 1–2. The hardest phase; the risks
are data (SAM balancing) and the solver, not the code.
→ **Detailed plan:** [`docs/phase-5-plan.md`](phase-5-plan.md).

### Nature extension (ENCORE) — *the ecosystem-service answer (planned)*
Maps sectors to their **dependencies on ecosystem services** (pollination, water, …) via the
ENCORE knowledge base, propagates those through the supply chain (reusing Engine 1's Leontief
machinery), and translates degradation scenarios into `NatureStress` shocks fed to the engines.
This is the nature-risk extension the platform was specifically designed to reach.
→ Roadmap [Phase 6](../roadmap.md).

### Pathway stack — *"a CGE that speaks IAM" (planned, Phase 7)*
Recursive-dynamic runs to 2050 driven by exogenous **NGFS** scenarios, with one-way **FaIR**
climate coupling (emissions → temperature). Includes a **temperature-target back-solve**: fix a
target (e.g. ≤ 1.75°C by 2050) and invert the forward chain to find the **carbon-price path**
that hits it, then run forward for the sector impacts — the credible, target-driven "IAM-ish"
mode. Not a process IAM — it consumes pathways others generate and adds sector/supply-chain
resolution. → Roadmap [Phase 7](../roadmap.md), detail in
[`docs/energy-and-temperature-plan.md`](energy-and-temperature-plan.md).

### Planned scenario inputs — *energy prices & temperature targets*
Two requested extensions that slot into the existing shock/module seams (no new engines):
- **Country-level energy prices** — an optional per-country, per-carrier energy-cost change
  applied *on top of* a carbon price; near-term, low-risk, reuses the same supply-chain
  propagation.
- **Temperature-target back-solving** — the Phase 7 feature above.

→ [`docs/energy-and-temperature-plan.md`](energy-and-temperature-plan.md).

### Web GUI — *making it usable*
A Streamlit app: browse data builds, **explore any build's matrices like a spreadsheet**, check
data quality, build datasets, run carbon-price scenarios, and explore results (price
decomposition waterfall, volume envelope, and the run's assumptions). New engines appear
automatically via the registry.
→ [`docs/gui.md`](gui.md).

---

## 4. How we keep it trustworthy

Three cross-cutting disciplines, applied at the same bar to every component:

- **Provenance on every result.** No number exists without its data build id, scenario hash,
  and a full assumption dump — printed on every GUI run page. This is the single
  highest-leverage credibility feature.
- **Documentation to equation level, with citations.** Every engine/module ships a
  model-description doc stating the method to equation level and citing the peer work it derives
  from. It's a definition-of-done criterion, not optional.
  → [`docs/documentation-standard.md`](documentation-standard.md), citations in
  [`docs/references.md`](references.md).
- **A standing validation suite.** Beyond unit tests, `cge validate` runs model-correctness
  checks (analytic known-answers, economic identities, cross-engine consistency) — the audit
  that the *numbers* are still right, not just that the code runs.
  → [`docs/validation.md`](validation.md).

The platform has been through **seven rounds of independent adversarial review** (units,
concurrency, crash-safety, validation edge cases), all remediated with the fixes and reasoning
recorded → [`docs/review-2026-07-remediation.md`](review-2026-07-remediation.md). The honest
current status: **cost answers are validated against live data; volume answers are indicative;
the CGE and nature extensions are planned.**

---

## 5. Where things stand

| Phase | What it delivers | Status |
|---|---|---|
| 0 — Foundations & contracts | the five contracts, engine registry, provenance | ✅ [status](phase-0-status.md) |
| 1 — Data layer | live EXIOBASE ingestion, quality, aggregation, store | ✅ [status](phase-1-status.md) |
| 2 — Engine 1 (price) | carbon-cost pass-through, validated on live EXIOBASE | ✅ [status](phase-2-status.md) |
| 3 — Web GUI | the six-page Streamlit app | ✅ [status](phase-3-status.md) |
| 4 — Engine 2 (volume) | partial-equilibrium volume response with bands | ✅ [status](phase-4-status.md) |
| 5 — Engine 3 (CGE) | general-equilibrium + revenue recycling | 📋 [plan](phase-5-plan.md) |
| 6 — Nature (ENCORE) | ecosystem-service exposure + nature stress | ⬜ [roadmap](../roadmap.md) |
| 7 — Pathway stack | NGFS-driven dynamics + FaIR climate | ⬜ [roadmap](../roadmap.md) |

Roughly **40% of the full build by effort** — but the completed half is the minimum viable
version of the original ask (cost + volume, on real data, with a GUI), and the harder economics
(CGE) is still ahead. Full plan and honest feasibility assessment: [`roadmap.md`](../roadmap.md).

---

## 6. Try it

```bash
pip install -e ".[dev,data,gui]"
cge build --test                                       # offline data build (no download)
cge run --scenario examples/carbon_price_io.yaml       # Engine 1: cost impacts
cge run --scenario examples/carbon_price_volume.yaml   # Engine 2: volume response
cge gui                                                # the web app
cge validate                                           # model-validation suite
```

## Document map

- **This overview** — start here.
- [`roadmap.md`](../roadmap.md) — the full plan, effort, dependencies, feasibility.
- **Model docs** (equation-level): [data layer](models/data-layer.md) ·
  [Engine 1 price](models/io-price-model.md) · [Engine 2 volume](models/partial-equilibrium.md).
- **Phase status**: [0](phase-0-status.md) · [1](phase-1-status.md) · [2](phase-2-status.md) ·
  [3](phase-3-status.md) · [4](phase-4-status.md) · [5 plan](phase-5-plan.md).
- **Feature plans**: [energy prices & temperature back-solve](energy-and-temperature-plan.md).
- **How we work**: [documentation standard](documentation-standard.md) ·
  [validation](validation.md) · [GUI](gui.md) · [ADRs](adr/) · [references](references.md).
- **Review history**: [remediation record](review-2026-07-remediation.md).
