# Roadmap: Open, Modular CGE/IAM Platform with Nature Extension

**Goal.** A modular, open-data economic model that can answer: *"What happens to the cost — and ideally the production volume — of good X under a carbon price?"*, extensible to nature-related risk via ecosystem-service dependencies and impacts (ENCORE), with a simple web GUI for data inspection, data quality, model builds, and model runs.

**Guiding principle.** Ship a useful answer early with the simplest defensible model (environmentally-extended input–output), then deepen the economics (partial equilibrium → simple CGE) behind a stable interface. Every phase produces something usable on its own.

---

## 1. Architecture

```
┌─────────────────────────── Web GUI (Streamlit) ───────────────────────────┐
│  Data catalogue │ Data quality │ Model builds │ Scenario runs │ Results   │
└──────────┬────────────────────────────┬───────────────────────┬──────────┘
           │                            │                       │
┌──────────▼──────────┐   ┌─────────────▼───────────┐   ┌───────▼──────────┐
│  Data layer         │   │  Model layer            │   │  Scenario layer  │
│  ingestion,         │   │  pluggable engines:     │   │  typed shocks:   │
│  harmonisation,     │──▶│  1. IO price model      │◀──│  carbon price,   │
│  concordances,      │   │  2. partial equilibrium │   │  nature stress,  │
│  quality metrics    │   │  3. simple CGE          │   │  productivity …  │
│  (parquet + DuckDB) │   │  4. nature/ENCORE       │   └──────────────────┘
└─────────────────────┘   └───────────┬─────────────┘
                          ┌───────────▼─────────────┐
                          │  Module slots:          │
                          │  climate (FaIR),        │
                          │  damages, dynamics      │
                          └─────────────────────────┘
```

Each model engine consumes the same harmonised data objects and emits results in a common schema, so the GUI and scenario layer never care which engine ran. New engines (or new data sources) plug in without touching the rest.

### 1.1 Design for extensibility

The modularity lives in five **contracts** — versioned schemas that modules talk through instead of importing each other. These are defined in Phase 0 and are the most leverage-per-hour work in the project:

1. **Harmonised data objects.** `IOSystem`, `SAM`, `SatelliteAccount`, `ElasticitySet`, `ConcordanceMap` — every data source (EXIOBASE now; FIGARO/ICIO/successors later) is an *adapter* that maps raw downloads into these objects. Engines never see raw source formats, so a second data source is a new adapter, not a refactor. Sector/region classifications are explicit metadata on every object, with concordances as first-class data.
2. **Typed shock vocabulary.** Scenarios are declarative files (YAML) composed of typed shocks: `CarbonPrice`, `EnergyPrice`, `ProductivityShock`, `DemandShift`, `TradeCost`, `NatureStress`, … each optionally a *time path*. Engines declare which shock types they understand; static engines take year-slices of a path. This is the key seam: the nature module, NGFS reader, and damage module all *emit shocks* in this vocabulary rather than talking to engines directly — so any future stress type (litigation risk, pandemic, tariff war) is a new shock class plus zero engine changes. (`EnergyPrice` — an exogenous per-country, per-carrier output-price change — is **implemented**: the carrier's price is pinned to the requested change (a boundary condition) and propagates downstream. See `docs/energy-and-temperature-plan.md`.)
3. **Engine protocol.** An engine declares its required inputs, supported shock types, and capabilities (`prices`, `volumes`, `general_equilibrium`, `dynamic`), registered in a plugin registry. The GUI renders run pages from this metadata — adding an engine adds a GUI option with no GUI code.
4. **Result schema.** All engines emit a common `ResultSet` (variable × sector × region × year, long format, parquet) with full provenance: data version, engine version, scenario hash, assumption dump. Comparison across engines/scenarios/data-sources is then a query, not a feature.
5. **Module slots for the pathway stack.** Climate (`emissions → temperature`) and damages (`temperature → shocks`) are interfaces with one implementation each (FaIR; DICE-style damage function) — swappable, and omittable.

**Deliberate non-goals of the abstraction:** no premature support for multiple solvers per engine, no microservices, no plugin distribution mechanism. Modularity here means clean seams inside one Python package, which is what a solo-maintained project can actually sustain.

### 1.2 Suggested stack

Python ≥3.11; `pymrio` for multi-regional input–output handling; `pandas`/`xarray` + parquet + DuckDB for data; `pydantic` for contracts/config; `scipy`/`pyomo` + IPOPT (open solver) for equilibrium computation; Streamlit for the GUI; `pytest` + a validation library (balance identities, known-answer tests); `uv` for environments; ruff + pre-commit; GitHub Actions CI.

### 1.3 Proposed repo layout

```
cge/
├── contracts/        # data objects, shock types, engine protocol, result schema (Phase 0)
├── data/
│   ├── adapters/     # exiobase.py, later figaro.py, icio.py
│   ├── concordance/  # concordance framework + stored maps (incl. ENCORE↔EXIOBASE)
│   ├── quality/      # balance checks, coverage metrics → QualityReport
│   └── store/        # parquet/DuckDB catalogue, versioning, provenance
├── engines/
│   ├── io_price/     # Engine 1: Leontief price pass-through
│   ├── partial_eq/   # Engine 2: elasticity-based volume response
│   ├── cge_static/   # Engine 3: static CGE (SAM build + pyomo model)
│   └── registry.py   # plugin registry the GUI reads
├── scenarios/        # shock classes, YAML loader, scenario library, NGFS reader
├── nature/           # ENCORE ingestion, exposure scoring, nature→shock translation
├── modules/          # climate/ (FaIR), damages/, dynamics/ (recursive wrapper)
├── gui/              # Streamlit app, one page-module per GUI area
├── validation/       # known-answer tests, toy economies, identity checks
└── docs/adr/         # architecture decision records
```

### 1.4 Core open data

| Dataset | Role | Access |
|---|---|---|
| EXIOBASE 3 | Multi-regional environmentally-extended IO tables (163 industries/200 products, 44 countries + 5 RoW regions), incl. CO₂/GHG satellite accounts | Free (Zenodo, CC BY-SA) |
| OECD ICIO / Eurostat FIGARO | Cross-check / alternative IO source | Free |
| ENCORE knowledge base | Sector → ecosystem-service dependencies and impact drivers | Free download (registration) |
| NGFS scenario database (IIASA) | Carbon price / GDP / population pathways to 2050+ | Free |
| Elasticities (literature: GTAP-published papers, USDA, meta-analyses) | Demand/substitution responses for PE and CGE engines | Free but scattered; see risks |
| UN Comtrade / FAOSTAT (optional) | Validation, physical volumes | Free |

Note: **GTAP itself (the standard CGE database) is licensed, not open** — this is the single biggest reason the roadmap builds the CGE calibration from EXIOBASE instead. That is done in the literature but is genuine work (Phase 5).

---

## 2. Phases

Effort assumes **one competent person, quantitative background, comfortable in Python but new to CGE modelling**, in full-time-equivalent (FTE) days/weeks. Part-time: scale accordingly. Ranges reflect how much polish/validation you invest. Each phase lists tasks, a definition of done (DoD), the key design decisions it forces, and its main risks.

**Phase headers are solo-project FTE totals — they must equal the sum of that phase's own task
estimates** (review P2, round 10: two phase headers previously understated their own task totals
by 4–10 weeks — Phase 5d said "4–8 wk" against task estimates summing to 8–12 wk, and Phase 8 said
"8–14 wk" against 8a+8b summing to roughly 13–23 wk; both are corrected below). Where a phase
splits into genuinely independent workstreams that **could** run in parallel with an additional
person (e.g. Phase 8's 8a validation vs 8b delivery), the header states the solo-FTE total, and a
parallel-capable **elapsed-time** estimate is called out separately in that phase's own text —
parallelism reduces calendar time for a team, never the total work for a solo builder.

### Phase 0 — Foundations & contracts (1–1.5 wk)

| # | Task | Effort |
|---|---|---|
| 0.1 | Repo, environment (uv), pyproject, ruff/pre-commit, pytest, CI | 0.5–1 d |
| 0.2 | Package skeleton per layout above | 0.5 d |
| 0.3 | Contracts v0: pydantic models for `IOSystem`, `SAM`, `SatelliteAccount`, `ElasticitySet`, `ConcordanceMap`; shock base class + `CarbonPrice`; `EngineProtocol` + registry; `ResultSet` schema; climate/damage slot interfaces | 2–3 d |
| 0.4 | Provenance & config: run manifest (data version, engine version, scenario hash, assumption dump), YAML config loader | 1 d |
| 0.5 | Toy fixtures: hand-built 3-sector/2-region economy with known analytic answers, used by all future engine tests | 1 d |
| 0.6 | Docs: ADR template + first ADRs (canonical data format, contract versioning policy) | 0.5 d |

**DoD:** `uv pip install -e .` works; a dummy engine registers itself, runs end-to-end on the toy economy via a YAML scenario, and emits a schema-valid `ResultSet` with provenance; CI green.
**Decisions forced:** pydantic vs plain dataclasses (recommend pydantic); canonical tabular format — long-format pandas vs xarray (recommend long-format + helpers; xarray only if it earns its keep later).
**Risks:** over-engineering the contracts before any real engine exists. Keep each contract to one file; evolve under version tags as phases land.
**Depends on:** nothing. **Unblocks:** everything.

### Phase 1 — Data layer: EXIOBASE ingestion + quality (2–3 wk)

| # | Task | Effort |
|---|---|---|
| 1.1 | Downloader/cacher for EXIOBASE 3 (Zenodo), version pinning, checksums, licence note propagation | 1–2 d |
| 1.2 | Adapter: pymrio parse → `IOSystem` + `SatelliteAccount` (GHG by gas, energy use); parquet store + DuckDB catalogue | 3–4 d |
| 1.3 | Aggregation machinery: sector/region aggregation driven by `ConcordanceMap`; produce the "small" interactive build (40–60 sectors × ~10 regions) alongside the full one | 2–3 d |
| 1.4 | Quality module → `QualityReport`: IO balance identities (row/column sums, supply=use), negative-value flags, imputation/RoW share, satellite-account coverage, year/version drift vs previous build | 2–3 d |
| 1.5 | Metadata registry: classifications, units, currency/price basis (basic prices), reference year handling | 1–2 d |
| 1.6 | Concordance framework: many-to-many weighted maps with validation (weights sum to 1, no orphans), stored as data files — reused for ENCORE (P6) and any second source (P7) | 2 d |

**DoD:** one command builds full + small datasets from a clean machine; `QualityReport` generated and stored; reproduction checks pass (e.g. global CO₂ total and a handful of published country totals within tolerance).
**Decisions forced:** product-by-product vs industry-by-industry tables (recommend pxp); monetary vs mixed-unit satellite handling; how aggressively to pre-aggregate for the interactive build.
**Risks:** memory — full MRIO is ~9800×9800 per matrix, ×matrices ×years; mitigate with float32, chunked parquet, and doing analysis on the small build by default. EXIOBASE contains small negatives (stock changes, subsidies) — flag, don't silently clip.
**Depends on:** P0. **Unblocks:** every engine.

### Phase 2 — Engine 1: IO carbon-cost pass-through (1–2 wk) ⭐ first real answers

| # | Task | Effort |
|---|---|---|
| 2.1 | Leontief price model: carbon cost per sector from emissions intensity × carbon price; full pass-through via (I−Aᵀ)⁻¹; scope options (scope 1 only; optionally include purchased electricity/energy explicitly) | 2–3 d |
| 2.2 | Scenario schema v1: `CarbonPrice` shock — level, gas coverage, region coverage, sector exemptions; YAML round-trip | 1–2 d |
| 2.3 | Result decomposition: direct vs upstream contribution per good, as **Neumann-series tier aggregates** (direct + first 2–3 upstream tiers + residual, summing to Δp). Note: these are aggregate tier contributions, **not** enumerated structural paths — full structural path analysis is a separate heavier method, not implemented. | 2–3 d |
| 2.4 | Validation: analytic tests on the toy economy; known-answer tests against published EXIOBASE carbon multipliers/footprints; assumption dump wired into `ResultSet` | 1–2 d |

**DoD:** CLI run of "€100/tCO₂, EU ETS-like coverage" returns Δprice for every good in every region with decomposition, in seconds on the small build; tests green; **model doc `docs/models/io-price-model.md` matches the implementation** (already drafted to equation level as the standard's worked example).
**Stated assumptions (documented in every result):** fixed technology, full cost pass-through, no substitution, no demand response, no volume effects. Because substitution would let firms avoid part of the cost, this tends to **over-state** the cost impact relative to a model with substitution — but it is not a proven upper bound over every model (supply/factor-market effects can push either way).
**Depends on:** P1. **Unblocks:** GUI v1 with real content; baseline for Engines 2–3; propagation machinery reused by P6.

### Phase 3 — GUI v1 (2–3 wk)

| # | Task | Effort |
|---|---|---|
| 3.1 | Streamlit scaffold: navigation, page-module convention, light theming | 1–2 d |
| 3.2 | Data catalogue page: datasets, versions, coverage, lineage, licence | 2 d |
| 3.2b | **Spreadsheet-style data explorer**: browse any build's matrices/accounts like an Excel sheet — sortable/filterable/scrollable grid over the A-matrix, final demand, and satellite intensities; pick sector/region slices; search labels; cell-level inspection; CSV export. Backed by DuckDB queries + a virtualised grid (`st.dataframe`/AgGrid) so the full ~9800² MRIO paginates without loading whole into the browser. This is the "look at the data like an Excel sheet" surface. | 2–3 d |
| 3.3 | Quality dashboard: render `QualityReport`s (incl. pipeline **consistency/plausibility checks**), drill-down to sector/region, build-over-build drift | 2–3 d |
| 3.4 | Build page: trigger/inspect data builds and aggregations; job wrapper (subprocess + log capture) so long jobs don't freeze the UI | 2 d |
| 3.5 | Scenario builder + run page: form → YAML scenario; engine picker driven by registry metadata (capabilities shown, unsupported shocks greyed out) | 2–3 d |
| 3.6 | Results explorer: sortable/filterable tables; charts (price impact by good/region, decomposition waterfall); per-run assumption printout; CSV/parquet export; side-by-side run comparison (v1: two runs) | 2–3 d |

**DoD:** a colleague can, without help, inspect data quality, run a carbon-price scenario, understand what assumptions produced the numbers, and export results.
**Decisions forced:** where runs execute (in-process for the small build is fine for v1; job queue is a P7 concern).
**Risks:** GUI scope creep — hold every page to "table + one chart + export" until engines stabilise.
**Depends on:** P1–P2. **Unblocks:** everyday usability; stakeholder demos.

### Phase 4 — Engine 2: partial-equilibrium volume response (2–4 wk) — ✅ IMPLEMENTED

> **Done** (see `docs/phase-4-status.md`). Production-volume response: a finite-change demand
> response Δy/y=(1+Δp)^ε−1 propagated through the Leontief quantity system x=(I−A)⁻¹y to give
> Δx/x, on Engine-1 prices, with a low/central/high elasticity band; functional default
> elasticity library; `partial_eq` engine + validation suite; GUI picks it up via the registry.
> Volume magnitudes are indicative (elasticity-dependent); the Armington nest and a curated
> elasticity set are documented follow-ups.

| # | Task | Effort |
|---|---|---|
| 4.1 | Elasticity library: schema (value, source citation, native classification + concordance, confidence tag, low/central/high range); initial population from literature (published GTAP parameter papers, USDA, meta-analyses) for the small build's sectors | 3–5 d (then ongoing) |
| 4.2 | PE engine: finite-change demand response Δy/y=(1+Δp)^ε−1 on Engine 1 prices, propagated through the Leontief quantity system x=(I−A)⁻¹y to give production volume Δx/x (a linear solve, not the old linear ε·Δp / fixed-point form); Armington domestic/import substitution specified but not implemented (v1) | 3–5 d |
| 4.3 | Uncertainty as a first-class output: run low/central/high elasticity bundles → result envelopes in `ResultSet` | 1–2 d |
| 4.4 | Validation: sign/magnitude sanity vs published carbon-price incidence studies; convergence tests; toy-economy analytics | 2–3 d |

**DoD:** volume impact *ranges* per sector/region for a carbon-price scenario; the GUI picks up the new engine purely via the registry (no GUI changes); every result carries its elasticity provenance; **model doc (equation-level demand/Armington response + per-parameter sourcing) exists** per the documentation standard.
**Risks:** elasticity gaps for many sectors — be explicit about defaults and tag them; resist inventing precision.
**Depends on:** P2 (elasticity gathering can start any time). **Unblocks:** volume answers; elasticity library feeds P5.

### Phase 4b — Macroeconomic aggregates: GVA, GDP, deflators; real vs nominal (1–2 wk PE tier; native in P5) — PE tier ✅ IMPLEMENTED

> **What the user asked for:** per time-step, **gross value added (GVA) per sector/country**,
> **GDP (and its growth) per country**, and — if possible — **interest rates and inflation**, so
> results can be read in **real or nominal** terms. This phase makes those first-class outputs and
> is explicit about which are genuine model outputs versus which need a macro closure the IO/CGE
> core does not have. Delivered in **two tiers** (indicative PE now; proper GE at P5).

**The economics, stated honestly (this drives the scope):**
- **GVA & GDP are native to this framework.** GVA is value added by sector (`IOSystem.value_added`
  already carries the base year); GDP by country is the sum of sectoral GVA (production approach).
  A carbon/energy shock changes them through the price and volume responses the engines already
  compute. So GVA/GDP *changes* are derivable — indicatively in PE, properly in the CGE.
- **Inflation = a price index (deflator), which is native.** An economy-wide price-level change
  (GDP deflator / CPI) is exactly what a price model produces; in the CGE the CPI is the standard
  numéraire. This is what makes **real vs nominal** well-defined: real = nominal deflated by the
  index. A crude aggregate deflator is available even in the PE tier (a value-added-weighted mean
  of sector price changes); the proper CPI/GDP-deflator distinction is a CGE output.
- **Interest rates are NOT a native IO/CGE output — flagged, not faked.** A static CGE has a
  **capital rental rate** (the real return to capital, an equilibrium factor price) — we *can*
  report that, and it is the closest honest analogue. A **nominal monetary interest rate** needs a
  monetary/macro-financial closure (a policy rule, money demand) the model does not contain;
  promising one from this core would be inventing precision. It is scoped as an **optional
  documented bolt-on** (a reduced-form Taylor-rule-style mapping from the model's inflation +
  output-gap outputs), clearly labelled illustrative — never a headline result.

| # | Task | Effort |
|---|---|---|
| 4b.1 | ✅ **GVA/GDP accounting layer (PE tier).** `cge.accounting` computes ΔGVA per sector/region (base-year value-added share × price/volume responses) and ΔGDP per region (VA-weighted), per time-step; adds `gva_change` / `gdp_change` result variables; applied engine-agnostically in the runner. Value added is derived from the IO identity, not a separate table. | done |
| 4b.2 | ✅ **Deflators & real-vs-nominal (PE tier).** Value-added-weighted GDP deflator per region; every aggregate emitted as both **nominal** and **real** (deflated by the index) in the `ResultSet` and GUI, provenance-tagged; `macro` validation suite + model doc (`docs/models/macro-aggregates.md`). | done |
| 4b.3 | **GVA/GDP/CPI as native CGE outputs (GE tier, folds into P5).** In the CGE these are model variables, not post-hoc arithmetic: factor income → GVA, CPI/GDP-deflator as the numéraire, real vs nominal exact. Report the **capital rental rate** as the honest "interest-rate-like" factor price. Cross-check the P5 aggregates against the 4b.1/4b.2 PE estimates (should be same-sign, GE typically smaller) | folded into P5.2/5.3 |
| 4b.4 | **GDP growth over time.** A path of GDP/GVA per region across the scenario's time-steps; a *growth-rate* series needs the recursive-dynamic wrapper (**7.1**) for capital/labour/productivity updating between years — static runs give a level per year, dynamics give genuine growth | with 7.1 |
| 4b.5 | **Optional interest-rate bolt-on (illustrative).** A documented reduced-form mapping (Taylor-rule-style) from the model's inflation + output-gap to a nominal policy rate, behind a clearly-labelled "illustrative macro-financial overlay" flag — off by default, never a headline | 3–5 d, optional |

**DoD (PE tier, 4b.1–4b.2):** every engine run reports GVA per sector/region, GDP per region, and an aggregate deflator per region per time-step, each available in real and nominal terms, provenance-tagged, in the `ResultSet` and GUI; a validation check ties ΔGDP to Σ ΔGVA and confirms real = nominal at zero inflation. **DoD (GE tier):** the CGE emits GVA/GDP/CPI and the capital rental rate as native variables reproducing the base-year SAM aggregates; the PE-vs-GE cross-check is green.
**Decisions forced:** deflator base (CPI vs GDP deflator — report both once in the CGE); whether to ship the interest-rate overlay at all (recommend: document it, build only if asked).
**Risks:** the PE-tier GVA/GDP are indicative and will move under GE substitution — label them exactly as Engine 2's volumes are labelled; **do not let the interest-rate overlay be read as a real forecast** — it is the item most likely to be over-interpreted, so it is opt-in and caveated.
**Depends on:** P4 (needs the volume response for real GVA) for the full PE tier; the deflator alone needs only P1/P2. **Unblocks:** real/nominal reporting everywhere; feeds the P5 credibility cross-checks and the P7 dynamic GDP-growth paths.

### Phase 5 — Engine 3: static CGE (6–12 wk) ⚠ the hard part — pilot core ✅ done; macro closure (5.2, 5d) reopened as outstanding, not new scope

> **Progress:** the **solver gate (5.0)**, **SAM construction & balancing (5.1)**, and the
> **correctness-first pilot (5.2a)** are built and green. `cge.engines.cge_static` calibrates to a
> hand-checkable balanced SAM **and to a SAM built from an aggregated EXIOBASE-shaped build** (the
> offline pymrio test MRIO — *not* live EXIOBASE yet; a live CGE gate is a follow-up), and in
> both cases passes the standard CGE battery — **benchmark replication** (to machine precision),
> **homogeneity**, and **Walras' law** — plus theory-consistent carbon-price responses (dirty
> output falls, dirty price rises, real GDP falls). SAM build (`cge.data.sam`): closed single-region
> construction from the IO identity, RAS balancer, and a SAM quality report (balance, aggregate
> preservation, adjustment audit, assumed capital share) surfaced in the run manifest; a failing SAM
> is rejected. Solver: IPOPT via pyomo when present, scipy fallback so CI needs no binary; a
> non-optimal solve raises. Model doc: `docs/models/cge-static.md`.
>
> **Revenue recycling (5.3) is in:** the carbon tax collects revenue R=Σ τ·e[i]·X[i] and recycles
> it to the household (lump_sum / labour_tax_cut). The model demonstrates the **revenue-recycling
> effect** (recycling offsets the welfare loss) and **sectoral reallocation** from dirty to clean —
> the headline GE features Engines 1–2 cannot show — with Walras holding under the recycled tax. A
> closed economy cannot destroy revenue, so a `none`-recycling positive carbon price defaults to
> lump_sum (recorded) and points to Engine 1 for the pure price-side view. Emits `welfare_change`,
> `carbon_revenue`, and per-factor price changes.
>
> **Now complete for the single-region economy:** the **Armington/CET open economy** (imports and
> exports respond to relative prices; a carbon price causes textbook carbon leakage), a **CES
> value-added nest** (non-unitary factor substitution — capital/labour substitution as relative
> factor prices move; NB *not* a double-dividend model — that needs a distortionary labour-tax
> wedge / heterogeneous households, a follow-up), **Armington elasticity sensitivity sweeps**
> (low/central/high envelopes), and a **non-zero-current-account closure** (foreign savings enter
> household income as the ROW capital transfer er·Sf). All replicate their benchmark to machine
> precision and pass the
> standard CGE battery.
>
> **Also complete:** **true multiple regions** with **bilateral trade** between build regions
> (`model_multi` — a closed global economy of R regions, each with an Armington composite over the
> domestic variety + imports from every partner and a CET transform into domestic sales + exports to
> every partner; region-specific factors and households). Every trade route has its own
> **destination-specific price** `pe[o,s,d]`, and every bilateral goods market clears explicitly
> (`M[d,s,o]=EX[o,s,d]`) rather than assuming a law-of-one-price reduction. Replicates its benchmark
> to machine precision, clears every bilateral and factor market under shock, and shows
> **cross-region carbon leakage** (a carbon price in one region relocates production and raises
> imports from partners); results are region-tagged.
>
> **Remaining, build-only (deferred):** the GE tier of the macro aggregates (native per-sector GVA +
> capital rental rate — the rental rate is already emitted as a factor price); a **live-EXIOBASE
> multi-region SAM build** — now explicitly owned as **§5.1b** below (review P2, round 10: this was
> acknowledged but unowned by any numbered task; Phase 7b/8 cannot credibly operate on a hand-built
> supplied multi-region SAM); and per-cell (rather than uniform) trade elasticities.
>
> **Correction (2026-07, prompted by an independent review): Phase 5 is not fully complete —
> government, savings/investment, and the energy nest were dropped silently, not carried forward.**
> The pilot above ("5.2a") is a genuine, correctness-first CGE core: it replicates its benchmark,
> satisfies homogeneity and Walras, and demonstrates revenue recycling, Armington/CET trade, CES
> value added, and true multi-region bilateral clearing. But **5.2's own original spec** (below)
> called for a **government/fiscal account**, **savings/investment** (capital accumulation, not
> just a factor endowment), and a **genuine energy nest** (KL–E–M, not KL with energy as a plain
> intermediate) — **none of which exist in the implemented model.** Carbon-tax revenue is a
> pass-through (collected and immediately recycled in the same period; there is no government
> balance sheet to hold it), there is no investment/savings mechanism (so **Phase 7.1's recursive
> dynamics, which explicitly need "savings/investment → next-year capital stock", currently have
> no real mechanism to update from** — capital accumulation there would be imposed bookkeeping, not
> a modelled decision), and energy is just another Leontief/CES input with no separable nest a
> carbon price can shift substitution within. These were listed as "remaining, later enhancement"
> in earlier drafts of this roadmap but dropped from the "remaining (deferred)" list above when
> Phase 5's header was marked complete — an honesty gap in the roadmap itself, not just the model.
> **They are reopened as Phase 5d below** — carried-forward Phase 5 debt, not new scope — and are a
> prerequisite for Phase 7b (a baseline/pathway harmonizer is low-value without a government
> account and an energy nest to harmonize NGFS's fiscal- and energy-transition pathways against).

> **Detailed plan: [`docs/phase-5-plan.md`](docs/phase-5-plan.md)** — solver-first sequencing,
> equation-level model structure, SAM balancing with an audit trail, the standard CGE
> correctness test battery (replication / homogeneity / Walras) plus cross-engine consistency
> checks, and honest scope. The outline below is the summary; the plan is the spec.

**5.0 Solver & environment (2–4 d) — do first.** pyomo needs a nonlinear solver (IPOPT). It is
not currently installed; the plan chooses IPOPT (via `idaes` binary) with a pure-Python scipy
fallback so CI stays solver-independent on the 2-sector toy. A solver abstraction records the
solver + termination status; a non-optimal solve raises (never returns numbers).

**5.1 SAM construction (2–4 wk) — ✅ closed single-region done; ✅ open SAM (hand-built + from a build) done**
- ✅ Map EXIOBASE flows into SAM accounts: sectors (activity=commodity, collapsed), factors (labour, capital), representative household — **single-region closed** economy (regions summed). `cge.data.sam.build_sam`. A **hand-built open SAM** (`cge.data.sam.toy_open`) with separate activity/commodity + rest-of-world accounts drives the open-economy variant, and **`build_open_sam`** constructs an open SAM (home region + rest-of-world, Armington/CET/ROW accounts) from an EXIOBASE-shaped **IOSystem** — balanced by construction, quality-gated, and it replicates its benchmark to machine precision (`open_replicates_on_built_sam`). A **live-EXIOBASE** open build (vs the offline test MRIO) remains the only open piece outstanding.
- ✅ Value added derived from the IO identity and split capital/labour by a documented assumption; RAS balancer (`balance.py`) for the thin-data path; SAM-specific `QualityReport` (`quality.py`): balance, **aggregate preservation**, adjustment audit, negative-cell + assumed-share flags — surfaced in the run manifest, and a failing SAM is rejected.
- **DoD (met for the closed single-region SAM):** balanced SAM reproducing source aggregates to 1e-6 with an audit trail; the CGE calibrates on it and replicates its benchmark to machine precision (`replicates_on_built_sam` validation check). Runs on the offline EXIOBASE-shaped test build; a live-EXIOBASE CGE gate is a follow-up.

**5.1b Live-data multi-region SAM build (3–5 wk) — new task (review P2, round 10): owns the
previously-unowned "live-EXIOBASE open/multi SAM build" and "IOSystem→multi-region-SAM builder"
deferred items, explicitly, since Phase 7b (harmonization) and Phase 8 (empirical validation)
cannot credibly operate on the current hand-built supplied multi-region SAM.**
- An IOSystem-driven multi-region SAM builder generalising `build_open_sam` (§5.1) from
  home-region + single ROW to **R build regions, each a genuine region** with its own household
  and bilateral trade to every other region — reusing the same construction pattern (balanced by
  construction, quality-gated).
- **Satellite alignment**: the multi-region carbon-cost path (currently supplied-SAM-only —
  `_run_multi` has no IOSystem/SatelliteAccount entry point, unlike the open model's
  `_run_open_from_io`) needs the same `carbon_cost_vector`-based per-year effective cost
  construction, aggregated per build region.
- **Final-demand attribution** using the by-region `final_demand_kind` machinery already built for
  the open path (§5.1's `open_fd_attribution` check), generalised to attribute each build region's
  own final demand rather than a single home region's.
- **Trade-materiality handling**: the live build's bilateral trade will be genuinely sparse (most
  region pairs don't trade every good) and may contain near-zero noise from aggregation/RAS —
  reuse `ROUTE_MATERIALITY_THRESHOLD` and `active_routes` (calibrate_multi.py) rather than
  reintroducing a bare `>0` check.
- **Topology validation**: a live build's region-trade graph must be checked for connectivity
  (`connected_components`) before calibration — a live multi-region build is far more likely than
  a hand-built toy to contain a genuinely disconnected pair of regions (e.g. two small economies
  with no direct recorded trade link after aggregation), which must be rejected with a clear
  message, not silently solved.
- **Live-data replication gate**: `multi_region_live_replicates_on_built_sam`, the multi-region
  analogue of `open_replicates_on_built_sam` — the multi-region CGE calibrates on the live-built
  SAM and replicates its benchmark to machine precision, gating this task's own DoD.
- **DoD:** an EXIOBASE-shaped multi-region SAM builds from a real (or realistically-aggregated)
  IOSystem + SatelliteAccount, passes the SAM quality report, and the multi-region CGE calibrates
  on it and replicates to machine precision, with topology validation and trade-materiality
  handling exercised by the live data (not just the hand-built toy fixtures).
- **Depends on:** §5.1 (the single-region open live build, whose patterns this generalises);
  `calibrate_multi.py`'s `active_routes`/`connected_components` (already built). **Unblocks:**
  Phase 7b (country/sector harmonization needs a live multi-region SAM, not a hand-built one) and
  Phase 8 (empirical validation/hindcasting needs real data to validate against).

**5.2 Model core (2–4 wk) — ✅ pilot done; government/investment/energy-nest reopened as 5d**
- ✅ Static CGE in pyomo/scipy: Armington imports / CET exports, household demand (Cobb-Douglas), carbon-tax revenue with recycling options (lump-sum vs labour-tax cut, as a same-period pass-through), square-model and degrees-of-freedom checks (proven square via the replication gate).
- ⏳ **Not built, reopened as 5d:** nested CES production with a genuine **energy nest** (KL–E–M, so carbon pricing can shift substitution *within* the energy bundle, not just KL); a **government/fiscal account** (the tax is collected and recycled same-period with no balance sheet — cannot carry a deficit/surplus or fund non-recycled spending); **investment/savings** (standard closure choices — savings-driven, fixed trade balance — were never implemented; there is no capital-accumulation mechanism for 7.1 to update between years).
- Pilot single-region model first; extend to multi-region only once the pilot passes 5.3's tests. (Done — see "Also complete" above and §8a in cge-static.md.)
- **DoD:** model solves from the SAM; equation/variable count documented; closures switchable by config. **Met for the pilot's own scope**; not met for the government/investment/energy-nest scope this section originally specified — see 5d.

**5.3 Calibration & credibility tests (2–4 wk) — ✅ correctness battery + revenue recycling + elasticity sensitivity sweeps done**
- ✅ Benchmark replication (zero shock reproduces the SAM to machine precision), ✅ homogeneity, ✅ Walras' law — in CI (toy + real EXIOBASE SAM).
- ✅ **Revenue recycling** (lump_sum / labour_tax_cut): the carbon tax collects R and recycles it; validated **revenue-recycling effect** (recycling offsets the welfare loss) and **sectoral reallocation** (dirty→clean), with Walras holding under the recycled tax. Emits welfare, carbon revenue, factor prices.
- ✅ Cross-engine sign consistency with Engine 2 (dirty sector contracts, same sign). ✅ Elasticity sensitivity sweeps (`armington_sensitivity_sweep`, low/central/high envelopes) are done for the open economy — see "Now complete for the single-region economy" above. A published-CGE literature bracket (cross-checking the sweep's envelope against values from the literature, not just internal sensitivity) remains a later enhancement.
- **DoD (met for the pilot):** replication + homogeneity + Walras + recycling-effect tests green in CI; equation-level model doc (`docs/models/cge-static.md`) exists with closures, recycling, and sources.

**Decisions forced:** nesting structure; closure defaults; single- vs multi-region sequencing (recommendation above); how much tax detail to fabricate vs omit.
**Risks:** SAM balancing is a known rabbit hole — timebox it and document rather than perfect; IPOPT convergence (mitigate: good starting values = benchmark data, gradual shock ramping); results sensitive to elasticities (mitigate: sweeps + Engine 2 cross-check, keep "toy but honest" framing).
**Depends on:** P1, P4 (elasticities). **Unblocks:** GE price *and* volume answers; P7.

### Phase 5d — Complete the macroeconomic closure (8–12 wk solo FTE) — reopened Phase 5 debt, not new scope

**This phase is not new scope.** Every item below was specified in 5.2's original design (a
government account, savings/investment, an energy nest) and is being reopened, not invented — see
the Phase 5 correction note above. It is sequenced as its own phase because it is now a real chunk
of independent work, and because it is a genuine **prerequisite for Phase 7b** (a baseline/pathway
harmonizer is low-value without a government account and an energy nest to harmonize NGFS's fiscal-
and energy-transition pathways against) and for **Phase 7.1's recursive dynamics** (which need a
real savings-investment mechanism to update capital between years — without this, capital
accumulation in the dynamic wrapper is imposed bookkeeping, not a modelled decision).

| # | Task | Effort |
|---|---|---|
| 5d.1 | **Government/fiscal account.** A tracked government balance sheet: carbon-tax revenue, subsidies, other tax instruments, and public spending as explicit SAM accounts (not a same-period collect-and-recycle pass-through); a `fiscal_balance` result variable; the closure choice (balanced budget vs deficit-financed) is a first-class, documented assumption | 1–2 wk |
| 5d.2 | **Savings and investment.** A savings-investment identity closing the household/government/RoW accounts; investment demand as a new final-demand component with its own sectoral composition (not folded into household consumption); the standard closure choices from the original 5.2 spec (savings-driven investment as default, fixed vs flexible trade balance as alternatives) | 1–2 wk |
| 5d.3 | **Capital accumulation, depreciation, and premature retirement.** Extends 5d.2 with a capital stock that depreciates and accumulates from investment — the actual mechanism Phase 7.1's recursive-dynamic wrapper needs between static solves (7.1 currently has no real capital-accumulation identity to call); premature asset retirement (e.g. stranded fossil capital under a carbon shock) as a documented, switchable option | 1 wk, shared with 7.1 |
| 5d.4 | **Labour market: employment/unemployment and wages.** Today's factor market clears with a fixed endowment and a flexible wage (full employment by construction); add a documented labour-market closure alternative (a wage floor/curve with involuntary unemployment) so factor-market outcomes aren't always "full employment, wage adjusts" by assumption | 1–2 wk |
| 5d.5 | **A genuine energy nest (KL–E–M).** Today energy is a plain Leontief/CES intermediate like any other; nest it separately (capital-labour, energy, materials) so a carbon price can shift substitution *within* the energy bundle (e.g. toward electricity, away from fossil fuels) rather than only across sectors. This is the single highest-value item for anything touching NGFS energy-transition pathways (Phase 7b) | 2–3 wk |
| 5d.6 | **Adaptation and transition investment.** A documented shock/response channel: a scenario can specify adaptation or transition capital expenditure (e.g. retrofitting, energy-efficiency investment) that competes with 5d.2's investment demand and is reported as its own line item, not absorbed silently into aggregate investment | 1 wk |
| 5d.7 | **Alternative external-balance and fiscal closures.** Beyond the fixed-trade-balance / lump-sum-recycling defaults already implemented: a flexible-trade-balance option and a government closure alternative (e.g. deficit-financed spending), each documented and switchable by config, with the standard correctness battery re-run under each | 1 wk |

**Standard scenario output set (DoD across 5d):** every CGE run — closed, open, or multi-region —
reports GDP, GVA, consumption, investment, employment, wages, a **relative cost-of-living /
GDP-deflator index** (review P2, round 10: NOT "inflation" — see below), trade, fiscal balance,
capital returns, emissions, and energy use as named result variables, each provenance-tagged and
each with an equation-level definition in `docs/models/cge-static.md`.

**"Inflation" corrected to a relative deflator (review P2, round 10).** An earlier draft of this
output set promised "inflation" as a standard per-run output. The model fixes the household's CPI
as the **numéraire** (Π pq^γ = 1, pinned by construction — see §4b.2 above, which already states
this correctly: "there is no separate inflation/deflator" for the CGE tier), so there is no
nominal/monetary anchor identifying an absolute price *level*, and therefore no absolute inflation
rate — only *relative* price movements (which good got more expensive relative to which, and
relative to the numéraire) are identified. Adding a government account, investment, or an energy
nest (5d.1/5d.2/5d.5) does not change this: none of them introduce a money-demand equation, a
central-bank reaction function, or a fixed money stock — the actual missing ingredient for
identifying a nominal price level. What the model CAN honestly report, and what "inflation" in
this output set actually means: a relative cost-of-living index (the wage-numéraire nominal GDP
reference already emitted by the closed/open CGE, or an analogous relative deflator), explicitly
labelled as relative, not an absolute inflation rate. If a genuine nominal anchor is wanted later,
that is the optional, clearly-labelled "illustrative macro-financial overlay" already scoped as
4b.5 — not a default output of every CGE run.
**Decisions forced:** government closure (balanced-budget default, recommend deficit-financed as
the documented alternative); investment closure (savings-driven default, per the original 5.2
plan); whether labour-market closure defaults to full employment or exposes unemployment as an
option (recommend: full employment default, unemployment as an explicit opt-in, clearly labelled).
**Risks:** this is real new equation-writing, not wiring — expect it to take the same "correctness
first" discipline as 5.0–5.3 (replication/homogeneity/Walras must still hold under every new
closure). The energy nest (5d.5) is the item most likely to be asked about by a CGE-literate
reviewer if skipped, so do not defer it in favour of the softer items (5d.6/5d.7).
**Depends on:** Phase 5 (5.0–5.3, done). **Unblocks:** Phase 7.1 (recursive dynamics — needs 5d.3's
capital-accumulation identity); Phase 7b (baseline/pathway harmonizer — needs 5d.1's government
account and 5d.5's energy nest to harmonize NGFS pathways against).

### Phase 6 — Nature extension via ENCORE (3–6 wk) — parallel with P4/P5

| # | Task | Effort |
|---|---|---|
| 6.1 | ENCORE ingestion: parse dependency ratings (production process × ecosystem service) and impact-driver ratings; map materiality classes to a documented numeric scale; version the snapshot | 2–3 d |
| 6.2 | ENCORE↔EXIOBASE concordance via the P1 framework, **seeded from published central-bank mappings** (DNB "Indebted to nature", ECB/EIOPA, World Bank) rather than built from scratch; document every weighting judgement | 1–2 wk |
| 6.3 | Exposure engine: direct dependency/impact scores per sector → upstream propagation through the Leontief inverse (reusing P2 machinery) → "good X depends on pollination/water/… directly and via inputs"; aggregation choice (max vs weighted mean materiality) exposed as a parameter, not buried | 1–2 wk |
| 6.4 | `NatureStress` shocks: degradation scenario → productivity shocks per sector/region scaled by dependency scores → fed to Engines 1/2/3 through the standard shock vocabulary; start from published scenario sets (NGFS nature scenarios, World Bank/PIK) | 1–2 wk |
| 6.5 | GUI: dependency/impact heatmaps (good × ecosystem service), supply-chain dependency drill-down, nature-scenario runner — within the P3 framework | 2–3 d |

**DoD:** for any good: ranked ecosystem-service dependencies (direct + upstream) and impact drivers, in the GUI; at least one `NatureStress` scenario runs end-to-end through an economic engine and produces a schema-valid `ResultSet`; **model doc exists** covering the propagation equations, the materiality→numeric scale, and the ENCORE↔EXIOBASE concordance with its published sources.
**Decisions forced:** the materiality→numeric scale (document it; it drives everything downstream); aggregation rule for propagated scores (max is conservative and defensible; weighted-mean is smoother — expose both); which published concordance to anchor on.
**Risks:** the concordance is judgement-heavy and the single biggest credibility surface here — cite sources per mapping and treat it as reviewable data, not code. Nature *scenario* design (6.4) is an open research question field-wide; lean on published scenarios and label outputs as illustrative, exactly as for damages in P7.
**Depends on:** P1 (concordance framework), P2 (propagation). GE-mode nature runs need P5, but 6.1–6.3 and 6.5 don't — **runs in parallel with P4/P5.** **Unblocks:** nature-related exposure and stress answers.

### Phase 6b — Physical nature-state and ecosystem-service scenario modelling (4–8 wk) — after P6

Phase 6 gives **exposure**: which sectors depend on which ecosystem services, and by how much.
Phase 6b adds the missing **state** layer: a physical model of the ecosystem services themselves,
so a "30% pollination decline" scenario is a modelled physical trajectory, not an assumed number
fed straight into 6.4's productivity shock. ENCORE stays the dependency/exposure layer (Phase 6)
unchanged; this phase sits upstream of it.

| # | Task | Effort |
|---|---|---|
| 6b.1 | **Physical ecosystem-service state variables.** For each service ENCORE scores (pollination, water, soil, forestry, fisheries, land-use, …), a documented physical state variable and its baseline level, sourced from published environmental accounts (not fabricated) | 1–2 wk |
| 6b.2 | **Spatial degradation and restoration pathways.** Scenario-driven trajectories for each state variable — degradation (deforestation, aquifer depletion, soil erosion) and restoration (reforestation, water-management investment) — spatially resolved where the source data supports it, aggregated to the model's regions otherwise | 1–2 wk |
| 6b.3 | **Channel-specific translation to productivity/input constraints.** A documented function per channel (water availability → agricultural yield; pollinator decline → crop-specific output; soil degradation → agricultural productivity; forestry/fisheries depletion → sectoral input availability) translating physical state into the `NatureStress` shocks Phase 6.4 already consumes | 1–2 wk |
| 6b.4 | **Thresholds, nonlinearities, and recovery assumptions.** Ecosystem degradation is not always linear (tipping points, slow recovery even after restoration) — expose threshold/nonlinearity and recovery-rate assumptions as first-class, documented parameters per channel, not hard-coded constants | 1 wk |
| 6b.5 | **Double-counting check against physical climate damages (Phase 7c).** Some nature degradation (e.g. heat-driven soil/water stress) is also a pathway in 7c's physical-risk channel library. A documented reconciliation rule (which channel owns which physical mechanism) and an automated check that a scenario invoking both does not apply the same physical effect twice — **this is a DoD criterion, not an afterthought**, since it is exactly the kind of gap a future review would catch | 1 wk |

**DoD:** a nature scenario specifies a physical degradation/restoration pathway (not a bare
productivity-shock number); it propagates through 6b.3's translation into the existing `NatureStress`
shock vocabulary and runs end-to-end through an economic engine; the double-counting check (6b.5)
passes for any scenario combining nature and climate-damage shocks; model doc covers every channel's
translation function with citations.
**Decisions forced:** which channels get a full physical-state model first (recommend: water and
agriculture — best-sourced published data); threshold/nonlinearity defaults (recommend: linear
default, documented nonlinear option per channel, not silently assumed away).
**Risks:** same as 6.4 — nature *scenario* design is a live research question field-wide; lean on
published pathways (IPBES, WWF, national ecosystem accounts) and label outputs as illustrative.
The double-counting check (6b.5) is easy to skip under schedule pressure and is the single most
likely thing a reviewer will ask about if this phase ships without it.
**Depends on:** P6 (ENCORE exposure/concordance — 6b feeds *into* 6.4's shock vocabulary, doesn't
replace it). **Unblocks:** physically-grounded nature scenarios; the double-counting reconciliation
also depends on 7c existing (or at least its channel list being drafted) to know what to check
against.

### Phase 7 — The pathway stack: "a CGE that speaks IAM" (6–12 wk for 7.1–7.3, then ongoing)

A true process IAM (GCAM/REMIND-class) is a multi-year team effort; the achievable and standard alternative — used by most financial-sector "IAM-based" tools — is to give the CGE dynamics and consume published pathways.

| # | Task | Effort |
|---|---|---|
| 7.1 | **Recursive-dynamic wrapper.** Solve any `general_equilibrium` engine year-by-year to 2050, updating capital (savings/investment → next-year capital stock with depreciation), labour (exogenous demographics), and productivity (exogenous trend) between static solves. No perfect foresight — dynamics are bookkeeping between solves, not a new solution concept. **Needs Phase 5d.3's capital-accumulation identity** — without it there is no real savings/investment mechanism to update capital from, and this would be imposed bookkeeping rather than a modelled decision | 4–8 wk |
| 7.2 | **NGFS scenario reader.** Adapter: open NGFS database (IIASA) → shock paths in the standard vocabulary (carbon price path, GDP/population trajectories per region) for Net Zero 2050 / Delayed Transition / Current Policies etc. Transition intelligence is inherited from the process IAMs that built the scenarios; our model adds sector/supply-chain resolution. **Raw NGFS pathways feed Phase 7b's harmonizer** before driving the model — imposing NGFS's own GDP/population path directly while also reporting GDP as an endogenous CGE result double-counts the same variable; 7b reconciles this | 1–2 wk |
| 7.3 | **Climate module (FaIR).** One-way coupling behind the climate slot: model emissions in → temperature path out, reported alongside economic results | 1–2 wk |
| 7.3b | **Temperature-target back-solve.** Given a temperature (or carbon-budget) target, invert the forward chain (carbon price → emissions → FaIR → temperature) to find the carbon-price path that hits it, then run forward for the sector impacts. **Corrected design (review P1, 2026-07): a single target is one scalar constraint, not enough to determine an entire multi-year price path** (per-year temperatures are dynamically coupled through cumulative emissions and climate inertia, not independent scalar roots) — so this is a genuine 1-D root-find only when it solves for **one scalar that scales a predetermined, documented price-path shape** (an NGFS trajectory or a parametric form), with an *empirically checked* monotonicity precondition (not assumed — recycling/leakage/substitution/rebound can break it for a given scenario configuration); flags infeasible or non-monotone cases rather than returning garbage. Choosing the path shape *itself* via optimisation is a distinct, harder follow-up, out of scope here. See `docs/energy-and-temperature-plan.md`. | 2–3 wk |
| 7.4 | **Damage feedback (optional, handle with care).** Temperature → damage function → productivity shocks fed back as standard shocks. **Distinct from 7.3b:** 7.3b uses only emissions→temperature (well-established); 7.4 adds temperature→economy through a damage function (the most contested object in climate economics — order-of-magnitude disagreement). Implement only with published functions (DICE, Burke–Hsiang–Miguel), labelled as scenario illustrations, with the choice surfaced as a first-class assumption in the GUI | 1–2 wk plumbing |
| 7.5 | **Ongoing hardening.** Second data source (FIGARO/ICIO) as a new adapter to test data sensitivity; more households/regions; scenario library + cross-run comparison in the GUI; job queue if runs get heavy; API layer if others need programmatic access | ongoing |

**DoD (7.1–7.3b):** pick an NGFS scenario OR a temperature target in the GUI → the recursive-dynamic engine produces a 2020→2050 path of sector prices/volumes per region, with the associated (or target-hitting) carbon-price and temperature paths, all provenance-tagged.
**Known limits even when complete:** no endogenous technological learning; energy as CES aggregates, not discrete technologies; no land-use module. The model traces the sectoral consequences of pathways others generate; it does not generate novel pathways. Say so in the docs.
**Depends on:** P5 for GE-mode dynamics (7.2 and 7.3 can be built earlier against Engines 1–2); 7.1 additionally needs 5d.3 (capital accumulation).

### Phase 7b — Baseline construction, pathway harmonization, and macroeconomic consistency (6–10 wk) — after 7.2, needs 5d

A scenario needs a credible counterfactual **baseline**, not just a sequence of shocks laid on top
of the benchmark year. Phase 7.2 gives raw access to NGFS's own trajectories; this phase reconciles
them with the model's own country/sector detail instead of naively overwriting it, and is the
direct consulting-grade upgrade from "we ran a shock on the benchmark SAM" to "we ran a shock on a
credible reference pathway."

| # | Task | Effort |
|---|---|---|
| 7b.1 | **Country and sector baselines.** Reconcile the model's benchmark-year SAM trajectory forward against published country/sector forecasts (IMF WEO, OECD Economic Outlook, national statistical-office projections where available); document the reconciliation method and every source per country | 1–2 wk |
| 7b.2 | **Structural trajectories.** Documented, sourced paths for structural change (sectoral GDP-share drift), productivity growth, population, labour-force participation, and emissions intensity — the exogenous drivers 7.1's recursive wrapper needs between static solves, replacing ad hoc "flat trend" assumptions | 1–2 wk |
| 7b.3 | **NGFS downscaling.** A transparent, documented method to downscale NGFS's coarse regions/sectors onto the model's finer EXIOBASE-shaped region/sector grid (e.g. GDP-share or historical-pattern based); the downscaling weights are provenance-tagged data, reviewable like the ENCORE concordance in Phase 6 | 1–2 wk |
| 7b.4 | **Reconciliation rules (the core deliverable).** A documented rule set so NGFS's own GDP/population path is not simultaneously **imposed** as an exogenous driver and **reported** as an endogenous CGE result for the same variable — pick which variables are exogenous inputs (population, and typically productivity trend) vs which the CGE computes (GDP, sectoral output, trade), and make that split an explicit, auditable assumption per scenario, not an implementation accident | 1–2 wk |
| 7b.5 | **Energy-mix, electrification, and technology-cost pathways.** Documented, sourced trajectories for the energy mix, electrification rate, and technology cost curves implied by each NGFS scenario, feeding the energy nest from Phase 5d.5 — this is the piece that actually makes "a CGE that speaks IAM" (Phase 7's own framing) speak the *energy-transition* part of IAM output, not just the carbon-price part | 1–2 wk |
| 7b.6 | **Automated cross-variable consistency checks.** A standing check (in the same spirit as `cge.validation`) that GDP, energy, emissions, carbon price, and temperature stay mutually consistent across a harmonized scenario — catches, e.g., a downscaled energy pathway that implies emissions inconsistent with the scenario's own carbon-price/temperature path | 1 wk |

**DoD:** running an NGFS scenario produces a harmonized baseline (not the raw NGFS numbers pasted
in, nor the model's benchmark year naively extrapolated) with every reconciliation rule, downscaling
weight, and structural-trajectory source documented and provenance-tagged; the consistency checks
(7b.6) run automatically and fail loudly on an internally inconsistent harmonized scenario.
**Decisions forced:** which variables are exogenous drivers vs endogenous CGE outputs (7b.4 — the
single most consequential judgement call in this phase, make it explicit); which published baseline
source to anchor country forecasts to when they disagree (recommend: IMF WEO as the default, with
the choice documented and switchable).
**Risks:** downscaling and reconciliation are judgement-heavy, similar in character to the ENCORE
concordance (Phase 6.2) — treat every weight and rule as reviewable data with a cited source, not
buried logic. This phase has the highest "looks precise, isn't" risk in the whole roadmap if the
reconciliation rules aren't made explicit and auditable.
**Depends on:** Phase 7.2 (NGFS reader — raw pathways to harmonize), Phase 5d (government account
+ energy nest to harmonize NGFS's fiscal and energy-transition content against — low-value without
them), and **§5.1b** (a live multi-region SAM — country/sector harmonization on the current
hand-built supplied SAM would not be credible; review P2, round 10). **Unblocks:** credible
baselines for every downstream pathway scenario; Phase 7c's driver decomposition (needs a
harmonized baseline to decompose deviations *from*).

### Phase 7c — Physical-risk channels, adaptation, uncertainty ensembles, and driver attribution (6–10 wk) — after 7b, P6b

Two related consulting-grade upgrades: (1) the generic temperature-damage function from 7.4 is too
aggregated for hazard-specific consulting scenarios — replace it with a channel library; (2) the
elasticity sweeps from Phase 5.3 are a good start on uncertainty but are single-parameter and don't
attribute *why* a result moved.

| # | Task | Effort |
|---|---|---|
| 7c.1 | **Physical-risk channel library.** Hazard-specific channels replacing 7.4's single generic damage function: heat-related labour productivity, capital destruction and accelerated depreciation (feeds Phase 5d.3), agriculture and water availability (shares a translation layer with Phase 6b.3 — see the double-counting check there), energy demand and generation efficiency, transport/infrastructure disruption. Each channel published-function-based and labelled illustrative, exactly as 7.4 already requires | 2–3 wk |
| 7c.2 | **Adaptation costs and avoided damages.** For each channel in 7c.1, a documented adaptation-investment option (competes with Phase 5d.6's transition investment) and the avoided-damage accounting it implies, so an adaptation scenario reports both the cost and the avoided loss, not just one side | 1 wk |
| 7c.3 | **Parameter ensembles and scenario ranges.** Beyond Phase 5.3's single-elasticity sweeps: ensembles over damage-function choice, elasticity sets, and data vintage, run as a documented probability-free range (not a false-precision probability distribution) | 1–2 wk |
| 7c.4 | **Uncertainty decomposition and sensitivity ranking.** Report model uncertainty (parameter/structural choice), data uncertainty (vintage/source), and scenario uncertainty (which NGFS/damage scenario) **separately**, with a sensitivity ranking of which inputs move the result most; quantile outputs where the ensemble supports them | 1–2 wk |
| 7c.5 | **Driver decomposition.** Decompose a scenario's deviation from its Phase 7b baseline into carbon-policy, energy-transition, physical-climate, and nature contributions, with explicit interaction terms where effects don't add independently (flagged, not silently summed) — this is the client-presentation deliverable the rest of 7c exists to support | 1–2 wk |

**DoD:** a scenario report includes a driver-decomposition chart (policy / energy / climate / nature
+ interaction terms) that sums to the total deviation from baseline; every physical-risk channel in
7c.1 has a model doc entry with its source function cited; sensitivity rankings and the
model/data/scenario uncertainty split are standard scenario outputs, not a bespoke one-off analysis.
**Decisions forced:** how to present interaction terms when effects are materially non-additive
(recommend: report the interaction as its own labelled term rather than allocating it to one driver).
**Risks:** same character as Phase 7.4 — published damage functions disagree by an order of
magnitude; treat every channel as a **labelled scenario illustration**, not a forecast, and say so
as loudly as 7.4 already does. Coordinate with Phase 6b.5's double-counting check so a scenario using
both nature-state and physical-climate channels doesn't apply the same physical mechanism twice.
**Depends on:** Phase 7b (a baseline to decompose deviations from); Phase 6b (the nature-state
channels the double-counting check needs to know about). **Unblocks:** hazard-specific consulting
scenarios; the driver-attribution deliverable Phase 8's client reporting packages consume directly.

### Phase 8 — Consulting delivery, model governance, and validation (13–23 wk solo FTE) — after 7b/7c

Everything through Phase 7c makes the platform a genuinely capable modelling engine. Phase 8 is
what converts that into something a consultancy can put its name behind in a paid engagement: **(1)
empirical evidence that the model's outputs are credible**, and **(2) the delivery/governance
infrastructure a real engagement needs.** These are split into two workstreams within the phase
because they have different risk profiles — validation is a research/credibility risk, delivery is
an engineering/process risk — but both gate "ready for paid client work," so they're one phase.

**8a. Empirical calibration and validation**

| # | Task | Effort |
|---|---|---|
| 8a.1 | **Historical hindcasts.** Re-run the model against an earlier data vintage and compare its projected trajectory to what actually happened, for at least one carbon-price or trade-shock episode with a documented real-world outcome | 1–2 wk |
| 8a.2 | **Calibration against observed sector responses.** Compare modelled sector responses to energy-price, trade, and carbon-price shocks against published empirical estimates (not just theory-consistent direction, which the standing validation suite already checks) | 1–2 wk |
| 8a.3 | **Cross-model comparison.** Compare headline results against GTAP-based published studies, NGFS's own macro outputs, and at least one other established model, on a shared scenario where possible; document where and why results diverge | 1–2 wk |
| 8a.4 | **Country/sector calibration targets with acceptance thresholds.** Documented, sourced target values (not just "plausible sign") and an explicit tolerance band per target, so a calibration run can pass/fail against a stated bar rather than a subjective read | 1 wk |
| 8a.5 | **Governed parameter registry.** Every elasticity, damage-function choice, and assumed share gets a registry entry: source, confidence level, applicable geography/sector, and a review date — the single source of truth `RunManifest.assumptions` should trace back to, replacing scattered per-module defaults | 1–2 wk |
| 8a.6 | **Independent model validation and a formal report.** A written validation report (methodology, results, limitations) reviewable by someone who did not build the model; documented limitations and an approved-use boundary **per module**, not just for the platform as a whole | 1–2 wk |

**8b. Consulting engagement and governance layer**

| # | Task | Effort |
|---|---|---|
| 8b.1 | **Engagement workspaces.** Frozen data build, assumption set, and model/engine version per engagement, so a client deliverable is reproducible after the platform itself has moved on | 1–2 wk |
| 8b.2 | **Client-specific classifications, concordances, and expert overrides.** Client sector/region mappings distinct from the platform's default concordances; expert overrides to any calibrated parameter that are **visible and versioned** (an override is data, reviewable like the ENCORE concordance, never a silent code edit) | 1–2 wk |
| 8b.3 | **Approved scenario packs and locked reference scenarios; batch comparison.** A curated, sign-off-gated set of reference scenarios per engagement; batch scenario comparison in the GUI (reusing Phase 7.5's scenario-library/cross-run-comparison groundwork) | 1–2 wk |
| 8b.4 | **Review, approval, and sign-off workflow.** A tracked state machine (draft → reviewed → approved) for scenario packs and deliverables, with an audit trail of who approved what and when | 1 wk |
| 8b.5 | **Deliverable generation.** Excel workbooks, report-quality charts, and an automatically-generated methodology appendix (assembled from the existing per-engine model docs and the run's own assumption dump — not hand-written per engagement); reproducible tables formatted for PowerPoint/Word | 1–2 wk |
| 8b.6 | **API, auth, RBAC, client-data isolation, audit logs.** Programmatic access (building on Phase 7.5's optional API layer) with authentication, role-based access control, per-client data isolation, and an audit log of who ran/viewed what | 2–3 wk |
| 8b.7 | **Commercial licensing and data-redistribution review.** A documented review of the licensing terms for every third-party data source in use (EXIOBASE, ENCORE, NGFS, and any future source) against the platform's intended commercial use and redistribution model — this is a legal/compliance gate, not an engineering task, but it blocks paid delivery if skipped | 1 wk, legal review |

**FTE vs elapsed time (review P2, round 10):** 8a totals ~6–11 wk solo FTE, 8b totals ~7–12 wk solo
FTE — **13–23 wk solo FTE combined**, which is what the phase header states. 8a (validation) and
8b (delivery) are genuinely independent workstreams with different risk profiles (research vs
engineering), so with a **second person** working 8b while the first works 8a, **elapsed time**
could compress toward the *larger* of the two workstreams (~7–12 wk) rather than their sum — but
that reduces calendar time for a team, not the ~13–23 wk of total work a solo builder still has to
do.

**DoD:** a validation report exists and is reviewable by someone outside the build; every
calibration target has a documented source and tolerance; an engagement can be stood up as a frozen
workspace, produce an approved scenario pack, and generate a client-ready deliverable (workbook +
charts + methodology appendix) without hand-editing the platform's own code or docs; the licensing
review is signed off for every data source actually in use.
**Decisions forced:** how much hindcast/cross-model-comparison evidence is "enough" before calling
the model validated for a given use case (recommend: document the acceptance bar per module up
front, in 8a.4, rather than deciding case-by-case under client pressure); build vs buy for
auth/RBAC (recommend: buy — this is not the platform's differentiator).
**Risks:** 8a is a genuine research effort, not a checklist — expect hindcasts and cross-model
comparisons to surface real disagreements, not just confirm the model. Treat that as the point, not
a failure: a validation report that only ever says "matches" is not credible. 8b.7 (licensing) is
easy to treat as a formality and is the one item here that can block commercial delivery entirely
if it surfaces a real restriction late.
**Depends on:** Phase 7b/7c (the model needs a harmonized baseline and driver attribution before a
hindcast or cross-model comparison is a fair test). **Unblocks:** paid consulting engagements.

---

## 3. Dependency graph

```
P0 ─▶ P1 ─▶ P2 ─▶ P3 (GUI v1)
            │
            ├──▶ P4 ─▶ P4b ─▶ P5 ─▶ P5d ─▶ P7 (7.1 needs 5d.3; 7.2 feeds 7b)
            │         (macro aggregates:         │
            │          GVA/GDP/deflator,         └─▶ P7b ─▶ P7c ─▶ P8
            │          real vs nominal;              (baseline/    (physical      (validation +
            │          native + exact in P5;          harmonizer,   channels,      governance +
            │          GDP *growth* needs 7.1)         needs P5d)   uncertainty,   delivery)
            │                                                       attribution,
            │                                                       needs P6b)
            └──▶ P6 ─▶ P6b (nature-state layer; P6b.5 double-counting check needs P7c)
                 (parallel with P4/P5; GE-mode nature runs need P5)
```

## 4. Effort & milestones

| Milestone | Cumulative FTE | You can… |
|---|---|---|
| End P3 | ~6–9 wk | Browse data & quality in the GUI; get supply-chain **cost** impacts of any carbon price on any good, with decomposition |
| End P4 | ~8–13 wk | Add production-**volume** responses (finite-change demand → Leontief propagation) with explicit uncertainty ranges |
| End P4b | ~9–15 wk | **GVA per sector/country, GDP per country, and a deflator** per time-step, in **real and nominal** terms (indicative PE tier; native and exact in the CGE) |
| End P6 (skipping P5) | ~11–19 wk | Nature dependency/impact exposure of any good, incl. via its supply chain, plus nature stress runs |
| End P5 | ~16–25 wk | General-equilibrium price + volume answers with carbon-tax revenue recycling (Armington/CET open economy + true multi-region bilateral trade — no government account, investment, or energy nest yet: see P5d; and the multi-region SAM is still hand-built, not live-EXIOBASE: see §5.1b) |
| End §5.1b | ~19–30 wk | A live-EXIOBASE-driven multi-region SAM build (satellite alignment, FD attribution, trade-materiality handling, topology validation, live-data replication gate) — the data-side prerequisite Phase 7b/8 need before they can credibly operate on multi-region output |
| End P5d | ~27–42 wk (revised — review P2, round 10: Phase 5d's own header corrected from 4–8 wk to 8–12 wk solo FTE to match its task total) | A government/fiscal account, savings-investment with capital accumulation, an energy nest, and the full standard scenario output set (GDP, GVA, consumption, investment, employment, wages, fiscal balance, capital returns, and a relative cost-of-living index — not "inflation", which the CPI-numéraire closure cannot identify) — the macro closure Phase 5 originally specified |
| End P6b | ~34–56 wk | Physically-grounded nature scenarios (degradation/restoration state, not a bare productivity-shock number) |
| Full incl. P7 pathway stack | 6–12 months | NGFS-driven dynamic pathways to 2050 with temperature reporting, multi-source data, scenario library |
| Full incl. P7b/7c | +4–5 months beyond P7 | Harmonized country/sector baselines reconciled to published forecasts; hazard-specific physical-risk channels; uncertainty ensembles with model/data/scenario decomposition; client-presentable driver attribution |
| Full incl. P8 | +3–5.5 months beyond P7c (revised — review P2, round 10: Phase 8's own header corrected from 8–14 wk to 13–23 wk solo FTE; a second person on 8b delivery while the first does 8a validation could compress **elapsed** time toward ~7–12 wk, not the total work) | Empirically validated (hindcasts, cross-model comparison, governed parameter registry) and consulting-delivery-ready (engagement workspaces, approval workflow, client deliverables, API/RBAC, licensing sign-off) |

Sequencing note: P3, P4, and P6a all deliver standalone value and can be reordered to taste. If forced to choose a minimal useful product, **P0→P1→P2→P3→P4→P6a** (cost + volumes + nature exposure, no CGE) is the highest value-per-week path and defers the hardest work. Within the consulting-grade extensions (§5.1b/P5d/P6b/P7b/P7c/P8), **P5d is the highest-priority single item** — it is reopened Phase 5 debt (not new scope), and P7b/P8's value is materially lower without a government account and energy nest to harmonize pathways against and validate. **§5.1b (the live multi-region SAM build) is the second priority** — P7b's harmonization and P8's empirical validation cannot credibly run against the current hand-built supplied multi-region SAM.

### Planned scenario-input extensions (detail in `docs/energy-and-temperature-plan.md`)

Two requested capabilities that fit the existing seams rather than adding new phases:

- **Country-level energy prices (✅ implemented).** An optional `EnergyPrice` shock — a
  per-country, per-carrier (coal / oil-gas / electricity) output-price change, applied *in
  addition to* a carbon price. The carrier's own price is **pinned** to the requested change (an
  exogenous boundary condition), and that price propagates downstream through the same Leontief
  inverse (reusing Engine 1/2 machinery). Available in Engine 1/2 and the GUI; richer in the CGE
  (Phase 5), where it would trigger KL-E-M substitution.
- **Temperature-target back-solving (Phase 7).** Given a temperature (or carbon-budget) target,
  invert the forward chain to find a scale factor on a **predetermined price-path shape** that
  hits it, then run forward for the sector impacts — the credible, target-driven "IAM-ish" mode.
  Added as roadmap task **7.3b** above; needs the Phase 7 climate module (FaIR) + recursive
  dynamics. A single target is one scalar constraint — it determines a scale factor on a
  documented path shape, not an arbitrary multi-year price path outright (see the corrected
  design in `docs/energy-and-temperature-plan.md`). Distinct from — and much more defensible
  than — temperature→economy *damage* feedback (7.4), which stays optional and illustrative for
  scientific reasons (damage-function disagreement).

---

## 5. Cross-cutting concerns

- **Testing & a standing validation suite.** Every engine ships with: analytic tests on the toy economy (P0.5), at least one known-answer test against published numbers, and identity checks (IO balance, CGE homogeneity/Walras). Beyond code-level unit tests, there is a **model-validation suite** (`cge.validation`, run by `scripts/validate.py` / `cge validate`, gated in CI) — a standing, human-readable audit that each model still reproduces its known answers and economic identities. Each engine/module adds a suite whose checks map to the properties stated in its model doc, preferring assertions against theory or published numbers over "looks plausible". Established with Engine 1 in Phase 2; see `docs/validation.md`. Being a DoD criterion, no engine is "done" without its suite.
- **Data correctness & pipeline consistency (enforced, not just tested).** Correctness and plausibility checks run at *every* stage that transforms data, and cross-stage invariants are verified *inside the pipeline* so bad data fails loudly at build time rather than surfacing as a wrong result later. Two tiers: **structural invariants** that must hold (finite values, aligned labels, square/productive `A`, existing Leontief inverse) raise and abort the build; **plausibility checks** (positive output, non-negative intensities, coverage, RoW share) and **cross-stage conservation** (aggregation preserves total output and final demand) become `QualityCheck`s in the stored `QualityReport`, with conservation failures treated as fatal. Established in Phase 1 (`cge.data.quality.consistency`); every later data transformation (SAM balancing P5, ENCORE concordance P6, NGFS ingestion P7) adds its own gate the same way.
- **Provenance everywhere.** No result exists without its data version, engine version, scenario hash, and assumption dump. This is what lets the tool be trusted and compared across runs.
- **Documentation as a deliverable — to equation level, with citations.** Every engine, module, and non-trivial data transformation ships a **model-description doc** stating the method *to equation level* (numbered equations, well-posedness argued) and citing the peer work it derives from (papers/textbooks in `docs/references.md`, institutional reports for applied choices). This is a **definition-of-done criterion** per phase, not optional — it is what makes results defensible to reviewers who know the field. The standard is `docs/documentation-standard.md`; the worked example is `docs/models/io-price-model.md`. Each doc's assumptions must match the engine's `RunManifest.assumptions`. ADRs record cross-cutting *why* (data format, closures, concordance choices).
- **Assumptions visible in the GUI.** Every run page prints the assumptions behind its numbers. For a screening/stress tool this is the single highest-leverage credibility feature.
- **A learning-path user guide (`docs/user-guide.md`) — kept current with each engine.** Distinct from the equation-level model docs (which target a CGE-literate reviewer): a *slow, hands-on* walkthrough that leads a new operator from the simplest run to the full feature set, **explaining the methodology conceptually** (what a Leontief inverse is, why a numéraire matters, what revenue recycling / carbon leakage / a double dividend *mean*) alongside **practical examples the reader runs themselves** after each step. It is a deliverable, not an afterthought: **every engine/feature that lands adds its guided step + a runnable example scenario before the phase is done.** The Phase 5 CGE arc (closed run → revenue recycling → open economy + carbon leakage → CES substitution → elasticity sweeps → multi-region bilateral trade) is done. **Owed, at the appropriate point in each new phase:** a 5d step (government account, savings/investment, the energy nest — what a fiscal closure and a KL–E–M nest mean conceptually, with a runnable example once implemented); a 6b step (physical nature-state scenarios, distinct from the existing 6.x ENCORE-exposure step); a 7b step (what a harmonized baseline is and why NGFS's raw numbers aren't run as-is); a 7c step (reading a driver-decomposition chart, what the uncertainty/sensitivity outputs mean); Phase 8 is delivery infrastructure, not new modelling — it does not need a guide arc of its own, but any new client-facing output format it introduces should get a short "how to read this deliverable" note.
- **Versioning.** Contracts are semver-tagged; **scenarios** are content-hashed (`scenario_hash`); **data builds** have deterministic human-readable ids (source-version-year-aggregation) and record their reference year and aggregation in provenance. Results record the build id, scenario hash, and versions that produced them. (Content-hashing data builds too — so identical inputs collide and changed inputs diverge — is a tracked improvement, not yet done.)

---

## 6. Honest feasibility assessment

**Very feasible (high confidence):** Phases 0–3 and 6a. The IO price model plus GUI is a genuinely useful, defensible tool, buildable by one person in ~2 months FTE, and is essentially what several central banks and financial institutions use for first-pass carbon-price and nature-exposure stress. ENCORE exposure scoring through supply chains is equally standard. Stopping after P3+P4+P6a already yields something consultancies charge real money for.

**Feasible but real work (medium confidence):** Phases 4 and 5. The mechanics of a small static CGE in pyomo are well-trodden; the risks are not coding:
- **Calibration data.** GTAP is closed; building a balanced SAM from EXIOBASE is done in the literature but takes iteration, and EXIOBASE's value-added/tax detail is thinner than GTAP's. Expect the SAM step to run long.
- **Elasticities.** No clean open elasticity database exists. You assemble values from papers; results (especially volumes) are sensitive to them. Mitigation: uncertainty ranges as first-class outputs, Engine 2 as a permanent cross-check.
- **Credibility ceiling.** A solo-built 40-sector CGE gives *indicative, directionally sound* volume responses, not defend-to-three-decimals numbers. Fine for a screening/stress instrument — which is what you described — but it won't match a GTAP-based model on precision, and CGE-literate reviewers will ask about closures and elasticities. Printing assumptions per run buys a lot of that back.

**Genuinely hard / manage expectations:**
- **A "real" process IAM** (endogenous energy system, dynamics, damage feedbacks à la GCAM/MESSAGEix/REMIND) is a multi-year, multi-person effort. The realistic "IAM" here is P7's recursive-dynamic CGE plus exogenous NGFS pathways — achievable, and what most financial-sector "IAM-based" tools actually are.
- **Nature stress *scenarios*** (6.4) and **damage feedbacks** (7.4): the plumbing is easy; deciding what "pollination declines 30%" or "2°C costs Y%" means quantitatively is a live research question. Lean on published scenario/damage sets rather than inventing shocks, and label outputs as illustrative.
- **Scope creep** is the top schedule risk. The modular design only pays off if each engine stays small. The single best decision available is to hold the P5 CGE to "toy but honest" for v1.

**The consulting-grade extensions (P5d, P6b, P7b, P7c, P8) are a second project, not a tail on the
first.** Everything through P7 is, at heart, one person building a correct, well-documented,
"toy but honest" model — feasible solo, on the timelines above. P5d/P6b/P7b/P7c/P8 are a different
kind of effort: they trade code volume for **empirical/data credibility work** (calibration
targets, hindcasts, cross-model comparison, harmonization judgement calls) and **delivery
infrastructure** (auth, RBAC, workflow, licensing) that a research-grade solo build does not
usually need. Be explicit about this distinction when scoping: shipping P7 does not mean P8 is
"just polish" — it is the difference between a research tool and a fee-earning deliverable.
- **P5d is feasible, medium confidence** — same character as P5 itself (well-trodden CGE mechanics, the risk is calibration/closure judgement, not code) and is genuinely a Phase 5 completion, not new risk.
- **P6b and P7c's physical channels are feasible, medium confidence, same caveat as P6.4/7.4** — the plumbing is straightforward; picking defensible channel-specific translation functions is the real work, and the honest answer is "lean on published functions, label as illustrative," same as the existing damage-function guidance.
- **P7b is feasible but judgement-heavy** — most similar in character to the ENCORE concordance (P6.2): not hard engineering, but every downscaling weight and reconciliation rule is a reviewable judgement call that will be scrutinized by a CGE-literate client reviewer. Treat it with the same "document every source, treat weights as data" discipline as P6.2, not as a data-plumbing task.
- **P8's validation workstream (8a) is genuinely hard, manage expectations directly.** Hindcasting and cross-model comparison for a solo-built model will likely surface real disagreements with GTAP-based studies and NGFS's own macro outputs — that is the point of doing it, not a sign it went wrong, but it means 8a needs to be scoped and communicated as an open-ended credibility investment, not a checklist with a fixed end date. **P8's delivery workstream (8b) is ordinary engineering** (auth, workflow, document generation) and should not be over-scheduled relative to 8a.

**Bottom line.** The tool is buildable as designed: ~2 months FTE to a useful GUI-driven cost tool, ~4–6 months FTE to volumes + nature exposure + a simple CGE, with the main risks in data (SAM balancing, elasticities) rather than software. Reaching genuinely defensible, paid-consulting-ready output (through P8) is a further ~6–12 months FTE beyond that, weighted toward the empirical-validation and harmonization judgement calls, not new code volume. The phrase to hold onto: *precise about costs, indicative about volumes, transparent about assumptions* — and, for the consulting-grade tier, *validated where we've checked, and honest about where we haven't.*