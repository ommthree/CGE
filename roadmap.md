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
2. **Typed shock vocabulary.** Scenarios are declarative files (YAML) composed of typed shocks: `CarbonPrice`, `EnergyPrice`, `ProductivityShock`, `DemandShift`, `TradeCost`, `NatureStress`, … each optionally a *time path*. Engines declare which shock types they understand; static engines take year-slices of a path. This is the key seam: the nature module, NGFS reader, and damage module all *emit shocks* in this vocabulary rather than talking to engines directly — so any future stress type (litigation risk, pandemic, tariff war) is a new shock class plus zero engine changes. (`EnergyPrice` — an exogenous per-country, per-carrier energy-cost change — is a planned near-term addition; it enters the cost vector exactly like a carbon cost and composes with it. See `docs/energy-and-temperature-plan.md`.)
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

### Phase 5 — Engine 3: simple static CGE (6–12 wk) ⚠ the hard part — 🔶 PILOT IMPLEMENTED (5.0 + 5.2a)

> **Progress:** the **solver gate (5.0)** and the **correctness-first 2-sector pilot (5.2a)** are
> built and green. `cge.engines.cge_static` calibrates to a hand-checkable balanced SAM and passes
> the standard CGE battery — **benchmark replication** (to machine precision), **homogeneity**, and
> **Walras' law** — plus theory-consistent carbon-price responses (dirty output falls, dirty price
> rises, real GDP falls). Solver abstraction: IPOPT via pyomo when present, scipy fallback so CI
> needs no binary; a non-optimal solve raises. Model doc: `docs/models/cge-static.md`.
>
> **Remaining for the full phase:** a real balanced **EXIOBASE SAM** (5.1b, the rabbit hole),
> **Armington/CET** open economy, **multiple regions**, **revenue recycling** + the carbon-price
> experiment (5.3), and the GE tier of the macro aggregates (native GVA/GDP/CPI + capital rental
> rate). The pilot is the provable core those build on.

> **Detailed plan: [`docs/phase-5-plan.md`](docs/phase-5-plan.md)** — solver-first sequencing,
> equation-level model structure, SAM balancing with an audit trail, the standard CGE
> correctness test battery (replication / homogeneity / Walras) plus cross-engine consistency
> checks, and honest scope. The outline below is the summary; the plan is the spec.

**5.0 Solver & environment (2–4 d) — do first.** pyomo needs a nonlinear solver (IPOPT). It is
not currently installed; the plan chooses IPOPT (via `idaes` binary) with a pure-Python scipy
fallback so CI stays solver-independent on the 2-sector toy. A solver abstraction records the
solver + termination status; a non-optimal solve raises (never returns numbers).

**5.1 SAM construction (2–4 wk)**
- Map EXIOBASE flows into SAM accounts: activities, commodities, factors (labour, capital), representative household, government, savings/investment, rest-of-world — at the small build's aggregation (~30–50 sectors; start with **one region + RoW**, multi-region after the pilot works).
- Fill EXIOBASE's thin spots (taxes less subsidies, margins, income flows) with documented assumptions; balance with RAS or cross-entropy; emit a SAM-specific `QualityReport` (imbalance before/after, adjustment magnitudes).
- **DoD:** balanced SAM reproducing EXIOBASE aggregates within tolerance, with an adjustment audit trail.

**5.2 Model core (2–4 wk)**
- Static CGE in pyomo/IPOPT: nested CES production (KL–E–M nesting so carbon pricing can shift the energy nest), Armington imports / CET exports, household demand (Cobb-Douglas first, LES if needed), government with carbon-tax revenue and recycling options (lump-sum vs labour-tax cut), investment, standard closure choices (recommend: savings-driven, fixed trade balance, numéraire = CPI), square-model and degrees-of-freedom checks.
- Pilot single-region model first; extend to multi-region only once the pilot passes 5.3's tests.
- **DoD:** model solves from the SAM; equation/variable count documented; closures switchable by config.

**5.3 Calibration & credibility tests (2–4 wk)**
- Benchmark replication: with zero shocks the model reproduces the base-year SAM to machine precision (the standard CGE correctness test).
- Homogeneity (doubling all prices changes nothing real) and Walras' law (one market clears residually) checks in CI.
- Elasticity sensitivity sweeps; comparison of carbon-price responses vs Engines 1–2 (prices should bracket, volumes should be same-sign) and vs published CGE carbon-price results for similar economies.
- **DoD:** replication + homogeneity + Walras tests green in CI; a short model-description document (equations, closures, elasticities and their sources) exists — this is what CGE-literate reviewers will ask for.

**Decisions forced:** nesting structure; closure defaults; single- vs multi-region sequencing (recommendation above); how much tax detail to fabricate vs omit.
**Risks:** SAM balancing is a known rabbit hole — timebox it and document rather than perfect; IPOPT convergence (mitigate: good starting values = benchmark data, gradual shock ramping); results sensitive to elasticities (mitigate: sweeps + Engine 2 cross-check, keep "toy but honest" framing).
**Depends on:** P1, P4 (elasticities). **Unblocks:** GE price *and* volume answers; P7.

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

### Phase 7 — The pathway stack: "a CGE that speaks IAM" (6–12 wk for 7.1–7.3, then ongoing)

A true process IAM (GCAM/REMIND-class) is a multi-year team effort; the achievable and standard alternative — used by most financial-sector "IAM-based" tools — is to give the CGE dynamics and consume published pathways.

| # | Task | Effort |
|---|---|---|
| 7.1 | **Recursive-dynamic wrapper.** Solve any `general_equilibrium` engine year-by-year to 2050, updating capital (savings/investment → next-year capital stock with depreciation), labour (exogenous demographics), and productivity (exogenous trend) between static solves. No perfect foresight — dynamics are bookkeeping between solves, not a new solution concept | 4–8 wk |
| 7.2 | **NGFS scenario reader.** Adapter: open NGFS database (IIASA) → shock paths in the standard vocabulary (carbon price path, GDP/population trajectories per region) for Net Zero 2050 / Delayed Transition / Current Policies etc. Transition intelligence is inherited from the process IAMs that built the scenarios; our model adds sector/supply-chain resolution | 1–2 wk |
| 7.3 | **Climate module (FaIR).** One-way coupling behind the climate slot: model emissions in → temperature path out, reported alongside economic results | 1–2 wk |
| 7.3b | **Temperature-target back-solve.** Given a temperature (or carbon-budget) target, invert the forward chain (carbon price → emissions → FaIR → temperature) to find the **carbon-price path** that hits it, then run forward for the sector impacts. A 1-D root-find per period (the chain is monotone), wrapping 7.1 + 7.3; reports the implied price path + resulting temperature, and flags infeasible targets rather than returning garbage. This is the credible, target-driven "IAM-ish" mode. See `docs/energy-and-temperature-plan.md`. | 1–2 wk |
| 7.4 | **Damage feedback (optional, handle with care).** Temperature → damage function → productivity shocks fed back as standard shocks. **Distinct from 7.3b:** 7.3b uses only emissions→temperature (well-established); 7.4 adds temperature→economy through a damage function (the most contested object in climate economics — order-of-magnitude disagreement). Implement only with published functions (DICE, Burke–Hsiang–Miguel), labelled as scenario illustrations, with the choice surfaced as a first-class assumption in the GUI | 1–2 wk plumbing |
| 7.5 | **Ongoing hardening.** Second data source (FIGARO/ICIO) as a new adapter to test data sensitivity; more households/regions; scenario library + cross-run comparison in the GUI; job queue if runs get heavy; API layer if others need programmatic access | ongoing |

**DoD (7.1–7.3b):** pick an NGFS scenario OR a temperature target in the GUI → the recursive-dynamic engine produces a 2020→2050 path of sector prices/volumes per region, with the associated (or target-hitting) carbon-price and temperature paths, all provenance-tagged.
**Known limits even when complete:** no endogenous technological learning; energy as CES aggregates, not discrete technologies; no land-use module. The model traces the sectoral consequences of pathways others generate; it does not generate novel pathways. Say so in the docs.
**Depends on:** P5 for GE-mode dynamics (7.2 and 7.3 can be built earlier against Engines 1–2).

---

## 3. Dependency graph

```
P0 ─▶ P1 ─▶ P2 ─▶ P3 (GUI v1)
            │
            ├──▶ P4 ─▶ P4b ─▶ P5 ─▶ P7
            │         (macro aggregates: GVA/GDP/deflator, real vs nominal;
            │          native + exact in P5; GDP *growth* needs 7.1)
            │
            └──▶ P6 (parallel with P4/P5; GE-mode nature runs need P5)
```

## 4. Effort & milestones

| Milestone | Cumulative FTE | You can… |
|---|---|---|
| End P3 | ~6–9 wk | Browse data & quality in the GUI; get supply-chain **cost** impacts of any carbon price on any good, with decomposition |
| End P4 | ~8–13 wk | Add production-**volume** responses (finite-change demand → Leontief propagation) with explicit uncertainty ranges |
| End P4b | ~9–15 wk | **GVA per sector/country, GDP per country, and a deflator** per time-step, in **real and nominal** terms (indicative PE tier; native and exact in the CGE) |
| End P6 (skipping P5) | ~11–19 wk | Nature dependency/impact exposure of any good, incl. via its supply chain, plus nature stress runs |
| End P5 | ~16–25 wk | General-equilibrium price + volume answers with carbon-tax revenue recycling |
| Full incl. P7 pathway stack | 6–12 months | NGFS-driven dynamic pathways to 2050 with temperature reporting, multi-source data, scenario library |

Sequencing note: P3, P4, and P6a all deliver standalone value and can be reordered to taste. If forced to choose a minimal useful product, **P0→P1→P2→P3→P4→P6a** (cost + volumes + nature exposure, no CGE) is the highest value-per-week path and defers the hardest work.

### Planned scenario-input extensions (detail in `docs/energy-and-temperature-plan.md`)

Two requested capabilities that fit the existing seams rather than adding new phases:

- **Country-level energy prices (near-term, low risk).** An optional `EnergyPrice` shock — a
  per-country, per-carrier (coal / oil-gas / electricity) energy-cost change, applied *in
  addition to* a carbon price. It enters the cost vector exactly like a carbon cost and
  propagates through the same Leontief inverse, so it reuses Engine 1/2 machinery (a new shock
  class + cost-assembly branch, ~3–5 d); richer in the CGE, where it triggers KL-E-M
  substitution. Fits Phases 2/4/5 as an incremental feature; blocks nothing.
- **Temperature-target back-solving (Phase 7).** Given a temperature (or carbon-budget) target,
  invert the forward chain to find the **carbon-price path** that hits it, then run forward for
  the sector impacts — the credible, target-driven "IAM-ish" mode. Added as roadmap task **7.3b**
  above; needs the Phase 7 climate module (FaIR) + recursive dynamics. Distinct from — and much
  more defensible than — temperature→economy *damage* feedback (7.4), which stays optional and
  illustrative for scientific reasons (damage-function disagreement).

---

## 5. Cross-cutting concerns

- **Testing & a standing validation suite.** Every engine ships with: analytic tests on the toy economy (P0.5), at least one known-answer test against published numbers, and identity checks (IO balance, CGE homogeneity/Walras). Beyond code-level unit tests, there is a **model-validation suite** (`cge.validation`, run by `scripts/validate.py` / `cge validate`, gated in CI) — a standing, human-readable audit that each model still reproduces its known answers and economic identities. Each engine/module adds a suite whose checks map to the properties stated in its model doc, preferring assertions against theory or published numbers over "looks plausible". Established with Engine 1 in Phase 2; see `docs/validation.md`. Being a DoD criterion, no engine is "done" without its suite.
- **Data correctness & pipeline consistency (enforced, not just tested).** Correctness and plausibility checks run at *every* stage that transforms data, and cross-stage invariants are verified *inside the pipeline* so bad data fails loudly at build time rather than surfacing as a wrong result later. Two tiers: **structural invariants** that must hold (finite values, aligned labels, square/productive `A`, existing Leontief inverse) raise and abort the build; **plausibility checks** (positive output, non-negative intensities, coverage, RoW share) and **cross-stage conservation** (aggregation preserves total output and final demand) become `QualityCheck`s in the stored `QualityReport`, with conservation failures treated as fatal. Established in Phase 1 (`cge.data.quality.consistency`); every later data transformation (SAM balancing P5, ENCORE concordance P6, NGFS ingestion P7) adds its own gate the same way.
- **Provenance everywhere.** No result exists without its data version, engine version, scenario hash, and assumption dump. This is what lets the tool be trusted and compared across runs.
- **Documentation as a deliverable — to equation level, with citations.** Every engine, module, and non-trivial data transformation ships a **model-description doc** stating the method *to equation level* (numbered equations, well-posedness argued) and citing the peer work it derives from (papers/textbooks in `docs/references.md`, institutional reports for applied choices). This is a **definition-of-done criterion** per phase, not optional — it is what makes results defensible to reviewers who know the field. The standard is `docs/documentation-standard.md`; the worked example is `docs/models/io-price-model.md`. Each doc's assumptions must match the engine's `RunManifest.assumptions`. ADRs record cross-cutting *why* (data format, closures, concordance choices).
- **Assumptions visible in the GUI.** Every run page prints the assumptions behind its numbers. For a screening/stress tool this is the single highest-leverage credibility feature.
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

**Bottom line.** The tool is buildable as designed: ~2 months FTE to a useful GUI-driven cost tool, ~4–6 months FTE to volumes + nature exposure + a simple CGE, with the main risks in data (SAM balancing, elasticities) rather than software. The phrase to hold onto: *precise about costs, indicative about volumes, transparent about assumptions.*