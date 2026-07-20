# Step-by-step user guide

A guided tour from the simplest possible run to the full current feature set. Each step builds on
the previous one. Follow it top to bottom the first time; afterwards it doubles as a reference.

**What exists today:** three engines — **Engine 1 (`io_price`)**, carbon/energy **cost**
pass-through; **Engine 2 (`partial_eq`)**, production-**volume** response; and **Engine 3
(`cge_static`)**, a general-equilibrium **CGE** with revenue recycling, an **open economy**
(Armington/CET, carbon leakage), **CES value added**, elasticity sweeps and **multiple regions**
(bilateral trade) — plus **macro aggregates** (GVA/GDP/deflators, real vs nominal) on every run, a
data layer (build, store, quality), and a web GUI. Nature (ENCORE) and the pathway stack are planned
(see [`roadmap.md`](../roadmap.md)); this guide covers what runs now.

This guide **teaches the economics as it goes** — each idea (a Leontief inverse, a numéraire,
revenue recycling, carbon leakage, factor substitution) is explained from first principles the first
time it appears, so no prior CGE background is assumed. Every step can be run two ways: on the
**command line** and in the **web GUI** — both are shown, so you can follow whichever you prefer.

> Conventions: shell commands are shown with a `$` prompt. Everything here works **offline** except
> the live EXIOBASE build in Step 6. If `cge` is not on your PATH, use `python -m cge` instead, or
> prefix with your virtualenv, e.g. `.venv/bin/cge`. The GUI is launched once (Step 0) and left
> running; each step says which page to use.

---

## Step 0 — Install and verify

```bash
$ pip install -e '.[gui]'      # the .[gui] extra pulls in Streamlit for the GUI
$ cge engines                  # should list io_price and partial_eq
```

You should see something like:

```
cge_static v0.5.0 [general_equilibrium, prices, volumes] — Static CGE pilot: GE price + volume response with factor-market feedback.
    shocks: carbon_price
io_price v0.6.0 [prices] — Leontief cost-push pass-through: Δprice of every good under a carbon price and/or an energy-carrier output-price change.
    shocks: carbon_price, energy_price
partial_eq v0.3.1 [prices, volumes] — Partial-equilibrium production volume: demand response via Leontief.
    shocks: carbon_price, energy_price
```

(Exact versions move as fixes land; run `cge engines` for the current list.)

If that prints, the install is good. Now confirm the model itself is sound:

```bash
$ cge validate                 # runs the standing model-correctness suite
```

Expect `ALL PASSED — N/N checks`. This suite is the model's self-audit — if it ever fails, stop and
investigate before trusting any result.

**Launch the GUI now and leave it open** in a browser tab — every step below shows both the CLI and
the GUI way, and the GUI page names referenced later assume it is already running:

```bash
$ cge gui                      # opens the Streamlit app (add --port 8502 to change the port)
```

The GUI's left sidebar has one page per area — **Catalogue, Explorer, Quality, Build, Run scenario,
Results**. We use **Run scenario** and **Results** the most.

---

## Step 1 — The simplest possible run (toy economy, carbon price)

The **toy economy** is a tiny built-in 3-sector × 2-region fixture. It needs no data build, so it is
the fastest way to see a real result. Run the bundled Engine 1 scenario:

```bash
$ cge run --scenario examples/carbon_price_io.yaml
```

> There is also `examples/carbon_price_toy.yaml`, but it runs the **`dummy`** engine — Phase-0
> plumbing, not economics. Use `carbon_price_io.yaml` (engine `io_price`) for your first real run.

You get three things, and they matter equally:

1. **The scenario name and hash** — the hash identifies these exact inputs.
2. **The assumptions block** — printed on *every* run. This is the credibility surface: it states
   the model, the units, and the interpretation caveats. Read it once now so you know it is there.
3. **The results table** — one row per good per output variable. `price_change` is the headline:
   a **fractional** change in the good's unit price (e.g. `0.06` = +6%).

Notice the `price_change_direct` / `price_change_upstream_tier_1…` rows: that is the **supply-chain
decomposition** — how much of a good's price rise is its own emissions cost versus cost inherited
from its inputs, tier by tier. They sum exactly to `price_change`.

**What you just saw:** Engine 1 turning a €/tonne carbon price into a per-good cost impact with full
supply-chain pass-through. That is the foundational answer everything else builds on.

---

## Step 2 — Read the same scenario file, then change it

Open `examples/carbon_price_io.yaml`. A scenario is declarative:

```yaml
name: "..."
engine: io_price          # which engine runs it
years: [2020]             # one or more time-steps
shocks:
  - type: carbon_price
    price: 100.0
    gases: [CO2]          # which greenhouse gases the price applies to
```

Try editing it (copy it first if you like):

- **Raise the price** to `250.0` and re-run — every price change scales linearly.
- **Restrict coverage** by adding `coverage_regions: [A]` under the shock — only region A's goods are
  taxed directly; other regions still move a little, through cross-region supply chains.
- **Add a time path** with `path: {2020: 0.0, 2030: 100.0}` and set `years: [2020, 2025, 2030]` —
  the price ramps and results vary by year.

Re-running after each edit shows how one lever at a time changes the answer. The assumptions block
and hash update to match.

---

## Step 3 — Add a volume response (Engine 2)

Engine 1 tells you about **cost**. To ask *"how much does produced quantity fall?"*, switch to
Engine 2:

```bash
$ cge run --scenario examples/carbon_price_volume.yaml
```

Same carbon price, but now the result also carries:

- `final_demand_change` — how demand for each good responds to its price rise (a finite-change
  elasticity response, bounded so it can never fall by more than 100%);
- `volume_change` — the **production** volume change (`Δx/x`), obtained by propagating the demand
  change through the Leontief quantity system, so a good's upstream suppliers fall too;
- three **scenario bands** — `low` / `central` / `high` — spanning demand-elasticity uncertainty.

Volume magnitudes are **indicative**, not precise (there is no clean open elasticity database), and
the assumptions block says so. Use them for screening and cross-checking, not as point forecasts.

**Key idea:** you did not change engines by editing code — the scenario's `engine:` field selected
`partial_eq`, and the GUI/CLI pick it up purely from the engine registry.

### Macro aggregates come for free

Every run — Engine 1 or Engine 2 — now also carries **macroeconomic aggregates**, added
automatically: `gva_change` per sector, `gdp_change` per country, and a `deflator` (inflation),
each in **nominal** and **real** terms (real = nominal deflated by the index). Region-level rows use
the sentinel sector `__economy__`. Look for them in the results table, e.g. filter for
`variable == gdp_change`.

Read them like this:
- On the **Engine 1** run (Step 1), nominal GDP change equals the deflator and **real GDP change is
  ~0** — a price-only model produces inflation but says nothing about real quantities. That is
  correct, not a bug.
- On the **Engine 2** run, **real GDP falls** under the carbon price (volumes drop by more, in real
  terms, than the price index rises), with a low/central/high band from the elasticity uncertainty.

This is the *indicative* tier (arithmetic on the engine outputs); the CGE will produce these as exact
equilibrium variables. One honest limit built into the labelling: there is **no monetary interest
rate** here — that needs a macro-financial closure the current engines don't have.
See [`docs/models/macro-aggregates.md`](models/macro-aggregates.md).

---

## Step 4 — Energy prices (and combining shocks)

An **energy-carrier output-price change** is a second lever, applied *on top of* a carbon price. The
bundled example combines both:

```bash
$ cge run --scenario examples/energy_price_io.yaml
```

Look at the `energy` sector's `price_change`: it rises by the **direct** energy-price change plus its
own upstream energy use. Energy-intensive goods (manufacturing) move more than light ones
(agriculture) — the whole point of modelling energy cost separately.

The scenario file shows the shape:

```yaml
shocks:
  - type: carbon_price
    price: 50.0
  - type: energy_price
    carrier: energy        # 'energy' on the toy build; 'electricity'/'energy_coal'/... on a coarse build
    change: 0.30           # +30% output-price rise
    # coverage_regions: [A]          # optional: restrict to countries
    # path: {2025: 0.0, 2035: 0.30}  # optional time path
```

A carbon price and an energy price in one scenario do **not** simply add on the carrier: the energy
shock **pins** the carrier's price to exactly the requested change (a boundary condition), which
*overrides* the carbon-induced price on that carrier — so the combined carrier result is the pinned
value, not carbon + energy. Non-carrier sectors reflect both effects. (Multiple *carbon* shocks do
add.) You can point the same scenario at Engine 2 (`engine: partial_eq`) for the *volume*
consequences.

Try: set `change: -0.1` (cheaper energy) and watch prices fall; set an unknown `carrier:` and see the
engine **reject** it rather than silently returning a zero-impact run.

---

## Step 5 — Build real data (offline)

The toy economy is for learning. For real answers you need a data build. The offline build uses
`pymrio`'s bundled test MRIO — no download, fully reproducible:

```bash
$ cge build --test          # writes a full build and an aggregated 'small' build to the store
$ cge data                  # list what is now in the store
```

`cge data` shows the build catalogue — ids, source, reference year, aggregation, and the worst
quality flag. Inspect a build's data-quality report:

```bash
$ cge quality <build_id>    # use an id from 'cge data'
```

The quality report is the pipeline's honesty layer: structural invariants (finite values, a
productive `A`, an existing Leontief inverse) plus plausibility and cross-stage conservation checks
(aggregation preserves total output and final demand). A build that violates a *structural* invariant
never makes it into the store.

**A deliberate currency guard you will hit here.** The offline `--test` build is denominated in
**USD** (the pymrio fixture's real currency). Engine 1 applies a euro-specific cost-share scaling and
therefore **refuses a non-EUR build** — this is correct behaviour, not a bug: it would rather reject
the run than return a wrongly-scaled number. So the `--test` build is for exercising the **data layer**
(build → store → quality → the Explorer in Step 8), while a *priced* `io_price`/`partial_eq` run needs
a **EUR** build, which the live EXIOBASE build (Step 6) provides. You can still point a run at a build
with `--data <build_id>`:

```bash
$ cge data                                                      # copy a build id
$ cge run --scenario examples/carbon_price_io.yaml --data <eur_small_build_id>
```

Use the **small (aggregated)** build — the engines are dense-only and capped, so the full ~9,800-good
MRIO is rejected with a clear message telling you to aggregate first.

---

## Step 6 — Build live EXIOBASE data (optional; downloads)

To use real EXIOBASE (a multi-hundred-MB download, needs network and disk):

```bash
$ cge build --exiobase --year 2019
```

This fetches the archive, adapts it into the harmonised data objects (normalising emission units to
tonnes), runs the quality gates, and stores both a full and a small build. Everything from Step 5
onward then works against the resulting build id. This is the only step that touches the network.

---

## Step 7 — General equilibrium: the CGE (Engine 3)

This is the longest step, because the CGE is where the economics gets interesting — and where the
ideas need explaining. We build it up in small pieces (7a–7f), each with the concept first, then a
run you do yourself (CLI **and** GUI).

### What a "general equilibrium" is, and why it is different

Engines 1 and 2 are **partial**: Engine 1 pushes a carbon cost through fixed input recipes; Engine 2
adds a demand response. Neither lets the *whole economy re-balance* — factor markets don't clear,
relative prices don't fully adjust, and nothing pins the overall price level.

A **Computable General Equilibrium (CGE)** model closes all of that. It finds a set of prices at
which **every market clears at once**: goods markets (supply = demand for each product), factor
markets (the labour and capital the economy owns are fully employed), and the government budget
(carbon revenue goes somewhere). "Equilibrium" means *no price wants to move*; "computable" means we
solve for those prices numerically. The pay-off: when you tax the dirty sector, the model shows not
just its cost rising but **capital and labour moving to the clean sector, households re-spending, and
imports/exports adjusting** — the second-round effects the partial engines cannot see.

Three ideas you'll meet, defined once here:

- **SAM (Social Accounting Matrix).** The CGE calibrates to a *balanced* square table of who-pays-whom
  in a base year: every account's income equals its spending. It's the double-entry photograph of the
  economy the model must reproduce before it's allowed to answer any question. We ship small,
  hand-checkable SAMs (`toy_cge`, `toy_cge_open`, `toy_cge_multi`) so you can run the CGE with no data
  build.
- **Numéraire.** Only *relative* prices matter in equilibrium (double everything and nothing real
  changes), so we fix one price index as the yardstick — here the household's cost-of-living index
  (CPI). All prices are then read "in CPI units", which is what makes *real* vs *nominal* well-defined.
- **Calibration & replication.** Before trusting a scenario, the model re-solves the base year with
  **no shock** and checks it reproduces the SAM exactly. If a zero shock doesn't give zero change, the
  benchmark is wrong and the run is refused. You'll see this as "benchmark replication".

### 7a — Your first CGE run (closed economy)

The **closed** economy (`toy_cge`) is the simplest: two sectors — `BRD` (a dirty, emissions-intensive
"bread" sector) and `MIL` (a cleaner "mill" sector) — two factors (capital `CAP`, labour `LAB`), and
one household. No trade yet.

**CLI.** Run a €50/tonne carbon price:

```bash
$ cge run --scenario examples/carbon_price_cge.yaml --data toy_cge
```

Read the output top to bottom:

1. The **assumptions block** now says *general_equilibrium* and describes the closure (CPI numéraire,
   fixed factor supply, revenue recycling). This is the credibility surface — it states exactly what
   the numbers mean.
2. `price_change` / `volume_change` per sector — but now these are **equilibrium** responses. The
   dirty sector's output falls; watch the clean sector's *rise*. That reallocation is the GE signal.
3. `factor_price_change` for CAP and LAB — the return to capital and the wage adjust as production
   shifts between sectors with different factor intensities. Engines 1–2 have no such thing.
4. `gdp_change_real` and `welfare_change` — the whole-economy summaries (defined in 7b).

**GUI.** On the **Run scenario** page: set **Data** to `toy_cge` (the engine auto-switches to
`cge_static`), leave **Carbon price** at 100, and click **Run**. Open the **Results** page: the price
and volume tables, the factor-price rows, and the full assumptions block are all there. This is the
same run as the CLI, in the browser.

> **Try it:** raise the price and re-run. Real GDP falls further, and the dirty→clean reallocation
> grows. There is no "energy price" control here — the CGE priced good is emissions, not a carrier.

### 7b — Revenue recycling and the "double dividend" (what it is, and what this model does *not* claim)

A carbon tax **collects revenue**. What happens to it matters enormously for the welfare result, so
the CGE makes it explicit — the `revenue_recycling` control:

- **`lump_sum`** — the government hands the revenue back to the household as a flat transfer.
- **`labour_tax_cut`** — the revenue funds a cut in labour taxes.
- **`none`** — revenue is *not* returned. In a closed economy this doesn't actually close (money
  vanishes, breaking the accounting), so the engine refuses it and defaults to `lump_sum`, telling you
  so. Use Engine 1 if you want the pure price-side view with no recycling.

**Why recycling matters (the concept).** Without recycling, a carbon tax is a pure loss to households.
*With* recycling, the money comes back — so the only remaining cost is the **distortion** (the economy
producing a slightly less efficient mix because relative prices changed). Returning the revenue
therefore **shrinks the welfare loss** substantially. That's the *revenue-recycling effect*, and you
can see it directly:

```bash
$ cge run --scenario examples/carbon_price_cge.yaml --data toy_cge   # recycling: lump_sum
```

Look at `welfare_change`: a small negative number. The revenue is returned, so what's left is just the
distortion.

**The "double dividend" — and the honest caveat.** The *double-dividend hypothesis* says recycling
carbon revenue through cutting an existing **distortionary** tax (like a labour tax that discourages
work) could give *two* wins: less pollution **and** a more efficient tax system. It's a real and
much-debated idea in the literature — but **this pilot does not model it**. With a single household
and no distortionary labour-tax wedge, `labour_tax_cut` and `lump_sum` are *economically identical*
here (the money reaches the same household either way). The model says so in its assumptions. A genuine
double dividend needs heterogeneous households or an explicit tax distortion — a documented follow-up.
We flag this because it is the single most over-claimed result in carbon-pricing tools, and this guide
would rather under-promise.

**GUI.** On **Run scenario** with `toy_cge`, open the **Carbon price** section and change **Revenue
recycling** between `lump_sum`, `labour_tax_cut` and `none`. Run each and compare `welfare_change` on
the **Results** page. You'll see `lump_sum` and `labour_tax_cut` give the *same* welfare (the caveat
above, made concrete), and `none` reports that it defaulted to `lump_sum`.

### 7c — Factor substitution: the CES value-added nest

In 7a the sectors combined capital and labour **Cobb-Douglas** (unit elasticity of substitution —
*not* fixed proportions; that's a stronger assumption called Leontief, where the factor ratio
never moves regardless of relative prices). Real firms can substitute **more or less easily than
Cobb-Douglas's unit elasticity**: if capital gets relatively cheaper, how much more of it they use
depends on that elasticity. The
CGE captures this with a **CES (constant-elasticity-of-substitution) value-added nest**, governed by
one number, `va_elast` (σ_va):

- **σ_va = 1** — the Cobb-Douglas default (moderate substitution).
- **σ_va < 1** — factors are *hard* to substitute (complements); a shock that changes relative factor
  prices causes a **larger** swing in those prices, because firms can't easily rebalance.
- **σ_va > 1** — factors substitute *easily*; relative factor prices move **less**.

This matters for carbon policy because taxing a capital-intensive dirty sector shifts demand between
capital and labour, and how easily firms substitute determines how much wages vs. the capital return
move.

**CLI.** Pass the elasticity through as an engine parameter (the example scenario reads it from the
data source, so use a tiny inline override via the Python API, or add `va_elast` to a copy of the
scenario). The simplest demonstration is the GUI slider below; on the CLI you can point at the same
data source and see the closed run — the elasticity is wired through `data_overrides` in the runner
and the GUI.

**GUI.** On **Run scenario** with `toy_cge`, open the **CGE options (elasticities)** expander and move
the **Value-added elasticity σ_va** slider to `0.4`, then run; note the `factor_price_change` rows.
Now set it to `2.5` and run again. The **relative** factor-price swing is *larger* at the low
elasticity — factors that can't substitute must reprice more. That's the CES nest biting, and it's the
channel behind serious factor-market and (in richer models) double-dividend analysis.

### 7d — Opening the economy: Armington imports, CET exports, and carbon leakage

The closed model can't relocate production abroad. The **open economy** (`toy_cge_open`) adds a
**rest-of-world (ROW)** account and two standard trade mechanisms:

- **Armington imports.** Each good is a blend of a *domestic* variety and an *imported* variety, which
  are **imperfect substitutes** (a CES aggregate with elasticity σ, `armington_elast`). When the
  domestic dirty good gets pricier, buyers shift toward imports — how much depends on σ.
- **CET exports.** Domestic output is split between home sales and exports by a *transformation*
  frontier (elasticity Ω, `cet_elast`). When home prices rise, producers export less.

Put these together and a carbon price produces the signature open-economy result — **carbon leakage**:
the taxed dirty sector's output falls, its **imports rise** (buyers substitute to un-taxed foreign
supply), and its **exports fall** (it's less competitive). Emissions don't disappear; they partly
*relocate abroad*. This is the single most important reason carbon policy is analysed in an open
economy, and the CGE shows it mechanically.

**CLI.**

```bash
$ cge run --scenario examples/carbon_price_cge.yaml --data toy_cge_open
```

New result variables appear: `import_change`, `export_change`, `exchange_rate_change`. For the dirty
sector `BRD` you should see **output down, imports up, exports down** — leakage. The exchange rate
moves to keep the trade balance closed.

**GUI.** On **Run scenario**, set **Data** to `toy_cge_open`. A **CGE options** expander now also
shows **Armington (import) elasticity σ** and **CET (export) elasticity Ω** sliders. Run with the
defaults, open **Results**, and find the `import_change` / `export_change` rows for `BRD`.

### 7e — How much leakage? Elasticity sweeps

The leakage magnitude hinges on the Armington elasticity σ — and no open elasticity database is
authoritative, so a single number would be false precision. The honest output is a **band**. Move the
**Armington elasticity** slider and watch the leakage change:

**GUI.** With `toy_cge_open`, set the Armington slider to **1.5**, run, and note `import_change` for
`BRD`. Now set it to **4.0** and run again. Imports rise *much* more at the higher elasticity — buyers
substitute to foreign supply more readily. Reporting the low/central/high envelope (say σ = 1.5 / 2 /
4) is how the model states leakage *honestly*: "somewhere in this range, depending on how substitutable
imports are". (The programmatic sweep that returns this envelope as one object is
`armington_sensitivity_sweep`, with full provenance for each band.)

**The lesson:** where a result is elasticity-sensitive, the band **is** the answer — exactly as
Engine 2's volume responses come with a low/central/high range. Precision about costs; ranges about
volumes and trade.

### 7f — Multiple regions: bilateral trade and cross-region leakage

The open economy trades with an anonymous "rest of world". The **multi-region** model
(`toy_cge_multi`) goes further: **two fully-modelled regions** (`N` = North, `S` = South), each with
its own sectors, factors, household, and production — trading *bilaterally* with each other. Now
"where does the production go?" has a concrete answer: **to the other region**.

Each region imports each good as an Armington blend over its **domestic variety and imports from the
partner region**, and exports via a CET split. So a carbon price in the North relocates dirty
production to the South — and you can *see* the South's output rise.

**CLI.**

```bash
$ cge run --scenario examples/carbon_price_cge.yaml --data toy_cge_multi
```

Results are now **region-tagged** (a `region` column with `N` and `S`). The default scenario taxes the
North's dirty sector. Read across regions:

- North `BRD`: **output falls**, **imports rise** (from the South).
- South `BRD`: **output rises** — the relocated production.
- Per-region `real_consumption_change`, `factor_price_change`, `welfare_change`, `carbon_revenue`.
  (`real_consumption_change` is a base-price household-consumption index, not production-side real
  GDP — only region North's CPI is pinned as numéraire, so a `pq·FD` deflation off that region isn't
  valid; see §7c/§7b for the closed/open engine's own GDP treatment.)

That North-loses / South-gains split in one good is **cross-region carbon leakage** made explicit — a
result neither the closed nor the single-region-open model can express.

**GUI.** On **Run scenario**, set **Data** to `toy_cge_multi` and run. On **Results**, filter the
`region` column to compare `N` and `S`: the taxed region's dirty output falls while the partner's rises.

> **Scope, honestly.** These are *toy* SAMs chosen to be hand-checkable. Every trade route has its
> own destination-specific price (no law-of-one-price shortcut), and every bilateral market clears
> explicitly under a shock, not just at the benchmark. Magnitudes are illustrative; the value is the
> **direction and mechanism** — which are textbook-correct and replicate their benchmark to machine
> precision. Running on real EXIOBASE-shaped data needs an IOSystem-driven multi-region SAM build,
> which (unlike the single-region open economy's `build_open_sam`) is a documented follow-up —
> today the multi-region model requires a supplied SAM.

---

## Step 8 — The web GUI (full tour)

You've already used the GUI's Run and Results pages throughout Step 7. Here is the whole app — walk
the pages left to right; they mirror this guide:

- **Catalogue** — the builds in your store (what `cge data` prints).
- **Explorer** — browse any build's matrices **like a spreadsheet**: the `A` matrix, final demand,
  satellite emissions. This is the "look at the data like an Excel sheet" view.
- **Quality** — the quality report per build, colour-coded by severity.
- **Build** — trigger a `--test` (or live) build from the browser.
- **Run scenario** — pick a **data source** (`toy` for Engines 1–2; `toy_cge` / `toy_cge_open` /
  `toy_cge_multi` for the CGE variants from Step 7; or a store build), pick an **engine** (it
  auto-selects `cge_static` for the CGE SAMs), set a **carbon price**, choose **revenue recycling**
  (GE engines), open **CGE options** for the value-added / Armington / CET **elasticity** sliders, and
  add any number of **energy-carrier price** shocks (Engines 1–2). Everything is composed into one
  scenario; the controls that appear are driven by the chosen engine's registry metadata.
- **Results** — the headline price table, the volume envelope (when Engine 2 ran), the CGE's
  **factor prices / trade / exchange-rate** rows, the **macroeconomic aggregates** (GDP/GVA/deflator,
  nominal and real; the CGE emits per-region GDP for `toy_cge_multi`), the supply-chain
  **decomposition waterfall**, the full assumptions block, and **exports** (CSV, or Parquet with the
  run manifest embedded so a downloaded result stays traceable to its inputs). For the multi-region
  run, filter the **region** column to compare `N` and `S`.

---

## Step 9 — Trust the numbers: provenance and validation

Two habits make results defensible:

1. **Keep the manifest.** Every result carries a `RunManifest`: the data build **and its generation**,
   the satellite and elasticity inputs (each with a content hash), the engine version, the scenario
   hash, and the full assumptions. Export the Parquet-plus-manifest from the Results page, or the
   manifest JSON, and a result stays reproducible and comparable across runs. If any substantive
   input changes — even the same build re-saved, or a different satellite — the manifest changes too.

2. **Run the validation suite** whenever you change anything:

   ```bash
   $ cge validate                       # all suites
   $ cge validate --suite io_price      # just one
   $ cge validate --strict              # exit non-zero on any failure (for scripts/CI)
   ```

   These are model-correctness checks tied to the equations in the model docs (analytic identities,
   known answers, linearity, additivity, bounded volumes). They are the standing proof that the model
   still does what its documentation claims.

---

## Where to go next

- **The method, to equation level:** [`docs/models/io-price-model.md`](models/io-price-model.md)
  (Engine 1, incl. §5a energy prices), [`docs/models/partial-equilibrium.md`](models/partial-equilibrium.md)
  (Engine 2), and [`docs/models/cge-static.md`](models/cge-static.md) (Engine 3 — the CGE, incl. the
  open economy §8, multi-region §8a, CES value added, and the closures behind Step 7).
- **The big picture:** [`docs/overview.md`](overview.md) — what the platform is and why.
- **What is coming:** [`roadmap.md`](../roadmap.md) — nature/ENCORE, the GE tier of the macro
  aggregates, and the NGFS/temperature pathway stack.
- **Planned scenario inputs:** [`docs/energy-and-temperature-plan.md`](energy-and-temperature-plan.md).
