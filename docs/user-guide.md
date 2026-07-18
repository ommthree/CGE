# Step-by-step user guide

A guided tour from the simplest possible run to the full current feature set. Each step builds on
the previous one. Follow it top to bottom the first time; afterwards it doubles as a reference.

**What exists today:** two engines — **Engine 1 (`io_price`)**, carbon/energy **cost** pass-through,
and **Engine 2 (`partial_eq`)**, production-**volume** response — plus a data layer (build, store,
quality) and a web GUI. The CGE (Engine 3), nature (ENCORE), macro aggregates (GVA/GDP), and the
pathway stack are planned (see [`roadmap.md`](../roadmap.md)); this guide covers what runs now.

> Conventions: shell commands are shown with a `$` prompt. Everything here works **offline** except
> the live EXIOBASE build in Step 6. If `cge` is not on your PATH, use `python -m cge` instead, or
> prefix with your virtualenv, e.g. `.venv/bin/cge`.

---

## Step 0 — Install and verify

```bash
$ pip install -e '.[gui]'      # the .[gui] extra pulls in Streamlit for Step 7
$ cge engines                  # should list io_price and partial_eq
```

You should see something like:

```
io_price v0.5.0 [prices] — Leontief cost-push pass-through: Δprice of every good under a carbon price and/or an energy-carrier output-price change.
    shocks: carbon_price, energy_price
partial_eq v0.3.0 [prices, volumes] — Partial-equilibrium production volume: demand response via Leontief.
    shocks: carbon_price, energy_price
```

If that prints, the install is good. Now confirm the model itself is sound:

```bash
$ cge validate                 # runs the standing model-correctness suite
```

Expect `ALL PASSED — N/N checks`. This suite is the model's self-audit — if it ever fails, stop and
investigate before trusting any result.

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

Two shocks in one scenario **compose additively** — the combined result equals the two run
separately, summed (a property the validation suite checks to machine precision). You can point the
same scenario at Engine 2 (`engine: partial_eq`) to get the *volume* consequences of an energy-price
rise.

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
(build → store → quality → the Explorer in Step 7), while a *priced* `io_price`/`partial_eq` run needs
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

## Step 7 — The web GUI

Everything above has a visual counterpart:

```bash
$ cge gui                    # launches the Streamlit app (add --port 8502 to change the port)
```

Walk the pages left to right — they mirror this guide:

- **Catalogue** — the builds in your store (what `cge data` prints).
- **Explorer** — browse any build's matrices **like a spreadsheet**: the `A` matrix, final demand,
  satellite emissions. This is the "look at the data like an Excel sheet" view.
- **Quality** — the quality report per build, colour-coded by severity.
- **Build** — trigger a `--test` (or live) build from the browser.
- **Run scenario** — pick a data source (`toy` or a build), pick an engine, set a carbon price and
  region coverage, and run. The engine list and its capabilities are rendered from the registry.
- **Results** — the headline price table, the volume envelope (when Engine 2 ran), the supply-chain
  **decomposition waterfall**, the full assumptions block, and **exports** (CSV, or Parquet with the
  run manifest embedded so a downloaded result stays traceable to its inputs).

> Note: the GUI Run page currently builds **carbon-price** scenarios. For **energy-price** or
> **combined** scenarios, use a YAML file with `cge run` as in Step 4 (a GUI control for energy
> prices is a small planned addition).

---

## Step 8 — Trust the numbers: provenance and validation

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
  (Engine 1, incl. §5a energy prices) and [`docs/models/partial-equilibrium.md`](models/partial-equilibrium.md)
  (Engine 2).
- **The big picture:** [`docs/overview.md`](overview.md) — what the platform is and why.
- **What is coming:** [`roadmap.md`](../roadmap.md) — the CGE, nature/ENCORE, macro aggregates
  (GVA/GDP/deflators, real vs nominal — Phase 4b), and the NGFS/temperature pathway stack.
- **Planned scenario inputs:** [`docs/energy-and-temperature-plan.md`](energy-and-temperature-plan.md).
