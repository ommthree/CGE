# Model description: Recursive dynamics (Phase 7.1)

- **Implements:** `cge.dynamics` (`recursive.py`), building on `cge.engines.cge_static.capital`
  (Phase 5d.3) and the engine's `factor_endowment_scale` hook.
- **Roadmap phase:** 7.1 (needs 5d.3's capital-accumulation identity).
- **Status: BUILT, single-region scope.** Runs end-to-end on the closed/gov CGE; open/multi-region
  are a documented follow-up. Magnitudes remain **illustrative** (toy calibration), like the rest of
  the CGE tier.

## 1. What it is (and is not)

The static CGE answers "what does the economy look like *this year* under a shock, vs the benchmark?"
Phase 7.1 makes it **recursive-dynamic**: it solves the same static CGE **year by year** to a horizon,
carrying the **capital stock** forward between solves. Each year is a normal static equilibrium of the
*same calibrated model*, re-scaled to that year's capital (and labour) endowment.

This is **bookkeeping between solves**, not a new solution concept. There is **no perfect foresight**:
agents in year *t* do not optimise over the future; the year-*t* equilibrium is solved, its investment
determines year-*(t+1)*'s capital, and we move on. This is the standard "recursive dynamic" closure
used by most applied CGE/IAM-style tools — distinct from an intertemporal (Ramsey) model.

## 2. The loop

Starting from the benchmark capital stock **K₀** (from the stock–flow bridge, §3):

1. **Solve year *t*** with the CAP endowment scaled to Kₜ and the LAB endowment scaled to that year's
   labour force Lₜ (and the productivity index).
2. **Read investment INVₜ** — the CGE's own savings-investment outcome for that year.
3. **Accumulate** (Phase 5d.3 perpetual inventory, with optional premature retirement rₜ):

   $$ K_{t+1} = (1-\delta)(1-r_t)\,K_t + INV_t $$

4. **Step the exogenous trends** — labour Lₜ₊₁ = Lₜ·(1+n), productivity by its trend — and advance.

The capital endowment in the CGE is the capital-**services** flow, proportional to the stock, so
scaling the stock by Kₜ/K₀ scales the services endowment by the same factor (the engine's
`factor_endowment_scale` hook). Labour scales the same way. **Productivity** enters as a Hicks-neutral
endowment-equivalent scale on both primary factors — a documented simplification; a genuine
sector-level TFP term is a follow-up.

Results are reported per year **relative to the original benchmark**, so capital accumulation and the
trends are **visible in the level path** (a growing stock raises output vs the benchmark). Two result
rows are added: `capital_stock` and `capital_growth`.

## 3. The stock–flow bridge and the implied benchmark growth

`capital_next` works in **stock** units, but the CGE's capital factor income is a **services flow**.
The bridge (Phase 5d.3, `benchmark_capital`) converts one to the other via the Jorgensonian user cost:

$$ \text{capital income} = u \cdot K_0, \quad u = \text{net\_return} + \delta \;\Rightarrow\; K_0 = \frac{\text{capital income}}{\text{net\_return}+\delta} $$

with documented defaults (net return 4%/yr, δ 5%/yr). The CGE manifest reports **K₀** and the
benchmark's **implied growth** g = INV₀/K₀ − δ under `capital_dynamics`. A **negative g** means the
benchmark's investment is *below* replacement (δ·K), so the stock would contract if stepped forward
unchanged — this is not hidden. On the toy SAM g ≈ −3.4%, so a zero-trend recursive run shows a gently
contracting capital path; a caller wanting a stationary or growing baseline re-anchors via the labour
and productivity trends (or, in a real build, a benchmark investment at replacement level).

## 4. Premature retirement (stranded assets)

`DynamicConfig.retirement = {year: fraction}` writes off a fraction of the *opening* stock in that
year before accumulation — e.g. fossil capital stranded by a carbon shock. It is an **exogenous
scenario input**, not a modelled investment decision (endogenous stranding — capital exiting because
its return fell below a threshold — is a documented future extension, exactly as `capital.py` notes).

## 5. Configuration & outputs

`DynamicConfig`: `depreciation` (δ, default 5%), `labour_growth` (n), `productivity_growth`,
`retirement` (per-year fractions). All trends default to **flat** — a zero-trend run is transparent
bookkeeping over the static solves, adding nothing implicit.

`run_recursive(scenario, config=…, data_source="toy_cge_gov")` returns a `DynamicPath`: the
concatenated per-year `ResultSet` (with the capital-path rows) plus `capital_stock` / `investment` /
`growth` dicts. The manifest's `recursive_dynamics` block records the mode ("no perfect foresight"),
horizon, δ, trends, retirement, K₀, and the capital path.

## 6. Scope & honesty

- **Single-region (closed/gov) only** for now — region-level capital, matching 5d.3's granularity.
  Open and multi-region capital accumulation are a documented follow-up.
- **No perfect foresight**; recursive bookkeeping, not intertemporal optimisation.
- Productivity is Hicks-neutral on primary factors (not sector-specific TFP yet).
- Magnitudes are illustrative (toy calibration); the value is the **mechanism** — a static CGE turned
  into a capital-carrying dynamic path, the backbone Phase 7.2 (NGFS) and 7.3 (climate) build on.

See `docs/models/macro-aggregates.md` for the GDP/GVA reporting and
[`roadmap.md`](../../roadmap.md) Phase 7 for the pathway stack this unblocks.
