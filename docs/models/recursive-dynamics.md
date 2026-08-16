# Model description: Recursive dynamics (Phase 7.1)

- **Implements:** `cge.dynamics` (`recursive.py`), building on `cge.engines.cge_static.capital`
  (Phase 5d.3) and the engine's `factor_endowment_scale` hook.
- **Roadmap phase:** 7.1 (needs 5d.3's capital-accumulation identity).
- **Status: BUILT, all three CGE variants.** Runs end-to-end on the closed/gov CGE, the open
  economy (Armington/CET + rest-of-world) — both carrying one aggregate capital stock — **and the
  multi-region CGE, which carries a per-region capital path** (each region's stock steps by its own
  investment). Dynamic-capable SAMs ship for each: `toy_cge_gov`, `toy_cge_open_gov`,
  `toy_cge_multi_gov`. Labour/productivity trends are exogenous and applied uniformly across regions
  (region-specific trends are a follow-up). Magnitudes remain **illustrative** (toy calibration),
  like the rest of the CGE tier.

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
`retirement` (per-year fractions), and `structural` (an optional sourced `StructuralTrajectory`,
Phase 7b.2). The flat scalars default to **flat** — a zero-trend run is transparent bookkeeping over
the static solves, adding nothing implicit. When `structural` is set it supersedes the flat scalars
with per-region sourced paths (§6); the flat scalars are the fallback for a run that names no
trajectory. Load the vendored trajectories with `cge.data.structural.load_structural_trajectories()`.

`run_recursive(scenario, config=…, data_source="toy_cge_gov")` returns a `DynamicPath`: the
concatenated per-year `ResultSet` (with the capital-path rows) plus `capital_stock` / `investment` /
`growth` dicts. The manifest's `recursive_dynamics` block records the mode ("no perfect foresight"),
horizon, δ, trends, retirement, K₀, and the capital path.

## 6. Scope & honesty

- **All three CGE variants** — the closed/gov SAM and the open economy carry one aggregate capital
  stock; the multi-region CGE carries a **per-region capital path**, each region's stock stepping by
  its own investment (`K_{t+1,r}=(1−δ)(1−r)K_{t,r}+INV_{t,r}`, region-level capital matching 5d.3's
  granularity). Any variant needs a savings-investment account to be dynamic-capable: `toy_cge_gov`,
  `toy_cge_open_gov`, `toy_cge_multi_gov`.
- **Structural trajectories (Phase 7b.2).** Supplying a sourced `StructuralTrajectory` on the
  `DynamicConfig` replaces the flat trend scalars with **documented, sourced, per-year** paths on
  two axes, each entry carrying its own citation and confidence (validated on load, like
  `ElasticitySet`); the wrapper compounds the sourced annual rates over the actual solve-year gaps.
  The vendored artifact `data/structural/trajectories_v1.json` (see `data/structural/NOTICE.md`) is
  real sourced data. Without a trajectory the flat scalars remain the fallback. The four drivers:
    - **Per-region** — labour-supply growth = population × labour-force participation, and labour
      productivity (TFP), applied as endowment scales. *(UN WPP 2024, ILO/World Bank, PWT 10.01.)*
    - **Per-sector `sector_productivity`** (structural change / GDP-share drift) — sector-biased TFP
      fed through the engine's existing per-sector θ multiplier, so the output mix shifts
      **endogenously** (a sector with faster productivity gains share); shares are a model result,
      not an imposed target. *(EU KLEMS.)*
    - **Per-sector `emissions_intensity`** (decarbonisation) — scales `carbon_cost_share`, so a
      decarbonising sector faces a smaller priced carbon wedge in the solve AND, via a base-year
      covered-emissions reference the wrapper feeds the engine, shows falling covered emissions
      measured against the base year. *(IEA WEO 2024 / NGFS Net Zero 2050.)*
- **No perfect foresight**; recursive bookkeeping, not intertemporal optimisation.
- Productivity is Hicks-neutral on primary factors (economy-wide and per-sector θ; no factor-biased
  or vintage-specific TFP).
- Magnitudes are illustrative (toy calibration); the value is the **mechanism** — a static CGE turned
  into a capital-carrying dynamic path, the backbone Phase 7.2 (NGFS) and 7.3 (climate) build on.

See `docs/models/macro-aggregates.md` for the GDP/GVA reporting and
[`roadmap.md`](../../roadmap.md) Phase 7 for the pathway stack this unblocks.
