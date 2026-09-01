# Model description: Recursive dynamics (Phase 7.1)

- **Implements:** `cge.dynamics` (`recursive.py`), building on `cge.engines.cge_static.capital`
  (Phase 5d.3) and the engine's `factor_endowment_scale` hook.
- **Roadmap phase:** 7.1 (needs 5d.3's capital-accumulation identity).
- **Status: BUILT, all three CGE variants.** Runs end-to-end on the closed/gov CGE, the open
  economy (Armington/CET + rest-of-world) — both carrying one aggregate capital stock — **and the
  multi-region CGE, which carries a per-region capital path** (each region's stock steps by its own
  investment). Dynamic-capable SAMs ship for each: `toy_cge_gov`, `toy_cge_open_gov`,
  `toy_cge_multi_gov`. Labour/productivity trends are exogenous. With the **flat** `DynamicConfig`
  scalars they are applied uniformly across regions; with a sourced **`StructuralTrajectory`**
  (Phase 7b.2) they are **per-region** (and per-sector for the sectoral-productivity and
  emissions-intensity drivers) — so region-specific trends now exist. Magnitudes remain
  **illustrative** (toy calibration), like the rest of the CGE tier.

  **Capital is stepped every calendar year** between the first and last requested year, not only on
  the (possibly sparse) requested years: the requested `years` are the *reporting* years, and the
  wrapper solves every intervening year internally so investment and depreciation accumulate
  annually. A sparse `[2025, 2030]` horizon therefore reports the SAME 2030 stock as the full
  `[2025, …, 2030]` horizon (review P1 2026-08-23).

  **Investment→capital is a REAL flow.** The wrapper steps the (real) capital stock with the engine's
  `investment_volume` output — investment demand valued at **benchmark prices** — not the nominal
  `investment` share, so investment-price movements do not masquerade as capital formation. For the
  multi CGE `investment_volume` is normalised by **global** GDP, matching the globally-normalised
  capital stock (review P1 2026-08-23).

## 1. What it is (and is not)

The static CGE answers "what does the economy look like *this year* under a shock, vs the benchmark?"
Phase 7.1 makes it **recursive-dynamic**: it solves the same static CGE **year by year** to a horizon,
carrying the **capital stock** forward between solves. Each year is a normal static equilibrium of the
*same calibrated model*, re-scaled to that year's capital (and labour) endowment.

This is **bookkeeping between solves**, not a new solution concept. There is **no perfect foresight**:
agents in year *t* do not optimise over the future; the year-*t* equilibrium is solved, its investment
determines year-*(t+1)*'s capital, and we move on. This is the standard "recursive dynamic" closure
used by most applied CGE/IAM-style tools — distinct from an intertemporal (Ramsey) model.

### Notation

| Symbol | Meaning | Units |
| --- | --- | --- |
| $t$ | calendar year (internal solve years are every year in $[t_0, T]$) | year |
| $K_t$ | capital **stock** at the start of year $t$ (per capital region $r$) | GDP-normalised |
| $INV_t$ | real investment volume in year $t$ (engine's `investment_volume`, benchmark prices) | share of benchmark GDP |
| $\delta$ | forward accumulation depreciation rate (`DynamicConfig.depreciation`) | /yr |
| $r_t$ | premature-retirement fraction of the opening stock in year $t$ | dimensionless |
| $\delta_c$ | user-cost **calibration** depreciation (fixed 5%, distinct from $\delta$) | /yr |
| $u,\ \text{net\_return}$ | Jorgensonian user cost and its net-return component | /yr |
| $L_t$ | labour-force scale in year $t$ | index (benchmark = 1) |
| $A_t$ | Hicks-neutral aggregate-productivity scale | index (benchmark = 1) |
| $\varphi_{s}$ | cumulative labour-augmenting factor on sector $s$'s labour input | index (benchmark = 1) |
| $s_L$ | benchmark labour share of a sector's value added | dimensionless |
| $\theta$ | the engine's per-sector Hicks-neutral productivity multiplier | index |

### Assumptions

1. **No perfect foresight.** Each year is a static equilibrium of the same calibrated model; agents
   do not optimise intertemporally. Encoded by the year-by-year loop; the manifest's
   `recursive_dynamics.mode` records "no perfect foresight".
2. **Capital services ∝ stock.** The CAP endowment scales linearly with the stock (eq $(4)$), so a
   stock ratio is an endowment ratio. No vintage structure.
3. **Real investment steps the stock.** Accumulation uses the engine's benchmark-priced
   `investment_volume`, not the nominal investment share, so investment-price movements are not
   miscounted as capital formation.
4. **Annual accumulation.** Capital is stepped every calendar year in $[t_0, T]$ regardless of which
   years are reported (eq $(1)$); reporting years are a strict subset.
5. **Two depreciation concepts are independent.** The calibration $\delta_c$ (fixed 5%) bootstraps
   $K_0$ (eq $(2)$); the forward $\delta$ (`DynamicConfig.depreciation`) drives eq $(1)$. They are
   never conflated (manifest labels them separately).
6. **Aggregate productivity is Hicks-neutral**, applied as an equal endowment-equivalent scale on
   both primary factors (eq $(3)$). Per-**sector** structural change is a **heuristic labour-
   augmenting composition lever** on the sector's labour input (eq $(5)$), not an identified
   technology term and not a claim of decomposed capital deepening (§6).
7. **Exogenous trends.** Labour and productivity paths are exogenous inputs — flat `DynamicConfig`
   scalars, or a sourced per-region/per-sector `StructuralTrajectory` (§6). Never implicit.

## 2. The loop

Starting from the benchmark capital stock **K₀** (from the stock–flow bridge, §3):

1. **Solve year *t*** with the CAP endowment scaled to Kₜ and the LAB endowment scaled to that year's
   labour force Lₜ (and the productivity index).
2. **Read investment INVₜ** — the CGE's own savings-investment outcome for that year.
3. **Accumulate** (Phase 5d.3 perpetual inventory, with optional premature retirement rₜ):

   $$ K_{t+1} = (1-\delta)(1-r_t)\,K_t + INV_t \tag{1} $$

4. **Step the exogenous trends** — labour Lₜ₊₁ = Lₜ·(1+n), productivity by its trend — and advance.

The capital endowment in the CGE is the capital-**services** flow, proportional to the stock, so
scaling the stock by Kₜ/K₀ scales the services endowment by the same factor (the engine's
`factor_endowment_scale` hook), equation $(4)$. Labour scales the same way. **Aggregate
productivity** enters as a Hicks-neutral endowment-equivalent scale on both primary factors.
**Sector-level** productivity is implemented (Phase 7b.2) as a **labour-augmenting** term on each
sector's labour input — not a Hicks-neutral θ — so the output mix shifts endogenously (§6, equation
$(5)$). It is a **heuristic composition lever**: the sourced series is observed labour productivity
(which mixes technology, capital deepening and utilisation), and this driver does not decompose those
out. Putting it on labour input keeps it a composition lever rather than an aggregate-level re-imposer
(the factor-augmenting form of [Solow1957]); it is NOT a claim that capital deepening has been
identified out — see §6 for the honest statement and the growth-accounting follow-up.

Concretely, the year-$t$ primary-factor endowment scales fed to the engine's `factor_endowment_scale`
hook are

$$ \text{lab\_scale}_t = L_t \cdot A_t, \qquad \text{cap\_scale}_t = \frac{K_t}{K_0} \cdot A_t \tag{3} $$

so a static solve at year $t$ is the benchmark model with its labour and capital services endowments
re-scaled by $(3)$; with $A_t = L_t = 1$ and $K_t = K_0$ (the base year, no trends) the scales are
unity and the solve is byte-identical to the static benchmark run. Because the CAP endowment is the
capital-services flow proportional to the stock,

$$ \frac{\partial\,\text{cap\_scale}_t}{\partial K_t} = \frac{A_t}{K_0} \tag{4} $$

i.e. a stock ratio maps one-to-one to a services-endowment ratio (assumption 2). The per-sector
labour-augmenting factor $\varphi_s$ scales **only** sector $s$'s labour entry. Under a **Cobb–
Douglas** value-added nest ($\sigma_{va}=1$, the default) the sector's VA unit cost then satisfies,
exactly,

$$ c^{VA}_s(\varphi_s) = \varphi_s^{-s_L}\, c^{VA}_s(1) \tag{5} $$

so a $\varphi_s > 1$ (faster sector labour productivity) lowers that sector's cost and it gains output
share endogenously ([Solow1957]). For a **CES** nest ($\sigma_{va}\neq1$) the exact unit-cost response
is *not* $\varphi_s^{-s_L}$ — it depends on $\sigma_{va}$ and the post-substitution factor shares; eq
$(5)$ is then a first-order ($s_L$-weighted) approximation, and the engine computes the exact CES cost
internally regardless. The important honesty point (see §6): the sourced sector series is observed
labour productivity, so $\varphi_s$ here is a **heuristic composition lever**, not an identified
labour-augmenting technology parameter.

Results are reported per year **relative to the original benchmark**, so capital accumulation and the
trends are **visible in the level path** (a growing stock raises output vs the benchmark). Two result
rows are added: `capital_stock` and `capital_growth`.

## 3. The stock–flow bridge and the implied benchmark growth

`capital_next` works in **stock** units, but the CGE's capital factor income is a **services flow**.
The bridge (Phase 5d.3, `benchmark_capital`) converts one to the other via the Jorgensonian user cost:

$$ \text{capital income} = u \cdot K_0, \quad u = \text{net\_return} + \delta \;\Rightarrow\; K_0 = \frac{\text{capital income}}{\text{net\_return}+\delta} \tag{2} $$

with documented defaults (net return 4%/yr, **calibration** δ 5%/yr). The CGE manifest reports **K₀**
and the benchmark's **implied growth** g = INV₀/K₀ − δ under `capital_dynamics`.

**Two distinct depreciation concepts (review P4 2026-08-28).** The δ in the user-cost bridge above is
a *calibration* rate — a historical, benchmark-consistent user-cost parameter used **only** to back
out the level of K₀ from the observed capital-income flow. It is fixed at the engine's documented 5%
and is deliberately **independent** of `DynamicConfig.depreciation`, which is the *forward*
accumulation rate applied in step 3's perpetual-inventory identity K_{t+1}=(1−δ)(1−r)K_t+INV_t. So a
run with `DynamicConfig.depreciation=0.20` still bootstraps K₀ off the 5% calibration bridge (the
benchmark's own historical user cost) and then depreciates the *forward* path at 20%. The manifest
labels the bridge's rate under `capital_dynamics.depreciation_rate` and the forward rate under
`recursive_dynamics.depreciation_rate` so the two are never conflated.

A **negative g** means the benchmark's investment is *below* replacement (δ·K), so the stock would
contract if stepped forward unchanged — this is not hidden. On the toy SAM g ≈ −3.4%, so a zero-trend
recursive run shows a gently contracting capital path; a caller wanting a stationary or growing
baseline re-anchors via the labour and productivity trends (or, in a real build, a benchmark
investment at replacement level).

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
    - **Per-region** — labour-supply growth = population growth compounded with labour-force
      participation growth, i.e. the exact multiplicative step (1+pop)(1+part) each year (both are
      *proportional* annual growth rates, not percentage-point changes), and labour productivity
      (TFP), applied as endowment scales. *(UN WPP 2024, ILO/World Bank, PWT 10.01.)*
    - **Per-sector `sector_productivity`** (structural composition drift) — a sourced
      **labour-productivity** series applied as a **labour-augmenting** term on the sector's labour
      input (a `ProductivityShock(mechanism="labour_augmenting")`), so the output mix shifts
      **endogenously** (a sector with faster measured productivity gains share); shares are a model
      result, not an imposed target. **Honesty (review P1 2026-08-31).** Observed labour productivity
      is $g_{Y/L}=g_A+s_K\,g_{K/L}+s_L\,g_\varphi$ — it mixes true labour-augmenting technology with
      capital deepening, utilisation and composition. This driver does **not** decompose those out, so
      $\varphi_s$ is a **heuristic, illustrative composition lever**, NOT an identified technology
      parameter, and we do **not** claim it removes capital-deepening double-counting. It is put on the
      labour input (not Hicks-neutral TFP) only so it stays a composition lever rather than re-imposing
      an aggregate level; a growth-accounting decomposition (needing sector K/L data the project does
      not vendor) is the documented follow-up. The per-sector **drift** is each sector's cumulative
      labour-productivity level relative to the **VA-share-weighted geometric mean** of all sectors'
      levels — weighted by each sector's **share of value added** $v_i=VA_i/\sum_j VA_j$ (review P1
      2026-08-31: previously the labour composition $s_L$, which weighted a 1%-of-economy sector like a
      99% one). The VA-weighted geometric mean of the biases is 1 **by construction**, so a large
      sector's drift dominates and the composition redistributes without re-imposing an aggregate
      level; this is a transparent normalisation, not an exact GE cost-neutrality. In **multi** mode
      the mean is **region-specific**, so an identical sector rate nets to a different bias per region
      (review P1b). *(EU KLEMS 2023 [EUKLEMS2023]; Penn World Table 10.01 [FeenstraPWT] for the
      aggregate reference.)*
    - **Per-sector `emissions_intensity`** (decarbonisation) — a **price-independent** engine hook
      (`emissions_intensity_scale`) that multiplies the sector's physical emission intensity (so
      covered emissions fall) AND its priced carbon wedge. Because it acts on the intensity the engine
      computed, it works on **real IO/satellite builds** where no `carbon_cost_share` is supplied
      (review P1a) — not only supplied-share toy builds. Via a base-year covered-emissions reference
      the wrapper feeds the engine (all three variants; a **per-region** reference for the multi CGE,
      where an **uncovered** region carries a zero reference without invalidating the covered
      regions), a decarbonising sector shows falling covered emissions measured against the base year.
      *(IEA WEO 2024 [IEA_WEO2024] / NGFS Net Zero 2050 [NGFS].)*
      **One limitation to state plainly.** With **no `CarbonPrice`** in the scenario there is no
      priced carbon and no covered-emissions output at all, so an `emissions_intensity` trajectory has
      **no observable effect** — pair it with a `CarbonPrice`.
  On a **real EXIOBASE build** the trajectory's archetype keys (N/S, BRD/MIL) are bound to the
  build's actual coarse-v3 region/sector labels by the provenance-carrying concordance
  `data/structural/concordance_v2.json` (via `structural_trajectories_for_build`). **v2 is
  GDP-weighted** (review P1 2026-08-29): World Bank regional aggregates mix income levels, so rather
  than assign one unweighted N/S archetype per block, v2 keeps **country-level** archetypes
  ([WorldBankIncome], FY2025 vintage) and blends the member countries' paths by **GDP weights**
  (`block_membership`) — a mixed block such as `RoW_Asia` sits *between* the pure N and S paths. The
  mapped trajectory carries a **composite** provenance naming both the concordance and the trajectory
  artifact; the loader validates the Provenance contract, that every archetype target is a known
  path, and that the v2 weights are finite, ≥ 0 and sum to 1 (review P2 2026-08-29). An unmapped
  label fails loudly, and `run_recursive` **rejects** a run whose trajectory does not differentiate
  any of the build's labels (every label falling through to `__all__`) unless the caller sets
  `DynamicConfig(allow_uniform_fallback=True)` — so a real build cannot silently collapse to the
  uniform default (review P2 2026-08-29).
- **No perfect foresight**; recursive bookkeeping, not intertemporal optimisation.
- Aggregate productivity is Hicks-neutral on primary factors; the per-sector series is a heuristic
  labour-augmenting composition lever (a factor on the sector's labour input, not a TFP transform,
  and not an identified technology term — no capital-deepening decomposition). No other factor-biased
  or vintage-specific technical change.
- Magnitudes are illustrative (toy calibration); the value is the **mechanism** — a static CGE turned
  into a capital-carrying dynamic path, the backbone Phase 7.2 (NGFS) and 7.3 (climate) build on.

## 7. Algorithm

`run_recursive` (in `cge.dynamics.recursive`):

1. **Probe.** Solve one no-shock static run at $t_0$ to read $K_0$ (per capital region) off the
   `capital_dynamics` manifest block (eq $(2)$), the model's sectors, and the per-(region, sector)
   benchmark labour shares $s_L$.
2. **Coverage gate.** If a `StructuralTrajectory` is supplied, reject the run when it differentiates
   none of the build's labels (all `__all__`) unless `allow_uniform_fallback=True` (§6).
3. **Expand physical nature once.** Any Phase-6b `nature_state` pathway is expanded over the FULL
   annual horizon before the loop, so its rate-form start year and recovery hysteresis are not reset
   by per-year slicing (review P1).
4. **Annual loop** over every year in $[t_0, T]$: build the endowment scales (eq $(3)$) and any
   per-sector `emissions_intensity_scale` / labour-augmenting `sector_productivity` shocks (eq $(5)$),
   call the static engine, read `investment_volume`, step the stock (eq $(1)$), and — for **reporting**
   years only — append the result rows. Static overrides passed via `data_overrides` (e.g. a nature
   exposure IOSystem + ENCORE for a physical run) are merged into every year's solve.
5. **Assemble** the `DynamicPath`: concatenated per-reporting-year `ResultSet`, capital path dicts,
   and a `recursive_dynamics` manifest block; the scenario hash is re-stamped over the ORIGINAL
   scenario (nature_state intact) plus the normalised `DynamicConfig`, and each solve-year's child
   hash is recorded.

Complexity is $O(T - t_0)$ static solves (every calendar year), independent of how many years are
reported. The hot path is the per-year static CGE solve; the wrapper itself is cheap bookkeeping.

## 8. Calibration & parameters

| Parameter | Default | Source |
| --- | --- | --- |
| user-cost net return | 4%/yr | [KingRebelo1999] |
| calibration $\delta_c$ (bridge) | 5%/yr | [MillerBlair2009] ch. 5 (PIM) |
| forward $\delta$ (`DynamicConfig.depreciation`) | 5%/yr | [MillerBlair2009] ch. 5; user-set |
| $K_0$ (services→stock) | derived, eq $(2)$ | [Jorgenson1963] |
| labour / participation paths | sourced per region | [UNWPP2024], [ILOSTAT] |
| aggregate TFP path | sourced per region | [FeenstraPWT] |
| sectoral labour productivity | sourced per sector | [EUKLEMS2023] |
| emissions-intensity path | illustrative central path | [IEA_WEO2024], [NGFS] |
| region/sector concordance | GDP-weighted, country-level | [WorldBankIncome] |

The vendored trajectory (`data/structural/trajectories_v1.json`) and concordance
(`data/structural/concordance_v2.json`) are documented, sourced, per-entry-cited artifacts; see
`data/structural/NOTICE.md` for licences and the reproducibility caveat (headline transcribed rates,
not a committed extraction pipeline).

## 9. Validation

Standing model-correctness checks live in `src/cge/validation/suites/dynamics.py` (run by
`cge validate`) and unit tests in `tests/test_dynamics.py` / `tests/test_structural_trajectories.py`:

- **Accumulation identity** eq $(1)$ holds at every consecutive step; a **sparse** horizon gives the
  same closing stock as the full annual horizon.
- The path **starts from the benchmark** (year 0, no shock ⇒ real GDP change 0); with $A=L=1$,
  $K=K_0$ the run is byte-identical to the static benchmark (eq $(3)$).
- **Premature retirement** lowers the closing stock; the multi-region CGE carries a genuinely
  per-region capital path.
- A sourced `StructuralTrajectory` makes the emerging region **outgrow** the developed one; sector
  labour-augmenting biases are **zero-mean** (a high-aggregate region is not uniformly negative) and
  per-region in multi mode; a declining `emissions_intensity` path drives **covered emissions down**
  against the base year — including on a **real IO-backed build** (opt-in `exiobase_live` suite,
  intensity engine-derived, not a supplied `carbon_cost_share`).
- A full **physical `nature_state`** pathway runs end-to-end through `run_recursive` (NatureStress →
  exposure → `ProductivityShock` → CGE), with a deeper degradation producing a larger output loss and
  the water-dependent sector hit harder.

## 10. References

Cited inline by key; full entries in [`docs/references.md`](../references.md): [MillerBlair2009],
[Jorgenson1963], [KingRebelo1999], [Solow1957], [FeenstraPWT], [EUKLEMS2023], [UNWPP2024], [ILOSTAT],
[WorldBankIncome], [IEA_WEO2024], [NGFS].

See `docs/models/macro-aggregates.md` for the GDP/GVA reporting and
[`roadmap.md`](../../roadmap.md) Phase 7 for the pathway stack this unblocks.
