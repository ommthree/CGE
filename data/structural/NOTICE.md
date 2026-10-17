# Structural trajectories — sources & licences

`trajectories_v1.json` holds **documented, sourced annual growth trajectories** for the Phase 7b.2
structural drivers: per-region **population** (total, plus a **15+ working-age** driver when the WPP
by-age file is vendored — the labour identity is 15+ population × 15+ LFPR; the by-age `/data`
endpoint is auth-gated, so absent it falls back to total population with an explicit caveat),
**labour-force participation** and **labour productivity**, plus per-sector **sectoral productivity**
and per-channel **emissions intensity** (blended per sector at the cumulative LEVEL, not an
annual-rate average — review-13). Each `(driver, key)` path carries its own citation and confidence in the artifact
itself; this file records the underlying sources and their licences. The values are the **output of a
committed, --check-gated extraction pipeline** (`scripts/extract_structural_sources.py` →
`build_structural_trajectories.py`) over the real published sources below — a compact,
review-friendly artifact, not a redistribution of the full databases. The pipeline is now built
(reproducible extraction, source-share MFP identification, per-driver/time-varying concordance
weights, pinned NGFS scenario with per-channel intensities, deterministic sensitivity cases); the
**remaining honest gaps** — EU KLEMS single-country breadth, the NGFS AEEI (FE/output) term, gas-
indexed emissions paths, per-region satellite-weighted channel blends, and quantified (probabilistic)
uncertainty — are tracked in `docs/structural-data-pipeline-plan.md`, so this is a strong research/
scenario scaffold approaching but not yet a fully engagement-grade consulting baseline. The shipped
`N`/`S` region and `BRD`/`MIL` sector keys are development/composition **archetypes**. On a real
EXIOBASE build, the provenance-carrying concordance `concordance_v3.json` (the default — per-driver,
time-varying weights + per-sector emissions-channel blends; see below) binds each real coarse-v3
build label to its archetype, and
`cge.data.structural.structural_trajectories_for_build(regions, sectors)` emits a trajectory keyed by
the build's OWN labels — so a real build has genuine country differentiation and sectoral composition
drift, not every label collapsing to the global `__all__` default. An unmapped build label fails
loudly rather than silently taking `__all__` (review P1c 2026-08-28).

## Region/sector concordance — `concordance_v3.json` (default)

`concordance_v3.json` is the default; `concordance_v2.json` (single static GDP weights) and
`concordance_v1.json` (a single unweighted archetype per coarse block) remain loadable for
back-compatibility. **v3 (review-9 1c) adds PER-DRIVER, TIME-VARYING `block_weights`**: population
blends by UN WPP per-country population shares that DRIFT across the 2025/2035/2050 knots,
productivity by PWT output shares; participation uses the static v2 GDP weights as an explicit,
declared proxy (no per-country labour-force series). Every trajectory region driver must be
explicitly classed and every block covered, so v3 can never silently fall back to the static
weights (review-10 P2#4). **v2 fixed the review-3 P1**: World Bank regional aggregates (`RoW_Asia`,
`RoW_Europe`, `RoW_America`) mix income levels, so a *single* unweighted N/S archetype could not
honestly come from an income rule — v2 keeps **country-level** archetypes and blends them per block.

- **`country_archetype`:** each country → `N` (advanced proxy) or `S` (emerging proxy) by World Bank
  income classification (FY2025 vintage; e.g. Russia → `N` under the current high-income
  classification, correcting the v1 RU→S vintage error).
  <https://datahelpdesk.worldbank.org/knowledgebase/articles/906519>
- **`block_membership`:** for each coarse-v3 build region, the member countries and their **GDP
  weights** (summing to 1 per block). A mixed block's archetype path is the GDP-weighted blend of its
  members' country paths (`_blend_region_path`), so `RoW_Asia` sits *between* the pure `N` and `S`
  paths rather than collapsing wholly to one. Weights are validated finite, ≥ 0 and sum-to-1 on load.
- **`sector_archetype`:** sector → `BRD`/`MIL` by the ISIC Rev.4 / EU KLEMS goods-vs-services split
  (goods-producing → `BRD`; distribution/transport/services → `MIL`).
  <https://unstats.un.org/unsd/classifications/Econ/isic>
- **Licence:** derived mapping (CC BY 4.0); underlying groupings CC BY 4.0.
- **Provenance:** a mapped trajectory carries a **composite** provenance
  (`structural-concordance-v2+structural-trajectories-v1`) naming both artifact identities, so the
  concordance source/version is auditable in the run manifest (review P2 2026-08-29).

### Aggregation-weighting limitations (review P1 2026-08-31 — documented, not yet resolved)

The GDP-weighted blend is a **documented simplification**, not an internally consistent aggregation.
Limitations 1 and 2 below were **RESOLVED by concordance v3** (review-9 1c) and are kept only to
record the history; limitation 3 still stands:

1. **~~One weight table for every region driver.~~ RESOLVED (v3).** v3's `block_weights` +
   `driver_weight_class` blend each driver with its OWN class: population (and 15+ working-age
   population) by UN WPP population shares, productivity by PWT output shares; participation still
   uses the static GDP weights as an explicit, declared proxy (no per-country labour-force series).
2. **~~Static weights over a long horizon.~~ RESOLVED (v3).** The population weights are YEAR-INDEXED
   (per-knot 2025/2035/2050 WPP shares), so a block's blend genuinely drifts over the horizon; output
   weights are held at PWT-2019.
3. **Residual `W*` blocks are coarse.** EXIOBASE's five rest-of-world aggregates (`WA`, `WL`, `WE`,
   `WF`, `WM`) are opaque multi-country regions with no within-block country detail, so each is
   assigned a single archetype. In particular **`RoW_MiddleEast` (`WM`) is 100% `S`** even though the
   real region contains high-income Gulf economies — an acknowledged mis-assignment that cannot be
   split without country detail EXIOBASE does not provide at this granularity. The other residuals
   (`WA`/`WL`/`WE`/`WF`) at least sit *inside* mixed blocks whose country members carry most of the
   weight, so their coarseness is diluted; `WM` stands alone and is the least reliable region path.

## Population — UN World Population Prospects 2024

- **Source:** United Nations, Department of Economic and Social Affairs, Population Division —
  *World Population Prospects 2024*, Medium variant. <https://population.un.org/wpp/>
- **Retrieved:** 2026-08-16.
- **Licence:** CC BY 3.0 IGO.

## Labour-force participation — ILOSTAT

- **Source:** ILOSTAT indicator `EAP_DWAP_SEX_AGE_RT` (labour-force participation rate, SEX_T total,
  15+ band). The implemented extraction reads **ILOSTAT only** — no World Bank series is used
  (review P3 2026-09-15). <https://ilostat.ilo.org/>
- **Retrieved:** 2026-09-09.
- **Licence:** CC BY 4.0.
- **Confidence: LOW.** Historical recent-window trend held flat; ±1%/yr outlier clip; a country
  needs ≥2 observations inside the recent ≤10-year window (else dropped, not back-filled from decades
  of history); unweighted archetype mean (no labour-force size weights in this file). Sensitivity-
  grade, not a central consulting input.

## Labour productivity / TFP — Penn World Table 10.01

- **Source:** Feenstra, Robert C., Robert Inklaar and Marcel P. Timmer (2015), "The Next Generation
  of the Penn World Table", *American Economic Review* 105(10), 3150-3182 — *Penn World Table
  version 10.01* (`rtfpna`, TFP growth at constant national prices).
  <https://www.rug.nl/ggdc/productivity/pwt/>
- **Retrieved:** 2026-08-16.
- **Licence:** CC BY 4.0.

## Sectoral productivity / MFP — EU KLEMS & INTANProd 2024 release

- **Source:** EU KLEMS & INTANProd 2024 growth-accounts release — per-hour TFP contribution
  (`LP1ConTFP`, the sourced `mfp` driver) and per-hour value-added growth (`LP1_G`,
  `sector_productivity`), by industry, **Austria** (the sole geography in the supplied workbook).
  <https://euklems-intanprod-llee.luiss.it/>
- **Retrieved:** 2026-09-09.
- **Licence:** CC BY 4.0 (EU KLEMS/INTANProd public release).
- **Note:** the shipped `BRD`/`MIL` sector keys are the toy model's illustrative sectors; the rates
  are headline goods-vs-services **labour-productivity** growth central estimates (output per hour),
  **absolute** per-sector rates (see the artifact `sector_productivity`). Value-added growth
  accounting gives `g_{Y/L}=g_MFP+s_K·g_{K/L}` — observed labour productivity is genuine MFP growth
  PLUS capital deepening. Since the recursive model accumulates capital separately, the wrapper
  **decomposes** the series into an identified MFP term (pipeline step 1b, review P2 2026-09-05).
  **The MFP is identified at SOURCE**, then applied through the receiving model's ACTUAL VA nest,
  ROUTED BY NEST TYPE so it is globally correct: a Cobb-Douglas sector via labour augmentation
  (`ln φ = ln(1+g_MFP)/s_L`, price-independent), a CES sector via the **value-added Hicks-neutral**
  engine channel (`mechanism="va_hicks_neutral"`, A_va = 1+g_MFP scaling the whole VA aggregate so
  the VA cost falls by 1/A_va at every price — a genuinely identified VA technology term for CES, not
  a benchmark-only surrogate). Two source-identification routes: (a) a sourced **`mfp`** series in the
  artifact is used directly; else (b) `g_MFP = g_{Y/L} − (1−s_L^src)·g_{K/L}` from
  `sector_productivity` + `capital_deepening` using the **source-period labour share**
  `source_labour_shares` s_L^src (NOT the receiving model's benchmark share — conflating them was a
  review finding). **As shipped, route (a) is active:** the `mfp` series is EXTRACTED from EU KLEMS —
  the workbook's own per-hour TFP contribution `LP1ConTFP` — so no capital-deepening subtraction is
  needed (review P1 2026-09-09; the earlier `LP2_G`/`CAP_QI` decomposition was dimensionally wrong —
  `LP2_G` is VA per *person*, not per hour — and is retired). The `source_labour_shares` (BRD 0.613,
  MIL 0.653) are the EU KLEMS latest-year LAB/VA_CP ratio — an APPROXIMATE source labour share, not
  yet an adjacent-period Törnqvist share (the consulting-grade refinement, see
  `docs/structural-data-pipeline-plan.md`). EU KLEMS here is **Austria only** (the sole geography in
  the supplied file), a single-country proxy for the sector archetypes. Without an `mfp` (or
  `capital_deepening`) series, or where a labour share or VA elasticity is missing, the sector falls
  back to the raw rate as a transparent heuristic; the run manifest records which mode ran per
  (region, sector). The per-sector *drift* normalises the cumulative MFP
  levels across sectors so the composition drift re-imposes no aggregate VA cost — the VA-share
  (`v_i=VA_i/ΣVA_j`) weighted aggregate log-cost effect Σ_i v_i·ln(1+g_MFP,i) is zeroed (the same
  criterion in MFP terms for both channels, since the CD labour cost effect s_L·ln φ = ln(1+g_MFP)
  equals the VA-channel ln A_va; the aggregate level is carried separately by the per-region TFP
  endowment scale). (This replaces the earlier `θ_dev = deviation ** s_L` conversion, the VA-share ×
  labour-share weighting, the simple-rate `(g_{Y/L}−s_K·g_{K/L})/s_L` arithmetic, and the
  benchmark-only CES labour-augmentation translation — all retracted.) On a real EXIOBASE
  build these sector keys map to actual industries via the structural concordance
  (`data/structural/concordance_v2.json`).

## Sectoral capital deepening — RETIRED (review P1 2026-09-09)

The EU KLEMS `capital_deepening` driver has been **removed**. It existed only to net capital
deepening out of observed labour productivity via `g_MFP = g_{Y/L} − s_K·g_{K/L}`, but the earlier
extraction used `LP2_G` (value added per *person* employed) and `CAP_QI` in a way that was
dimensionally inconsistent (mixing a per-person LP with a per-hour capital-services index) and
produced the wrong goods-vs-services ordering. The sourced MFP is now taken **directly** from the EU
KLEMS workbook's own per-hour TFP-contribution series `LP1ConTFP` (route (a) above), which already
nets out capital deepening at source — so no separate `capital_deepening` series is needed or
shipped. The identity and CES/CD routing above still apply; only the source of the MFP term changed.

## Emissions intensity (per channel) — NGFS Phase 5, REMIND-MAgPIE 3.3-4.8 / Below 2°C

- **Source:** NGFS Phase 5 scenario explorer, model `REMIND-MAgPIE 3.3-4.8`, scenario **`Below 2°C`**,
  region `World`. <https://data.ece.iiasa.ac.at/ngfs/>
- **Retrieved:** 2026-09-19.
- **Licence:** NGFS scenario data terms (cited, not redistributed).
- **Per-channel intensity (review-11 P1 2026-09-19):** emissions intensity is built **per channel**,
  each CO₂ over the activity that actually drives it — NOT one BRD/MIL bundle over final energy, and
  NOT CO₂/GDP:
  - **electricity** = `CO2|Supply|Electricity / Secondary Energy|Electricity` (generation intensity)
  - **industry** = `CO2|Demand|Industry / Final Energy|Industry` (fuel combustion)
  - **process** = `CO2|Industrial Processes / (Production|Steel + |Cement)` (physical output driven)
  - **transport** / **buildings** = their CO₂ over their final energy
  - **`__all__`** = (Energy + Industrial Processes) CO₂ / total Final Energy (fallback)

  The concordance (`sector_emissions_channel`) maps each model sector to a **weighted BLEND** of these
  channels (emissions are mixed-source: metals = process + combustion), so electricity gets the
  generation-decarbonisation path, not industry combustion. The blend is taken **at the cumulative
  LEVEL** — a sector's factor is `Fₛ(t) = Σ wₖ Fₖ(t)`, `Fₖ(t) = ∏_{τ<t}(1+rₖ,τ)` — because the
  weights are baseline emission-source shares; a per-year average of the channel *rates* would
  silently rebalance those shares every year (~5% level error on the shipped metals/chemicals paths,
  review-13 P1#2). The weights must be a non-empty convex combination (validated — review-13 P2).
  Model/scenario/region are an **explicit validated tuple** (`_NGFS`), and **every configured
  CO₂/activity variable is MANDATORY** (a missing one raises — no silent economy-wide degrade). A knot
  is emitted at every positive source year, each the CAGR to the next; the electricity channel turns
  net-negative ~2090 (grid with net CCS) so it is **truncated at its last positive year** and held
  flat, and a channel that turns non-positive within the client horizon (before 2050) raises (the
  Net-Zero net-negative case needing a gross-intensity + removals split).
- **KNOWN GAP → `confidence = low`:** the energy-per-output term FE/output (autonomous energy
  efficiency) is **not sourceable per sector** from NGFS (it publishes no sector output/value-added)
  and the default Leontief CGE does not reproduce it. So the drivers are **combustion / energy-
  emissions intensities, NOT the full CO₂/output** the engine multiplier nominally represents; they
  omit the efficiency term and thereby understate decarbonisation. This is documented, not fabricated;
  closing it needs a sector-output source + an engine channel for AEEI. AFOLU/removals stay a separate
  concern (excluded). The per-channel series are fetched by `scripts/fetch_ngfs_sectoral.py` (pyam).

## Reproducibility (review 2026-09-09 / 2026-09-15 — the transcription caveat is retired)

The shipped figures are the **output of a committed extraction pipeline**, not hand-transcribed
headline rates. `scripts/extract_structural_sources.py` reads the raw published files
(`data/structural/sources/raw/`, git-ignored) and writes `data/structural/sources/inputs.json`
(source digest); `scripts/build_structural_trajectories.py` assembles `trajectories_v1.json` from
that digest. **Scope of the CI gate (stated precisely):** CI runs `--check` on **digest→artifact**
(fully validated in CI) and on **raw→digest** — but the raw files are NOT in CI (large, separately
licensed), so that second check is a no-op there and only validates locally when the raw files are
present. So the pipeline is **locally reproducible from separately-acquired inputs**; it is not a
claim that CI re-derives the digest from raw sources. To make the raw→digest step auditable without
redistributing the files, `sources/raw_manifest.json` records each raw file's SHA-256 + byte size —
re-acquire the cited releases and compare. Rates are **proportional** annual growth (log/delta-log
sources are converted with `expm1`). PWT 10.01 ends in **2019**, so the productivity knots are held
forward (an assumption, flagged as such per-entry); WPP/NGFS supply genuinely forward-looking values.
**Aggregation weights (review-9 1c 2026-09-16):** `concordance_v3.json` blends each region driver with
its OWN, YEAR-INDEXED weight class — population by UN WPP per-country population shares that drift
across 2025/2035/2050, productivity by PWT output shares; participation uses the static v2 GDP weights
(no per-country labour-force series). The EXIOBASE W* residual aggregates keep their v2 proxy share.

**Sensitivity cases (review-9 1e; relabelled review-10 P1#2; per-driver direction review-11 P2#2):**
`trajectories_v1_adverse.json` / `_favourable.json` (built by `scripts/build_uncertainty_sets.py`)
are **DETERMINISTIC STRESS CASES, NOT statistical uncertainty intervals or output bounds** — NGFS
scenarios are not forecasts and carry no probabilities. Direction is **per driver**: adverse = lower
growth for productivity/population/participation/sector-productivity, and slower (less-negative)
decarbonisation for emissions intensity. EU KLEMS sector drivers use the empirical trend-window
envelope (2017–21 central / 2015–19 / 2010–19 pre-COVID); other drivers a confidence-tiered stress.
Use `structural_trajectory_variants()` to sweep {adverse, central, favourable}, and
`structural_sensitivity_bounds()` for the **realised per-output min/max** (the input labels do not
bound a nonlinear CGE output). Genuine, probability-weighted **quantified uncertainty remains open**.

**Remaining limitations** (not defects): EU KLEMS is single-country (Austria — a breadth expansion is
the main open item); forward participation/productivity knots hold the recent trend flat; the region
archetypes are N/S and the productivity/MFP sector archetypes BRD/MIL; the NGFS emissions channels
capture combustion/energy-emissions intensity but **not** the FE/output efficiency term (AEEI — not
sourceable per sector from NGFS, not reproduced by the Leontief CGE); and quantified probabilistic
uncertainty is not built (the sensitivity cases are deterministic). See
`docs/structural-data-pipeline-plan.md` (1a-breadth, 1d-AEEI).

## Reproducing / updating

To refresh against a new data vintage: drop the updated raw files into
`data/structural/sources/raw/` (see its README for exact downloads/filenames) and run
`python scripts/extract_structural_sources.py` then `python scripts/build_structural_trajectories.py`
(bump `source_version`/`retrieved` in the digest provenance). The `--check` mode of each script
fails if the committed artifact is stale versus the raw sources. The `StructuralTrajectory` contract
validates every entry on load (finite rates in a plausible band, known drivers, per-entry source +
confidence), so a malformed or unsourced trajectory fails loudly.
