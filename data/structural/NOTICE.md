# Structural trajectories — sources & licences

`trajectories_v1.json` holds **documented, sourced annual growth trajectories** for the Phase 7b.2
structural drivers: per-region **population**, **labour-force participation** and **labour
productivity**, plus per-sector **sectoral productivity** and **emissions intensity**. Each
`(driver, key)` path carries its own citation and confidence in the artifact itself; this file
records the underlying sources and their licences. The figures are **headline trend rates** taken
from the published sources below — a compact, review-friendly artifact, not a re-distribution of the
full source databases. **These are illustrative research-scaffold inputs, not an engagement-grade
consulting baseline** — see `docs/structural-data-pipeline-plan.md` for the reproducible-extraction,
source-share-identification, per-driver-weighting, scenario-pinning and uncertainty work that
consulting use requires, and the interim conservative posture until it lands. The shipped `N`/`S` region and `BRD`/`MIL` sector keys are development/
composition **archetypes**. On a real EXIOBASE build, the provenance-carrying concordance
`concordance_v2.json` (the default; see below) binds each real coarse-v3 build label to its archetype, and
`cge.data.structural.structural_trajectories_for_build(regions, sectors)` emits a trajectory keyed by
the build's OWN labels — so a real build has genuine country differentiation and sectoral composition
drift, not every label collapsing to the global `__all__` default. An unmapped build label fails
loudly rather than silently taking `__all__` (review P1c 2026-08-28).

## Region/sector concordance — `concordance_v2.json` (default)

`concordance_v2.json` is the default; `concordance_v1.json` (a single unweighted archetype per
coarse block) remains loadable for back-compatibility. **v2 fixes the review-3 P1**: World Bank
regional aggregates (`RoW_Asia`, `RoW_Europe`, `RoW_America`) mix income levels, so a *single*
unweighted N/S archetype could not honestly come from an income-classification rule. v2 therefore
keeps **country-level** archetypes and blends them per block with documented **GDP weights**.

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
Three limitations to state plainly (they bound how much a real-build regional trajectory should be
trusted; resolving them is a follow-up that needs per-driver weight data this artifact does not yet
vendor):

1. **One weight table for every region driver.** The single `block_membership` GDP-share table is
   applied to *all three* per-region drivers — population, labour-force participation and
   productivity. That is not the internally consistent choice: population growth should be aggregated
   from population levels (or population weights), labour-force growth from working-age/labour-force
   levels, and productivity growth from output/value-added (or Divisia) weights. Using GDP shares for
   all three is a transparent common-proxy approximation, defensible only because the whole artifact
   is illustrative. Per-driver weight tables are the documented follow-up.
2. **Static weights over a long horizon.** The GDP shares are a single reference-year snapshot held
   fixed across the whole scenario (e.g. to 2050). As economies grow at different rates the true
   shares drift, so a fixed 2025 weight progressively misweights later years. A time-varying weight
   (or re-derived shares per decade) is the follow-up.
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

## Labour-force participation — ILOSTAT / World Bank

- **Source:** ILO modelled estimates (ILOSTAT) and World Bank World Development Indicators —
  labour-force participation rate. <https://ilostat.ilo.org/>
- **Retrieved:** 2026-08-16.
- **Licence:** CC BY 4.0.

## Labour productivity / TFP — Penn World Table 10.01

- **Source:** Feenstra, Robert C., Robert Inklaar and Marcel P. Timmer (2015), "The Next Generation
  of the Penn World Table", *American Economic Review* 105(10), 3150-3182 — *Penn World Table
  version 10.01* (`rtfpna`, TFP growth at constant national prices).
  <https://www.rug.nl/ggdc/productivity/pwt/>
- **Retrieved:** 2026-08-16.
- **Licence:** CC BY 4.0.

## Sectoral productivity — EU KLEMS 2023 release

- **Source:** EU KLEMS & INTANProd 2023 release — sectoral labour-productivity growth by industry.
  <https://euklems-intanprod-llee.luiss.it/>
- **Retrieved:** 2026-08-16.
- **Licence:** CC BY 4.0 (EU KLEMS/INTANProd public release).
- **Note:** the shipped `BRD`/`MIL` sector keys are the toy model's illustrative sectors; the rates
  are headline goods-vs-services **labour-productivity** growth central estimates (output per hour),
  **absolute** per-sector rates (see the artifact `sector_productivity`). Value-added growth
  accounting gives `g_{Y/L}=g_MFP+s_K·g_{K/L}` — observed labour productivity is genuine MFP growth
  PLUS capital deepening. Since the recursive model accumulates capital separately, the wrapper
  **decomposes** the series into an identified MFP term (pipeline step 1b, review P2 2026-09-05).
  **The MFP is identified at SOURCE**, then translated into a labour-augmentation through the
  receiving model's ACTUAL VA nest (CES too, solved numerically; CD reduces to the closed form). Two
  source-identification routes: (a) a sourced **`mfp`** series in the artifact is used directly; else
  (b) `g_MFP = g_{Y/L} − (1−s_L^src)·g_{K/L}` from `sector_productivity` + `capital_deepening` using
  the **source-period labour share** `source_labour_shares` s_L^src (NOT the receiving model's
  benchmark share — the two are different quantities, and conflating them was a review finding). The
  shipped `source_labour_shares` (BRD 0.58, MIL 0.62, `__all__` 0.60) are **illustrative headline
  values**; an EU KLEMS adjacent-period Törnqvist extraction is the consulting-grade follow-up (see
  `docs/structural-data-pipeline-plan.md`). Without a `capital_deepening`/`mfp` series, or where a
  labour share or VA elasticity is missing, the sector falls back to the raw rate as a transparent
  heuristic (deepening not removed); the run manifest records which mode ran per (region, sector).
  Either way it is a
  `ProductivityShock(mechanism="labour_augmenting")` on the sector's labour input. Under a
  Cobb-Douglas VA nest the sector's VA cost falls by φ^{−s_L} exactly (s_L = labour share of VA); for
  CES the engine computes the exact cost. The per-sector *drift* is each sector's cumulative φ level
  relative to an **aggregate-neutral geometric mean** of all sectors' levels — weighted by
  `v_i·s_{L,i}` (each sector's SHARE of value added `v_i=VA_i/ΣVA_j` TIMES its labour share, which
  zeros the aggregate CD log-cost effect Σ_i v_i·s_{L,i}·ln φ_i; review P1 2026-08-31 fixed the weight
  from labour composition to v_i, and review P1 2026-09-03 added the missing s_{L,i}). (This replaces
  the earlier `θ_dev = deviation ** s_L` conversion and the simple-rate `(g_{Y/L}−s_K·g_{K/L})/s_L`
  arithmetic.) On a real EXIOBASE
  build these sector keys map to actual industries via the structural concordance
  (`data/structural/concordance_v2.json`).

## Sectoral capital deepening — EU KLEMS 2023 release

- **Source:** EU KLEMS & INTANProd 2023 release — sector **capital services per hour worked** growth
  (`g_{K/L}`), from the capital-services and hours-worked accounts by industry.
  <https://euklems-intanprod-llee.luiss.it/>
- **Retrieved:** 2026-08-16.
- **Licence:** CC BY 4.0 (EU KLEMS/INTANProd public release).
- **Note:** the shipped `capital_deepening` rates are headline central estimates (goods-producing
  sectors are more capital-intensive and deepen faster ~1.0–1.2%/yr; services ~0.6%/yr). They exist
  to net capital deepening out of observed labour productivity so the sector-productivity driver
  becomes a genuine labour-augmenting technology term (CD finite-change mapping
  `1+g_φ=[(1+g_{Y/L})/(1+g_{K/L})^{s_K}]^{1/s_L}`; s_K=1−s_L from the engine's benchmark factor
  shares) rather than double-counting the capital the recursive model already accumulates. `confidence = low` (illustrative headline rates, not a reproduced
  extraction — see the reproducibility caveat below).

## Emissions intensity — IEA WEO 2024 / NGFS Net Zero 2050

- **Source:** IEA *World Energy Outlook 2024* and NGFS *Net Zero 2050* scenario — CO₂-intensity of
  output decline. <https://www.iea.org/reports/world-energy-outlook-2024> ;
  <https://www.ngfs.net/ngfs-scenarios-portal/>
- **Retrieved:** 2026-08-16.
- **Licence:** IEA terms of use (WEO figures cited, not redistributed); NGFS scenario data resources
  (open, cited).
- **Caveat (review 2026-08-23):** the shipped rates are a **single headline decarbonisation path**
  synthesised from these sources. WEO contains **multiple scenarios** (STEPS, APS, NZE) and NGFS
  spans multiple models/regions/variables; this artifact does **not** yet pin a specific IEA
  scenario / NGFS model / region / variable / aggregation. It is an illustrative-of-method central
  path, `confidence = medium`; a reproducible transform selecting one scenario+model+variable is a
  documented follow-up.

## Reproducibility caveat (review 2026-08-23)

These figures are **headline trend rates transcribed from the cited sources**, not the output of a
committed extraction/aggregation pipeline. In particular PWT 10.01 covers **1950–2019**, so the
2025/2040 productivity knots rest on an (unstated) extrapolation of the historical trend rather than
a direct source value. WPP 2024, ILOSTAT and NGFS do publish explicit forward-looking versioned
datasets that could support a fully reproducible transform (region aggregation → sector mapping →
rate calculation). Treat the shipped `v1` artifact as a **documented, review-friendly illustrative
central path**, not a reproducibly-derived projection; the per-entry `confidence` reflects this.

## Reproducing / updating

The vendored artifact is the small derived object the code and tests consume. To refresh it against
a new data vintage, edit `trajectories_v1.json` with the updated headline rates and their citations
(bump `source_version` and `retrieved`), or point the loader at a build-specific artifact. The
`StructuralTrajectory` contract validates every entry on load (finite rates in a plausible band,
known drivers, per-entry source + confidence), so a malformed or unsourced trajectory fails loudly.
