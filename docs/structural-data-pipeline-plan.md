# Plan — consulting-grade structural-data pipeline (Phase 7b.2 readiness)

**Status: IN PROGRESS (2026-09-06). Steps 1a and 1b BUILT (mechanism + reproducible-build shape);
real source extractions (1a data), per-driver/time-varying weights (1c), scenario pinning (1d) and
uncertainty sets (1e) remain.** This document scopes the work needed to move
the Phase 7b.2 structural trajectories from an *illustrative research scaffold* to an
*engagement-grade baseline* fit for climate-risk consulting. It is the response to the standing
**P1-readiness** finding raised across reviews 4–6: the shipped `data/structural/trajectories_v1.json`
and `concordance_v2.json` are transparently disclosed as headline/illustrative (see
`data/structural/NOTICE.md`), which is honest but is **not** the same as a reproducible, sourced,
uncertainty-quantified dataset. It also absorbs the **review-6 P1** on identification: the current
growth-accounting decomposition is a *model-calibrated Cobb-Douglas* approximation, and the rigorous
fix (source-period MFP estimation translated through the actual production nest) belongs here.

The engine and wrapper are sound and reviewed; this plan is about the **inputs**, not the code paths
that consume them. Nothing in here blocks scenario-experimentation use of the model today — it blocks
representing the *bundled trajectory* as a client baseline.

## 0. What "consulting-grade" means here

A trajectory artifact is engagement-grade when every number in it is:

1. **Reproducible** — produced by a committed extraction script from a pinned source release, not
   hand-entered; re-running the build reproduces the artifact's CONTENT (the `--check` gate compares
   parsed JSON, so incidental formatting differences are ignored while every value must match).
2. **Attributed at the value level** — each rate names its source dataset, release, table/variable,
   geography, industry, vintage year(s), and the transformation applied.
3. **Method-consistent** — the growth-accounting identity uses *source-economy* factor shares
   (adjacent-period average / Törnqvist), not the receiving CGE model's benchmark shares.
4. **Weighted per driver and over time** — population, participation and productivity aggregations
   use their *own* weights (population, labour force, output), and weights vary over the horizon.
5. **Uncertainty-quantified** — every headline rate ships with a documented low/central/high band
   and a sensitivity procedure, so a client result carries an interval, not a point.
6. **Scenario-pinned** — the emissions-intensity path names a specific NGFS/IEA model, scenario,
   region, variable and aggregation, versioned to a release.

The current artifact meets (2) partially (per-entry citation prose + confidence) and none of
(1),(3),(4),(5),(6) rigorously. The gaps below are ordered by how much they change a client answer.

## 1. Gaps and the concrete remediation

### 1a. Reproducible extraction (replaces hand-entered headline rates)
**PARTIALLY DONE — build shape landed 2026-09-06; real extractions still to come.** The reproducible
*path* now exists: `data/structural/sources/inputs.json` is a structured source digest (per-driver,
per-key knot rates each with source + confidence), and `scripts/build_structural_trajectories.py`
assembles `trajectories_v1.json` from it (`--check` gates drift; wired into CI). Today the digest
holds the same illustrative headline figures, so the build is numerically a no-op — the point is that
real per-country/industry pulls now drop into the digest without touching the build or the schema.
**Now (data):** the digest values are still headline central estimates, not reproducible pulls.
**Target (data):** replace each digest entry with a committed extraction from a pinned source release:
- **Population** — UN WPP 2024, Medium variant, per-region population growth by 5-year period,
  interpolated to annual. Vendored input digest + release id.
- **Participation** — ILOSTAT modelled estimates + World Bank labour-force participation, per region.
- **Aggregate TFP** — PWT 10.01 `rtfpna`. **PWT 10.01 ends in 2019**, so any 2025–2040 value is an
  *extrapolation assumption*, not an observation. The build must (i) mark the historical window vs
  the assumption window explicitly in provenance, and (ii) take the forward assumption from a named,
  defensible convergence rule (e.g. conditional convergence to a frontier growth rate), not an
  unlabelled hand number.
- **Sector labour productivity & capital deepening** — EU KLEMS & INTANProd 2023 release, per
  country-industry: output-per-hour growth and capital-services-per-hour growth. **EU KLEMS covers
  principally the EU, UK, US and Japan through ~2020**; applying its archetype rates to every model
  region is an assumption that must be labelled per region (see 1c).

**Acceptance:** re-running the build reproduces the committed artifact; a CI job (opt-in, needs the
vendored source digests) diffs a fresh build against the committed file.

### 1b. Source-period factor shares for the growth-accounting identity (review-6 P1, rigorous fix)
**DONE (mechanism) 2026-09-06 — real EU KLEMS Törnqvist shares still to extract (part of 1a data).**
The two-stage identification below is now implemented: MFP is identified at SOURCE using a
`source_labour_shares` s_L^src (a first-class trajectory field), or a sourced `mfp` series is used
directly; then MFP is translated into the labour-augmenting shock through the receiving model's
**actual** VA nest — CES solved numerically (`_mfp_to_labour_aug_log`), CD as the closed form. CES
sectors are now IDENTIFIED, not dropped to the heuristic. The shipped `source_labour_shares` (BRD
0.58 / MIL 0.62 / __all__ 0.60) are illustrative headline values; the EU KLEMS Törnqvist extraction
that replaces them is part of the 1a data work.
**Was (review-6 P1):** the wrapper netted capital deepening out with the CD mapping
`1+g_φ = [(1+g_{Y/L})/(1+g_{K/L})^{s_K}]^{1/s_L}`, using the **receiving model's benchmark share**
for the source-side step and applying it **only to σ_va=1 sectors** (CES fell back to heuristic).
**Why it is only an approximation:** standard growth accounting (OECD productivity methodology) uses
*source-economy*, adjacent-period average (Törnqvist) cost shares to identify MFP, and the mapping
from an MFP series into a *labour-augmenting* shock in a CES nest depends on the elasticity and
calibration, not just s_L.
**Target — a two-stage identification, done in the data build, not the wrapper:**
1. **Estimate sector MFP at source** using EU KLEMS source-period Törnqvist factor shares:
   `g_MFP = g_{Y/L} − s_K^{src}·g_{K/L}` with `s_K^{src}` the average-period source capital share.
   Ship `mfp` as a first-class sourced series alongside `sector_productivity` and `capital_deepening`.
2. **Translate MFP → labour-augmenting shock through the receiving nest.** For σ_va=1 this is the
   current closed form; for σ_va≠1 solve the finite-change labour-augmentation that reproduces the
   MFP-implied unit-cost change under the sector's actual CES dual, numerically per sector.

**Acceptance:** the identified g_φ no longer depends on the receiving model's benchmark share for
the source-side step; the manifest records `identification=source_share_tornqvist`; CES sectors are
identified (not dropped to heuristic); a known-answer test pins both the σ=1 and a σ≠1 case.

### 1c. Per-driver, time-varying aggregation weights (replaces one static GDP table)
**Now:** one static GDP-share `block_membership` table blends country archetypes into coarse regions,
and the SAME weights are applied to population, participation AND productivity, held fixed to a single
year (documented caveat in `NOTICE.md`); residual `W*` blocks are coarse single-archetype aggregates
(e.g. `RoW_MiddleEast`=100% S).
**Target:**
- **Per-driver weights:** population growth blended by *population* weights, participation by
  *labour-force* weights, productivity by *output/GDP* weights.
- **Time-varying weights:** weights indexed to the projection year (population/GDP shares drift), so a
  block's 2040 blend differs from its 2025 blend.
- **Residual blocks:** replace 100%-single-archetype `W*` blocks with genuine multi-country blends
  where source data exists, or explicitly widen their uncertainty band (1e) where it does not.

**Acceptance:** the concordance carries per-driver weight tables with their own provenance; the
per-driver validation added in review-6 (each present driver must resolve every mapped archetype)
already guards the artifact against silent cross-driver divergence.

### 1d. Pinned emissions-intensity scenario
**Now:** `emissions_intensity` rates cite "IEA WEO 2024 / NGFS Net Zero 2050" in prose without an
exact model/scenario/variable/region/aggregation.
**Target:** pin a specific NGFS phase + model (e.g. `NGFS Phase 5, REMIND-MAgPIE, Net Zero 2050`),
the exact variable (e.g. `Emissions|CO2|Energy` intensity per output), region mapping and temporal
aggregation, versioned to the NGFS release, with the extraction in the build script.

**Acceptance:** manifest names the full scenario tuple; a reader can re-pull the same series.

### 1e. Uncertainty / sensitivity sets
**Now:** a per-entry `confidence` string (low/medium/high), no quantified band.
**Target:** every rate ships low/central/high; the wrapper already accepts custom trajectories, so a
sensitivity run is a sweep over {low, central, high} trajectory variants. Add a documented procedure
and a helper that emits the three variants from the build.

**Acceptance:** a client-facing result reports an interval from the low/high trajectories, not a
single point.

## 2. Sequencing

1. **Extraction skeleton (1a)** — script + provenance manifest + CI reproduce-check, reproducing the
   *current* headline numbers first (no numeric change), to lock the pipeline shape.
2. **Emissions scenario pin (1d)** and **per-driver weights (1c)** — independent, parallelisable.
3. **Source-share MFP identification (1b)** — depends on the EU KLEMS extraction from (1a).
4. **Uncertainty sets (1e)** — last, once central series are reproducible.

Each step lands its own artifact version bump (`trajectories_v2`, `_v3`, …) with the prior retained
for back-compat, matching how `concordance_v1`→`v2` was handled.

## 3. What does NOT change

- The engine, the recursive wrapper, the labour-augmenting channel, the aggregate-neutral
  normalisation, the log-space decomposition, and the per-(region,sector) identified/heuristic
  bookkeeping are all reviewed and stay as-is; this plan only changes the **inputs** they consume and
  the **source-side** identification step (1b), which moves into the data build.
- The honest disclosure in `NOTICE.md` stays until each gap is closed; as a step lands, its caveat is
  removed and replaced by the reproducible provenance.

## 4. Interim posture (shipped in review-6/7 remediation)

Until this plan is fully executed, the shipped behaviour is deliberately conservative and honestly
labelled:
- MFP is identified at source (a sourced `mfp` series, or LP+deepening netted with a
  `source_labour_shares` s_L^src) and applied through the sector's actual nest, routed by nest type.
  The manifest's `sector_productivity_mode` records the fine-grained mode per (region, sector):
  `sourced_mfp` / `derived_source_share` (source-identified), `model_share_approx` (LP+deepening but
  only the receiving model's benchmark share available — an approximation), or `raw_lp_heuristic`.
- **CES sectors are now GLOBALLY identified** (review P1 2026-09-06, closing the earlier
  benchmark-only limitation): a CES sector routes through a dedicated **value-added Hicks-neutral
  engine channel** (`ProductivityShock(mechanism="va_hicks_neutral")`) — a per-sector multiplier
  A_va = 1+g_MFP that scales the whole VA aggregate, so the VA unit cost falls by exactly 1/A_va at
  **every** price vector, reproducing the source MFP shift globally (not just at benchmark prices).
  Cobb-Douglas sectors use the price-independent labour-augmentation map ln φ = ln(1+g_MFP)/s_L. The
  benchmark-calibrated CES labour-augmentation translation (and its feasibility failures) is retired.
- The shipped `source_labour_shares` are illustrative headline values (validated finite in (0,1] with
  value-level provenance); the EU KLEMS Törnqvist extraction that replaces them is part of the 1a data
  work.
- The bundled trajectory must not be represented to a client as an engagement-grade baseline; it is a
  scenario-experimentation scaffold.
