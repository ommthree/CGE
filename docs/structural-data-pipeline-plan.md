# Plan — consulting-grade structural-data pipeline (Phase 7b.2 readiness)

**Status: IN PROGRESS (2026-09-07). Steps 1a + 1b DONE and the digest is now EXTRACTED FROM REAL
DATA for all six drivers (PWT / WPP / ILOSTAT / EU KLEMS / NGFS). Remaining: broaden EU KLEMS beyond
the single supplied country + add a per-sector NGFS split (1a-data breadth), per-driver/time-varying
weights (1c), fuller scenario pinning (1d), and uncertainty sets (1e).** This document scopes the
work needed to move
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

**Current state (2026-09-09):** the artifact now meets **(1)** (committed extraction scripts, both
`--check`-gated in CI), **(2)** (value-level provenance: dataset/release/table/geography/window/
transform per rate), **(6)** (NGFS scenario pinned to an explicit validated tuple), and **(3)/(4)**
partially (MFP taken directly from EU KLEMS `LP1ConTFP` so no benchmark-share step is used for the
sourced sectors; aggregations are member-weighted with each driver's own weight — but weights are
static and the source labour share is an approximate LAB/VA ratio, not an adjacent-period Törnqvist
share). **(5)** (low/central/high uncertainty bands) is not yet built. The remaining gaps below —
breadth (EU KLEMS single-country, NGFS economy-wide), time-varying weights, Törnqvist shares, and
uncertainty sets — are ordered by how much they change a client answer.

## 1. Gaps and the concrete remediation

### 1a. Reproducible extraction (replaces hand-entered headline rates)
**DONE — build shape landed 2026-09-06; real extractions landed 2026-09-07 and were hardened
2026-09-09 (review-8).** The reproducible *path*: `data/structural/sources/inputs.json` is a
structured source digest (per-driver, per-key knot rates each with a value-level source + confidence),
and `scripts/build_structural_trajectories.py` assembles `trajectories_v1.json` from it (`--check`
gates drift; wired into CI). `scripts/extract_structural_sources.py` (also `--check`-gated in CI)
rebuilds the digest from the raw files. The digest values are **real extractions**, not headline
estimates. The forward-window handling below still applies:
- **Aggregate TFP** — PWT 10.01 `rtfpna`. **PWT 10.01 ends in 2019**, so any 2025–2040 value is an
  *extrapolation assumption*, not an observation; the per-entry provenance marks the historical
  window and the forward knot as a held-trend assumption. A named convergence rule for the forward
  knot (conditional convergence to a frontier rate) remains a possible refinement.
- **Sector productivity / MFP** — EU KLEMS & INTANProd 2024 release, per country-industry. **EU KLEMS
  covers principally the EU, UK, US and Japan** and the supplied file is Austria only; applying its
  archetype rates to every model region is an assumption labelled per region (see 1c).

**EXTRACTED FROM REAL DATA 2026-09-07, HARDENED 2026-09-09 (review-8)**
(`scripts/extract_structural_sources.py`): every driver in the committed digest is a real extraction:
- **PWT 10.01** `rtfpna` — GDP-weighted (rgdpo) over mapped countries per archetype, annualised
  log-change over the 2000–2019 window, `expm1`→proportional; mapped countries only (no default-to-S).
- **UN WPP 2024** — per-country population growth at each knot, **population-weighted** to the
  archetype AND the `__all__` aggregate (not a mean of the archetypes).
- **ILOSTAT LFPR** (`EAP_DWAP_SEX_AGE_RT`, SEX_T, 15+ band) — recent-window annualised trend,
  `expm1`→proportional, **±1%/yr outlier clip**, **≥2015 common-vintage cutoff** (stale series
  dropped, not projected forward decades), UNWEIGHTED archetype mean, held forward, **`confidence
  = low`**.
- **EU KLEMS & INTANProd 2024** growth-accounts workbook (Austria — the sole geography supplied):
  the sourced `mfp` driver is the workbook's own **per-hour TFP contribution `LP1ConTFP`** (used
  directly, review-8 P1); `sector_productivity` is per-hour VA growth `LP1_G` (context); labour share
  is `LAB/VA_CP`; VA weight is `VA_CP`; VA-weighted to archetypes and `__all__`. **No `LP2_G`/`CAP_QI`
  capital-deepening decomposition** — it was dimensionally wrong (`LP2_G` is VA per *person*, not per
  hour) and is retired.
- **NGFS Phase 5** REMIND-MAgPIE 3.3-4.8 / **Below 2°C** / World — emissions intensity **derived** as
  `Emissions|CO2 / GDP|PPP|Counterfactual without damage`, selected by an **explicit validated tuple**
  (`_NGFS`, exact region + single-series variables, resolve-to-exactly-one scenario/model). A knot is
  emitted at **every source year** (each the CAGR to the next), so the path reproduces the source
  intensity at every source year (review-9 P1 — the earlier sparse 2025/2040 pair annualised the
  2040→2100 tail and overstated 2050 intensity ~30%); endpoints validated finite/positive.

All rates are **proportional** annual growth (log/delta-log sources converted with `expm1`, review-8
P3). Each maps source countries/industries to the N/S and BRD/MIL archetypes via committed
`archetype_maps/` and aggregates with the documented per-driver weights. Raw files live in
`data/structural/sources/raw/` (git-ignored, separately licensed; see its README). Parsing is covered
by byte-level fixtures in `tests/test_structural_extractors.py`. **Known scope of THIS extraction
(honest limits, tracked as 1c–1e below):** EU KLEMS is a single country (Austria); NGFS gives an
economy-wide intensity path (no per-sector split); the productivity/participation forward knots hold a
recent historical trend flat.

**Acceptance:** re-running the build reproduces the committed artifact; the extractor's `--check`
diffs a fresh extraction against the committed digest once raw files are present.

### 1b. Source-period factor shares for the growth-accounting identity (review-6 P1, rigorous fix)
**DONE (mechanism) 2026-09-06; CES made globally identified 2026-09-06 — real EU KLEMS Törnqvist
shares still to extract (part of 1a data).** Two-stage identification: MFP is identified at SOURCE
using a `source_labour_shares` s_L^src (a first-class trajectory field), or a sourced `mfp` series is
used directly; then MFP is applied through the receiving model's **actual** VA nest, routed by nest
type — Cobb-Douglas via labour augmentation (`ln φ = ln(1+g_MFP)/s_L`, price-independent), CES via a
**value-added Hicks-neutral engine channel** (`va_hicks_neutral`, A_va = 1+g_MFP scaling the whole VA
aggregate so the VA cost falls by 1/A_va at ALL prices — globally identified, not benchmark-only).
CES sectors are now GLOBALLY identified. The `source_labour_shares` are now EXTRACTED from EU KLEMS
(LAB/VA_CP per section: BRD ≈ 0.61, MIL ≈ 0.65, Austria), no longer illustrative.
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
**DONE (review-9 1c, 2026-09-16).** `concordance_v3.json` (built by `scripts/build_concordance_v3.py`
from v2 + the vendored WPP/PWT) adds a `block_weights` structure and a `driver_weight_class` map:
- **Per-driver weights:** population growth blends by **UN WPP per-country population** shares;
  productivity by **PWT `rgdpo` output** shares; participation still uses the static v2 GDP weights
  (no per-country labour-force series is available — documented proxy).
- **Time-varying weights:** the population class is indexed to the knot years (2025/2035/2050) — a
  block's 2050 blend genuinely differs from its 2025 blend as demographics drift (e.g. Indonesia's
  share of RoW_Asia rises while Korea's falls). Output weights are held at PWT-2019.
- **Residual blocks:** the EXIOBASE rest-of-region aggregates (`W*`, e.g. `RoW_MiddleEast`=100% S)
  have no single ISO3, so they RETAIN their v2 residual share (a documented emerging-economy proxy)
  with the named members renormalised around it; their uncertainty is widened via 1e.
The loader (`_blend_region_path`) selects the driver's weight class and the year-appropriate weights;
a v2 concordance (no `block_weights`) still loads and falls back to the single static GDP weights.

**Acceptance (met):** the concordance carries per-driver, year-indexed weight tables with their own
provenance + a `--check` gate; `_validate_v3_block_weights` enforces each table sums to 1 over mapped
countries; the mapped trajectory's provenance stamps the per-driver/time-varying basis; the review-6
per-driver coverage validation still guards against silent cross-driver divergence.

### 1d. Pinned emissions-intensity scenario
**SCENARIO PINNED 2026-09-09 (review-8); PATH FIDELITY FIXED 2026-09-15 (review-9); per-sector split
still open.** The extraction pins an **explicit validated tuple** (`_NGFS` in
`scripts/extract_structural_sources.py`): NGFS Phase 5, model `REMIND-MAgPIE 3.3-4.8`, scenario
`Below 2°C`, region `World`, intensity **derived** as `Emissions|CO2 / GDP|PPP|Counterfactual without
damage`. A knot is emitted at every source year (each the CAGR to the next), so the piecewise path
reproduces the source intensity at every source year — the earlier sparse 2025/2040 pair annualised
the 2040→2100 tail and held that gentle average from 2040, overstating 2050 intensity ~30% (review-9
P1). Endpoints are validated finite/positive before exponentiation (a net-negative-emissions scenario
raises rather than emitting a negative/NaN scale). Selection resolves the scenario/model to exactly
one published value (raises on none/ambiguous) and requires the CO₂/GDP series each as a single
series — no silent cross-region/variable averaging.
**PER-SECTOR SPLIT DONE (review-9 1d, 2026-09-16).** `scripts/fetch_ngfs_sectoral.py` pulls the
sector-resolved CO₂ series from the IIASA NGFS Phase 5 explorer (via `pyam`, an offline one-off) and
vendors them into the IAMC-wide `ngfs_phase5.csv`; `extract_ngfs_emissions(sector_bundles=…)` then
emits a per-archetype intensity path = the SUM of each bundle's CO₂ over the SAME economy-wide GDP
(NGFS has no sectoral GDP). BRD (goods) = industry energy demand + industrial processes + energy
supply; MIL (services) = transport + residential/commercial; AFOLU is excluded (land use, net-
negative mid-century). Goods decarbonise faster than services (BRD 2050 intensity ≈ 0.13 of 2025 vs
MIL ≈ 0.31), each reproducing its source path at every knot. An economy-wide-only export degrades
gracefully (sectors skipped, `__all__` still emitted).

**Acceptance (met):** manifest names the full scenario tuple; a reader can re-pull the same series
(incl. the sectoral fetch script); per-sector paths are emitted and validated.

### 1e. Uncertainty / sensitivity sets
**DONE (review-9 1e, 2026-09-16).** `scripts/build_uncertainty_sets.py` emits
`trajectories_v1_low.json` / `trajectories_v1_high.json` alongside the central artifact:
- **EU KLEMS sector drivers** (`mfp`, `sector_productivity`): an EMPIRICAL band = the min/max across
  the workbook's alternative trend windows {2017–21 (central), 2015–21, 2010–21} — directly capturing
  the COVID-window sensitivity review-9 flagged.
- **All other drivers**: a confidence-tiered relative band (low ±50% / medium ±25% / high ±10%) on
  the central annual rate; "low" is uniformly the weaker-growth / slower-decarbonisation world.
`cge.data.structural.structural_trajectory_variants(regions, sectors)` returns the three mapped
trajectories for a sweep; a `--check` gate keeps the variants reproducible.

**Acceptance (met):** a client result can report an interval by running {low, central, high} and
comparing — the variants are monotone (low ≤ central ≤ high for growth) and provenance-stamped.

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
- The `source_labour_shares` are now EXTRACTED from EU KLEMS (LAB/VA_CP per ISIC section, Austria),
  validated finite in (0,1] with value-level provenance; broadening beyond the single supplied country
  is part of the 1a-data breadth follow-up.
- The bundled trajectory must not be represented to a client as an engagement-grade baseline; it is a
  scenario-experimentation scaffold.
