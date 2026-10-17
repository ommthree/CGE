# Plan — consulting-grade structural-data pipeline (Phase 7b.2 readiness)

**Status: as of review-13 (2026-10-17), steps 1a/1b/1c/1d/1e are built — the digest is EXTRACTED
FROM REAL DATA (PWT / WPP / ILOSTAT / EU KLEMS / NGFS), with per-driver/time-varying
concordance weights (1c, `concordance_v3.json`), PER-CHANNEL NGFS emissions intensities blended per
sector at the LEVEL (review-13 P1#2 — Σ wₖ·Fₖ(t), baseline source shares, not an annual-rate
average), and adverse/central/favourable deterministic sensitivity cases (1e). The labour identity
uses the **15+ population** driver (`working_age_population`, WPP by-age indicator-70 downloader +
dedicated 15+ concordance weight class) when the by-age file is vendored; otherwise it falls back to
total population with an EXPLICIT documented caveat (the by-age `/data` endpoint is auth-gated).
Remaining honest gaps (NOT yet engagement-grade): EU KLEMS single-country breadth; the NGFS AEEI
(FE/output) term + gas-indexed emissions paths + per-region satellite-weighted channel blends (all
sign-posted below); and quantified (probabilistic) uncertainty. This document scopes the work that
moved** the Phase 7b.2 structural trajectories from an *illustrative research scaffold* toward an
*engagement-grade baseline* fit for climate-risk consulting. It is the response to the standing
**P1-readiness** finding raised across reviews 4–6: the shipped `data/structural/trajectories_v1.json`
and concordance were originally disclosed as headline/illustrative (see `data/structural/NOTICE.md`),
which is honest but is **not** the same as a reproducible, sourced dataset. It also absorbs the
**review-6 P1** on identification: the current growth-accounting decomposition is a *model-calibrated
Cobb-Douglas* approximation, and the rigorous fix (source-period MFP estimation translated through
the actual production nest) belongs here.

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

**Current state (2026-09-19):** the artifact meets **(1)** (committed extraction scripts, all
`--check`-gated in CI), **(2)** (value-level provenance per rate), **(4)** (per-driver, year-indexed
concordance weights — population by WPP, productivity by PWT output; review-9 1c), **(6)** (NGFS
scenario pinned to an explicit validated tuple, per-channel intensities routed by the concordance;
review-11 1d), and **(3)** partially (MFP taken directly from EU KLEMS `LP1ConTFP`; the source labour
share is still an approximate LAB/VA ratio, not an adjacent-period Törnqvist share). **(5)** is met
only as **deterministic sensitivity cases** (adverse/central/favourable) — genuine probability-
weighted **quantified uncertainty is still open**. Two substantive gaps remain, ordered by how much
they change a client answer: EU KLEMS single-country **breadth**, and the NGFS **AEEI/FE-output** term
(unsourceable per sector + needs an engine channel). See 1a-breadth and 1d-AEEI below.

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
- **NGFS Phase 5** REMIND-MAgPIE 3.3-4.8 / **Below 2°C** / World — emissions intensity derived **PER
  CHANNEL** (review-11 P1), each channel's CO₂ over the activity that drives it: electricity =
  `CO2|Supply|Electricity / Secondary Energy|Electricity`, industry = `CO2|Demand|Industry /
  FE|Industry`, process = `CO2|Industrial Processes / (Production|Steel + Cement)`, transport /
  buildings over their final energy, energy_supply = `(Energy|Supply − Electricity) / FE`, `__all__`
  = `(Energy + Industrial Processes) CO2 / total FE`. This is **not** `CO2/GDP` (the earlier form
  double-counted the CGE-endogenous output-share change, review-10 P1) and **not** one economy-wide
  bundle. Selected by an **explicit validated tuple** (`_NGFS`, exact region + resolve-to-exactly-one
  scenario/model); every configured CO₂ and activity variable is mandatory (missing → raise); unit
  compatibility is checked within each sum **and between a co2 minuend and its co2_subtract
  subtrahend** (review-13 P3#5). A knot is emitted at **every source year** (each the CAGR to the
  next), so the path reproduces the source intensity at every source year (review-9 P1 — the earlier
  sparse 2025/2040 pair annualised the 2040→2100 tail and overstated 2050 intensity ~30%); endpoints
  validated finite/positive. The per-channel paths are **blended per model sector at the LEVEL**
  (Σ wₖ·Fₖ(t), baseline source shares — review-13 P1#2) by `sector_emissions_channel` in the
  concordance. KNOWN GAP: the AEEI `FE/output` term is not sourceable per sector from NGFS → these
  are combustion/energy-emissions intensities, not full CO₂/output (**LOW confidence**); the path is
  CO₂-only (a CO₂-only CarbonPrice is required; gas-indexed paths are a future refinement).

All rates are **proportional** annual growth (log/delta-log sources converted with `expm1`, review-8
P3). Each maps source countries/industries to the N/S and BRD/MIL archetypes via committed
`archetype_maps/` and aggregates with the documented per-driver weights. Raw files live in
`data/structural/sources/raw/` (git-ignored, separately licensed; see its README). Parsing is covered
by byte-level fixtures in `tests/test_structural_extractors.py`. **Known scope of THIS extraction
(honest limits, tracked as 1c–1e below):** EU KLEMS is a single country (Austria); the NGFS
per-channel intensities are World-level and region-uniform (per-region/per-source refinement is a
future item) and omit the AEEI `FE/output` term; the productivity/participation forward knots hold a
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

### 1d. Pinned emissions-intensity scenario — per-channel, blended per sector
**SCENARIO PINNED (review-8); PATH FIDELITY (review-9); CO₂/activity (review-10); PER-CHANNEL + BLEND
(review-11/12).** The extraction pins an explicit validated tuple (`_NGFS`): NGFS Phase 5, model
`REMIND-MAgPIE 3.3-4.8`, scenario `Below 2°C`, region `World`. A knot is emitted at every source year
(each the CAGR to the next); endpoints validated finite/positive; the electricity channel's
net-negative ~2090 tail is truncated at its last positive year, and a channel non-positive within the
client horizon (before 2050) raises (the Net-Zero removals case).

**Per-channel intensity.** `scripts/fetch_ngfs_sectoral.py` (pyam, offline) vendors, per channel, a
CO₂ numerator + the activity that drives it; `extract_ngfs_emissions(intensity_specs=…)` emits
Σ(CO₂)/Σ(activity) per channel: electricity = CO₂|Supply|Electricity / Secondary Energy|Electricity,
industry = CO₂|Demand|Industry / FE|Industry, process = CO₂|Industrial Processes / (Steel + Cement),
energy_supply = (Energy|Supply − Electricity) / FE, transport/buildings over their FE, `__all__` =
(Energy + Industrial Processes) CO₂ / total FE. Every configured variable is mandatory (missing →
raise).

**Per-sector channel BLEND (review-12 P1#2).** The concordance's `sector_emissions_channel` maps each
model sector to a WEIGHTED BLEND of channels (not one), because emissions are mixed-source:
electricity → {electricity}, metals → {process 0.6, industry 0.4}, minerals → {process 0.7, industry
0.3}, chemicals → {industry 0.7, process 0.3}, coal/oil_gas → {energy_supply}, agriculture →
{`__all__`} (its energy CO₂; AFOLU excluded), services/buildings → {buildings}. The loader blends the
channels **at the cumulative-LEVEL** (review-13 P1#2): a sector's intensity factor is `Fₛ(t) = Σ wₖ
Fₖ(t)` with `Fₖ(t) = ∏_{τ<t}(1+rₖ,τ)` — the weights are baseline emission-source shares, so blending
LEVELS is correct; the earlier per-year average of channel *rates* (`∏_t(1+Σ wₖ rₖ,t)`) silently
rebalanced the source shares every year (~5% level error on the shipped metals/chemicals paths). The
emitted per-interval rate reproduces the blended level under the engine's annual compounding. The
weights must be a non-empty **convex combination** (numeric, finite, non-negative, sum to 1 —
validated, review-13 P2) and cover every sector (no silent `__all__` fallthrough, review-12 P2#4).
They are documented fixed values; a **per-region satellite-weighted blend** (reading the build's
combustion/process/supply satellite split) is a sign-posted future refinement.

**KNOWN GAPS (documented, not fabricated):**
- **1d-AEEI:** the FE/output autonomous-energy-efficiency term is NOT sourceable per sector from NGFS
  (no sector output/VA) and the Leontief CGE does not reproduce it — the drivers are combustion/
  energy-emissions intensities, NOT the full CO₂/output the engine multiplier nominally represents;
  confidence LOW. Closing it needs a sector-output source + an engine AEEI channel.
- **1d-gas:** the paths are CO₂-only (NGFS CO₂ series). A non-CO₂/CO₂e `CarbonPrice` combined with the
  structural emissions trajectory is REJECTED (review-12 P1#1) rather than silently scaled with the
  CO₂ path. Gas-indexed structural paths (NGFS does publish CH₄/N₂O per sector) applied pre-aggregation
  per gas are the full fix — sign-posted; needs the engine to apply gas×component scales before
  aggregating the carbon cost share.
- **region uniformity:** the source path is World, duplicated across model regions; per-region paths
  are a follow-up.

**Acceptance:** manifest names the full scenario tuple; a reader can re-pull each channel's series
(the fetch script); per-channel paths are emitted, validated, and routed by the concordance. AEEI
term: outstanding (data + engine).

### 1e. Sensitivity cases (deterministic) — quantified uncertainty still open
**SENSITIVITY CASES DONE (review-9 1e; relabelled review-10 P1#2; per-driver direction review-11
P2#2). Probabilistic quantified uncertainty: OPEN.** `scripts/build_uncertainty_sets.py` emits
`trajectories_v1_adverse.json` / `_favourable.json` as **deterministic stress cases** (NOT statistical
intervals — NGFS scenarios carry no probabilities):
- **EU KLEMS sector drivers**: an EMPIRICAL envelope = the per-archetype min/max across the trend
  windows {2017–21 (central, COVID), 2015–19, 2010–19 (pre-COVID)}.
- **Other drivers**: a confidence-tiered relative stress (low ±50% / medium ±25% / high ±10%), with
  direction PER DRIVER — adverse = lower growth for productivity/population/participation, slower
  (less-negative) decarbonisation for emissions intensity.
`structural_trajectory_variants(regions, sectors)` returns {adverse, central, favourable};
`structural_sensitivity_bounds()` reports the REALISED per-output min/max (the input labels do not
bound a nonlinear output). A `--check` gate keeps them reproducible.

**Acceptance:** a client result can report a deterministic sensitivity range by running the three
cases; the cases are provenance-stamped as non-probabilistic. **Still open:** genuine quantified
(probability-weighted) uncertainty — WPP variants, multiple NGFS models/scenarios, empirically
grounded driver ranges — remains a follow-up, so "uncertainty-quantified" in §0 is only PARTIALLY met.

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
