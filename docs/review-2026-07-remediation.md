# Independent review (2026-07) — remediation

An independent model review found real, reproducible defects in Engine 1 and the data layer,
and correctly judged the "implemented & validated" / "Phase 1–3 complete" claims premature.
This records what was fixed, what remains, and the honest status.

## Critical (fixed)

| Finding | Fix |
|---|---|
| **Carbon-price units wrong by ~1e9.** EXIOBASE F is kg/M€; code did τ·e with no conversion. | Adapter reads the extension's `.unit` metadata and normalises every gas flow to **tonnes**; engine applies **1e‑6 (M€→€)** so τ·e is a dimensionless cost share. `units_plausible_magnitude` + `known_answer_full_pipeline` checks added. |
| **Live build path broken.** `download_exiobase3` returns a log, not a path; real extension is `satellite`, not matched by `emiss\|ghg`. | `fetch_exiobase` now locates and returns the year's archive; extension matcher includes `satellite`. (Live download still untested offline by design — see remaining.) |
| **Gas / time-path semantics ignored.** Engine always used CO2e; copied one year to all; `Shock.at()` didn't exist. | Engine honours `shock.gases` (GWP-weighted sum of selected per-gas rows) and recomputes **per year** via a real `CarbonPrice.price_at(year)` with piecewise-linear path interpolation. `gas_selection_distinct` + `time_path_varies_by_year` checks added. `revenue_recycling` now **rejected** (not silently ignored) — it has no meaning in a price model. |

## High (fixed)

- **Admissible-A policy.** Negative coefficients broke the non-negative pass-through
  guarantee (a positive tax could lower a price). The engine now **rejects** A with
  entries below −1e‑9 in addition to the ρ(A)<1 check. Test + counterexample added.
- **Full/small provenance.** `Provenance` now carries `build_id` and `aggregation`; the
  store preserves them; the manifest's `data_source` is the build id. The confirmed
  counterexample (both builds reporting `EXIOBASE-test test`) is resolved.
- **NaN final demand.** The structural gate now rejects non-finite final demand.
- **Weak `ResultSet.validate_schema`.** Now also rejects unexpected columns, NaN/inf
  values, and invalid scenario-band labels.
- **Non-atomic store writes.** `save` writes to a staging dir and atomically replaces the
  build dir, eliminating partial writes and stale satellite/value-added files.
- **Silent zero-fill of missing satellite labels.** Now an alignment error (raises).
- **Exports omitted provenance.** Results export adds a manifest JSON download and embeds
  the manifest in the Parquet file metadata.
- **Sparse full-MRIO overclaim.** Docs corrected: the engine is **dense**, suitable for the
  small build only; the sparse path is intended but **not implemented**. Restricted, not
  claimed.

## Remaining (tracked, not yet done)

- **Live EXIOBASE known-answer test** (published CO₂ multipliers within tolerance) — needs
  the multi-GB download; the one validation still missing before real-build numbers are
  quoted quantitatively. A small real-format archive fixture would let the adapter's archive
  selection and `satellite` extension parsing be tested offline.
- **Sparse full-MRIO path** — implement or keep the engine restricted to small builds.
- **Content-hashed data build ids** — currently deterministic/human-readable.
- **Scope option** (scope-1 vs embodied-energy) — spec mentions it; deferred with a note
  that adding purchased-energy emissions to the buyer would double-count emitter liability
  unless modelled as a distinct scope-2 liability.
- **Curated small-build concordance** — the default sector map is still a first-word
  placeholder; the concordance and its hash are not yet in result provenance.
- **GUI data explorer** loads the whole parquet before slicing (not DuckDB-paginated);
  fine for the small build, not the full MRIO.
- **Non-blocking build page** — the subprocess isolates the work but the page still reads
  its output synchronously; a job queue is Phase 7.
- **EXIOBASE 3.9.x** — pinned at 3.8.2; a newer release exists.

## Honest status after remediation

- **Toy economy:** correct and validated to a hand-derived known answer, with units, gas
  selection, time paths, and admissibility all covered.
- **Real EXIOBASE small build:** the pipeline is now dimensionally correct and should give
  plausible fractional price changes, but **is not yet validated against published EXIOBASE
  numbers** — treat real-build results as indicative pending that test.
- **Full MRIO:** not supported (dense-only); use the small build.

Phase status docs and the model doc have been updated to reflect this; "validated" now means
"validated on the toy + internal identities", explicitly not "validated against live data".

---

## Second review round (fixed)

A second independent review found further real defects — including one the first round
*introduced*:

- **Multi-gas cross-multiplication (new bug, high).** The round-1 fix unioned all gases and
  summed all prices, then multiplied — cross-multiplying one shock's price against another's
  gas. Rewritten: each shock contributes `price(year) × its own gases' intensity × its own
  coverage`, summed. Counterexample (CO2@€100 + CH4@€10) now gives 0.0128, not 0.0528.
- **Store overwrite still not atomic (high).** The old dir was deleted before the rename, so
  a rename failure lost the build. Rewritten to move the old build aside, swap staging in,
  then drop the backup — with restore-on-failure; per-pid staging avoids writer races. Test
  simulates a swap failure and asserts the prior build survives.
- **Live archive selection (high).** Removed the fallback to "any archive for the year"; a
  `pxp` request now refuses to return an `ixi` file.
- **Path bypassed the non-negative-price check (high).** `CarbonPrice.path` values are now
  validated ≥ 0.
- **Unknown/partial gas silently aggregated (high).** `_gas_intensity` now raises for an
  unknown gas or a partially-missing mix, instead of falling back to CO2e (which taxed all
  gases).
- **Small-build-only not enforced (high).** The engine now rejects builds above
  `MAX_DENSE_PRODUCTS` and validates `io.unit == MEUR` and `t/MEUR` intensities before the
  1e-6 scaling. The engine header no longer claims a sparse path exists.
- **Doc/theory fixes.** "Upper bound" softened to "expected to over-state vs substitution,
  not a proven bound"; the non-negative-inverse statement now states the required A ≥ 0;
  tier aggregates explicitly distinguished from structural path analysis; the scope option
  is documented as **not implemented** (with the double-counting rationale); "verbatim
  assumptions" corrected to "paraphrase kept consistent"; fixed test-count claims.
- **Contract hardening.** `ResultSet.validate_schema` rejects string values, duplicates and
  bad bands; `RunManifest` rejects empty assumptions; `Classification` rejects duplicate
  labels; `ConcordanceMap` rejects negative weights; `NatureStress.severity` bounded [0,1].
- **Packaging.** The `[gui]` extra now includes duckdb/pyarrow/pymrio (the GUI imports the
  data store and Parquet exporter). The GUI results page labels changes as **percent** and
  states they are fractional cost-only.

---

## Third review round (fixed)

A third review confirmed the round-2 fixes and found four more real issues:

- **Unit guard permitted a 1000× error (high).** The check only tested a `/MEUR` *suffix*, so
  `kg/MEUR` passed and was treated as `t/MEUR`. Now the engine requires **exact** units per
  row (`t/MEUR` for gases, `tCO2e/MEUR` for CO2e), rejects missing units, and requires
  `currency == EUR` and `unit == MEUR`. Engine version bumped **0.2.0 → 0.3.0**.
- **Size cap ran after the dense eigenvalue computation (high).** `_assert_productive` (which
  calls `eigvals`) ran before the product cap, so a full MRIO still risked OOM. The cap now
  runs **first**, before `to_numpy`/`eigvals`/solve. Test asserts `eigvals` is not reached.
- **Store "truly atomic" claim was wrong (high).** A directory rename can't atomically replace
  a non-empty dir, so there is a brief window where the canonical path is absent. Reworded to
  **recoverable, not strictly atomic**, and added `_recover_interrupted` (runs on store open)
  that restores a build left in `.bak` by a crash mid-swap. Tested.
- **Engine version not bumped (high).** Fixed (now 0.3.0), so manifests are distinguishable.

Medium items also fixed: `gases` must be non-empty/unique and cannot mix `CO2e` with
component gases (validated at construction *and* in the engine); `path` rejects NaN;
`ConcordanceMap` rejects NaN weights; `RunManifest` rejects empty assumptions at construction
(not just via `build`); `ResultSet` rejects a string `year`; the adapter verifies A is square
and reindexes final demand onto A's products (rejects mis-ordered/misaligned sources); phase
status text de-staled ("upper bound"/"CO2e by default" removed); the README quickstart installs
`.[dev,data,gui]` (the tests need them).

Still deferred (unchanged): live EXIOBASE known-answer test, sparse full-MRIO, content-hashed
build ids, curated concordance, DuckDB-paginated explorer, async builds, EXIOBASE 3.9.x, and
the strictly-atomic (single-syscall) store swap.

---

## Live EXIOBASE known-answer — DONE (the gate every review flagged)

Downloaded EXIOBASE 3 (2019, pxp, ~690 MB) and ran the pipeline on real data
(`tests/test_exiobase_known_answer.py`, opt-in via `CGE_EXIOBASE_ARCHIVE`; 4 tests pass):

- **Adapter reproduces the full MRIO** — 9800×9800, 49 regions × 200 products, and detects
  the real `satellite` extension (the round-1 fix, now confirmed on real data).
- **Global CO₂ preserved** — adapter intensity×output equals pymrio's raw satellite total to
  `rtol 1e-6`, proving the kg→tonne conversion and `e=F/x` construction on real data.
- **Magnitude correct** — 30.0 Gt, the expected EXIOBASE 2019 production-accounting figure.
- **Engine 1 runs end-to-end on real data** — a €100/t run on a coarse EUR build (14 sectors ×
  16 regions) gives fractional price changes (mean +15.7%, coal +80–164% across regions), coal
  the most exposed sector. Qualitatively and quantitatively sensible.

Also fixed while doing this: the default `default_maps` first-word grouping produced ~1005
sectors (exceeding the engine's dense cap) — replaced with a functional ~14-sector keyword
grouping + continental region folding, so a real build is actually runnable. Engine 1's
status is now "validated on toy + live EXIOBASE"; real-build numbers from a *coarse* build are
usable and directionally sound (a curated concordance would sharpen sector-level precision).

---

## Sixth review round (fixed)

A sixth review (post live-validation) found two highs plus mediums:

- **Negative gas could still cancel inside a single multi-gas shock (high).** The round-5 fix
  checked the *aggregated* per-shock vector, so a negative CO₂ inside a single `["CO2","CH4"]`
  shock was hidden by a positive CH₄. Now **each gas row is checked before GWP aggregation**,
  and the check honours coverage (an uncovered negative row can be excluded via coverage, as
  the error suggests). Reproduced and now rejected.
- **Catalogue not cross-process/transactional-safe (high).** Per-build locks didn't serialise
  the shared DuckDB catalogue; concurrent saves collided on DuckDB's file lock and lost rows,
  and there was no rollback if the catalogue update failed after the filesystem swap. Now:
  **all** catalogue access (init/upsert/read) goes through a global catalogue lock + retry;
  the upsert is a single DELETE+INSERT transaction; and the filesystem backup is retained
  until the catalogue commit succeeds, rolling files back to the prior build on failure. A
  4-process concurrent-write test confirms all rows land.

Mediums fixed: recovery now runs on **direct** `load`/`load_meta`/`load_quality` (GUI
`frames()` calls `load()` directly); the live-test fixture parses the year from the archive
filename; the live checks are also registered as a **gated `exiobase_live` validation suite**
(present in `cge validate` only when a real archive is set), matching the docs; the coarse map
keywords now catch anthracite/coke/diesel/refinery/biogas (were in "other") and fold more
economies to continents; `_region_row_labels` recognises aggregated `RoW_*` labels. Docs:
column-sum condition corrected to *sufficient not necessary*; the Stadler-2018 citation
qualified (it covers through 2011, not 2019); adapter docstring de-MEUR-hardcoded; the live
gate honestly re-described as strong integration/sanity, **not** a published-multiplier
benchmark (still outstanding). Engine version left at v0.4.0 — the gas fix tightens rejection
of already-invalid input, not the numeric behaviour of valid runs.

---

## Fourth review round (fixed)

A fourth review confirmed round-3 and found two material residuals (both introduced by my
round-3 fixes) plus several mediums:

- **Store recovery deleted a live writer's staging (high).** `_recover_interrupted`
  unconditionally removed every `.tmp`, so opening a second store deleted a concurrent
  writer's staging; PID-based staging names also collided within a process. Replaced with a
  **per-build writer lock** (pid lockfile with liveness check): saves use a uuid staging dir
  and a lock; recovery only cleans up builds whose lock is *stale* (dead pid), leaving live
  writers untouched, and a concurrent save on a held build is refused. Two tests.
- **Gas without a GWP factor got GWP=1 (high).** `GWP100_AR5.get(g, 1.0)` treated a tonne of
  SF6 (GWP ~23500) as a tonne of CO2e. Now any component gas lacking an explicit GWP factor
  is rejected.

Mediums fixed: `CONTRACTS_VERSION` bumped 0.1.0 → **0.2.0** (the tightened validation is
breaking); negative selected intensities rejected; coverage labels validated against the
build's classification (a typo no longer yields a silent zero-impact run); `CarbonPrice.price`
must be finite (rejects `inf`); `CO2e` mixed with component gases rejected **at construction**
(not only in the engine); the manifest records **per-year** shock contributions (not just the
last year); the adapter takes `currency`/`monetary_unit` as parameters (the USD test fixture
is now labelled USD/MUSD, so io_price correctly refuses it instead of trusting a false EUR
label). Docs de-staled: engine header "verbatim" → "paraphrase", model doc/phase status →
v0.3.0, example YAML "upper bound" → "over-states vs substitution", units note clarifies F is
a mass total.

---

## Fifth review round (fixed)

A fifth review confirmed round-4 and found three highs (all from round-4 changes) plus
mediums:

- **Satellite unit still hardcoded per-MEUR (high).** After the adapter took `monetary_unit`,
  `_ghg_satellite` still labelled intensities `t/MEUR`, so an MUSD build had MUSD IO metadata
  but per-MEUR satellite units. The denominator now derives from `monetary_unit` (MUSD build →
  `t/MUSD`).
- **Staging failure leaked the writer lock (high).** `staging.mkdir()` ran before the
  try/finally, so a mkdir failure kept the lock forever and blocked all future saves. Moved
  inside the try so the finally always releases the lock. Tested.
- **Negative intensities cancelled against positive gases (high).** The negative check ran on
  the summed cost, so a negative CO₂ intensity plus a positive CH₄ contribution passed. The
  check now runs **per shock, before aggregation**, on each gas's masked intensity. Tested with
  the CO₂=−100 / CH₄=10 counterexample.

Mediums fixed: recovery now runs on enumerate/read (`build_ids`/`has`) too, so a long-lived
process recovers a crashed subprocess's build without recreating the store; `build_ids`
excludes internal `.tmp`/`.bak` dirs; the catalogue update moved **inside the lock** so two
writers can't cross metadata and files; the example YAML / phase status no longer recommend
the USD `exiobase-test-small` build (io_price refuses it), and a **EUR-relabelled build test
restores the store→engine seam** the round-4 change had dropped; engine bumped to **v0.4.0**
(behaviour changed) with docs synced; the contracts semver policy is corrected to standard 0.x
semantics (a breaking change bumps the minor while major is 0). The PID-lockfile protocol has
small theoretical races (create/write, stale reclaim) — noted; an OS-backed file lock is a
possible future hardening.

---

## Seventh review round (fixed)

One high plus mediums:

- **Hard-kill between swap and catalogue commit could keep uncommitted files and delete the
  valid backup (high).** With `final` installed but the catalogue not yet committed, both
  `final` (uncommitted) and `.bak` (valid) existed; recovery kept `final` and deleted `.bak`,
  leaving files at v2 but catalogue at v1. Fixed with a **commit marker** (`.uncommitted`
  written into the build dir, removed only after the catalogue commits): recovery sees the
  marker and restores the backup, discarding the uncommitted files. Tested with the exact
  crash-window sequence.

Mediums fixed:
- **Global lock race** — replaced the PID-file lock (create-then-write window) with an
  **OS-backed `fcntl.flock`** (atomic; auto-released on death).
- **Aggregation changed build identity silently** — `build_from_pymrio` takes a
  `concordance_id`; a named concordance encodes its **version + content hash** in the small
  build id and `aggregation` field (which flow into provenance and the run manifest). Default
  coarse map bumped to `coarse-v2`.
- **Keyword false positives** — added high-priority exceptions so oil-seeds→agriculture,
  nuclear-fuel→electricity, motor-fuel-retail→trade, biogasification-waste→waste; removed the
  over-broad `fuel`/`oil ` keywords. (Still a keyword heuristic, documented as such.)
- **Live provenance** — the `exiobase_live` suite and standalone fixture parse the archive
  year (no hardcoded 2019); the live **end-to-end engine** check is now in `cge validate`.

Docs de-staled: phase-1/phase-2 status and data-layer.md no longer say live checks are
pending; "coal most exposed" → "coal among the most exposed"; roadmap "upper bound" softened
and "structural paths" corrected to Neumann-tier aggregates.

Remaining genuinely open (documented, not defects): an **independent published-footprint**
comparison; a **curated sector concordance**; sparse full-MRIO; content-hashed full-build ids;
EXIOBASE 3.9.x.

## Follow-up review round (Engine 2 rewrite + store) — fixed

A later review of the Engine-2 production-volume rewrite and the store's generation-based
recovery found two blockers and several mediums; all fixed:

- **P1 — legacy crash recovery could keep uncommitted data and delete the committed backup.**
  A store left by the *predecessor marker-based* implementation, hard-killed between the swap and
  the catalogue commit, has a legacy `.uncommitted` marker inside `final` and NULL generations
  everywhere; the generation-only recovery treated NULL==NULL as committed and discarded the
  valid backup. Recovery now decides commit state **fail-safe** (`_final_is_committed`): it
  honours the legacy `.uncommitted` marker, treats corrupt/unreadable final metadata as
  uncommitted (never as a legacy NULL match), and only keeps `final` when commit is positively
  proven. Regression tests reproduce the exact predecessor crash and the corrupt-metadata case.
- **P1 — run manifests identified only the IO system, not the satellite that determines prices.**
  A different satellite (generation or doubled emissions) moved prices but left the manifest
  identical. Both engines now record an `inputs` list — an identity (build id + generation) and a
  content hash for **every** substantive input (IO system, satellite, and, for partial_eq, the
  elasticity set). Tests assert a changed satellite moves the manifest.
- **P2 — fallback elasticity band was invisible in the manifest.** The hash covered only explicit
  values and per-good records held only the central value. The manifest now records each good's
  full applied (low, central, high) triple and an `effective_content_hash` over the *applied*
  triples, so a changed fallback band moves the manifest. Test added.
- **P2 — elasticity classification was unenforced.** An unrelated `classification` was accepted
  and matched by coincidental sector names. partial_eq now rejects an elasticity set whose
  classification is not compatible (coarse-sector family or exact match) with the build's sector
  classification. Test added.
- **P2 — version identifiers advanced.** `CONTRACTS_VERSION` 0.2.0 → **0.2.1** (additive:
  `generation` + manifest input identities); `io_price` 0.4.0 → **0.5.0**; `partial_eq` 0.2.0 →
  **0.3.0** (classification enforcement + manifest semantics).
- **P3 — docs de-staled.** `partial-equilibrium.md` (equation-2 mislabel, "first-order"/"optional
  Armington" wording, v0.3.0), `phase-4-status.md` (test/check counts), `roadmap.md` (old linear
  ε·Δp / fixed-point wording) corrected. Elasticity `source` strings documented honestly as
  descriptive attributions, not per-value citations (curated citations remain a follow-up).

---

# Independent review (2026-07, open-economy round) — remediation

A second, deeper review of the **open-economy CGE** (Armington/CET + CES + sweeps) found nine real,
reproduced defects — all confirmed against HEAD and now fixed. The closed pilot was judged sound;
the open variant's *supported domain* was narrower than documented and several closed-path
safeguards were bypassed on the new open path. Engine `cge_static` bumped **0.3.0 → 0.4.0**.

## P1 (fixed)

- **Open dispatch bypassed the SAM + shock-control gates.** `_run_open` returned before any
  validation ran, so an unbalanced open SAM and unknown gases/coverage were accepted. Added
  `_validate_open_sam` (structure, unique/aligned axes, finite, non-negative, balanced) **before**
  calibration, and explicit rejection of gas selection / spatial coverage the single dimensionless
  cost share cannot express. Tests added.
- **Foreign-savings closure did not replicate a non-zero current account.** Household income omits
  the ROW transfer, so a SAM with `Sf ≠ 0` would not replicate. **Restricted** the pilot to a
  balanced current account: `calibrate_open` rejects `Σ M ≠ Σ E` with guidance; a ROW-transfer
  closure is a documented follow-up. Added `open_nonzero_foreign_savings_rejected` to the standing
  validation suite (as the review requested) + a pytest.
- **Zero-export sector crashed calibration** (`0 ** (-Ω) = inf`). **Fixed** with masked branching
  (safe base in the CET/Armington price duals and quantity splits), so structural zeros — common in
  real trade data — run warning-free. Test with a non-exporting sector added.
- **Open manifest could not identify the inputs that drive results.** It recorded only the SAM and
  the *first* elasticity. Now records the **effective per-year carbon-cost vector (hashed)**, the
  **full per-sector** Armington/CET/VA elasticity vectors, solver backend/status, the
  recycling-defaulted flag, foreign savings, and accurate open/CES assumptions (`OPEN_ASSUMPTIONS`).
  The **closed** manifest now also records `va_elast`. Version bumped for the model additions.

## P2 (fixed)

- **Recycling fixed point could stop above tolerance.** The 50-iteration undamped loop had no
  convergence check. **Replaced with the closed-form solution** `I = factor_income/(1−k)` (income is
  linear at fixed prices), with a `k ≥ 1` guard and an independent income-identity check. The budget
  identity now holds exactly.
- **Elasticity + sweep-band inputs under-validated.** Added `_elast_vector`: finite, strictly
  positive, exact scalar-or-`(ns,)` shape (a length-1 vector no longer silently broadcasts; a
  one-element VA no longer raises a raw IndexError), applied to VA/Armington/CET in **both** the open
  and closed calibrations. The sweep now requires strictly ordered `low < central < high`.
- **Open system was overdetermined with tautological rows** (9 residuals vs 7 unknowns for 2
  sectors; the composite-market rows were identically zero because `_quantities` already solves those
  markets). **Removed** the redundant rows → a genuinely square `2·ns + nf + 1` system; corrected the
  module + model-doc equation count. (This also removes the latent risk that the IPOPT adapter, which
  builds exactly *n* constraints, would silently ignore excess equations.)
- **Open `gdp_change_real` changed the metric's meaning.** It reported unweighted `Σ FD` while the
  closed model reports CPI-weighted expenditure `p·FD`. **Aligned** the open path to `pq·FD` — the
  same real-GDP contract — so it equals the CD welfare move at the CPI numéraire.

## P3 (fixed)

- **Docs de-staled.** `cge-static.md` (open economy no longer "pending"; CES value-added equation
  (1′) added; the square reduced system + balanced-current-account restriction documented; v0.4.0),
  `README.md` (open economy is in, not pending), `roadmap.md` §5.1 (hand-built open SAM done),
  `user-guide.md` (engine version), and the engine's `ASSUMPTIONS`/docstring (the open run no longer
  claims Armington is a later phase) all reconciled.

**Verification:** 229 pytest passed / 5 skipped (18 new regression tests); `cge validate` 44/44;
ruff check + format clean. **Remaining (deferred, larger effort):** true multi-region with bilateral
trade, a non-zero-foreign-savings ROW closure, and a live-EXIOBASE open-SAM build.

---

# Independent review (2026-07, open-economy round 2) — remediation

A third review (of HEAD `5184a91`) confirmed the round-1 open-economy fixes held and found seven
further defects, all reproduced. Two P1s concerned *arbitrary supplied SAMs* (not the toy), which is
why the reviewer would not yet call such runs safe. All fixed; still engine `cge_static` 0.4.0
(bug/robustness fixes, no new model surface).

## P1 (fixed)

- **A balanced SAM could pass validation without belonging to the implemented model.** The
  validators check axes/balance/finiteness/names but not the economic topology, so a balanced SAM
  with an unsupported flow (an offsetting household↔commodity loop) was accepted and reported a
  zero-shock result of exactly zero by comparing a non-replicating base with itself. Added a
  **universal post-calibration replication gate** (open + closed): derive the state at benchmark
  prices and assert every calibrated quantity ($X/FD/F$ closed; $Z/D/E/M/Q/FD/F$ open) matches to
  $10^{-6}$, else refuse the run.
- **SAM fingerprints ignored axis labels.** The fingerprint stored `accounts` + the flattened array
  but not the matrix's row/column order, so relabelling both axes (a different economy, since
  calibration reads by label) gave an identical hash and manifest. Now **canonicalised by account
  label** with the ordered labels included.

## P2 (fixed)

- **The recycling guard could abort the solver at its initial point.** The closed-form income
  correction raised on `k ≥ 1` at *any* trial price vector, but the engine starts shocked solves at
  benchmark prices where `k` can momentarily exceed 1 even though a valid equilibrium (`k < 1`)
  exists — the review reproduced this at cc=[1.4, 0.35] (benchmark k=1.006, equilibrium k=0.801).
  Now the guard raises only in **strict mode (the accepted equilibrium)**; during solver iteration a
  **smooth positive floor** on the denominator `(1−k)` keeps income finite and the residual
  C¹-continuous, and the solver uses a small **deterministic multi-start** so a benchmark start on
  the k-ridge still finds the equilibrium. A genuinely infeasible equilibrium is still refused.
- **Manifest misdescribed CES + overclaimed the double dividend + a false default flag.** The nest
  description is now derived from the calibrated `va_elast` (CES vs Cobb-Douglas), the
  double-dividend claim is removed (the model has no distortionary labour-tax wedge;
  `labour_tax_cut` ≡ `lump_sum` with one household), and `recycling_defaulted_from_none` is true only
  when a positive-revenue scenario actually defaulted from `none`.
- **Sweep provenance was incomplete.** `armington_sensitivity_sweep` now returns a
  `SweepResult` carrying the tidy `bands` DataFrame **plus** the exact elasticities, swept parameter,
  engine version, scenario hash, SAM identity, and each band's full manifest — so an exported sweep
  is identifiable.
- **The standing suite did not substantiate the "passes the full battery" claim for the open model.**
  Added open **homogeneity**, **Walras + trade-balance** under a carbon shock, and **income-identity**
  checks to `cge validate`, and replaced the non-zero-foreign-savings fixture with a **genuinely
  balanced** Sf≠0 SAM (adding the ROW→HOH capital transfer the old one omitted). Suite: 44 → 47.

## P3 (fixed)

- Zero-**import** sector with `arm_elast < 1` no longer emits a divide-by-zero from `arm_scale`
  (the unused `np.where` branch used a safe base); roadmap 5.3 header no longer says sweeps are
  pending; remaining CES↔double-dividend associations removed from roadmap + model doc; the
  non-zero-Sf regression fixture is now genuinely balanced (with a matching description).

**Verification:** 236 pytest passed / 5 skipped; `cge validate` 47/47; ruff check + format clean.
**Remaining (deferred, larger effort):** true multi-region with bilateral trade, a
non-zero-foreign-savings ROW closure, and a live-EXIOBASE open-SAM build.

## Eighth review round (2026-07) — multi-region redesign + IO double-price fix (fixed)

An independent validation-agent review of HEAD `bb54a1d` found two release-blocking P0s and
several P1/P2s. Both P0s were verified genuine against current code before any fix; per the
review's own ordering, the multi-region redesign was done first, then the double-price fix, then
the rest, then this documentation reconciliation last.

### P0 (fixed)

- **The multi-region model did not clear bilateral trade or the factor market dropped by Walras.**
  The earlier law-of-one-price reduction (`pd[r,s]` shared by domestic and export sales, CET
  allocating *quantities* only) let a machine-zero solver residual coexist with a ~15.2% bilateral
  trade discrepancy and a ~0.70% gap on the dropped factor market — the advertised South-production
  increase and cross-region leakage could have been an artefact of unbalanced demand and supply,
  not a genuine equilibrium. **Redesigned model_multi.py**: every trade route carries its own
  **destination-specific price** `pe[o,s,d]`; the Armington composite and CET transform are each
  CES duals over domestic + every partner's route price; import demand `M` and export supply `EX`
  are computed **separately** from their duals and reconciled by an **explicit bilateral
  market-clearing residual** `M[d,s,o]=EX[o,s,d]` for every `o≠d`, added to the square system
  (`2·nr·ns + nr·(nr−1)·ns + nf·nr` unknowns/residuals). Post-fix: bilateral discrepancy
  2.08e-17 (was 15.2%), every factor market gap 0 (was 0.70%). New standing checks
  (`test_multi_bilateral_markets_clear_under_shock`, `test_multi_all_factor_markets_clear_under_shock`,
  and `multi_region_markets_clear_under_shock` in the `cge validate` suite) assert both under a
  live carbon shock, not just at the benchmark.
- **The IO-backed open-economy path applied the carbon price twice.** `carbon_cost_vector`
  (Engine 1's per-year cost builder) already multiplies price × intensity × 1e-6; the multi-region
  and open IO paths stuffed that finished cost into `carbon_cost_share` and then let `_run_open`
  re-multiply by the price, quadrupling the response to a doubled price instead of doubling it.
  Fixed by carrying an **effective per-year cost dict** (`{year: cc[ns]}`, already price-included)
  end-to-end from `_open_effective_cc_from_io` through `_run_open`, which now consumes it verbatim
  with an explicit "do NOT re-multiply" guard at the point of use. Verified: response ratio at
  price=100 vs price=50 is 2.000 (was ~4 with the bug). The gas/coverage rejection loop that
  previously ran unconditionally in `_run_open` (correct for a supplied dimensionless share, wrong
  for the IO path, which **honours** gases/coverage/paths via `carbon_cost_vector`) is now gated on
  whether the run is IO-backed; new regression tests pin price linearity, a price path that starts
  at zero and later goes positive, gas selection (`gases=["CO2e"]`) matching the CO2 result on the
  fixture's identical intensity rows, two stacked shocks doubling the response, coverage excluding
  the home region replicating the benchmark, an unknown coverage label being rejected, and the
  manifest carrying both `EffectiveCarbonCost` and the `SatelliteAccount` identity.

### P1 (fixed)

- **Multi-region omitted the universal post-calibration replication gate** the closed/open models
  already had (refusing a balanced-but-uncalibratable SAM topology). Added
  `_assert_multi_replicates`, called after the benchmark solve, checking prices, `Z`, `D`, `M`,
  `EX`, `FD`, and `F` against the calibrated benchmark.
- **Multi-region manifest recorded only the SAM identity** (two runs with carbon shares 0.1 vs 0.3
  produced identical manifests) **and inherited a false "law of one price" assumption string from
  the closed/open models.** The manifest now includes a hashed `EffectiveCarbonCostMatrix` input,
  restores the `SatelliteAccount` identity on the IO-backed path, and `model_variant`/`trade`/
  `closure` describe the actual destination-specific-price, explicit-bilateral-clearing,
  no-exchange-rate/no-external-ROW/one-dropped-factor-market closure.
- **A non-zero current account only worked for import deficits.** `build_open_sam` wrote the signed
  net foreign savings `Sf=ΣM−ΣE` into a single `HOH→ROW` cell, so an export-surplus region (`Sf<0`)
  produced a **negative SAM cell** and failed the non-negativity quality gate outright — every valid
  exporter economy was unusable. Fixed to write the transfer **in the direction of the flow**:
  `HOH→ROW` for a deficit, `ROW→HOH` for a surplus, both non-negative; `calibrate_open` now reads
  the **net** transfer (inflow cell minus outflow cell) rather than assuming one direction. Verified
  on the test build's other home region (`B`, an exporter): builds, passes quality, calibrates, and
  replicates.
- **The open-SAM builder could not identify home final demand**, because the EXIOBASE adapter
  collapsed pymrio's `Y` consuming-region columns into one aggregate column — the imported share of
  home final demand had to be *imputed* from the intermediate-use import ratio rather than measured.
  Extended the harmonised data contract: `IOSystem.final_demand` may now carry **one column per
  consuming region** (`fd_by_region()` exposes the split when present; every existing consumer that
  only wants totals is unaffected, since `.sum(axis=1)` still works). The adapter now retains the
  per-region split; `aggregate_io` aggregates it through the region bridge on top of the existing
  producing-label bridge; `build_open_raw_sam` uses it when present for an **exact** home-final-demand
  and import attribution, falling back to the documented imputation on legacy builds. Both paths are
  explicit in provenance and the quality report: a new `open_fd_attribution` check is PASS
  ("measured") or WARN ("imputed_import_share (synthetic: ...)"), and the SAM's provenance notes
  record which attribution was used. Verified: on the test build, the two single-region reductions
  (home=A, home=B) are now **exact mirror images** (A's exports equal B's imports and vice versa) —
  something the imputed construction could not guarantee.
- **Multi-region's `gdp_change_real` was current-price consumption mislabelled as real GDP**
  (already fixed in an earlier pass this round, retained here for the record): renamed to
  `real_consumption_change`, a base-price (Laspeyres) household-consumption index — documented as
  NOT production-side real GDP, since only region 0's CPI is pinned.

### P2 (fixed)

- Multi-region's trade elasticities (`arm_elast`/`cet_elast`) now go through the same strict
  `_elast_vector` validation as closed/open (finite, positive, correct shape) — negative or zero
  elasticities are rejected with a clear error instead of silently producing nonsense CES/CET
  scales.
- Multi-region's recycling mode is validated the same way as closed/open: an unsupported mode
  raises; a `none` request on a positive-revenue scenario is recorded as
  `recycling_defaulted_from_none` rather than silently collapsing.
- **Documentation reconciled last** (per the review's explicit ordering), only after the P0/P1 gates
  above passed: `docs/user-guide.md` no longer calls Cobb-Douglas value added "fixed proportions"
  (that's Leontief; Cobb-Douglas has unit elasticity of substitution) and its multi-region section
  now names `real_consumption_change` and describes destination-specific prices instead of the
  law of one price; `README.md`, `docs/overview.md`, and `docs/models/cge-static.md` (header, §8a,
  and the "not yet modelled" line) now describe multi-region as implemented with bilateral clearing
  rather than "pending"; `roadmap.md`'s Phase 5 status no longer says the multi-region model
  "currently law-of-one-price" and separates the still-pending items (IOSystem-driven multi-region
  SAM build, per-cell trade elasticities) from what this round completed.

**Verification:** 268 pytest passed / 5 skipped; `cge validate` 51/51; ruff check + format clean.
Engine version bumped `0.5.0` → `0.5.1` (multi-region equation redesign + IO carbon-cost fix are
behavior-changing).
**Remaining (deferred, larger effort):** an IOSystem-driven multi-region SAM build (the multi-region
model still requires a supplied SAM); per-cell (rather than uniform) trade elasticities; a
live-EXIOBASE SAM build for all three CGE variants.

## Ninth review round (2026-07) — sparse-trade rank deficiency + missing-satellite gap (fixed)

A follow-up independent review of HEAD `ea5a060` (round 8's fixes) found no new P0 — both prior P0s
held — but two P1s in the *implemented* model, plus several P2/P3s. Fixed in order: both P1s, then
the P2s, then documentation last, matching this project's standing remediation discipline.

### P1 (fixed)

- **Sparse bilateral trade made the multi-region equilibrium rank-deficient.** Every possible
  directed route `(o,s,d)` got a live price unknown `pe[o,s,d]` and a clearing residual, even when
  benchmark trade on that route was genuinely zero — but nothing in `_armington_price`/
  `_cet_price`/`_quantities` ever *reads* a zero-share route's price (they gate on `share > 0`), so
  that unknown was a free, unpinned direction. Reproduced exactly as described: on a 3-region toy
  SAM with one structurally-zero trade route, the benchmark Jacobian's rank was 26 against 30
  unknowns, and perturbing the unused route's price by 3× left every residual at machine-zero
  (2.22e-16) — the solver's "convergence" did not pin a unique equilibrium. Real SAMs are commonly
  sparse (most region pairs don't trade every good), so per the review's own framing, fixed by
  **packing only active routes** rather than rejecting sparse topology: added
  `MultiCalibratedModel.active_routes` (routes with `M0>0` or `EX0>0` at benchmark), and
  `model_multi.py`'s `_pe_from_flat`/`_unpack`/`residuals`/`n_unknowns` now pack/residual only those
  routes, fixing every inactive route's price at 1 (never solved for, never read). Post-fix on the
  same fixture: 26 unknowns (not 30), Jacobian full rank, benchmark and shocked replication both to
  machine precision. New `toy_multi_sparse_sam()` fixture (3 regions, one sector genuinely untraded
  on two routes) plus regression tests pinning full Jacobian rank and correct replication.
- **A positive carbon price on the IO-backed open path with no satellite was silently accepted as
  a zero-impact run.** `_open_effective_cc_from_io` returned `None` whenever `sat is None`,
  identically to a genuine zero price or a coverage selection that excludes the home region — but
  a *missing* satellite with a *requested* positive price is a data error, not a legitimate zero,
  and the closed IO path already raises on exactly this case. Reproduced: a €100/t run with no
  `SatelliteAccount` supplied ran to completion with `emissions_priced=False` and zero impact, no
  error. Fixed by adding the same gate `_run_open_from_io` — a positive price with `sat is None`
  now raises before calibration, naming the missing input, matching the closed path's message.
  Distinct legitimate-zero cases (zero price with no satellite; a satellite present but coverage
  excludes home) are unaffected and still produce a genuine, unflagged zero. Also fixed a related
  gap the review's rationale implied: the `SatelliteAccount` identity was previously recorded in
  the manifest only when the effective cost came out non-zero — it's now recorded whenever a
  satellite was actually consulted, even if its effective cost is legitimately zero, so the
  manifest reflects that real data was read rather than that pricing happened to be non-zero.

### P2 (fixed)

- **`fd_by_region()` inferred the final-demand column shape from `set(columns) == set(regions)`,**
  which silently misclassified an *incomplete* by-region frame (e.g. one region's column dropped by
  an upstream bug) as a legacy single-aggregate-column build — the set-equality check just failed
  and fell through to `None`, with no signal the data was actually corrupted. Fixed by adding an
  **explicit discriminator field**, `IOSystem.final_demand_kind: Literal["aggregate", "by_region"]`,
  set once by whoever constructs the object (the EXIOBASE adapter, `aggregate_io`, or the store on
  load — threaded through a new `BuildMeta.final_demand_kind` sidecar field for round-tripping) and
  validated for completeness/uniqueness against `regions` whenever marked `by_region` — a dropped or
  duplicated region column now raises at construction instead of silently degrading. `fd_by_region()`
  reads the discriminator directly; it no longer re-derives the shape from the column set.
- **The multi-region numéraire rationale was theoretically wrong.** A comment/docstring claimed
  `pq·FD` is unsuitable as a real-consumption measure because "only region 0's CPI is pinned" and
  other regions' `pq` is in an "unpinned scale" — false in a connected system: one global numéraire
  fixes the common nominal scale for *every* region, and verified numerically that every region's
  `pq` is a single, fully-determined number at the solved equilibrium (perturbing any region's
  solved `pq` by 1% breaks a residual). The actual reason `pq·FD` is unsuitable: it is
  **current-price nominal expenditure**, conflating the quantity change with the composite-price
  change; valuing `FD` at **base** (benchmark) prices isolates the real quantity effect. The emitted
  `real_consumption_change` index itself was already correct — only the stated rationale was wrong.
  Corrected in `engine.py`, `docs/models/cge-static.md`, and `docs/user-guide.md`; added a
  regression test proving every region's price is genuinely pinned (not just region 0's).
- **Zero-impact runs still silently overwrote the requested recycling mode.** On the open and
  multi-region paths, `if recycling == "none": recycling = "lump_sum"` ran unconditionally whenever
  `none` was requested, regardless of whether anything was actually priced — so a zero-impact run
  explicitly requesting `revenue_recycling="none"` had its manifest report
  `recycling_mode: "lump_sum"` next to `recycling_defaulted_from_none: False`, which is internally
  contradictory (if it wasn't defaulted, why does it show a different mode than requested?). The
  closed-economy path already had the correct guard (`if recycling == "none" and emissions_priced`);
  the open/multi paths now match it — the switch is gated on `recycling_defaulted`, the same flag
  recorded in the manifest, so the two can never disagree again. Numerically inert either way
  (`derive_open_state`/`derive_multi_state` already skip the recycling fixed-point when `cc == 0`),
  but the manifest is now internally consistent.
- **The GUI Results page exposed none of `import_change`/`export_change`/`factor_price_change`/
  `exchange_rate_change`/`welfare_change`/`carbon_revenue`, and `has_macro()` ignored
  `real_consumption_change`** entirely (so the multi-region macro section never rendered at all) —
  these were only reachable via download, contradicting the user guide's instruction to find them
  on Results. Added `has_trade`/`trade_table`/`exchange_rate_table`,
  `has_factor_prices`/`factor_price_table`, and `has_welfare`/`welfare_table` to `results_view.py`;
  wired three new sections (Trade, Factor prices, Welfare & carbon revenue) into the Results page;
  fixed `has_macro`/`macro_gdp_table` to recognise `real_consumption_change` (labelled explicitly as
  "NOT GDP"). New Streamlit-free unit tests for every new helper plus an `AppTest` smoke test
  confirming all three new sections actually render for an open-economy CGE run.

### P3 (fixed)

- Remaining documentation staleness: `engine.py`'s module docstring still said "two variants" and
  called multi-region "a documented follow-up" (it's implemented and heavily hardened this round);
  `docs/overview.md` still said "multi-region ahead" in two places and didn't list the new Phase 5d;
  `docs/user-guide.md` still claimed "the CGE emits per-region GDP for `toy_cge_multi`" (renamed to
  `real_consumption_change` two rounds ago) and didn't mention the new Trade/Factor/Welfare Results
  sections; `docs/models/cge-static.md` §8a still said "one direction per unordered pair" for the
  bilateral route unknowns (both directed routes are packed when active) and implied the
  square-system formula was still `nr·(nr−1)·ns` (now `n_active ≤ nr·(nr−1)·ns`). All corrected.

**Verification:** 289 pytest passed / 5 skipped; `cge validate` 51/51; ruff check + format clean.
Engine version bumped `0.5.1` → `0.5.2` (sparse-route packing changes the equilibrium's unknown
structure; the missing-satellite gate and recycling-default fix are behavior-changing).
**Remaining (deferred, larger effort, unchanged from round 8):** an IOSystem-driven multi-region SAM
build (the multi-region model still requires a supplied SAM); per-cell (rather than uniform) trade
elasticities; a live-EXIOBASE SAM build for all three CGE variants.
