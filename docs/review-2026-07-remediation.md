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
