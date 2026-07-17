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
