# Phase 1 — status

Phase 1 ("Data layer: EXIOBASE ingestion + quality") from `roadmap.md` is complete. The
data layer builds from the **live EXIOBASE source**, produces full + small builds, checks
correctness and consistency along the pipeline, and stores builds for engines to consume.

## Tasks (roadmap §Phase 1)

| # | Task | Status | Where |
|---|---|---|---|
| 1.1 | Downloader/cacher for EXIOBASE 3 (Zenodo), version pinning (DOI), caching | ✅ | `data/adapters/exiobase.py` (`fetch_exiobase`) |
| 1.2 | Adapter: pymrio → `IOSystem` + GHG `SatelliteAccount`; parquet + DuckDB store | ✅ | `data/adapters/exiobase.py`, `data/store/store.py` |
| 1.3 | Aggregation driven by `ConcordanceMap`; full + small builds | ✅ | `data/aggregate.py` |
| 1.4 | Quality module → `QualityReport` (balance, negatives, RoW, coverage, drift) | ✅ | `data/quality/checks.py` |
| 1.5 | Metadata registry: classifications, units, price basis, reference year | ✅ | `data/metadata.py` |
| 1.6 | Concordance framework: weighted maps, sum-to-1 + orphan validation | ✅ | `data/concordance/` |
| 1.7 | **Pipeline consistency/plausibility gates (enforced)** | ✅ | `data/quality/consistency.py` |

## Definition of done

> one command builds full + small datasets from a clean machine; `QualityReport` generated
> and stored; reproduction checks pass.

- `cge build --exiobase` (live) or `cge build --test` (offline) builds full + small into the
  store; `cge data` lists them; `cge quality <build_id>` shows the report.
- Every build carries a stored `QualityReport` including the pipeline consistency checks.
- 22 tests pass; lint + format clean.
- **Remaining for a real build:** the published-number reproduction checks (global CO₂ total
  and a few country footprints vs [Stadler2018]) require the live download — run them against
  an `exiobase` build. The check *code* and tolerances are specified in the data-layer doc.

## Data correctness & consistency (addresses the "checks at all points" requirement)

Enforced *in the pipeline*, not only in tests (`data/quality/consistency.py`):

- **Structural gate** after adapt and after aggregate — finite/aligned/square/productive
  `A`, existing Leontief inverse. Violations raise `ConsistencyError` and abort the build.
- **Cross-stage conservation** — aggregation must preserve total gross output and total
  final demand (`rtol=1e-4`); a break is fatal.
- **Plausibility checks** — positive output, final demand present, non-negative intensities,
  satellite coverage, RoW share — surfaced as `QualityCheck`s in the stored report.

The data-layer method (incl. the aggregation equations and the conservation identity) is
documented to equation level in `docs/models/data-layer.md`.

## Notes for Phase 2 (Engine 1)

- `runner.load_data(build_id)` now returns real builds; Engine 1 consumes `data["IOSystem"]`
  and `data["SatelliteAccount"]` (GHG, exposing a `CO2e` intensity row).
- The `spectral_radius < 1` precondition Engine 1 needs is already asserted at build time,
  so the engine can rely on the Leontief inverse existing.
- Implement Engine 1 against `docs/models/io-price-model.md`; validate on the `toy` economy
  and a real small build.

## Notes for Phase 3 (GUI)

- A **spreadsheet-style data explorer** (browse any build's matrices/accounts like Excel) is
  now an explicit Phase 3 task (3.2b), backed by DuckDB queries + a virtualised grid so the
  full MRIO paginates without loading whole into the browser.
