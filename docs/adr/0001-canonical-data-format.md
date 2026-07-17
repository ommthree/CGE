# ADR-0001: Canonical tabular format is long-format pandas

- **Status:** accepted
- **Date:** 2026-07-17
- **Phase:** 0

## Context

Data objects (IO matrices, satellite accounts) and results need a canonical in-memory
representation. The candidates were wide/matrix-oriented pandas, `xarray` (labelled
N-dimensional arrays), and long-format ("tidy") pandas. The choice ripples through every
engine, the store, and the GUI, so it is worth an explicit decision.

The workloads differ:
- **Matrices** (the A-matrix, Leontief inverse) are inherently 2-D and linear-algebra
  heavy — naturally matrix-oriented.
- **Results** are a ragged collection of (variable, sector, region, year, scenario) →
  value, queried and compared every which way.

## Decision

- **Results** use **long format** with fixed columns (`RESULT_COLUMNS`). One row per
  observation. This is the interface every engine emits.
- **Matrix payloads** inside data objects stay as 2-D pandas DataFrames indexed by the
  sector×region product — engines do linear algebra directly on them.
- `xarray` is **not** adopted now; it may be reconsidered in Phase 7 if multi-region ×
  multi-year × multi-scenario result cubes make long format awkward for the GUI.

## Consequences

- Cross-engine, cross-scenario, cross-data-source comparison becomes a `groupby`/filter,
  not bespoke code — directly serves the "comparison is a query" goal.
- Long format serialises cleanly to parquet and loads into DuckDB for the GUI.
- Uncertainty bands (low/central/high) live in a `scenario` column rather than as
  separate result objects — the P4 uncertainty story falls out for free.
- Cost: long format is not how you do matrix algebra, hence the deliberate split
  (matrices stay wide inside data objects). Engines convert at their boundary.

## Alternatives considered

- **xarray everywhere:** elegant for the N-dimensional result cube, but adds a dependency
  and a mental model for a solo maintainer before it clearly earns its keep. Deferred, not
  rejected forever.
- **Wide results (sectors as columns):** compact for one variable/one year, but every new
  dimension (region, year, scenario, variable) forces a reshape; comparison logic sprawls.
