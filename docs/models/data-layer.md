# Model description: Data layer (EXIOBASE ingestion, aggregation, quality)

- **Implements:** `cge.data.*` (adapters, aggregate, quality, store)
- **Roadmap phase:** 1
- **Status:** implemented (live EXIOBASE path + offline test path)

## 1. Purpose & scope

Turn a raw multi-regional input–output (MRIO) database into the harmonised data objects
engines consume, at two resolutions (a full build and an interactive "small" build), with a
machine-readable quality verdict and enforced consistency along the pipeline.

**In scope:** ingestion from the live EXIOBASE source, mapping to contracts, economically
correct sector/region aggregation, quality + consistency checks, versioned storage.

**Not modelled here:** any economics. This layer produces data; engines produce results.

## 2. Notation

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{A}$ | technical-coefficient matrix, $A_{ij}$ = input $i$ per unit output $j$ | $n\times n$ |
| $\mathbf{Z}$ | inter-industry flow matrix | $n\times n$ |
| $\mathbf{x}$ | gross output per product; $\hat{\mathbf{x}}$ its diagonal | $n\times 1$ |
| $\mathbf{f}$ | final demand | $n\times 1$ |
| $\mathbf{e}$ | satellite totals (e.g. emissions) per product | $k\times n$ |
| $\mathbf{B}$ | bridge (aggregation) matrix, $B_{ts}$ = weight of source $s$ in target $t$ | $m\times n$ |

Primed symbols denote the aggregated (coarse) system, dimension $m < n$.

## 3. Assumptions

1. EXIOBASE product-by-product tables at basic prices, one reference year per build.
2. GHG accounts combined to CO₂e via GWP-100 (AR5): CO₂=1, CH₄=28, N₂O=265.
3. Satellite accounts are stored as **intensities** (per unit gross output), so aggregation
   converts to totals and back.
4. Aggregation weights are a partition (each source maps to exactly one target, weight 1)
   for the default small build; the framework supports fractional weights for future maps.

## 4. Derivation — aggregation

Coefficients cannot be averaged; the economically correct procedure aggregates *flows*
[MillerBlair2009, §4.3]. Recover flows and output from the coefficient system:

$$
\mathbf{Z} = \mathbf{A}\,\hat{\mathbf{x}}, \qquad
\mathbf{x} = (\mathbf{I}-\mathbf{A})^{-1}\mathbf{f}. \tag{1}
$$

Aggregate flows, output and final demand with the bridge $\mathbf{B}$ (target×source):

$$
\mathbf{Z}' = \mathbf{B}\,\mathbf{Z}\,\mathbf{B}^{\!\top}, \qquad
\mathbf{x}' = \mathbf{B}\,\mathbf{x}, \qquad
\mathbf{f}' = \mathbf{B}\,\mathbf{f}. \tag{2}
$$

Recompute coefficients on the aggregated system:

$$
\mathbf{A}' = \mathbf{Z}'\,\hat{\mathbf{x}}'^{-1}. \tag{3}
$$

For a **multi-dimensional** (sector×region) label set, the combined bridge is the Kronecker
product of the region and sector bridges: $B_{(t_r t_s),(r s)} = B^{\text{reg}}_{t_r r}\,
B^{\text{sec}}_{t_s s}$.

Satellite **intensities** $\mathbf{s}=\mathbf{e}/\mathbf{x}$ are aggregated by returning to
totals first: $\mathbf{e}' = (\mathbf{s}\odot\mathbf{x})\,\mathbf{B}^{\!\top}$, then
$\mathbf{s}' = \mathbf{e}'/\mathbf{x}'$.

**Conservation (correctness property).** Because $\mathbf{B}$ is a partition
($\mathbf{1}^{\!\top}\mathbf{B} = \mathbf{1}^{\!\top}$), equations (2) preserve totals:
$\mathbf{1}^{\!\top}\mathbf{x}' = \mathbf{1}^{\!\top}\mathbf{x}$ and
$\mathbf{1}^{\!\top}\mathbf{f}' = \mathbf{1}^{\!\top}\mathbf{f}$. This identity is **checked
in the pipeline** (`check_aggregation_conserves`), not merely in tests — a violation fails
the build.

## 5. Algorithm & pipeline consistency

Pipeline: `fetch (live Zenodo) → parse → adapt → [gate 1] → quality → store`, and for the
small build `→ aggregate → [gate 2] → quality → store`.

Two enforced consistency gates (`cge.data.quality.consistency`):

- **Gate 1 — structural** (`assert_structural`): square, label-aligned, finite $\mathbf{A}$;
  aligned final demand and satellites; productive economy $\rho(\mathbf{A})<1$. Violations
  raise `ConsistencyError` — there is no usable build otherwise.
- **Gate 2 — cross-stage conservation**: equations (2) must hold to `rtol=1e-4` (float32
  storage + solve round-off). A break means the aggregation is wrong; the build fails.

Plausibility checks (soft, → warnings): positive gross output, non-empty final demand,
non-negative emission intensities. These join the stored `QualityReport`.

**Complexity.** Aggregation is dense matrix products $O(mn^2)$ plus one $n\times n$ solve;
seconds on the full MRIO. Full-MRIO memory is the real cost (roadmap P1 risk) — mitigated by
float32 parquet storage and doing analysis on the small build by default.

## 6. Data sourcing

- **Live:** `pymrio.download_exiobase3` from the pinned Zenodo DOI (`EXIOBASE_DOI`), product
  system, one year. Cached; re-runs skip existing files. This is the default real path.
- **Offline:** `pymrio.load_test()` — a tiny bundled MRIO with identical structure — runs
  the entire pipeline in CI without a multi-GB download. The test system's stressors are
  aliased onto CO₂ so a GHG account exists downstream; real EXIOBASE needs no alias.

## 7. Validation

`tests/test_data_layer.py`: adapter mapping; concordance orphan/weight invariants;
**aggregation conservation** (total output and final demand preserved, exact to `rtol`);
quality-report content; structural guard rejects NaN / non-productive systems; cross-stage
conservation checks reach the stored report; store round-trip; runner loads a build id.

Live-data known-answer checks (global CO₂ total and a few published country footprints vs
[Stadler2018]) are the remaining item requiring the actual download — the roadmap P1 DoD;
run against a real build.

## 8. References

[MillerBlair2009] (flow-based aggregation §4.3, Leontief identities); [Stadler2018],
[Wood2015] (EXIOBASE 3). Full entries in [`../references.md`](../references.md).
