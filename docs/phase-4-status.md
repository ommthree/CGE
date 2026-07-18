# Phase 4 — status

Phase 4 ("Engine 2: partial-equilibrium volume response") is implemented. The platform now
answers the **volume** half of the original question: how much does the produced quantity of
each good change under a carbon price, with an explicit uncertainty band.

## Tasks (roadmap §Phase 4)

| # | Task | Status | Where |
|---|---|---|---|
| 4.1 | Elasticity library: schema (value, source, low/central/high, confidence); initial literature-sourced set | ✅ (functional defaults; curated set is follow-up) | `data/elasticities/` |
| 4.2 | PE engine: finite-change demand Δy/y=(1+Δp)^ε−1 propagated through Leontief x=(I−A)⁻¹y → Δx/x (production), on Engine-1 prices | ✅ | `engines/partial_eq/engine.py` |
| 4.3 | Uncertainty as a first-class output: low/central/high band → envelope | ✅ | engine emits a `volume_change` row per band |
| 4.4 | Validation: volume-sign / bounded > −100% / Leontief propagation / band-order / pass-through / provenance | ✅ (published-incidence comparison is follow-up) | `validation/suites/partial_eq.py`, `tests/test_partial_eq.py` |

## Definition of done

> volume impact *ranges* per sector/region for a carbon-price scenario; the GUI picks up the
> new engine purely via the registry; every result carries its elasticity provenance; model
> doc exists.

All satisfied:
- `cge run --scenario examples/carbon_price_volume.yaml` returns Δx/x (production volume) per
  good with a low/central/high band; negative (a carbon price reduces volumes),
  energy/manufacturing hit hardest.
- The engine appears in `cge engines` and the GUI Run page **purely via the registry** (no
  GUI code changed to add it); the Results page shows a volume-envelope table when a result
  has `volume_change` rows.
- Every result carries `elasticity_used` per good and a manifest count of goods using the
  default elasticity; Engine 1's price caveats are carried into the manifest too.
- Model doc `docs/models/partial-equilibrium.md` matches the code.
- The offline test suite passes (live-network tests skipped offline); the `partial_eq` validation
  suite (6 checks) and the full standing suite pass; lint + format clean. (Run `cge validate` and
  `pytest` for current totals — they grow as later phases add suites, e.g. Phase 4b's `macro`.)

## Design

- **Reuses Engine 1** for prices (single source of truth — no reimplementation of the price
  model), then applies the demand response. So Engine 1's units/gas/coverage correctness and
  the whole 2026-07 remediation carry straight through.
- **Uncertainty is first-class**: three elasticity bands → a `volume_change` row per band, so
  the answer is an envelope, not a point. This is the roadmap's mitigation for the fact that
  volume results are elasticity-sensitive.
- **Contract-clean**: adds `Capability.VOLUMES`; the GUI/CLI discovered it with no changes.

## Honest scope

- **Volume answers are indicative, not precise.** There is no clean open elasticity database;
  the default set (`data/elasticities/library.py`) is assembled, ranged, and tagged by
  confidence, with a flagged default for unmatched goods. Treat the envelope as screening-grade.
- **Own-price only in v1.** The Armington domestic/import substitution nest (model doc eq. 2)
  is specified but not implemented — it needs a build with a domestic/import price split.
- **Follow-ups:** a curated, sector-matched elasticity set (feeds the CGE too); the Armington
  nest; comparison against published carbon-price incidence/volume studies.

## Notes for Phase 5 (CGE)

- The elasticity library and its `ElasticitySet` contract are shared with the CGE calibration.
- Engine 2 remains useful after the CGE exists as a fast, transparent cross-check on GE volume
  responses (prices should bracket, volumes should be same-sign).
