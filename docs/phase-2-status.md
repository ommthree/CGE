# Phase 2 — status

Phase 2 ("Engine 1: IO carbon-cost pass-through") is implemented: the change in the price of
every good under a carbon price, with full supply-chain pass-through and a direct-vs-upstream
decomposition.

> **Post-review (2026-07, engine v0.2.0).** An independent review found real defects — a ~1e9
> units error, a broken live-build path, and ignored gas/time-path controls — since fixed. See
> [`review-2026-07-remediation.md`](review-2026-07-remediation.md). Honest status: **validated
> on the toy economy and internal identities (incl. a hand-derived known answer, units, gas
> selection, time paths); NOT yet validated against live EXIOBASE published multipliers**, and
> the engine is dense/small-build-only. Treat real-build numbers as indicative until the live
> known-answer test is added.

## Tasks (roadmap §Phase 2)

| # | Task | Status | Where |
|---|---|---|---|
| 2.1 | Leontief price model: Δp = (I−Aᵀ)⁻¹·τ·e via linear solve; productivity guard | ✅ | `engines/io_price/engine.py` |
| 2.2 | `CarbonPrice` scenario: level, gas/region/sector coverage; YAML round-trip | ✅ | contract `shocks.py` + `examples/carbon_price_io.yaml` |
| 2.3 | Decomposition: direct + Neumann-series upstream tiers + residual (sums to Δp) | ✅ | `engine.py` (`decompose`) |
| 2.4 | Validation: analytic + identity checks; assumptions in `RunManifest` | ✅ (live known-answer pending) | `validation/suites/io_price.py` |

## Definition of done

> CLI run of "€100/tCO₂" returns Δprice for every good in every region with decomposition,
> in seconds; tests green; model doc matches the implementation.

- `cge run --scenario examples/carbon_price_io.yaml` returns Δp + decomposition for every
  label (toy and any store build, e.g. `--data exiobase-test-small`), in milliseconds.
- `cge engines` lists `io_price`; the GUI/CLI will pick it up purely via the registry.
- Unit tests + the validation suite pass; lint + format clean. (Counts grow with each
  review round — see the current `pytest`/`cge validate` output rather than a fixed number.)
- Model doc `docs/models/io-price-model.md` matches the code (manifest assumptions restate
  the engine's `ASSUMPTIONS`); status is "implemented; validated on toy + identities, **not**
  against live EXIOBASE" (see `review-2026-07-remediation.md`).

## Validation suite (addresses the "comprehensive validation tests + script" request)

New standing model-validation subsystem (`cge.validation`, `docs/validation.md`):

- Framework (`framework.py`): `@check`-registered checks grouped into named suites, run by a
  registry; a check that raises is a failure, not a crash.
- Runner: `scripts/validate.py` and `cge validate` (`--suite`, `--strict`, `--markdown`),
  plus `tests/test_validation.py` so CI fails on any model regression.
- Suites: `io_price` (checks mapping to the model doc §7 — analytic, units, gas selection,
  time path, coverage, well-posedness, known-answer) and `data_layer` (economic identities
  on a built dataset).

**The suite already earned its keep:** it caught that the well-posedness guard relied on the
linear solve failing, which `np.linalg.solve` does not do for merely non-productive matrices.
The engine now checks ρ(A) explicitly (documented in the model doc's implementation note).

## What's genuine vs pending

- **Genuine now:** correct Leontief pass-through, decomposition, coverage filtering, uses the
  `CO2e` intensity on real builds and `CO2` on the toy — all validated against theory.
- **Pending live data:** the known-answer check reproducing published EXIOBASE CO₂
  multipliers within tolerance [Stadler2018] needs an `exiobase` build. It's specified in the
  model doc §7 and the validation doc; add it to the `io_price` suite once real data is built.

## Notes for Phase 3 (GUI)

- The results explorer can render the decomposition directly: each good has `price_change`
  plus `price_change_direct` / `price_change_upstream_tier_{1,2,3}` / `..._residual` rows —
  a ready-made waterfall.
- Every run's `RunManifest.assumptions` carries the model's caveats (incl. "UPPER BOUND on
  cost impact; NO volume effects") for the assumptions printout.
