# Phase 2 — status

Phase 2 ("Engine 1: IO carbon-cost pass-through") from `roadmap.md` is complete. The
platform now produces **real economic answers**: the change in the price of every good under
a carbon price, with full supply-chain pass-through and a direct-vs-upstream decomposition.

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
- 33 unit tests + the validation suite pass; lint + format clean.
- Model doc `docs/models/io-price-model.md` matches the code (assumptions generated from the
  engine's `ASSUMPTIONS`), and its status is now "implemented & validated".

## Validation suite (addresses the "comprehensive validation tests + script" request)

New standing model-validation subsystem (`cge.validation`, `docs/validation.md`):

- Framework (`framework.py`): `@check`-registered checks grouped into named suites, run by a
  registry; a check that raises is a failure, not a crash.
- Runner: `scripts/validate.py` and `cge validate` (`--suite`, `--strict`, `--markdown`),
  plus `tests/test_validation.py` so CI fails on any model regression.
- Suites: `io_price` (9 checks mapping to the model doc §7) and `data_layer` (4 economic
  identities on a built dataset).

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
