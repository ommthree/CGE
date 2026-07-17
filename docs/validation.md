# Validation suite

Model-correctness validation — distinct from both unit tests and data quality reports:

- **Unit tests** (`pytest`) check that *code* behaves as written; they gate every change in CI.
- **Data quality/consistency** (`QualityReport`, `cge quality`) check that a *data build* is
  sound.
- **Validation suite** (this) checks that a *model* reproduces its known answers and
  economic identities — the standing, human-readable audit that the numbers a model
  produces are still correct against theory and published results.

The three overlap by design: the validation checks are also invoked from `pytest`
(`tests/test_validation.py`) so CI fails on any model regression, while `scripts/validate.py`
runs them as a report with a CI-friendly exit code.

## Running

```bash
cge validate                       # all suites, text report
cge validate --suite io_price      # one suite
cge validate --strict              # non-zero exit if any check fails (CI)

python scripts/validate.py --markdown docs/validation-report.md   # write a report
```

## How it is structured

`cge.validation.framework` provides a tiny structure mirroring the data-quality contract:

- **`ValidationResult`** — one check's outcome (passed, message, measured value, tolerance).
- **`Suite`** — a named group of checks; **`registry`** holds all suites.
- **`@check(suite, name)`** — decorator registering a function as a check. The function
  returns `(passed, message[, value[, tolerance]])`.
- **`run_all(only=…)`** → **`RunSummary`** — executes suites and aggregates; a check that
  raises is recorded as a failure, never crashes the run.

Suites live in `cge.validation.suites.*`; importing that package registers them. **Adding a
model adds a suite module and one import line — nothing else changes**, the same plug-in
pattern as engines.

## Current suites

| Suite | Guards | Doc |
|---|---|---|
| `io_price` | Engine 1 Leontief price model: analytic exactness, zero-shock, linearity, pass-through ≥ direct, decomposition sums to total, coverage filtering, well-posedness guard, end-to-end | [`models/io-price-model.md`](models/io-price-model.md) §7 |
| `data_layer` | Built dataset economic identities: Leontief inverse exists, aggregation conserves output & final demand, stored quality passes | [`models/data-layer.md`](models/data-layer.md) §7 |

## Convention: every model adds validation

Per the documentation standard, an engine/module is not "done" until it ships (a) an
equation-level model doc and (b) a validation suite whose checks map to that doc's stated
properties. Prefer checks that assert **against theory or a published number** (known-answer,
identity) over "looks plausible". The `io_price` suite is the reference example — its checks
name the equation or assumption each one guards.

## Known-answer checks against published data

Checks that need a live EXIOBASE build (e.g. reproducing published CO₂ multipliers within
tolerance, [Stadler2018]) are the highest-value validation but require the download. Add them
to the relevant suite guarded to skip cleanly when no live build is present, so `cge validate`
still runs offline but tightens automatically once real data is built.
