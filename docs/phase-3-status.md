# Phase 3 — status

Phase 3 ("GUI v1") from `roadmap.md` is complete. A Streamlit web app now covers the whole
everyday workflow — inspect data, browse it like a spreadsheet, check quality, build
datasets, run carbon-price scenarios, and explore results — over the Phase 0–2 stack.

## Tasks (roadmap §Phase 3)

| # | Task | Status | Where |
|---|---|---|---|
| 3.1 | Streamlit scaffold, navigation, page-module convention | ✅ | `gui/app.py`, `gui/pages/` |
| 3.2 | Data catalogue page | ✅ | `gui/pages/catalogue.py` |
| 3.2b | **Spreadsheet-style data explorer** | ✅ | `gui/pages/explorer.py` |
| 3.3 | Quality dashboard (incl. consistency/plausibility checks) | ✅ | `gui/pages/quality.py` |
| 3.4 | Build page with subprocess job wrapper + live log | ✅ | `gui/pages/build.py`, `service.start_build` |
| 3.5 | Scenario builder + run page (registry-driven engine picker) | ✅ | `gui/pages/run.py` |
| 3.6 | Results explorer: table, decomposition waterfall, assumptions, export | ✅ | `gui/pages/results.py`, `gui/results_view.py` |

## Definition of done

> a colleague can, without help, inspect data quality, run a carbon-price scenario,
> understand what assumptions produced the numbers, and export results.

All satisfied. Launch with `cge gui`; the six pages cover the full loop. Every results view
prints the run's assumptions (the credibility surface), and results export to CSV/Parquet.

- 46 tests pass (adds `test_gui_service.py` + `test_gui_pages.py`), lint + format clean.
- Every page verified to **render without exception** via Streamlit's `AppTest`, run in CI
  (needs the `[gui]` extra, now installed there).

## Design

Pages are thin renderers over two Streamlit-free modules (`service.py`, `results_view.py`)
so the logic is unit-testable without a server, and pages never reach into store/runner/
registry internals. See `docs/gui.md`.

## A real fix this phase surfaced

The GUI test caught that `run_scenario` always used the process-default store, ignoring any
store the caller (the service) held — so a GUI pointed at a non-default store would fail to
resolve its own builds. `run_scenario`/`load_data` now accept an optional `store`, and the
service threads its own through. This is exactly the kind of integration gap unit tests on
isolated modules miss and end-to-end page tests catch.

## Deferred (intentionally)

- In-process runs (job queue is Phase 7); side-by-side run comparison scoped for later. Held
  to "table + one chart + export" per page to avoid the GUI scope-creep the roadmap warns of.

## Notes for later phases

- New engines (Phase 4 partial-equilibrium, Phase 5 CGE) appear in the Run page's engine
  picker automatically via registry metadata — no GUI changes needed, only possibly a new
  shock control if they consume a new shock type.
- The results waterfall is Engine-1-specific (`price_change_*` rows); generalise the
  decomposition view when a second engine emits different variables.
