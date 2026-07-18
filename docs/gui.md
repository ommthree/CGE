# Web GUI (Phase 3)

A Streamlit app for the everyday workflow: see what data exists, browse it like a
spreadsheet, check data quality, build datasets, run carbon-price and energy-price scenarios,
and explore results — without touching the CLI.

## Launch

```bash
pip install -e ".[gui]"     # if not already installed
cge gui                     # or: streamlit run src/cge/gui/app.py
```

Then open the local URL Streamlit prints (default http://localhost:8501).

## Pages

| Page | Purpose | Task |
|---|---|---|
| **Data catalogue** | What builds exist — versions, size, quality verdict, provenance, licence. | 3.2 |
| **Data explorer** | Spreadsheet-style grid over any build's A-matrix / final demand / satellite intensities. Filter rows & columns by label, scroll, export the slice to CSV. | 3.2b |
| **Data quality** | The stored `QualityReport` per build, incl. pipeline consistency/plausibility checks, worst issues first. | 3.3 |
| **Build data** | Trigger an offline test build or a live EXIOBASE build; streams the job log live (background subprocess so the UI never freezes). | 3.4 |
| **Run scenario** | Pick data + engine + a carbon price and/or any number of energy-carrier price shocks (carrier + % change + region coverage), composed into one scenario and run in-process. Both the engine picker and which shock controls appear are driven by registry metadata (`supported_shocks`), so new engines/shocks appear with no page changes. | 3.5 |
| **Results** | Headline Δprice table, supply-chain decomposition waterfall per good, the run's assumptions, CSV/Parquet export. | 3.6 |

## Architecture

Pages never import the store/runner/registry directly. Two Streamlit-free modules hold all
the logic, so it is unit-tested without a server:

- **`gui/service.py`** — the `GuiService` façade: enumerate builds, load frames, read
  quality, list engines, run scenarios, start builds. Holds a `DataStore` and threads it
  into the runner so GUI runs resolve builds from the store the GUI browses.
- **`gui/results_view.py`** — reshape a `ResultSet` into headline tables and waterfall
  series.

`gui/app.py` wires pages into Streamlit's native multipage navigation. Each page module
exposes a single `render()`.

## Scale discipline (the spreadsheet view)

The A-matrix is products×products — up to ~9800² on the full MRIO. The explorer never
renders it whole: the user filters row/column labels and a hard cell cap (200k) truncates
with a visible warning, so a mis-slice can't try to draw millions of cells. `st.dataframe`
virtualises the visible rows.

## Testing

- **`tests/test_gui_service.py`** — the Streamlit-free logic (service + results reshaping),
  including that waterfall parts sum to the headline value.
- **`tests/test_gui_pages.py`** — every page rendered headless via Streamlit's `AppTest`,
  asserting no exception. Runs in CI (needs the `[gui]` extra).

## Deliberately deferred (roadmap notes)

- Runs execute in-process; a job queue is a Phase 7 concern (fine for the small build).
- Side-by-side run comparison is scoped for a later pass; v1 keeps every page to
  "table + one chart + export" to avoid scope creep.
