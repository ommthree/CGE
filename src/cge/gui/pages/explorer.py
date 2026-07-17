"""Spreadsheet-style data explorer (task 3.2b).

Browse any build's matrices and accounts like an Excel sheet: pick a frame (A-matrix,
final demand, satellite intensities), filter/search rows and columns by label, and view a
sortable, scrollable grid with CSV export.

Scale discipline: the A-matrix is products×products (up to ~9800² on the full MRIO). We
never render it whole — the user slices to a manageable window (row/column label filters +
a hard cap on displayed cells) before the grid renders. ``st.dataframe`` virtualises the
visible rows so scrolling stays responsive.
"""

from __future__ import annotations

import streamlit as st

from cge.gui.service import FrameView, get_service

# Hard cap on rendered cells so a mis-slice can't try to draw millions of cells.
MAX_CELLS = 200_000


def _filter_labels(labels: list[str], query: str) -> list[str]:
    if not query:
        return labels
    q = query.lower()
    return [x for x in labels if q in str(x).lower()]


def _render_grid(view: FrameView) -> None:
    df = view.df
    st.caption(view.description)

    rows = [str(r) for r in df.index]
    cols = [str(c) for c in df.columns]

    c1, c2 = st.columns(2)
    row_q = c1.text_input("Filter rows (label contains)", key=f"{view.name}_row")
    col_q = c2.text_input("Filter columns (label contains)", key=f"{view.name}_col")

    keep_rows = _filter_labels(rows, row_q)
    keep_cols = _filter_labels(cols, col_q)

    # Cap columns first (matrices are wide); tell the user if we truncated.
    n_cells = len(keep_rows) * len(keep_cols)
    if n_cells > MAX_CELLS:
        max_cols = max(1, MAX_CELLS // max(len(keep_rows), 1))
        st.warning(
            f"Slice has {n_cells:,} cells (> {MAX_CELLS:,}). Showing the first "
            f"{max_cols:,} columns — narrow the filters to see more."
        )
        keep_cols = keep_cols[:max_cols]

    sub = df.loc[
        [df.index[rows.index(r)] for r in keep_rows], [df.columns[cols.index(c)] for c in keep_cols]
    ]

    st.write(f"**{sub.shape[0]:,} rows × {sub.shape[1]:,} columns**")
    st.dataframe(sub, width="stretch")
    st.download_button(
        "Download this slice (CSV)",
        sub.to_csv().encode(),
        file_name=f"{view.name}_slice.csv",
        mime="text/csv",
    )


def render() -> None:
    st.title("🔎 Data explorer")
    st.caption("Browse a build's matrices and accounts like a spreadsheet.")
    svc = get_service()

    build_ids = svc.build_ids()
    if not build_ids:
        st.info("No data builds yet — create one on the **Build data** page.")
        return

    build_id = st.selectbox("Build", build_ids)
    frames = svc.frames(build_id)
    frame_name = st.radio("Frame", list(frames), horizontal=True)
    _render_grid(frames[frame_name])
