"""Build data page (task 3.4): trigger a data build and stream its log.

Builds run in a **subprocess** (via the service's job wrapper) so a crash in the build can't
take down the app, and the log streams live. Note: the current implementation reads the
subprocess output synchronously within ``render()``, so *this browser tab* is busy until the
build finishes — the subprocess keeps the work out of the app process but does not make the
page interactive during a long download. A fully non-blocking build (background thread +
polling, or a job queue) is deferred to Phase 7; for now prefer the offline test build in the
UI and run long EXIOBASE builds from the CLI (`cge build --exiobase`).
"""

from __future__ import annotations

import streamlit as st

from cge.gui.service import get_service


def render() -> None:
    st.title("🏗️ Build data")
    st.caption(
        "Create a data build. The offline test build needs no download; the live "
        "EXIOBASE build fetches from Zenodo (large)."
    )
    svc = get_service()

    kind = st.radio(
        "Source",
        ["Offline test build (no download)", "Live EXIOBASE build"],
        help="Start with the test build to try the pipeline; use EXIOBASE for real data.",
    )
    is_test = kind.startswith("Offline")
    year = 2019
    if not is_test:
        year = st.number_input("EXIOBASE year", min_value=1995, max_value=2022, value=2019)
        st.warning("The live build downloads several GB and can take a long time.")

    if st.button("Start build", type="primary"):
        proc = svc.start_build(test=is_test, year=int(year))
        st.write("Build started — streaming log:")
        log_area = st.empty()
        lines: list[str] = []
        # Stream stdout line by line into the UI.
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip())
            log_area.code("\n".join(lines[-400:]))
        code = proc.wait()
        if code == 0:
            st.success("Build finished. See the **Data catalogue** and **Data quality** pages.")
        else:
            st.error(f"Build failed (exit {code}). Check the log above.")

    st.divider()
    st.subheader("Existing builds")
    cat = svc.catalogue()
    if cat.empty:
        st.write("None yet.")
    else:
        st.dataframe(cat, width="stretch", hide_index=True)
