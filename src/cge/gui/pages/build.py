"""Build data page (task 3.4): trigger a data build and stream its log.

Builds run as a background subprocess (via the service's job wrapper) so a long EXIOBASE
download doesn't freeze the UI; the page streams stdout live and refreshes the catalogue
when the job finishes.
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
