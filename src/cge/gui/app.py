"""Streamlit entry point.

Run with:  streamlit run src/cge/gui/app.py   (or:  cge gui)

Navigation uses Streamlit's native multipage API (``st.navigation`` / ``st.Page``), one
module per page under ``gui/pages``. Pages call the ``GuiService`` façade; none reach into
the store/runner/registry directly (keeps the UI decoupled — see the service module).
"""

from __future__ import annotations

import streamlit as st

from cge.gui.pages import build as build_page
from cge.gui.pages import catalogue as catalogue_page
from cge.gui.pages import explorer as explorer_page
from cge.gui.pages import quality as quality_page
from cge.gui.pages import results as results_page
from cge.gui.pages import run as run_page


def main() -> None:
    st.set_page_config(page_title="CGE / IAM platform", page_icon="🌍", layout="wide")

    pages = [
        st.Page(catalogue_page.render, title="Data catalogue", icon="📚", default=True),
        st.Page(explorer_page.render, title="Data explorer", icon="🔎"),
        st.Page(quality_page.render, title="Data quality", icon="✅"),
        st.Page(build_page.render, title="Build data", icon="🏗️"),
        st.Page(run_page.render, title="Run scenario", icon="▶️"),
        st.Page(results_page.render, title="Results", icon="📊"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
