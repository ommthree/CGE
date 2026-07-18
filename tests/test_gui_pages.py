"""Smoke tests: every GUI page renders without raising, via Streamlit's AppTest.

These complement test_gui_service.py (which covers the Streamlit-free logic): here we
actually execute each page's render() in a headless Streamlit runtime and assert no
exception surfaced. Requires a data build in the default store, so we make one first.
"""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from cge.data.build import build_test  # noqa: E402

PAGES = ["catalogue", "explorer", "quality", "build", "run", "results"]

_RESULTS_SETUP = """
import streamlit as st
from cge.scenarios.loader import Scenario
from cge.contracts.shocks import CarbonPrice
from cge.runner import run_scenario
sc = Scenario(name="t", engine="io_price", years=[2020], shocks=[CarbonPrice(price=100.0)])
st.session_state["last_result"] = run_scenario(sc, data_source="toy")
st.session_state["last_scenario"] = sc
"""


@pytest.fixture(scope="module", autouse=True)
def _ensure_build():
    # Pages read from the default store; ensure at least one build exists.
    build_test()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    extra = _RESULTS_SETUP if page == "results" else ""
    src = f"from cge.gui.pages import {page}\n{extra}\n{page}.render()\n"
    at = AppTest.from_string(src, default_timeout=60)
    at.run()
    assert not at.exception, f"{page} raised: {[getattr(e, 'value', e) for e in at.exception]}"


def test_run_page_energy_price_branch_renders():
    """Exercise the Run page's energy-price controls: set the shock count to 1 so the carrier /
    change / coverage widgets render, then trigger a run — the combined carbon+energy scenario
    must execute without error and produce a result."""
    src = "from cge.gui.pages import run\nrun.render()\n"
    at = AppTest.from_string(src, default_timeout=60).run()
    # Data source 'toy' (first selectbox), engine io_price (first engine selectbox).
    # Set the number of energy-price shocks (the number_input labelled below) to 1 and re-run.
    n_input = next(ni for ni in at.number_input if ni.label == "Number of energy-price shocks")
    n_input.set_value(1)
    at.run()
    assert not at.exception, f"energy-price branch raised: {at.exception}"
    # The carrier selectbox now exists; run the scenario.
    assert any(sb.label.startswith("Carrier #1") for sb in at.selectbox)
    at.button[0].click().run()
    assert not at.exception, f"combined run raised: {at.exception}"
    assert at.session_state["last_result"] is not None
