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


_CGE_OPEN_RESULTS_SETUP = """
import streamlit as st
from cge.scenarios.loader import Scenario
from cge.contracts.shocks import CarbonPrice
from cge.runner import run_scenario
sc = Scenario(name="t", engine="cge_static", years=[2020], shocks=[CarbonPrice(price=50.0)])
st.session_state["last_result"] = run_scenario(sc, data_source="toy_cge_open")
st.session_state["last_scenario"] = sc
"""


def test_results_page_renders_trade_factor_welfare_sections_for_open_cge():
    """P2 (review round 9): the Results page must render its new Trade / Factor prices /
    Welfare & carbon revenue sections for an open-economy CGE run, not just the price/volume/macro
    tables the io_price/partial_eq engines produce."""
    src = f"from cge.gui.pages import results\n{_CGE_OPEN_RESULTS_SETUP}\nresults.render()\n"
    at = AppTest.from_string(src, default_timeout=60)
    at.run()
    assert not at.exception, (
        f"results page raised: {[getattr(e, 'value', e) for e in at.exception]}"
    )
    headers = [h.value for h in at.subheader]
    assert "Trade" in headers
    assert "Factor prices" in headers
    assert "Welfare & carbon revenue" in headers


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


def test_run_page_exposes_gases_and_recycling_for_cge():
    """Review P3: the Run page exposes gas selection (all engines) and a revenue-recycling control
    for a general-equilibrium engine (the CGE), and a CGE run on a build works end to end."""
    src = "from cge.gui.pages import run\nrun.render()\n"
    at = AppTest.from_string(src, default_timeout=120).run()
    # Gas selection is always present.
    assert any(ms.label == "Gases priced" for ms in at.multiselect)
    # Pick the CGE engine + a real build (the fixture built one) → recycling control appears.
    data_sel = at.selectbox[0]  # 'Data'
    build = next((o for o in data_sel.options if o != "toy"), None)
    engine_sel = at.selectbox[1]  # 'Engine'
    if build is None or "cge_static" not in engine_sel.options:
        return  # environment without a build / CGE — nothing to assert
    data_sel.set_value(build)
    engine_sel.set_value("cge_static")
    at.run()
    assert not at.exception, f"CGE run page raised: {at.exception}"
    assert any(sb.label == "Revenue recycling" for sb in at.selectbox)
    at.button[0].click().run()
    assert not at.exception, f"CGE run raised: {at.exception}"
    res = at.session_state["last_result"]
    assert res is not None and res.manifest.engine_name == "cge_static"
