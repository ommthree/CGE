"""Run scenario page (task 3.5): build a scenario from a form and run it.

The engine picker is driven entirely by registry metadata — capabilities and supported
shocks are read from ``EngineMeta``, so a new engine appears here with no page changes. v1
exposes the carbon-price shock (Engine 1); the form generalises as more shock types land.
The result is stashed in session state for the Results page.
"""

from __future__ import annotations

import streamlit as st

from cge.contracts.shocks import CarbonPrice
from cge.gui.service import get_service
from cge.scenarios.loader import Scenario


def render() -> None:
    st.title("▶️ Run scenario")
    svc = get_service()

    build_ids = svc.build_ids()
    data_options = ["toy"] + build_ids
    data_source = st.selectbox(
        "Data", data_options, help="'toy' is the built-in fixture; others are store builds."
    )

    engines = svc.engines()
    # Only offer engines that can actually price a carbon shock in v1.
    names = [e.name for e in engines]
    engine_name = st.selectbox(
        "Engine", names, index=names.index("io_price") if "io_price" in names else 0
    )
    meta = svc.engine_meta(engine_name)

    with st.expander("Engine capabilities", expanded=False):
        st.write(
            {
                "version": meta.version,
                "capabilities": [c.value for c in meta.capabilities],
                "supported_shocks": meta.supported_shocks,
                "required_data": meta.required_data,
                "description": meta.description,
            }
        )

    st.subheader("Carbon price shock")
    if "carbon_price" not in meta.supported_shocks:
        st.warning(f"Engine {engine_name!r} does not support carbon-price shocks.")
        return

    price = st.slider("Carbon price (currency / tCO₂e)", 0.0, 500.0, 100.0, step=10.0)
    labels = svc.label_axis(data_source) if data_source != "toy" else []
    regions = sorted({lab.split(":", 1)[0] for lab in labels}) if labels else []
    coverage_regions = st.multiselect("Region coverage (empty = all regions)", regions, default=[])
    year = st.number_input("Year", min_value=1995, max_value=2100, value=2020)

    if st.button("Run", type="primary"):
        shock = CarbonPrice(price=price, coverage_regions=coverage_regions)
        scenario = Scenario(
            name=f"Carbon €{price:.0f}/t on {data_source}",
            description="Built in the GUI run page.",
            engine=engine_name,
            years=[int(year)],
            shocks=[shock],
        )
        try:
            result = svc.run(scenario, data_source=data_source)
        except Exception as exc:  # surface engine/data errors in the UI, don't crash
            st.error(f"Run failed: {exc}")
            return
        # Stash for the Results page.
        st.session_state["last_result"] = result
        st.session_state["last_scenario"] = scenario
        st.success(f"Ran {scenario.name}. Open the **Results** page to explore.")
        st.metric("Products priced", int((result.data["variable"] == "price_change").sum()))
