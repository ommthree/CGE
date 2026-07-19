"""Run scenario page (task 3.5): build a scenario from a form and run it.

The engine picker and the available shock controls are driven entirely by registry metadata
(``EngineMeta.supported_shocks``), so a new engine/shock appears here with no page changes. The
form composes a **carbon price** and any number of **energy-carrier price** shocks into one
scenario — they add, exactly as the engines compose them. The result is stashed in session state
for the Results page.
"""

from __future__ import annotations

import streamlit as st

from cge.contracts.shocks import CarbonPrice, EnergyPrice
from cge.gui.service import get_service
from cge.scenarios.loader import Scenario

# Carriers to *suggest* first in the picker when present (coarse-build energy sectors); the picker
# still offers every sector in the build, so a differently-named energy sector works too.
_SUGGESTED_CARRIERS = ("energy", "energy_coal", "energy_oil_gas", "electricity")


def render() -> None:
    st.title("▶️ Run scenario")
    svc = get_service()

    build_ids = svc.build_ids()
    data_options = ["toy"] + build_ids
    data_source = st.selectbox(
        "Data", data_options, help="'toy' is the built-in fixture; others are store builds."
    )

    engines = svc.engines()
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

    # Region coverage options come from the build's labels (empty for the toy → text is clearer).
    labels = svc.label_axis(data_source) if data_source != "toy" else []
    regions = sorted({lab.split(":", 1)[0] for lab in labels}) if labels else []
    year = st.number_input("Year", min_value=1995, max_value=2100, value=2020)

    shocks: list = []

    # -- Carbon price ----------------------------------------------------------
    if "carbon_price" in meta.supported_shocks:
        st.subheader("Carbon price")
        use_carbon = st.checkbox("Apply a carbon price", value=True)
        if use_carbon:
            price = st.slider("Carbon price (currency / tCO₂e)", 0.0, 500.0, 100.0, step=10.0)
            c_regions = st.multiselect(
                "Carbon price — region coverage (empty = all)", regions, default=[], key="c_regions"
            )
            # Gas selection: which GHGs the price applies to (must exist in the build's GHG
            # account). Default CO2. 'CO2e' cannot be mixed with component gases.
            gases = st.multiselect(
                "Gases priced", ["CO2", "CH4", "N2O", "CO2e"], default=["CO2"], key="c_gases"
            ) or ["CO2"]
            # Revenue recycling — only meaningful for a general-equilibrium engine (the CGE); the
            # cost-push engines (io_price/partial_eq) have no government budget and reject it.
            recycling = "none"
            if "general_equilibrium" in [c.value for c in meta.capabilities]:
                recycling = st.selectbox(
                    "Revenue recycling",
                    ["lump_sum", "labour_tax_cut", "none"],
                    help="How the carbon revenue is returned. A closed CGE cannot destroy it, so "
                    "'none' auto-defaults to lump_sum; use Engine 1 for a pure price-side view.",
                    key="c_recycling",
                )
            if price > 0:
                shocks.append(
                    CarbonPrice(
                        price=price,
                        gases=gases,
                        coverage_regions=c_regions,
                        revenue_recycling=recycling,
                    )
                )
    else:
        st.info(f"Engine {engine_name!r} does not support carbon-price shocks.")

    # -- Energy-carrier price shocks ------------------------------------------
    if "energy_price" in meta.supported_shocks:
        st.subheader("Energy-carrier prices")
        st.caption(
            "A rise (or fall) in an energy carrier's **output price** — the carrier's price is "
            "pinned to exactly this change and propagates downstream through the supply chain."
        )
        sectors = svc.sectors(data_source)
        # Order suggested carriers first, then the rest, so the common choice is at the top.
        carriers = [c for c in _SUGGESTED_CARRIERS if c in sectors] + [
            s for s in sectors if s not in _SUGGESTED_CARRIERS
        ]
        n = st.number_input(
            "Number of energy-price shocks", min_value=0, max_value=5, value=0, step=1
        )
        for i in range(int(n)):
            cols = st.columns([2, 2, 3])
            carrier = cols[0].selectbox(f"Carrier #{i + 1}", carriers, key=f"ep_carrier_{i}")
            change_pct = cols[1].number_input(
                f"Change % #{i + 1}",
                min_value=-100.0,
                max_value=500.0,
                value=30.0,
                step=5.0,
                key=f"ep_change_{i}",
            )
            e_regions = cols[2].multiselect(
                f"Regions #{i + 1} (empty = all)", regions, default=[], key=f"ep_regions_{i}"
            )
            shocks.append(
                EnergyPrice(carrier=carrier, change=change_pct / 100.0, coverage_regions=e_regions)
            )

    # -- Run -------------------------------------------------------------------
    if not shocks:
        st.warning("Add at least one shock (a carbon price and/or an energy-price shock) to run.")
        return

    summary = ", ".join(_describe(s) for s in shocks)
    if st.button("Run", type="primary"):
        scenario = Scenario(
            name=f"{summary} on {data_source}",
            description="Built in the GUI run page.",
            engine=engine_name,
            years=[int(year)],
            shocks=shocks,
        )
        try:
            result = svc.run(scenario, data_source=data_source)
        except Exception as exc:  # surface engine/data errors in the UI, don't crash
            st.error(f"Run failed: {exc}")
            return
        st.session_state["last_result"] = result
        st.session_state["last_scenario"] = scenario
        st.success(f"Ran {scenario.name}. Open the **Results** page to explore.")
        st.metric("Products priced", int((result.data["variable"] == "price_change").sum()))


def _describe(shock) -> str:
    """A short human label for a shock, for the scenario name."""
    if isinstance(shock, CarbonPrice):
        return f"Carbon €{shock.price:.0f}/t"
    if isinstance(shock, EnergyPrice):
        return f"{shock.carrier} {shock.change:+.0%}"
    return shock.type
