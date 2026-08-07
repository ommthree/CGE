"""Nature exposure & scenarios (Phase 6.5).

Three things, all within the P3 framework and driven by ``GuiService``:

1. **Dependency heatmap** — good × ecosystem-service exposure (direct, or direct + upstream), the
   output of the exposure engine (Phase 6.3). A green→red gradient reads "which goods depend on
   which services, and how much".
2. **Supply-chain drill-down** — for one good, its DIRECT dependency on each service vs. the TOTAL
   (direct + inherited-from-inputs) exposure, so the upstream propagation the exposure engine exists
   to surface is visible per good.
3. **Nature-scenario runner** — pick services to degrade and by how much, and run the whole
   ENCORE→exposure→``NatureStress``→``ProductivityShock``→engine chain end-to-end (Phase 6.4),
   showing the resulting per-good output response.

The shipped ENCORE ratings are a small, published-sourced **illustrative fixture** (the gated real
export drops into the same contract with no code change) — surfaced honestly at the top of the page.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from cge.gui.service import get_service

_ILLUSTRATIVE = (
    "The ENCORE dependency ratings here are a small hand-entered, **published-sourced illustrative "
    "fixture** on the toy economy, not the full (registration-gated) ENCORE knowledge base — which "
    "drops into the same contract with no code change. Read the numbers as *illustrative of the "
    "method*, not as calibrated risk. See `docs/models/nature-encore.md`."
)


def _gradient(df: pd.DataFrame):
    """A green(low)→red(high) background gradient on a [0,1] exposure frame, or the plain frame if
    the Styler/matplotlib path is unavailable in this Streamlit build."""
    try:
        return df.style.background_gradient(cmap="RdYlGn_r", vmin=0.0, vmax=1.0).format("{:.2f}")
    except (ImportError, ValueError):
        return df.round(2)


def _heatmap_section(svc) -> None:
    st.subheader("Dependency heatmap")
    st.caption("How much each good depends on each ecosystem service (0 = none, 1 = fully).")
    rule = st.radio(
        "Upstream aggregation",
        ["weighted_mean", "max"],
        horizontal=True,
        help=(
            "How a good inherits its inputs' dependencies: weighted_mean is a noisy-OR risk "
            "measure (total ≥ direct); max is the worst-in-chain screen. Both are exposed as a "
            "parameter, not buried."
        ),
    )
    direct, total, _io = svc.nature_exposure(rule=rule)
    view = st.radio(
        "Exposure", ["Direct + upstream (total)", "Direct only"], horizontal=True
    )
    frame = direct if view == "Direct only" else total
    st.dataframe(_gradient(frame), width="stretch")
    st.download_button(
        "Download exposure (CSV)",
        frame.to_csv().encode(),
        file_name="nature_exposure.csv",
        mime="text/csv",
    )


def _drilldown_section(svc) -> None:
    st.subheader("Supply-chain drill-down")
    st.caption(
        "For one good: its OWN dependency on each service (direct) vs. the TOTAL once dependencies "
        "inherited through its input supply chain are added — the upstream channel the exposure "
        "engine exists to surface."
    )
    direct, total, _io = svc.nature_exposure()
    good = st.selectbox("Good", list(total.index))
    table = pd.DataFrame(
        {"direct": direct.loc[good], "total (incl. upstream)": total.loc[good]}
    )
    table["upstream contribution"] = table["total (incl. upstream)"] - table["direct"]
    st.dataframe(_gradient(table.drop(columns="upstream contribution")), width="stretch")
    st.bar_chart(table[["direct", "upstream contribution"]])


def _scenario_section(svc) -> None:
    st.subheader("Nature-scenario runner")
    st.caption(
        "Degrade one or more ecosystem services and run the whole chain — exposure → NatureStress "
        "→ per-good ProductivityShock → economic engine — end-to-end."
    )
    services = svc.nature_services()
    chosen = st.multiselect(
        "Services to degrade", services, default=services[:1]
    )
    stresses: list[tuple[str, float]] = []
    for s in chosen:
        sev = st.slider(
            f"Severity — {s} (fraction of the service lost)",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            key=f"sev_{s}",
        )
        stresses.append((s, sev))

    engine = st.radio(
        "Economic engine",
        ["partial_eq", "cge_static"],
        horizontal=True,
        help=(
            "partial_eq is the partial-equilibrium supply hit; cge_static is the "
            "general-equilibrium response (factor reallocation, relative prices, cross-region "
            "leakage on the 2-region toy economy)."
        ),
    )

    if not st.button("Run nature scenario", type="primary"):
        return
    if not stresses:
        st.warning("Pick at least one service to degrade.")
        return
    with st.spinner("Running the nature scenario…"):
        result = svc.run_nature(stresses=stresses, engine=engine, years=[2020])

    df = result.data
    vol = df[(df["variable"] == "volume_change") & (df["scenario"] == "central")]
    if vol.empty:  # some variants band only central; fall back to any volume rows
        vol = df[df["variable"] == "volume_change"]
    if "region" in vol.columns:
        vol = vol.assign(good=vol["region"].astype(str) + ":" + vol["sector"].astype(str))
    else:
        vol = vol.assign(good=vol["sector"].astype(str))
    out = vol.set_index("good")["value"].sort_values()
    st.write("**Output response by good** (Δ output, central scenario)")
    st.bar_chart(out)
    st.caption(
        "Every good loses output under a degradation; the most-exposed goods (agriculture) lose "
        "most. In the CGE, an un-degraded region's goods can *gain* output as production relocates "
        "(the nature analogue of carbon leakage)."
    )
    st.dataframe(out.rename("volume_change").to_frame(), width="stretch")


def render() -> None:
    st.title("🌱 Nature exposure & scenarios")
    st.caption("Ecosystem-service dependency exposure and nature-degradation scenarios (Phase 6).")
    st.info(_ILLUSTRATIVE)
    svc = get_service()

    _heatmap_section(svc)
    st.divider()
    _drilldown_section(svc)
    st.divider()
    _scenario_section(svc)
