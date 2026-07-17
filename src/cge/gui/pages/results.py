"""Results page (task 3.6): headline table, decomposition waterfall, assumptions, export.

Reads the last run from session state (set by the Run page). Charts use altair. The
assumptions printout is mandatory — it is the credibility surface for a screening tool.
"""

from __future__ import annotations

import altair as alt
import streamlit as st

from cge.gui import results_view as rv


def render() -> None:
    st.title("📊 Results")
    result = st.session_state.get("last_result")
    scenario = st.session_state.get("last_scenario")
    if result is None:
        st.info("No run yet. Build a scenario on the **Run scenario** page.")
        return

    st.caption(
        f"Scenario: **{scenario.name}**  ·  engine `{result.manifest.engine_name}`  ·  "
        f"data `{result.manifest.data_source}`  ·  hash `{result.manifest.scenario_hash}`"
    )

    stats = rv.summary_stats(result)
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Goods", stats["goods"])
        c2.metric("Mean Δprice", f"{stats['mean']:.2f}")
        c3.metric("Max Δprice", f"{stats['max']:.2f}")
        c4.metric("Min Δprice", f"{stats['min']:.2f}")

    # -- headline table --------------------------------------------------------
    st.subheader("Price change by good")
    table = rv.headline_table(result)
    st.dataframe(table, width="stretch", hide_index=True)

    # -- decomposition waterfall ----------------------------------------------
    pairs = rv.goods_with_decomposition(result)
    if pairs:
        st.subheader("Supply-chain decomposition")
        label = st.selectbox(
            "Good",
            [f"{r}:{s}" for r, s in pairs],
            help="Direct emissions cost vs cost inherited from upstream inputs, by tier.",
        )
        region, sector = label.split(":", 1)
        wf = rv.waterfall(result, region=region, sector=sector)
        if not wf.empty:
            chart = (
                alt.Chart(wf)
                .mark_bar()
                .encode(
                    x=alt.X("part:N", sort=list(wf["part"]), title="contribution"),
                    y=alt.Y("value:Q", title="Δ price"),
                    tooltip=["part", "value"],
                )
            )
            st.altair_chart(chart, width="stretch")

    # -- assumptions (mandatory credibility surface) ---------------------------
    st.subheader("Assumptions behind these numbers")
    st.json(result.manifest.assumptions)

    # -- export ----------------------------------------------------------------
    st.subheader("Export")
    c1, c2 = st.columns(2)
    c1.download_button(
        "Results (CSV)",
        result.data.to_csv(index=False).encode(),
        file_name="results.csv",
        mime="text/csv",
    )
    c2.download_button(
        "Results (Parquet)",
        _to_parquet(result.data),
        file_name="results.parquet",
        mime="application/octet-stream",
    )


def _to_parquet(df) -> bytes:
    import io

    buf = io.BytesIO()
    df.to_parquet(buf)
    return buf.getvalue()
