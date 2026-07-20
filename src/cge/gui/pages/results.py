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
    is_cge = result.manifest.engine_name == "cge_static"

    if rv.has_volume(result) and is_cge:
        st.info(
            "Δprice is a **fractional** change in the unit price index (baseline = 1), shown as "
            "a **percent**. This is a **general-equilibrium** run: the volume response below comes "
            "from the CGE (input substitution + factor markets), not a partial-equilibrium "
            "elasticity model. Magnitudes are indicative (pilot calibration)."
        )
    elif rv.has_volume(result):
        st.info(
            "Δprice is a **fractional** change in the unit price index (baseline = 1), shown as "
            "a **percent** (e.g. +6.0% = a 6% price rise). This engine also estimates the "
            "**production-volume** response below (indicative — elasticity-dependent)."
        )
    else:
        st.info(
            "Δprice is a **fractional** change in the unit price index (baseline = 1), shown as "
            "a **percent** (e.g. +6.0% = a 6% price rise). **Cost impact only** — this engine "
            "models no volume response (use the partial-equilibrium engine for volumes)."
        )
    stats = rv.summary_stats(result)
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Goods", stats["goods"])
        c2.metric("Mean Δprice", f"{stats['mean'] * 100:+.2f}%")
        c3.metric("Max Δprice", f"{stats['max'] * 100:+.2f}%")
        c4.metric("Min Δprice", f"{stats['min'] * 100:+.2f}%")

    # -- headline table (fractional + percent) ---------------------------------
    st.subheader("Price change by good")
    table = rv.headline_table(result).copy()
    table["change_%"] = (table["value"] * 100).round(3)
    st.dataframe(table, width="stretch", hide_index=True)

    # -- volume response -------------------------------------------------------
    if rv.has_volume(result):
        st.subheader("Volume change by good")
        if is_cge:
            st.caption(
                "Δx/x — the **general-equilibrium** change in each sector's output, from the CGE "
                "(input substitution + factor markets clearing). With revenue recycling a carbon "
                "price **reallocates** output from dirty to clean sectors rather than only "
                "shrinking it. Indicative magnitudes (pilot calibration). Shown as percent."
            )
        else:
            st.caption(
                "Δx/x — fractional change in **production** (gross-output) volume, shown as "
                "percent. A carbon price raises prices; the finite-change demand response "
                "Δy/y=(1+Δp)^ε−1 cuts final demand; that propagates through the Leontief system "
                "x=(I−A)⁻¹y so upstream suppliers fall too. low/central/high span the "
                "demand-elasticity uncertainty (partial-equilibrium engine — **indicative**). "
                "Negative = volume falls."
            )
        env = rv.volume_envelope(result).copy()
        for b in ("low", "central", "high"):
            if b in env.columns:
                env[f"{b}_%"] = (env[b] * 100).round(2)
        show = ["region", "sector", "year"] + [
            f"{b}_%" for b in ("low", "central", "high") if f"{b}_%" in env.columns
        ]
        st.dataframe(env[show], width="stretch", hide_index=True)

    # -- macro aggregates ------------------------------------------------------
    if rv.has_macro(result):
        st.subheader("Macroeconomic aggregates")
        if is_cge:
            st.caption(
                "GDP and welfare are **native CGE equilibrium outputs** — not a post-hoc roll-up. "
                "The household's CD consumer price index is the **numéraire**, so *real* GDP is in "
                "CPI units and there is **no separate inflation/deflator** (a wage-numéraire "
                "nominal figure is shown for reference). Shown as percent."
            )
        else:
            st.caption(
                "GDP and value added rolled up from the per-good responses (**indicative PE "
                "tier**, not a general-equilibrium result). **Nominal** includes the price effect; "
                "**real** deflates it by the region's value-added-weighted GDP deflator. A "
                "price-only run (Engine 1) shows inflation with ~0 real GDP; a volume run "
                "(Engine 2) shows real GDP falling. Shown as percent."
            )
        gdp = rv.macro_gdp_table(result).copy()
        for c in (
            "GDP Δ (nominal)",
            "GDP Δ (real)",
            "GDP Δ (nominal, wage-numéraire)",
            "deflator (inflation)",
        ):
            if c in gdp.columns:
                gdp[c] = (gdp[c] * 100).round(2)
        st.dataframe(gdp, width="stretch", hide_index=True)
        with st.expander("Value added by sector (nominal & real)", expanded=False):
            gva = rv.macro_gva_table(result).copy()
            for c in ("GVA Δ (nominal)", "GVA Δ (real)"):
                if c in gva.columns:
                    gva[c] = (gva[c] * 100).round(2)
            st.dataframe(gva, width="stretch", hide_index=True)

    # -- trade (open / multi-region CGE) ---------------------------------------
    if rv.has_trade(result):
        st.subheader("Trade")
        st.caption(
            "Δ imports / Δ exports per sector, from the Armington/CET trade block — a "
            "general-equilibrium output, not a banded estimate. Shown as percent."
        )
        trade = rv.trade_table(result).copy()
        for c in ("Imports Δ", "Exports Δ"):
            if c in trade.columns:
                trade[c] = (trade[c] * 100).round(2)
        st.dataframe(trade, width="stretch", hide_index=True)
        er = rv.exchange_rate_table(result)
        if er is not None:
            er = er.copy()
            er["Exchange rate Δ"] = (er["Exchange rate Δ"] * 100).round(2)
            st.caption(
                "Exchange rate: the single-region open economy's numéraire-adjacent price of "
                "foreign currency (the multi-region model has no exchange rate — trade is "
                "entirely among the build's own regions)."
            )
            st.dataframe(er, width="stretch", hide_index=True)

    # -- factor prices ----------------------------------------------------------
    if rv.has_factor_prices(result):
        st.subheader("Factor prices")
        st.caption(
            "The wage and capital-rental-rate change as production reallocates between sectors "
            "with different factor intensities. Shown as percent."
        )
        fp = rv.factor_price_table(result).copy()
        fp["Factor price Δ"] = (fp["Factor price Δ"] * 100).round(2)
        st.dataframe(fp, width="stretch", hide_index=True)

    # -- welfare & carbon revenue -------------------------------------------------
    if rv.has_welfare(result):
        st.subheader("Welfare & carbon revenue")
        st.caption(
            "Welfare Δ is the household's utility change (Cobb-Douglas index over consumption); "
            "carbon revenue is the tax collected, as a share of benchmark GDP. Shown as percent."
        )
        wf_table = rv.welfare_table(result).copy()
        for c in ("Welfare Δ", "Carbon revenue (share of GDP)"):
            if c in wf_table.columns:
                wf_table[c] = (wf_table[c] * 100).round(2)
        st.dataframe(wf_table, width="stretch", hide_index=True)

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

    # -- export (with provenance) ----------------------------------------------
    st.subheader("Export")
    st.caption("Exports carry the run manifest so results stay traceable to their inputs.")
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Results (CSV)",
        result.data.to_csv(index=False).encode(),
        file_name="results.csv",
        mime="text/csv",
        help="Data only; download the manifest alongside for provenance.",
    )
    c2.download_button(
        "Results (Parquet + manifest)",
        _to_parquet(result),
        file_name="results.parquet",
        mime="application/octet-stream",
        help="Parquet with the run manifest embedded in file metadata.",
    )
    c3.download_button(
        "Manifest (JSON)",
        result.manifest.model_dump_json(indent=2).encode(),
        file_name="manifest.json",
        mime="application/json",
    )


def _to_parquet(result) -> bytes:
    """Parquet bytes with the run manifest embedded in the file's key-value metadata, so a
    downloaded result file remains traceable to the data build and scenario that produced it."""
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(result.data, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"cge_manifest"] = result.manifest.model_dump_json().encode()
    table = table.replace_schema_metadata(meta)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()
