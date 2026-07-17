"""Data catalogue page (task 3.2): what data we have, versions, coverage, lineage, licence."""

from __future__ import annotations

import streamlit as st

from cge.gui.service import get_service


def render() -> None:
    st.title("📚 Data catalogue")
    st.caption("Datasets in the store — versions, size, quality verdict, provenance.")
    svc = get_service()

    cat = svc.catalogue()
    if cat.empty:
        st.info(
            "No data builds yet. Go to **Build data** to create one (offline test build "
            "needs no download)."
        )
        return

    st.dataframe(cat, width="stretch", hide_index=True)

    st.subheader("Build detail")
    build_id = st.selectbox("Build", svc.build_ids())
    if not build_id:
        return
    meta = svc.build_meta(build_id)
    col1, col2, col3 = st.columns(3)
    col1.metric("Products (labels)", len(svc.label_axis(build_id)))
    col2.metric("Reference year", meta.reference_year)
    col3.metric("Aggregation", meta.aggregation)

    with st.expander("Provenance & licence", expanded=True):
        st.write(
            {
                "source": meta.source,
                "source_version": meta.source_version,
                "licence": meta.licence,
                "price_basis": meta.price_basis,
                "currency": meta.currency,
                "unit": meta.monetary_unit,
                "retrieved": meta.retrieved,
                "notes": meta.notes,
            }
        )
