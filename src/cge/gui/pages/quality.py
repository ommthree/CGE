"""Data quality dashboard (task 3.3): render a build's QualityReport, incl. the pipeline
consistency/plausibility checks, with the worst issues surfaced first."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from cge.contracts.quality import Severity
from cge.gui.service import get_service

_SEV_ICON = {Severity.PASS: "✅", Severity.WARN: "⚠️", Severity.FAIL: "❌"}
_SEV_ORDER = {Severity.FAIL: 0, Severity.WARN: 1, Severity.PASS: 2}


def render() -> None:
    st.title("✅ Data quality")
    st.caption("Balance identities, plausibility, and pipeline consistency checks per build.")
    svc = get_service()

    build_ids = svc.build_ids()
    if not build_ids:
        st.info("No data builds yet — create one on the **Build data** page.")
        return

    build_id = st.selectbox("Build", build_ids)
    report = svc.quality(build_id)
    if report is None:
        st.warning(f"No quality report stored for {build_id!r}.")
        return

    summary = report.summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall", _SEV_ICON[report.worst] + " " + report.worst.value)
    c2.metric("Pass", summary.get("pass", 0))
    c3.metric("Warn", summary.get("warn", 0))
    c4.metric("Fail", summary.get("fail", 0))

    # Worst-first table.
    rows = sorted(report.checks, key=lambda c: _SEV_ORDER[c.severity])
    df = pd.DataFrame(
        [
            {
                "": _SEV_ICON[c.severity],
                "check": c.name,
                "message": c.message,
                "value": c.value,
                "tolerance": c.tolerance,
            }
            for c in rows
        ]
    )
    st.dataframe(df, width="stretch", hide_index=True)

    failed = [c for c in report.checks if c.severity == Severity.FAIL]
    if failed:
        st.error(f"{len(failed)} check(s) FAILED — this build should not be trusted as-is.")
    elif any(c.severity == Severity.WARN for c in report.checks):
        st.info("Some warnings — review before relying on edge-case results.")
    else:
        st.success("All checks passed.")
