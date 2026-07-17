"""Validation suite for the data layer — economic identities on a built dataset.

Runs against the offline test build (so it needs no download); the same checks apply to a
live EXIOBASE build. Guards the invariants documented in docs/models/data-layer.md.
"""

from __future__ import annotations

import tempfile

import numpy as np

from cge.data.build import build_test
from cge.data.quality.consistency import total_output
from cge.data.store import DataStore
from cge.validation.framework import check

SUITE = "data_layer"


def _build():
    tmp = tempfile.mkdtemp()
    store = DataStore(tmp)
    written = build_test(store=store)
    return store, written


@check(SUITE, "leontief_inverse_exists")
def _productive():
    """ρ(A) < 1 on the built system (precondition for every downstream engine)."""
    store, written = _build()
    io = store.load(written["full"])["IOSystem"]
    rho = float(np.max(np.abs(np.linalg.eigvals(io.A.to_numpy(dtype=float)))))
    return rho < 1.0, f"ρ(A) = {rho:.4f}", rho, 1.0


@check(SUITE, "aggregation_conserves_output")
def _agg_output():
    """Aggregation to the small build preserves total gross output (flow-based agg)."""
    store, written = _build()
    fine = store.load(written["full"])["IOSystem"]
    coarse = store.load(written["small"])["IOSystem"]
    xf, xc = total_output(fine).sum(), total_output(coarse).sum()
    rel = abs(xf - xc) / abs(xf) if xf else 0.0
    return rel < 1e-4, f"output {xf:.6g} → {xc:.6g} (rel {rel:.2e})", rel, 1e-4


@check(SUITE, "aggregation_conserves_final_demand")
def _agg_fd():
    """Aggregation preserves total final demand."""
    store, written = _build()
    fine = store.load(written["full"])["IOSystem"]
    coarse = store.load(written["small"])["IOSystem"]
    ff = float(fine.final_demand.sum().sum())
    fc = float(coarse.final_demand.sum().sum())
    rel = abs(ff - fc) / abs(ff) if ff else 0.0
    return rel < 1e-4, f"final demand {ff:.6g} → {fc:.6g} (rel {rel:.2e})", rel, 1e-4


@check(SUITE, "stored_quality_passes")
def _quality():
    """The stored quality report for both builds has no FAIL-severity checks."""
    store, written = _build()
    ok = all(store.load_quality(b).passed for b in written.values())
    return ok, f"quality reports pass for builds: {list(written.values())}"
