"""Tests for SAM construction from an EXIOBASE build (roadmap Phase 5.1).

Covers: the raw SAM is balanced and preserves the source aggregates; the quality report gates
it; the CGE pilot calibrates on the REAL SAM and replicates its benchmark (the 5.1b gate); and
the engine runs end-to-end on a build through the runner.
"""

import tempfile

import numpy as np
import pytest

from cge.contracts.shocks import CarbonPrice
from cge.data.build import build_test
from cge.data.sam import build_raw_sam, build_sam
from cge.data.sam.balance import is_balanced
from cge.data.store import DataStore
from cge.engines.cge_static import model as M
from cge.engines.cge_static.calibrate import calibrate
from cge.engines.cge_static.solver import solve


@pytest.fixture(scope="module")
def small_build_io():
    store = DataStore(tempfile.mkdtemp())
    build_test(store=store)
    small = next(b for b in store.build_ids() if b != "exiobase-test")
    return store.load(small)["IOSystem"], store, small


def test_raw_sam_is_balanced_and_preserves_aggregates(small_build_io):
    io, _store, _bid = small_build_io
    raw = build_raw_sam(io)
    assert is_balanced(raw.sam.matrix)
    m = raw.sam.matrix
    # Value added (factor cols into sectors) = final demand (HOH col) = source aggregates.
    sam_va = m.loc[["CAP", "LAB"], raw.sectors].to_numpy().sum()
    sam_fd = m.loc[raw.sectors, "HOH"].sum()
    assert np.isclose(sam_va, raw.source_value_added, rtol=1e-9)
    assert np.isclose(sam_fd, raw.source_final_demand, rtol=1e-9)
    assert np.isclose(sam_va, sam_fd, rtol=1e-9)  # GDP identity


def test_build_sam_quality_passes(small_build_io):
    io, _store, _bid = small_build_io
    sam, report, sectors = build_sam(io)
    assert report.passed
    names = {c.name for c in report.checks}
    assert {"sam_balanced", "preserves_final_demand", "preserves_value_added"} <= names
    # The capital-share assumption is recorded (the audit trail).
    assert any(c.name == "assumed_capital_share" for c in report.checks)
    assert len(sectors) == 3


def test_capital_share_out_of_range_rejected(small_build_io):
    io, _store, _bid = small_build_io
    with pytest.raises(ValueError, match="capital_share"):
        build_raw_sam(io, capital_share=1.5)


def test_cge_calibrates_and_replicates_on_real_sam(small_build_io):
    """THE 5.1b gate: the pilot CGE calibrates on the real EXIOBASE SAM and replicates its
    benchmark to machine precision (proves the model works on real data, not just the toy)."""
    io, _store, _bid = small_build_io
    sam, _report, sectors = build_sam(io)
    cal = calibrate(sam, sectors=sectors, factors=["CAP", "LAB"])
    # Benchmark residual is zero (normalised levels), so replication is exact.
    assert np.max(np.abs(M.residuals(cal, M.initial_guess(cal)))) < 1e-9
    sol = solve(lambda z: M.residuals(cal, z), M.initial_guess(cal) * 1.05, prefer="scipy")
    ns = len(sectors)
    st = M.derive_state(cal, sol.x[:ns], sol.x[ns:])
    assert np.allclose(sol.x, 1.0, atol=1e-8)  # all prices return to 1
    assert np.allclose(st.X, cal.X0, rtol=1e-6)  # outputs replicate


def test_engine_runs_on_real_build_via_runner(small_build_io):
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    _io, store, bid = small_build_io
    sc = Scenario(name="cge", engine="cge_static", years=[2020], shocks=[CarbonPrice(price=50.0)])
    res = run_scenario(sc, data_source=bid, store=store)
    d = res.data
    assert (d["variable"] == "price_change").any()
    assert (d["variable"] == "gdp_change_real").any()
    # SAM quality surfaced in the manifest, and emissions were priced from the satellite.
    assert res.manifest.assumptions["sam_quality"]["worst"] == "pass"
    assert res.manifest.assumptions["emissions_priced"] is True
    # Carbon revenue is collected and recycling defaulted to a closed-economy mode.
    rev = d[(d["variable"] == "carbon_revenue")]["value"].iloc[0]
    assert rev > 0
    assert res.manifest.assumptions["recycling_mode"] in ("lump_sum", "labour_tax_cut")
    # A non-zero GE response (the pymrio test fixture's emission intensities are small, so the
    # magnitude is tiny — but with the correct 1e-6 M→currency scaling it is finite and non-zero,
    # not the ~1e6-too-large blowup the units bug produced).
    prices = d[d["variable"] == "price_change"]["value"]
    assert 0 < prices.abs().max() < 1.0
    # Emissions provenance is recorded (satellite + effective cost-share), not just the SAM.
    input_names = {i.get("name") for i in res.manifest.assumptions["inputs"]}
    assert {"SAM", "EffectiveCarbonCostShare", "SatelliteAccount"} <= input_names


def test_zero_shock_replicates_on_real_build(small_build_io):
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    _io, store, bid = small_build_io
    sc = Scenario(name="cge0", engine="cge_static", years=[2020], shocks=[CarbonPrice(price=0.0)])
    res = run_scenario(sc, data_source=bid, store=store)
    assert res.data["value"].abs().max() < 1e-7  # every change ~0 at zero carbon price
