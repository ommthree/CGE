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


def test_cge_calibrates_and_replicates_on_built_sam(small_build_io):
    """THE 5.1b gate: the pilot CGE calibrates on a SAM built from an EXIOBASE-shaped build (the
    offline pymrio test MRIO, not live EXIOBASE) and replicates its benchmark to machine precision
    (proves the SAM→calibrate→solve pipeline works on structured multi-region data)."""
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


# -- open SAM from a real build (Phase 5 deferred: live-EXIOBASE open-SAM build) --------------
def test_build_open_sam_balanced_and_quality_passes(small_build_io):
    """An OPEN SAM built from the multi-region build (home region + rest-of-world) is balanced by
    construction and passes the SAM quality gates, with a_<s>/c_<s>/ROW accounts."""
    from cge.data.sam import build_open_sam

    io, _store, _bid = small_build_io
    sam, report, sectors = build_open_sam(io, home_region="A")
    assert is_balanced(sam.matrix, tol=1e-6)
    assert report.passed
    assert "ROW" in sam.accounts
    assert all(f"a_{s}" in sam.accounts and f"c_{s}" in sam.accounts for s in sectors)


def test_open_cge_calibrates_and_replicates_on_built_open_sam(small_build_io):
    """The open CGE calibrates on a SAM built from an EXIOBASE-shaped build and replicates its
    benchmark to machine precision — the open analogue of the 5.1b gate, proving the
    IOSystem→open-SAM→calibrate→solve pipeline works on structured multi-region data."""
    from cge.data.sam import build_open_sam
    from cge.engines.cge_static import model_open as MO
    from cge.engines.cge_static.calibrate_open import calibrate_open

    io, _store, _bid = small_build_io
    sam, _report, sectors = build_open_sam(io, home_region="A")
    cal = calibrate_open(sam, sectors=sectors, factors=["CAP", "LAB"])
    ns = len(sectors)
    sol = solve(
        lambda z: MO.residuals(cal, z, recycling="lump_sum"),
        MO.initial_guess(cal) * 1.03,
        prefer="scipy",
    )
    st = MO.derive_open_state(
        cal,
        sol.x[:ns],
        sol.x[ns : 2 * ns],
        sol.x[2 * ns : 2 * ns + 2],
        float(sol.x[-1]),
        recycling="lump_sum",
        strict=True,
    )
    assert sol.residual_norm < 1e-8
    assert np.allclose(st.Z, cal.Z0, atol=1e-6)
    assert np.allclose(st.M, cal.M0, atol=1e-6)
    assert np.allclose(st.E, cal.E0, atol=1e-6)


def test_engine_open_run_from_iosystem(small_build_io):
    """The engine builds an open SAM from an IOSystem when open_home_region is set, dispatches to
    the open path, and replicates on a zero shock (the full IOSystem→open-CGE wiring)."""
    from cge.engines.cge_static.engine import CGEStaticEngine

    io, store, bid = small_build_io
    sat = store.load(bid)["SatelliteAccount"]
    res = CGEStaticEngine().run(
        data={"IOSystem": io, "SatelliteAccount": sat, "open_home_region": "A"},
        shocks=[CarbonPrice(price=0.0)],
        years=[2020],
    )
    assert res.data["value"].abs().max() < 1e-6  # zero-shock replication
    assert "open economy" in res.manifest.assumptions["model_variant"]
    assert res.manifest.assumptions["sam_quality"]["worst"] == "pass"
    assert (res.data["variable"] == "import_change").any()


def test_open_sam_unknown_home_region_rejected(small_build_io):
    """An unknown home region is rejected (the home economy must be one of the build's regions)."""
    from cge.data.sam import build_open_sam

    io, _store, _bid = small_build_io
    with pytest.raises(ValueError, match="not in build regions"):
        build_open_sam(io, home_region="Z")


# -- multi-region SAM from a real build (Phase 5.1b) -----------------------------------------
def test_build_multi_sam_balanced_and_quality_passes(small_build_io):
    """A multi-region SAM built from the multi-region build (every region a genuine region with
    bilateral trade) is balanced by construction and passes the SAM quality gates, with
    a_<r>_<s>/c_<r>_<s>/<f>_<r>/HOH_<r> accounts and NO ROW account (closed global economy)."""
    from cge.data.sam import build_multi_sam

    io, _store, _bid = small_build_io
    sam, report, regions, sectors = build_multi_sam(io)
    assert is_balanced(sam.matrix, tol=1e-6)
    assert report.passed
    assert "ROW" not in sam.accounts
    assert len(regions) >= 2
    for r in regions:
        assert f"HOH_{r}" in sam.accounts
        assert all(f"a_{r}_{s}" in sam.accounts and f"c_{r}_{s}" in sam.accounts for s in sectors)
    # Final demand is MEASURED (the build carries by-region final demand), not imputed.
    assert any(
        c.name == "open_fd_attribution" and c.severity.value == "pass" for c in report.checks
    )


def test_build_multi_sam_preserves_aggregates(small_build_io):
    """The multi-region reduction preserves the source EXIOBASE aggregates (gross output, final
    demand, value added) — the conservation-through-a-transform gate, summed over all regions."""
    from cge.data.sam import build_multi_raw_sam

    io, _store, _bid = small_build_io
    raw = build_multi_raw_sam(io)
    m = raw.sam.matrix
    factor_rows = [f"{f}_{r}" for r in raw.regions for f in ("CAP", "LAB")]
    va_cols = [f"a_{r}_{s}" for r in raw.regions for s in raw.sectors]
    fd_rows = [f"c_{r}_{s}" for r in raw.regions for s in raw.sectors]
    fd_cols = [f"HOH_{r}" for r in raw.regions]
    sam_va = m.loc[factor_rows, va_cols].to_numpy().sum()
    sam_fd = m.loc[fd_rows, fd_cols].to_numpy().sum()
    assert np.isclose(sam_va, raw.source_value_added, rtol=1e-9)
    assert np.isclose(sam_fd, raw.source_final_demand, rtol=1e-9)
    assert np.isclose(sam_va, sam_fd, rtol=1e-6)  # global GDP identity


def test_multi_cge_calibrates_and_replicates_on_built_sam(small_build_io):
    """THE Phase 5.1b DoD: the multi-region CGE calibrates on a SAM built from an EXIOBASE-shaped
    build and replicates its benchmark to machine precision (the full
    IOSystem→multi-SAM→calibrate→solve pipeline on structured multi-region data)."""
    from cge.data.sam import build_multi_sam
    from cge.engines.cge_static import model_multi as MM
    from cge.engines.cge_static.calibrate_multi import calibrate_multi

    io, _store, _bid = small_build_io
    sam, _report, regions, sectors = build_multi_sam(io)
    cal = calibrate_multi(sam, regions=regions, sectors=sectors, factors=["CAP", "LAB"])
    assert cal.active_routes  # genuine bilateral trade survives materiality
    assert len(cal.connected_components) == 1  # connected region graph
    sol = solve(
        lambda z: MM.residuals(cal, z, recycling="lump_sum"),
        MM.initial_guess(cal) * 1.03,
        prefer="scipy",
    )
    st = MM.unpack_state(cal, sol.x, recycling="lump_sum", strict=True)
    assert sol.residual_norm < 1e-8
    assert np.allclose(st.Z, cal.Z0, atol=1e-6)
    assert np.allclose(st.M, cal.M0, atol=1e-6)


def test_no_retained_route_lacks_a_clearing_residual(small_build_io):
    """Review P1 (2026-07-26): the builder's route-drop threshold and the calibrator's
    ``active_routes`` threshold now use ONE threshold and ONE (global) denominator, so there is no
    'retained but inactive' route — every route with a nonzero Armington import share is in
    ``active_routes`` (has a clearing residual), and every active route clears at the solution."""
    from cge.data.sam import build_multi_sam
    from cge.engines.cge_static import model_multi as MM
    from cge.engines.cge_static.calibrate_multi import calibrate_multi

    io, _store, _bid = small_build_io
    sam, _report, regions, sectors = build_multi_sam(io)
    cal = calibrate_multi(sam, regions=regions, sectors=sectors, factors=["CAP", "LAB"])
    active = set(cal.active_routes)
    # Every route USED by demand (nonzero import share) must be an ACTIVE (cleared) route.
    for oi in range(cal.nr):
        for di in range(cal.nr):
            if oi == di:
                continue
            for si in range(cal.ns):
                if cal.arm_share_m[di, si, oi] > 0.0:
                    assert (oi, si, di) in active, (
                        f"route {regions[oi]}->{regions[di]} [{sectors[si]}] has a nonzero import "
                        "share but no clearing residual"
                    )
    # And every active route actually clears at the benchmark solution.
    sol = solve(lambda z: MM.residuals(cal, z), MM.initial_guess(cal) * 1.02, prefer="scipy")
    st = MM.unpack_state(cal, sol.x, strict=True)
    worst = max(
        (abs(float(st.M[di, si, oi] - st.EX[oi, si, di])) for (oi, si, di) in cal.active_routes),
        default=0.0,
    )
    assert worst < 1e-8


def test_build_multi_sam_needs_by_region_final_demand(small_build_io):
    """A build with only an AGGREGATE final-demand column cannot be reduced to a multi-region SAM
    (each region's own final demand cannot be attributed without inventing the split) — rejected,
    not imputed (unlike the single-region-open builder, which can impute a single home region)."""
    from cge.contracts.data_objects import IOSystem
    from cge.data.sam import build_multi_sam

    io, _store, _bid = small_build_io
    # Collapse to an aggregate FD column (IOSystem defaults final_demand_kind to "aggregate").
    agg = IOSystem(
        provenance=io.provenance,
        sectors=io.sectors,
        regions=io.regions,
        price_basis=io.price_basis,
        currency=io.currency,
        unit=io.unit,
        A=io.A,
        final_demand=io.final_demand.sum(axis=1).to_frame("final_demand"),
    )
    assert agg.fd_by_region() is None
    with pytest.raises(ValueError, match="BY CONSUMING REGION|by-region"):
        build_multi_sam(agg)


def test_build_multi_sam_rejects_disconnected_topology():
    """A DISCONNECTED region-trade graph (two regions with no bilateral trade between them) is
    rejected with a TopologyError — the single-numéraire multi-region closure is under-determined
    on a disconnected graph. Built by hand so the disconnection is unambiguous."""

    import pandas as pd

    from cge.contracts.data_objects import Classification, IOSystem, Provenance
    from cge.data.sam import TopologyError, build_multi_sam

    prov = Provenance(
        source="x", source_version="1", licence="x", reference_year=2020, retrieved="2026-07-26"
    )
    regions, sectors = ["A", "B"], ["g"]
    labels = ["A:g", "B:g"]
    # Block-diagonal A: A and B never buy from each other. FD is within-region only. No trade link.
    A = pd.DataFrame([[0.2, 0.0], [0.0, 0.2]], index=labels, columns=labels)
    fd = pd.DataFrame({"A": [80.0, 0.0], "B": [0.0, 90.0]}, index=labels)
    io = IOSystem(
        provenance=prov,
        sectors=Classification(name="s", kind="sector", labels=sectors),
        regions=Classification(name="r", kind="region", labels=regions),
        A=A,
        final_demand=fd,
        unit="MEUR",
        currency="EUR",
        final_demand_kind="by_region",
    )
    with pytest.raises(TopologyError, match="DISCONNECTED"):
        build_multi_sam(io)


def test_engine_multi_run_from_iosystem(small_build_io):
    """The engine builds a multi-region SAM from an IOSystem when multi_region=True, dispatches to
    the multi path, and replicates on a zero shock (the full IOSystem→multi-CGE wiring)."""
    from cge.engines.cge_static.engine import CGEStaticEngine

    io, store, bid = small_build_io
    sat = store.load(bid)["SatelliteAccount"]
    res = CGEStaticEngine().run(
        data={"IOSystem": io, "SatelliteAccount": sat, "multi_region": True},
        shocks=[CarbonPrice(price=0.0)],
        years=[2020],
    )
    assert res.data["value"].abs().max() < 1e-6  # zero-shock replication
    assert res.manifest.assumptions["sam_quality"]["worst"] == "pass"
    assert (res.data["variable"] == "import_change").any()
    assert len(res.manifest.assumptions["regions"]) >= 2


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


def test_build_open_sam_both_home_regions_and_surplus_closure(small_build_io):
    """P1 regressions, both at once. (1) Export-surplus closure: the surplus region builds a VALID
    open SAM — the surplus is closed by a household → ROW outflow cell, not a negative ROW → HOH
    entry (previously the surplus region failed the non-negativity gate). (2) Measured home final
    demand: with FD retained by consuming region the two single-region reductions are MIRROR
    images — A's exports equal B's imports and vice versa — which the imputed construction could
    not achieve. Both regions run end-to-end and replicate on a zero shock."""
    from cge.data.sam import build_open_sam
    from cge.engines.cge_static.engine import CGEStaticEngine

    io, _store, _bid = small_build_io
    trade = {}
    for home in ("A", "B"):
        sam, report, sectors = build_open_sam(io, home_region=home)
        assert report.passed
        m = sam.matrix
        assert m.to_numpy().min() >= 0.0  # no negative cells either way round
        assert any(
            c.name == "open_fd_attribution" and c.severity.value == "pass" for c in report.checks
        )  # measured, not imputed
        trade[home] = (
            sum(m.loc[f"a_{s}", "ROW"] for s in sectors),  # exports
            sum(m.loc["ROW", f"c_{s}"] for s in sectors),  # imports
            float(m.loc["HOH", "ROW"]),
            float(m.loc["ROW", "HOH"]),
        )
        res = CGEStaticEngine().run(
            data={"IOSystem": io, "open_home_region": home},
            shocks=[CarbonPrice(price=0.0)],
            years=[2020],
        )
        assert res.data["value"].abs().max() < 1e-6  # zero-shock replication
    # Mirror consistency: one region's exports are the other's imports (measured attribution).
    assert trade["A"][0] == pytest.approx(trade["B"][1], rel=1e-6)
    assert trade["B"][0] == pytest.approx(trade["A"][1], rel=1e-6)
    # The surplus region lends abroad (HOH→ROW cell), the deficit region receives (ROW→HOH) —
    # exactly one direction populated on each side.
    (ea, ma, in_a, out_a), (eb, mb, in_b, out_b) = trade["A"], trade["B"]
    assert ea > ma and out_a > 0 and in_a == 0.0  # A: surplus → outflow
    assert mb > eb and in_b > 0 and out_b == 0.0  # B: deficit → inflow


def test_fd_by_region_discriminator_is_explicit_not_inferred(small_build_io):
    """P2 (review round 9): fd_by_region() reads the EXPLICIT final_demand_kind discriminator, not
    a column-set-equality guess. The real fixture from the adapter is labelled by_region and
    behaves accordingly. THE P1 follow-up (review round 10): the reverse mislabelling —
    genuinely multi-column (by-region-shaped) data labelled "aggregate" — must now be REJECTED at
    construction, not silently accepted and then routed through synthetic imputation
    (final_demand_kind='aggregate' means exactly one column, enforced the same way 'by_region'
    enforces completeness)."""
    from cge.contracts.data_objects import IOSystem

    io, _store, _bid = small_build_io
    assert io.final_demand_kind == "by_region"  # the adapter sets this explicitly
    assert io.fd_by_region() is not None

    # Multi-column data mislabelled "aggregate" is now rejected outright.
    with pytest.raises(ValueError, match="final_demand_kind='aggregate' requires exactly one"):
        IOSystem(
            provenance=io.provenance,
            sectors=io.sectors,
            regions=io.regions,
            price_basis=io.price_basis,
            currency=io.currency,
            unit=io.unit,
            A=io.A,
            final_demand=io.final_demand,
            final_demand_kind="aggregate",
        )


def test_fd_by_region_rejects_incomplete_or_duplicate_columns(small_build_io):
    """THE P2 regression: an INCOMPLETE by-region frame (one region's column dropped — e.g. by a
    bug upstream) must be REJECTED at construction, not silently misclassified as a legacy
    aggregate. Previously: dropping region B's column made the column-set check fail, so
    fd_by_region() silently fell through to None and the corrupted data was never flagged."""
    from cge.contracts.data_objects import IOSystem

    io, _store, _bid = small_build_io

    # Drop one region's column: an incomplete by-region frame, not a valid legacy aggregate.
    with pytest.raises(ValueError, match="final-demand column is missing"):
        IOSystem(
            provenance=io.provenance,
            sectors=io.sectors,
            regions=io.regions,
            price_basis=io.price_basis,
            currency=io.currency,
            unit=io.unit,
            A=io.A,
            final_demand=io.final_demand.drop(columns=[io.regions.labels[0]]),
            final_demand_kind="by_region",
        )

    # A duplicated region column is equally invalid.
    dup = io.final_demand.copy()
    dup.columns = [io.regions.labels[0]] * len(dup.columns)  # collapse to one repeated label
    with pytest.raises(ValueError, match="duplicate consuming-region columns"):
        IOSystem(
            provenance=io.provenance,
            sectors=io.sectors,
            regions=io.regions,
            price_basis=io.price_basis,
            currency=io.currency,
            unit=io.unit,
            A=io.A,
            final_demand=dup,
            final_demand_kind="by_region",
        )


def test_build_open_sam_imputed_fd_is_explicit(small_build_io):
    """P1: a legacy build with only an AGGREGATE final-demand column still builds (the documented
    import-share imputation), but the synthetic construction is EXPLICIT — a WARN quality check and
    a provenance note — rather than silent."""
    from cge.contracts.data_objects import IOSystem
    from cge.data.sam import build_open_sam

    io, _store, _bid = small_build_io
    legacy = IOSystem(
        provenance=io.provenance,
        sectors=io.sectors,
        regions=io.regions,
        price_basis=io.price_basis,
        currency=io.currency,
        unit=io.unit,
        A=io.A,
        final_demand=io.final_demand.sum(axis=1).to_frame("final_demand"),  # collapse the split
    )
    assert legacy.fd_by_region() is None
    sam, report, _sectors = build_open_sam(legacy, home_region="A")
    check = next(c for c in report.checks if c.name == "open_fd_attribution")
    assert check.severity.value == "warn"  # synthetic → visible, not hidden
    assert "imputed" in check.message
    assert "SYNTHETIC" in sam.provenance.notes


# -- IO-backed OPEN path: effective carbon cost (review P0: price was applied twice) ----------
def _run_open_io(small_build_io, shocks, years):
    from cge.engines.cge_static.engine import CGEStaticEngine

    io, store, bid = small_build_io
    sat = store.load(bid)["SatelliteAccount"]
    return CGEStaticEngine().run(
        data={"IOSystem": io, "SatelliteAccount": sat, "open_home_region": "A"},
        shocks=shocks,
        years=years,
    )


def test_open_io_carbon_cost_linear_in_price(small_build_io):
    """THE P0 regression: on the IO-backed open path the response is ~linear in the carbon price.
    The double-application bug multiplied the price twice, so doubling the price QUADRUPLED the
    (small) response; with the fix the ratio is ~2."""
    r50 = _run_open_io(small_build_io, [CarbonPrice(price=50.0)], [2020])
    r100 = _run_open_io(small_build_io, [CarbonPrice(price=100.0)], [2020])

    def _max_price(res):
        d = res.data
        return d[d["variable"] == "price_change"]["value"].abs().max()

    ratio = _max_price(r100) / _max_price(r50)
    assert abs(ratio - 2.0) < 0.05, f"response not linear in price: ratio={ratio:.3f}"


def test_open_io_price_path_zero_then_positive(small_build_io):
    """A price PATH is honoured per year on the IO-backed open path: a year priced at zero
    replicates the benchmark while a later positive year moves it."""
    res = _run_open_io(
        small_build_io, [CarbonPrice(price=0.0, path={2020: 0.0, 2030: 100.0})], [2020, 2030]
    )
    d = res.data
    assert d[d["year"] == 2020]["value"].abs().max() < 1e-7  # unpriced year replicates
    assert d[d["year"] == 2030]["value"].abs().max() > 0  # priced year responds


def test_open_io_gas_selection_and_multi_shock_composition(small_build_io):
    """Gas selection is HONOURED on the IO-backed open path (carbon_cost_vector applies it), not
    rejected like on the supplied-SAM path: gases=['CO2e'] runs, and — because the fixture's CO2e
    row equals its CO2 row — reproduces the CO2 result exactly (it read the requested row). Two
    stacked shocks compose: the effective cost doubles, so the near-linear response ~doubles."""
    r_co2 = _run_open_io(small_build_io, [CarbonPrice(price=100.0)], [2020])
    r_co2e = _run_open_io(small_build_io, [CarbonPrice(price=100.0, gases=["CO2e"])], [2020])
    v1 = r_co2.data[r_co2.data["variable"] == "price_change"]["value"].to_numpy()
    v2 = r_co2e.data[r_co2e.data["variable"] == "price_change"]["value"].to_numpy()
    assert np.allclose(v1, v2, atol=1e-12)  # identical intensity rows → identical result

    r_two = _run_open_io(
        small_build_io, [CarbonPrice(price=100.0), CarbonPrice(price=100.0)], [2020]
    )

    def _max_price(res):
        d = res.data
        return d[d["variable"] == "price_change"]["value"].abs().max()

    ratio = _max_price(r_two) / _max_price(r_co2)
    assert abs(ratio - 2.0) < 0.05, f"shocks do not compose: ratio={ratio:.3f}"


def test_open_io_coverage_honoured(small_build_io):
    """Spatial coverage is honoured on the IO-backed open path: pricing only the OTHER region
    leaves the home economy unpriced (benchmark replication), and an unknown coverage label is
    rejected up front rather than silently pricing nothing."""
    res = _run_open_io(small_build_io, [CarbonPrice(price=100.0, coverage_regions=["B"])], [2020])
    assert res.data["value"].abs().max() < 1e-7  # home region A is outside the coverage
    with pytest.raises(ValueError, match="coverage"):
        _run_open_io(small_build_io, [CarbonPrice(price=100.0, coverage_regions=["ZZ"])], [2020])


def test_open_io_manifest_records_effective_cost_and_satellite(small_build_io):
    """The IO-backed open manifest carries the hashed effective carbon-cost matrix (price-included,
    named EffectiveCarbonCost — not the share) AND the SatelliteAccount identity (review P1)."""
    res = _run_open_io(small_build_io, [CarbonPrice(price=100.0)], [2020])
    input_names = {i.get("name") for i in res.manifest.assumptions["inputs"]}
    assert {"SAM", "EffectiveCarbonCost", "SatelliteAccount"} <= input_names


def test_open_io_missing_satellite_with_positive_price_rejected(small_build_io):
    """THE second P1 regression (2026-07 review round 9): a positive carbon price on the IO-backed
    open path with NO SatelliteAccount supplied must raise — the same gate the closed IO path
    already has — rather than being silently accepted as a zero-impact run with
    emissions_priced=False. Previously reproduced: €100/t ran with exactly zero impact and no
    error, contradicting the closed path's own validation."""
    from cge.engines.cge_static.engine import CGEStaticEngine

    io, _store, _bid = small_build_io
    with pytest.raises(ValueError, match="SatelliteAccount"):
        CGEStaticEngine().run(
            data={"IOSystem": io, "open_home_region": "A"},  # no SatelliteAccount
            shocks=[CarbonPrice(price=100.0)],
            years=[2020],
        )


def test_open_io_zero_price_without_satellite_is_a_genuine_zero(small_build_io):
    """A ZERO carbon price with no satellite is a legitimate zero-impact run, distinct from the
    missing-satellite-with-positive-price case above — no satellite is needed when nothing is
    being priced."""
    from cge.engines.cge_static.engine import CGEStaticEngine

    io, _store, _bid = small_build_io
    res = CGEStaticEngine().run(
        data={"IOSystem": io, "open_home_region": "A"},
        shocks=[CarbonPrice(price=0.0)],
        years=[2020],
    )
    assert res.data["value"].abs().max() < 1e-9


def test_open_io_coverage_excludes_home_still_records_satellite_provenance(small_build_io):
    """A satellite that IS supplied but whose effective cost comes out zero (coverage prices only
    the other region) is still a genuine zero, distinct from a missing satellite — and the
    manifest must still record that a real SatelliteAccount was consulted (review P1 follow-up:
    provenance was previously dropped whenever the effective cost happened to be zero)."""
    res = _run_open_io(small_build_io, [CarbonPrice(price=100.0, coverage_regions=["B"])], [2020])
    assert (
        res.data["value"].abs().max() < 1e-7
    )  # home region A is outside the coverage: a real zero
    input_names = {i.get("name") for i in res.manifest.assumptions["inputs"]}
    assert "SatelliteAccount" in input_names
    assert "EffectiveCarbonCost" not in input_names  # nothing was actually priced


def test_zero_shock_replicates_on_real_build(small_build_io):
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    _io, store, bid = small_build_io
    sc = Scenario(name="cge0", engine="cge_static", years=[2020], shocks=[CarbonPrice(price=0.0)])
    res = run_scenario(sc, data_source=bid, store=store)
    assert res.data["value"].abs().max() < 1e-7  # every change ~0 at zero carbon price
