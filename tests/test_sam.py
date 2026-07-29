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
    # Value added (factor cols into sectors) = TOTAL final demand (over HOH + GOV + SAVINV when the
    # institution split routed them, review P1 round 13) = source aggregates.
    fd_cols = [c for c in ("HOH", "GOV", "SAVINV") if c in m.columns]
    sam_va = m.loc[["CAP", "LAB"], raw.sectors].to_numpy().sum()
    sam_fd = m.loc[raw.sectors, fd_cols].to_numpy().sum()
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
    # The IO build now carries GOV/SAVINV (institution split, review P1 round 13), so name them.
    cal = calibrate(
        sam,
        sectors=sectors,
        factors=["CAP", "LAB"],
        institutions={"household": "HOH", "government": "GOV", "savings_investment": "SAVINV"},
    )
    # Benchmark residual is zero (normalised levels), so replication is exact.
    assert np.max(np.abs(M.residuals(cal, M.initial_guess(cal)))) < 1e-9
    sol = solve(lambda z: M.residuals(cal, z), M.initial_guess(cal) * 1.05, prefer="scipy")
    ns = len(sectors)
    st = M.derive_state(cal, sol.x[:ns], sol.x[ns:])
    assert np.allclose(sol.x, 1.0, atol=1e-8)  # all prices return to 1
    assert np.allclose(st.X, cal.X0, rtol=1e-6)  # outputs replicate


def test_built_sam_routes_government_and_investment_from_fd_institutions(small_build_io):
    """Review P1 (round 13, 2026-07-28): the IO→SAM pipeline now routes government consumption and
    gross capital formation from the EXIOBASE FD institution split to GOV / SAVINV (financed
    by imputed tax / savings), so the macro closures run on a REAL built SAM — not only on
    hand-built fixtures. The CGE calibrates with a government AND an investment account."""
    from cge.engines.cge_static.calibrate import calibrate

    io, _store, _bid = small_build_io
    assert io.fd_by_institution() is not None  # the build carries the institution split
    sam, report, sectors = build_sam(io)
    assert "GOV" in sam.accounts and "SAVINV" in sam.accounts
    assert report.passed  # FD aggregate still preserved across HOH + GOV + SAVINV
    cal = calibrate(
        sam,
        sectors=sectors,
        factors=["CAP", "LAB"],
        institutions={"household": "HOH", "government": "GOV", "savings_investment": "SAVINV"},
    )
    assert cal.has_government and cal.has_investment
    # The imputation is provenance-flagged (tax/savings not sourced).
    assert "IMPUTED" in sam.provenance.notes


def test_built_sam_institutions_can_be_disabled(small_build_io):
    """With institutions=False the builder lumps all final demand into the household (the older
    behaviour), so no GOV/SAVINV accounts appear — a caller can opt out."""
    from cge.data.sam.build import build_raw_sam

    io, _store, _bid = small_build_io
    raw = build_raw_sam(io, institutions=False)
    assert "GOV" not in raw.sam.accounts and "SAVINV" not in raw.sam.accounts


def test_corrupt_institution_split_is_rejected_not_rescaled():
    """Review P1 (round 14, 2026-07-28): a corrupt institution split (totalling far from the build's
    final demand) is REJECTED, not silently rescaled into a plausible SAM. The review supplied a 0.4
    split against 200 total FD; the builder amplified it 500× into household+government demand of
    ~100 each with a passing quality report. Now both the IOSystem contract and the builder reject a
    large split↔FD discrepancy."""
    import pandas as pd

    from cge.contracts.data_objects import Classification, IOSystem, Provenance
    from cge.data.sam.build import build_raw_sam

    prov = Provenance(
        source="x", source_version="1", licence="x", reference_year=2020, retrieved="2026-07-28"
    )
    labels = ["R:a", "R:b"]
    A = pd.DataFrame([[0.1, 0.2], [0.05, 0.1]], index=labels, columns=labels)
    fd = pd.DataFrame({"R": [120.0, 80.0]}, index=labels)  # total final demand 200
    sep = "|"
    fbi = pd.DataFrame(  # totals 0.4 — three orders of magnitude too small
        {
            f"R{sep}household": [0.2, 0.1],
            f"R{sep}government": [0.05, 0.02],
            f"R{sep}investment": [0.02, 0.01],
        },
        index=labels,
    )
    io = IOSystem(
        provenance=prov,
        sectors=Classification(name="s", kind="sector", labels=["a", "b"]),
        regions=Classification(name="r", kind="region", labels=["R"]),
        A=A,
        final_demand=fd,
        final_demand_kind="by_region",
        final_demand_by_institution=fbi,
    )
    with pytest.raises(ValueError, match="does not decompose|split does not|rel gap"):
        build_raw_sam(io)


def _one_region_io_with_split(fd_sectors, split_by_inst):
    """A one-region two-sector IOSystem carrying a by-institution split, for the per-sector
    consistency tests. ``split_by_inst`` maps institution → [sector_a, sector_b]."""
    import pandas as pd

    from cge.contracts.data_objects import Classification, IOSystem, Provenance

    prov = Provenance(
        source="x", source_version="1", licence="x", reference_year=2020, retrieved="2026-07-28"
    )
    labels = ["R:a", "R:b"]
    A = pd.DataFrame([[0.1, 0.2], [0.05, 0.1]], index=labels, columns=labels)
    fd = pd.DataFrame({"R": list(fd_sectors)}, index=labels)
    sep = "|"
    fbi = pd.DataFrame({f"R{sep}{inst}": vec for inst, vec in split_by_inst.items()}, index=labels)
    return IOSystem(
        provenance=prov,
        sectors=Classification(name="s", kind="sector", labels=["a", "b"]),
        regions=Classification(name="r", kind="region", labels=["R"]),
        A=A,
        final_demand=fd,
        final_demand_kind="by_region",
        final_demand_by_institution=fbi,
    )


def test_institution_split_with_matching_region_total_but_wrong_sector_shape_is_rejected():
    """Review P1 (round 15, 2026-07-28): a split whose REGION total matches final demand but whose
    SECTOR distribution is wildly off is REJECTED — an aggregate/region-total check alone passed it
    and the builder then silently rewrote a sector by ~11,900%. Reviewer's exact case: split totals
    [1, 199] against a final demand of [120, 80] (both sum to 200). Both the IOSystem contract
    (construction time) and the builder must reject it per-sector."""
    from cge.contracts.data_objects import IOSystem
    from cge.data.sam.build import build_raw_sam

    split = {"household": [1.0, 199.0], "government": [0.0, 0.0], "investment": [0.0, 0.0]}
    # The contract rejects at construction (sector 'a' split 1 exceeds nothing, but sector 'b'
    # split 199 EXCEEDS its final demand 80).
    with pytest.raises(ValueError, match="SECTOR-BY-SECTOR|per-sector|EXCEEDING"):
        _one_region_io_with_split([120.0, 80.0], split)

    # And even if a caller bypasses the contract (an AGGREGATE-final-demand build, which the
    # contract's per-sector check does not cover because there is no per-region column to compare),
    # the builder's per-sector reconciliation still rejects it.
    import pandas as pd

    from cge.contracts.data_objects import Classification, Provenance

    prov = Provenance(
        source="x", source_version="1", licence="x", reference_year=2020, retrieved="2026-07-28"
    )
    labels = ["R:a", "R:b"]
    io = IOSystem(
        provenance=prov,
        sectors=Classification(name="s", kind="sector", labels=["a", "b"]),
        regions=Classification(name="r", kind="region", labels=["R"]),
        A=pd.DataFrame([[0.1, 0.2], [0.05, 0.1]], index=labels, columns=labels),
        final_demand=pd.DataFrame({"total": [120.0, 80.0]}, index=labels),  # single aggregate col
        final_demand_kind="aggregate",
        final_demand_by_institution=pd.DataFrame(
            {"R|household": [1.0, 199.0], "R|government": [0.0, 0.0], "R|investment": [0.0, 0.0]},
            index=labels,
        ),
    )
    with pytest.raises(ValueError, match="SECTOR-BY-SECTOR|does not decompose|per-sector|rel gap"):
        build_raw_sam(io)


def test_institution_split_consistent_per_sector_still_builds():
    """The tightened per-sector guard does NOT reject a legitimate split that decomposes final
    demand sector-by-sector (regression guard against over-rejection)."""
    split = {
        "household": [80.0, 50.0],
        "government": [25.0, 20.0],
        "investment": [15.0, 10.0],
    }  # per sector: a=120, b=80 — exactly final demand
    io = _one_region_io_with_split([120.0, 80.0], split)
    sam, report, _sectors = build_sam(io)
    assert "GOV" in sam.accounts and "SAVINV" in sam.accounts
    assert report.passed


def _two_region_io_with_split(fd_by_region, split):
    """A 2-region 1-sector IOSystem. ``fd_by_region`` maps consuming region → [producerA, producerB]
    final demand; ``split`` maps '<consuming_region>|<institution>' → [producerA, producerB]."""
    import pandas as pd

    from cge.contracts.data_objects import Classification, IOSystem, Provenance

    prov = Provenance(
        source="x", source_version="1", licence="x", reference_year=2020, retrieved="2026-07-30"
    )
    labels = ["A:g", "B:g"]
    A = pd.DataFrame([[0.1, 0.05], [0.05, 0.1]], index=labels, columns=labels)
    fd = pd.DataFrame({d: v for d, v in fd_by_region.items()}, index=labels)
    fbi = pd.DataFrame({col: v for col, v in split.items()}, index=labels)
    return IOSystem(
        provenance=prov,
        sectors=Classification(name="s", kind="sector", labels=["g"]),
        regions=Classification(name="r", kind="region", labels=["A", "B"]),
        A=A,
        final_demand=fd,
        final_demand_kind="by_region",
        final_demand_by_institution=fbi,
    )


def test_offsetting_cross_region_institution_corruption_is_rejected():
    """Review P1 (round 16, 2026-07-30): the institution validator must check EVERY (product,
    consuming-region) cell, not just the domestic diagonal. final_demand columns are the CONSUMING
    region; rows are the PRODUCING label. The earlier validator derived the region from the
    producing label and only checked that column, so an imported-final-demand cell (producer A →
    consumer B) was never validated. Reviewer's exact case: cross-region final demand 20 and 10, an
    institution split of 30 and 0 for those cells, with matching domestic cells and matching
    sector/global totals — accepted, and 30 was misattributed to government. It must now be rejected
    (the B-consumed split total 30 exceeds B's final demand of that product)."""
    sep = "|"
    # Consuming-region final demand: A consumes [A:g=100, B:g=20]; B consumes [A:g=10, B:g=90].
    fd_by_region = {"A": [100.0, 20.0], "B": [10.0, 90.0]}
    # Corrupt split: for product A:g consumed by B, government=30 (> the 10 final demand there),
    # offset so the producing-region-A domestic column and the sector/global totals still match.
    split = {
        f"A{sep}household": [100.0, 0.0],
        f"A{sep}government": [0.0, 0.0],
        f"A{sep}investment": [0.0, 0.0],
        f"B{sep}household": [0.0, 90.0],  # product A:g consumed by B → household 0
        f"B{sep}government": [30.0, 0.0],  # product A:g consumed by B → gov 30 (> fd 10) — corrupt
        f"B{sep}investment": [0.0, 0.0],
    }
    with pytest.raises(ValueError, match="consumed by region|EXCEEDING|round 16"):
        _two_region_io_with_split(fd_by_region, split)


def test_consistent_cross_region_institution_split_is_accepted():
    """Regression guard against over-rejection: a split that decomposes final demand in EVERY
    (product, consuming-region) cell is accepted."""
    sep = "|"
    fd_by_region = {"A": [100.0, 20.0], "B": [10.0, 90.0]}
    split = {
        f"A{sep}household": [70.0, 15.0],
        f"A{sep}government": [20.0, 3.0],
        f"A{sep}investment": [10.0, 2.0],  # A-consumed: A:g=100, B:g=20 ✓
        f"B{sep}household": [7.0, 60.0],
        f"B{sep}government": [2.0, 20.0],
        f"B{sep}investment": [1.0, 10.0],  # B-consumed: A:g=10, B:g=90 ✓
    }
    io = _two_region_io_with_split(fd_by_region, split)  # constructs without error
    assert io.fd_by_institution() is not None


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


def _tiny_route_io(cross_ab, cross_ba=None):
    """A 2-region EUR IOSystem with controllable (possibly ASYMMETRIC) tiny cross-region flows, to
    exercise the dust-rejection path (review P1 round 13, 2026-07-28)."""
    import pandas as pd

    from cge.contracts.data_objects import Classification, IOSystem, Provenance

    cross_ba = cross_ab if cross_ba is None else cross_ba
    prov = Provenance(
        source="x", source_version="1", licence="x", reference_year=2020, retrieved="2026-07-28"
    )
    regions, sectors = ["A", "B"], ["g"]
    labels = [f"{r}:g" for r in regions]
    A = pd.DataFrame([[0.2, cross_ba], [cross_ab, 0.2]], index=labels, columns=labels)
    fd = pd.DataFrame({"A": [80.0, cross_ba], "B": [cross_ab, 90.0]}, index=labels)
    return IOSystem(
        provenance=prov,
        sectors=Classification(name="s", kind="sector", labels=sectors),
        regions=Classification(name="r", kind="region", labels=regions),
        A=A,
        final_demand=fd,
        unit="MEUR",
        currency="EUR",
        final_demand_kind="by_region",
    )


def test_build_multi_sam_rejects_asymmetric_dust_route():
    """Review P1 (round 13, 2026-07-28): an ASYMMETRIC dust route is REJECTED with a clear domain
    error — NOT zeroed-and-RAS-rebalanced (that transformation is not balance-preserving for
    asymmetric dust, and the earlier version made RAS fail to converge). This is the exact failure
    mechanism the previous symmetric test did not cover. The error advice points to
    aggregate_dust_regions or LOWERING materiality (not raising it — classifies MORE as dust)."""
    from cge.data.sam import build_multi_sam

    with pytest.raises(ValueError, match="dust trade route|LOWER"):
        build_multi_sam(_tiny_route_io(3e-6, cross_ba=1e-6))


def _dust_region_io(regions, A_data, fd_data, sectors=("g",)):
    """A small IOSystem with the given A and by-region final demand, for the dust-region aggregation
    workflow tests."""
    import pandas as pd

    from cge.contracts.data_objects import Classification, IOSystem, Provenance

    labels = [f"{r}:{s}" for r in regions for s in sectors]
    prov = Provenance(
        source="x", source_version="1", licence="x", reference_year=2020, retrieved="2026-07-28"
    )
    return IOSystem(
        provenance=prov,
        sectors=Classification(name="s", kind="sector", labels=list(sectors)),
        regions=Classification(name="r", kind="region", labels=list(regions)),
        A=pd.DataFrame(A_data, index=labels, columns=labels),
        final_demand=pd.DataFrame(fd_data, index=labels),
        unit="MEUR",
        currency="EUR",
        final_demand_kind="by_region",
    )


def test_aggregate_dust_regions_folds_low_trade_pairs():
    """Review P1 (round 14, 2026-07-28): the explicit upstream workflow — aggregate_dust_regions
    folds a region that only DUST-trades with one partner into it, producing a coarser build with NO
    dust that build_multi_sam accepts. Region C dust-links only to A, so it merges into {A+C},
    giving a clean 2-region {A+C, B} SAM."""
    from cge.data.sam import build_multi_sam
    from cge.data.sam.build import aggregate_dust_regions

    regions = ["A", "B", "C"]
    labels = ["A:g", "B:g", "C:g"]
    A = {c: {r: 0.0 for r in labels} for c in labels}
    for r in labels:
        A[r][r] = 0.2
    A["B:g"]["A:g"] = 0.06  # A buys from B (genuine)
    A["A:g"]["B:g"] = 0.06  # B buys from A (genuine)
    A["A:g"]["C:g"] = 1e-6  # C dust-buys from A
    A["C:g"]["A:g"] = 1e-6  # A dust-buys from C
    fd = {"A": [100.0, 5.0, 1e-4], "B": [5.0, 100.0, 0.0], "C": [1e-4, 0.0, 80.0]}
    io = _dust_region_io(regions, {k: [A[k][r] for r in labels] for k in labels}, fd)
    coarse, _sats, grouping = aggregate_dust_regions(io, [])
    assert grouping["C"] == grouping["A"] and grouping["B"] != grouping["A"]  # C folded into A
    sam, report, regs, _secs = build_multi_sam(coarse)  # no dust remains → builds
    assert report.passed and len(regs) == 2


def test_aggregate_dust_regions_merges_flows_not_original_routes():
    """Review P1 (round 15, 2026-07-28): dust must be judged on the SUMMED flow between MERGED
    groups at ``build_multi_sam``'s granularity, NOT on the original constituent routes. Three
    regions A, B, C: A↔B is genuine; C has one dust route to A (A→C sub-threshold) so C folds into
    A, but C's OTHER flows (C→A, C→B) are genuine. After the merge, {A+C}↔B is the SUM of A's and
    C's flows to/from B and clears the threshold — so a VALID 2-group build survives. The earlier
    code re-tested the original A→C constituent route after the merge and kept collapsing, wrongly
    rejecting this valid coarse multi-region system."""
    from cge.data.sam import build_multi_sam
    from cge.data.sam.build import aggregate_dust_regions

    mat = 0.01  # threshold ≈ 2.32; A→C and B→C (≈1.63) are dust, C→A and C→B (≈5.07) are genuine
    regions = ["A", "B", "C"]
    labels = ["A:g", "B:g", "C:g"]
    A = {c: {r: 0.0 for r in labels} for c in labels}
    for r in labels:
        A[r][r] = 0.2
    A["B:g"]["A:g"] = 0.10  # A buys from B (genuine, large)
    A["A:g"]["B:g"] = 0.10  # B buys from A (genuine, large)
    A["A:g"]["C:g"] = 0.03  # C buys from A
    A["C:g"]["A:g"] = 0.03  # A buys from C
    A["B:g"]["C:g"] = 0.03  # C buys from B
    A["C:g"]["B:g"] = 0.03  # B buys from C
    fd = {"A": [100.0, 5.0, 0.5], "B": [5.0, 100.0, 0.5], "C": [0.5, 0.5, 20.0]}
    io = _dust_region_io(regions, {k: [A[k][r] for r in labels] for k in labels}, fd)

    coarse, _sats, grouping = aggregate_dust_regions(io, [], materiality=mat)
    n_groups = len(set(grouping.values()))
    assert n_groups == 2, f"expected a 2-group aggregation, got {grouping}"
    assert grouping["C"] == grouping["A"] and grouping["B"] != grouping["A"]  # C folded into A
    sam, report, regs, _secs = build_multi_sam(coarse, materiality=mat)  # summed flows clear it
    assert report.passed and len(regs) == 2


def test_aggregate_dust_regions_enforces_same_materiality_range_as_builder():
    """Review P2 (round 16, 2026-07-30): aggregate_dust_regions must enforce the SAME materiality
    range as build_multi_sam [1e-6, 0.1). Otherwise a value the helper accepts (e.g. 0) produces a
    build that build_multi_sam then rejects at the same setting — an inconsistent contract."""
    from cge.data.sam.build import aggregate_dust_regions

    io = _dust_region_io(
        ["A", "B"],
        {"A:g": [0.2, 0.05], "B:g": [0.05, 0.2]},
        {"A": [50.0, 5.0], "B": [5.0, 50.0]},
    )
    with pytest.raises(ValueError, match="materiality must be in"):
        aggregate_dust_regions(io, [], materiality=0.0)
    with pytest.raises(ValueError, match="materiality must be in"):
        aggregate_dust_regions(io, [], materiality=0.5)


def test_aggregate_dust_regions_reports_when_no_multi_structure_survives():
    """When a build's inter-region trade is ENTIRELY dust (the offline pymrio test MRIO's case),
    aggregation collapses to one region and raises an honest error rather than returning a
    degenerate single-region 'multi' SAM (review P1 round 14)."""
    from cge.data.adapters.exiobase import adapt_pymrio, load_exiobase_test
    from cge.data.sam.build import aggregate_dust_regions

    pio = load_exiobase_test()
    io, sats = adapt_pymrio(
        pio,
        source="EXIOBASE-test",
        source_version="test",
        reference_year=2011,
        gas_aliases={"emission_type1": "CO2"},
        currency="USD",
        monetary_unit="MUSD",
    )
    with pytest.raises(ValueError, match="collapsed to a single region|entirely dust"):
        aggregate_dust_regions(io, sats)


def test_aggregate_dust_regions_output_is_always_dust_free_or_honestly_collapses():
    """Property-style (review round 15): across a sweep of materiality thresholds on a random-ish
    4-region fixture, aggregate_dust_regions must ALWAYS either (a) return a coarse build that
    build_multi_sam accepts at the SAME threshold (no surviving dust — the invariant the workflow
    promises), or (b) raise the honest single-region collapse. It must NEVER return a coarse build
    that build_multi_sam then rejects for dust (that would be the round-14/15 bug class)."""
    import numpy as np

    from cge.data.sam import build_multi_sam
    from cge.data.sam.build import aggregate_dust_regions

    rng = np.random.default_rng(20260729)
    regions = ["A", "B", "C", "D"]
    labels = [f"{r}:g" for r in regions]
    for _ in range(8):
        A = {c: {r: 0.0 for r in labels} for c in labels}
        for r in labels:
            A[r][r] = 0.2
        # Random small-to-moderate cross-region coefficients (a mix of genuine and dust flows).
        for ri in labels:
            for rj in labels:
                if ri != rj:
                    A[ri][rj] = float(rng.choice([0.0, 0.005, 0.02, 0.08], p=[0.3, 0.3, 0.2, 0.2]))
        fd = {r: [40.0 if f"{r}:g" == lb else 0.5 for lb in labels] for r in regions}
        io = _dust_region_io(regions, {k: [A[k][r] for r in labels] for k in labels}, fd)
        for mat in (0.004, 0.008, 0.015, 0.03):
            try:
                coarse, _s, grouping = aggregate_dust_regions(io, [], materiality=mat)
            except ValueError as e:
                assert "collapsed to a single region" in str(e) or "entirely dust" in str(e)
                continue
            # The coarse build MUST pass build_multi_sam at the same threshold — no dust survived.
            sam, report, regs, _sec = build_multi_sam(coarse, materiality=mat)
            assert report.passed, f"mat={mat} grouping={grouping} produced a build that failed QA"
            assert len(regs) == len(set(grouping.values()))


def test_build_multi_sam_clean_build_still_works():
    """A build with NO dust (all cross flows either genuine trade or exactly zero) still builds,
    balances, and calibrates with every share aligned to a clearing route."""
    from cge.data.sam import build_multi_sam
    from cge.data.sam.balance import is_balanced
    from cge.engines.cge_static import model_multi as MM
    from cge.engines.cge_static.calibrate_multi import calibrate_multi

    # Genuine (above-threshold) two-way trade — no dust.
    sam, _report, regions, sectors = build_multi_sam(_tiny_route_io(0.05))
    assert is_balanced(sam.matrix, tol=1e-6)
    cal = calibrate_multi(sam, regions=regions, sectors=sectors, factors=["CAP", "LAB"])
    active = set(cal.active_routes)
    for oi in range(cal.nr):
        for di in range(cal.nr):
            if oi == di:
                continue
            for si in range(cal.ns):
                assert not (cal.arm_share_m[di, si, oi] > 0.0 and (oi, si, di) not in active)
                assert not (cal.cet_share_e[oi, si, di] > 0.0 and (oi, si, di) not in active)
    sol = solve(lambda z: MM.residuals(cal, z), MM.initial_guess(cal) * 1.02, prefer="scipy")
    st = MM.unpack_state(cal, sol.x, strict=True)
    worst = max(
        (abs(float(st.M[di, si, oi] - st.EX[oi, si, di])) for (oi, si, di) in cal.active_routes),
        default=0.0,
    )
    assert worst < 1e-8


def test_build_multi_sam_materiality_bounds_validated():
    """The user-facing ``materiality`` has a bounded, documented contract: at least the calibrator's
    threshold (so the built SAM passes the calibrator's dust gate) and below 0.1 of GDP (so real
    trade is not erased). Out-of-range values are rejected (review P1, 2026-07-27)."""
    from cge.data.sam import build_multi_sam
    from cge.engines.cge_static.calibrate_multi import ROUTE_MATERIALITY_THRESHOLD

    io = _tiny_route_io(1e-3)
    with pytest.raises(ValueError, match="materiality must be in"):
        build_multi_sam(io, materiality=ROUTE_MATERIALITY_THRESHOLD / 10)  # too small
    with pytest.raises(ValueError, match="materiality must be in"):
        build_multi_sam(io, materiality=0.5)  # too large — would erase real trade


def test_supplied_sam_with_dust_route_is_rejected():
    """A SUPPLIED (not built) multi-region SAM carrying a sub-threshold dust route is rejected at
    calibration with a clear message — not silently calibrated into a 35%-imbalance equilibrium the
    solver accepts at machine-zero residual (review P1, 2026-07-27). Built by hand with one genuine
    route and one dust route so the dust is unambiguous and the SAM is otherwise balanced."""
    import pandas as pd

    from cge.contracts.data_objects import SAM, Provenance
    from cge.engines.cge_static.calibrate_multi import calibrate_multi

    regions, sectors = ["N", "S"], ["BRD", "MIL"]
    accounts = []
    for r in regions:
        accounts += [f"a_{r}_{s}" for s in sectors] + [f"c_{r}_{s}" for s in sectors]
    accounts += [f"{f}_{r}" for r in regions for f in ("CAP", "LAB")] + [
        f"HOH_{r}" for r in regions
    ]
    m = pd.DataFrame(0.0, index=accounts, columns=accounts)
    for r in regions:
        for s in sectors:
            m.loc[f"a_{r}_{s}", f"c_{r}_{s}"] = 100.0
    # A genuine, balanced BRD route N<->S (keeps the graph connected) and a DUST MIL route N->S.
    m.loc["a_N_BRD", "c_S_BRD"] = 15.0
    m.loc["a_S_BRD", "c_N_BRD"] = 15.0
    m.loc["a_N_MIL", "c_S_MIL"] = 1e-7  # dust: ~2.9e-10 of the ~340 GDP
    for r in regions:
        for s in sectors:
            output = float(m.loc[f"a_{r}_{s}", :].sum())
            m.loc[f"CAP_{r}", f"a_{r}_{s}"] = output / 2.0
            m.loc[f"LAB_{r}", f"a_{r}_{s}"] = output / 2.0
    for r in regions:
        for s in sectors:
            com = f"c_{r}_{s}"
            m.loc[com, f"HOH_{r}"] = float(m[com].sum()) - float(m.loc[com].sum())
    for r in regions:
        m.loc[f"HOH_{r}", f"CAP_{r}"] = float(m.loc[f"CAP_{r}", :].sum())
        m.loc[f"HOH_{r}", f"LAB_{r}"] = float(m.loc[f"LAB_{r}", :].sum())
    prov = Provenance(
        source="t", source_version="1", licence="x", reference_year=0, retrieved="2026-07-27"
    )
    sam = SAM(provenance=prov, accounts=accounts, matrix=m)
    with pytest.raises(ValueError, match="dust trade route"):
        calibrate_multi(sam, regions=regions, sectors=sectors, factors=["CAP", "LAB"])


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
    d = res.data
    # Every CHANGE variable is ~0 at a zero carbon price (benchmark replication). LEVEL outputs
    # (gov_spending / investment / savings — GDP shares that exist now the IO build carries GOV /
    # SAVINV, review P1 round 13) are non-zero benchmark levels, not changes — exclude them.
    levels = {"gov_spending", "investment", "savings", "fiscal_balance", "carbon_revenue"}
    changes = d[~d["variable"].isin(levels)]
    assert changes["value"].abs().max() < 1e-7
