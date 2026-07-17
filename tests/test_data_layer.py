"""Phase 1 tests: the data layer builds, aggregates, quality-checks and stores offline.

All tests use pymrio's bundled test MRIO (no download), so they run in CI. The live
EXIOBASE path shares the same adapter/aggregate/quality/store code, so exercising it here
covers the pipeline logic; only the download itself is untested offline (by design).
"""

import numpy as np
import pytest

from cge.contracts.data_objects import ConcordanceMap, Provenance
from cge.contracts.quality import Severity
from cge.data.adapters.exiobase import adapt_pymrio, load_exiobase_test
from cge.data.aggregate import aggregate_io
from cge.data.build import build_test
from cge.data.concordance.concordance import bridge_matrix, check_covers, one_to_one
from cge.data.metadata import BuildMeta
from cge.data.quality import build_quality_report, drift_report
from cge.data.store import DataStore


def _prov():
    return Provenance(
        source="t", source_version="1", licence="none", reference_year=2011, retrieved="2026-07-17"
    )


# -- adapter ------------------------------------------------------------------
def test_adapter_maps_pymrio_to_contracts():
    pio = load_exiobase_test()
    io, sats = adapt_pymrio(
        pio,
        source="t",
        source_version="test",
        reference_year=2011,
        gas_aliases={"emission_type1": "CO2"},
    )
    # labels are region:sector and square A
    assert io.A.shape[0] == io.A.shape[1]
    assert all(":" in c for c in io.A.columns)
    # GHG satellite built with a CO2e row
    assert any(s.name == "GHG" for s in sats)
    ghg = next(s for s in sats if s.name == "GHG")
    assert "CO2e" in ghg.data.index


# -- concordance --------------------------------------------------------------
def test_concordance_orphan_detection_and_bridge():
    cmap = one_to_one(
        {"a": "X", "b": "X", "c": "Y"},
        from_classification="src",
        to_classification="dst",
        provenance=_prov(),
    )
    assert check_covers(cmap, ["a", "b", "c"]) == []
    assert check_covers(cmap, ["a", "z"]) == ["z"]
    B = bridge_matrix(cmap, ["a", "b", "c"])
    assert B.shape == (2, 3)  # 2 targets × 3 sources
    assert B.loc["X", "a"] == 1.0 and B.loc["Y", "c"] == 1.0


def test_bridge_raises_on_orphan():
    cmap = one_to_one(
        {"a": "X"}, from_classification="s", to_classification="d", provenance=_prov()
    )
    with pytest.raises(ValueError, match="unmapped"):
        bridge_matrix(cmap, ["a", "b"])


def test_concordance_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        ConcordanceMap(
            provenance=_prov(),
            from_classification="s",
            to_classification="d",
            weights={"a": {"X": 0.7, "Y": 0.2}},  # 0.9
        )


# -- aggregation --------------------------------------------------------------
def test_aggregation_conserves_output_and_final_demand():
    """The key economic-correctness invariant: aggregating flows (not coefficients) leaves
    total gross output and total final demand unchanged."""
    pio = load_exiobase_test()
    io, sats = adapt_pymrio(
        pio,
        source="t",
        source_version="test",
        reference_year=2011,
        gas_aliases={"emission_type1": "CO2"},
    )
    sectors, regions = list(pio.get_sectors()), list(pio.get_regions())
    sec_map = {s: ["p", "e", "m"][i % 3] for i, s in enumerate(sectors)}
    reg_map = {r: ("A" if i < len(regions) // 2 else "B") for i, r in enumerate(regions)}
    scm = one_to_one(sec_map, from_classification="s", to_classification="ss", provenance=_prov())
    rcm = one_to_one(reg_map, from_classification="r", to_classification="rr", provenance=_prov())
    meta = BuildMeta(
        build_id="t",
        source="t",
        source_version="test",
        reference_year=2011,
        licence="x",
        retrieved="2026-07-17",
    )
    s_io, s_sats, _ = aggregate_io(
        io,
        sats,
        sector_cmap=scm,
        region_cmap=rcm,
        meta=meta,
        new_build_id="ts",
        aggregation_name="small",
    )

    def total_output(system):
        A = system.A.to_numpy(float)
        f = system.final_demand.sum(axis=1).to_numpy(float)
        return np.linalg.solve(np.eye(A.shape[0]) - A, f).sum()

    assert np.isclose(total_output(io), total_output(s_io), rtol=1e-6)
    assert np.isclose(io.final_demand.sum().sum(), s_io.final_demand.sum().sum(), rtol=1e-6)
    # aggregated system is smaller and still productive
    assert s_io.A.shape[0] < io.A.shape[0]
    assert float(np.max(np.abs(np.linalg.eigvals(s_io.A.to_numpy(float))))) < 1.0


# -- quality ------------------------------------------------------------------
def test_quality_report_flags_and_passes():
    pio = load_exiobase_test()
    io, sats = adapt_pymrio(
        pio,
        source="t",
        source_version="test",
        reference_year=2011,
        gas_aliases={"emission_type1": "CO2"},
    )
    report = build_quality_report("b", io, sats)
    names = {c.name for c in report.checks}
    assert "spectral_radius" in names
    assert report.passed  # test system is productive
    # spectral radius check passes
    rho = next(c for c in report.checks if c.name == "spectral_radius")
    assert rho.severity == Severity.PASS and rho.value < 1.0


def test_drift_report_flags_material_change():
    pio = load_exiobase_test()
    io, sats = adapt_pymrio(pio, source="t", source_version="test", reference_year=2011)
    r1 = build_quality_report("b1", io, sats)
    # perturb A to shift the spectral radius, rebuild
    io2 = io.model_copy()
    io2.A.iloc[:] = io.A.to_numpy(float) * 1.5
    r2 = build_quality_report("b2", io2, sats)
    drift = drift_report("b", r2, r1)
    sr_drift = next(c for c in drift.checks if c.name == "drift_spectral_radius")
    assert sr_drift.severity == Severity.WARN  # 50% change > 10% threshold


# -- store + end to end -------------------------------------------------------
def test_build_test_end_to_end(tmp_path):
    store = DataStore(tmp_path)
    written = build_test(store=store)
    assert "full" in written and "small" in written

    # catalogue lists both builds
    cat = store.catalogue()
    assert set(cat["build_id"]) == {written["full"], written["small"]}

    # round-trip: loaded objects match engine-expected keys
    data = store.load(written["full"])
    assert "IOSystem" in data and "SatelliteAccount" in data
    assert store.load_quality(written["full"]).passed

    # small build is genuinely smaller
    assert store.load(written["small"])["IOSystem"].A.shape[0] < data["IOSystem"].A.shape[0]


def test_structural_guard_rejects_broken_build():
    """The structural consistency gate rejects non-finite / non-productive systems."""
    from cge.data.quality import ConsistencyError, assert_structural

    pio = load_exiobase_test()
    io, sats = adapt_pymrio(pio, source="t", source_version="test", reference_year=2011)
    assert_structural(io, sats)  # valid build passes

    # NaN in A -> raise
    bad = io.model_copy()
    A = bad.A.copy()
    A.iloc[0, 0] = np.nan
    bad.A = A
    with pytest.raises(ConsistencyError, match="non-finite"):
        assert_structural(bad, sats)

    # non-productive (rho >= 1) -> raise
    bad2 = io.model_copy()
    bad2.A = io.A * 5.0
    with pytest.raises(ConsistencyError, match="Leontief inverse"):
        assert_structural(bad2, sats)


def test_aggregation_conservation_check_reports_pass():
    """The cross-stage conservation check passes for a correct aggregation and is stored
    in the build's quality report (so a broken aggregation would surface, not hide)."""
    from cge.data.quality import check_aggregation_conserves

    pio = load_exiobase_test()
    io, sats = adapt_pymrio(
        pio,
        source="t",
        source_version="test",
        reference_year=2011,
        gas_aliases={"emission_type1": "CO2"},
    )
    sectors, regions = list(pio.get_sectors()), list(pio.get_regions())
    scm = one_to_one(
        {s: ["p", "e", "m"][i % 3] for i, s in enumerate(sectors)},
        from_classification="s",
        to_classification="ss",
        provenance=_prov(),
    )
    rcm = one_to_one(
        {r: ("A" if i < len(regions) // 2 else "B") for i, r in enumerate(regions)},
        from_classification="r",
        to_classification="rr",
        provenance=_prov(),
    )
    meta = BuildMeta(
        build_id="t",
        source="t",
        source_version="test",
        reference_year=2011,
        licence="x",
        retrieved="2026-07-17",
    )
    s_io, _, _ = aggregate_io(
        io,
        sats,
        sector_cmap=scm,
        region_cmap=rcm,
        meta=meta,
        new_build_id="ts",
        aggregation_name="small",
    )
    report = check_aggregation_conserves(io, s_io)
    assert report.passed
    assert {c.name for c in report.checks} == {
        "aggregation_conserves_output",
        "aggregation_conserves_final_demand",
    }


def test_stored_small_build_carries_conservation_checks(tmp_path):
    """End to end: the pipeline folds cross-stage conservation checks into the stored
    small-build quality report."""
    store = DataStore(tmp_path)
    written = build_test(store=store)
    q = store.load_quality(written["small"])
    names = {c.name for c in q.checks}
    assert "aggregation_conserves_output" in names
    assert "aggregation_conserves_final_demand" in names
    assert q.passed


def test_runner_loads_from_store(tmp_path, monkeypatch):
    """The runner's load_data dispatches to the store for a build id (not just 'toy')."""
    import cge.data.store.store as store_mod
    from cge.runner import load_data

    store = DataStore(tmp_path)
    build_test(store=store)
    # point the default store at our tmp store
    monkeypatch.setattr(store_mod, "_default", store)

    data = load_data("exiobase-test")
    assert "IOSystem" in data
    with pytest.raises(ValueError, match="Unknown data source"):
        load_data("does-not-exist")
