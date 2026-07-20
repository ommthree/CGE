"""Phase 1 tests: the data layer builds, aggregates, quality-checks and stores offline.

All tests use pymrio's bundled test MRIO (no download), so they run in CI. The live
EXIOBASE path shares the same adapter/aggregate/quality/store code, so exercising it here
covers the pipeline logic; only the download itself is untested offline (by design).
"""

import json

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


def test_store_roundtrip_preserves_final_demand_kind(tmp_path):
    """P2 (review round 9): final_demand_kind is an explicit discriminator, not re-derived from
    the parquet's columns on load — a round-tripped build must come back labelled the same way it
    went in (by_region for a real adapter build, aggregate for the small aggregated derivative if
    its final demand collapses to one column)."""
    store = DataStore(tmp_path)
    written = build_test(store=store)
    full = store.load(written["full"])["IOSystem"]
    assert full.final_demand_kind == "by_region"
    assert full.fd_by_region() is not None
    small = store.load(written["small"])["IOSystem"]
    # The aggregation preserves the by-region split when the source had it.
    assert small.final_demand_kind == "by_region"
    assert small.fd_by_region() is not None


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


def test_store_save_preserves_build_on_swap_failure(tmp_path):
    """If the staging→final swap fails, the pre-existing build must be restored, not lost
    (review: the old dir was deleted before the rename)."""
    import pathlib
    from unittest import mock

    store = DataStore(tmp_path)
    build_test(store=store)
    bid = "exiobase-test"
    before = sorted(p.name for p in (store.builds_dir / bid).iterdir())

    pio = load_exiobase_test()
    io, sats = adapt_pymrio(
        pio,
        source="EXIOBASE-test",
        source_version="test",
        reference_year=2011,
        gas_aliases={"emission_type1": "CO2"},
    )
    meta = store.load_meta(bid)

    orig = pathlib.Path.replace

    def flaky(self, target):
        if ".tmp" in str(self) and str(target).endswith("/" + bid):
            raise OSError("simulated staging→final failure")
        return orig(self, target)

    with mock.patch.object(pathlib.Path, "replace", flaky), pytest.raises(OSError):
        store.save(meta=meta, io=io, satellites=sats)

    assert (store.builds_dir / bid).exists()
    after = sorted(p.name for p in (store.builds_dir / bid).iterdir())
    assert after == before  # prior build intact


def test_store_recovers_build_from_backup_on_open(tmp_path):
    """A hard-kill mid-swap leaves data in a .bak with the canonical path absent AND a stale
    lock (dead writer). Opening the store must recover it — but only because the lock is
    stale, not for a live writer."""
    import shutil

    store = DataStore(tmp_path)
    build_test(store=store)
    bid = "exiobase-test"
    final = store.builds_dir / bid
    # Simulate crash after old→bak, before staging→final, with a STALE lock (pid never runs).
    bak = store.builds_dir / f".{bid}.bak"
    shutil.move(str(final), str(bak))
    (store.builds_dir / f".{bid}.lock").write_text("999999")  # a pid that isn't alive
    assert not final.exists()

    store2 = DataStore(tmp_path)  # recovery on open
    assert store2.has(bid)
    assert not (store.builds_dir / f".{bid}.lock").exists()  # stale lock cleaned


def _concurrent_save_worker(root, bid):  # module-level so multiprocessing 'spawn' can import
    from cge.data.adapters.exiobase import adapt_pymrio, load_exiobase_test
    from cge.data.metadata import BuildMeta
    from cge.data.store import DataStore

    store = DataStore(root)
    io, sats = adapt_pymrio(
        load_exiobase_test(),
        source="E",
        source_version="v",
        reference_year=2011,
        gas_aliases={"emission_type1": "CO2"},
    )
    meta = BuildMeta(
        build_id=bid,
        source="E",
        source_version="v",
        reference_year=2011,
        licence="x",
        currency="USD",
        monetary_unit="MUSD",
        retrieved="2026-07-18",
    )
    store.save(meta=meta, io=io, satellites=sats)


def test_concurrent_catalogue_writes_all_land(tmp_path):
    """Four processes saving different builds must all end up in the catalogue — DuckDB's
    cross-process file lock is handled by the global lock + retry (review)."""
    import multiprocessing as mp

    DataStore(tmp_path)  # init the catalogue table first
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_concurrent_save_worker, args=(str(tmp_path), f"build{i}"))
        for i in range(4)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    cat = DataStore(tmp_path).catalogue()
    assert set(cat["build_id"]) == {f"build{i}" for i in range(4)}


def test_catalogue_failure_rolls_back_filesystem(tmp_path):
    """If the catalogue update fails after the filesystem swap, files roll back to the prior
    build so files and catalogue never diverge (review)."""
    from unittest import mock

    import cge.data.store.store as store_mod

    store = DataStore(tmp_path)
    build_test(store=store)
    bid = "exiobase-test"
    v1_labels = store.load(bid)["IOSystem"].A.shape[0]

    io, sats = adapt_pymrio(
        load_exiobase_test(),
        source="EXIOBASE-test",
        source_version="test-v2",
        reference_year=2011,
        gas_aliases={"emission_type1": "CO2"},
        currency="USD",
        monetary_unit="MUSD",
    )
    meta = store.load_meta(bid).model_copy(update={"source_version": "test-v2"})

    with (
        mock.patch.object(store, "_catalogue_upsert", side_effect=RuntimeError("catalogue down")),
        pytest.raises(RuntimeError, match="catalogue down"),
    ):
        store.save(meta=meta, io=io, satellites=sats)

    # The prior build (v1) must still be present and loadable — not left at v2.
    assert store.has(bid)
    assert store.load(bid)["IOSystem"].A.shape[0] == v1_labels
    assert store.load_meta(bid).source_version == "test"  # rolled back
    _ = store_mod  # ensure import used


def test_legacy_catalogue_migrates_and_keeps_legacy_build(tmp_path):
    """A pre-generation catalogue (no `generation` column) with a stale lock on a legacy build
    must: (1) migrate the schema before recovery reads it (no BinderException); (2) NOT delete
    the legacy build whose catalogue+meta both have NULL generation (review P1a)."""
    import duckdb

    store = DataStore(tmp_path)
    build_test(store=store)
    bid = "exiobase-test"

    # Downgrade the catalogue to the legacy 8-column schema (no generation).
    con = duckdb.connect(str(store.catalogue_path))
    con.execute(
        "CREATE TABLE builds_old AS SELECT build_id, source, source_version, reference_year, "
        "aggregation, n_labels, quality_worst, retrieved FROM builds"
    )
    con.execute("DROP TABLE builds")
    con.execute("ALTER TABLE builds_old RENAME TO builds")
    con.close()
    # Make the build legacy (strip its generation) and leave a stale lock.
    mp = store.builds_dir / bid / "meta.json"
    m = json.loads(mp.read_text())
    m.pop("generation", None)
    mp.write_text(json.dumps(m))
    (store.builds_dir / f".{bid}.lock").write_text("999999")

    store2 = DataStore(tmp_path)  # migrate + recover on open
    assert store2.has(bid)  # legacy build survived (matching NULL generations = committed)


def test_legacy_marker_crash_restores_backup_not_uncommitted_final(tmp_path):
    """The exact predecessor crash state (review P1): a store written by the OLD marker-based
    implementation, hard-killed between the swap and the catalogue commit. On disk: catalogue at
    v1 (legacy NULL generation), '.bak' at v1, 'final' at an uncommitted v2 carrying the legacy
    '.uncommitted' marker and a NULL generation, plus a stale lock. Recovery must honour the
    marker: discard the uncommitted v2 and restore the committed v1 backup — never keep v2 and
    delete the backup."""
    import shutil

    import duckdb

    store = DataStore(tmp_path)
    build_test(store=store)
    bid = "exiobase-test"
    final = store.builds_dir / bid
    bak = store.builds_dir / f".{bid}.bak"

    # Downgrade catalogue to legacy schema (NULL generation) — the migration will re-add it.
    con = duckdb.connect(str(store.catalogue_path))
    con.execute(
        "CREATE TABLE builds_old AS SELECT build_id, source, source_version, reference_year, "
        "aggregation, n_labels, quality_worst, retrieved FROM builds"
    )
    con.execute("DROP TABLE builds")
    con.execute("ALTER TABLE builds_old RENAME TO builds")
    con.close()

    # v1 committed files → .bak (legacy: strip generation).
    shutil.copytree(final, bak)
    mb = json.loads((bak / "meta.json").read_text())
    mb.pop("generation", None)
    mb["source_version"] = "v1-committed"
    (bak / "meta.json").write_text(json.dumps(mb))

    # 'final' = uncommitted v2 with the legacy .uncommitted marker and NULL generation.
    mf = json.loads((final / "meta.json").read_text())
    mf.pop("generation", None)
    mf["source_version"] = "v2-uncommitted"
    (final / "meta.json").write_text(json.dumps(mf))
    (final / ".uncommitted").write_text("")  # the predecessor's mid-write commit marker
    (store.builds_dir / f".{bid}.lock").write_text("999999")  # stale lock

    recovered = DataStore(tmp_path).load_meta(bid)
    assert recovered.source_version == "v1-committed"  # backup restored, not the uncommitted v2
    assert not bak.exists()  # backup consumed by the restore


def test_recovery_treats_corrupt_final_metadata_as_uncommitted(tmp_path):
    """Corrupt/unreadable final metadata must NOT be mistaken for a legacy NULL-generation
    committed build (review P1: _read_generation returned None for unreadable meta, which then
    matched a legacy NULL catalogue row). With a valid backup present, recovery restores it."""
    import shutil

    store = DataStore(tmp_path)
    build_test(store=store)
    bid = "exiobase-test"
    final = store.builds_dir / bid
    bak = store.builds_dir / f".{bid}.bak"

    shutil.copytree(final, bak)
    mb = json.loads((bak / "meta.json").read_text())
    mb["source_version"] = "v1-committed"
    (bak / "meta.json").write_text(json.dumps(mb))

    (final / "meta.json").write_text("{ this is not valid json ")  # corrupt final meta
    (store.builds_dir / f".{bid}.lock").write_text("999999")

    recovered = DataStore(tmp_path).load_meta(bid)
    assert recovered.source_version == "v1-committed"  # corrupt final rejected, backup restored


def test_recovery_discards_uncommitted_final_restores_backup(tmp_path):
    """Crash BEFORE the catalogue commit: 'final' holds files whose generation does NOT match
    the catalogue → recovery must discard them and restore the backup (review P1)."""
    import shutil

    store = DataStore(tmp_path)
    build_test(store=store)
    bid = "exiobase-test"
    committed_gen = store._committed_generation(bid)  # the v1 generation in the catalogue
    final = store.builds_dir / bid
    bak = store.builds_dir / f".{bid}.bak"

    # v1 → .bak; 'final' holds pretend-v2 with a DIFFERENT generation the catalogue never saw.
    shutil.copytree(final, bak)
    m = json.loads((final / "meta.json").read_text())
    m["generation"] = "uncommitted-v2-gen"
    m["source_version"] = "v2-uncommitted"
    (final / "meta.json").write_text(json.dumps(m))
    (store.builds_dir / f".{bid}.lock").write_text("999999")  # stale lock (dead pid)
    assert m["generation"] != committed_gen

    recovered = DataStore(tmp_path).load_meta(bid)
    assert recovered.generation == committed_gen  # backup (committed v1) restored
    assert recovered.source_version != "v2-uncommitted"
    assert not bak.exists()


def test_recovery_keeps_committed_final_after_post_commit_crash(tmp_path):
    """Crash AFTER the catalogue commit but before the backup was dropped: 'final's generation
    MATCHES the catalogue, so recovery must KEEP it and drop the stale backup — NOT roll back
    to v1 (review P1: the inverse post-commit window)."""
    import json as _json
    import shutil

    store = DataStore(tmp_path)
    build_test(store=store)
    bid = "exiobase-test"
    final = store.builds_dir / bid
    bak = store.builds_dir / f".{bid}.bak"

    # The current 'final' is committed (its generation == catalogue's). Simulate a leftover
    # backup from a prior generation, plus a stale lock — as if the process died right after
    # the catalogue commit but before dropping the backup.
    committed_gen = store._committed_generation(bid)
    readable, final_gen = store._read_generation(final)
    assert readable and final_gen == committed_gen
    shutil.copytree(final, bak)  # a stale backup (older files) left around
    old = _json.loads((bak / "meta.json").read_text())
    old["source_version"] = "v1-old"
    (bak / "meta.json").write_text(_json.dumps(old))
    (store.builds_dir / f".{bid}.lock").write_text("999999")  # stale lock

    recovered = DataStore(tmp_path).load_meta(bid)
    assert recovered.generation == committed_gen  # committed files KEPT, not rolled back
    assert recovered.source_version != "v1-old"
    assert not bak.exists()  # stale backup dropped


def test_concordance_change_yields_distinct_build_id(tmp_path):
    """A different named concordance must produce a different small-build id + aggregation label,
    so old and new builds are distinguishable in manifests (review)."""
    from cge.data.build import build_from_pymrio

    def _build(cid):
        store = DataStore(tmp_path / cid)
        sectors = list(load_exiobase_test().get_sectors())
        regions = list(load_exiobase_test().get_regions())
        return build_from_pymrio(
            load_exiobase_test(),
            source="E",
            source_version="v",
            reference_year=2011,
            build_id="b",
            store=store,
            make_small=True,
            small_sector_map={s: "x" for s in sectors},
            small_region_map={r: "y" for r in regions},
            concordance_id=cid,
            gas_aliases={"emission_type1": "CO2"},
            currency="USD",
            monetary_unit="MUSD",
        )["small"]

    id_a = _build("conc-a")
    id_b = _build("conc-b")
    assert id_a != id_b


def test_custom_maps_with_different_contents_yield_distinct_ids(tmp_path):
    """Even under the DEFAULT concordance_id ('custom'), two different maps must produce
    different build ids — a caller changing a custom map must not silently overwrite a
    numerically different build under the same id (review P2)."""
    from cge.data.build import build_from_pymrio

    def _build(subdir, sector_group):
        store = DataStore(tmp_path / subdir)
        sectors = list(load_exiobase_test().get_sectors())
        regions = list(load_exiobase_test().get_regions())
        return build_from_pymrio(
            load_exiobase_test(),
            source="E",
            source_version="v",
            reference_year=2011,
            build_id="b",
            store=store,
            make_small=True,
            small_sector_map={s: sector_group(i) for i, s in enumerate(sectors)},
            small_region_map={r: "y" for r in regions},
            gas_aliases={"emission_type1": "CO2"},
            currency="USD",
            monetary_unit="MUSD",
        )["small"]  # default concordance_id="custom"

    id_1 = _build("m1", lambda i: ["a", "b"][i % 2])  # 2 sector groups
    id_2 = _build("m2", lambda i: ["a", "b", "c"][i % 3])  # 3 sector groups — different map
    assert id_1 != id_2


def test_recovery_runs_on_direct_load(tmp_path):
    """A stale-lock crash leaving the build in .bak must be recovered by a *direct* load()
    (not only by construction/has) — GUI frames() calls load() directly (review)."""
    import shutil

    store = DataStore(tmp_path)
    build_test(store=store)
    bid = "exiobase-test"
    final = store.builds_dir / bid
    shutil.move(str(final), str(store.builds_dir / f".{bid}.bak"))
    (store.builds_dir / f".{bid}.lock").write_text("999999")  # stale lock (dead pid)
    assert not final.exists()

    # Direct load on the SAME store instance must self-heal (recovery is not construction-only).
    data = store.load(bid)
    assert "IOSystem" in data


def test_build_ids_excludes_internal_dirs(tmp_path):
    """.tmp staging and .bak backup dirs must not appear as builds (review)."""
    store = DataStore(tmp_path)
    build_test(store=store)
    (store.builds_dir / ".exiobase-test.somehex.tmp").mkdir()
    (store.builds_dir / ".exiobase-test.bak").mkdir()
    ids = store.build_ids()
    assert all(not i.startswith(".") for i in ids)
    assert "exiobase-test" in ids


def test_store_lock_released_on_staging_failure(tmp_path):
    """A staging mkdir failure must release the writer lock, not leak it (review)."""
    import pathlib
    from unittest import mock

    store = DataStore(tmp_path)
    build_test(store=store)
    io, sats = adapt_pymrio(
        load_exiobase_test(),
        source="EXIOBASE-test",
        source_version="test",
        reference_year=2011,
        gas_aliases={"emission_type1": "CO2"},
    )
    meta = store.load_meta("exiobase-test")
    orig = pathlib.Path.mkdir

    def flaky(self, *a, **k):
        if ".tmp" in str(self):
            raise OSError("mkdir fail")
        return orig(self, *a, **k)

    with mock.patch.object(pathlib.Path, "mkdir", flaky), pytest.raises(OSError):
        store.save(meta=meta, io=io, satellites=sats)
    assert not (store.builds_dir / ".exiobase-test.lock").exists()  # lock released


def test_store_to_engine_seam_eur_build(tmp_path):
    """Exercise the store→engine seam with a EUR-relabelled build (the default USD test build
    is correctly refused by io_price; this keeps the seam covered — review)."""
    import cge.data.store.store as store_mod
    from cge.contracts.shocks import CarbonPrice
    from cge.data.build import build_from_pymrio
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    store = DataStore(tmp_path)
    written = build_from_pymrio(
        load_exiobase_test(),
        source="EXIOBASE-test-eur",
        source_version="test",
        reference_year=2011,
        build_id="eur-build",
        store=store,
        make_small=True,
        small_sector_map={
            s: ["p", "e", "m"][i % 3] for i, s in enumerate(load_exiobase_test().get_sectors())
        },
        small_region_map={
            r: ("A" if i < 3 else "B") for i, r in enumerate(load_exiobase_test().get_regions())
        },
        gas_aliases={"emission_type1": "CO2"},
        currency="EUR",
        monetary_unit="MEUR",  # EUR so io_price accepts it
    )
    monkey = pytest.MonkeyPatch()
    monkey.setattr(store_mod, "_default", store)
    try:
        sc = Scenario(
            name="seam", engine="io_price", years=[2020], shocks=[CarbonPrice(price=100.0)]
        )
        result = run_scenario(sc, data_source=written["small"], store=store)
        assert (result.data["variable"] == "price_change").any()
    finally:
        monkey.undo()


def test_store_recovery_leaves_live_writer_untouched(tmp_path):
    """Opening a second store must NOT delete a live writer's staging (review counterexample)."""
    import os

    store = DataStore(tmp_path)
    bid = "wip"
    lock = store.builds_dir / f".{bid}.lock"
    lock.write_text(str(os.getpid()))  # a LIVE writer (this process)
    staging = store.builds_dir / f".{bid}.deadbeef.tmp"
    staging.mkdir()
    (staging / "marker").write_text("live")

    DataStore(tmp_path)  # second store open
    assert staging.exists()  # live staging untouched
    lock.unlink()


def test_structural_gate_rejects_nan_final_demand():
    """A NaN in final demand must fail the structural gate (review: it previously passed all
    gates because NaN comparisons silently evaluate false)."""
    from cge.data.quality import ConsistencyError, assert_structural

    pio = load_exiobase_test()
    io, sats = adapt_pymrio(pio, source="t", source_version="test", reference_year=2011)
    bad = io.model_copy()
    fd = bad.final_demand.copy()
    fd.iloc[0, 0] = np.nan
    bad.final_demand = fd
    with pytest.raises(ConsistencyError, match="final_demand contains non-finite"):
        assert_structural(bad, sats)


def test_adapter_ghg_intensity_in_tonnes_per_meur():
    """Units fix: the GHG satellite is in t/MEUR (kg source unit converted to tonnes),
    and carries per-gas + CO2e rows with correct unit labels."""
    pio = load_exiobase_test()
    _, sats = adapt_pymrio(
        pio,
        source="t",
        source_version="test",
        reference_year=2011,
        gas_aliases={"emission_type1": "CO2"},
    )
    ghg = next(s for s in sats if s.name == "GHG")
    assert ghg.units["CO2"] == "t/MEUR"
    assert ghg.units["CO2e"] == "tCO2e/MEUR"
    # kg→t conversion makes intensities ~1000× smaller than the raw kg/MEUR numbers.
    assert ghg.data.loc["CO2"].abs().max() < 1.0  # test fixture is low-intensity


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
