"""The data store: persist and load data builds; catalogue them in DuckDB.

Layout on disk (under ``root``)::

    <root>/
      catalogue.duckdb            # index of builds (for the GUI/runner to enumerate)
      builds/<build_id>/
        meta.json                 # BuildMeta
        A.parquet                 # technical coefficients (index/cols = labels)
        final_demand.parquet
        value_added.parquet
        satellite_<name>.parquet  # one per SatelliteAccount
        satellite_<name>.units.json
        quality.json              # QualityReport (optional)

Numeric payloads are parquet (float32 to keep MRIO builds manageable, per roadmap P1
risks). Everything else is JSON so it stays diffable and human-readable.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

try:
    import fcntl  # POSIX advisory file locks
except ImportError:  # pragma: no cover - Windows
    fcntl = None

from cge.contracts.data_objects import (
    Classification,
    IOSystem,
    Provenance,
    SatelliteAccount,
)
from cge.contracts.quality import QualityReport
from cge.data.metadata import BuildMeta

DEFAULT_ROOT = Path("data_store")

# Sentinel written into a build dir while it is installed but not yet catalogue-committed.
# Its presence after a crash tells recovery the files are uncommitted (see _recover_interrupted).
UNCOMMITTED_MARKER = ".uncommitted"

# Catalogue write retry (DuckDB rejects concurrent cross-process write connections).
_CATALOGUE_RETRIES = 50
_CATALOGUE_RETRY_DELAY = 0.1  # seconds
_GLOBAL_LOCK_TIMEOUT = 30.0  # seconds to wait for the catalogue lock before giving up


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # signal 0: existence check, doesn't actually signal
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    return True


def _lock_is_live(lock: Path) -> bool:
    """A lock is live iff it holds the pid of a running process."""
    try:
        pid = int(lock.read_text().strip())
    except (OSError, ValueError):
        return False
    return _pid_alive(pid)


def _acquire_lock(lock: Path) -> None:
    """Create the per-build writer lock, or raise if a *live* writer already holds it. A stale
    lock (dead pid) is reclaimed. This serialises concurrent saves to the same build."""
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if _lock_is_live(lock):
            raise RuntimeError(
                f"build {lock.name[1:-5]!r} is being written by another process; "
                f"concurrent save refused."
            ) from None
        lock.unlink(missing_ok=True)  # stale lock from a dead writer; reclaim
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))


@contextmanager
def _global_lock(lock: Path, timeout: float = _GLOBAL_LOCK_TIMEOUT):
    """OS-backed exclusive cross-process lock via ``fcntl.flock`` (advisory).

    Unlike the PID-file per-build lock, this is **atomic** — there is no create-then-write
    window where another process mistakes an empty file for a stale lock (review). The OS
    releases the lock automatically if the holder dies, so no stale-reclamation is needed.
    Blocks (polling non-blocking flock) up to ``timeout`` so catalogue writers *serialise*
    rather than fail. On platforms without ``fcntl`` (Windows), falls back to no-op — the
    per-connection retry in ``_with_catalogue`` still handles DuckDB contention."""
    if fcntl is None:  # non-POSIX; rely on the DuckDB retry loop
        yield
        return
    fd = os.open(lock, os.O_CREAT | os.O_RDWR)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"timed out waiting for catalogue lock {lock}") from None
                time.sleep(0.02)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _meta_to_provenance(meta: BuildMeta) -> Provenance:
    return Provenance(
        source=meta.source,
        source_version=meta.source_version,
        licence=meta.licence,
        reference_year=meta.reference_year,
        retrieved=meta.retrieved,
        build_id=meta.build_id,
        aggregation=meta.aggregation,
        notes=meta.notes,
    )


class DataStore:
    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.builds_dir = self.root / "builds"
        self.catalogue_path = self.root / "catalogue.duckdb"
        self.builds_dir.mkdir(parents=True, exist_ok=True)
        self._recover_interrupted()
        self._init_catalogue()

    def _recover_interrupted(self) -> None:
        """Recover from a save hard-killed mid-swap, WITHOUT touching another writer's active
        work. A build is only recovered/cleaned if its writer lock is *stale* (no live holder).

        Recovery is driven by a **commit marker** (``UNCOMMITTED_MARKER`` inside the build dir),
        removed only after the catalogue commit succeeds, so recovery can tell an *installed but
        uncommitted* build from a fully-committed one:

        - ``final`` present WITH the marker ⇒ crash between swap and catalogue commit; the files
          are uncommitted (catalogue still describes the old build), so discard them and restore
          the ``.bak`` if present (the previous version unconditionally kept ``final`` and
          deleted the valid backup — review).
        - ``final`` present WITHOUT the marker ⇒ committed; drop any stale ``.bak``.
        - only ``.bak`` (no ``final``) ⇒ crash before the swap; restore it.

        Live staging/backups (lock held by a running process) are left untouched."""
        for lock in self.builds_dir.glob(".*.lock"):
            build_id = lock.name[1:-5]  # strip leading '.' and trailing '.lock'
            if _lock_is_live(lock):
                continue  # a writer is active on this build; hands off
            final = self.builds_dir / build_id
            bak = self.builds_dir / f".{build_id}.bak"
            final_uncommitted = final.exists() and (final / UNCOMMITTED_MARKER).exists()

            if final_uncommitted:
                # Swap happened, catalogue never committed → the on-disk 'final' is not the
                # committed build. Discard it and restore the backup (the last committed state).
                shutil.rmtree(final, ignore_errors=True)
                if bak.exists():
                    bak.replace(final)
            elif final.exists():
                shutil.rmtree(bak, ignore_errors=True)  # committed; backup is stale
            elif bak.exists():
                bak.replace(final)  # crash before the swap; restore the prior build

            for tmp in self.builds_dir.glob(f".{build_id}.*.tmp"):
                shutil.rmtree(tmp, ignore_errors=True)
            lock.unlink(missing_ok=True)

    # -- catalogue -------------------------------------------------------------
    def _with_catalogue(self, fn):
        """Run ``fn(con)`` on a fresh DuckDB connection, serialised by the global catalogue
        lock and retried on DuckDB's cross-process file-lock error. All catalogue access
        (init, upsert, read) goes through here so two processes never hold DuckDB's own
        file lock at once (review: concurrent init/write collided)."""
        cat_lock = self.catalogue_path.with_suffix(".lock")
        with _global_lock(cat_lock):
            last_err: Exception | None = None
            for _ in range(_CATALOGUE_RETRIES):
                try:
                    con = duckdb.connect(str(self.catalogue_path))
                    try:
                        return fn(con)
                    finally:
                        con.close()
                except duckdb.IOException as exc:  # another process holds DuckDB's file lock
                    last_err = exc
                    time.sleep(_CATALOGUE_RETRY_DELAY)
            raise RuntimeError(f"catalogue access failed after retries: {last_err}")

    def _init_catalogue(self) -> None:
        self._with_catalogue(
            lambda con: con.execute(
                """
                CREATE TABLE IF NOT EXISTS builds (
                    build_id      VARCHAR PRIMARY KEY,
                    source        VARCHAR,
                    source_version VARCHAR,
                    reference_year INTEGER,
                    aggregation   VARCHAR,
                    n_labels      INTEGER,
                    quality_worst VARCHAR,
                    retrieved     VARCHAR
                )
                """
            )
        )

    def _catalogue_upsert(self, meta: BuildMeta, n_labels: int, quality_worst: str | None) -> None:
        """Upsert one build's catalogue row: DELETE + INSERT in a single transaction (so the
        row is never briefly absent), serialised and retried via ``_with_catalogue``."""

        def _do(con) -> None:
            con.execute("BEGIN TRANSACTION")
            con.execute("DELETE FROM builds WHERE build_id = ?", [meta.build_id])
            con.execute(
                "INSERT INTO builds VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    meta.build_id,
                    meta.source,
                    meta.source_version,
                    meta.reference_year,
                    meta.aggregation,
                    n_labels,
                    quality_worst,
                    meta.retrieved,
                ],
            )
            con.execute("COMMIT")

        self._with_catalogue(_do)

    def catalogue(self) -> pd.DataFrame:
        """Return the build catalogue as a DataFrame (what the GUI lists)."""
        return self._with_catalogue(
            lambda con: con.execute("SELECT * FROM builds ORDER BY build_id").fetchdf()
        )

    def build_ids(self) -> list[str]:
        # Recover any crashed-subprocess builds first, then exclude internal dot-dirs
        # (.tmp staging, .bak backups) — they are not builds (review).
        self._recover_interrupted()
        return sorted(
            p.name for p in self.builds_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
        )

    def has(self, build_id: str) -> bool:
        self._recover_interrupted()
        return (self.builds_dir / build_id / "meta.json").exists()

    # -- write -----------------------------------------------------------------
    def save(
        self,
        *,
        meta: BuildMeta,
        io: IOSystem,
        satellites: list[SatelliteAccount],
        quality: QualityReport | None = None,
    ) -> Path:
        """Persist a build with crash-safe replacement.

        Write to a staging dir, then swap it into place. Because a POSIX directory rename
        cannot atomically replace a *non-empty* existing directory in one syscall, the swap
        is: move the old build to a ``.bak`` alongside, rename staging → final, drop the
        backup. This is **recoverable, not strictly atomic** — there is a brief window where
        the canonical path is absent, and a hard kill in that window leaves the data in the
        backup, which ``_recover_interrupted`` restores. Recovery runs on store construction
        AND on enumerate/read (``build_ids``/``has``/``load``), so a long-lived process picks
        up a crashed subprocess's build without recreating the store. On an ordinary exception
        the backup is restored immediately, so an existing build is never lost. Avoids partial
        writes and stale files from in-place overwrites.
        """
        final = self.builds_dir / meta.build_id
        backup = self.builds_dir / f".{meta.build_id}.bak"
        # Unique staging dir per save (uuid), so two same-process saves never collide.
        staging = self.builds_dir / f".{meta.build_id}.{uuid.uuid4().hex}.tmp"
        lock = self.builds_dir / f".{meta.build_id}.lock"

        # Serialise writers to this build: fail fast if another live writer holds the lock.
        _acquire_lock(lock)
        try:
            # Inside the try so a mkdir failure still releases the lock (review: it was before
            # the try, so a failed mkdir permanently leaked the lock).
            staging.mkdir(parents=True)
            (staging / "meta.json").write_text(meta.model_dump_json(indent=2))
            io.A.astype("float32").to_parquet(staging / "A.parquet")
            io.final_demand.astype("float32").to_parquet(staging / "final_demand.parquet")
            if not io.value_added.empty:
                io.value_added.astype("float32").to_parquet(staging / "value_added.parquet")
            for sat in satellites:
                sat.data.astype("float32").to_parquet(staging / f"satellite_{sat.name}.parquet")
                (staging / f"satellite_{sat.name}.units.json").write_text(
                    json.dumps(sat.units, indent=2)
                )
            if quality is not None:
                (staging / "quality.json").write_text(quality.model_dump_json(indent=2))
            # Mark staging UNCOMMITTED; the marker is removed only after the catalogue commits.
            # If a crash leaves 'final' with this marker, recovery knows the files are not the
            # committed build and restores the backup instead of keeping them (review).
            (staging / UNCOMMITTED_MARKER).write_text(meta.build_id)

            had_existing = final.exists()
            if had_existing:
                if backup.exists():
                    shutil.rmtree(backup)
                final.replace(backup)  # move old aside (recoverable marker)
            try:
                staging.replace(final)  # atomic when 'final' is now absent
            except OSError:
                if had_existing:
                    backup.replace(final)  # restore the prior build
                raise
            # Commit the catalogue BEFORE clearing the marker / dropping the backup. Roll the
            # filesystem back to the backup if the catalogue update fails, so files and catalogue
            # never end up at different revisions (review). Held under the per-build lock.
            try:
                self._catalogue_upsert(
                    meta,
                    n_labels=len(io.A.columns),
                    quality_worst=(quality.worst.value if quality else None),
                )
            except Exception:
                shutil.rmtree(final, ignore_errors=True)
                if had_existing:
                    backup.replace(final)  # restore the prior files to match the catalogue
                raise
            # Catalogue committed → clear the marker (now 'final' is the committed build), then
            # drop the backup. Order matters: marker cleared before backup removal so a crash
            # here leaves a committed 'final' + stale backup, which recovery drops harmlessly.
            (final / UNCOMMITTED_MARKER).unlink(missing_ok=True)
            if had_existing:
                shutil.rmtree(backup, ignore_errors=True)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            lock.unlink(missing_ok=True)
        return final

    # -- read ------------------------------------------------------------------
    def load_meta(self, build_id: str) -> BuildMeta:
        self._recover_interrupted()  # a direct load must also self-heal a crashed swap (review)
        d = self.builds_dir / build_id
        return BuildMeta.model_validate_json((d / "meta.json").read_text())

    def load(self, build_id: str) -> dict:
        """Return harmonised data objects for a build, keyed by type name — the exact
        shape engines expect from the runner's ``data`` argument."""
        self._recover_interrupted()  # self-heal before a direct load (GUI frames() calls here)
        d = self.builds_dir / build_id
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"No build {build_id!r} in {self.builds_dir}")
        meta = self.load_meta(build_id)
        prov = _meta_to_provenance(meta)

        A = pd.read_parquet(d / "A.parquet")
        final_demand = pd.read_parquet(d / "final_demand.parquet")
        value_added = (
            pd.read_parquet(d / "value_added.parquet")
            if (d / "value_added.parquet").exists()
            else pd.DataFrame()
        )
        sectors, regions = _classifications_from_labels(list(A.columns))
        io = IOSystem(
            provenance=prov,
            sectors=sectors,
            regions=regions,
            price_basis=meta.price_basis,
            currency=meta.currency,
            unit=meta.monetary_unit,
            A=A,
            final_demand=final_demand,
            value_added=value_added,
        )

        sats: dict[str, SatelliteAccount] = {}
        for parquet in sorted(d.glob("satellite_*.parquet")):
            name = parquet.stem.removeprefix("satellite_")
            units_path = d / f"satellite_{name}.units.json"
            units = json.loads(units_path.read_text()) if units_path.exists() else {}
            sats[name] = SatelliteAccount(
                provenance=prov,
                name=name,
                units=units,
                data=pd.read_parquet(parquet),
            )

        out: dict = {"IOSystem": io}
        # Engines ask for "SatelliteAccount"; expose the GHG one under that key by
        # convention, and all of them under their explicit names.
        if "GHG" in sats:
            out["SatelliteAccount"] = sats["GHG"]
        out["satellites"] = sats
        return out

    def load_quality(self, build_id: str) -> QualityReport | None:
        self._recover_interrupted()
        p = self.builds_dir / build_id / "quality.json"
        return QualityReport.model_validate_json(p.read_text()) if p.exists() else None


def _classifications_from_labels(labels: list[str]) -> tuple[Classification, Classification]:
    """Reconstruct sector/region classifications from ``region:sector`` labels."""
    regions: list[str] = []
    sectors: list[str] = []
    for label in labels:
        region, sector = label.split(":", 1)
        if region not in regions:
            regions.append(region)
        if sector not in sectors:
            sectors.append(sector)
    return (
        Classification(name="build-sectors", kind="sector", labels=sectors),
        Classification(name="build-regions", kind="region", labels=regions),
    )


_default: DataStore | None = None


def default_store() -> DataStore:
    global _default
    if _default is None:
        _default = DataStore()
    return _default
