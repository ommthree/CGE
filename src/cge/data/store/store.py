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
from pathlib import Path

import duckdb
import pandas as pd

from cge.contracts.data_objects import (
    Classification,
    IOSystem,
    Provenance,
    SatelliteAccount,
)
from cge.contracts.quality import QualityReport
from cge.data.metadata import BuildMeta

DEFAULT_ROOT = Path("data_store")


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
        """Restore builds left in a ``.bak`` by a save that was hard-killed mid-swap (the
        brief window where the canonical path is absent). Also sweeps stale ``.tmp`` staging
        dirs. Makes the recoverable-swap strategy self-healing on next open."""
        for bak in self.builds_dir.glob(".*.bak"):
            build_id = bak.name[1:-4]  # strip leading '.' and trailing '.bak'
            final = self.builds_dir / build_id
            if not final.exists():
                bak.replace(final)  # crash happened after old→bak, before staging→final
            else:
                shutil.rmtree(bak, ignore_errors=True)  # final present ⇒ backup is stale
        for tmp in self.builds_dir.glob(".*.tmp"):
            shutil.rmtree(tmp, ignore_errors=True)

    # -- catalogue -------------------------------------------------------------
    def _init_catalogue(self) -> None:
        con = duckdb.connect(str(self.catalogue_path))
        con.execute(
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
        con.close()

    def _catalogue_upsert(self, meta: BuildMeta, n_labels: int, quality_worst: str | None) -> None:
        con = duckdb.connect(str(self.catalogue_path))
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
        con.close()

    def catalogue(self) -> pd.DataFrame:
        """Return the build catalogue as a DataFrame (what the GUI lists)."""
        con = duckdb.connect(str(self.catalogue_path))
        df = con.execute("SELECT * FROM builds ORDER BY build_id").fetchdf()
        con.close()
        return df

    def build_ids(self) -> list[str]:
        return sorted(p.name for p in self.builds_dir.iterdir() if p.is_dir())

    def has(self, build_id: str) -> bool:
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
        backup, which ``_recover_interrupted`` (called on load/enumerate) restores. On an
        ordinary exception the backup is restored immediately, so an existing build is never
        lost. Avoids partial writes and stale files from in-place overwrites.
        """
        final = self.builds_dir / meta.build_id
        staging = self.builds_dir / f".{meta.build_id}.{os.getpid()}.tmp"
        backup = self.builds_dir / f".{meta.build_id}.bak"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        try:
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
            if had_existing:
                shutil.rmtree(backup, ignore_errors=True)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

        self._catalogue_upsert(
            meta,
            n_labels=len(io.A.columns),
            quality_worst=(quality.worst.value if quality else None),
        )
        return final

    # -- read ------------------------------------------------------------------
    def load_meta(self, build_id: str) -> BuildMeta:
        d = self.builds_dir / build_id
        return BuildMeta.model_validate_json((d / "meta.json").read_text())

    def load(self, build_id: str) -> dict:
        """Return harmonised data objects for a build, keyed by type name — the exact
        shape engines expect from the runner's ``data`` argument."""
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
