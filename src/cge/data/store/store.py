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
        notes=meta.notes,
    )


class DataStore:
    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.builds_dir = self.root / "builds"
        self.catalogue_path = self.root / "catalogue.duckdb"
        self.builds_dir.mkdir(parents=True, exist_ok=True)
        self._init_catalogue()

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
        d = self.builds_dir / meta.build_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(meta.model_dump_json(indent=2))

        io.A.astype("float32").to_parquet(d / "A.parquet")
        io.final_demand.astype("float32").to_parquet(d / "final_demand.parquet")
        if not io.value_added.empty:
            io.value_added.astype("float32").to_parquet(d / "value_added.parquet")

        for sat in satellites:
            sat.data.astype("float32").to_parquet(d / f"satellite_{sat.name}.parquet")
            (d / f"satellite_{sat.name}.units.json").write_text(json.dumps(sat.units, indent=2))

        if quality is not None:
            (d / "quality.json").write_text(quality.model_dump_json(indent=2))

        self._catalogue_upsert(
            meta,
            n_labels=len(io.A.columns),
            quality_worst=(quality.worst.value if quality else None),
        )
        return d

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
