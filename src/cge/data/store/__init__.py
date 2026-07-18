"""Parquet + DuckDB data store.

A build is persisted as a directory of parquet files (the numeric payloads) plus JSON
sidecars (metadata/provenance/quality). A DuckDB catalogue indexes builds so the GUI and
runner can enumerate "what data we have" without loading it. This is the concrete backing
for ``runner.load_data`` (which was a toy-only stub after Phase 0).
"""

from cge.data.store.store import UNCOMMITTED_MARKER, DataStore, default_store

__all__ = ["DataStore", "default_store", "UNCOMMITTED_MARKER"]
