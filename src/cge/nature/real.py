"""Load the **real** ENCORE knowledge base vendored under ``data/encore/`` (Phase 6, 2026-08-09).

This is the counterpart to ``fixture.py`` (which builds a small *synthetic* object for offline
tests). Here we ingest the actual ENCORE May-2026 ratings via ``load_encore_ratings_wide`` — real
ISIC-coded production processes, real ecosystem-service vocabulary, real ND/N-A cells. The data is
CC BY-SA 4.0; see ``data/encore/NOTICE.md`` for attribution.

**Scope honesty.** This gives you the real *dependency* (and *pressure*) objects. It does NOT yet
give you a real nature *scenario* end-to-end, because ENCORE's processes are keyed by ISIC and the
economy's sectors are EXIOBASE/toy labels — bridging them needs a real ENCORE↔EXIOBASE concordance,
which is the next (deferred) piece. So ``sector_scores`` on the real object requires a concordance
that covers your economy's sectors; the synthetic toy concordance does NOT map real ISIC codes.
"""

from __future__ import annotations

from pathlib import Path

from cge.contracts.data_objects import Provenance
from cge.nature.encore import EncoreDependencies, load_encore_ratings_wide

# Repo-root-relative default location of the vendored ENCORE files (data/encore/…). Resolved from
# this file so it works regardless of the process's working directory.
_DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "encore"
_ENCORE_DIR = _DEFAULT_ROOT / "ENCORE files"
_DEPENDENCY_CSV = "06. Dependency mat ratings.csv"
_PRESSURE_CSV = "07. Pressure mat ratings.csv"

# Attribution baked into every real-ENCORE run's provenance (CC BY-SA 4.0 requirement).
_ENCORE_SOURCE = "ENCORE knowledge base (ENCORE Partners: UNEP-WCMC, UNEP FI, Global Canopy)"
_ENCORE_VERSION = "Updated ENCORE knowledge base May 2026"
_ENCORE_LICENCE = "CC BY-SA 4.0"


def encore_data_available(root: str | Path | None = None) -> bool:
    """True if the vendored ENCORE dependency file is present (it may be absent in a checkout that
    excluded the data). Lets tests/GUI degrade gracefully rather than raise on import."""
    base = Path(root) if root else _ENCORE_DIR
    return (base / _DEPENDENCY_CSV).exists()


def _provenance() -> Provenance:
    return Provenance(
        source=_ENCORE_SOURCE,
        source_version=_ENCORE_VERSION,
        licence=_ENCORE_LICENCE,
        reference_year=2026,
        retrieved="2026-08-09",
        notes=(
            "Real ENCORE ratings vendored under data/encore/ (CC BY-SA 4.0; see NOTICE.md). "
            "Materiality ratings are indicators of POTENTIAL significance, not calibrated "
            "output/TFP elasticities — the severity→productivity mapping remains a scenario "
            "assumption."
        ),
    )


def real_encore_dependencies(root: str | Path | None = None) -> EncoreDependencies:
    """The real ENCORE **dependency** ratings (production process × ecosystem service) as an
    ``EncoreDependencies`` object, with attribution provenance. ND cells are kept distinct."""
    base = Path(root) if root else _ENCORE_DIR
    path = base / _DEPENDENCY_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"ENCORE dependency file not found at {path}. It is vendored under data/encore/ "
            "(CC BY-SA 4.0); ensure the data directory is present in this checkout."
        )
    return load_encore_ratings_wide(str(path), provenance=_provenance(), kind="dependency")


def real_encore_pressures(root: str | Path | None = None) -> EncoreDependencies:
    """The real ENCORE **pressure / impact-driver** ratings, typed ``kind='impact'``. Ingested and
    provenance-stamped, but NOT yet consumed by an engine (dependencies drive the productivity
    channel; a pressure/impact channel is a documented follow-up)."""
    base = Path(root) if root else _ENCORE_DIR
    path = base / _PRESSURE_CSV
    if not path.exists():
        raise FileNotFoundError(f"ENCORE pressure file not found at {path}.")
    return load_encore_ratings_wide(str(path), provenance=_provenance(), kind="impact")
