"""Load the **real** ENCORE knowledge base vendored under ``data/encore/`` (Phase 6, 2026-08-09).

This is the counterpart to ``fixture.py`` (which builds a small *synthetic* object for offline
tests). Here we ingest the actual ENCORE May-2026 ratings via ``load_encore_ratings_wide`` — real
ISIC-coded production processes, real ecosystem-service vocabulary, real ND/N-A cells. The data is
CC BY-SA 4.0; see ``data/encore/NOTICE.md`` for attribution.

**Scope.** This exposes the real *dependency* and *pressure* objects AND the real EXIOBASE↔ENCORE
concordance (``real_encore_concordance``, built in ``concordance_build``), so a real nature scenario
runs end-to-end against a real-EXIOBASE-labelled economy. Note that the synthetic *toy* concordance
does NOT map real ISIC codes — use ``real_encore_concordance`` (or your own) with the real objects.
The concordance is **equal-weighted (a documented v1 assumption)** and ENCORE ratings are indicators
of *potential* significance, not calibrated elasticities — so results stay illustrative of the
method, not calibrated risk.
"""

from __future__ import annotations

from pathlib import Path

from cge.contracts.data_objects import Provenance
from cge.nature.encore import EncoreDependencies, load_encore_ratings_wide

# Repo-root-relative default DATASET ROOT — the directory that contains BOTH ``ENCORE files/`` and
# ``Crosswalk tables/`` (data/encore/…). Resolved from this file so it works regardless of the
# process's working directory. Every public ``root`` argument uses this SAME convention (review P2
# round 5 2026-08-13: the dependency and concordance loaders previously disagreed on what ``root``
# meant).
_DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "encore"
_ENCORE_SUBDIR = "ENCORE files"
_CROSSWALK_SUBDIR = "Crosswalk tables"
_DEPENDENCY_CSV = "06. Dependency mat ratings.csv"
_PRESSURE_CSV = "07. Pressure mat ratings.csv"
_CROSSWALK_CSV = "EXIOBASE - NACE Rev. 2 - ISIC Rev. 4 - ISIC Rev. 5.csv"


def _encore_dir(root: str | Path | None) -> Path:
    """The ``ENCORE files/`` directory under the dataset root."""
    return (Path(root) if root else _DEFAULT_ROOT) / _ENCORE_SUBDIR


def _crosswalk_path(root: str | Path | None) -> Path:
    """The EXIOBASE↔ISIC crosswalk CSV under the dataset root."""
    return (Path(root) if root else _DEFAULT_ROOT) / _CROSSWALK_SUBDIR / _CROSSWALK_CSV


# Attribution baked into every real-ENCORE run's provenance (CC BY-SA 4.0 requirement).
_ENCORE_SOURCE = "ENCORE knowledge base (ENCORE Partners: UNEP-WCMC, UNEP FI, Global Canopy)"
_ENCORE_VERSION = "Updated ENCORE knowledge base May 2026"
_ENCORE_LICENCE = "CC BY-SA 4.0"


def encore_data_available(root: str | Path | None = None) -> bool:
    """True if the vendored ENCORE dependency file is present (it may be absent in a checkout that
    excluded the data). Lets tests/GUI degrade gracefully rather than raise on import."""
    return (_encore_dir(root) / _DEPENDENCY_CSV).exists()


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
    ``EncoreDependencies`` object, with attribution provenance. ND cells are kept distinct.

    ``root`` is the DATASET ROOT (the directory containing ``ENCORE files/`` and ``Crosswalk
    tables/``), the same convention as every other loader here."""
    path = _encore_dir(root) / _DEPENDENCY_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"ENCORE dependency file not found at {path}. It is vendored under data/encore/ "
            "(CC BY-SA 4.0); ensure the data directory is present in this checkout."
        )
    return load_encore_ratings_wide(str(path), provenance=_provenance(), kind="dependency")


def real_encore_pressures(root: str | Path | None = None) -> EncoreDependencies:
    """The real ENCORE **pressure / impact-driver** ratings, typed ``kind='impact'``. Ingested and
    provenance-stamped, but NOT yet consumed by an engine (dependencies drive the productivity
    channel; a pressure/impact channel is a documented follow-up). ``root`` is the dataset root."""
    path = _encore_dir(root) / _PRESSURE_CSV
    if not path.exists():
        raise FileNotFoundError(f"ENCORE pressure file not found at {path}.")
    return load_encore_ratings_wide(str(path), provenance=_provenance(), kind="impact")


def real_encore_concordance(root: str | Path | None = None):
    """The real **EXIOBASE → ENCORE** concordance, derived from the vendored crosswalk against the
    real dependency processes. Returns ``(ConcordanceMap, ConcordanceAudit)``. Equal-weighted v1
    (a documented assumption — see ``concordance_build`` and the audit). This is what lets a real
    nature scenario run against a real EXIOBASE-shaped economy. ``root`` is the dataset root (same
    convention as ``real_encore_dependencies``)."""
    from cge.nature.concordance_build import build_exiobase_encore_concordance

    dep = real_encore_dependencies(root)
    return build_exiobase_encore_concordance(dep, crosswalk_path=_crosswalk_path(root))
