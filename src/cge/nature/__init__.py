"""Nature extension via ENCORE (Phase 6).

Ecosystem-service dependency exposure: ingest ENCORE ratings (``encore``), map them onto the
economy's sectors (``concord``), and propagate them through the input–output structure to a per-good
direct + upstream exposure (``exposure``). See ``docs/models/nature-encore.md``.

The nature→shock translation (a ``NatureStress`` → ``ProductivityShock`` scaled by exposure, fed to
the economic engines) is Phase 6.4, a documented follow-up.
"""

from cge.nature.concord import broadcast_to_goods, sector_scores
from cge.nature.encore import (
    MATERIALITY_SCALE,
    EncoreDependencies,
    load_encore_csv,
    materiality_to_score,
)
from cge.nature.exposure import compute_exposure

__all__ = [
    "MATERIALITY_SCALE",
    "EncoreDependencies",
    "load_encore_csv",
    "materiality_to_score",
    "sector_scores",
    "broadcast_to_goods",
    "compute_exposure",
]
