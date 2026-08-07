"""Nature extension via ENCORE (Phase 6).

Ecosystem-service dependency exposure: ingest ENCORE ratings (``encore``), map them onto the
economy's sectors (``concord``), propagate them through the input–output structure to a per-good
direct + upstream exposure (``exposure``), and translate a ``NatureStress`` degradation into
per-good ``ProductivityShock``s the economic engines consume (``translate``). See
``docs/models/nature-encore.md``.

``build_nature_shocks`` runs the whole chain end-to-end; Engine 2 (``partial_eq``) consumes the
resulting ``ProductivityShock``s as a supply-side output hit.
"""

from cge.nature.concord import broadcast_to_goods, sector_scores
from cge.nature.encore import (
    MATERIALITY_SCALE,
    EncoreDependencies,
    load_encore_csv,
    materiality_to_score,
)
from cge.nature.exposure import compute_exposure
from cge.nature.translate import build_nature_shocks, nature_to_productivity

__all__ = [
    "MATERIALITY_SCALE",
    "EncoreDependencies",
    "load_encore_csv",
    "materiality_to_score",
    "sector_scores",
    "broadcast_to_goods",
    "compute_exposure",
    "nature_to_productivity",
    "build_nature_shocks",
]
