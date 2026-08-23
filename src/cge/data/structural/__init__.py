"""Structural-trajectory data for the recursive-dynamic wrapper (Phase 7b.2).

The exogenous driver paths (population, labour-force participation, labour productivity) the Phase
7.1 wrapper steps between static solves, as documented, sourced, provenance-tagged trajectories that
replace the ad-hoc flat trends. The small vendored artifact under ``data/structural/`` is the
review-friendly derived object the code consumes; see ``data/structural/NOTICE.md`` for sources.
"""

from cge.data.structural.library import (
    default_structural_trajectories,
    load_structural_trajectories,
)

__all__ = ["load_structural_trajectories", "default_structural_trajectories"]
