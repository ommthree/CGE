"""Recursive-dynamic driver for the static CGE (Phase 7.1).

Solves a ``general_equilibrium`` engine **year by year** to a horizon, carrying the capital stock
forward between static solves via the Phase-5d.3 accumulation identity (and optional exogenous
labour and productivity trends). This is *bookkeeping between solves* — no perfect foresight, no new
solution concept — exactly as the roadmap specifies. See ``docs/models/recursive-dynamics.md``.
"""

from cge.dynamics.recursive import (
    DynamicConfig,
    DynamicPath,
    run_recursive,
)

__all__ = ["DynamicConfig", "DynamicPath", "run_recursive"]
