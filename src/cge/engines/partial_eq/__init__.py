"""Engine 2 — partial-equilibrium volume response (Phase 4).

First-order production-volume response to carbon-price-driven price changes: Δq/q = ε·Δp,
evaluated across low/central/high demand-elasticity bands. Reuses Engine 1 for the prices.
Full method (equation level) in ``docs/models/partial-equilibrium.md``. Importing this package
registers the engine.
"""

from cge.engines.partial_eq.engine import PartialEqEngine

__all__ = ["PartialEqEngine"]
