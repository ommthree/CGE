"""Engine 2 — partial-equilibrium volume response (Phase 4).

Production-volume response to carbon-price-driven price changes: a finite-change demand
response Δy/y=(1+Δp)^ε−1 propagated through the Leontief quantity system x=(I−A)⁻¹y to give
Δx/x (production), evaluated across low/central/high demand-elasticity bands. Reuses Engine 1
for the prices. Full method (equation level) in ``docs/models/partial-equilibrium.md``.
Importing this package registers the engine.
"""

from cge.engines.partial_eq.engine import PartialEqEngine

__all__ = ["PartialEqEngine"]
