"""Engine 1 — Leontief carbon-cost price model.

Full method, to equation level with citations, is specified in
``docs/models/io-price-model.md``. The core solve is equation (5),
``Δp = (I − Aᵀ)⁻¹ · τ · e``, computed as a linear solve rather than an explicit inverse.
Importing this package registers the engine.
"""

from cge.engines.io_price.engine import IOPriceEngine, decompose, price_change

__all__ = ["IOPriceEngine", "price_change", "decompose"]
