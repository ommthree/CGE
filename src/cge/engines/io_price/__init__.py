"""Engine 1 — Leontief carbon-cost price model (Phase 2, not yet implemented).

Full method, to equation level with citations, is specified in
``docs/models/io-price-model.md``. Implement this engine against that doc: the
``RunManifest.assumptions`` it emits must match the doc's assumption list, and the
core solve is equation (5), ``Delta_p = (I - A^T)^-1 . tau . e``, computed as a linear
solve rather than an explicit inverse.
"""
