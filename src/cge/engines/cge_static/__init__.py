"""Engine 3 — static computable general equilibrium (Phase 5).

Importing this package registers the engine (side effect). The solver (scipy/pyomo) is imported
lazily inside ``solver.py``, so importing the package is safe without the ``[cge]`` extra.
"""

from cge.engines.cge_static.engine import CGEStaticEngine

__all__ = ["CGEStaticEngine"]
