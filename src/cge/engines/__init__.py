"""Engines. Importing this package registers the available engines with the shared
``registry`` (import side effect is intentional and is how the GUI/CLI discover them).

The dummy engine (Phase 0) exercises the contracts without real economics; io_price
(Engine 1, Phase 2) is the Leontief carbon-cost price model; partial_eq (Engine 2, Phase 4)
is the partial-equilibrium volume response; cge_static (Engine 3, Phase 5) is the static CGE.

cge_static's solver uses scipy/pyomo lazily (only when a run actually solves), so importing
the package is safe on the light core install; a run without the ``[cge]`` extra raises a clear
ImportError at solve time.
"""

from cge.engines import (
    cge_static,  # noqa: F401  (registration side effect)
    dummy,  # noqa: F401  (registration side effect)
    io_price,  # noqa: F401  (registration side effect)
    partial_eq,  # noqa: F401  (registration side effect)
)
