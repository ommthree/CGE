"""Engines. Importing this package registers the available engines with the shared
``registry`` (import side effect is intentional and is how the GUI/CLI discover them).

Phase 0 ships only the dummy engine, which exercises every contract end-to-end without
doing real economics. Real engines (io_price, partial_eq, cge_static) land in later phases.
"""

from cge.engines import dummy  # noqa: F401  (registration side effect)
