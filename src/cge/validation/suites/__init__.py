"""Validation suites. Importing this package registers every suite (side effect), so the
runner can discover them. Add a module here per engine/model and import it below.
"""

from cge.validation.suites import (  # noqa: F401
    data_layer,
    exiobase_live,
    io_price,
    macro,
    partial_eq,
)

__all__ = ["io_price", "data_layer", "exiobase_live", "partial_eq", "macro"]
