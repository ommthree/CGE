"""Validation suites. Importing this package registers every suite (side effect), so the
runner can discover them. Add a module here per engine/model and import it below.
"""

from cge.validation.suites import data_layer, io_price  # noqa: F401

__all__ = ["io_price", "data_layer"]
