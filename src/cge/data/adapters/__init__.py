"""Data-source adapters (task 1.2).

Each adapter maps a raw source into the harmonised data objects (contract 1). Engines
never see raw formats, so a second source (FIGARO/ICIO, P7) is a new adapter here, not a
refactor anywhere else (ADR-0002).

EXIOBASE is the only adapter in Phase 1. It works against any parsed pymrio ``IOSystem``,
so it maps both a full live download and pymrio's tiny built-in test system (used by the
offline tests) through identical code.
"""

from cge.data.adapters.exiobase import (
    adapt_pymrio,
    fetch_exiobase,
    load_exiobase_test,
    parse_exiobase,
)

__all__ = ["adapt_pymrio", "fetch_exiobase", "parse_exiobase", "load_exiobase_test"]
