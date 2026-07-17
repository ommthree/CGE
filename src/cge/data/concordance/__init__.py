"""Concordance framework (task 1.6).

Many-to-many weighted maps between classifications, with validation (weights sum to 1,
no orphaned source labels) and the operations that use them: building a bridge matrix and
applying it to aggregate an IO system. Reused for aggregation (P1), ENCORE↔EXIOBASE (P6),
and any second data source (P7).
"""

from cge.data.concordance.concordance import (
    bridge_matrix,
    check_covers,
    load_concordance,
    save_concordance,
)

__all__ = ["bridge_matrix", "check_covers", "load_concordance", "save_concordance"]
