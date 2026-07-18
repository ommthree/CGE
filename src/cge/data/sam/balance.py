"""SAM balance checking and RAS balancing (roadmap Phase 5.1b).

A SAM must be square with row sum = column sum for every account. Real data is not; the toy is.
This module provides the balance check used by the quality gate and a **RAS** biproportional
balancer for the real-data path (Phase 5.1 on an EXIOBASE build). RAS is the simple, transparent
first choice; cross-entropy [Robinson2001] is the principled alternative when RAS struggles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def imbalance(matrix: pd.DataFrame) -> pd.Series:
    """Per-account row-sum − column-sum. Zero everywhere ⇔ balanced."""
    return matrix.sum(axis=1) - matrix.sum(axis=0)


def is_balanced(matrix: pd.DataFrame, *, tol: float = 1e-6) -> bool:
    """True iff every account's row sum equals its column sum within ``tol``."""
    return bool(imbalance(matrix).abs().max() <= tol)


def ras_balance(
    matrix: pd.DataFrame, targets: pd.Series, *, tol: float = 1e-9, max_iter: int = 1000
) -> pd.DataFrame:
    """Biproportional (RAS) balancing to a common row/column total ``targets`` per account.

    Iteratively scales rows then columns so each account's row and column sums approach
    ``targets[account]``. Requires a target total per account (row total = column total for a
    balanced SAM). Preserves zero cells (a structural zero stays zero). Returns the balanced
    matrix; raises if it does not converge (never returns an unbalanced result silently).

    This is the standard RAS procedure [MillerBlair2009, §7.4]; the real-data SAM path uses it,
    the toy needs no balancing (it is built balanced).
    """
    m = matrix.to_numpy(dtype=float).copy()
    t = targets.reindex(matrix.index).to_numpy(dtype=float)
    if np.any(t < 0):
        raise ValueError("RAS targets must be non-negative")
    for _ in range(max_iter):
        row_sums = m.sum(axis=1)
        r = np.divide(t, row_sums, out=np.ones_like(t), where=row_sums > 0)
        m = m * r[:, None]
        col_sums = m.sum(axis=0)
        s = np.divide(t, col_sums, out=np.ones_like(t), where=col_sums > 0)
        m = m * s[None, :]
        resid = max(np.max(np.abs(m.sum(axis=1) - t)), np.max(np.abs(m.sum(axis=0) - t)))
        if resid < tol:
            return pd.DataFrame(m, index=matrix.index, columns=matrix.columns)
    raise RuntimeError(f"RAS did not converge in {max_iter} iterations (max residual {resid:.2e})")
