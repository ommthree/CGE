"""The nature exposure engine (Phase 6.3).

Given ENCORE **direct** dependency scores per production process and the economy's input–output
structure, compute each good's **total** dependency on each ecosystem service — direct *plus* the
dependency inherited through its supply chain (a good that buys a lot of agricultural input inherits
agriculture's pollination/water dependency, even if it has no direct dependency itself).

**The propagation, stated plainly.** Let ``D`` be the goods × service **direct** dependency matrix
(from ENCORE, via ``MATERIALITY_SCALE``) and ``A`` the technical-coefficient matrix
(``A[i, j]`` = units of good *i* used to make one unit of good *j*, so column *j* sums to good *j*'s
intermediate-input share of its gross output — the remainder is value added). Exposure is a **risk**
measure: a good is *at least* as exposed as its own direct dependency, and its supply chain can only
*add* exposure, never remove it — so ``total ≥ direct`` always. Two aggregation rules combine the
upstream contribution, the modelling choice the roadmap says to expose, not bury:

- ``"weighted_mean"`` — the good's own direct dependency, raised by an input-weighted contribution
  from its inputs' exposure, scaled by the remaining headroom so it stays in [0, 1] (a "noisy-OR"):

      E[j, k] = D[j, k] + (1 − D[j, k]) · Σ_i A[i, j]·E[i, k]

  Because ``A``'s columns sum to < 1 (value added is the remainder) the recursion damps and settles
  to a unique fixed point. Smooth, and input-intensity-weighted, so a small dirty input contributes
  a little.
- ``"max"`` — conservative screening: a good is as exposed as the MOST-exposed thing anywhere in its
  supply chain, regardless of that input's share (any nonzero ``A[i, j]`` counts as "uses").

Both are monotone, bounded in [0, 1], and satisfy ``total ≥ direct``; both are solved by fixed-point
iteration. The direct-only scores are also returned so a GUI can show "direct vs total".

See ``docs/models/nature-encore.md`` for the equations and sourcing.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

AggregationRule = Literal["weighted_mean", "max"]


def compute_exposure(
    A: pd.DataFrame,
    direct: pd.DataFrame,
    *,
    rule: AggregationRule = "weighted_mean",
    max_iter: int = 1000,
    tol: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Total dependency exposure per good × service.

    ``A``: goods × goods technical coefficients (index == columns == the goods labels; column *j*
    sums to good *j*'s intermediate-input share, < 1 because value added is the remainder).
    ``direct``: goods × service direct dependency scores in [0, 1] (from ENCORE; reindexed onto
    ``A``'s goods, missing goods → 0 direct dependency).

    Returns ``(total, direct_aligned)`` — both goods × service DataFrames in [0, 1].
    """
    goods = list(A.index)
    if list(A.columns) != goods:
        raise ValueError("exposure: A must be square with identical row/column (goods) labels")
    # Align direct scores onto the economy's goods; a good ENCORE doesn't rate has 0 direct
    # dependency (its exposure, if any, is entirely upstream).
    D = direct.reindex(index=goods).fillna(0.0)
    services = list(D.columns)
    Dv = D.to_numpy(dtype=float)
    a = A.to_numpy(dtype=float)
    rho = a.sum(axis=0)  # good j's intermediate-input share of gross output (value added = 1 − rho)
    if float(rho.max()) >= 1.0:
        raise ValueError(
            f"exposure: a good's intermediate-input share ≥ 1 (max {rho.max():.4f}); the economy "
            "is not productive (no value added), so dependency propagation would not damp"
        )

    if rule == "weighted_mean":
        # E[j] = D[j] + (1 − D[j]) · Σ_i A[i,j]·E[i]  (noisy-OR: upstream only adds, scaled by
        # headroom). Monotone non-decreasing in E, bounded above by 1; Σ_i A[i,j] < 1 makes it a
        # contraction, so iterate to the unique fixed point.
        aT = a.T
        Ev = Dv.copy()
        for _ in range(max_iter):
            upstream = aT @ Ev  # Σ_i A[i,j]·E[i] per good j
            new = Dv + (1.0 - Dv) * upstream
            new = np.clip(new, 0.0, 1.0)
            if np.max(np.abs(new - Ev)) < tol:
                Ev = new
                break
            Ev = new
    elif rule == "max":
        # E[j, k] = max( D[j, k], max over inputs i (A[i,j] > 0) of E[i, k] ). Not linear → iterate
        # to a fixed point. Monotone non-decreasing and bounded above by 1, so it converges.
        reaches = a.T > 0  # reaches[j, i] True if good j uses good i as an input
        Ev = Dv.copy()
        for _ in range(max_iter):
            upstream = np.zeros_like(Ev)
            for j in range(Ev.shape[0]):
                inputs = np.where(reaches[j])[0]
                if inputs.size:
                    upstream[j] = Ev[inputs].max(axis=0)
            new = np.maximum(Dv, upstream)
            if np.max(np.abs(new - Ev)) < tol:
                Ev = new
                break
            Ev = new
    else:
        raise ValueError(f"unknown aggregation rule {rule!r}; use 'weighted_mean' or 'max'")

    total = pd.DataFrame(Ev, index=goods, columns=services)
    return total, D
