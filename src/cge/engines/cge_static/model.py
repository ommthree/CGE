"""The static CGE model as a residual system (roadmap Phase 5.2c).

Given a ``CalibratedModel`` and a shock, this builds the vector of equilibrium residuals
``F(z) = 0`` that the solver drives to zero. The pilot model (see docs/models/cge-static.md):

- **Prices** are the unknowns: commodity prices ``p[i]`` and factor prices ``w[f]``. Everything
  else (outputs, demands, income) is a closed-form function of prices, so the equilibrium is a
  small square system in ``(p, w)``.
- **Production:** Leontief intermediates + Cobb-Douglas value added. The Cobb-Douglas VA **unit
  cost** is ``pv[i] = (1/av[i])·Π_f (w[f]/β[f,i])^{β[f,i]}``; the zero-profit condition is
  ``p[i] = Σ_j ax[j,i]·p[j] + pv[i]``.
- **Household:** Cobb-Douglas demand ``FD[i] = γ[i]·I/p[i]`` from income ``I = Σ_f w[f]·FF[f]``.
- **Goods market:** ``X = (I − ax)⁻¹ FD`` (output meets intermediate + final demand).
- **Factor market:** demand ``F[f,i] = β[f,i]·pv[i]·X[i]/w[f]``; clearing ``Σ_i F[f,i] = FF[f]``.
- **Closure / numéraire:** the consumer price index is fixed to 1 (``Σ_i γ[i]·p[i] = 1``),
  pinning the price level. By Walras' law one market clears residually, so the CPI equation
  replaces one redundant factor-clearing equation — keeping the system square.

**Carbon price** enters as a per-unit cost on each sector's emissions (reusing the Engine-1
emission intensities, so units stay consistent): it adds ``τ·e[i]`` to sector ``i``'s unit cost in
the zero-profit condition. Revenue recycling is handled by the engine (Phase 5.3); the pilot
residual keeps the tax as a pure cost wedge (``none`` recycling).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cge.engines.cge_static.calibrate import CalibratedModel


@dataclass(frozen=True)
class ModelState:
    """Everything derived from an equilibrium price vector — the reported quantities."""

    p: np.ndarray  # commodity prices [i]
    w: np.ndarray  # factor prices [f]
    pv: np.ndarray  # value-added unit cost [i]
    X: np.ndarray  # gross output [i]
    F: np.ndarray  # factor demand [f, i]
    FD: np.ndarray  # household final demand [i]
    income: float  # household income


def _va_unit_cost(cal: CalibratedModel, w: np.ndarray) -> np.ndarray:
    """Cobb-Douglas value-added unit cost pv[i] = (1/av[i])·Π_f (w[f]/β[f,i])^{β[f,i]}."""
    # w[:,None] broadcast over sectors; guard β=0 (a factor unused in a sector) by skipping it.
    beta = cal.beta  # [f,i]
    ratio = np.where(beta > 0, (w[:, None] / np.where(beta > 0, beta, 1.0)), 1.0)
    return (1.0 / cal.av) * np.prod(np.power(ratio, beta), axis=0)


def derive_state(cal: CalibratedModel, p: np.ndarray, w: np.ndarray) -> ModelState:
    """Close the model at equilibrium prices (p, w): compute VA cost, outputs, demands and income.

    The carbon tax does not appear here — it is a cost wedge in the zero-profit *price* condition
    (``residuals``), so by the time we have equilibrium prices the tax is already reflected in
    them; quantities follow from prices alone."""
    ns = len(cal.sectors)
    pv = _va_unit_cost(cal, w)
    income = float(np.dot(w, cal.endowment))
    FD = cal.gamma * income / p  # Cobb-Douglas household demand
    # Goods market: X = ax·X + FD  ⇒  X = (I − ax)⁻¹ FD. ax is [input j, output i]; the intermediate
    # demand for good i is Σ_j ax[i,j]·X[j], so the Leontief operator uses ax directly.
    X = np.linalg.solve(np.eye(ns) - cal.ax, FD)
    # Factor demand (Shephard on the value-added cost va_share·pv·X, split by CD share β).
    va_cost = cal.va_share * pv * X  # [i] total VA payment per sector
    F = cal.beta * va_cost[None, :] / w[:, None]  # [f,i]
    return ModelState(p=p, w=w, pv=pv, X=X, F=F, FD=FD, income=income)


def residuals(
    cal: CalibratedModel,
    z: np.ndarray,
    *,
    carbon_cost: np.ndarray | None = None,
    drop_factor: int = 0,
) -> np.ndarray:
    """Equilibrium residual vector F(z) for z = [p (ns), w (nf)].

    Components (square by Walras + numéraire):
    - ns zero-profit conditions: p[i] − (Σ_j ax[j,i]·p[j] + pv[i] + τ·e[i]) = 0;
    - (nf − 1) factor-market clearing: Σ_i F[f,i] − FF[f] = 0 for all f except ``drop_factor``
      (dropped by Walras' law);
    - 1 numéraire: Σ_i γ[i]·p[i] − 1 = 0 (fix the CPI).

    ``z`` accepts an object-dtype array (pyomo vars) so the same residual builds the IPOPT model;
    it uses only +, −, ×, ÷ and np.dot-free elementwise algebra where that matters.
    """
    ns = len(cal.sectors)
    nf = len(cal.factors)
    p = z[:ns]
    w = z[ns : ns + nf]
    cc = np.zeros(ns) if carbon_cost is None else np.asarray(carbon_cost, dtype=float)

    state = derive_state(cal, np.asarray(p, dtype=float), np.asarray(w, dtype=float))

    res = []
    # Zero-profit: p[i] = Σ_j ax[j,i]·p[j] + va_share[i]·pv[i] + carbon cost.
    for i in range(ns):
        intermediate = sum(cal.ax[j, i] * p[j] for j in range(ns))
        res.append(p[i] - (intermediate + cal.va_share[i] * state.pv[i] + cc[i]))
    # Factor clearing (drop one by Walras).
    for f in range(nf):
        if f == drop_factor:
            continue
        res.append(float(state.F[f, :].sum()) - cal.endowment[f])
    # Numéraire: CPI = Σ γ[i]·p[i] = 1.
    res.append(sum(cal.gamma[i] * p[i] for i in range(ns)) - 1.0)
    return np.array(res, dtype=float if not _is_object(z) else object)


def _is_object(z) -> bool:
    return getattr(z, "dtype", None) is not None and z.dtype == object


def initial_guess(cal: CalibratedModel) -> np.ndarray:
    """Benchmark starting point: all prices = 1 (the calibration point). z = [p, w]."""
    return np.ones(len(cal.sectors) + len(cal.factors))


def n_unknowns(cal: CalibratedModel) -> int:
    return len(cal.sectors) + len(cal.factors)
