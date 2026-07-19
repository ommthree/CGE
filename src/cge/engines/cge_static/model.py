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
- **Closure / numéraire:** the household's exact Cobb-Douglas price index (its cost of living) is
  fixed to 1 (``Π_i p[i]^γ[i] = 1``), pinning the price level in CPI units. By Walras' law one
  market clears residually, so this equation replaces one redundant factor-clearing equation —
  keeping the system square. Because the CPI *is* the numéraire, there is no separate inflation
  ("deflator") to report; real quantities and relative prices are the outputs.

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
    income: float  # household income (incl. any recycled carbon revenue)
    carbon_revenue: float  # τ·Σ e[i]·X[i] collected by government
    factor_income: float  # Σ_f w[f]·FF[f] (pre-transfer)


def _va_unit_cost(cal: CalibratedModel, w: np.ndarray) -> np.ndarray:
    """Cobb-Douglas value-added unit cost pv[i] = (1/av[i])·Π_f (w[f]/β[f,i])^{β[f,i]}."""
    # w[:,None] broadcast over sectors; guard β=0 (a factor unused in a sector) by skipping it.
    beta = cal.beta  # [f,i]
    ratio = np.where(beta > 0, (w[:, None] / np.where(beta > 0, beta, 1.0)), 1.0)
    return (1.0 / cal.av) * np.prod(np.power(ratio, beta), axis=0)


def derive_state(
    cal: CalibratedModel,
    p: np.ndarray,
    w: np.ndarray,
    *,
    carbon_cost: np.ndarray | None = None,
    recycling: str = "none",
) -> ModelState:
    """Close the model at equilibrium prices (p, w): compute VA cost, outputs, demands and income.

    **Revenue recycling.** The carbon tax collects ``R = Σ_i cc[i]·X[i]`` (cc = τ·e is the per-unit
    emissions cost). In a **closed** economy the revenue must circulate — money cannot vanish, or
    the circular flow (and Walras' law) does not close. So the household receives R:
    - ``lump_sum`` — government returns R to the household as a lump-sum transfer; income = factor
      income + R.
    - ``labour_tax_cut`` — revenue rebates a labour tax. In this **single-household** pilot the
      household owns both factors, so a labour rebate and a lump-sum transfer give the *same*
      aggregate household income and hence the same real allocation; the two modes are therefore
      equivalent here. They diverge only once the model has heterogeneous households (a labour vs
      capital household) or a distortionary labour-tax wedge to cut — the "double-dividend" channel
      — a documented follow-up.
    - ``none`` — revenue is NOT returned. This does not close a closed economy (the leaked value
      breaks Walras' law); the engine rejects it and points the user to Engine 1 for the pure
      price-side view, or to a recycling mode for a proper GE run.

    Because R depends on X which depends on income which depends on R, the fixed point is solved in
    closed form: with FD = γ·I/p and X = (I−ax)⁻¹·FD, R = I·(cc·(I−ax)⁻¹·(γ/p)), so
    I = factor_income / (1 − k) where k = cc·(I−ax)⁻¹·(γ/p) is the marginal-revenue coefficient."""
    ns = len(cal.sectors)
    cc = np.zeros(ns) if carbon_cost is None else np.asarray(carbon_cost, dtype=float)
    pv = _va_unit_cost(cal, w)
    factor_income = float(np.dot(w, cal.endowment))

    leontief = np.linalg.inv(np.eye(ns) - cal.ax)  # (I − ax)⁻¹, small and dense
    demand_per_income = cal.gamma / p  # FD = I · demand_per_income
    # Carbon revenue is returned to the household (closed economy: it must circulate). k = extra
    # revenue generated per unit of household income; 0 when there are no emissions/tax. ``none``
    # is rejected by the engine (it does not close), so any mode reaching here recycles.
    recycles = recycling != "none"
    k = float(cc @ (leontief @ demand_per_income)) if recycles else 0.0
    if k >= 1.0:
        # Runaway recycling (revenue ≥ income) — the closed economy is ill-posed here; refuse
        # rather than divide through a non-positive denominator.
        raise ValueError(f"revenue-recycling fixed point diverges (k={k:.3f} ≥ 1)")
    income = factor_income / (1.0 - k)

    FD = income * demand_per_income  # Cobb-Douglas household demand
    X = leontief @ FD  # goods-market clearing
    carbon_revenue = float(cc @ X)
    # Factor demand (Shephard on the value-added cost va_share·pv·X, split by CD share β).
    va_cost = cal.va_share * pv * X  # [i] total VA payment per sector
    F = cal.beta * va_cost[None, :] / w[:, None]  # [f,i]
    return ModelState(
        p=p,
        w=w,
        pv=pv,
        X=X,
        F=F,
        FD=FD,
        income=income,
        carbon_revenue=carbon_revenue,
        factor_income=factor_income,
    )


def residuals(
    cal: CalibratedModel,
    z: np.ndarray,
    *,
    carbon_cost: np.ndarray | None = None,
    recycling: str = "none",
    drop_factor: int = 0,
) -> np.ndarray:
    """Equilibrium residual vector F(z) for z = [p (ns), w (nf)].

    Components (square by Walras + numéraire):
    - ns zero-profit conditions: p[i] − (Σ_j ax[j,i]·p[j] + pv[i] + τ·e[i]) = 0;
    - (nf − 1) factor-market clearing: Σ_i F[f,i] − FF[f] = 0 for all f except ``drop_factor``
      (dropped by Walras' law);
    - 1 numéraire: Π_i p[i]^γ[i] − 1 = 0 (fix the exact CD consumer price index — its own
      cost-of-living index — so the deflator relative to it is 1 by construction, not an AM-GM
      artifact of pinning the arithmetic Σγp while reporting the geometric Πp^γ; review P1).

    ``recycling`` selects how carbon revenue is returned to the household (see ``derive_state``).

    ``z`` accepts an object-dtype array (pyomo vars) so the same residual builds the IPOPT model;
    it uses only +, −, ×, ÷ and np.dot-free elementwise algebra where that matters.
    """
    ns = len(cal.sectors)
    nf = len(cal.factors)
    p = z[:ns]
    w = z[ns : ns + nf]
    cc = np.zeros(ns) if carbon_cost is None else np.asarray(carbon_cost, dtype=float)

    state = derive_state(
        cal,
        np.asarray(p, dtype=float),
        np.asarray(w, dtype=float),
        carbon_cost=cc,
        recycling=recycling,
    )

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
    # Numéraire: the household's exact Cobb-Douglas price index (cost of living) Π_i p[i]^γ[i] = 1.
    # Using the CD price index itself as numéraire keeps it CONSISTENT with the reported welfare
    # and real-GDP deflation: the deflator relative to it is 1 by construction, not an AM-GM
    # artifact of pinning the *arithmetic* Σγp=1 while measuring the *geometric* Πp^γ (review P1).
    cpi = 1.0
    for i in range(ns):
        cpi = cpi * p[i] ** cal.gamma[i]
    res.append(cpi - 1.0)
    return np.array(res, dtype=float if not _is_object(z) else object)


def _is_object(z) -> bool:
    return getattr(z, "dtype", None) is not None and z.dtype == object


def initial_guess(cal: CalibratedModel) -> np.ndarray:
    """Benchmark starting point: all prices = 1 (the calibration point). z = [p, w]."""
    return np.ones(len(cal.sectors) + len(cal.factors))


def n_unknowns(cal: CalibratedModel) -> int:
    return len(cal.sectors) + len(cal.factors)
