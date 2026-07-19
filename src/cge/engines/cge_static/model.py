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
    """Value-added unit cost pv[i]. σ_va = 1 ⇒ Cobb-Douglas
    ``pv = (1/av)·Π_f (w_f/β_f)^{β_f}``; σ_va ≠ 1 ⇒ CES
    ``pv = (1/av)·[Σ_f δ_f^σ w_f^{1-σ}]^{1/(1-σ)}``. Computed per sector so a mix of σ works."""
    ns = len(cal.sectors)
    pv = np.empty(ns)
    for i in range(ns):
        s = cal.va_elast[i]
        if abs(s - 1.0) < 1e-12:
            b = cal.beta[:, i]
            ratio = np.where(b > 0, w / np.where(b > 0, b, 1.0), 1.0)
            pv[i] = (1.0 / cal.av[i]) * np.prod(np.power(ratio, b))
        else:
            d = cal.va_ces_share[:, i]
            pv[i] = (1.0 / cal.av[i]) * np.power(np.sum(d**s * w ** (1.0 - s)), 1.0 / (1.0 - s))
    return pv


def _factor_demand(cal: CalibratedModel, w: np.ndarray, pv: np.ndarray, va_cost: np.ndarray):
    """Factor demand F[f,i] by Shephard's lemma on the VA cost. CD: F = β·va_cost/w; CES:
    F = va_cost·(1/av)·(pv·av)^σ·δ^σ·w^{-σ}. ``va_cost`` = va_share·pv·X is total VA payment."""
    ns, nf = len(cal.sectors), len(cal.factors)
    F = np.empty((nf, ns))
    for i in range(ns):
        s = cal.va_elast[i]
        if abs(s - 1.0) < 1e-12:
            F[:, i] = cal.beta[:, i] * va_cost[i] / w
        else:
            d = cal.va_ces_share[:, i]
            unit = (1.0 / cal.av[i]) * (pv[i] * cal.av[i]) ** s * d**s * w ** (-s)
            F[:, i] = unit * (va_cost[i] / pv[i])  # va_cost/pv = VA quantity
    return F


# Smooth positive floor on the recycling denominator (1−k). Identity for x ≫ δ, asymptotes to δ as
# x → −∞, C¹-continuous everywhere — so an exploratory trial point with k ≥ 1 yields a finite income
# and a residual with a restoring gradient (never a flat plateau or a raised exception). δ is small
# enough not to perturb any real equilibrium (which has 1−k well above it) beyond solver tolerance.
_DENOM_FLOOR = 1e-6


def _safe_denom(x: float) -> float:
    """max(x, δ) smoothed: δ·(1 + softplus((x−δ)/δ)) with softplus(t)=log(1+e^t)."""
    t = (x - _DENOM_FLOOR) / _DENOM_FLOOR
    softplus = np.log1p(np.exp(-abs(t))) + max(t, 0.0)  # numerically-stable log(1+e^t)
    return _DENOM_FLOOR * (1.0 + softplus)


def derive_state(
    cal: CalibratedModel,
    p: np.ndarray,
    w: np.ndarray,
    *,
    carbon_cost: np.ndarray | None = None,
    recycling: str = "none",
    strict: bool = False,
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
    if strict and k >= 1.0 - 1e-12:
        # Runaway recycling (revenue ≥ income) AT THE ACCEPTED EQUILIBRIUM — the closed economy is
        # ill-posed here; refuse rather than return numbers (review P2).
        raise ValueError(f"revenue-recycling fixed point diverges (k={k:.3f} ≥ 1)")
    # A trial price vector during the solve can hit k≥1 even when a valid equilibrium (k<1) exists
    # elsewhere; raising there makes the residual discontinuously unavailable and can abort a solve
    # that starts at benchmark prices (review P2). So in non-strict (exploratory) mode we use a
    # SMOOTH floor on the denominator (1−k): it is the identity for 1−k ≥ δ and asymptotes to δ as
    # 1−k → −∞, keeping income finite and the residual C¹-continuous with a gradient that steers the
    # solver back toward the feasible region — not a flat plateau.
    income = factor_income / _safe_denom(1.0 - k)

    FD = income * demand_per_income  # Cobb-Douglas household demand
    X = leontief @ FD  # goods-market clearing
    carbon_revenue = float(cc @ X)
    # Factor demand (Shephard on the value-added cost va_share·pv·X). CD or CES per sector.
    va_cost = cal.va_share * pv * X  # [i] total VA payment per sector
    F = _factor_demand(cal, w, pv, va_cost)  # [f,i]
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
