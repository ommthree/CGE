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
    # Government account (Phase 5d.1). Zero/empty when cal.has_government is False — the pilot's
    # pre-5d.1 behaviour (100% of carbon revenue recycled straight to the household) is preserved
    # exactly in that case; these fields are then reported as an all-zero no-op government.
    GD: np.ndarray  # government final demand [i]
    gov_income: float  # government income (its share of carbon revenue, + benchmark gov_income0)
    fiscal_balance: float  # government income − government spending (≡0 under balanced_budget)


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
    gov_closure: str = "balanced_budget",
) -> ModelState:
    """Close the model at equilibrium prices (p, w): compute VA cost, outputs, demands and income.

    **Revenue recycling.** The carbon tax collects ``R = Σ_i cc[i]·X[i]`` (cc = τ·e is the per-unit
    emissions cost). In a **closed** economy the revenue must circulate — money cannot vanish, or
    the circular flow (and Walras' law) does not close.

    **No government account (``cal.has_government`` False — pre-5d.1 behaviour, unchanged):** the
    household receives R directly:
    - ``lump_sum`` — government returns R to the household as a lump-sum transfer; income = factor
      income + R.
    - ``labour_tax_cut`` — revenue rebates a labour tax. In this **single-household** pilot the
      household owns both factors, so a labour rebate and a lump-sum transfer give the *same*
      aggregate household income and hence the same real allocation; the two modes are therefore
      equivalent here.
    - ``none`` — revenue is NOT returned. This does not close a closed economy (the leaked value
      breaks Walras' law); the engine rejects it.

    **With a government account (Phase 5d.1, ``cal.has_government`` True):** the government, not
    the household, collects R (plus the benchmark direct tax ``gov_tax_rate0·factor_income``) and
    spends it on its own Cobb-Douglas demand vector ``gov_gamma``. Under ``balanced_budget`` (the
    only closure 5d.1 implements; ``deficit_financed`` is 5d.7), government spending exactly
    exhausts government income each period, so ``fiscal_balance ≡ 0`` and total final demand is
    ``FD + GD`` — the household's income no longer includes carbon revenue at all (it goes to
    government instead), which is the intended generalisation: recycling is now a real
    institutional transfer, not a same-period pass-through to the same account that pays the tax.
    Note the reported ``welfare`` (CD utility over household FD) therefore values HOUSEHOLD
    consumption only — government-provided goods carry no utility here, a documented 5d.1 scope
    choice.

    Because R depends on X which depends on income which depends on R, the fixed point is solved in
    closed form. Without government: with FD = γ·I/p and X = (I−ax)⁻¹·FD,
    R = I·(cc·(I−ax)⁻¹·(γ/p)), so I = factor_income / (1 − k) where k is the marginal-revenue
    coefficient. With government: household income is factor income net of the benchmark tax
    (T = gov_tax_rate0·factor_income), fixed given prices — so only gov_income has a fixed point:
    gov_income = T + R, R = gov_income·kg + R0 where kg = cc·(I−ax)⁻¹·(gov_gamma/p) is government
    spending's own marginal-revenue coefficient and R0 = cc·(I−ax)⁻¹·FD is the revenue generated
    by (fixed) household demand alone — so gov_income = (T + R0) / (1 − kg)."""
    ns = len(cal.sectors)
    cc = np.zeros(ns) if carbon_cost is None else np.asarray(carbon_cost, dtype=float)
    pv = _va_unit_cost(cal, w)
    factor_income = float(np.dot(w, cal.endowment))

    leontief = np.linalg.inv(np.eye(ns) - cal.ax)  # (I − ax)⁻¹, small and dense
    demand_per_income = cal.gamma / p  # FD = I · demand_per_income
    recycles = recycling != "none"

    if not cal.has_government:
        # Pre-5d.1 behaviour, bit-for-bit unchanged: carbon revenue recycles straight to the
        # household. k = extra revenue generated per unit of household income; 0 when there are no
        # emissions/tax. ``none`` is rejected by the engine (it does not close), so any mode
        # reaching here recycles.
        k = float(cc @ (leontief @ demand_per_income)) if recycles else 0.0
        if strict and k >= 1.0 - 1e-12:
            # Runaway recycling (revenue ≥ income) AT THE ACCEPTED EQUILIBRIUM — refuse rather than
            # return numbers (review P2).
            raise ValueError(f"revenue-recycling fixed point diverges (k={k:.3f} ≥ 1)")
        # A trial price vector during the solve can hit k≥1 even when a valid equilibrium (k<1)
        # exists elsewhere; in non-strict (exploratory) mode use a SMOOTH floor on (1−k): identity
        # for 1−k ≥ δ, asymptotes to δ as 1−k → −∞, keeping income finite and the residual
        # C¹-continuous — not a flat plateau.
        income = factor_income / _safe_denom(1.0 - k)
        FD = income * demand_per_income
        GD = np.zeros(ns)
        gov_income = 0.0
        fiscal_balance = 0.0
    else:
        if gov_closure != "balanced_budget":
            raise ValueError(
                f"unsupported gov_closure {gov_closure!r} (5d.1 only implements "
                "'balanced_budget'; 'deficit_financed' is Phase 5d.7)"
            )
        # Household income is factor income net of the benchmark direct tax — carbon revenue no
        # longer passes through it. The tax is levied as a RATE on factor income (rate·w·FF), so
        # the benchmark government replicates exactly and homogeneity survives (see calibrate).
        tax = cal.gov_tax_rate0 * factor_income
        income = factor_income - tax
        FD = income * demand_per_income
        gov_demand_per_income = cal.gov_gamma / p  # GD = gov_income · gov_demand_per_income
        if recycles:
            r0 = float(cc @ (leontief @ FD))  # revenue generated by (fixed) household demand
            kg = float(cc @ (leontief @ gov_demand_per_income))  # marginal revenue from Σgov spend
        else:
            r0, kg = 0.0, 0.0
        if strict and kg >= 1.0 - 1e-12:
            raise ValueError(f"government revenue-recycling fixed point diverges (kg={kg:.3f} ≥ 1)")
        gov_income = (tax + r0) / _safe_denom(1.0 - kg)
        GD = gov_income * gov_demand_per_income
        fiscal_balance = 0.0  # balanced_budget: spending exactly exhausts income, by construction

    X = leontief @ (FD + GD)  # goods-market clearing
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
        GD=GD,
        income=income,
        gov_income=gov_income,
        fiscal_balance=fiscal_balance,
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
    gov_closure: str = "balanced_budget",
) -> np.ndarray:
    """Equilibrium residual vector F(z) for z = [p (ns), w (nf)].

    Components (square by Walras + numéraire):
    - ns zero-profit conditions: p[i] − (Σ_j ax[j,i]·p[j] + pv[i] + τ·e[i]) = 0;
    - (nf − 1) factor-market clearing: Σ_i F[f,i] − FF[f] = 0 for all f except ``drop_factor``
      (dropped by Walras' law);
    - 1 numéraire: Π_i p[i]^γ[i] − 1 = 0 (fix the exact CD consumer price index — its own
      cost-of-living index — so the deflator relative to it is 1 by construction, not an AM-GM
      artifact of pinning the arithmetic Σγp while reporting the geometric Πp^γ; review P1).

    ``recycling`` selects how carbon revenue is returned (see ``derive_state``); ``gov_closure``
    selects the government's financing closure when ``cal.has_government`` (Phase 5d.1) — the
    government account adds no new unknown/equation here: ``balanced_budget`` makes government
    demand ``GD`` an algebraic function of prices exactly like household demand ``FD``, so the
    system stays square in ``(p, w)`` with no new residual line.

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
        gov_closure=gov_closure,
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
