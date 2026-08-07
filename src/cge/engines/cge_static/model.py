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
    # Savings-investment account (Phase 5d.2). Zeros when cal.has_investment is False.
    ID: np.ndarray  # investment final demand [i]
    savings: float  # household savings (= nominal investment under savings_driven)
    # Labour market (Phase 5d.4). 0 under the default full-employment closure; positive when a
    # wage floor binds (labour supply exceeds employed labour F[LAB]).
    unemployment: float = 0.0
    # Adaptation/transition investment (Phase 5d.6). Nominal adaptation spending, part of the total
    # investment ID (it crowds out ordinary investment under savings_driven — same total, split).
    adaptation_investment: float = 0.0


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
    F = va_cost·(1/av)·(pv·av)^σ·δ^σ·w^{-σ}. ``va_cost`` = pv·(VA quantity) is total VA payment
    (VA quantity = va_share·X in the flat model, or the KL quantity from the energy nest)."""
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


def _leontief_and_va(
    cal: CalibratedModel,
    p: np.ndarray,
    pv: np.ndarray,
    cc: np.ndarray,
    productivity: np.ndarray | None = None,
):
    """Return ``(leontief, va_qty_per_x, cc_eff)`` — the (I − A(p))⁻¹ intermediate-inverse, the
    value-added (KL) quantity per unit output, and an **effective per-output carbon cost**.

    **Flat model** (no energy nest): A = the fixed Leontief ``cal.ax``, VA quantity per unit output
    = ``cal.va_share``, and cc_eff = cc (the carbon cost is already a per-output add-on) — all
    price-independent and bit-identical to the pre-5d.5 code.

    **Energy nest** (Phase 5d.5): intermediate demand is price-responsive (energy substitutes as
    relative prices move), so A(p)[j,i] = intermediate use of commodity j per unit output i comes
    from ``nest_demands`` at unit output; VA quantity per unit output = the KL quantity per unit
    output. **cc_eff = cc** — the carbon cost is a per-OUTPUT wedge, the same as the flat model
    (review remediation 2026-07-26): total revenue = Σ_i cc[i]·X[i] with or without the nest, so
    enabling the nest cannot change a scenario's emissions/revenue meaning. The nest still shifts
    substitution away from taxed energy because a taxed fossil sector's output price rises via
    zero-profit and reaches energy inputs through pq (NOT via a separate energy-price add-on, which
    was the original formulation and silently dropped process/household emissions).

    **Productivity shock (Phase 6.4 GE tier).** ``productivity`` is a per-sector Hicks-neutral
    output multiplier θ[i] (θ=1 ⇒ no shock; a nature degradation lands here as θ<1). A sector with
    productivity θ needs ``1/θ`` times its whole input bundle — intermediates, value added, AND the
    per-output carbon cost — to make one unit of output. Scaling the effective coefficients by 1/θ
    at this single choke point is what makes it consistent everywhere downstream: zero-profit gives
    ``p[i] = unit_cost[i]/θ[i]`` (a degraded sector's price rises), and goods-market and factor
    clearing read the same scaled requirements, so a less-productive sector draws proportionally
    more inputs/factors per unit output with no separate bookkeeping. θ=1 is byte-identical to the
    pre-6.4 code (the ``inv_theta`` factors are exactly 1), so benchmark replication / homogeneity /
    Walras are untouched. NB θ scales *technology*, not the carbon *wedge's* physical meaning:
    covered emissions per unit output are unchanged, only more output-inputs are needed per unit."""
    ns = len(cal.sectors)
    inv_theta = np.ones(ns) if productivity is None else 1.0 / np.asarray(productivity, dtype=float)
    if not cal.has_energy_nest:
        # Scale each sector i's input column by 1/θ[i]: ax[:, i]/θ[i], va_share[i]/θ[i], cc[i]/θ[i].
        ax_eff = cal.ax * inv_theta[np.newaxis, :]
        leontief = np.linalg.inv(np.eye(ns) - ax_eff)
        return leontief, cal.va_share * inv_theta, cc * inv_theta
    from cge.engines.cge_static.energy_nest import nest_demands

    nest = cal.energy_nest
    unit_x = np.ones(ns)
    energy_use, materials_use, kl_qty = nest_demands(nest, p, pv, cc, unit_x)  # per unit output
    a = np.zeros((ns, ns))  # A(p)[j, i] = commodity j per unit output i
    for k, j in enumerate(nest.energy_idx):
        a[j, :] += energy_use[k, :]
    for k, j in enumerate(nest.mat_idx):
        a[j, :] += materials_use[k, :]
    # Productivity scales the whole nested input bundle per unit output by 1/θ[i] (column i), the
    # same Hicks-neutral treatment as the flat model.
    a = a * inv_theta[np.newaxis, :]
    leontief = np.linalg.inv(np.eye(ns) - a)
    # cc_eff = cc/θ: the carbon cost is a per-OUTPUT wedge (review remediation 2026-07-26), same as
    # the flat model, so total revenue = Σ_i cc[i]·X[i] with or without the nest. The nest still
    # substitutes away from taxed energy because a taxed fossil sector's output price rises via
    # zero-profit and flows into the energy-input price through pq.
    return leontief, kl_qty * inv_theta, cc * inv_theta


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


def _hh_demand(income, demand_per_income, s, cal, p, inv_closure, d_adapt, strict):
    """Household consumption ``FD``, investment demand ``ID`` and nominal ``savings`` given a fixed
    household income (used by the government branches, where income is pinned by the tax/fiscal
    closure rather than by a recycling fixed point). Factors out the shared savings/investment
    split so ``balanced_budget`` and ``deficit_financed`` don't duplicate it."""
    ns = len(cal.sectors)
    if not cal.has_investment:
        return income * demand_per_income, np.zeros(ns), 0.0
    if inv_closure == "savings_driven":
        savings = s * income
        FD = (1.0 - s) * income * demand_per_income
        ID = savings * cal.inv_gamma / p + d_adapt  # adaptation re-allocation (Phase 5d.6)
        return FD, ID, savings
    # fixed_real
    ID = cal.INV0.copy()
    savings = float(np.dot(p, ID))
    if strict and income - savings <= 0:
        raise ValueError(
            f"fixed_real investment ({savings:.6g}) exceeds household disposable income "
            f"({income:.6g}); no consumption left."
        )
    return (income - savings) * demand_per_income, ID, savings


def _demand_decomposition(cal, p, s, inv_closure, d_adapt):
    """Return ``(u, d0)`` such that the household's total FINAL demand (consumption ``FD`` +
    investment ``ID``) equals ``I·u + d0`` — the income-linear part ``u`` (= ∂(FD+ID)/∂I) and the
    income-INDEPENDENT part ``d0`` (demand at I=0). This is the SINGLE source of truth for the
    revenue-recycling fixed point's marginal coefficient, so it can never drift from ``_hh_demand``
    (review P1 round 14): the earlier code hard-coded the savings-driven derivative even for
    ``fixed_real``, giving a false equilibrium.

    - No investment account: FD = I·(γ/p), ID = 0 ⇒ u = γ/p, d0 = 0.
    - ``savings_driven``: FD = (1−s)·I·(γ/p), ID = s·I·(inv_gamma/p) + d_adapt ⇒
      u = (1−s)·γ/p + s·inv_gamma/p, d0 = d_adapt (nominally zero-sum).
    - ``fixed_real``: ID = INV0 (constant); FD = (I − p·INV0)·(γ/p) ⇒ u = γ/p (consumption only),
      d0 = INV0 − (p·INV0)·(γ/p) = INV0 − savings·(γ/p) + d_adapt (the fixed investment plus the
      fixed downward shift in consumption that pays for it)."""
    demand_per_income = cal.gamma / p
    if not cal.has_investment:
        return demand_per_income, np.zeros(len(cal.sectors)) + d_adapt
    if inv_closure == "savings_driven":
        u = (1.0 - s) * cal.gamma / p + s * cal.inv_gamma / p
        return u, d_adapt.copy() if hasattr(d_adapt, "copy") else d_adapt
    # fixed_real: investment is constant INV0, only consumption is income-linear.
    savings = float(np.dot(p, cal.INV0))
    u = demand_per_income  # γ/p — consumption only
    d0 = cal.INV0 - savings * demand_per_income + d_adapt
    return u, d0


def derive_state(
    cal: CalibratedModel,
    p: np.ndarray,
    w: np.ndarray,
    *,
    carbon_cost: np.ndarray | None = None,
    recycling: str = "none",
    strict: bool = False,
    gov_closure: str = "balanced_budget",
    inv_closure: str = "savings_driven",
    carbon_revenue_recipient: str = "government",
    labour_floor: float | None = None,
    adapt_amount: float = 0.0,
    adapt_gamma: np.ndarray | None = None,
    productivity: np.ndarray | None = None,
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
    spends it on its own Cobb-Douglas demand vector ``gov_gamma``. Under ``balanced_budget``
    (default), government spending exactly exhausts government income each period, so
    ``fiscal_balance ≡ 0`` and total final demand is
    ``FD + GD`` — the household's income no longer includes carbon revenue at all (it goes to
    government instead), which is the intended generalisation: recycling is now a real
    institutional transfer, not a same-period pass-through to the same account that pays the tax.
    Note the reported ``welfare`` (CD utility over household FD) therefore values HOUSEHOLD
    consumption only — government-provided goods carry no utility here, a documented 5d.1 scope
    choice.

    **Savings and investment (Phase 5d.2, ``cal.has_investment``):** a savings-investment account
    turns part of household income into investment demand ``ID`` with its own sectoral composition
    ``inv_gamma``, under one of two closures:
    - ``savings_driven`` (default, the original Phase 5 spec): the household saves the calibrated
      rate ``s`` of disposable income; nominal investment = savings exactly (the S=I identity is
      substituted in closed form, so the system stays square with NO new unknowns — the identity
      holds by construction and is verified in strict mode);
    - ``fixed_real``: the investment *quantity* vector is fixed at its benchmark ``INV0``;
      household savings adjust residually to finance it (consumption income = I − p·INV0).
    Savings carry no utility in this static model (standard static-CGE treatment): reported
    welfare is CD utility over consumption ``FD`` only.

    Because R depends on X which depends on income which depends on R, the fixed point is solved in
    closed form. Without government: with FD = γ·I/p and X = (I−ax)⁻¹·FD,
    R = I·(cc·(I−ax)⁻¹·(γ/p)), so I = factor_income / (1 − k) where k is the marginal-revenue
    coefficient (with investment, the per-unit-income demand vector becomes
    ((1−s)γ + s·inv_gamma)/p under savings_driven, or gains a fixed INV0 part under fixed_real —
    both still linear in income, so the same closed form applies). With government: household
    income is factor income net of the benchmark tax, fixed given prices — so only gov_income has
    a fixed point: gov_income = (T + R0) / (1 − kg), R0 the revenue from the (price-fixed)
    household + investment demand, kg government spending's own marginal-revenue coefficient."""
    ns = len(cal.sectors)
    cc = np.zeros(ns) if carbon_cost is None else np.asarray(carbon_cost, dtype=float)
    pv = _va_unit_cost(cal, w)

    # (I − A(p))⁻¹ and VA quantity per unit output. Flat model: fixed Leontief + va_share. Energy
    # nest (Phase 5d.5): both price-responsive (energy substitutes as the carbon-inclusive energy
    # price moves) — computed from the nest's Shephard demands, so goods-market clearing and factor
    # demand stay consistent. cc is zero here for the flat model's recycling coefficients; the
    # nest reads the actual carbon cost so substitution responds to it (see below).
    leontief, va_qty_per_x, cc_eff = _leontief_and_va(cal, p, pv, cc, productivity)
    demand_per_income = cal.gamma / p  # FD = I · demand_per_income
    recycles = recycling != "none"
    if cal.has_investment and inv_closure not in ("savings_driven", "fixed_real"):
        raise ValueError(
            f"unsupported inv_closure {inv_closure!r}; 5d.2 implements 'savings_driven' "
            "(default) and 'fixed_real'."
        )
    s = cal.sav_rate0 if cal.has_investment else 0.0

    # Adaptation/transition investment (Phase 5d.6): an exogenous nominal amount ``adapt_amount``
    # earmarked to the ``adapt_gamma`` sectoral composition, financed from the SAME investment
    # budget — so it CROWDS OUT ordinary investment (same total, different split), not a free lunch.
    # It appears as a fixed re-allocation demand d_adapt = A·(adapt_gamma − inv_gamma)/p, which is
    # nominally ZERO-SUM (p·d_adapt = A·(1−1) = 0), added to ID's income-proportional part. Only
    # meaningful under savings_driven with an investment account; the engine gates that.
    if adapt_amount > 0.0:
        if not cal.has_investment:
            raise ValueError(
                "adaptation investment needs a savings-investment account (Phase 5d.2) to crowd "
                "out; none is present."
            )
        if inv_closure != "savings_driven":
            raise ValueError(
                f"adaptation investment is only modelled under the savings_driven closure (it "
                f"crowds out ordinary investment from the same savings pool); got {inv_closure!r}."
            )
        if adapt_gamma is None:
            raise ValueError("adaptation investment needs a sectoral composition adapt_gamma.")
    adapt_active = cal.has_investment and adapt_amount > 0.0 and adapt_gamma is not None
    if adapt_active:
        d_adapt = adapt_amount * (adapt_gamma - cal.inv_gamma) / p  # [i]; p·d_adapt = 0
        c_adapt = float(cc_eff @ (leontief @ d_adapt)) if recycles else 0.0
    else:
        d_adapt = np.zeros(ns)
        c_adapt = 0.0

    # Labour-market closure (Phase 5d.4). Default: flexible wage, full employment — factor income
    # values labour at the full fixed endowment. Wage-floor alternative: when a floor is configured
    # and binds (the residual system pins w[LAB]=floor; see ``residuals``), the household earns only
    # its EMPLOYED labour (labour DEMAND, not supply), so it is poorer and there is unemployment.
    # Employed labour scales linearly with factor income (the whole quantity chain does at fixed
    # prices), so factor income is a scalar fixed point: FI = capital income + w_L·L_emp(FI). We
    # solve it by iteration (a contraction with ratio w_L·ℓ < 1) rather than re-deriving all six
    # income branches — each pass reuses ``_close`` unchanged. With no floor (or a slack floor)
    # the loop runs once with L_emp = the endowment, i.e. exactly the pre-5d.4 behaviour.
    lab = cal.factors.index("LAB") if "LAB" in cal.factors else None
    floor_active = labour_floor is not None and lab is not None

    def _close(factor_income: float):
        """Given a factor income, derive (FD, GD, ID, X, F, income, gov_income, savings,
        fiscal_balance). Everything here is the pre-5d.4 body, made a function of factor income so
        the labour-floor fixed point can iterate it."""
        if not cal.has_government:
            # Pre-5d.1/5d.2 behaviour when no accounts are declared, bit-for-bit unchanged: carbon
            # revenue recycles straight to the household. With an investment account, the fixed
            # point is the same shape — demand is still linear in income:
            #   savings_driven: per-unit-income demand u = ((1−s)γ + s·inv_gamma)/p, I = FI/(1−k);
            #   fixed_real: demand = γ(I − p·INV0)/p + INV0, so R = k·(I − pI0) + cInv and
            #     I = (FI − k·pI0 + cInv)/(1−k) with pI0 = p·INV0, cInv = cc·L·INV0.
            if not cal.has_investment:
                k = float(cc_eff @ (leontief @ demand_per_income)) if recycles else 0.0
                if strict and k >= 1.0 - 1e-12:
                    # Runaway recycling (revenue ≥ income) AT THE ACCEPTED EQUILIBRIUM — refuse
                    # rather than return numbers (review P2).
                    raise ValueError(f"revenue-recycling fixed point diverges (k={k:.3f} ≥ 1)")
                # A trial price vector during the solve can hit k≥1 even when a valid equilibrium
                # (k<1) exists elsewhere; in non-strict (exploratory) mode use a SMOOTH floor on
                # (1−k): identity for 1−k ≥ δ, asymptotes to δ as 1−k → −∞, keeping income finite
                # and the residual C¹-continuous — not a flat plateau.
                income = factor_income / _safe_denom(1.0 - k)
                FD = income * demand_per_income
                ID = np.zeros(ns)
                savings = 0.0
            elif inv_closure == "savings_driven":
                u = ((1.0 - s) * cal.gamma + s * cal.inv_gamma) / p
                k = float(cc_eff @ (leontief @ u)) if recycles else 0.0
                if strict and k >= 1.0 - 1e-12:
                    raise ValueError(f"revenue-recycling fixed point diverges (k={k:.3f} ≥ 1)")
                # Adaptation adds a fixed zero-sum re-allocation to investment demand: it generates
                # carbon revenue c_adapt that recycles into income but does NOT change the total
                # investment budget (p·d_adapt = 0).
                income = (factor_income + c_adapt) / _safe_denom(1.0 - k)
                savings = s * income
                FD = (1.0 - s) * income * demand_per_income
                ID = savings * cal.inv_gamma / p + d_adapt
            else:  # fixed_real
                ID = cal.INV0.copy()
                p_inv = float(np.dot(p, ID))
                if recycles:
                    k = float(cc_eff @ (leontief @ demand_per_income))
                    c_inv = float(cc_eff @ (leontief @ ID))
                else:
                    k, c_inv = 0.0, 0.0
                if strict and k >= 1.0 - 1e-12:
                    raise ValueError(f"revenue-recycling fixed point diverges (k={k:.3f} ≥ 1)")
                income = (factor_income - k * p_inv + c_inv) / _safe_denom(1.0 - k)
                savings = p_inv  # residual saving that finances the fixed real investment
                if strict and income - p_inv <= 0:
                    raise ValueError(
                        f"fixed_real investment ({p_inv:.6g}) exceeds household income "
                        f"({income:.6g}); no consumption left. Lower the shock or use "
                        "savings_driven."
                    )
                FD = (income - p_inv) * demand_per_income
            GD = np.zeros(ns)
            gov_income = 0.0
            fiscal_balance = 0.0
        else:
            if gov_closure not in ("balanced_budget", "deficit_financed"):
                raise ValueError(
                    f"unsupported gov_closure {gov_closure!r}; 5d.1/5d.7 implement "
                    "'balanced_budget' and 'deficit_financed'."
                )
            # Household income is factor income net of the benchmark direct tax. The tax is a RATE
            # on factor income (rate·w·FF), so the benchmark government replicates and homogeneity
            # survives. Under deficit_financed the deficit is financed by CROWDING OUT investment
            # (below), not by the household absorbing the residual (review P1 round 12).
            tax = cal.gov_tax_rate0 * factor_income
            gov_demand_per_income = cal.gov_gamma / p

            if gov_closure == "balanced_budget":
                # Government spending exactly exhausts its income each period (fiscal_balance ≡ 0).
                # Who receives carbon revenue R is carbon_revenue_recipient (review P1 round 13):
                if carbon_revenue_recipient == "government":
                    # R funds the government alongside the tax; the household pays the tax and gets
                    # no carbon revenue. gov_income = (tax + R) with R recycled into gov demand
                    # (a fixed point because gov spending itself is a taxed demand: coefficient kg).
                    income = factor_income - tax
                    FD, ID, savings = _hh_demand(
                        income, demand_per_income, s, cal, p, inv_closure, d_adapt, strict
                    )
                    if recycles:
                        r0 = float(cc_eff @ (leontief @ (FD + ID)))
                        kg = float(cc_eff @ (leontief @ gov_demand_per_income))
                    else:
                        r0, kg = 0.0, 0.0
                    if strict and kg >= 1.0 - 1e-12:
                        raise ValueError(
                            f"government revenue-recycling fixed point diverges (kg={kg:.3f} ≥ 1)"
                        )
                    gov_income = (tax + r0) / _safe_denom(1.0 - kg)
                    GD = gov_income * gov_demand_per_income
                elif carbon_revenue_recipient == "household":
                    # R recycles to the HOUSEHOLD (as under lump-sum); the government spends only
                    # the tax, so GD = tax·γ^g. Household income fixed point I = FI − tax + R_hh,
                    # R_hh = cc_eff·L·(FD+ID+GD). Total household demand FD+ID = I·u + d0 from the
                    # SHARED decomposition (correct for savings_driven AND fixed_real, review P1
                    # round 14), so R_hh = I·k + c with k = cc_eff·L·u and c = cc_eff·L·(d0 + GD),
                    # and I = (FI − tax + c)/(1 − k).
                    gov_income = tax
                    GD = gov_income * gov_demand_per_income
                    if recycles:
                        u, d0 = _demand_decomposition(cal, p, s, inv_closure, d_adapt)
                        k = float(cc_eff @ (leontief @ u))
                        c_gd = float(cc_eff @ (leontief @ (d0 + GD)))
                    else:
                        k, c_gd = 0.0, 0.0
                    if strict and k >= 1.0 - 1e-12:
                        raise ValueError(
                            f"household revenue-recycling fixed point diverges (k={k:.3f} ≥ 1)"
                        )
                    income = (factor_income - tax + c_gd) / _safe_denom(1.0 - k)
                    FD, ID, savings = _hh_demand(
                        income, demand_per_income, s, cal, p, inv_closure, d_adapt, strict
                    )
                else:
                    raise ValueError(
                        f"unsupported carbon_revenue_recipient {carbon_revenue_recipient!r}; use "
                        "'government' (default) or 'household'."
                    )
                fiscal_balance = 0.0
            else:  # deficit_financed (Phase 5d.7, redesigned 2026-07-26)
                # Government spends a FIXED REAL amount (its benchmark real level gov_income0·γ^g),
                # financed by the direct tax PLUS a genuine deficit that draws on the national
                # savings pool — i.e. the deficit CROWDS OUT private investment 1-for-1 (standard
                # static financing closure). This is a real financing account, not the earlier
                # household-residual pass-through (which cancelled the tax out of demand, and was
                # economically a variable lump-sum transfer — review P1, 2026-07-26).
                #
                #   household disposable income  I = factor_income − tax + R_household   (tax BITES)
                #   government budget            gov_income = tax + R_gov
                #   deficit                      def = p·GD − gov_income   (financed from savings)
                #   investment                   p·ID = private_savings − def   (crowding out)
                #
                # Deficit financing therefore needs a savings-investment account to draw on — the
                # financing channel. Without one there is nowhere for the deficit to come from, so
                # the closure is rejected (the engine gates this too).
                if not cal.has_investment:
                    raise ValueError(
                        "deficit_financed needs a savings-investment (SAVINV) account to finance "
                        "the deficit — the deficit crowds out private investment from the national "
                        "savings pool; with no such account there is no financing channel."
                    )
                if inv_closure != "savings_driven":
                    raise ValueError(
                        "deficit_financed is modelled under the savings_driven investment closure "
                        "(the deficit crowds out savings-financed investment); got "
                        f"{inv_closure!r}."
                    )
                GD = cal.gov_income0 * cal.gov_gamma  # fixed real quantity (price-independent)
                p_gd = float(np.dot(p, GD))  # nominal government spending
                inv_dir = cal.inv_gamma / p  # nominal 1 of investment buys this quantity vector
                # Who receives carbon revenue R is a SEPARATE choice (carbon_revenue_recipient),
                # NOT implied by the fiscal closure (review P1, 2026-07-27). Two closed forms:
                #
                #  government (default): R funds the government alongside the tax, so
                #     gov_income = tax + R,  def = p·GD − (tax + R),  p·ID = s_priv − def
                #     household  I = FI − tax  (no carbon revenue — it is the government's).
                #     Here R depends on ID (revenue base) and ID depends on R (via the deficit), a
                #     scalar linear fixed point in nominal net investment.
                #
                #  household: R recycles to the household (as under lump_sum), the government is
                #     financed by the tax alone, so def = p·GD − tax is exogenous given prices:
                #     I = FI − tax + R_hh,  R_hh = cc_eff·L·(FD + ID + GD),  p·ID = s_priv − def.
                #
                # Both keep GD fixed real and finance the deficit by crowding out investment.
                if carbon_revenue_recipient == "government":
                    income = factor_income - tax  # household gets no carbon revenue
                    private_savings = s * income if cal.has_investment else 0.0
                    FD = (1.0 - s) * income * demand_per_income
                    # Solve p·ID (nominal net investment) from the R/deficit fixed point. With
                    #   R = a·pID + b,  a = cc_eff·L·inv_dir,  b = cc_eff·L·(FD + d_adapt + GD)
                    #   pID = s_priv − (p·GD − tax − R) = s_priv − p·GD + tax + R
                    # ⇒ pID·(1 − a) = s_priv − p·GD + tax + b.
                    if recycles:
                        a = float(cc_eff @ (leontief @ inv_dir))
                        b = float(cc_eff @ (leontief @ (FD + d_adapt + GD)))
                    else:
                        a, b = 0.0, 0.0
                    if strict and a >= 1.0 - 1e-12:
                        raise ValueError(
                            f"deficit-financed (gov recipient) fixed point diverges (a={a:.3f} ≥ 1)"
                        )
                    p_id_net = (private_savings - p_gd + tax + b) / _safe_denom(1.0 - a)
                    R = a * p_id_net + b
                    gov_income = tax + R
                elif carbon_revenue_recipient == "household":
                    # Government financed by the tax alone; deficit exogenous given prices.
                    gov_income = tax
                    deficit = p_gd - gov_income
                    if recycles:
                        u_income = (1.0 - s) * cal.gamma / p + s * inv_dir  # ∂(FD+ID)/∂I
                        const_dem = GD + d_adapt - deficit * inv_dir
                        k = float(cc_eff @ (leontief @ u_income))
                        c0 = float(cc_eff @ (leontief @ const_dem))
                    else:
                        k, c0 = 0.0, 0.0
                    if strict and k >= 1.0 - 1e-12:
                        raise ValueError(
                            f"deficit-financed income fixed point diverges (k={k:.3f} ≥ 1)"
                        )
                    income = (factor_income - tax + c0) / _safe_denom(1.0 - k)
                    private_savings = s * income if cal.has_investment else 0.0
                    FD = (1.0 - s) * income * demand_per_income
                    p_id_net = private_savings - deficit
                else:
                    raise ValueError(
                        f"unsupported carbon_revenue_recipient {carbon_revenue_recipient!r}; use "
                        "'government' (default) or 'household'."
                    )
                # Crowding-out feasibility: the NET investment budget (after the deficit) must be
                # positive, the adaptation earmark cannot exceed it, and the per-sector
                # investment demand must be non-negative componentwise (review P1, 2026-07-27: the
                # earlier guard compared adaptation to GROSS savings, so def could drive a sector
                # sector's ID negative while the solver still reported a machine-zero residual).
                if strict and p_id_net <= 0:
                    raise ValueError(
                        f"deficit exhausts private investment (net budget {p_id_net:.6g} ≤ 0); "
                        "fixed real government spending is infeasible at this tax rate."
                    )
                if strict and adapt_active and adapt_amount > p_id_net + 1e-12:
                    raise ValueError(
                        f"adaptation investment ({adapt_amount:.6g}) exceeds the NET investment "
                        f"budget after the deficit ({p_id_net:.6g}); it cannot crowd out more than "
                        "what is left after financing the deficit."
                    )
                ID = p_id_net * inv_dir + d_adapt  # crowded-out investment (adaptation preserved)
                if strict and np.any(ID < -1e-12):
                    raise ValueError(
                        f"deficit-financed investment demand is negative in some sector "
                        f"(min {float(ID.min()):.3e}); the deficit + adaptation over-committed the "
                        "investment budget."
                    )
                savings = private_savings
                fiscal_balance = gov_income - p_gd  # < 0 ⇒ deficit financed from savings

        X = leontief @ (FD + GD + ID)  # goods-market clearing
        # Total VA payment per sector = pv · (VA quantity). VA quantity per unit output is
        # cal.va_share (flat) or the price-responsive KL quantity per unit output (energy nest).
        va_cost = pv * va_qty_per_x * X  # [i]
        F = _factor_demand(cal, w, pv, va_cost)  # [f,i]
        return FD, GD, ID, X, F, income, gov_income, savings, fiscal_balance

    # Labour-floor fixed point. Full employment: factor income = w·endowment. Floor binding: labour
    # income counts EMPLOYED labour F[LAB].sum(), so we iterate factor income to consistency (a
    # contraction — each pass shrinks the gap by w_L·(∂L_emp/∂FI) < 1). Not floor-active ⇒ one pass
    # with the full endowment, identical to before.
    factor_income = float(np.dot(w, cal.endowment))
    result = _close(factor_income)
    if floor_active:
        capital_income = factor_income - w[lab] * cal.endowment[lab]  # non-labour factor income
        for _ in range(100):
            employed = float(result[4][lab, :].sum())  # F[LAB].sum() at the current guess
            fi_new = capital_income + w[lab] * employed
            if abs(fi_new - factor_income) <= 1e-13 * max(1.0, abs(fi_new)):
                factor_income = fi_new
                result = _close(factor_income)
                break
            factor_income = fi_new
            result = _close(factor_income)
    FD, GD, ID, X, F, income, gov_income, savings, fiscal_balance = result
    carbon_revenue = float(cc_eff @ X)  # cc_eff = cc (flat) or energy-weighted (nest)
    # Savings-investment identity check (strict mode; Phase 5d.2 Tier 2): under savings_driven,
    # nominal investment must equal household savings exactly. Adaptation (Phase 5d.6) preserves
    # this exactly — d_adapt is nominally zero-sum, so p·ID = savings still, by construction.
    # EXCEPTION: under deficit_financed, the government deficit crowds out investment, so
    # p·ID = savings − deficit by design; that closure's own guard (p_id_net > 0) covers it instead.
    if (
        strict
        and cal.has_investment
        and inv_closure == "savings_driven"
        and gov_closure != "deficit_financed"
    ):
        resid = float(np.dot(p, ID)) - savings
        if abs(resid) > 1e-9 * max(1.0, abs(savings)):  # pragma: no cover - guards the closed form
            raise ValueError(f"savings-investment identity not satisfied (residual {resid:.3e}).")
    # Adaptation crowding-out guard (Phase 5d.6): the earmarked adaptation cannot exceed the total
    # investment budget, or ordinary investment would go negative (a nonsensical over-earmark).
    # deficit_financed has its OWN (stricter, net-of-deficit + componentwise) guard inline above —
    # here ``savings`` is the GROSS budget, the right denominator only when there is no deficit.
    if (
        strict
        and adapt_active
        and gov_closure != "deficit_financed"
        and adapt_amount > savings + 1e-12
    ):
        raise ValueError(
            f"adaptation investment ({adapt_amount:.6g}) exceeds total investment ({savings:.6g}); "
            "it cannot crowd out more than the whole budget. Lower the adaptation amount."
        )
    # Unemployment (Phase 5d.4): labour supply less employed labour (0 unless a floor binds).
    unemployment = 0.0
    if lab is not None:
        unemployment = float(cal.endowment[lab] - F[lab, :].sum())
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
        ID=ID,
        savings=savings,
        unemployment=unemployment,
        adaptation_investment=adapt_amount if adapt_active else 0.0,
    )


def residuals(
    cal: CalibratedModel,
    z: np.ndarray,
    *,
    carbon_cost: np.ndarray | None = None,
    recycling: str = "none",
    drop_factor: int = 0,
    gov_closure: str = "balanced_budget",
    inv_closure: str = "savings_driven",
    carbon_revenue_recipient: str = "government",
    labour_floor: float | None = None,
    adapt_amount: float = 0.0,
    adapt_gamma: np.ndarray | None = None,
    productivity: np.ndarray | None = None,
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
    selects the government's financing closure when ``cal.has_government`` (Phase 5d.1);
    ``inv_closure`` selects the savings-investment closure when ``cal.has_investment`` (Phase
    5d.2: ``savings_driven`` or ``fixed_real``). Neither account adds a new unknown/equation:
    government demand ``GD`` and investment demand ``ID`` are both algebraic functions of prices
    exactly like household demand ``FD`` (the savings-investment identity is substituted in
    closed form), so the system stays square in ``(p, w)`` with no new residual line — the
    square-count re-derivation the phase plan required is therefore trivial: the count is
    unchanged, verified by the existing square-system test.

    ``labour_floor`` (Phase 5d.4) selects the wage-floor labour-market closure. When set, the
    **LAB factor-clearing row is replaced by the wage pin** ``w[LAB] − floor = 0``: the labour
    market no longer clears on quantity (demand may fall short of supply — the shortfall is
    reported as ``unemployment``), it clears on the pinned wage instead. The system stays exactly
    square — one clearing row swapped for one pin row, same count. The engine imposes this only
    in the regime where the floor genuinely binds (the unconstrained wage would sit below it),
    solving the default full-employment system first (see ``engine._solve``); off-regime the floor
    is passed as ``None`` and this is the pre-5d.4 residual exactly.

    ``z`` accepts an object-dtype array (pyomo vars) so the same residual builds the IPOPT model;
    it uses only +, −, ×, ÷ and np.dot-free elementwise algebra where that matters.
    """
    ns = len(cal.sectors)
    nf = len(cal.factors)
    p = z[:ns]
    w = z[ns : ns + nf]
    cc = np.zeros(ns) if carbon_cost is None else np.asarray(carbon_cost, dtype=float)
    lab = cal.factors.index("LAB") if "LAB" in cal.factors else None

    state = derive_state(
        cal,
        np.asarray(p, dtype=float),
        np.asarray(w, dtype=float),
        carbon_cost=cc,
        recycling=recycling,
        gov_closure=gov_closure,
        inv_closure=inv_closure,
        carbon_revenue_recipient=carbon_revenue_recipient,
        labour_floor=labour_floor,
        adapt_amount=adapt_amount,
        adapt_gamma=adapt_gamma,
        productivity=productivity,
    )

    # Per-sector productivity multiplier θ[i] (Phase 6.4 GE tier): a sector with productivity θ
    # needs 1/θ times its whole input bundle per unit output, so its zero-profit unit cost divides
    # by θ. θ=1 (the default) is byte-identical to the pre-6.4 residual — the ``inv_theta`` factors
    # are exactly 1 — so replication / homogeneity / Walras are untouched.
    inv_theta = (
        np.ones(ns) if productivity is None else 1.0 / np.asarray(productivity, dtype=float)
    )

    res = []
    if cal.has_energy_nest:
        # Zero-profit with the KL-E-M nest (Phase 5d.5): p[i] = px[i]/θ[i], the nest's output unit
        # cost scaled by the productivity requirement. Carbon is a per-OUTPUT wedge added to px
        # (review remediation 2026-07-26) — NOT an add-on inside the nest — so the emissions/revenue
        # contract is identical to the flat model.
        from cge.engines.cge_static.energy_nest import nest_unit_cost

        px = nest_unit_cost(cal.energy_nest, np.asarray(p, dtype=float), state.pv, cc)
        for i in range(ns):
            res.append(p[i] - px[i] * inv_theta[i])
    else:
        # Flat model: p[i] = (Σ_j ax[j,i]·p[j] + va_share[i]·pv[i] + cc[i])/θ[i]. (Object-dtype
        # safe for the dormant pyomo hook.)
        for i in range(ns):
            intermediate = sum(cal.ax[j, i] * p[j] for j in range(ns))
            unit_cost = intermediate + cal.va_share[i] * state.pv[i] + cc[i]
            res.append(p[i] - unit_cost * inv_theta[i])
    # Factor clearing (drop one by Walras). Under a binding wage floor (Phase 5d.4), the LAB row
    # becomes the wage pin w[LAB] = floor instead of quantity-clearing — labour demand ≤ supply is
    # then slack, the gap reported as unemployment.
    #
    # The dropped (Walras) market MUST be a non-LAB factor when the floor is active: the LAB row is
    # the wage pin and must be present, so dropping LAB would silently discard the pin and let the
    # wage fall below the floor unenforced (review P1, 2026-07-26 — a valid LAB-only model reported
    # labour_floor_bound=true while the actual wage sat below the floor). If LAB is the only factor
    # there is no non-LAB market to drop as the numéraire-market, so the wage-floor closure is not
    # well posed and is rejected outright.
    if labour_floor is not None and lab is not None:
        non_lab = [f for f in range(nf) if f != lab]
        if not non_lab:
            raise ValueError(
                "the wage-floor labour-market closure needs at least one non-LAB factor to drop as "
                "the Walras market (the LAB row is the wage pin and cannot also be the dropped "
                f"market); this model has only {cal.factors} — reject or add a capital factor."
            )
        if drop_factor == lab:
            drop_factor = non_lab[0]  # never drop the LAB (pin) row when the floor is active
    for f in range(nf):
        if f == drop_factor:
            continue
        if labour_floor is not None and f == lab:
            res.append(w[lab] - labour_floor)
        else:
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
