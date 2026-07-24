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


def _leontief_and_va(cal: CalibratedModel, p: np.ndarray, pv: np.ndarray, cc: np.ndarray):
    """Return ``(leontief, va_qty_per_x, cc_eff)`` — the (I − A(p))⁻¹ intermediate-inverse, the
    value-added (KL) quantity per unit output, and an **effective per-output carbon cost**.

    **Flat model** (no energy nest): A = the fixed Leontief ``cal.ax``, VA quantity per unit output
    = ``cal.va_share``, and cc_eff = cc (the carbon cost is already a per-output add-on) — all
    price-independent and bit-identical to the pre-5d.5 code.

    **Energy nest** (Phase 5d.5): intermediate demand is price-responsive (energy substitutes as
    its carbon-inclusive price moves), so A(p)[j,i] = intermediate use of commodity j per unit
    output i comes from ``nest_demands`` at unit output; VA quantity per unit output = the KL
    quantity per unit output. The carbon cost attaches to the *energy commodities* (it raises their
    price inside the nest), so the carbon revenue per unit of a sector's output is
    cc_eff[i] = Σ_{j∈energy} cc[j]·a_energy[j,i] — a linear functional of output, so the existing
    recycling/government fixed points (using cc_eff @ leontief @ demand) carry over unchanged."""
    ns = len(cal.sectors)
    if not cal.has_energy_nest:
        leontief = np.linalg.inv(np.eye(ns) - cal.ax)
        return leontief, cal.va_share, cc
    from cge.engines.cge_static.energy_nest import nest_demands

    nest = cal.energy_nest
    unit_x = np.ones(ns)
    energy_use, materials_use, kl_qty = nest_demands(nest, p, pv, cc, unit_x)  # per unit output
    a = np.zeros((ns, ns))  # A(p)[j, i] = commodity j per unit output i
    cc_eff = np.zeros(ns)
    for k, j in enumerate(nest.energy_idx):
        a[j, :] += energy_use[k, :]
        cc_eff += cc[j] * energy_use[k, :]  # carbon revenue per unit output from energy commodity j
    for k, j in enumerate(nest.mat_idx):
        a[j, :] += materials_use[k, :]
    leontief = np.linalg.inv(np.eye(ns) - a)
    return leontief, kl_qty, cc_eff


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
    inv_closure: str = "savings_driven",
    labour_floor: float | None = None,
    adapt_amount: float = 0.0,
    adapt_gamma: np.ndarray | None = None,
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
    leontief, va_qty_per_x, cc_eff = _leontief_and_va(cal, p, pv, cc)
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
            if gov_closure != "balanced_budget":
                raise ValueError(
                    f"unsupported gov_closure {gov_closure!r} (5d.1 only implements "
                    "'balanced_budget'; 'deficit_financed' is Phase 5d.7)"
                )
            # Household income is factor income net of the benchmark direct tax — carbon revenue no
            # longer passes through it. The tax is levied as a RATE on factor income (rate·w·FF),
            # so the benchmark government replicates exactly and homogeneity survives (calibrate).
            tax = cal.gov_tax_rate0 * factor_income
            income = factor_income - tax
            if not cal.has_investment:
                FD = income * demand_per_income
                ID = np.zeros(ns)
                savings = 0.0
            elif inv_closure == "savings_driven":
                savings = s * income
                FD = (1.0 - s) * income * demand_per_income
                # Adaptation crowds out ordinary investment (zero-sum re-allocation d_adapt); the
                # household's income is already fixed by the tax here, so only ID's split moves.
                ID = savings * cal.inv_gamma / p + d_adapt
            else:  # fixed_real
                ID = cal.INV0.copy()
                savings = float(np.dot(p, ID))
                if strict and income - savings <= 0:
                    raise ValueError(
                        f"fixed_real investment ({savings:.6g}) exceeds household disposable "
                        f"income ({income:.6g}); no consumption left."
                    )
                FD = (income - savings) * demand_per_income
            gov_demand_per_income = cal.gov_gamma / p  # GD = gov_income · gov_demand_per_income
            if recycles:
                # Revenue from the (price-fixed) household + investment demand.
                r0 = float(cc_eff @ (leontief @ (FD + ID)))
                kg = float(cc_eff @ (leontief @ gov_demand_per_income))  # marginal from Σgov spend
            else:
                r0, kg = 0.0, 0.0
            if strict and kg >= 1.0 - 1e-12:
                raise ValueError(
                    f"government revenue-recycling fixed point diverges (kg={kg:.3f} ≥ 1)"
                )
            gov_income = (tax + r0) / _safe_denom(1.0 - kg)
            GD = gov_income * gov_demand_per_income
            fiscal_balance = 0.0  # balanced_budget: spending exhausts income, by construction

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
    if strict and cal.has_investment and inv_closure == "savings_driven":
        resid = float(np.dot(p, ID)) - savings
        if abs(resid) > 1e-9 * max(1.0, abs(savings)):  # pragma: no cover - guards the closed form
            raise ValueError(f"savings-investment identity not satisfied (residual {resid:.3e}).")
    # Adaptation crowding-out guard (Phase 5d.6): the earmarked adaptation cannot exceed the total
    # investment budget, or ordinary investment would go negative (a nonsensical over-earmark).
    if strict and adapt_active and adapt_amount > savings + 1e-12:
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
    labour_floor: float | None = None,
    adapt_amount: float = 0.0,
    adapt_gamma: np.ndarray | None = None,
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
        labour_floor=labour_floor,
        adapt_amount=adapt_amount,
        adapt_gamma=adapt_gamma,
    )

    res = []
    if cal.has_energy_nest:
        # Zero-profit with the KL-E-M nest (Phase 5d.5): p[i] = px[i], the nest's output unit cost
        # (carbon attaches to energy commodities inside the nest, not as a flat per-output add-on).
        from cge.engines.cge_static.energy_nest import nest_unit_cost

        px = nest_unit_cost(cal.energy_nest, np.asarray(p, dtype=float), state.pv, cc)
        for i in range(ns):
            res.append(p[i] - px[i])
    else:
        # Flat model: p[i] = Σ_j ax[j,i]·p[j] + va_share[i]·pv[i] + carbon cost. (Object-dtype
        # safe for the dormant pyomo hook.)
        for i in range(ns):
            intermediate = sum(cal.ax[j, i] * p[j] for j in range(ns))
            res.append(p[i] - (intermediate + cal.va_share[i] * state.pv[i] + cc[i]))
    # Factor clearing (drop one by Walras). Under a binding wage floor (Phase 5d.4), the LAB row
    # becomes the wage pin w[LAB] = floor instead of quantity-clearing — labour demand ≤ supply is
    # then slack, the gap reported as unemployment.
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
