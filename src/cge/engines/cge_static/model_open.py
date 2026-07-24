"""The OPEN-economy static CGE as a residual system (Phase 5 — Armington/CET).

Small open economy with an Armington import composite and CET export transformation
[Hosoe2010, ch. 7]. World prices are fixed (= 1 in foreign currency); the exchange rate ``er`` is
the price of foreign currency. Foreign savings ``Sf`` is fixed at its benchmark level (a standard
closure), so the trade balance clears through relative prices.

**Unknowns** ``z = [pd (ns), pq (ns), w (nf), er]``:
- ``pd`` — domestic-sales price of each commodity;
- ``pq`` — Armington composite price (what intermediates + the household pay);
- ``w`` — factor prices; ``er`` — exchange rate.

Everything else is derived by CES/CET **dual** (price) identities and Shephard/Hotelling:
- import/export prices ``pm = pe = er`` (world price 1);
- **Armington price** (CES cost fn): ``pq = (1/A)[δ^σ pd^{1-σ} + (1-δ)^σ pm^{1-σ}]^{1/(1-σ)}``;
- **CET price** (revenue fn): ``pz = (1/B)[ξ^{-Ω} pd^{1+Ω} + (1-ξ)^{-Ω} pe^{1+Ω}]^{1/(1+Ω)}``;
- **zero profit** ties pz to input costs: ``pz_i = Σ_j ax[j,i] pq_j + va_share_i·pv_i + cc_i``;
- household demand ``FD_i = γ_i I / pq_i`` with income ``I = Σ_f w_f FF_f + carbon revenue``;
- Armington/CET quantity splits (D, M, E) from Shephard/Hotelling on Q_i and Z_i.

**Residuals — an exactly square system in the 2·ns + nf + 1 unknowns:**
- ns Armington-price identities (pin ``pq``);
- ns zero-profit (pz) identities (pin ``pd`` via ``pz(pd,pe)``);
- (nf−1) factor-market clearing (one dropped by Walras' law);
- 1 trade balance ``Σ pm_i M_i − Σ pe_i E_i − er·Sf = 0`` (pins ``er``);
- 1 numéraire ``Π_i pq_i^{γ_i} = 1`` (the household CPI over the composite good).
Total: 2·ns + (nf−1) + 1 + 1 = **2·ns + nf + 1 = n_unknowns**.

Composite-market clearing ``Q_i = Σ_j ax[i,j] Z_j + FD_i`` is **not** an independent residual: it is
solved *algebraically* inside ``_quantities`` (``Q = (I − ax·diag(ratio))⁻¹ FD``, ``Z = ratio·Q``),
so the market clears by construction. Adding it as a residual row would be tautological (identically
zero) and overdetermine the system on paper. ``pz`` is a *derived* price (the CET dual of ``pd``),
not an independent unknown — hence 2·ns price unknowns, not 3·ns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cge.engines.cge_static.calibrate_open import OpenCalibratedModel
from cge.engines.cge_static.model import _safe_denom


@dataclass(frozen=True)
class OpenModelState:
    pd: np.ndarray  # domestic-sales prices [i]
    pq: np.ndarray  # composite prices [i]
    pz: np.ndarray  # activity output prices [i]
    w: np.ndarray  # factor prices [f]
    er: float  # exchange rate
    Z: np.ndarray  # activity output [i]
    D: np.ndarray  # domestic sales [i]
    E: np.ndarray  # exports [i]
    M: np.ndarray  # imports [i]
    Q: np.ndarray  # composite supply [i]
    FD: np.ndarray  # household final demand [i]
    F: np.ndarray  # factor demand [f, i]
    income: float
    carbon_revenue: float
    # Government account (Phase 5d.1) — zero/empty when cal.has_government is False (pre-5d.1
    # behaviour preserved exactly; see the closed model's ModelState for the same convention).
    GD: np.ndarray  # government final demand on the composite [i]
    gov_income: float  # government income (benchmark tax + carbon revenue)
    fiscal_balance: float  # income − spending (≡0 under balanced_budget)
    # Savings-investment account (Phase 5d.2). Zeros when cal.has_investment is False.
    ID: np.ndarray  # investment final demand on the composite [i]
    savings: float  # household savings (investment = savings + er·Sf under savings_driven)


def _va_unit_cost(cal: OpenCalibratedModel, w: np.ndarray) -> np.ndarray:
    """VA unit cost per activity. σ_va = 1 ⇒ Cobb-Douglas; σ_va ≠ 1 ⇒ CES (per sector)."""
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


def _factor_demand(cal: OpenCalibratedModel, w: np.ndarray, pv: np.ndarray, va_cost: np.ndarray):
    """Factor demand by Shephard on the VA cost. CD or CES per sector (mirrors the closed model)."""
    ns, nf = len(cal.sectors), len(cal.factors)
    F = np.empty((nf, ns))
    for i in range(ns):
        s = cal.va_elast[i]
        if abs(s - 1.0) < 1e-12:
            F[:, i] = cal.beta[:, i] * va_cost[i] / w
        else:
            d = cal.va_ces_share[:, i]
            unit = (1.0 / cal.av[i]) * (pv[i] * cal.av[i]) ** s * d**s * w ** (-s)
            F[:, i] = unit * (va_cost[i] / pv[i])
    return F


def _armington_price(cal: OpenCalibratedModel, pd: np.ndarray, pm: np.ndarray) -> np.ndarray:
    """CES cost function for the composite: pq = (1/A)[δ^σ pd^{1-σ}+(1-δ)^σ pm^{1-σ}]^{1/(1-σ)}.
    A non-traded good (δ=1, share_m=0) collapses to pq = pd/A. The import term is masked where
    share_m=0 so ``0**σ`` never produces a spurious 0/NaN (structural zeros are supported)."""
    sigma = cal.arm_elast
    e = 1.0 - sigma
    # Use a safe base (1.0 where the share is 0) so the *unselected* np.where branch never evaluates
    # a singular power; the mask then zeroes that term. Keeps output warning-free for structural
    # zeros (a non-traded good has share_m=0).
    share_m = np.where(cal.arm_share_m > 0, cal.arm_share_m, 1.0)
    m_term = np.where(cal.arm_share_m > 0, share_m**sigma * pm**e, 0.0)
    inner = cal.arm_share_d**sigma * pd**e + m_term
    return (1.0 / cal.arm_scale) * np.power(inner, 1.0 / e)


def _cet_price(cal: OpenCalibratedModel, pd: np.ndarray, pe: np.ndarray) -> np.ndarray:
    """CET revenue function for output: pz = (1/B)[ξ^{-Ω} pd^{1+Ω}+(1-ξ)^{-Ω} pe^{1+Ω}]^{1/(1+Ω)}
    (verified via Hotelling; the negative share exponents distinguish the convex CET frontier).
    A non-exporting good (share_e=0) collapses to pz = pd/B; the export term is **masked** where
    share_e=0 so ``0**(-Ω)`` never produces ``inf`` (review P1: structural zeros are supported)."""
    omega = cal.cet_elast
    e = 1.0 + omega
    share_e = np.where(cal.cet_share_e > 0, cal.cet_share_e, 1.0)  # safe base; masked below
    e_term = np.where(cal.cet_share_e > 0, share_e ** (-omega) * pe**e, 0.0)
    inner = cal.cet_share_d ** (-omega) * pd**e + e_term
    return (1.0 / cal.cet_scale) * np.power(inner, 1.0 / e)


def derive_open_state(
    cal: OpenCalibratedModel,
    pd: np.ndarray,
    pq: np.ndarray,
    w: np.ndarray,
    er: float,
    *,
    carbon_cost: np.ndarray | None = None,
    recycling: str = "lump_sum",
    strict: bool = False,
    gov_closure: str = "balanced_budget",
    inv_closure: str = "savings_driven",
) -> OpenModelState:
    """Close the open model at prices (pd, pq, w, er): derive all quantities by CES/CET duals.

    ``strict`` controls the recycling k≥1 guard. During solver iteration (``strict=False``) a
    trial price vector may momentarily give k≥1 even when a valid equilibrium (k<1) exists
    elsewhere; raising there would make the residual function discontinuously unavailable and can
    abort a solve that starts at benchmark prices (review P2). So in non-strict mode we **clamp**
    the income multiplier to a large-but-finite value (steering the solver away smoothly) instead
    of raising. On the **final accepted equilibrium** the engine calls with ``strict=True``, which
    raises if k≥1 — so a genuinely infeasible equilibrium is still refused, never returned.

    **Government account (Phase 5d.1, ``cal.has_government``):** exactly the closed model's
    design, with the open income identity. The household pays the benchmark direct tax
    (rate·factor income); the GOVERNMENT collects carbon revenue and spends tax+revenue on its
    own CD demand vector under ``balanced_budget``. Z is linear in the total demand vector at
    fixed prices, so the government fixed point solves in closed form:
    gov_income = (T + R0)/(1 − kg), R0 = cc·Z(FD), kg = cc·Z(gov_gamma/pq).

    **Savings-investment account (Phase 5d.2, ``cal.has_investment``):** the genuine open-economy
    closure change — the er·Sf capital transfer RE-ROUTES from household income into the
    investment pool: household income = factor income − tax (no Sf), and investment = household
    savings + er·Sf (foreign savings finance domestic investment, not consumption). Closures:
    ``savings_driven`` (default; S = s·I_hh, nominal investment = S + er·Sf exactly) and
    ``fixed_real`` (investment quantities pinned at INV0; household savings adjust residually,
    S = pq·INV0 − er·Sf)."""
    ns = len(cal.sectors)
    cc = np.zeros(ns) if carbon_cost is None else np.asarray(carbon_cost, dtype=float)
    pm = er * np.ones(ns)  # world import price 1
    pe = er * np.ones(ns)  # world export price 1
    pv = _va_unit_cost(cal, w)
    pz = _cet_price(cal, pd, pe)
    # Effective per-output carbon cost (Phase 5d.5): cc for the flat model, energy-weighted for the
    # nest. Carbon revenue and all recycling/government marginal-revenue coefficients use cc_eff·Z,
    # so they stay linear in output whether or not the nest is active. The nest reads pv for its
    # KL-vs-energy split. _quantities() below is passed pv/cc so its A(p) matches.
    cc_eff = _intermediate_coeffs(cal, pq, pv, cc)[2]

    # Household income: factor income + (WITHOUT a savings-investment account) the net ROW capital
    # transfer + recycled carbon revenue. Foreign savings Sf is a fixed foreign-currency inflow
    # valued at the exchange rate (er·Sf); with a savings-investment account it finances
    # INVESTMENT instead (Phase 5d.2). At fixed prices every quantity is LINEAR in the demand
    # vector (FD = γ·I/pq, Q solves a linear system in demand, Z = ratio·Q), so every fixed point
    # below solves in closed form, guarded k < 1.
    factor_income = float(np.dot(w, cal.endowment))
    recycles = recycling != "none" and np.any(cc != 0.0)
    if cal.has_investment and inv_closure not in ("savings_driven", "fixed_real"):
        raise ValueError(
            f"unsupported inv_closure {inv_closure!r}; 5d.2 implements 'savings_driven' "
            "(default) and 'fixed_real'."
        )
    s = cal.sav_rate0 if cal.has_investment else 0.0
    sf = er * cal.foreign_savings  # the capital-account inflow in domestic currency

    if not cal.has_government:
        if not cal.has_investment:
            # Pre-5d behaviour, unchanged: er·Sf and recycled revenue both go to the household.
            base_income = factor_income + sf
            if recycles:
                _, z_unit, *_ = _quantities(
                    cal, pd, pq, pm, pe, pz, cal.gamma * 1.0 / pq, pv=pv, carbon_cost=cc
                )
                k = float(cc_eff @ z_unit)
                if strict and k >= 1.0 - 1e-12:
                    raise ValueError(
                        f"open recycling fixed point diverges: carbon revenue ≥ income "
                        f"(k={k:.6g} ≥ 1) at the equilibrium; the recycled transfer would exceed "
                        "factor income. Lower the carbon price."
                    )
                # Non-strict: a smooth positive floor on (1−k) keeps income finite and the
                # residual C¹-continuous with a restoring gradient when a trial point has k≥1
                # (review P2); it is the identity for a real equilibrium.
                income = base_income / _safe_denom(1.0 - k)
            else:
                income = base_income
            FD = cal.gamma * income / pq
            ID = np.zeros(ns)
            savings = 0.0
        elif inv_closure == "savings_driven":
            # Household income = factor income + recycled revenue (NO er·Sf — it funds
            # investment). Demand has an income-proportional part u·I and a fixed part
            # inv_gamma·sf/pq, so R = k·I + c0 and I = (FI + c0)/(1−k).
            u = ((1.0 - s) * cal.gamma + s * cal.inv_gamma) / pq
            id_sf = cal.inv_gamma * sf / pq  # the Sf-financed part of investment demand
            if recycles:
                _, z_u, *_ = _quantities(cal, pd, pq, pm, pe, pz, u, pv=pv, carbon_cost=cc)
                k = float(cc_eff @ z_u)
                _, z_sf, *_ = _quantities(cal, pd, pq, pm, pe, pz, id_sf, pv=pv, carbon_cost=cc)
                c0 = float(cc_eff @ z_sf)
            else:
                k, c0 = 0.0, 0.0
            if strict and k >= 1.0 - 1e-12:
                raise ValueError(f"open recycling fixed point diverges (k={k:.6g} ≥ 1)")
            income = (factor_income + c0) / _safe_denom(1.0 - k)
            savings = s * income
            FD = (1.0 - s) * income * cal.gamma / pq
            ID = (savings + sf) * cal.inv_gamma / pq
        else:  # fixed_real
            ID = cal.INV0.copy()
            p_inv = float(np.dot(pq, ID))
            if recycles:
                _, z_unit, *_ = _quantities(
                    cal, pd, pq, pm, pe, pz, cal.gamma * 1.0 / pq, pv=pv, carbon_cost=cc
                )
                k = float(cc_eff @ z_unit)
                _, z_id, *_ = _quantities(cal, pd, pq, pm, pe, pz, ID, pv=pv, carbon_cost=cc)
                c_inv = float(cc_eff @ z_id)
            else:
                k, c_inv = 0.0, 0.0
            if strict and k >= 1.0 - 1e-12:
                raise ValueError(f"open recycling fixed point diverges (k={k:.6g} ≥ 1)")
            # Savings = pq·INV0 − er·Sf (residual); consumption income = I − savings. With
            # recycling, R = k·(I − savings) + c_inv and I = FI + R:
            savings = p_inv - sf
            income = (factor_income - k * savings + c_inv) / _safe_denom(1.0 - k)
            if strict and income - savings <= 0:
                raise ValueError(
                    f"fixed_real investment leaves no consumption (savings {savings:.6g} ≥ "
                    f"income {income:.6g}); lower the shock or use savings_driven."
                )
            FD = (income - savings) * cal.gamma / pq
        GD = np.zeros(ns)
        gov_income = 0.0
    else:
        if gov_closure != "balanced_budget":
            raise ValueError(
                f"unsupported gov_closure {gov_closure!r} (5d.1 only implements "
                "'balanced_budget'; 'deficit_financed' is Phase 5d.7)"
            )
        # Household: factor income − direct tax, + er·Sf ONLY without a savings-investment
        # account; carbon revenue never enters it.
        tax = cal.gov_tax_rate0 * factor_income
        if not cal.has_investment:
            income = factor_income + sf - tax
            FD = cal.gamma * income / pq
            ID = np.zeros(ns)
            savings = 0.0
        elif inv_closure == "savings_driven":
            income = factor_income - tax
            savings = s * income
            FD = (1.0 - s) * income * cal.gamma / pq
            ID = (savings + sf) * cal.inv_gamma / pq
        else:  # fixed_real
            income = factor_income - tax
            ID = cal.INV0.copy()
            savings = float(np.dot(pq, ID)) - sf
            if strict and income - savings <= 0:
                raise ValueError(
                    f"fixed_real investment leaves no consumption (savings {savings:.6g} ≥ "
                    f"disposable income {income:.6g})."
                )
            FD = (income - savings) * cal.gamma / pq
        if recycles:
            _, z_fd, *_ = _quantities(cal, pd, pq, pm, pe, pz, FD + ID, pv=pv, carbon_cost=cc)
            r0 = float(cc_eff @ z_fd)  # revenue from the (price-fixed) household+investment demand
            _, z_unit_g, *_ = _quantities(
                cal, pd, pq, pm, pe, pz, cal.gov_gamma * 1.0 / pq, pv=pv, carbon_cost=cc
            )
            kg = float(cc_eff @ z_unit_g)  # marginal revenue per unit of government spending
        else:
            r0, kg = 0.0, 0.0
        if strict and kg >= 1.0 - 1e-12:
            raise ValueError(
                f"open government recycling fixed point diverges (kg={kg:.6g} ≥ 1); lower the "
                "carbon price."
            )
        gov_income = (tax + r0) / _safe_denom(1.0 - kg)
        GD = cal.gov_gamma * gov_income / pq

    Q, Z, D, E, M = _quantities(cal, pd, pq, pm, pe, pz, FD + GD + ID, pv=pv, carbon_cost=cc)
    carbon_revenue = float(cc_eff @ Z)  # cc_eff = cc (flat) or energy-weighted (nest)
    # Independent check of the income identity we just solved (cheap; catches any regression in the
    # linearity assumption above). Only in strict mode — the non-strict clamp deliberately violates
    # the identity to steer the solver, so checking it there would defeat the smooth-penalty design.
    if strict:
        if not cal.has_government:
            base = factor_income + (sf if not cal.has_investment else 0.0)
            resid = income - (base + (carbon_revenue if recycles else 0.0))
        else:
            resid = gov_income - (cal.gov_tax_rate0 * factor_income + carbon_revenue)
        if abs(resid) > 1e-9 * max(1.0, abs(income)):  # pragma: no cover - guards the closed form
            raise ValueError(f"open income identity not satisfied (residual {resid:.3e}).")
        # Savings-investment identity (Phase 5d.2): nominal investment = savings + er·Sf, under
        # either closure (fixed_real defines savings as exactly this residual).
        if cal.has_investment:
            resid_si = float(np.dot(pq, ID)) - (savings + sf)
            if abs(resid_si) > 1e-9 * max(1.0, abs(savings) + abs(sf)):  # pragma: no cover
                raise ValueError(
                    f"open savings-investment identity not satisfied (residual {resid_si:.3e})."
                )
    # Factor demand (Shephard on VA cost). VA quantity per unit output is va_share (flat) or the
    # price-responsive KL quantity per output from the nest (Phase 5d.5).
    va_qty_per_z = (
        cal.va_share if not cal.has_energy_nest else _intermediate_coeffs(cal, pq, pv, cc)[1]
    )
    va_cost = va_qty_per_z * pv * Z
    F = _factor_demand(cal, w, pv, va_cost)
    return OpenModelState(
        pd=pd,
        pq=pq,
        pz=pz,
        w=w,
        er=er,
        Z=Z,
        D=D,
        E=E,
        M=M,
        Q=Q,
        FD=FD,
        F=F,
        income=income,
        carbon_revenue=carbon_revenue,
        GD=GD,
        gov_income=gov_income,
        fiscal_balance=0.0,  # balanced_budget: spending exhausts income by construction
        ID=ID,
        savings=savings,
    )


def _intermediate_coeffs(cal, pq, pv, carbon_cost):
    """Intermediate-demand coefficient matrix ``a[j,i]`` (composite j per unit activity output i),
    the value-added (KL) quantity per unit output, and the effective per-output carbon cost.

    Flat model: fixed ``cal.ax`` and ``cal.va_share`` (price-independent, bit-identical to
    pre-5d.5), cc_eff = cc. Energy nest (Phase 5d.5): intermediates are the Armington COMPOSITE
    commodities, so the nest substitutes over composite prices (imports included) and the carbon
    cost attaches to the energy composites; a(p), the KL quantity, and cc_eff are all
    price-responsive — cc_eff[i] = Σ_{j∈energy} cc[j]·a_energy[j,i], a linear functional of
    output, so the recycling/government fixed points carry over unchanged."""
    ns = len(cal.sectors)
    cc = np.zeros(ns) if carbon_cost is None else np.asarray(carbon_cost, dtype=float)
    if not cal.has_energy_nest:
        return cal.ax, cal.va_share, cc
    from cge.engines.cge_static.energy_nest import nest_demands

    nest = cal.energy_nest
    energy_use, materials_use, kl_qty = nest_demands(nest, pq, pv, cc, np.ones(ns))
    a = np.zeros((ns, ns))  # a[j, i] = composite commodity j per unit output i
    cc_eff = np.zeros(ns)
    for k, j in enumerate(nest.energy_idx):
        a[j, :] += energy_use[k, :]
        cc_eff += cc[j] * energy_use[k, :]
    for k, j in enumerate(nest.mat_idx):
        a[j, :] += materials_use[k, :]
    return a, kl_qty, cc_eff


def _quantities(cal, pd, pq, pm, pe, pz, FD, *, pv=None, carbon_cost=None):
    """Given prices and final demand, solve for composite Q, output Z, and the D/E/M splits.

    Composite market: Q_i = Σ_j a[i,j] Z_j + FD_i, with ``a`` the intermediate-demand coefficients
    (fixed Leontief, or the price-responsive energy-nest demands — see ``_intermediate_coeffs``;
    ``pv`` and ``carbon_cost`` are only read when the nest is active). Domestic sales D are shared
    between supplying Q (Armington) and coming from Z (CET); at equilibrium the CET domestic supply
    equals the Armington domestic demand. Using the CES/CET Shephard shares (functions of prices
    only), write D_i = s^Q_i Q_i (Armington domestic input per composite) and D_i = s^Z_i Z_i (CET
    domestic output per activity output), so Z_i = (s^Q_i / s^Z_i) Q_i. Then the composite market is
    linear in Q: Q = a^T diag(s^Q/s^Z) Q + FD."""
    ns = len(FD)
    sigma = cal.arm_elast
    omega = cal.cet_elast
    A, B = cal.arm_scale, cal.cet_scale
    ax = cal.ax if not cal.has_energy_nest else _intermediate_coeffs(cal, pq, pv, carbon_cost)[0]
    # Armington (Shephard on the composite cost fn): per unit composite Q,
    #   D/Q = (1/A)(pq·A)^σ δ^σ pd^{-σ};  M/Q = (1/A)(pq·A)^σ (1-δ)^σ pm^{-σ}.
    sQ = (1.0 / A) * np.power(pq * A, sigma) * cal.arm_share_d**sigma * np.power(pd, -sigma)
    sM = np.where(
        cal.arm_share_m > 0,
        (1.0 / A) * np.power(pq * A, sigma) * cal.arm_share_m**sigma * np.power(pm, -sigma),
        0.0,
    )
    # CET (Hotelling on the revenue fn): per unit activity output Z,
    #   D/Z = (1/B)(pz·B)^{-Ω} ξ^{-Ω} pd^{Ω};  E/Z = (1/B)(pz·B)^{-Ω} (1-ξ)^{-Ω} pe^{Ω}.
    sZd = (1.0 / B) * np.power(pz * B, -omega) * cal.cet_share_d ** (-omega) * np.power(pd, omega)
    share_e = np.where(cal.cet_share_e > 0, cal.cet_share_e, 1.0)  # safe base; masked below
    sZe = np.where(
        cal.cet_share_e > 0,
        (1.0 / B) * np.power(pz * B, -omega) * share_e ** (-omega) * np.power(pe, omega),
        0.0,
    )
    # Z in terms of Q via domestic-sales consistency: D = sQ·Q (Armington) = sZd·Z (CET) ⇒
    # Z = (sQ/sZd)·Q.
    ratio = sQ / sZd  # [i]
    # Composite market: Q_i = Σ_j a[i,j] Z_j + FD_i = Σ_j a[i,j] ratio_j Q_j + FD_i.
    coeff = ax * ratio[None, :]  # a[i,j]·ratio[j] (a = fixed Leontief or price-responsive nest)
    Q = np.linalg.solve(np.eye(ns) - coeff, FD)
    Z = ratio * Q
    D = sQ * Q
    M = sM * Q
    E = sZe * Z
    return Q, Z, D, E, M


def residuals(
    cal: OpenCalibratedModel,
    z: np.ndarray,
    *,
    carbon_cost: np.ndarray | None = None,
    recycling: str = "lump_sum",
    drop_factor: int = 0,
    inv_closure: str = "savings_driven",
) -> np.ndarray:
    ns = len(cal.sectors)
    nf = len(cal.factors)
    pd = np.asarray(z[:ns], dtype=float)
    pq = np.asarray(z[ns : 2 * ns], dtype=float)
    w = np.asarray(z[2 * ns : 2 * ns + nf], dtype=float)
    er = float(z[2 * ns + nf])
    cc = np.zeros(ns) if carbon_cost is None else np.asarray(carbon_cost, dtype=float)

    st = derive_open_state(
        cal, pd, pq, w, er, carbon_cost=cc, recycling=recycling, inv_closure=inv_closure
    )
    pm = er * np.ones(ns)
    pe = er * np.ones(ns)
    pv = _va_unit_cost(cal, w)

    res = []
    # Armington price identity: pq = CES cost of (pd, pm).  [ns rows]
    pq_id = _armington_price(cal, pd, pm)
    res.extend((pq - pq_id).tolist())
    # Zero-profit: activity output price pz(pd,pe) = input cost. Flat: Σ_j ax[j,i]·pq_j + va·pv +
    # cc. Energy nest (Phase 5d.5): the input cost is the nest's output unit cost over composite
    # prices (carbon attaches to the energy composites inside the nest). [ns rows]
    pz = _cet_price(cal, pd, pe)
    if cal.has_energy_nest:
        from cge.engines.cge_static.energy_nest import nest_unit_cost

        px = nest_unit_cost(cal.energy_nest, pq, pv, cc)
        for i in range(ns):
            res.append(pz[i] - px[i])
    else:
        for i in range(ns):
            intermediate = sum(cal.ax[j, i] * pq[j] for j in range(ns))
            res.append(pz[i] - (intermediate + cal.va_share[i] * pv[i] + cc[i]))
    # NB: composite-market clearing Q_i = Σ_j ax[i,j] Z_j + FD_i is NOT an independent residual — it
    # is solved *algebraically* inside _quantities() (Q = (I − ax·diag(ratio))⁻¹ FD, Z = ratio·Q),
    # so (Q − (ax·Z + FD)) ≡ 0 by construction and would be a tautological row (review P2). The
    # composite market is therefore closed by construction, not by a solver equation.
    # Factor clearing (drop one by Walras).  [nf−1 rows]
    for f in range(nf):
        if f == drop_factor:
            continue
        res.append(float(st.F[f, :].sum()) - cal.endowment[f])
    # Trade balance: Σ pm·M − Σ pe·E − er·Sf = 0.  [1 row]
    res.append(float(pm @ st.M - pe @ st.E - er * cal.foreign_savings))
    # Numéraire: household CPI over the composite good, Π pq_i^γ_i = 1.  [1 row]
    cpi = float(np.prod(np.power(pq, cal.gamma)))
    res.append(cpi - 1.0)
    # Total: ns + ns + (nf−1) + 1 + 1 = 2·ns + nf + 1 = n_unknowns — an exactly square system.
    return np.array(res, dtype=float)


def initial_guess(cal: OpenCalibratedModel) -> np.ndarray:
    """Benchmark start: all prices and the exchange rate = 1."""
    return np.ones(2 * len(cal.sectors) + len(cal.factors) + 1)


def n_unknowns(cal: OpenCalibratedModel) -> int:
    return 2 * len(cal.sectors) + len(cal.factors) + 1
