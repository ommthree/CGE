"""The MULTI-REGION static CGE as a residual system (Phase 5.4 — bilateral trade).

Closed global economy of ``R`` regions trading bilaterally [Hosoe2010, ch. 7 generalised]. Each
region has an Armington import composite over **every partner region** and a CET transformation of
output into domestic sales + **exports to every partner**. Factors are region-specific.

**Bilateral prices + market clearing (the corrected design).** Each traded route has its own price
``pe[o,s,d]`` — the price region ``d`` pays for good ``s`` imported from origin ``o`` (= the price
``o`` receives on that export). The Armington composite in ``d`` mixes the domestic variety (price
``pd[d,s]``) with imports at ``pe[o,s,d]``; the CET in ``o`` splits output between domestic sales
(``pd[o,s]``) and exports at ``pe[o,s,d]``. Each bilateral goods market clears explicitly:
``M[d,s,o] = EX[o,s,d]`` — the equation that pins ``pe`` and makes the result a genuine equilibrium
(the earlier law-of-one-price reduction left these markets **uncleared**). One global numéraire pins
region-0's composite CPI.

**Unknowns** ``z = [pd (nr·ns), pq (nr·ns), pe (n_active), w (nf·nr)]``:
- ``pd[r,s]`` — domestic-sales price of region ``r``'s good ``s``;
- ``pq[r,s]`` — Armington composite price in region ``r``;
- ``pe[o,s,d]`` — bilateral export/import price on route ``o→d`` for good ``s`` (``o≠d``), **one
  unknown per ordered pair that actually trades** — ``n_active = len(cal.active_routes)``, a subset
  of the ``nr·(nr−1)·ns`` possible directed routes;
- ``w[f,r]`` — region-``r`` factor prices.

**Only routes with genuine benchmark trade get a price unknown** (review P1). A route with zero
benchmark trade is never read by ``_armington_price``/``_cet_price``/``_quantities`` (they gate on
the calibrated share being ``>0``), so packing a live, unpinned price unknown for it — as an earlier
version did for every possible ``(o,s,d)`` — left the equilibrium **rank-deficient**: the Jacobian's
rank fell by exactly the number of zero routes, and perturbing an unused route's price left every
residual unchanged at machine-zero. Fixing an inactive route's price at 1 (never solved for) removes
the free direction without changing any quantity, since nothing downstream reads it.

**Residuals — square in ``2·nr·ns + n_active + nf·nr``:**
- ``nr·ns`` Armington-price identities (pin ``pq``);
- ``nr·ns`` zero-profit identities (pin ``pd`` via the CET dual ``pz``);
- ``n_active`` bilateral goods-market clearings ``M[d,s,o] = EX[o,s,d]`` (pin ``pe``), one per
  active route only;
- ``nf·nr − 1`` factor-market clearings (one dropped globally by Walras);
- 1 numéraire.
The **domestic** goods market clears algebraically inside ``_quantities`` (domestic Armington demand
= domestic CET supply, per region-sector); a region's current account then closes by Walras once the
bilateral and factor markets clear.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cge.engines.cge_static.calibrate_multi import MultiCalibratedModel


@dataclass(frozen=True)
class MultiModelState:
    pd: np.ndarray  # [r, s] domestic-sales prices
    pq: np.ndarray  # [r, s] composite prices
    pe: np.ndarray  # [o, s, d] bilateral export/import price on route o→d (own slot unused)
    pz: np.ndarray  # [r, s] activity output prices
    w: np.ndarray  # [f, r] factor prices
    Z: np.ndarray  # [r, s] activity output
    D: np.ndarray  # [r, s] domestic sales
    EX: np.ndarray  # [o, s, d] export supply from o to d
    M: np.ndarray  # [d, s, o] import demand of d from o
    Q: np.ndarray  # [r, s] composite supply
    FD: np.ndarray  # [r, s] household final demand
    F: np.ndarray  # [f, r, s] factor demand
    income: np.ndarray  # [r]
    carbon_revenue: np.ndarray  # [r]
    # Government accounts (Phase 5d.1) — zeros when cal.has_government is False (pre-5d.1
    # behaviour preserved exactly; same convention as the closed/open models).
    GD: np.ndarray  # [r, s] government final demand
    gov_income: np.ndarray  # [r] government income (benchmark tax + own-region carbon revenue)
    fiscal_balance: np.ndarray  # [r] income − spending (≡0 under balanced_budget)
    # Savings-investment accounts (Phase 5d.2). Zeros when cal.has_investment is False.
    ID: np.ndarray  # [r, s] investment final demand
    savings: np.ndarray  # [r] household savings (investment_r = savings_r + Sf_r)


def _va_unit_cost(cal: MultiCalibratedModel, w: np.ndarray) -> np.ndarray:
    """VA unit cost [r,s]. σ_va=1 ⇒ Cobb-Douglas; ≠1 ⇒ CES (per region-sector). ``w`` is [f,r]."""
    nr, ns = cal.nr, cal.ns
    pv = np.empty((nr, ns))
    for ri in range(nr):
        wr = w[:, ri]
        for si in range(ns):
            s = cal.va_elast[ri, si]
            if abs(s - 1.0) < 1e-12:
                b = cal.beta[:, ri, si]
                ratio = np.where(b > 0, wr / np.where(b > 0, b, 1.0), 1.0)
                pv[ri, si] = (1.0 / cal.av[ri, si]) * np.prod(np.power(ratio, b))
            else:
                d = cal.va_ces_share[:, ri, si]
                pv[ri, si] = (1.0 / cal.av[ri, si]) * np.power(
                    np.sum(d**s * wr ** (1.0 - s)), 1.0 / (1.0 - s)
                )
    return pv


def _factor_demand(cal, w, pv, va_cost):
    """Factor demand [f,r,s] by Shephard on the VA cost (CD or CES per region-sector)."""
    nr, ns, nf = cal.nr, cal.ns, cal.nf
    F = np.empty((nf, nr, ns))
    for ri in range(nr):
        wr = w[:, ri]
        for si in range(ns):
            s = cal.va_elast[ri, si]
            if abs(s - 1.0) < 1e-12:
                F[:, ri, si] = cal.beta[:, ri, si] * va_cost[ri, si] / wr
            else:
                d = cal.va_ces_share[:, ri, si]
                unit = (
                    (1.0 / cal.av[ri, si]) * (pv[ri, si] * cal.av[ri, si]) ** s * d**s * wr ** (-s)
                )
                F[:, ri, si] = unit * (va_cost[ri, si] / pv[ri, si])
    return F


def _import_price(cal: MultiCalibratedModel, ri: int, si: int, oi: int, pd, pe) -> float:
    """Price region ``ri`` pays for good ``si`` from origin ``oi`` — its own domestic price on the
    own slot, else the bilateral route price ``pe[oi,si,ri]``."""
    return pd[ri, si] if oi == ri else pe[oi, si, ri]


def _armington_price(cal: MultiCalibratedModel, pd: np.ndarray, pe: np.ndarray) -> np.ndarray:
    """Composite price pq[d,s] = CES cost over the domestic variety (price pd[d,s]) and imports from
    each origin o at the bilateral price pe[o,s,d]:
        pq = (1/A)[δ_d·pd_d^{1-σ} + Σ_o δ_o·pe[o,s,d]^{1-σ}]^{1/(1-σ)}."""
    nr, ns = cal.nr, cal.ns
    sigma = cal.arm_elast
    e = 1.0 - sigma
    pq = np.empty((nr, ns))
    for di in range(nr):
        for si in range(ns):
            term = cal.arm_share_d[di, si] ** sigma[di, si] * pd[di, si] ** e[di, si]
            for oi in range(nr):
                dm = cal.arm_share_m[di, si, oi]
                if dm > 0:
                    p_imp = pe[oi, si, di]
                    term += dm ** sigma[di, si] * p_imp ** e[di, si]
            pq[di, si] = (1.0 / cal.arm_scale[di, si]) * term ** (1.0 / e[di, si])
    return pq


def _cet_price(cal: MultiCalibratedModel, pd: np.ndarray, pe: np.ndarray) -> np.ndarray:
    """Activity output price pz[o,s] = CET revenue function over domestic sales (price pd[o,s]) and
    exports to each destination d at price pe[o,s,d]:
        pz = (1/B)[ξ_d^{-Ω}·pd_o^{1+Ω} + Σ_d ξ_{e,d}^{-Ω}·pe[o,s,d]^{1+Ω}]^{1/(1+Ω)}."""
    nr, ns = cal.nr, cal.ns
    omega = cal.cet_elast
    e = 1.0 + omega
    pz = np.empty((nr, ns))
    for oi in range(nr):
        for si in range(ns):
            term = cal.cet_share_d[oi, si] ** (-omega[oi, si]) * pd[oi, si] ** e[oi, si]
            for di in range(nr):
                xe = cal.cet_share_e[oi, si, di]
                if xe > 0:
                    term += xe ** (-omega[oi, si]) * pe[oi, si, di] ** e[oi, si]
            pz[oi, si] = (1.0 / cal.cet_scale[oi, si]) * term ** (1.0 / e[oi, si])
    return pz


def _quantities(cal, pd, pq, pe, pz, FD):
    """Given prices and per-region final demand FD[r,s], solve for composite supply Q, output Z, the
    domestic split D, bilateral import demand M[d,s,o] and export supply EX[o,s,d].

    Armington (Shephard on the composite cost fn), per unit composite Q[d,s]:
      domestic  sD  = (1/A)(pq·A)^σ δ_d^σ pd_d^{-σ};
      import-o  sM_o = (1/A)(pq·A)^σ δ_o^σ pe[o,s,d]^{-σ}.
    CET (Hotelling on the revenue fn), per unit output Z[o,s]:
      domestic  sZd  = (1/B)(pz·B)^{-Ω} ξ_d^{-Ω} pd_o^{Ω};
      export-d  sZe_d = (1/B)(pz·B)^{-Ω} ξ_{e,d}^{-Ω} pe[o,s,d]^{Ω}.

    The **domestic** market clears by construction — domestic Armington demand sD·Q equals domestic
    CET supply sZd·Z, giving Z[r,s] = (sD/sZd)·Q — so we solve each region's composite market
    (linear in Q) block by block. Bilateral import demand M and export supply EX are then returned
    **separately**; the residual system clears M[d,s,o] = EX[o,s,d] by choosing pe."""
    nr, ns = cal.nr, cal.ns
    sigma = cal.arm_elast
    omega = cal.cet_elast
    A, B = cal.arm_scale, cal.cet_scale

    # Armington unit shares (per unit composite Q[d,s]).
    sD = np.zeros((nr, ns))
    sM = np.zeros((nr, ns, nr))  # [d,s,o] import of o per unit Q[d,s]
    for di in range(nr):
        for si in range(ns):
            base = (1.0 / A[di, si]) * (pq[di, si] * A[di, si]) ** sigma[di, si]
            sD[di, si] = (
                base * cal.arm_share_d[di, si] ** sigma[di, si] * pd[di, si] ** (-sigma[di, si])
            )
            for oi in range(nr):
                dm = cal.arm_share_m[di, si, oi]
                if dm > 0:
                    p_imp = pe[oi, si, di]
                    sM[di, si, oi] = base * dm ** sigma[di, si] * p_imp ** (-sigma[di, si])
    # CET unit shares (per unit output Z[o,s]).
    sZd = np.zeros((nr, ns))
    sZe = np.zeros((nr, ns, nr))  # [o,s,d] export to d per unit Z[o,s]
    for oi in range(nr):
        for si in range(ns):
            base = (1.0 / B[oi, si]) * (pz[oi, si] * B[oi, si]) ** (-omega[oi, si])
            sZd[oi, si] = (
                base * cal.cet_share_d[oi, si] ** (-omega[oi, si]) * pd[oi, si] ** omega[oi, si]
            )
            for di in range(nr):
                xe = cal.cet_share_e[oi, si, di]
                if xe > 0:
                    sZe[oi, si, di] = (
                        base * xe ** (-omega[oi, si]) * pe[oi, si, di] ** omega[oi, si]
                    )

    # Domestic consistency: Z[r,s] = (sD[r,s]/sZd[r,s]) · Q[r,s].
    ratio = sD / sZd  # [r,s]
    # Composite market per region (block-diagonal — intermediates are within-region):
    #   Q[r,s] = Σ_i ax[r,s,i]·ratio[r,i]·Q[r,i] + FD[r,s].
    Q = np.zeros((nr, ns))
    for ri in range(nr):
        coeff = cal.ax[ri] * ratio[ri][None, :]  # ax[r,s,i]·ratio[r,i] → [s,i]
        Q[ri] = np.linalg.solve(np.eye(ns) - coeff, FD[ri])
    Z = ratio * Q
    D = sD * Q
    M = np.zeros((nr, ns, nr))  # [d,s,o] import DEMAND of d from o
    EX = np.zeros((nr, ns, nr))  # [o,s,d] export SUPPLY of o to d
    for si in range(ns):
        for di in range(nr):
            for oi in range(nr):
                if oi != di:
                    M[di, si, oi] = sM[di, si, oi] * Q[di, si]
        for oi in range(nr):
            for di in range(nr):
                if di != oi:
                    EX[oi, si, di] = sZe[oi, si, di] * Z[oi, si]
    return Q, Z, D, EX, M


def derive_multi_state(
    cal: MultiCalibratedModel,
    pd: np.ndarray,
    pq: np.ndarray,
    pe: np.ndarray,
    w: np.ndarray,
    *,
    carbon_cost: np.ndarray | None = None,
    recycling: str = "lump_sum",
    strict: bool = False,
    gov_closure: str = "balanced_budget",
    inv_closure: str = "savings_driven",
) -> MultiModelState:
    """Close the multi-region model at prices (pd, pq, pe, w): derive all quantities and per-region
    income. ``carbon_cost`` is [r,s]. Income per region = factor income + Sf transfer + recycled
    carbon revenue. ``strict`` raises on a diverging recycling fixed point (k≥1) at the accepted
    equilibrium; non-strict clamps it (keeps the residual continuous during the solve).

    **Government accounts (Phase 5d.1, ``cal.has_government``):** the closed/open design, one
    government per region. Each region's household pays its benchmark direct tax
    (rate_r·factor income_r); each region's GOVERNMENT collects its own region's carbon revenue
    and spends tax+revenue on its own CD demand vector under ``balanced_budget``. The per-region
    fixed point is exact because ``_quantities`` is block-diagonal in demand per region at fixed
    prices (region r's output depends only on region r's demand): gov_income_r =
    (T_r + R0_r)/(1 − kg_r).

    **Savings-investment accounts (Phase 5d.2, ``cal.has_investment``):** one per region; the
    bilateral capital transfer Sf_r RE-ROUTES from household income into the region's investment
    pool: investment_r = household savings_r + Sf_r. Closures ``savings_driven`` (default) and
    ``fixed_real`` mirror the open model, per region."""
    nr, ns = cal.nr, cal.ns
    cc = np.zeros((nr, ns)) if carbon_cost is None else np.asarray(carbon_cost, dtype=float)
    pv = _va_unit_cost(cal, w)
    pz = _cet_price(cal, pd, pe)

    # Per-region income: base = factor income + (WITHOUT a savings-investment layer) the fixed
    # foreign-savings transfer Sf (held at its calibrated nominal level, the fixed-Sf closure).
    # Recycled carbon revenue adds k·I when no government collects it.
    factor_income = (w * cal.endowment).sum(axis=0)  # [r]
    recycles = recycling != "none" and np.any(cc != 0.0)
    if cal.has_investment and inv_closure not in ("savings_driven", "fixed_real"):
        raise ValueError(
            f"unsupported inv_closure {inv_closure!r}; 5d.2 implements 'savings_driven' "
            "(default) and 'fixed_real'."
        )
    s = cal.sav_rate0 if cal.has_investment else np.zeros(nr)
    sf = cal.foreign_savings  # [r] fixed nominal capital-account inflow

    def _k_of(demand):
        """Per-region marginal carbon revenue of a demand array [r,s] (exact: block-diagonal)."""
        _, z_d, *_ = _quantities(cal, pd, pq, pe, pz, demand)
        return (cc * z_d).sum(axis=1)

    if not cal.has_government:
        if not cal.has_investment:
            # Pre-5d behaviour, unchanged: Sf and recycled revenue both go to the household.
            base_income = factor_income + sf  # [r]
            income = base_income.copy()
            if recycles:
                k = _k_of(cal.gamma / pq)  # [r]
                if strict and np.any(k >= 1.0 - 1e-12):
                    bad = int(np.argmax(k))
                    raise ValueError(
                        f"multi-region recycling fixed point diverges in region "
                        f"{cal.regions[bad]}: carbon revenue ≥ income (k={k[bad]:.6g} ≥ 1). "
                        "Lower the carbon price."
                    )
                k = np.clip(k, None, 1.0 - 1e-9)
                income = base_income / (1.0 - k)
            FD = cal.gamma * income[:, None] / pq
            ID = np.zeros((nr, ns))
            savings = np.zeros(nr)
        elif inv_closure == "savings_driven":
            # Household income = factor income + recycled revenue (Sf funds investment). Demand
            # has an income-proportional part u·I and a fixed Sf part, so I = (FI + c0)/(1−k).
            u = ((1.0 - s)[:, None] * cal.gamma + s[:, None] * cal.inv_gamma) / pq
            id_sf = cal.inv_gamma * sf[:, None] / pq
            if recycles:
                k = _k_of(u)
                c0 = _k_of(id_sf)
            else:
                k, c0 = np.zeros(nr), np.zeros(nr)
            if strict and np.any(k >= 1.0 - 1e-12):
                bad = int(np.argmax(k))
                raise ValueError(
                    f"multi-region recycling fixed point diverges in region "
                    f"{cal.regions[bad]} (k={k[bad]:.6g} ≥ 1); lower the carbon price."
                )
            k = np.clip(k, None, 1.0 - 1e-9)
            income = (factor_income + c0) / (1.0 - k)
            savings = s * income
            FD = (1.0 - s)[:, None] * income[:, None] * cal.gamma / pq
            ID = (savings + sf)[:, None] * cal.inv_gamma / pq
        else:  # fixed_real
            ID = cal.INV0.copy()
            p_inv = (pq * ID).sum(axis=1)  # [r]
            savings = p_inv - sf
            if recycles:
                k = _k_of(cal.gamma / pq)
                c_inv = _k_of(ID)
            else:
                k, c_inv = np.zeros(nr), np.zeros(nr)
            if strict and np.any(k >= 1.0 - 1e-12):
                bad = int(np.argmax(k))
                raise ValueError(
                    f"multi-region recycling fixed point diverges in region "
                    f"{cal.regions[bad]} (k={k[bad]:.6g} ≥ 1); lower the carbon price."
                )
            k = np.clip(k, None, 1.0 - 1e-9)
            income = (factor_income - k * savings + c_inv) / (1.0 - k)
            if strict and np.any(income - savings <= 0):
                bad = int(np.argmax(savings - income))
                raise ValueError(
                    f"fixed_real investment leaves no consumption in region "
                    f"{cal.regions[bad]}; lower the shock or use savings_driven."
                )
            FD = (income - savings)[:, None] * cal.gamma / pq
        GD = np.zeros((nr, ns))
        gov_income = np.zeros(nr)
    else:
        if gov_closure != "balanced_budget":
            raise ValueError(
                f"unsupported gov_closure {gov_closure!r} (5d.1 only implements "
                "'balanced_budget'; 'deficit_financed' is Phase 5d.7)"
            )
        # Household per region: factor income − direct tax, + Sf ONLY without a savings-
        # investment layer; no carbon revenue in it.
        tax = cal.gov_tax_rate0 * factor_income  # [r]
        if not cal.has_investment:
            income = factor_income + sf - tax
            FD = cal.gamma * income[:, None] / pq
            ID = np.zeros((nr, ns))
            savings = np.zeros(nr)
        elif inv_closure == "savings_driven":
            income = factor_income - tax
            savings = s * income
            FD = (1.0 - s)[:, None] * income[:, None] * cal.gamma / pq
            ID = (savings + sf)[:, None] * cal.inv_gamma / pq
        else:  # fixed_real
            income = factor_income - tax
            ID = cal.INV0.copy()
            savings = (pq * ID).sum(axis=1) - sf
            if strict and np.any(income - savings <= 0):
                bad = int(np.argmax(savings - income))
                raise ValueError(
                    f"fixed_real investment leaves no consumption in region {cal.regions[bad]}."
                )
            FD = (income - savings)[:, None] * cal.gamma / pq
        if recycles:
            r0 = _k_of(FD + ID)  # [r] revenue from the (price-fixed) household+investment demand
            kg = _k_of(cal.gov_gamma / pq)  # [r] marginal revenue per unit of own-gov spending
        else:
            r0, kg = np.zeros(nr), np.zeros(nr)
        if strict and np.any(kg >= 1.0 - 1e-12):
            bad = int(np.argmax(kg))
            raise ValueError(
                f"multi-region government recycling fixed point diverges in region "
                f"{cal.regions[bad]} (kg={kg[bad]:.6g} ≥ 1); lower the carbon price."
            )
        kg = np.clip(kg, None, 1.0 - 1e-9)
        gov_income = (tax + r0) / (1.0 - kg)  # [r]
        GD = cal.gov_gamma * gov_income[:, None] / pq

    Q, Z, D, EX, M = _quantities(cal, pd, pq, pe, pz, FD + GD + ID)
    carbon_revenue = (cc * Z).sum(axis=1)  # [r]
    va_cost = cal.va_share * pv * Z
    F = _factor_demand(cal, w, pv, va_cost)
    return MultiModelState(
        pd=pd,
        pq=pq,
        pe=pe,
        pz=pz,
        w=w,
        Z=Z,
        D=D,
        EX=EX,
        M=M,
        Q=Q,
        FD=FD,
        F=F,
        income=income,
        carbon_revenue=carbon_revenue,
        GD=GD,
        gov_income=gov_income,
        fiscal_balance=np.zeros(nr),  # balanced_budget by construction
        ID=ID,
        savings=savings,
    )


def _unpack(cal, z):
    """Split the flat unknown vector into (pd, pq, pe, w)."""
    nr, ns, nf = cal.nr, cal.ns, cal.nf
    o1 = nr * ns
    o2 = 2 * nr * ns
    n_pe = len(cal.active_routes)
    o3 = o2 + n_pe
    pd = np.asarray(z[:o1], dtype=float).reshape(nr, ns)
    pq = np.asarray(z[o1:o2], dtype=float).reshape(nr, ns)
    pe = _pe_from_flat(cal, np.asarray(z[o2:o3], dtype=float))
    w = np.asarray(z[o3 : o3 + nf * nr], dtype=float).reshape(nf, nr)
    return pd, pq, pe, w


def _pe_from_flat(cal, flat: np.ndarray) -> np.ndarray:
    """Expand the packed bilateral prices into a full [o,s,d] array. Only **active** routes (review
    P1: genuine benchmark trade, ``cal.active_routes``) get an unknown; a structurally-zero route
    keeps its price fixed at 1 — no unknown, no residual, since nothing in ``_armington_price`` /
    ``_cet_price`` / ``_quantities`` reads a zero-share route's price anyway (they gate on
    ``share > 0``), so fixing it at 1 changes nothing while removing a rank-deficient free
    direction."""
    nr, ns = cal.nr, cal.ns
    pe = np.ones((nr, ns, nr))
    for idx, (oi, si, di) in enumerate(cal.active_routes):
        pe[oi, si, di] = flat[idx]
    return pe


def residuals(
    cal: MultiCalibratedModel,
    z: np.ndarray,
    *,
    carbon_cost: np.ndarray | None = None,
    recycling: str = "lump_sum",
    drop_factor: int = 0,
    inv_closure: str = "savings_driven",
) -> np.ndarray:
    nr, ns, nf = cal.nr, cal.ns, cal.nf
    pd, pq, pe, w = _unpack(cal, z)
    cc = np.zeros((nr, ns)) if carbon_cost is None else np.asarray(carbon_cost, dtype=float)

    st = derive_multi_state(
        cal, pd, pq, pe, w, carbon_cost=cc, recycling=recycling, inv_closure=inv_closure
    )
    pv = _va_unit_cost(cal, w)
    pz = _cet_price(cal, pd, pe)

    res: list[float] = []
    # 1. Armington price identity: pq = CES cost of (pd_d, {pe[o,s,d]}).  [nr·ns]
    pq_id = _armington_price(cal, pd, pe)
    res.extend((pq - pq_id).ravel().tolist())
    # 2. Zero-profit: pz(pd,pe) = Σ_j ax[r,j,i]·pq[r,j] + va_share·pv + cc.  [nr·ns]
    for ri in range(nr):
        for ii in range(ns):
            intermediate = float(np.dot(cal.ax[ri, :, ii], pq[ri]))
            res.append(pz[ri, ii] - (intermediate + cal.va_share[ri, ii] * pv[ri, ii] + cc[ri, ii]))
    # 3. Bilateral goods-market clearing: import demand = export supply on each ACTIVE route only
    # (review P1: a route with zero benchmark trade got a free price unknown and no way to pin it —
    # Jacobian rank-deficient by exactly the number of zero routes; see cal.active_routes).
    for oi, si, di in cal.active_routes:
        res.append(st.M[di, si, oi] - st.EX[oi, si, di])
    # 4. Factor clearing per region-factor, dropping ONE globally by Walras.  [nf·nr − 1]
    flat = 0
    for fi in range(nf):
        for ri in range(nr):
            if flat != drop_factor:
                res.append(float(st.F[fi, ri, :].sum()) - cal.endowment[fi, ri])
            flat += 1
    # 5. Numéraire: region-0 household CPI over its composite good.  [1]
    cpi0 = float(np.prod(np.power(pq[0], cal.gamma[0])))
    res.append(cpi0 - 1.0)
    return np.array(res, dtype=float)


def initial_guess(cal: MultiCalibratedModel) -> np.ndarray:
    """Benchmark start: all prices = 1."""
    return np.ones(n_unknowns(cal))


def n_unknowns(cal: MultiCalibratedModel) -> int:
    return 2 * cal.nr * cal.ns + len(cal.active_routes) + cal.nf * cal.nr


def unpack_state(
    cal, x, *, carbon_cost=None, recycling="lump_sum", strict=True, inv_closure="savings_driven"
):
    """Convenience: derive the model state from a solved unknown vector (used by the engine)."""
    pd, pq, pe, w = _unpack(cal, x)
    return derive_multi_state(
        cal,
        pd,
        pq,
        pe,
        w,
        carbon_cost=carbon_cost,
        recycling=recycling,
        strict=strict,
        inv_closure=inv_closure,
    )
