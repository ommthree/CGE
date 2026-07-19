"""The MULTI-REGION static CGE as a residual system (Phase 5.4 — bilateral trade).

Closed global economy of ``R`` regions trading bilaterally [Hosoe2010, ch. 7 generalised]. Each
region has an Armington import composite over **every partner region** and a CET transformation of
output into domestic sales + **exports to every partner**. Factors are region-specific.

**Price convention (law of one price).** A producer sells its good at a single supply price
``pd[r,s]`` whether domestically or as an export; the CET allocates *quantities* between domestic
sales and exports, and region ``d``'s import price of good ``s`` from origin ``o`` is ``pd[o,s]``.
(A separate export/border price is a documented refinement.) One global numéraire pins region-0's
composite CPI.

**Unknowns** ``z = [pd (nr·ns), pq (nr·ns), w (nf·nr)]``:
- ``pd[r,s]`` — producer/domestic price of region ``r``'s good ``s`` (also its export price);
- ``pq[r,s]`` — Armington composite price in region ``r``;
- ``w[f,r]`` — region-``r`` factor prices.

Derived by CES/CET duals: composite price (Armington cost fn over ``pd[r,s]`` + partners'
``pd[o,s]``), output price ``pz`` (CET revenue fn), factor demand + quantities (Shephard/Hotelling).

**Residuals — square in ``2·nr·ns + nf·nr``:**
- ``nr·ns`` Armington-price identities (pin ``pq``);
- ``nr·ns`` zero-profit identities (pin ``pd`` via the CET dual ``pz``);
- ``nf·nr − 1`` factor-market clearings (one dropped globally by Walras);
- 1 numéraire.
Composite-market clearing is solved algebraically inside ``_quantities`` (not a residual), exactly
as in the single-region open model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cge.engines.cge_static.calibrate_multi import MultiCalibratedModel


@dataclass(frozen=True)
class MultiModelState:
    pd: np.ndarray  # [r, s] producer/domestic prices
    pq: np.ndarray  # [r, s] composite prices
    pz: np.ndarray  # [r, s] activity output prices
    w: np.ndarray  # [f, r] factor prices
    Z: np.ndarray  # [r, s] activity output
    D: np.ndarray  # [r, s] domestic sales
    EX: np.ndarray  # [r, s, d] exports to region d
    M: np.ndarray  # [r, s, o] imports from region o
    Q: np.ndarray  # [r, s] composite supply
    FD: np.ndarray  # [r, s] household final demand
    F: np.ndarray  # [f, r, s] factor demand
    income: np.ndarray  # [r]
    carbon_revenue: np.ndarray  # [r]


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


def _armington_price(cal: MultiCalibratedModel, pd: np.ndarray) -> np.ndarray:
    """Composite price pq[r,s] = CES cost over the domestic variety (price pd[r,s]) and imports from
    each partner o (price pd[o,s]): pq = (1/A)[δ_d·pd_r^{1-σ} + Σ_o δ_o·pd_o^{1-σ}]^{1/(1-σ)}."""
    nr, ns = cal.nr, cal.ns
    sigma = cal.arm_elast
    e = 1.0 - sigma
    pq = np.empty((nr, ns))
    for ri in range(nr):
        for si in range(ns):
            term = cal.arm_share_d[ri, si] ** sigma[ri, si] * pd[ri, si] ** e[ri, si]
            for oi in range(nr):
                dm = cal.arm_share_m[ri, si, oi]
                if dm > 0:
                    term += dm ** sigma[ri, si] * pd[oi, si] ** e[ri, si]
            pq[ri, si] = (1.0 / cal.arm_scale[ri, si]) * term ** (1.0 / e[ri, si])
    return pq


def _cet_price(cal: MultiCalibratedModel, pd: np.ndarray) -> np.ndarray:
    """Activity output price pz[r,s] = CET revenue fn. With law-of-one-price the domestic and every
    export sink share the producer price pd[r,s], so pz = (1/B)[Σ_k ξ_k^{-Ω}]^{1/(1+Ω)}·pd — the
    revenue index collapses to a constant × pd at a common price (the frontier is homogeneous)."""
    nr, ns = cal.nr, cal.ns
    omega = cal.cet_elast
    e = 1.0 + omega
    pz = np.empty((nr, ns))
    for ri in range(nr):
        for si in range(ns):
            coef = cal.cet_share_d[ri, si] ** (-omega[ri, si])
            for di in range(nr):
                xe = cal.cet_share_e[ri, si, di]
                if xe > 0:
                    coef += xe ** (-omega[ri, si])
            pz[ri, si] = (1.0 / cal.cet_scale[ri, si]) * coef ** (1.0 / e[ri, si]) * pd[ri, si]
    return pz


def _quantities(cal, pd, pq, pz, FD):
    """Given prices and per-region final demand FD[r,s], solve for composite Q, output Z and the
    domestic/import/export splits. The composite market in each region is linear in that region's Q
    once the domestic/import Shephard shares (functions of prices) are known; imports come from
    partners' output. We solve the GLOBAL linear system for all Q[r,s] jointly (imports couple
    regions).

    Shephard (Armington): per unit composite Q[r,s], domestic sD = (1/A)(pq·A)^σ δ_d^σ pd_r^{-σ};
    import input from o, sM_o = (1/A)(pq·A)^σ δ_o^σ pd_o^{-σ}.
    Hotelling (CET): per unit output Z[r,s], domestic sink sZd = (1/B)(pz·B)^{-Ω} ξ_d^{-Ω} pd_r^{Ω};
    export sink to d, sZe_d = (1/B)(pz·B)^{-Ω} ξ_{e,d}^{-Ω} pd_r^{Ω}.
    Consistency: domestic demand D[r,s] = sD·Q[r,s] must equal domestic CET supply sZd·Z[r,s], and
    import M[d,s,o] = sM_o·Q[d,s] must equal export EX[o,s,d] = sZe_d(o)·Z[o,s]."""
    nr, ns = cal.nr, cal.ns
    sigma = cal.arm_elast
    omega = cal.cet_elast
    A, B = cal.arm_scale, cal.cet_scale

    # Armington unit shares.
    sD = np.zeros((nr, ns))
    sM = np.zeros((nr, ns, nr))  # [r,s,o] import of o per unit Q[r,s]
    for ri in range(nr):
        for si in range(ns):
            base = (1.0 / A[ri, si]) * (pq[ri, si] * A[ri, si]) ** sigma[ri, si]
            sD[ri, si] = (
                base * cal.arm_share_d[ri, si] ** sigma[ri, si] * pd[ri, si] ** (-sigma[ri, si])
            )
            for oi in range(nr):
                dm = cal.arm_share_m[ri, si, oi]
                if dm > 0:
                    sM[ri, si, oi] = base * dm ** sigma[ri, si] * pd[oi, si] ** (-sigma[ri, si])
    # CET unit shares.
    sZd = np.zeros((nr, ns))
    sZe = np.zeros((nr, ns, nr))  # [r,s,d] export to d per unit Z[r,s]
    for ri in range(nr):
        for si in range(ns):
            base = (1.0 / B[ri, si]) * (pz[ri, si] * B[ri, si]) ** (-omega[ri, si])
            sZd[ri, si] = (
                base * cal.cet_share_d[ri, si] ** (-omega[ri, si]) * pd[ri, si] ** omega[ri, si]
            )
            for di in range(nr):
                xe = cal.cet_share_e[ri, si, di]
                if xe > 0:
                    sZe[ri, si, di] = base * xe ** (-omega[ri, si]) * pd[ri, si] ** omega[ri, si]

    # Domestic consistency gives Z[r,s] = (sD[r,s]/sZd[r,s]) · Q[r,s].
    ratio = sD / sZd  # [r,s]

    # Composite market per (r,s): Q[r,s] = Σ_j ax[r,j? ] ... intermediates + FD, where intermediates
    # use composite (r,s) by activity (r,i): Σ_i ax[r,s,i]·Z[r,i]. With Z = ratio·Q:
    #   Q[r,s] = Σ_i ax[r,s,i]·ratio[r,i]·Q[r,i] + FD[r,s].
    # This is block-diagonal per region (intermediates are within-region); solve region by region.
    Q = np.zeros((nr, ns))
    for ri in range(nr):
        coeff = cal.ax[ri] * ratio[ri][None, :]  # ax[r, s, i]·ratio[r,i]  → [s, i]
        Q[ri] = np.linalg.solve(np.eye(ns) - coeff, FD[ri])
    Z = ratio * Q
    D = sD * Q
    M = np.zeros((nr, ns, nr))
    EX = np.zeros((nr, ns, nr))
    for ri in range(nr):
        for si in range(ns):
            for oi in range(nr):
                M[ri, si, oi] = sM[ri, si, oi] * Q[ri, si]  # r imports s from o
            for di in range(nr):
                EX[ri, si, di] = sZe[ri, si, di] * Z[ri, si]  # r exports s to d
    return Q, Z, D, EX, M


def derive_multi_state(
    cal: MultiCalibratedModel,
    pd: np.ndarray,
    pq: np.ndarray,
    w: np.ndarray,
    *,
    carbon_cost: np.ndarray | None = None,
    recycling: str = "lump_sum",
) -> MultiModelState:
    """Close the multi-region model at prices (pd, pq, w): derive all quantities and per-region
    income. ``carbon_cost`` is [r,s] (per-region-sector cost wedge). Income per region = factor
    income + er·Sf (er=1 in the single-numéraire reduction) + recycled carbon revenue."""
    nr, ns = cal.nr, cal.ns
    cc = np.zeros((nr, ns)) if carbon_cost is None else np.asarray(carbon_cost, dtype=float)
    pv = _va_unit_cost(cal, w)
    pz = _cet_price(cal, pd)

    # Per-region income closed form (linear in income, like the open model): base = factor income +
    # foreign-savings transfer; recycled carbon revenue adds k·I. Region incomes couple only through
    # prices (already fixed here), so each region's income solves independently.
    factor_income = (w * cal.endowment).sum(axis=0)  # [r]
    base_income = factor_income + cal.foreign_savings  # er=1 in the single-numéraire reduction
    income = base_income.copy()
    if recycling != "none" and np.any(cc != 0.0):
        # z_unit[r,s] = Z[r,s] at unit income for region r; k[r] = Σ_s cc·z_unit.
        FD_unit = cal.gamma / pq  # per-unit-income FD
        _, Zc, *_ = _quantities(cal, pd, pq, pz, FD_unit)
        k = (cc * Zc).sum(axis=1)  # [r]
        k = np.clip(k, None, 1.0 - 1e-9)  # feasibility floor (strict check is at the engine level)
        income = base_income / (1.0 - k)

    FD = cal.gamma * income[:, None] / pq
    Q, Z, D, EX, M = _quantities(cal, pd, pq, pz, FD)
    carbon_revenue = (cc * Z).sum(axis=1)  # [r]
    va_cost = cal.va_share * pv * Z
    F = _factor_demand(cal, w, pv, va_cost)
    return MultiModelState(
        pd=pd,
        pq=pq,
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
    )


def residuals(
    cal: MultiCalibratedModel,
    z: np.ndarray,
    *,
    carbon_cost: np.ndarray | None = None,
    recycling: str = "lump_sum",
    drop_factor: int = 0,
) -> np.ndarray:
    nr, ns, nf = cal.nr, cal.ns, cal.nf
    pd = np.asarray(z[: nr * ns], dtype=float).reshape(nr, ns)
    pq = np.asarray(z[nr * ns : 2 * nr * ns], dtype=float).reshape(nr, ns)
    w = np.asarray(z[2 * nr * ns : 2 * nr * ns + nf * nr], dtype=float).reshape(nf, nr)
    cc = np.zeros((nr, ns)) if carbon_cost is None else np.asarray(carbon_cost, dtype=float)

    st = derive_multi_state(cal, pd, pq, w, carbon_cost=cc, recycling=recycling)
    pv = _va_unit_cost(cal, w)
    pz = _cet_price(cal, pd)

    res: list[float] = []
    # 1. Armington price identity: pq = CES cost of (pd_r, {pd_o}).  [nr·ns]
    pq_id = _armington_price(cal, pd)
    res.extend((pq - pq_id).ravel().tolist())
    # 2. Zero-profit: pz(pd) = Σ_j ax[r,j,i]·pq[r,j] + va_share·pv + cc.  [nr·ns]
    for ri in range(nr):
        for ii in range(ns):
            intermediate = float(np.dot(cal.ax[ri, :, ii], pq[ri]))
            res.append(pz[ri, ii] - (intermediate + cal.va_share[ri, ii] * pv[ri, ii] + cc[ri, ii]))
    # 3. Factor clearing per region-factor, dropping ONE globally by Walras.  [nf·nr − 1]
    flat = 0
    for fi in range(nf):
        for ri in range(nr):
            if flat != drop_factor:
                res.append(float(st.F[fi, ri, :].sum()) - cal.endowment[fi, ri])
            flat += 1
    # 4. Numéraire: region-0 household CPI over its composite good, Π pq[0,s]^γ[0,s] = 1.  [1]
    cpi0 = float(np.prod(np.power(pq[0], cal.gamma[0])))
    res.append(cpi0 - 1.0)
    return np.array(res, dtype=float)


def initial_guess(cal: MultiCalibratedModel) -> np.ndarray:
    """Benchmark start: all prices = 1."""
    return np.ones(2 * cal.nr * cal.ns + cal.nf * cal.nr)


def n_unknowns(cal: MultiCalibratedModel) -> int:
    return 2 * cal.nr * cal.ns + cal.nf * cal.nr
