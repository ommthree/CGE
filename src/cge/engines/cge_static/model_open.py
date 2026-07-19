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

**Residuals (square):**
- ns Armington-price identities;   - ns zero-profit (pz) identities;
- ns composite-market clearing ``Q_i = Σ_j ax[i,j] Z_j + FD_i``;
- (nf−1) factor-market clearing (one dropped by Walras);
- 1 trade balance ``Σ pm_i M_i − Σ pe_i E_i − er·Sf = 0``;
- 1 numéraire ``Π_i pq_i^{γ_i} = 1`` (the household CPI over the composite good).
Total: 3·ns + nf + 1, matching the 2·ns + nf + 1 unknowns plus ns pz unknowns — see ``n_unknowns``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cge.engines.cge_static.calibrate_open import OpenCalibratedModel


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


def _va_unit_cost(cal: OpenCalibratedModel, w: np.ndarray) -> np.ndarray:
    beta = cal.beta
    ratio = np.where(beta > 0, w[:, None] / np.where(beta > 0, beta, 1.0), 1.0)
    return (1.0 / cal.av) * np.prod(np.power(ratio, beta), axis=0)


def _armington_price(cal: OpenCalibratedModel, pd: np.ndarray, pm: np.ndarray) -> np.ndarray:
    """CES cost function for the composite: pq = (1/A)[δ^σ pd^{1-σ}+(1-δ)^σ pm^{1-σ}]^{1/(1-σ)}.
    A non-traded good (δ=1, share_m=0) collapses to pq = pd/A."""
    sigma = cal.arm_elast
    e = 1.0 - sigma
    inner = cal.arm_share_d**sigma * pd**e + cal.arm_share_m**sigma * pm**e
    return (1.0 / cal.arm_scale) * np.power(inner, 1.0 / e)


def _cet_price(cal: OpenCalibratedModel, pd: np.ndarray, pe: np.ndarray) -> np.ndarray:
    """CET revenue function for output: pz = (1/B)[ξ^{-Ω} pd^{1+Ω}+(1-ξ)^{-Ω} pe^{1+Ω}]^{1/(1+Ω)}
    (verified via Hotelling; the negative share exponents distinguish the convex CET frontier)."""
    omega = cal.cet_elast
    e = 1.0 + omega
    inner = cal.cet_share_d ** (-omega) * pd**e + cal.cet_share_e ** (-omega) * pe**e
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
) -> OpenModelState:
    """Close the open model at prices (pd, pq, w, er): derive all quantities by CES/CET duals."""
    ns = len(cal.sectors)
    cc = np.zeros(ns) if carbon_cost is None else np.asarray(carbon_cost, dtype=float)
    pm = er * np.ones(ns)  # world import price 1
    pe = er * np.ones(ns)  # world export price 1
    pv = _va_unit_cost(cal, w)
    pz = _cet_price(cal, pd, pe)

    # Household income = factor income + recycled carbon revenue. Solve the recycling fixed point
    # by iterating on income (X depends on FD depends on income depends on revenue) — a couple of
    # passes converge; the residual system's solver drives the outer prices anyway.
    factor_income = float(np.dot(w, cal.endowment))
    income = factor_income
    for _ in range(50):
        FD = cal.gamma * income / pq
        # Composite supply Q must meet intermediate + final demand. Intermediates use composite j
        # per unit activity output; but activity output Z is itself set by CET from D. Solve the
        # linear intermediate system for Z given FD via the domestic/composite identities below.
        # Quantities from CES/CET Shephard/Hotelling (given prices), scaled to meet demand:
        # First get unit split ratios, then scale by the market-clearing Q.
        Q, Z, D, E, M = _quantities(cal, pd, pq, pm, pe, pz, FD)
        carbon_revenue = float(cc @ Z)
        new_income = factor_income + (carbon_revenue if recycling != "none" else 0.0)
        if abs(new_income - income) < 1e-13 * max(1.0, abs(income)):
            income = new_income
            break
        income = new_income
    FD = cal.gamma * income / pq
    Q, Z, D, E, M = _quantities(cal, pd, pq, pm, pe, pz, FD)
    carbon_revenue = float(cc @ Z)
    # Factor demand (Shephard on VA cost).
    va_cost = cal.va_share * pv * Z
    F = cal.beta * va_cost[None, :] / w[:, None]
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
    )


def _quantities(cal, pd, pq, pm, pe, pz, FD):
    """Given prices and final demand, solve for composite Q, output Z, and the D/E/M splits.

    Composite market: Q_i = Σ_j ax[i,j] Z_j + FD_i.  Domestic sales D are shared between supplying
    Q (Armington) and coming from Z (CET); at equilibrium the CET domestic supply equals the
    Armington domestic demand. Using the CES/CET Shephard shares (functions of prices only), write
    D_i = s^Q_i Q_i (Armington domestic input per composite) and D_i = s^Z_i Z_i (CET domestic
    output per activity output), so Z_i = (s^Q_i / s^Z_i) Q_i. Then the composite market is linear
    in Q:  Q = ax^T diag(s^Q/s^Z) Q + FD."""
    ns = len(FD)
    sigma = cal.arm_elast
    omega = cal.cet_elast
    A, B = cal.arm_scale, cal.cet_scale
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
    sZe = np.where(
        cal.cet_share_e > 0,
        (1.0 / B) * np.power(pz * B, -omega) * cal.cet_share_e ** (-omega) * np.power(pe, omega),
        0.0,
    )
    # Z in terms of Q via domestic-sales consistency: D = sQ·Q (Armington) = sZd·Z (CET) ⇒
    # Z = (sQ/sZd)·Q.
    ratio = sQ / sZd  # [i]
    # Composite market: Q_i = Σ_j ax[i,j] Z_j + FD_i = Σ_j ax[i,j] ratio_j Q_j + FD_i.
    coeff = cal.ax * ratio[None, :]  # ax[i,j]·ratio[j]
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
) -> np.ndarray:
    ns = len(cal.sectors)
    nf = len(cal.factors)
    pd = np.asarray(z[:ns], dtype=float)
    pq = np.asarray(z[ns : 2 * ns], dtype=float)
    w = np.asarray(z[2 * ns : 2 * ns + nf], dtype=float)
    er = float(z[2 * ns + nf])
    cc = np.zeros(ns) if carbon_cost is None else np.asarray(carbon_cost, dtype=float)

    st = derive_open_state(cal, pd, pq, w, er, carbon_cost=cc, recycling=recycling)
    pm = er * np.ones(ns)
    pe = er * np.ones(ns)
    pv = _va_unit_cost(cal, w)

    res = []
    # Armington price identity: pq = CES cost of (pd, pm).
    pq_id = _armington_price(cal, pd, pm)
    res.extend((pq - pq_id).tolist())
    # Zero-profit: activity output price pz(pd,pe) = intermediate composite cost + VA + carbon.
    pz = _cet_price(cal, pd, pe)
    for i in range(ns):
        intermediate = sum(cal.ax[j, i] * pq[j] for j in range(ns))
        res.append(pz[i] - (intermediate + cal.va_share[i] * pv[i] + cc[i]))
    # Composite market clearing: Q_i = Σ_j ax[i,j] Z_j + FD_i.
    intermediate_use = cal.ax @ st.Z
    res.extend((st.Q - (intermediate_use + st.FD)).tolist())
    # Factor clearing (drop one by Walras).
    for f in range(nf):
        if f == drop_factor:
            continue
        res.append(float(st.F[f, :].sum()) - cal.endowment[f])
    # Trade balance: Σ pm·M − Σ pe·E − er·Sf = 0.
    res.append(float(pm @ st.M - pe @ st.E - er * cal.foreign_savings))
    # Numéraire: household CPI over the composite good, Π pq_i^γ_i = 1.
    cpi = float(np.prod(np.power(pq, cal.gamma)))
    res.append(cpi - 1.0)
    return np.array(res, dtype=float)


def initial_guess(cal: OpenCalibratedModel) -> np.ndarray:
    """Benchmark start: all prices and the exchange rate = 1."""
    return np.ones(2 * len(cal.sectors) + len(cal.factors) + 1)


def n_unknowns(cal: OpenCalibratedModel) -> int:
    return 2 * len(cal.sectors) + len(cal.factors) + 1
