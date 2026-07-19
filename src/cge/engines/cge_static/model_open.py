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
) -> OpenModelState:
    """Close the open model at prices (pd, pq, w, er): derive all quantities by CES/CET duals.

    ``strict`` controls the recycling k≥1 guard. During solver iteration (``strict=False``) a
    trial price vector may momentarily give k≥1 even when a valid equilibrium (k<1) exists
    elsewhere; raising there would make the residual function discontinuously unavailable and can
    abort a solve that starts at benchmark prices (review P2). So in non-strict mode we **clamp**
    the income multiplier to a large-but-finite value (steering the solver away smoothly) instead
    of raising. On the **final accepted equilibrium** the engine calls with ``strict=True``, which
    raises if k≥1 — so a genuinely infeasible equilibrium is still refused, never returned."""
    ns = len(cal.sectors)
    cc = np.zeros(ns) if carbon_cost is None else np.asarray(carbon_cost, dtype=float)
    pm = er * np.ones(ns)  # world import price 1
    pe = er * np.ones(ns)  # world export price 1
    pv = _va_unit_cost(cal, w)
    pz = _cet_price(cal, pd, pe)

    # Household income = factor income + recycled carbon revenue. At fixed prices every quantity is
    # LINEAR in income (FD = γ·I/pq, Q solves a linear system in FD, Z = ratio·Q), so the recycling
    # fixed point I = factor_income + cc·Z(I) is linear and solves in closed form — no iteration, no
    # silent non-convergence (review P2). Writing Z = I·z_unit with z_unit = Z evaluated at I=1:
    #   I = factor_income + k·I,  k = cc·z_unit  ⇒  I = factor_income / (1 − k),  guarded k < 1.
    factor_income = float(np.dot(w, cal.endowment))
    if recycling != "none" and np.any(cc != 0.0):
        _, z_unit, *_ = _quantities(cal, pd, pq, pm, pe, pz, cal.gamma * 1.0 / pq)
        k = float(cc @ z_unit)
        if strict and k >= 1.0 - 1e-12:
            raise ValueError(
                f"open recycling fixed point diverges: carbon revenue ≥ income (k={k:.6g} ≥ 1) "
                "at the equilibrium; the recycled transfer would exceed factor income. Lower the "
                "carbon price."
            )
        # Non-strict: a smooth positive floor on (1−k) keeps income finite and the residual
        # C¹-continuous with a restoring gradient when a trial point has k≥1 (review P2); it is the
        # identity for a real equilibrium (1−k well above the floor).
        income = factor_income / _safe_denom(1.0 - k)
    else:
        income = factor_income
    FD = cal.gamma * income / pq
    Q, Z, D, E, M = _quantities(cal, pd, pq, pm, pe, pz, FD)
    carbon_revenue = float(cc @ Z)
    # Independent check of the income identity we just solved (cheap; catches any regression in the
    # linearity assumption above). Only in strict mode — the non-strict clamp deliberately violates
    # the identity to steer the solver, so checking it there would defeat the smooth-penalty design.
    if strict:
        resid = income - (factor_income + (carbon_revenue if recycling != "none" else 0.0))
        if abs(resid) > 1e-9 * max(1.0, abs(income)):  # pragma: no cover - guards the closed form
            raise ValueError(f"open income identity not satisfied (residual {resid:.3e}).")
    # Factor demand (Shephard on VA cost).
    va_cost = cal.va_share * pv * Z
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
    share_e = np.where(cal.cet_share_e > 0, cal.cet_share_e, 1.0)  # safe base; masked below
    sZe = np.where(
        cal.cet_share_e > 0,
        (1.0 / B) * np.power(pz * B, -omega) * share_e ** (-omega) * np.power(pe, omega),
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
    # Armington price identity: pq = CES cost of (pd, pm).  [ns rows]
    pq_id = _armington_price(cal, pd, pm)
    res.extend((pq - pq_id).tolist())
    # Zero-profit: activity output price pz(pd,pe) = intermediate composite cost + VA + carbon.
    # [ns rows]
    pz = _cet_price(cal, pd, pe)
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
