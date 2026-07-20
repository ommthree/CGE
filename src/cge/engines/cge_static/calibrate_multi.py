"""Calibration for the MULTI-REGION static CGE (Phase 5.4 — bilateral trade).

Generalises the single-region-open calibration to ``R`` regions that trade bilaterally, in a closed
global economy [Hosoe2010, ch. 7 generalised]. Each region ``r`` has the same sectors, its own
(immobile) factors and household, and:

- **Armington** (per region ``r``, commodity ``s``): the composite ``Q[r,s]`` is a CES over the
  region's own domestic variety ``D[r,s]`` and imports from **every other region** ``M[r,s,o]``
  (``o≠r``). Benchmark shares read off the bilateral trade block.
- **CET** (per region ``r``, sector ``s``): output ``Z[r,s]`` is transformed into domestic sales
  ``DS[r,s]`` and exports to **every other region** ``EX[r,s,d]`` (``d≠r``).
- Bilateral consistency: ``M[d,s,o] = EX[o,s,d]`` at benchmark (region ``d``'s import of ``s`` from
  ``o`` = region ``o``'s export of ``s`` to ``d``).

World prices are the model's own equilibrium prices (no external ROW); the benchmark exchange rate
per region is 1. Foreign savings per region ``Sf[r] = Σ imports − Σ exports`` is closed by a
household capital transfer (a globally zero-sum vector); the model fixes it at benchmark.

All benchmark prices = 1, so every share reads off the balanced SAM at unit prices. For a nested CES
over ``n`` sources with a single elasticity, the source shares are ``δ_k ∝ V_k^{1/σ}`` (value shares
raised to ``1/σ``) and the scale makes the composite price 1 at unit prices — the same Shephard
calibration as the 2-source open model, generalised to ``n`` sources.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cge.contracts.data_objects import SAM
from cge.engines.cge_static.calibrate_open import _elast_vector

DEFAULT_ARMINGTON_ELAST = 2.0
DEFAULT_CET_ELAST = 2.0


@dataclass(frozen=True)
class MultiCalibratedModel:
    """Benchmark parameters for the multi-region CGE. All benchmark prices = 1.

    Array axes: ``r`` = region, ``s`` = sector, ``o``/``d`` = partner region. Region-pair arrays are
    ``[r, s, partner]`` with the own-region slot carrying the domestic term (0 for pure trade
    arrays)."""

    regions: list[str]
    sectors: list[str]
    factors: list[str]  # base factor names (per region: f"{f}_{r}")
    # technology
    ax: np.ndarray  # [r, j, i] intermediate composite j per unit activity output i in region r
    va_share: np.ndarray  # [r, i]
    beta: np.ndarray  # [f, r, i] CD value-added factor shares (over factors, per region-sector)
    av: np.ndarray  # [r, i]
    va_elast: np.ndarray  # [r, i]
    va_ces_share: np.ndarray  # [f, r, i]
    gamma: np.ndarray  # [r, i] household budget shares (sum to 1 per region)
    endowment: np.ndarray  # [f, r] factor endowment per region
    # Armington (imports): composite Q[r,s] over domestic D[r,s] + imports from o≠r.
    arm_elast: np.ndarray  # [r, s]
    arm_scale: np.ndarray  # [r, s]
    arm_share_d: np.ndarray  # [r, s] domestic share δ^d
    arm_share_m: np.ndarray  # [r, s, o] import share from region o (0 on the own slot)
    # CET (exports): output Z[r,s] over domestic DS[r,s] + exports to d≠r.
    cet_elast: np.ndarray  # [r, s]
    cet_scale: np.ndarray  # [r, s]
    cet_share_d: np.ndarray  # [r, s] domestic-sales share ξ^d
    cet_share_e: np.ndarray  # [r, s, d] export share to region d (0 on the own slot)
    foreign_savings: np.ndarray  # [r] Sf = Σ imports − Σ exports (globally zero-sum)
    # benchmark quantities (= money at unit prices), GDP-normalised
    Z0: np.ndarray  # [r, s] activity output
    D0: np.ndarray  # [r, s] domestic sales (used by own composite)
    EX0: np.ndarray  # [r, s, d] exports to region d
    M0: np.ndarray  # [r, s, o] imports from region o
    Q0: np.ndarray  # [r, s] composite supply
    FD0: np.ndarray  # [r, s] household final demand
    F0: np.ndarray  # [f, r, s] factor demand
    INT0: np.ndarray  # [r, j, i] intermediate use

    @property
    def gdp0(self) -> float:
        return float(self.F0.sum())

    @property
    def nr(self) -> int:
        return len(self.regions)

    @property
    def ns(self) -> int:
        return len(self.sectors)

    @property
    def nf(self) -> int:
        return len(self.factors)

    @property
    def active_routes(self) -> list[tuple[int, int, int]]:
        """Ordered ``(o, s, d)`` triples for routes with genuine benchmark trade — ``M0[d,s,o] > 0``
        (equivalently ``EX0[o,s,d] > 0``; the two coincide at a balanced benchmark). A route absent
        from this list has **no** price unknown and **no** clearing residual (review P1): packing an
        unknown for a structurally-zero route left its price undetermined — perturbing it by 1.5
        left every residual at machine-zero, a genuine rank deficiency (Jacobian rank dropped by
        exactly the number of zero routes), so the solver's "convergence" on a sparse SAM did not
        pin a unique equilibrium. Real SAMs are commonly sparse (most region pairs don't trade every
        good), so we pack only active routes rather than rejecting sparse topology outright."""
        routes = []
        for oi in range(self.nr):
            for di in range(self.nr):
                if oi == di:
                    continue
                for si in range(self.ns):
                    if self.M0[di, si, oi] > 0 or self.EX0[oi, si, di] > 0:
                        routes.append((oi, si, di))
        return routes


def calibrate_multi(
    sam: SAM,
    *,
    regions: list[str],
    sectors: list[str],
    factors: list[str],
    arm_elast: float = DEFAULT_ARMINGTON_ELAST,
    cet_elast: float = DEFAULT_CET_ELAST,
    va_elast: float = 1.0,
) -> MultiCalibratedModel:
    """Calibrate the multi-region CGE from a globally-balanced multi-region SAM with
    ``a_<r>_<s>``/``c_<r>_<s>`` activity/commodity accounts, per-region factors ``<f>_<r>`` and
    households ``HOH_<r>``. Elasticities are scalars applied to every region-sector (a per-cell
    vector is a later refinement)."""
    m = sam.matrix
    nr, ns = len(regions), len(sectors)

    def a(r, s):
        return f"a_{r}_{s}"

    def c(r, s):
        return f"c_{r}_{s}"

    # Benchmark flows (prices = 1 ⇒ money = quantity).
    D0 = np.array([[m.loc[a(r, s), c(r, s)] for s in sectors] for r in regions], dtype=float)
    EX0 = np.zeros((nr, ns, nr))  # [r, s, d] region r exports s to d
    M0 = np.zeros((nr, ns, nr))  # [r, s, o] region r imports s from o
    for ri, r in enumerate(regions):
        for si, s in enumerate(sectors):
            for di, d in enumerate(regions):
                if d == r:
                    continue
                EX0[ri, si, di] = m.loc[a(r, s), c(d, s)]  # r's activity → d's composite
                M0[ri, si, di] = m.loc[a(d, s), c(r, s)]  # d's activity → r's composite = r imports

    Z0 = D0 + EX0.sum(axis=2)  # activity output = domestic sales + all exports
    INT0 = np.zeros((nr, ns, ns))  # [r, j, i] composite j used by activity i in region r
    for ri, r in enumerate(regions):
        for ji, j in enumerate(sectors):
            for ii, i in enumerate(sectors):
                INT0[ri, ji, ii] = m.loc[c(r, j), a(r, i)]
    F0 = np.array(
        [[[m.loc[f"{f}_{r}", a(r, s)] for s in sectors] for r in regions] for f in factors],
        dtype=float,
    )
    FD0 = np.array([[m.loc[c(r, s), f"HOH_{r}"] for s in sectors] for r in regions], dtype=float)
    # Composite use of commodity s in region r = Σ_i INT0[r, s, i] + FD (s used by each activity i).
    # INT0[r,j,i] is composite j used by activity i, so use of commodity s = INT0[r,s,:].sum().
    Q0 = INT0.sum(axis=2) + FD0

    # Normalise by global GDP.
    scale = float(F0.sum())
    if scale <= 0:
        raise ValueError("multi-region SAM has non-positive total value added; cannot calibrate")
    D0, EX0, M0, Z0, INT0, F0, FD0, Q0 = (
        arr / scale for arr in (D0, EX0, M0, Z0, INT0, F0, FD0, Q0)
    )

    if float(Z0.min()) <= 0 or float(Q0.min()) <= 0 or float(D0.min()) <= 0:
        raise ValueError("multi-region SAM has non-positive output / composite / domestic sales")

    # Value added, technology, household (per region-sector; mirrors the open model).
    VA0 = Z0 - INT0.sum(axis=1)  # output − Σ_j INT[r,j,i]
    if float(VA0.min()) <= 0:
        raise ValueError("multi-region SAM has non-positive value added for some activity")
    va_share = VA0 / Z0
    ax = INT0 / Z0[:, None, :]
    beta = F0 / VA0[None, :, :]
    sigma_va = np.full((nr, ns), float(_elast_vector(va_elast, 1, "va_elast")[0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        w_ces = np.where(F0 > 0, np.power(F0, 1.0 / sigma_va[None, :, :]), 0.0)
        va_ces_share = w_ces / w_ces.sum(axis=0)[None, :, :]
    av = np.empty((nr, ns))
    for ri in range(nr):
        for si in range(ns):
            s = sigma_va[ri, si]
            if abs(s - 1.0) < 1e-12:
                b = beta[:, ri, si]
                av[ri, si] = np.prod(
                    np.power(np.where(b > 0, 1.0 / np.where(b > 0, b, 1.0), 1.0), b)
                )
            else:
                d = va_ces_share[:, ri, si]
                av[ri, si] = np.power(np.sum(d**s), 1.0 / (1.0 - s))
    gamma = FD0 / FD0.sum(axis=1)[:, None]
    endowment = F0.sum(axis=2)  # [f, r]

    # Validate each elasticity is finite and strictly positive (reuse the open model's validator),
    # then broadcast to every region-sector; σ=1 is the singular Cobb-Douglas case (review P2: a
    # negative elasticity used to be accepted).
    arm_scalar = float(_elast_vector(arm_elast, 1, "arm_elast")[0])
    cet_scalar = float(_elast_vector(cet_elast, 1, "cet_elast")[0])
    arm = np.full((nr, ns), arm_scalar)
    cet = np.full((nr, ns), cet_scalar)
    if abs(arm_scalar - 1.0) < 1e-9 or abs(cet_scalar - 1.0) < 1e-9:
        raise ValueError(
            "Armington/CET elasticities must be ≠ 1 (σ=1 is the Cobb-Douglas special case, not "
            "implemented for trade); typical values are 1.5–5."
        )

    # Armington: composite Q[r,s] over sources {domestic D[r,s]} ∪ {imports M[r,s,o]}. For a single
    # elasticity the source shares are δ_k ∝ V_k^{1/σ} (value shares^{1/σ}); scale makes price 1.
    arm_scale, arm_share_d, arm_share_m = _calibrate_arm(D0, M0, Q0, arm)
    # CET: output Z[r,s] over sinks {domestic DS[r,s]} ∪ {exports EX[r,s,d]}. Revenue-fn dual with
    # negative exponents (convex frontier), generalised to n sinks.
    cet_scale, cet_share_d, cet_share_e = _calibrate_cet(D0, EX0, Z0, cet)

    foreign_savings = M0.sum(axis=(1, 2)) - EX0.sum(axis=(1, 2))  # [r]

    return MultiCalibratedModel(
        regions=list(regions),
        sectors=list(sectors),
        factors=list(factors),
        ax=ax,
        va_share=va_share,
        beta=beta,
        av=av,
        va_elast=sigma_va,
        va_ces_share=va_ces_share,
        gamma=gamma,
        endowment=endowment,
        arm_elast=arm,
        arm_scale=arm_scale,
        arm_share_d=arm_share_d,
        arm_share_m=arm_share_m,
        cet_elast=cet,
        cet_scale=cet_scale,
        cet_share_d=cet_share_d,
        cet_share_e=cet_share_e,
        foreign_savings=foreign_savings,
        Z0=Z0,
        D0=D0,
        EX0=EX0,
        M0=M0,
        Q0=Q0,
        FD0=FD0,
        F0=F0,
        INT0=INT0,
    )


def _calibrate_arm(D0, M0, Q0, arm):
    """Armington CES over {domestic} ∪ {imports from each o}. Shares δ_k ∝ V_k^{1/σ} normalised to
    sum to 1; scale A so the composite price = 1 at unit prices:
        Q = A·[δ_d·D^ρ + Σ_o δ_o·M_o^ρ]^{1/ρ}, ρ=(σ−1)/σ, all benchmark prices 1.
    With value shares v_k = V_k/Q and ρ, at unit prices the calibrated A = Q / [Σ_k δ_k V_k^ρ]^{1/ρ}
    where δ_k = v_k^{1/σ} / Σ_l v_l^{1/σ}."""
    nr, ns, _ = M0.shape
    rho = (arm - 1.0) / arm  # [r,s]
    share_d = np.zeros((nr, ns))
    share_m = np.zeros_like(M0)
    scale = np.zeros((nr, ns))
    for ri in range(nr):
        for si in range(ns):
            sigma = arm[ri, si]
            vals = np.concatenate([[D0[ri, si]], M0[ri, si, :]])  # [domestic, imports...]
            pos = vals > 0
            w = np.zeros_like(vals)
            w[pos] = np.power(vals[pos], 1.0 / sigma)
            delta = w / w.sum()
            r_ = rho[ri, si]
            inner = np.sum(np.where(pos, delta * np.power(np.where(pos, vals, 1.0), r_), 0.0))
            scale[ri, si] = Q0[ri, si] / np.power(inner, 1.0 / r_)
            share_d[ri, si] = delta[0]
            share_m[ri, si, :] = delta[1:]
    return scale, share_d, share_m


def _calibrate_cet(D0, EX0, Z0, cet):
    """CET over {domestic sales} ∪ {exports to each d}. Revenue-max dual: share_k ∝ V_k^{−1/Ω}
    (note the NEGATIVE exponent — convex transformation frontier), scale B so price = 1 at unit
    prices: Z = B·[ξ_d·DS^κ + Σ_d ξ_d·EX_d^κ]^{1/κ}, κ=(Ω+1)/Ω."""
    nr, ns, _ = EX0.shape
    kappa = (cet + 1.0) / cet
    share_d = np.zeros((nr, ns))
    share_e = np.zeros_like(EX0)
    scale = np.zeros((nr, ns))
    for ri in range(nr):
        for si in range(ns):
            omega = cet[ri, si]
            vals = np.concatenate([[D0[ri, si]], EX0[ri, si, :]])
            pos = vals > 0
            w = np.zeros_like(vals)
            w[pos] = np.power(vals[pos], -1.0 / omega)  # negative exponent
            xi = w / w.sum()
            k_ = kappa[ri, si]
            # Quantity aggregator uses the PLAIN shares ξ_k (the ^{-Ω} appears only in the price
            # dual): Z = B·[Σ_k ξ_k · y_k^κ]^{1/κ} ⇒ B = Z / [Σ_k ξ_k · V_k^κ]^{1/κ}. (Matches the
            # single-region open model's cet_scale; the earlier ξ^{-Ω} here made pz ≠ 1.)
            inner = np.sum(np.where(pos, xi * np.power(np.where(pos, vals, 1.0), k_), 0.0))
            scale[ri, si] = Z0[ri, si] / np.power(inner, 1.0 / k_)
            share_d[ri, si] = xi[0]
            share_e[ri, si, :] = xi[1:]
    return scale, share_d, share_e
