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
from cge.engines.cge_static.energy_nest import (
    DEFAULT_ENERGY_ELAST,
    DEFAULT_KL_E_ELAST,
    DEFAULT_KLE_M_ELAST,
    calibrate_energy_nest,
)

DEFAULT_ARMINGTON_ELAST = 2.0
DEFAULT_CET_ELAST = 2.0

# Materiality threshold for "does this route genuinely trade" (review P1, round 10): benchmark
# flows are GDP-normalised (divided by total global value added — see calibrate_multi below), so
# this is a *share of global GDP*, comparable across builds. A route below this is treated as
# structurally zero even if its raw SAM cell is a nonzero float — numerical noise from a prior
# aggregation/RAS-balancing step can leave a route at ~1e-10 of GDP, which is not "trade" in any
# economic sense but IS enough to make `> 0` pack a price unknown for it. That unknown is then
# almost entirely unconstrained (the residual system's sensitivity to it is only as large as the
# route's own tiny share), producing a near-singular Jacobian (condition number in the 1e12 range)
# that a tolerance-based solver can accept as "converged" while the route's price is actually free
# to many significant figures.
ROUTE_MATERIALITY_THRESHOLD = 1e-6


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
    # Government accounts (Phase 5d.1; None ⇒ no governments declared — pre-5d.1 behaviour). One
    # ``GOV_<r>`` account PER REGION (all-or-nothing: a partial layout is rejected — mixing
    # government and household revenue routing across regions is more likely a mistake than a
    # design). Mirrors the closed/open models' fields, one entry per region.
    gov_gamma: np.ndarray | None = None  # [r, s] government CD demand shares (sum→1 per region)
    gov_income0: np.ndarray | None = None  # [r] benchmark government income (GDP-normalised)
    GD0: np.ndarray | None = None  # [r, s] benchmark government demand (GDP-normalised)
    gov_tax_rate0: np.ndarray | None = None  # [r] benchmark tax rate on regional factor income
    # Savings-investment accounts (Phase 5d.2; None ⇒ pre-5d.2 behaviour, the bilateral capital
    # transfer Sf_r goes to households). One ``SAVINV_<r>`` per region, all-or-nothing. With them,
    # each region's investment = its household savings + its Sf_r (capital transfers route between
    # the SAVINV accounts, not between households).
    inv_gamma: np.ndarray | None = None  # [r, s] investment demand composition (sum→1 per region)
    INV0: np.ndarray | None = None  # [r, s] benchmark investment demand (GDP-normalised)
    sav_rate0: np.ndarray | None = None  # [r] savings rate on regional disposable income
    # Energy nest (Phase 5d.5; None ⇒ flat Leontief production, bit-identical to pre-5d.5). ONE
    # EnergyNest per region — each region's nest is calibrated over that region's own composite
    # intermediate flows, so a carbon price shifts substitution within each region's energy bundle.
    energy_nests: list | None = None  # list[EnergyNest] of length nr, or None

    @property
    def gdp0(self) -> float:
        return float(self.F0.sum())

    @property
    def has_government(self) -> bool:
        return self.gov_gamma is not None

    @property
    def has_investment(self) -> bool:
        return self.inv_gamma is not None

    @property
    def has_energy_nest(self) -> bool:
        return self.energy_nests is not None

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
        """Ordered ``(o, s, d)`` triples for routes with genuine (materially above-threshold)
        benchmark trade — ``M0[d,s,o] > ROUTE_MATERIALITY_THRESHOLD`` (equivalently
        ``EX0[o,s,d]``; the two coincide at a balanced benchmark, both GDP-normalised). A route
        absent from this list has **no** price unknown and **no** clearing residual (review P1):
        packing an unknown for a structurally-zero route left its price undetermined — perturbing
        it left every residual at machine-zero, a genuine rank deficiency, so the solver's
        "convergence" on a sparse SAM did not pin a unique equilibrium. Real SAMs are commonly
        sparse (most region pairs don't trade every good), so we pack only active routes rather
        than rejecting sparse topology outright. The threshold (not a bare ``> 0``) additionally
        excludes numerical dust — a route at ~1e-10 of GDP from upstream aggregation/RAS noise is
        not "trade" in any economic sense, but a bare `>0` check would pack a price unknown for it
        anyway, producing a near-singular (condition number ~1e12) Jacobian that a tolerance-based
        solver can accept as converged while that route's price is actually free (round-10
        follow-up review)."""
        routes = []
        for oi in range(self.nr):
            for di in range(self.nr):
                if oi == di:
                    continue
                for si in range(self.ns):
                    if (
                        self.M0[di, si, oi] > ROUTE_MATERIALITY_THRESHOLD
                        or self.EX0[oi, si, di] > ROUTE_MATERIALITY_THRESHOLD
                    ):
                        routes.append((oi, si, di))
        return routes

    @property
    def connected_components(self) -> list[set[int]]:
        """Partition the ``nr`` regions into connected components under `active_routes` (undirected
        connectivity — a route in either direction links the two regions). A single global
        numéraire and a single globally-dropped factor-market equation are only a valid closure
        when the region graph is **connected**: two regions with no active trade route between
        them (directly or via a chain of intermediaries) are, mathematically, two independent
        economies glued into one residual system with one numéraire and one dropped equation too
        few between them — the extra region's overall price level is genuinely underdetermined,
        not just numerically delicate (review P1, round 10: reproduced a disconnected 2-region SAM
        where scaling one region's entire price vector by 1.7× left every residual unchanged to
        within 1e-15, i.e. a real (not just near-) singular direction the Jacobian's
        default-tolerance rank check can fail to flag)."""
        parent = list(range(self.nr))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for oi, _si, di in self.active_routes:
            union(oi, di)
        groups: dict[int, set[int]] = {}
        for ri in range(self.nr):
            groups.setdefault(find(ri), set()).add(ri)
        return list(groups.values())


def calibrate_multi(
    sam: SAM,
    *,
    regions: list[str],
    sectors: list[str],
    factors: list[str],
    arm_elast: float = DEFAULT_ARMINGTON_ELAST,
    cet_elast: float = DEFAULT_CET_ELAST,
    va_elast: float = 1.0,
    government: bool = False,
    savings_investment: bool = False,
    energy_sectors: list[str] | None = None,
    energy_elasticities: dict[str, float] | None = None,
) -> MultiCalibratedModel:
    """Calibrate the multi-region CGE from a globally-balanced multi-region SAM with
    ``a_<r>_<s>``/``c_<r>_<s>`` activity/commodity accounts, per-region factors ``<f>_<r>`` and
    households ``HOH_<r>``. Elasticities are scalars applied to every region-sector (a per-cell
    vector is a later refinement).

    ``government=True`` (Phase 5d.1) reads one ``GOV_<r>`` account per region (by convention,
    like ``HOH_<r>`` — all regions must have one): each region's government buys its OWN region's
    composites (its ``c_<r>_<s>`` column) financed by a ``HOH_<r>``→``GOV_<r>`` direct tax,
    stored as a rate on that region's factor income. Cross-region government flows, production/
    factor taxes and transfers are rejected explicitly (5d follow-ups).

    ``savings_investment=True`` (Phase 5d.2) reads one ``SAVINV_<r>`` per region (all-or-nothing).
    Each buys its own region's composites (investment demand), financed by its household's savings
    (a rate on regional disposable income) **plus that region's foreign savings Sf_r — the
    bilateral capital transfers must route between the SAVINV accounts, not between households**
    (the 5d.2 closure change: capital flows finance investment, not consumption; a
    cross-region HOH↔HOH transfer is rejected when SAVINV accounts are present)."""
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
    GD0 = None
    gov_tax0 = None
    if government:
        missing = [f"GOV_{r}" for r in regions if f"GOV_{r}" not in sam.accounts]
        if missing:
            raise ValueError(
                f"government=True needs a GOV_<r> account for every region; missing {missing} "
                "(a partial government layout is rejected — all regions or none)."
            )
        GD0 = np.array(
            [[m.loc[c(r, s), f"GOV_{r}"] for s in sectors] for r in regions], dtype=float
        )
        gov_tax0 = np.array([m.loc[f"GOV_{r}", f"HOH_{r}"] for r in regions], dtype=float)
        # 5d.1 discipline (same as closed/open): the ONLY supported government flows are own-region
        # commodity purchases financed by the own household's direct tax. Check every other cell in
        # each GOV row/column is zero.
        for ri, r in enumerate(regions):
            g = f"GOV_{r}"
            own_com = {c(r, s) for s in sectors}
            bad_in = [
                a
                for a in sam.accounts
                if a != f"HOH_{r}" and abs(float(m.loc[g, a])) > 1e-9 and a != g
            ]
            bad_out = [
                a
                for a in sam.accounts
                if a not in own_com and abs(float(m.loc[a, g])) > 1e-9 and a != g
            ]
            if bad_in:
                raise ValueError(
                    f"{g} receives from accounts {bad_in}: only a HOH_{r}→{g} direct tax is "
                    "modelled (Phase 5d follow-up for production/factor taxes, cross-region "
                    "flows)."
                )
            if bad_out:
                raise ValueError(
                    f"{g} pays accounts {bad_out}: a region's government may only buy its own "
                    "region's composites (its c_<r>_<s> column); transfers and cross-region "
                    "purchases are Phase 5d follow-ups."
                )
            if abs(gov_tax0[ri] - float(GD0[ri].sum())) > 1e-6 * max(1.0, gov_tax0[ri]):
                raise ValueError(
                    f"{g} is unbalanced: benchmark receipts {gov_tax0[ri]:.6g} ≠ spending "
                    f"{float(GD0[ri].sum()):.6g}."
                )
    INV0 = None
    sav0 = None
    if savings_investment:
        missing = [f"SAVINV_{r}" for r in regions if f"SAVINV_{r}" not in sam.accounts]
        if missing:
            raise ValueError(
                f"savings_investment=True needs a SAVINV_<r> account for every region; missing "
                f"{missing} (all regions or none)."
            )
        INV0 = np.array(
            [[m.loc[c(r, s), f"SAVINV_{r}"] for s in sectors] for r in regions], dtype=float
        )
        sav0 = np.array([m.loc[f"SAVINV_{r}", f"HOH_{r}"] for r in regions], dtype=float)
        savinv_names = {f"SAVINV_{r}" for r in regions}
        for r in regions:
            si_acc = f"SAVINV_{r}"
            own_com = {c(r, s) for s in sectors}
            allowed_in = {f"HOH_{r}"} | (savinv_names - {si_acc})
            bad_in = [
                a
                for a in sam.accounts
                if a not in allowed_in and abs(float(m.loc[si_acc, a])) > 1e-9 and a != si_acc
            ]
            allowed_out = own_com | (savinv_names - {si_acc})
            bad_out = [
                a
                for a in sam.accounts
                if a not in allowed_out and abs(float(m.loc[a, si_acc])) > 1e-9 and a != si_acc
            ]
            if bad_in:
                raise ValueError(
                    f"{si_acc} receives from accounts {bad_in}: only HOH_{r} savings and "
                    "inter-SAVINV capital transfers are modelled (Phase 5d follow-up)."
                )
            if bad_out:
                raise ValueError(
                    f"{si_acc} pays accounts {bad_out}: a region's savings-investment account "
                    "may only buy its own region's composites or transfer to other SAVINV "
                    "accounts (the bilateral capital account)."
                )
        # With SAVINV accounts, capital transfers must route between them — a cross-region
        # household transfer would put Sf back into consumption income (the pre-5d.2 routing).
        hoh_names = [f"HOH_{r}" for r in regions]
        bad_hoh = [
            (a, b)
            for a in hoh_names
            for b in hoh_names
            if a != b and abs(float(m.loc[a, b])) > 1e-9
        ]
        if bad_hoh:
            raise ValueError(
                f"with SAVINV accounts, cross-region household transfers {bad_hoh} are rejected: "
                "bilateral capital must route between the SAVINV accounts (foreign savings "
                "finance investment, not consumption)."
            )
    # Composite use of commodity s in region r = Σ_i INT0[r, s, i] + FD (s used by each activity i)
    # + government demand (Phase 5d.1) + investment demand (Phase 5d.2). INT0[r,j,i] is composite j
    # used by activity i, so use of commodity s = INT0[r,s,:].sum().
    Q0 = (
        INT0.sum(axis=2)
        + FD0
        + (GD0 if GD0 is not None else 0.0)
        + (INV0 if INV0 is not None else 0.0)
    )

    # Normalise by global GDP.
    scale = float(F0.sum())
    if scale <= 0:
        raise ValueError("multi-region SAM has non-positive total value added; cannot calibrate")
    D0, EX0, M0, Z0, INT0, F0, FD0, Q0 = (
        arr / scale for arr in (D0, EX0, M0, Z0, INT0, F0, FD0, Q0)
    )
    if GD0 is not None:
        GD0, gov_tax0 = GD0 / scale, gov_tax0 / scale
    if INV0 is not None:
        INV0, sav0 = INV0 / scale, sav0 / scale

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

    # Government parameters (Phase 5d.1): per-region CD demand shares (falling back to that
    # region's household gamma for a zero column) and the benchmark tax as a RATE on the region's
    # own factor income (replication + homogeneity both survive; same as closed/open).
    gov_gamma = None
    gov_income0 = None
    gov_tax_rate0 = None
    if GD0 is not None:
        gov_income0 = GD0.sum(axis=1)  # [r]
        gov_gamma = np.where(
            (gov_income0 > 0)[:, None],
            GD0 / np.where(gov_income0 > 0, gov_income0, 1.0)[:, None],
            gamma,
        )
        gov_tax_rate0 = gov_tax0 / endowment.sum(axis=0)  # [r] tax / regional factor income

    # Savings-investment parameters (Phase 5d.2): per-region composition and savings rate on
    # regional DISPOSABLE income, plus the per-region balance check investment = savings + Sf_r —
    # which implicitly validates that the inter-SAVINV capital transfers settle each region's
    # current account (the direct transfer cells are free to settle it any zero-sum way).
    inv_gamma = None
    sav_rate0 = None
    if INV0 is not None:
        inv_total = INV0.sum(axis=1)  # [r]
        for ri, r in enumerate(regions):
            expected = sav0[ri] + foreign_savings[ri]
            if abs(inv_total[ri] - expected) > 1e-6 * max(1.0, abs(expected)):
                raise ValueError(
                    f"SAVINV_{r} is unbalanced: investment {inv_total[ri]:.6g} ≠ household "
                    f"savings {sav0[ri]:.6g} + foreign savings {foreign_savings[ri]:.6g}; the "
                    "inter-SAVINV capital transfers do not settle the region's current account."
                )
        inv_gamma = np.where(
            (inv_total > 0)[:, None],
            INV0 / np.where(inv_total > 0, inv_total, 1.0)[:, None],
            gamma,
        )
        disposable0 = endowment.sum(axis=0) - (gov_tax0 if gov_tax0 is not None else 0.0)
        sav_rate0 = sav0 / disposable0  # [r]

    # Energy nest (Phase 5d.5): ONE nest per region, calibrated over that region's own composite
    # intermediate flows (INT0[r] is [j,i], VA0[r] is [i], Z0[r] is [i]). Opt-in; with no
    # energy_sectors declared production stays flat Leontief (bit-identical to pre-5d.5).
    energy_nests = None
    if energy_sectors:
        unknown = [e for e in energy_sectors if e not in sectors]
        if unknown:
            raise ValueError(f"energy_sectors {unknown} are not in the sector list {sectors}")
        e_idx = np.array([sectors.index(e) for e in energy_sectors], dtype=int)
        el = energy_elasticities or {}
        kle_m = el.get("kle_m", DEFAULT_KLE_M_ELAST)
        kl_e = el.get("kl_e", DEFAULT_KL_E_ELAST)
        en = el.get("energy", DEFAULT_ENERGY_ELAST)
        energy_nests = [
            calibrate_energy_nest(
                INT0[ri],
                VA0[ri],
                Z0[ri],
                e_idx,
                kle_m_elast=kle_m,
                kl_e_elast=kl_e,
                energy_elast=en,
            )
            for ri in range(nr)
        ]

    model = MultiCalibratedModel(
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
        gov_gamma=gov_gamma,
        gov_income0=gov_income0,
        GD0=GD0,
        gov_tax_rate0=gov_tax_rate0,
        inv_gamma=inv_gamma,
        INV0=INV0,
        sav_rate0=sav_rate0,
        energy_nests=energy_nests,
    )

    # A single global numéraire + a single globally-dropped factor-market equation is only a valid
    # closure when the region graph is CONNECTED (review P1, round 10). Two or more regions with no
    # active trade route linking them (directly or via intermediaries) are, mathematically, distinct
    # economies sharing one residual system with one numéraire and one dropped equation too few
    # between them — each additional component's overall price level is genuinely underdetermined,
    # not merely numerically delicate. Reject rather than silently solve a system with an
    # unidentified direction; per-component numéraire/Walras closures are a documented refinement
    # (roadmap), not yet implemented.
    components = model.connected_components
    if len(components) > 1:
        named = [sorted(model.regions[i] for i in comp) for comp in components]
        raise ValueError(
            f"multi-region SAM's region-trade graph is disconnected into {len(components)} "
            f"components ({named}); a single global numéraire and one globally-dropped factor "
            "market only identify a connected trade network. Each disconnected group needs its "
            "own numéraire/Walras closure (not yet implemented) — supply a SAM where every region "
            "reaches every other region via active trade routes, or split the disconnected groups "
            "into separate single-economy runs."
        )
    return model


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
