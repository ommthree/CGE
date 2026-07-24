"""Nested KL-E-M production (Phase 5d.5 — the energy nest).

Replaces flat Leontief-in-everything production with a **nested CES** structure, so a carbon price
can shift substitution *within* the energy bundle (toward electricity, away from fossil fuels), not
only across sectors. The single highest-value item in Phase 5d [roadmap], and what Phase 7b's
NGFS energy-transition harmonisation needs.

**Opt-in and backward-compatible.** The nest activates only when a build declares which commodities
are energy (``energy_sectors``, explicit — not string-matched, the same discipline as the
``institutions`` role-mapping). With no energy sectors declared the model is exactly the flat
Leontief pilot (bit-identical), so every pre-5d.5 run is unchanged.

**The nest** [Hosoe2010, standard KL-E-M], per sector ``i``:

    outer:   X_i   = A_i [ δ_i KLE_i^ρ_i + (1-δ_i) M_i^ρ_i ]^{1/ρ_i}       (KLE vs. materials)
    middle:  KLE_i = A^KLE_i [ δ^kl_i KL_i^ρ^kle_i + (1-δ^kl_i) E_i^ρ^kle_i ]^{1/ρ^kle_i}
    inner:   E_i   = A^E_i [ Σ_e δ^e_{ei} E_{ei}^ρ^e_i ]^{1/ρ^e_i}          (CES over energy goods)
             KL_i  = the existing value-added composite (CD/CES over factors) — unchanged.
    M_i is a **Leontief** aggregate of the non-energy intermediates (fixed proportions, as today).

Substitution elasticities σ relate to the CES exponents by ρ = (σ-1)/σ. σ = 1 is the Cobb-Douglas
special case; the module handles it explicitly (the ρ formulae are singular there).

**What this module provides** (the shared engine, called from all three ``calibrate_*``/``model_*``
variants so the CES algebra lives in ONE place):
- ``calibrate_energy_nest``: back out the scale/share parameters from benchmark flows at unit
  prices, so the benchmark reproduces exactly.
- ``nest_unit_cost``: the dual — output unit cost ``px`` given factor/commodity prices and the
  per-energy-commodity carbon cost. This replaces the flat ``Σ ax·p + va_share·pv + cc`` in the
  zero-profit condition.
- ``nest_demands``: Shephard/quantity demands — energy-commodity intermediate use, materials use,
  and the value-added quantity — per unit of output, so goods-market clearing and factor demand
  stay consistent.

All arrays are per-sector (closed/open); the multi-region variant calls this per region-sector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Documented default substitution elasticities [Hosoe2010, ch. 6 / standard CGE practice], each
# overridable per build. Central values in the usual range for aggregate CGE energy nests.
DEFAULT_KLE_M_ELAST = 0.5  # outer: KLE composite vs. materials (low — materials hard to substitute)
DEFAULT_KL_E_ELAST = 0.5  # middle: capital-labour vs. energy (energy-capital substitution)
DEFAULT_ENERGY_ELAST = 1.5  # inner: across energy commodities (electricity vs. fossil — easier)


@dataclass(frozen=True)
class EnergyNest:
    """Calibrated KL-E-M nest parameters for one build. Indexed by sector ``i``; the energy
    sub-nest is indexed by ``(energy-commodity, sector)``. All benchmark prices = 1.

    ``energy_idx`` are the indices (into the sector list) of the energy commodities; ``mat_idx``
    the non-energy (materials) commodities. A sector with **no** energy intermediate use still has
    a well-defined nest (its energy share is 0 and the KLE composite is just KL)."""

    energy_idx: np.ndarray  # [n_e] indices of energy commodities among the sectors
    mat_idx: np.ndarray  # [n_m] indices of non-energy (materials) commodities
    # Outer nest (KLE vs. materials), per sector.
    kle_m_elast: np.ndarray  # [i] σ outer
    outer_scale: np.ndarray  # [i] A
    outer_share_kle: np.ndarray  # [i] δ (KLE share); materials share is 1-δ
    # Middle nest (KL vs. energy), per sector.
    kl_e_elast: np.ndarray  # [i] σ middle
    mid_scale: np.ndarray  # [i] A^KLE
    mid_share_kl: np.ndarray  # [i] δ^kl (KL share); energy share is 1-δ^kl
    # Inner energy nest (CES over energy commodities), per sector.
    energy_elast: np.ndarray  # [i] σ inner
    energy_scale: np.ndarray  # [i] A^E
    energy_share: (
        np.ndarray
    )  # [e, i] δ^e (over energy commodities, sums to 1 per sector with E use)
    # Materials Leontief coefficients (fixed proportions within the materials aggregate), per
    # sector: materials-commodity m used per unit of the materials composite M_i.
    mat_coeff: np.ndarray  # [m, i] non-energy intermediate m per unit M_i
    # Benchmark composite quantities per unit output (for calibration round-tripping / tests).
    kle0: np.ndarray  # [i] KLE per unit X at benchmark
    mat0: np.ndarray  # [i] M per unit X at benchmark
    kl0: np.ndarray  # [i] KL per unit KLE at benchmark
    e0: np.ndarray  # [i] E per unit KLE at benchmark


def _ces_unit_cost(shares: np.ndarray, prices: np.ndarray, sigma: float, scale: float) -> float:
    """Dual unit cost of the CES aggregator Y = scale·[Σ_k δ_k y_k^ρ]^{1/ρ}, ρ=(σ-1)/σ, whose
    cost-minimising demand shares are calibrated as δ_k ∝ v_k^{1/σ} (the same convention as the
    Armington/CET nests, so quantities round-trip — see ``_calibrate_ces_shares``). Its dual:
        c(p) = (1/scale)·[Σ_k δ_k^σ p_k^{1-σ}]^{1/(1-σ)}   (σ≠1)
        c(p) = (1/scale)·Π_k (p_k/δ_k)^{δ_k}               (σ=1, Cobb-Douglas)
    ``shares`` and ``prices`` are aligned 1-D arrays over the active inputs (δ summing to 1)."""
    if abs(sigma - 1.0) < 1e-12:
        pos = shares > 0
        return (1.0 / scale) * float(np.prod(np.power(prices[pos] / shares[pos], shares[pos])))
    inner = float(np.sum(shares**sigma * prices ** (1.0 - sigma)))
    return (1.0 / scale) * inner ** (1.0 / (1.0 - sigma))


def _ces_input_per_output(
    shares: np.ndarray, prices: np.ndarray, sigma: float, scale: float, unit_cost: float
) -> np.ndarray:
    """Shephard demand: input quantity per unit composite output, y_k/Y = ∂c/∂p_k, for the CES
    aggregator above. Returns an array aligned with ``shares``/``prices``.
        σ≠1: y_k/Y = (1/scale)·(scale·c)^σ·δ_k^σ·p_k^{-σ}
        σ=1: y_k/Y = δ_k·c/p_k                                (Cobb-Douglas)"""
    if abs(sigma - 1.0) < 1e-12:
        return shares * unit_cost / prices
    return (1.0 / scale) * (scale * unit_cost) ** sigma * shares**sigma * prices ** (-sigma)


def _calibrate_ces_shares(values: np.ndarray, sigma: float) -> tuple[np.ndarray, float]:
    """Calibrate CES shares δ and scale A from benchmark input VALUES at unit prices, so the
    aggregator reproduces the benchmark quantities exactly (the Armington/CET convention: at unit
    prices the cost-min demand y_k/Y = δ_k^σ·A^{σ-1}, which equals the value share v_k iff
    δ_k ∝ v_k^{1/σ}). Returns (δ over the inputs summing to 1, scale A making unit cost 1).

    ``values`` are the benchmark input values (0 for an absent input — its δ is 0)."""
    total = float(values.sum())
    if total <= 0:
        # No inputs at all — a degenerate composite; caller must guard (share vector all zero).
        return np.zeros_like(values), 1.0
    v = values / total  # value shares
    pos = v > 0
    if abs(sigma - 1.0) < 1e-12:
        delta = v  # Cobb-Douglas: shares are the value shares
        scale = float(np.prod(np.power(1.0 / delta[pos], delta[pos])))
        return delta, scale
    w = np.zeros_like(v)
    w[pos] = np.power(v[pos], 1.0 / sigma)
    delta = w / w.sum()
    scale = float(np.sum(delta[pos] ** sigma)) ** (1.0 / (1.0 - sigma))
    return delta, scale


def calibrate_energy_nest(
    Z0: np.ndarray,
    VA0: np.ndarray,
    X0: np.ndarray,
    energy_idx: np.ndarray,
    *,
    kle_m_elast: float = DEFAULT_KLE_M_ELAST,
    kl_e_elast: float = DEFAULT_KL_E_ELAST,
    energy_elast: float = DEFAULT_ENERGY_ELAST,
) -> EnergyNest:
    """Calibrate the KL-E-M nest from benchmark flows at unit prices (money = quantity).

    - ``Z0`` [j,i]: benchmark intermediate use of commodity j by sector i.
    - ``VA0`` [i]: benchmark value added (KL) of sector i.
    - ``X0`` [i]: benchmark gross output of sector i.
    - ``energy_idx``: indices (into the commodity axis) of the energy commodities.

    All flows must already be GDP-normalised (as the calibrators do). At benchmark every price and
    composite unit price is 1, so shares are value shares and scales make each composite price 1.
    The calibration reproduces the benchmark exactly (Tier-1 replication)."""
    ns = X0.shape[0]
    energy_idx = np.asarray(energy_idx, dtype=int)
    mat_idx = np.array([j for j in range(ns) if j not in set(energy_idx.tolist())], dtype=int)

    # Benchmark value flows (= quantities at unit prices).
    E_by_sector = Z0[energy_idx, :]  # [n_e, i] energy commodity e used by sector i
    M_by_sector = Z0[mat_idx, :]  # [n_m, i] materials commodity m used by sector i
    E_tot = E_by_sector.sum(axis=0)  # [i] total energy value used by sector i
    M_tot = M_by_sector.sum(axis=0)  # [i] total materials value
    KL_tot = VA0  # [i] value added is the KL composite value

    kle_m = np.full(ns, float(kle_m_elast))
    kl_e = np.full(ns, float(kl_e_elast))
    en = np.full(ns, float(energy_elast))
    for name, s in (("kle_m", kle_m), ("kl_e", kl_e), ("energy", en)):
        if np.any(s <= 0) or not np.all(np.isfinite(s)):
            raise ValueError(f"{name} elasticity must be finite and strictly positive")

    # Composite value benchmarks (value = quantity at unit prices).
    KLE_tot = KL_tot + E_tot  # [i] KLE composite value = KL + energy
    n_e = len(energy_idx)

    # Per-sector CES calibration (δ ∝ v^{1/σ}, scale makes unit cost 1) — the same convention as
    # the Armington/CET nests, so benchmark quantities round-trip exactly.
    outer_share_kle = np.empty(ns)  # δ on KLE (materials share = 1-δ)
    outer_scale = np.empty(ns)
    mid_share_kl = np.empty(ns)  # δ^kl on KL (energy share = 1-δ^kl)
    mid_scale = np.empty(ns)
    energy_share = np.zeros((n_e, ns))  # δ^e over energy commodities
    energy_scale = np.ones(ns)
    for i in range(ns):
        # Outer: KLE vs materials.
        d_out, s_out = _calibrate_ces_shares(np.array([KLE_tot[i], M_tot[i]]), float(kle_m[i]))
        outer_share_kle[i], outer_scale[i] = d_out[0], s_out
        # Middle: KL vs energy composite.
        d_mid, s_mid = _calibrate_ces_shares(np.array([KL_tot[i], E_tot[i]]), float(kl_e[i]))
        mid_share_kl[i], mid_scale[i] = d_mid[0], s_mid
        # Inner: across energy commodities (only if the sector uses energy).
        if E_tot[i] > 0:
            d_e, s_e = _calibrate_ces_shares(E_by_sector[:, i], float(en[i]))
            energy_share[:, i], energy_scale[i] = d_e, s_e

    # Materials Leontief coefficients: materials-commodity m per unit of the materials composite
    # M_i (fixed proportions). With M_i's benchmark quantity = M_tot, coeff = flow / M_tot.
    with np.errstate(divide="ignore", invalid="ignore"):
        mat_coeff = np.where(M_tot[None, :] > 0, M_by_sector / M_tot[None, :], 0.0)  # [m,i]

    # Benchmark composite quantities per unit of their parent (stored for round-tripping / tests).
    with np.errstate(divide="ignore", invalid="ignore"):
        kle0 = np.where(X0 > 0, KLE_tot / X0, 0.0)
        mat0 = np.where(X0 > 0, M_tot / X0, 0.0)
        kl0 = np.where(KLE_tot > 0, KL_tot / KLE_tot, 0.0)
        e0 = np.where(KLE_tot > 0, E_tot / KLE_tot, 0.0)

    return EnergyNest(
        energy_idx=energy_idx,
        mat_idx=mat_idx,
        kle_m_elast=kle_m,
        outer_scale=outer_scale,
        outer_share_kle=outer_share_kle,
        kl_e_elast=kl_e,
        mid_scale=mid_scale,
        mid_share_kl=mid_share_kl,
        energy_elast=en,
        energy_scale=energy_scale,
        energy_share=energy_share,
        mat_coeff=mat_coeff,
        kle0=kle0,
        mat0=mat0,
        kl0=kl0,
        e0=e0,
    )


def nest_unit_cost(
    nest: EnergyNest,
    pq: np.ndarray,
    pv: np.ndarray,
    carbon_cost: np.ndarray,
) -> np.ndarray:
    """Output unit cost px[i] (the dual of the whole KL-E-M nest) — replaces the flat
    ``Σ_j ax[j,i]·p[j] + va_share[i]·pv[i] + cc[i]`` in the zero-profit condition.

    - ``pq`` [j]: composite commodity prices (what intermediates pay).
    - ``pv`` [i]: value-added (KL) unit cost per sector — as computed today.
    - ``carbon_cost`` [j]: per-unit carbon cost on commodity j; **attaches to the energy
      commodities specifically** (the emissions tax on fossil energy use), so the effective energy
      price is pq[j] + carbon_cost[j] for an energy commodity j. This is what makes the carbon
      price bite the energy nest — the whole point of 5d.5.

    Bottom-up: energy composite cost pE from energy-commodity prices; KLE cost from (pv, pE);
    output cost from (pKLE, pM) with pM the Leontief materials cost."""
    ns = len(pv)
    px = np.empty(ns)
    e_idx, m_idx = nest.energy_idx, nest.mat_idx
    p_energy_eff = pq[e_idx] + carbon_cost[e_idx]  # carbon attaches to energy commodities
    p_mat = pq[m_idx]
    for i in range(ns):
        # Inner: energy composite unit cost (only if the sector uses energy).
        if nest.energy_share[:, i].sum() > 0:
            pE = _ces_unit_cost(
                nest.energy_share[:, i],
                p_energy_eff,
                float(nest.energy_elast[i]),
                float(nest.energy_scale[i]),
            )
        else:
            pE = 1.0  # unused (KL share of KLE is 1)
        # Middle: KLE unit cost over (KL=pv, energy=pE).
        kl_share = float(nest.mid_share_kl[i])
        pKLE = _ces_unit_cost(
            np.array([kl_share, 1.0 - kl_share]),
            np.array([pv[i], pE]),
            float(nest.kl_e_elast[i]),
            float(nest.mid_scale[i]),
        )
        # Materials: Leontief aggregate cost (fixed proportions).
        pM = float(np.dot(nest.mat_coeff[:, i], p_mat))
        # Outer: output cost over (KLE=pKLE, materials=pM). A sector with no materials has δ=1.
        kle_share = float(nest.outer_share_kle[i])
        if kle_share >= 1.0 - 1e-15:
            px[i] = pKLE / float(nest.outer_scale[i])
        else:
            px[i] = _ces_unit_cost(
                np.array([kle_share, 1.0 - kle_share]),
                np.array([pKLE, pM]),
                float(nest.kle_m_elast[i]),
                float(nest.outer_scale[i]),
            )
    return px


def nest_demands(
    nest: EnergyNest,
    pq: np.ndarray,
    pv: np.ndarray,
    carbon_cost: np.ndarray,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given prices and gross output ``X`` [i], return
    (``energy_use`` [e,i], ``materials_use`` [j,i], ``kl_qty`` [i]) — the intermediate demand for
    each energy commodity, each materials commodity, and the value-added (KL) quantity, all as
    TOTAL quantities (not per-unit). Consistent with ``nest_unit_cost`` by Shephard's lemma, so the
    goods market clears and factor demand keys off ``kl_qty``.

    ``carbon_cost`` attaches to energy prices exactly as in ``nest_unit_cost``."""
    ns = len(pv)
    n_e, n_m = len(nest.energy_idx), len(nest.mat_idx)
    e_idx, m_idx = nest.energy_idx, nest.mat_idx
    p_energy_eff = pq[e_idx] + carbon_cost[e_idx]
    p_mat = pq[m_idx]
    energy_use = np.zeros((n_e, ns))
    materials_use = np.zeros((n_m, ns))
    kl_qty = np.zeros(ns)
    for i in range(ns):
        kle_share = float(nest.outer_share_kle[i])
        # Recompute the composite unit costs (bottom-up) so the demands are internally consistent.
        if nest.energy_share[:, i].sum() > 0:
            pE = _ces_unit_cost(
                nest.energy_share[:, i],
                p_energy_eff,
                float(nest.energy_elast[i]),
                float(nest.energy_scale[i]),
            )
        else:
            pE = 1.0
        kl_share = float(nest.mid_share_kl[i])
        pKLE = _ces_unit_cost(
            np.array([kl_share, 1.0 - kl_share]),
            np.array([pv[i], pE]),
            float(nest.kl_e_elast[i]),
            float(nest.mid_scale[i]),
        )
        pM = float(np.dot(nest.mat_coeff[:, i], p_mat))
        px_i = (
            pKLE / float(nest.outer_scale[i])
            if kle_share >= 1.0 - 1e-15
            else _ces_unit_cost(
                np.array([kle_share, 1.0 - kle_share]),
                np.array([pKLE, pM]),
                float(nest.kle_m_elast[i]),
                float(nest.outer_scale[i]),
            )
        )
        # Outer split: KLE and materials composite per unit X.
        if kle_share >= 1.0 - 1e-15:
            kle_per_x, m_per_x = 1.0 / float(nest.outer_scale[i]), 0.0
        else:
            outer = _ces_input_per_output(
                np.array([kle_share, 1.0 - kle_share]),
                np.array([pKLE, pM]),
                float(nest.kle_m_elast[i]),
                float(nest.outer_scale[i]),
                px_i,
            )
            kle_per_x, m_per_x = float(outer[0]), float(outer[1])
        KLE = kle_per_x * X[i]
        Mcomp = m_per_x * X[i]
        # Materials Leontief: each materials commodity = coeff · M composite.
        materials_use[:, i] = nest.mat_coeff[:, i] * Mcomp
        # Middle split: KL and energy composite per unit KLE.
        mid = _ces_input_per_output(
            np.array([kl_share, 1.0 - kl_share]),
            np.array([pv[i], pE]),
            float(nest.kl_e_elast[i]),
            float(nest.mid_scale[i]),
            pKLE,
        )
        kl_qty[i] = float(mid[0]) * KLE
        E = float(mid[1]) * KLE
        # Inner split: each energy commodity per unit E.
        if nest.energy_share[:, i].sum() > 0:
            inner = _ces_input_per_output(
                nest.energy_share[:, i],
                p_energy_eff,
                float(nest.energy_elast[i]),
                float(nest.energy_scale[i]),
                pE,
            )
            energy_use[:, i] = inner * E
    return energy_use, materials_use, kl_qty
