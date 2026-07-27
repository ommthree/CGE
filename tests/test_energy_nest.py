"""Tests for the standalone KL-E-M energy nest (Phase 5d.5).

The nest's CES algebra is the hardest part of 5d.5, so it is unit-tested in isolation here
(against hand-computable benchmarks) before being wired into the three model variants — the same
de-risking discipline as capital.py. The two non-negotiables: at benchmark (unit prices, no carbon
cost) every composite unit price is 1 and demands reproduce the benchmark flows exactly; and the
carbon-price signs are right (energy price up ⇒ substitute away from energy).
"""

import numpy as np
import pytest

from cge.engines.cge_static.energy_nest import (
    calibrate_energy_nest,
    nest_demands,
    nest_unit_cost,
)

# A hand-built 3-commodity benchmark: sector 0 uses energy (commodity 1) + materials (commodity 2)
# + value added; sector 1 similar with a different mix; commodity 2 is a pure materials sector.
# Z0[j, i] = commodity j used by sector i. Energy is commodity index 1.
_Z0 = np.array(
    [
        [0.0, 5.0, 3.0],  # commodity 0 (materials) used by sectors 0,1,2
        [8.0, 6.0, 0.0],  # commodity 1 (ENERGY) used by sectors 0,1
        [4.0, 0.0, 2.0],  # commodity 2 (materials) used by sectors 0,2
    ],
    dtype=float,
)
_VA0 = np.array([20.0, 15.0, 10.0])  # value added (KL) per sector
_ENERGY = np.array([1])  # commodity 1 is energy


def _X0():
    # Gross output = Σ intermediate cost + value added (unit prices).
    return _Z0.sum(axis=0) + _VA0


def _cal(**kw):
    return calibrate_energy_nest(_Z0, _VA0, _X0(), _ENERGY, **kw)


def test_energy_and_materials_partition():
    nest = _cal()
    assert nest.energy_idx.tolist() == [1]
    assert nest.mat_idx.tolist() == [0, 2]


def test_shares_sum_correctly():
    nest = _cal()
    # Energy shares over energy commodities sum to 1 for sectors that use energy (0, 1), 0 for 2.
    assert nest.energy_share[:, 0].sum() == pytest.approx(1.0)
    assert nest.energy_share[:, 1].sum() == pytest.approx(1.0)
    assert nest.energy_share[:, 2].sum() == pytest.approx(0.0)


def test_benchmark_unit_cost_is_one():
    """THE calibration replication check: at unit prices with no carbon cost, every sector's output
    unit cost is exactly 1 (so the zero-profit price p = px reproduces p = 1)."""
    nest = _cal()
    ns = len(_VA0)
    pq = np.ones(ns)
    pv = np.ones(ns)
    cc = np.zeros(ns)
    px = nest_unit_cost(nest, pq, pv, cc)
    assert np.allclose(px, 1.0, atol=1e-12)


def test_benchmark_demands_reproduce_flows():
    """At unit prices, nest_demands reproduces the exact benchmark energy/materials/VA flows."""
    nest = _cal()
    ns = len(_VA0)
    X = _X0()
    energy_use, materials_use, kl_qty = nest_demands(
        nest, np.ones(ns), np.ones(ns), np.zeros(ns), X
    )
    # Energy commodity (index 1) use per sector matches Z0[1].
    assert np.allclose(energy_use[0, :], _Z0[1, :], atol=1e-10)
    # Materials commodities (indices 0, 2) match Z0[0], Z0[2].
    assert np.allclose(materials_use[0, :], _Z0[0, :], atol=1e-10)
    assert np.allclose(materials_use[1, :], _Z0[2, :], atol=1e-10)
    # Value-added quantity matches VA0.
    assert np.allclose(kl_qty, _VA0, atol=1e-10)


def test_carbon_is_a_per_output_wedge():
    """Review remediation (2026-07-26): the carbon cost is a per-OUTPUT wedge, exactly as in the
    flat model — ``nest_unit_cost`` raises sector ``i``'s output cost by ``cc[i]`` and NOTHING
    else. So a cost on an ENERGY COMMODITY index raises only THAT sector's output cost (because it
    happens to be indexed there), not the cost of every energy-using sector — the old formulation
    (add-on to energy-input prices) reinterpreted an output-intensity vector as a fuel-use tax and
    dropped process/household emissions. Substitution is driven by pq, tested separately."""
    nest = _cal()
    ns = len(_VA0)
    cc = np.zeros(ns)
    cc[1] = 0.5  # per-output wedge on sector 1
    px = nest_unit_cost(nest, np.ones(ns), np.ones(ns), cc)
    assert px[1] == pytest.approx(1.5, abs=1e-12)  # sector 1's output cost += cc[1] exactly
    assert px[0] == pytest.approx(1.0, abs=1e-12)  # OTHER sectors unchanged (no fuel-use add-on)
    assert px[2] == pytest.approx(1.0, abs=1e-12)
    # A cost on a NON-energy sector (0) is now honoured — the whole point of the fix.
    cc0 = np.array([0.4, 0.0, 0.0])
    assert nest_unit_cost(nest, np.ones(ns), np.ones(ns), cc0)[0] == pytest.approx(1.4, abs=1e-12)


def test_carbon_substitutes_away_from_energy_via_price():
    """THE Tier-2 sign test (5d.5's deliverable), under the corrected contract: substitution is
    driven by the energy-COMMODITY PRICE (pq), which is what a taxed fossil sector's output tax
    raises via zero-profit — NOT by a separate carbon add-on. A higher energy-commodity price makes
    an energy-using sector substitute AWAY from energy (lower energy input per unit output)."""
    nest = _cal()
    ns = len(_VA0)
    X = _X0()
    e0, _m0, _kl0 = nest_demands(nest, np.ones(ns), np.ones(ns), np.zeros(ns), X)
    energy_intensity_base = e0[0, 0] / X[0]
    # Energy commodity (index 1) is more expensive — as it would be after its output is taxed.
    pq = np.ones(ns)
    pq[1] = 1.5
    e1, _m1, _kl1 = nest_demands(nest, pq, np.ones(ns), np.zeros(ns), X)
    energy_intensity_shocked = e1[0, 0] / X[0]
    assert energy_intensity_shocked < energy_intensity_base  # substituted away from energy


def test_within_energy_substitution_toward_cheaper_via_price():
    """Within the energy composite, a HIGHER PRICE on one energy commodity (its output tax passing
    through pq) shifts the mix toward the cheaper one (electricity vs. fossil) — the intra-energy
    substitution, now driven by relative prices, not a carbon add-on."""
    Z0 = np.array(
        [
            [2.0, 0.0, 0.0],  # materials commodity 0 used by sector 0
            [5.0, 0.0, 0.0],  # energy commodity 1 used by sector 0
            [5.0, 0.0, 0.0],  # energy commodity 2 used by sector 0
        ]
    )
    VA0 = np.array([10.0, 8.0, 8.0])
    X0 = Z0.sum(axis=0) + VA0
    nest = calibrate_energy_nest(Z0, VA0, X0, np.array([1, 2]), energy_elast=2.0)
    ns = 3
    pq = np.ones(ns)
    pq[1] = 1.6  # energy commodity 1 pricier (its output tax passed through)
    e_use, _m, _kl = nest_demands(nest, pq, np.ones(ns), np.zeros(ns), X0)
    e_base, _, _ = nest_demands(nest, np.ones(ns), np.ones(ns), np.zeros(ns), X0)
    assert e_use[0, 0] < e_base[0, 0]  # pricier energy commodity used less
    assert e_use[1, 0] > e_base[1, 0]  # cheaper energy commodity used more


def test_sector_with_no_energy_is_pure_leontief_materials():
    """A sector that uses no energy has a well-defined nest: its KLE composite is just KL, and its
    output cost is the Leontief materials + VA cost — no NaN from the empty energy sub-nest."""
    # Square 2×2: sector 0 uses only materials (commodity 0); commodity 1 is energy but unused.
    Z0 = np.array([[6.0, 0.0], [0.0, 0.0]])
    VA0 = np.array([14.0, 5.0])
    X0 = Z0.sum(axis=0) + VA0
    nest = calibrate_energy_nest(Z0, VA0, X0, np.array([1]))
    ns = 2
    px = nest_unit_cost(nest, np.ones(ns), np.ones(ns), np.zeros(ns))
    assert px[0] == pytest.approx(1.0, abs=1e-12)
    # A carbon cost on the (unused) energy commodity does nothing to sector 0.
    cc = np.array([0.0, 0.9])
    assert nest_unit_cost(nest, np.ones(ns), np.ones(ns), cc)[0] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_bad_elasticity_rejected(bad):
    with pytest.raises(ValueError, match="elasticity"):
        _cal(energy_elast=bad)


def test_homogeneity_of_unit_cost():
    """The dual is homogeneous of degree 1 in prices: scaling all prices by k scales px by k (a
    basic CES-cost property; a good guard against a mis-specified scale)."""
    nest = _cal()
    ns = len(_VA0)
    px1 = nest_unit_cost(nest, np.ones(ns), np.ones(ns), np.zeros(ns))
    k = 2.5
    px2 = nest_unit_cost(nest, k * np.ones(ns), k * np.ones(ns), np.zeros(ns))
    assert np.allclose(px2, k * px1, atol=1e-10)
