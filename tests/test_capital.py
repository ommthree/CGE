"""Tests for the capital-accumulation identity (Phase 5d.3).

The identity is standalone and stateless — these pin its analytic correctness, boundary
validation, and the premature-retirement option, following the plan's DoD. Phase 7.1's
recursive-dynamic wrapper will call this function; here we test it in isolation.
"""

import numpy as np
import pytest

from cge.engines.cge_static.capital import (
    DEFAULT_DEPRECIATION_RATE,
    capital_next,
)


def test_identity_analytic_known_answer():
    """K_{t+1} = (1−δ)K_t + INV, hand-computed: (1−0.05)·100 + 10 = 105.0."""
    assert capital_next(100.0, 10.0, depreciation=0.05) == pytest.approx(105.0)


def test_default_depreciation_rate_is_five_percent():
    assert DEFAULT_DEPRECIATION_RATE == 0.05
    # With the default and zero investment, the stock decays by exactly 5%.
    assert capital_next(200.0, 0.0) == pytest.approx(190.0)


def test_zero_depreciation_zero_retirement_is_pure_accumulation():
    assert capital_next(100.0, 25.0, depreciation=0.0, retirement=0.0) == pytest.approx(125.0)


def test_steady_state_investment_holds_stock_constant():
    """Replacement investment INV = δ·K leaves the stock unchanged (the stationary point)."""
    k = 100.0
    delta = 0.05
    assert capital_next(k, delta * k, depreciation=delta) == pytest.approx(k)


def test_retirement_lowers_next_stock_all_else_equal():
    """The plan's DoD check: a nonzero retirement fraction produces a strictly lower K_{t+1}."""
    base = capital_next(100.0, 10.0, depreciation=0.05, retirement=0.0)
    stranded = capital_next(100.0, 10.0, depreciation=0.05, retirement=0.2)
    assert stranded < base
    # Hand value: (1−0.05)(1−0.2)·100 + 10 = 0.95·0.8·100 + 10 = 86.0.
    assert stranded == pytest.approx(86.0)


def test_retirement_and_depreciation_compose_multiplicatively():
    """Both apply to the opening stock: surviving fraction is (1−δ)(1−r), not (1−δ−r)."""
    k, delta, r = 100.0, 0.1, 0.1
    assert capital_next(k, 0.0, depreciation=delta, retirement=r) == pytest.approx(81.0)
    # (1−δ−r) would give 80.0 — confirm we are NOT doing the additive (wrong) version.
    assert capital_next(k, 0.0, depreciation=delta, retirement=r) != pytest.approx(80.0)


def test_elementwise_per_region_vector():
    """The identity broadcasts elementwise, so it works per-region (or per-region-sector)."""
    k = np.array([100.0, 50.0])
    inv = np.array([10.0, 20.0])
    out = capital_next(k, inv, depreciation=0.05)
    assert np.allclose(out, [105.0, 67.5])


def test_per_element_depreciation_and_retirement():
    """δ and r can themselves be arrays (per-region rates / a sector-targeted stranding shock)."""
    k = np.array([100.0, 100.0])
    inv = np.array([0.0, 0.0])
    out = capital_next(k, inv, depreciation=np.array([0.05, 0.10]), retirement=np.array([0.0, 0.5]))
    assert np.allclose(out, [95.0, 45.0])  # (0.95)100 ; (0.90)(0.50)100


def test_result_is_non_negative_by_construction():
    """With valid inputs the stock cannot go negative even under full retirement + full
    depreciation and zero investment — it floors at the new investment vintage."""
    assert capital_next(100.0, 3.0, depreciation=1.0, retirement=1.0) == pytest.approx(3.0)
    assert capital_next(100.0, 0.0, depreciation=1.0) == pytest.approx(0.0)


# -- boundary validation (reject, don't clamp) --------------------------------
def test_negative_capital_rejected():
    with pytest.raises(ValueError, match="capital stock K_t must be non-negative"):
        capital_next(-1.0, 10.0)


def test_negative_investment_rejected():
    with pytest.raises(ValueError, match="investment must be non-negative"):
        capital_next(100.0, -5.0)


@pytest.mark.parametrize("bad", [-0.1, 1.5, np.nan, np.inf])
def test_out_of_range_depreciation_rejected(bad):
    with pytest.raises(ValueError, match="depreciation rate"):
        capital_next(100.0, 10.0, depreciation=bad)


@pytest.mark.parametrize("bad", [-0.1, 1.5, np.nan])
def test_out_of_range_retirement_rejected(bad):
    with pytest.raises(ValueError, match="retirement fraction"):
        capital_next(100.0, 10.0, retirement=bad)


def test_non_finite_stock_rejected():
    with pytest.raises(ValueError, match="must be finite"):
        capital_next(np.inf, 10.0)


def test_phase_7_1_usage_shape_round_trips():
    """Sanity: a per-region stock steps forward and could feed straight back in (what Phase 7.1's
    loop does), staying a finite non-negative float array of the same shape."""
    k = np.array([1.0, 2.0, 0.5])  # region-level capital, GDP-normalised scale
    inv = np.array([0.06, 0.11, 0.02])
    for _ in range(5):  # a few steps, as the recursive wrapper would
        k = capital_next(k, inv)
        assert k.shape == (3,)
        assert np.all(np.isfinite(k)) and np.all(k >= 0)


# -- benchmark_capital adapter (the Phase 7.1 entry point) --------------------
def test_benchmark_capital_closed():
    from cge.data.sam import toy_sam
    from cge.engines.cge_static.calibrate import calibrate
    from cge.engines.cge_static.capital import benchmark_capital

    cal = calibrate(toy_sam(), sectors=["BRD", "MIL"], factors=["CAP", "LAB"])
    k0 = benchmark_capital(cal)
    assert k0.shape == (1,)  # single-region → length-1 vector
    # CAP endowment at the GDP-normalised benchmark = capital's share of factor income.
    assert k0[0] == pytest.approx(cal.endowment[cal.factors.index("CAP")])


def test_benchmark_capital_multi_is_per_region():
    from cge.data.sam.toy_multi import REGIONS, SECTORS, toy_multi_sam
    from cge.engines.cge_static.calibrate_multi import calibrate_multi
    from cge.engines.cge_static.capital import benchmark_capital

    cal = calibrate_multi(toy_multi_sam(), regions=REGIONS, sectors=SECTORS, factors=["CAP", "LAB"])
    k0 = benchmark_capital(cal)
    assert k0.shape == (cal.nr,)  # one entry per region
    fi = cal.factors.index("CAP")
    assert np.allclose(k0, cal.endowment[fi, :])


def test_benchmark_capital_steps_forward():
    """The two pieces compose: benchmark_capital → capital_next is exactly what Phase 7.1 does."""
    from cge.data.sam import toy_sam
    from cge.engines.cge_static.calibrate import calibrate
    from cge.engines.cge_static.capital import benchmark_capital, capital_next

    cal = calibrate(toy_sam(), sectors=["BRD", "MIL"], factors=["CAP", "LAB"])
    k0 = benchmark_capital(cal)
    k1 = capital_next(k0, np.array([0.05]))  # a small investment flow
    assert k1.shape == k0.shape
    assert np.all(np.isfinite(k1)) and np.all(k1 >= 0)


def test_benchmark_capital_requires_capital_factor():
    """A model with no CAP factor has no capital stock to track — reject, don't guess."""
    from dataclasses import replace

    from cge.data.sam import toy_sam
    from cge.engines.cge_static.calibrate import calibrate
    from cge.engines.cge_static.capital import benchmark_capital

    cal = calibrate(toy_sam(), sectors=["BRD", "MIL"], factors=["CAP", "LAB"])
    # Relabel the factor list so no factor is named CAP (the identity has no capital otherwise).
    no_cap = replace(cal, factors=["LND", "LAB"])
    with pytest.raises(ValueError, match="no 'CAP' factor"):
        benchmark_capital(no_cap)
