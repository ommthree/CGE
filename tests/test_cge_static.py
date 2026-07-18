"""Unit tests for Engine 3 (static CGE pilot): SAM, calibration, model, engine.

Pins the CGE correctness battery (replication, homogeneity, Walras) and the pilot's economic
behaviour, plus calibration exactness and the engine wrapper. All run on the hand-checkable
2-sector toy SAM via the scipy solver fallback (no IPOPT binary needed).
"""

from dataclasses import replace

import numpy as np
import pytest

from cge.contracts.engine import registry
from cge.contracts.shocks import CarbonPrice
from cge.data.sam import toy_sam
from cge.data.sam.balance import imbalance, is_balanced, ras_balance
from cge.engines.cge_static import model as M
from cge.engines.cge_static.calibrate import calibrate
from cge.engines.cge_static.solver import solve

_SECTORS = ["BRD", "MIL"]
_FACTORS = ["CAP", "LAB"]
_EMISSIONS = np.array([2.0, 0.5])  # BRD dirty, MIL clean


def _cal():
    return calibrate(toy_sam(), sectors=_SECTORS, factors=_FACTORS)


def _solve(cal, carbon_cost=None, drop_factor=0):
    cc = np.zeros(len(cal.sectors)) if carbon_cost is None else carbon_cost
    sol = solve(
        lambda z: M.residuals(cal, z, carbon_cost=cc, drop_factor=drop_factor),
        M.initial_guess(cal),
        prefer="scipy",
    )
    ns = len(cal.sectors)
    return sol, M.derive_state(cal, sol.x[:ns], sol.x[ns:])


# -- SAM ----------------------------------------------------------------------
def test_toy_sam_is_balanced():
    sam = toy_sam()
    assert is_balanced(sam.matrix)
    assert imbalance(sam.matrix).abs().max() < 1e-9
    # GDP identity: total value added = total final demand = 181
    va = sam.matrix.loc[_FACTORS, _SECTORS].to_numpy().sum()
    fd = sam.matrix.loc[_SECTORS, "HOH"].to_numpy().sum()
    assert np.isclose(va, 181.0) and np.isclose(fd, 181.0)


def test_ras_balances_an_unbalanced_matrix():
    import pandas as pd

    m = pd.DataFrame([[1.0, 2.0], [3.0, 1.0]], index=["a", "b"], columns=["a", "b"])
    targets = pd.Series({"a": 4.0, "b": 4.0})
    balanced = ras_balance(m, targets)
    assert np.allclose(balanced.sum(axis=1), balanced.sum(axis=0), atol=1e-8)
    assert np.allclose(balanced.sum(axis=1), [4.0, 4.0], atol=1e-8)


# -- calibration --------------------------------------------------------------
def test_calibration_reproduces_benchmark_parameters():
    cal = _cal()
    assert np.allclose(cal.X0, [100.0, 120.0])
    assert np.allclose(cal.ax, [[0.0, 0.2], [0.15, 0.0]])  # input j per unit output i
    assert np.allclose(cal.va_share, [0.85, 0.8])
    assert np.allclose(cal.beta, [[0.5, 0.5], [0.5, 0.5]])  # 50/50 CAP/LAB
    assert np.allclose(cal.gamma.sum(), 1.0)
    assert np.allclose(cal.endowment, [90.5, 90.5])
    assert np.isclose(cal.gdp0, 181.0)


def test_zero_profit_holds_at_benchmark():
    """Σ_j ax[j,i] + va_share[i] = 1 at benchmark prices (the calibration identity)."""
    cal = _cal()
    for i in range(len(cal.sectors)):
        assert np.isclose(cal.ax[:, i].sum() + cal.va_share[i], 1.0)


# -- CGE correctness battery --------------------------------------------------
def test_benchmark_replication():
    cal = _cal()
    _sol, st = _solve(cal)
    assert np.allclose(st.p, 1.0, atol=1e-8)
    assert np.allclose(st.X, cal.X0, atol=1e-6)
    assert np.allclose(st.FD, cal.FD0, atol=1e-6)
    assert np.allclose(st.F, cal.F0, atol=1e-6)


def test_homogeneity_degree_zero():
    cal = _cal()
    sol, st = _solve(cal)
    k = 2.5
    cal_k = replace(cal, endowment=cal.endowment * k, X0=cal.X0 * k, F0=cal.F0 * k, Z0=cal.Z0 * k)
    sol_k, st_k = _solve(cal_k)
    assert np.allclose(sol.x, sol_k.x, atol=1e-8)  # prices unchanged
    assert np.allclose(st_k.X, k * st.X, atol=1e-6)  # reals scale


def test_walras_law():
    cal = _cal()
    _sol, st = _solve(cal, drop_factor=0)
    excess = float(st.F[0, :].sum()) - cal.endowment[0]
    assert abs(excess) < 1e-8


# -- economic behaviour -------------------------------------------------------
def test_carbon_price_contracts_dirty_sector():
    cal = _cal()
    _b, base = _solve(cal)
    _s, st = _solve(cal, carbon_cost=0.2 * _EMISSIONS)
    assert st.X[0] < base.X[0]  # dirty BRD output falls
    assert st.p[0] / st.p[1] > base.p[0] / base.p[1]  # dirty price rises relative to clean
    assert st.FD.sum() < base.FD.sum()  # real GDP falls


def test_larger_carbon_price_larger_response():
    """Monotonicity: a bigger carbon price contracts the dirty sector more."""
    cal = _cal()
    _, s1 = _solve(cal, carbon_cost=0.1 * _EMISSIONS)
    _, s2 = _solve(cal, carbon_cost=0.2 * _EMISSIONS)
    _b, base = _solve(cal)
    assert (base.X[0] - s2.X[0]) > (base.X[0] - s1.X[0]) > 0


# -- engine wrapper -----------------------------------------------------------
def test_engine_registered_with_ge_capability():
    import cge.engines  # noqa: F401

    meta = registry.get("cge_static").meta
    caps = [c.value for c in meta.capabilities]
    assert "general_equilibrium" in caps and "prices" in caps and "volumes" in caps


def test_engine_zero_shock_replicates():
    eng = registry.get("cge_static")
    res = eng.run(data={"SAM": toy_sam()}, shocks=[CarbonPrice(price=0.0)], years=[2020])
    assert res.data["value"].abs().max() < 1e-8  # every change is zero


def test_engine_carbon_price_emits_ge_outputs():
    eng = registry.get("cge_static")
    data = {"SAM": toy_sam(), "emission_intensity": {"BRD": 2.0, "MIL": 0.5}}
    res = eng.run(data=data, shocks=[CarbonPrice(price=0.1)], years=[2020])
    d = res.data
    for v in ("price_change", "volume_change", "factor_price_change", "gdp_change_real"):
        assert (d["variable"] == v).any(), v
    # dirty sector volume falls
    brd = d[(d["variable"] == "volume_change") & (d["sector"] == "BRD")]["value"].iloc[0]
    assert brd < 0
    assert res.manifest.assumptions["emissions_priced"] is True


def test_engine_cross_check_signs_with_engine2_intuition():
    """Cross-engine consistency: the CGE's carbon-price volume changes are the same sign
    (negative) as the partial-equilibrium intuition — both say a carbon price cuts output."""
    eng = registry.get("cge_static")
    data = {"SAM": toy_sam(), "emission_intensity": {"BRD": 2.0, "MIL": 0.5}}
    res = eng.run(data=data, shocks=[CarbonPrice(price=0.15)], years=[2020])
    vol = res.data[res.data["variable"] == "volume_change"]["value"]
    assert (vol < 0).all()  # every sector's volume falls, same sign as Engine 2


def test_engine_rejects_revenue_recycling():
    eng = registry.get("cge_static")
    with pytest.raises(ValueError, match="none"):
        eng.run(
            data={"SAM": toy_sam()},
            shocks=[CarbonPrice(price=0.1, revenue_recycling="lump_sum")],
            years=[2020],
        )
