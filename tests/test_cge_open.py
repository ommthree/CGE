"""Tests for the open-economy CGE (Phase 5 — Armington imports / CET exports).

Pins benchmark replication (activity output, domestic sales, imports, exports), the CES/CET
calibration identities, carbon leakage direction, homogeneity, and the engine dispatch.
"""

import numpy as np
import pytest

from cge.contracts.engine import registry
from cge.contracts.shocks import CarbonPrice
from cge.data.sam import toy_open_sam
from cge.data.sam.balance import is_balanced
from cge.engines.cge_static import model_open as MO
from cge.engines.cge_static.calibrate_open import calibrate_open
from cge.engines.cge_static.solver import solve

_SECTORS = ["BRD", "MIL"]
_FACTORS = ["CAP", "LAB"]
_EMISSIONS = np.array([2.0, 0.5])


def _cal(**kw):
    return calibrate_open(toy_open_sam(), sectors=_SECTORS, factors=_FACTORS, **kw)


def _solve(cal, carbon_cost=None):
    ns, nf = len(cal.sectors), len(cal.factors)
    cc = np.zeros(ns) if carbon_cost is None else carbon_cost
    sol = solve(
        lambda z: MO.residuals(cal, z, carbon_cost=cc, recycling="lump_sum"),
        MO.initial_guess(cal) * 1.04,
        prefer="scipy",
    )
    st = MO.derive_open_state(
        cal,
        sol.x[:ns],
        sol.x[ns : 2 * ns],
        sol.x[2 * ns : 2 * ns + nf],
        float(sol.x[-1]),
        carbon_cost=cc,
        recycling="lump_sum",
    )
    return sol, st


# -- SAM ----------------------------------------------------------------------
def test_open_sam_balanced_and_trade_balanced():
    sam = toy_open_sam()
    assert is_balanced(sam.matrix)
    m = sam.matrix
    imports = m.loc["ROW"].sum()  # ROW receives import payments
    exports = m["ROW"].sum()  # ROW pays for exports
    assert np.isclose(imports, exports)  # zero foreign savings at benchmark


# -- calibration --------------------------------------------------------------
def test_open_calibration_benchmark_prices_are_one():
    """The Armington composite price and CET output price both equal 1 at benchmark prices."""
    cal = _cal()
    one = np.ones(len(cal.sectors))
    assert np.allclose(MO._armington_price(cal, one, one), 1.0, atol=1e-12)
    assert np.allclose(MO._cet_price(cal, one, one), 1.0, atol=1e-12)


def test_open_calibration_reproduces_benchmark_quantities():
    """Composite supply Q = intermediate use + final demand; D+M = Q; D+E = Z at benchmark."""
    cal = _cal()
    assert np.allclose(cal.D0 + cal.M0, cal.Q0, atol=1e-12)
    assert np.allclose(cal.D0 + cal.E0, cal.Z0, atol=1e-12)


# -- CGE correctness ----------------------------------------------------------
def test_open_benchmark_replication():
    cal = _cal()
    sol, st = _solve(cal)
    assert np.allclose(sol.x, 1.0, atol=1e-7)  # all prices + er return to 1
    assert np.allclose(st.Z, cal.Z0, atol=1e-7)
    assert np.allclose(st.D, cal.D0, atol=1e-7)
    assert np.allclose(st.E, cal.E0, atol=1e-7)
    assert np.allclose(st.M, cal.M0, atol=1e-7)


def test_open_homogeneity():
    """Scaling factor endowments scales real trade/output; prices and the exchange rate unchanged
    (no money illusion in the open model)."""
    from dataclasses import replace

    cal = _cal()
    sol, st = _solve(cal)
    k = 1.5
    cal_k = replace(cal, endowment=cal.endowment * k)
    sol_k, st_k = _solve(cal_k)
    assert np.allclose(sol.x, sol_k.x, atol=1e-6)  # prices + er unchanged
    assert np.allclose(st_k.Z, k * st.Z, atol=1e-6)  # real output scales


def test_open_carbon_price_causes_leakage():
    """The signature open-economy result: a carbon price on the dirty sector cuts its output,
    RAISES its imports (leakage) and CUTS its exports (competitiveness)."""
    cal = _cal()
    _b, base = _solve(cal)
    _s, st = _solve(cal, carbon_cost=0.15 * _EMISSIONS)
    assert st.Z[0] < base.Z[0]  # dirty output falls
    assert st.M[0] > base.M[0]  # imports of the dirty good rise (leakage)
    assert st.E[0] < base.E[0]  # exports of the dirty good fall
    assert st.Z[1] > base.Z[1]  # clean sector expands


def test_open_trade_balance_holds():
    """At the shocked equilibrium the value trade balance still closes (Σ pm·M = Σ pe·E)."""
    cal = _cal()
    _s, st = _solve(cal, carbon_cost=0.15 * _EMISSIONS)
    pm = st.er * np.ones(len(cal.sectors))
    pe = st.er * np.ones(len(cal.sectors))
    assert np.isclose(float(pm @ st.M), float(pe @ st.E), atol=1e-8)


def test_higher_armington_elasticity_more_leakage():
    """Monotonicity: a higher Armington elasticity → more import substitution under the tax."""
    cc = 0.15 * _EMISSIONS
    _lo, lo = _solve(_cal(arm_elast=1.5), carbon_cost=cc)
    _hi, hi = _solve(_cal(arm_elast=4.0), carbon_cost=cc)
    _blo, blo = _solve(_cal(arm_elast=1.5))
    _bhi, bhi = _solve(_cal(arm_elast=4.0))
    leak_lo = lo.M[0] / blo.M[0] - 1.0
    leak_hi = hi.M[0] / bhi.M[0] - 1.0
    assert leak_hi > leak_lo  # more elastic → more import leakage


# -- engine dispatch ----------------------------------------------------------
def test_engine_dispatches_to_open_on_row_account():
    eng = registry.get("cge_static")
    res = eng.run(data={"SAM": toy_open_sam()}, shocks=[CarbonPrice(price=0.0)], years=[2020])
    assert res.data["value"].abs().max() < 1e-7  # zero-shock replication
    assert "open economy" in res.manifest.assumptions["model_variant"]
    for v in ("import_change", "export_change", "exchange_rate_change"):
        assert (res.data["variable"] == v).any(), v


def test_engine_open_carbon_run_emits_leakage():
    eng = registry.get("cge_static")
    data = {"SAM": toy_open_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}}
    res = eng.run(data=data, shocks=[CarbonPrice(price=0.1)], years=[2020])
    d = res.data
    brd_imp = d[(d["variable"] == "import_change") & (d["sector"] == "BRD")]["value"].iloc[0]
    brd_vol = d[(d["variable"] == "volume_change") & (d["sector"] == "BRD")]["value"].iloc[0]
    assert brd_vol < 0 and brd_imp > 0  # output down, imports up (leakage)


def test_open_requires_carbon_cost_share_for_positive_price():
    eng = registry.get("cge_static")
    with pytest.raises(ValueError, match="requires a 'carbon_cost_share'"):
        eng.run(data={"SAM": toy_open_sam()}, shocks=[CarbonPrice(price=100.0)], years=[2020])
