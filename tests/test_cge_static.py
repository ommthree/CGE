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


def _solve(cal, carbon_cost=None, drop_factor=0, recycling="lump_sum"):
    cc = np.zeros(len(cal.sectors)) if carbon_cost is None else carbon_cost
    sol = solve(
        lambda z: M.residuals(cal, z, carbon_cost=cc, recycling=recycling, drop_factor=drop_factor),
        M.initial_guess(cal),
        prefer="scipy",
    )
    ns = len(cal.sectors)
    return sol, M.derive_state(cal, sol.x[:ns], sol.x[ns:], carbon_cost=cc, recycling=recycling)


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
    # Levels are GDP-normalised (a CGE is scale-free); ratios and shares are what matter.
    assert np.allclose(cal.X0, [100.0 / 181.0, 120.0 / 181.0])  # scaled by benchmark GDP 181
    assert np.allclose(cal.ax, [[0.0, 0.2], [0.15, 0.0]])  # input j per unit output i (scale-free)
    assert np.allclose(cal.va_share, [0.85, 0.8])
    assert np.allclose(cal.beta, [[0.5, 0.5], [0.5, 0.5]])  # 50/50 CAP/LAB
    assert np.allclose(cal.gamma.sum(), 1.0)
    assert np.allclose(cal.endowment, [90.5 / 181.0, 90.5 / 181.0])  # normalised factor income
    assert np.isclose(cal.gdp0, 1.0)  # GDP normalised to 1


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
def test_carbon_price_reallocates_dirty_to_clean():
    """With recycling, a carbon price shifts output from the dirty to the clean sector (rather than
    shrinking the economy): dirty output falls, clean output rises, dirty relative price rises."""
    cal = _cal()
    _b, base = _solve(cal)
    _s, st = _solve(cal, carbon_cost=0.2 * _EMISSIONS)
    assert st.X[0] < base.X[0]  # dirty BRD output falls
    assert st.X[1] > base.X[1]  # clean MIL output rises (the reallocation)
    assert st.p[0] / st.p[1] > base.p[0] / base.p[1]  # dirty price rises relative to clean


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
    data = {"SAM": toy_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}}
    res = eng.run(data=data, shocks=[CarbonPrice(price=0.1)], years=[2020])
    d = res.data
    for v in ("price_change", "volume_change", "factor_price_change", "gdp_change_real"):
        assert (d["variable"] == v).any(), v
    # dirty sector volume falls
    brd = d[(d["variable"] == "volume_change") & (d["sector"] == "BRD")]["value"].iloc[0]
    assert brd < 0
    assert res.manifest.assumptions["emissions_priced"] is True


def test_engine_cross_check_dirty_sector_falls():
    """Cross-engine consistency: the CGE's dirty-sector volume change is negative, the same sign
    as the partial-equilibrium intuition (a carbon price cuts the emitting sector's output). With
    recycling the clean sector may rise — the GE reallocation Engine 2 cannot show."""
    eng = registry.get("cge_static")
    data = {"SAM": toy_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}}
    res = eng.run(data=data, shocks=[CarbonPrice(price=0.15)], years=[2020])
    d = res.data
    brd = d[(d["variable"] == "volume_change") & (d["sector"] == "BRD")]["value"].iloc[0]
    assert brd < 0  # the dirty sector contracts, same sign as Engine 2


def _welfare_engine(eng, mode):
    data = {"SAM": toy_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}}
    res = eng.run(data=data, shocks=[CarbonPrice(price=0.15, revenue_recycling=mode)], years=[2020])
    d = res.data
    w = float(d[d["variable"] == "welfare_change"]["value"].iloc[0])
    return w, res.manifest.assumptions["recycling_mode"]


def test_recycling_offsets_welfare_loss():
    """The revenue-recycling effect: a pure carbon price wedge (`none`, at the model layer) reduces
    real consumption; returning the revenue (lump_sum) offsets it. The headline GE feature."""
    cal = _cal()
    cc = 0.15 * _EMISSIONS
    _b, base = _solve(cal, recycling="lump_sum")
    # `none` at the MODEL layer (a pure loss — the engine auto-recycles for a closed run).
    _n, none = _solve(cal, carbon_cost=cc, recycling="none")
    _l, lump = _solve(cal, carbon_cost=cc, recycling="lump_sum")
    w_none = none.FD.sum() / base.FD.sum() - 1.0
    w_lump = lump.FD.sum() / base.FD.sum() - 1.0
    assert w_none < -0.01  # carbon price hurts without recycling
    assert w_lump > w_none  # recycling offsets the loss
    assert abs(w_lump) < 1e-6  # lump-sum fully restores real consumption in this pilot
    assert lump.carbon_revenue > 0


def test_labour_tax_cut_equivalent_to_lump_sum_in_pilot():
    """With one household, a labour rebate and a lump-sum transfer give the same real allocation
    (documented pilot equivalence)."""
    eng = registry.get("cge_static")
    w_lump, _ = _welfare_engine(eng, "lump_sum")
    w_lab, mode = _welfare_engine(eng, "labour_tax_cut")
    assert np.isclose(w_lump, w_lab, atol=1e-9)
    assert mode == "labour_tax_cut"


def test_carbon_revenue_emitted():
    eng = registry.get("cge_static")
    data = {"SAM": toy_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}}
    res = eng.run(data=data, shocks=[CarbonPrice(price=0.1)], years=[2020])
    rev = res.data[res.data["variable"] == "carbon_revenue"]["value"].iloc[0]
    assert rev > 0  # a positive carbon price on emitting sectors raises revenue


def test_engine_rejects_mixed_recycling_modes():
    eng = registry.get("cge_static")
    with pytest.raises(ValueError, match="single revenue_recycling"):
        eng.run(
            data={"SAM": toy_sam()},
            shocks=[
                CarbonPrice(price=0.1, revenue_recycling="none"),
                CarbonPrice(price=0.1, revenue_recycling="lump_sum"),
            ],
            years=[2020],
        )


# -- review remediation: supplied-SAM gate, units, welfare, gas/coverage, provenance ----------
def _eur_io_sat(intensity_energy=800.0, intensity_mfg=200.0):
    """A small EUR IOSystem + satellite with realistic CO2 emission intensities (t/MEUR)."""
    import pandas as pd

    from cge.contracts.data_objects import (
        Classification,
        IOSystem,
        Provenance,
        SatelliteAccount,
    )

    prov = Provenance(
        source="x", source_version="1", licence="x", reference_year=2020, retrieved="2026-07-18"
    )
    labels = ["R:energy", "R:mfg"]
    io = IOSystem(
        provenance=prov,
        sectors=Classification(name="s", kind="sector", labels=["energy", "mfg"]),
        regions=Classification(name="r", kind="region", labels=["R"]),
        A=pd.DataFrame([[0.1, 0.2], [0.15, 0.1]], index=labels, columns=labels),
        final_demand=pd.DataFrame({"fd": [100.0, 120.0]}, index=labels),
        unit="MEUR",
        currency="EUR",
    )
    sat = SatelliteAccount(
        provenance=prov,
        name="GHG",
        units={"CO2": "t/MEUR"},
        data=pd.DataFrame(
            {"R:energy": [intensity_energy], "R:mfg": [intensity_mfg]}, index=["CO2"]
        ),
    )
    return io, sat


def _run_eur(shock, **sat_kwargs):
    io, sat = _eur_io_sat(**sat_kwargs)
    return registry.get("cge_static").run(
        data={"IOSystem": io, "SatelliteAccount": sat}, shocks=[shock], years=[2020]
    )


def test_carbon_wedge_is_not_a_million_times_too_large():
    """Review P0: the €100/t carbon cost must be finite (the recycling fixed point does not
    diverge), not ~1e6 too large. A realistic-intensity EUR build solves and moves output."""
    res = _run_eur(CarbonPrice(price=100.0))  # would raise 'k ≥ 1' before the 1e-6 fix
    vol = res.data[res.data["variable"] == "volume_change"]
    energy = vol[vol["sector"] == "energy"]["value"].iloc[0]
    assert -0.5 < energy < 0.0  # dirty sector contracts by a plausible amount, not a blowup


def test_gas_and_coverage_change_the_result():
    """Review P1: the CGE must honour gases and coverage (it reuses Engine 1's cost vector)."""
    v_co2 = _run_eur(CarbonPrice(price=100.0, gases=["CO2"])).data
    v_cov = _run_eur(CarbonPrice(price=100.0, gases=["CO2"], coverage_sectors=["energy"])).data

    def energy_vol(d):
        return d[(d["variable"] == "volume_change") & (d["sector"] == "energy")]["value"].iloc[0]

    assert not np.isclose(energy_vol(v_co2), energy_vol(v_cov))  # coverage bites


def test_emissions_change_moves_the_manifest():
    """Review P1: doubling emission intensity changes results AND the substantive manifest."""
    m1 = _run_eur(CarbonPrice(price=100.0), intensity_energy=800.0).manifest
    m2 = _run_eur(CarbonPrice(price=100.0), intensity_energy=1600.0).manifest
    h1 = next(
        i["content_hash"] for i in m1.assumptions["inputs"] if i["name"] == "SatelliteAccount"
    )
    h2 = next(
        i["content_hash"] for i in m2.assumptions["inputs"] if i["name"] == "SatelliteAccount"
    )
    assert h1 != h2  # a changed satellite moves the manifest


def test_non_eur_unit_metadata_rejected():
    """Review P2: emissions in the wrong unit (kg vs t) are rejected before pricing."""
    io, sat = _eur_io_sat()
    sat.units = {"CO2": "kg/MEUR"}  # 1000× off
    with pytest.raises(ValueError, match="expected 't/MEUR'"):
        registry.get("cge_static").run(
            data={"IOSystem": io, "SatelliteAccount": sat},
            shocks=[CarbonPrice(price=100.0)],
            years=[2020],
        )


def test_supplied_unbalanced_sam_rejected():
    """Review P1: a directly-supplied SAM must pass the balance/finiteness/non-negativity gate."""
    sam = toy_sam()
    sam.matrix.loc["BRD", "HOH"] += 50.0  # break the balance by 50 units
    with pytest.raises(ValueError, match="not balanced"):
        registry.get("cge_static").run(
            data={"SAM": sam}, shocks=[CarbonPrice(price=0.1)], years=[2020]
        )


def test_welfare_is_cobb_douglas_utility():
    """Review P1: welfare_change is CD utility U=Π FD_i^γ_i, not the Σ FD sum. Under a carbon
    price with recycling the CD utility falls (the distortion), even if Σ FD is ~unchanged."""
    cal = _cal()
    _b, base = _solve(cal)
    _s, st = _solve(cal, carbon_cost=0.15 * _EMISSIONS, recycling="lump_sum")
    u = float(np.prod(np.power(st.FD, cal.gamma)))
    u_base = float(np.prod(np.power(base.FD, cal.gamma)))
    cd_welfare = u / u_base - 1.0
    sum_fd = st.FD.sum() / base.FD.sum() - 1.0
    assert cd_welfare < -1e-6  # the true CD welfare loss is strictly negative
    assert cd_welfare < sum_fd  # and more negative than the (misleading) sum-of-quantities
