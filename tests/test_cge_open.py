"""Tests for the open-economy CGE (Phase 5 — Armington imports / CET exports).

Pins benchmark replication (activity output, domestic sales, imports, exports), the CES/CET
calibration identities, carbon leakage direction, homogeneity, and the engine dispatch.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from cge.contracts.data_objects import SAM, Provenance
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


def test_armington_sensitivity_sweep_bands():
    """The sensitivity sweep returns a low/central/high envelope; higher Armington elasticity → more
    import leakage for the dirty sector (the leakage channel is elasticity-sensitive)."""
    from cge.engines.cge_static.engine import armington_sensitivity_sweep

    res = armington_sensitivity_sweep(
        {"SAM": toy_open_sam(), "carbon_cost_share": {"BRD": 2.0, "MIL": 0.5}},
        [CarbonPrice(price=0.1)],
        elasticities=(1.5, 2.0, 4.0),
    )
    sw = res.bands
    assert set(sw.columns) == {"sector", "variable", "low", "central", "high"}
    brd_imp = sw[(sw["sector"] == "BRD") & (sw["variable"] == "import_change")].iloc[0]
    assert brd_imp["high"] > brd_imp["central"] > brd_imp["low"] > 0  # more elastic → more leakage
    # Provenance is complete (review P2): exact elasticities, version, scenario hash, SAM identity,
    # and each band's full manifest are retained, so an exported sweep is identifiable.
    assert res.elasticities == (1.5, 2.0, 4.0)
    assert res.swept_parameter == "armington_elast"
    assert res.engine_version and res.scenario_hash and res.sam_identity
    assert set(res.manifests) == {"low", "central", "high"}


def test_armington_elasticity_one_rejected():
    """σ = 1 is the Cobb-Douglas trade special case (singular here); rejected with guidance."""
    with pytest.raises(ValueError, match="must be ≠ 1"):
        calibrate_open(toy_open_sam(), sectors=_SECTORS, factors=_FACTORS, arm_elast=1.0)


# -- 2026-07 review remediation -----------------------------------------------
#
# The findings below are the ones the independent review reproduced; each test pins the fix.


def _build_open_sam(exports: dict, imports: dict, domestic: dict) -> SAM:
    """Assemble a fully-balanced 2-sector open SAM from trade/production dicts (test helper).

    ``exports``/``imports``/``domestic`` are per-sector money flows; value added is the residual of
    activity output over intermediate purchases, split 50/50, and household final demand is the
    residual of each commodity's supply over its intermediate use. When aggregate imports ≠ exports
    (a non-zero current account), the ROW account is balanced by a **ROW→HOH capital transfer**
    equal to the net capital inflow (Σ imports − Σ exports) — so the matrix balances for *any* trade
    inputs, including Sf ≠ 0, and the household spends that inflow."""
    accounts = ["a_BRD", "a_MIL", "c_BRD", "c_MIL", "CAP", "LAB", "HOH", "ROW"]
    inter = {("c_MIL", "a_BRD"): 24.0, ("c_BRD", "a_MIL"): 15.0}
    m = pd.DataFrame(0.0, index=accounts, columns=accounts)
    for s in _SECTORS:
        m.loc[f"a_{s}", f"c_{s}"] = domestic[s]
        m.loc[f"a_{s}", "ROW"] = exports[s]
        m.loc["ROW", f"c_{s}"] = imports[s]
    for (com, act), v in inter.items():
        m.loc[com, act] = v
    for s in _SECTORS:
        output = domestic[s] + exports[s]
        ii = sum(m.loc[c, f"a_{s}"] for c in ("c_BRD", "c_MIL"))
        va = output - ii
        m.loc["CAP", f"a_{s}"] = va / 2.0
        m.loc["LAB", f"a_{s}"] = va / 2.0
    # Net capital inflow that finances a trade deficit (0 when trade is balanced): ROW → HOH.
    m.loc["HOH", "ROW"] = sum(imports.values()) - sum(exports.values())
    for s in _SECTORS:
        m.loc[f"c_{s}", "HOH"] = m[f"c_{s}"].sum() - m.loc[f"c_{s}"].sum()
    m.loc["HOH", "CAP"] = m.loc["CAP", ["a_BRD", "a_MIL"]].sum()
    m.loc["HOH", "LAB"] = m.loc["LAB", ["a_BRD", "a_MIL"]].sum()
    prov = Provenance(
        source="test",
        source_version="v",
        licence="n/a",
        reference_year=0,
        retrieved=date.today().isoformat(),
        notes="test open SAM",
    )
    return SAM(provenance=prov, accounts=accounts, matrix=m)


def test_open_dispatch_rejects_unbalanced_sam():
    """P1: the open path now gates the SAM before calibration (it previously bypassed the check)."""
    eng = registry.get("cge_static")
    bad = toy_open_sam()
    bad.matrix.iloc[0, 1] += 7.0  # break balance in one cell
    with pytest.raises(ValueError, match="not balanced"):
        eng.run(data={"SAM": bad}, shocks=[], years=[2020])


def test_open_dispatch_rejects_unsupported_shock_controls():
    """P1: gas selection and spatial coverage are rejected on the open path, not ignored."""
    eng = registry.get("cge_static")
    data = {"SAM": toy_open_sam(), "carbon_cost_share": {"BRD": 0.2, "MIL": 0.05}}
    with pytest.raises(ValueError, match="cannot select gases"):
        eng.run(data=data, shocks=[CarbonPrice(price=1.0, gases=["CH4"])], years=[2020])
    with pytest.raises(ValueError, match="coverage"):
        eng.run(data=data, shocks=[CarbonPrice(price=1.0, coverage_regions=["ZZ"])], years=[2020])


def test_open_zero_export_sector_runs():
    """P1: a balanced SAM with a non-exporting sector calibrates and solves (no 0**-Ω = inf)."""
    sam = _build_open_sam(
        exports={"BRD": 30.0, "MIL": 0.0},  # MIL exports nothing
        imports={"BRD": 12.0, "MIL": 18.0},  # aggregate trade balanced (30 = 30)
        domestic={"BRD": 80.0, "MIL": 110.0},
    )
    assert is_balanced(sam.matrix, tol=1e-9)
    cal = calibrate_open(sam, sectors=_SECTORS, factors=_FACTORS)
    assert cal.cet_share_e[1] == 0.0  # structural zero recorded
    pz = MO._cet_price(cal, np.ones(2), np.ones(2))
    assert np.all(np.isfinite(pz))  # the CET dual no longer blows up
    eng = registry.get("cge_static")
    res = eng.run(data={"SAM": sam}, shocks=[], years=[2020])
    assert not res.data["value"].isna().any()


def test_open_replication_gate_rejects_unsupported_topology():
    """P1 (round 2): a balanced SAM with flows outside the implemented topology (an offsetting
    household↔commodity loop) passes the structural validators but does NOT replicate; the
    post-calibration replication gate rejects it rather than reporting changes vs a wrong base."""
    sam = toy_open_sam()
    sam.matrix.loc["c_BRD", "HOH"] += 10.0
    sam.matrix.loc["HOH", "c_BRD"] += 10.0
    assert is_balanced(sam.matrix, tol=1e-9)  # still balanced — passes the structural gate
    eng = registry.get("cge_static")
    with pytest.raises(ValueError, match="does not replicate"):
        eng.run(data={"SAM": sam}, shocks=[CarbonPrice(price=0.0)], years=[2020])


def test_open_sam_fingerprint_is_label_sensitive():
    """P1 (round 2): the SAM fingerprint depends on account labels, not just the numeric block, so
    two economies with permuted axes but an identical value array get distinct run identities."""
    from cge.engines.cge_static.engine import _sam_fingerprint

    s1 = toy_open_sam()
    acc = list(s1.accounts)
    rotated = acc[1:] + acc[:1]
    m2 = pd.DataFrame(
        s1.matrix.to_numpy(), index=rotated, columns=rotated
    )  # same array, new labels
    s2 = SAM(provenance=s1.provenance, accounts=acc, matrix=m2)
    assert _sam_fingerprint(s1) != _sam_fingerprint(s2)


def test_open_nonzero_foreign_savings_replicates():
    """A GENUINELY balanced SAM with a **non-zero current account** (imports 40 > exports 30 ⇒
    Sf = 10, with a ROW→HOH capital transfer of 10 closing the ROW account) calibrates and
    replicates its benchmark to machine precision. Foreign savings enter household income as er·Sf
    (Phase 5 deferred: the ROW closure lifts the earlier balanced-current-account restriction)."""
    sam = _build_open_sam(
        exports={"BRD": 20.0, "MIL": 10.0},
        imports={"BRD": 22.0, "MIL": 18.0},  # Σ imports 40 > Σ exports 30 → Sf = 10 ≠ 0
        domestic={"BRD": 80.0, "MIL": 110.0},
    )
    assert is_balanced(sam.matrix, tol=1e-9)
    cal = calibrate_open(sam, sectors=_SECTORS, factors=_FACTORS)
    assert cal.foreign_savings > 0  # positive net capital inflow
    sol, st = _solve(cal)
    assert np.allclose(sol.x, 1.0, atol=1e-6)  # benchmark prices + er = 1
    assert np.allclose(st.Z, cal.Z0, atol=1e-6)
    assert np.allclose(st.M, cal.M0, atol=1e-6)
    assert np.allclose(st.FD, cal.FD0, atol=1e-6)


def test_open_row_transfer_must_match_net_trade():
    """The SAM's ROW→household transfer must equal net foreign savings (Σimports−Σexports); a
    mismatched ROW capital account is rejected rather than silently non-replicating."""
    sam = _build_open_sam(
        exports={"BRD": 20.0, "MIL": 10.0},
        imports={"BRD": 22.0, "MIL": 18.0},
        domestic={"BRD": 80.0, "MIL": 110.0},
    )
    sam.matrix.loc["HOH", "ROW"] += 5.0  # break the ROW capital-account balance
    sam.matrix.loc["HOH", "CAP"] -= 5.0  # keep the matrix balanced overall
    with pytest.raises(ValueError, match="ROW→household transfer"):
        calibrate_open(sam, sectors=_SECTORS, factors=_FACTORS)


def test_open_zero_import_sector_arm_below_one_warning_free():
    """P3: a non-importing sector with arm_elast < 1 (ρ < 0) calibrates without a divide-by-zero
    warning — the unused import term in arm_scale no longer evaluates 0**ρ."""
    import warnings

    sam = _build_open_sam(
        exports={"BRD": 15.0, "MIL": 15.0},
        imports={"BRD": 0.0, "MIL": 30.0},  # BRD imports nothing; aggregate trade balanced
        domestic={"BRD": 80.0, "MIL": 110.0},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cal = calibrate_open(sam, sectors=_SECTORS, factors=_FACTORS, arm_elast=0.5)
    assert cal.arm_share_m[0] == 0.0  # BRD is non-traded on the import side
    assert np.all(np.isfinite(cal.arm_scale))


def test_open_manifest_distinguishes_carbon_shares_and_elasticities():
    """P1: two runs differing only in carbon shares (or an elasticity vector) get different
    manifests — the substantive inputs are recorded, not just the SAM and first elasticity."""
    eng = registry.get("cge_static")
    base = {"SAM": toy_open_sam()}

    def _assum(share, **extra):
        res = eng.run(
            data={**base, "carbon_cost_share": share, **extra},
            shocks=[CarbonPrice(price=0.1)],
            years=[2020],
        )
        return res.manifest.assumptions

    a1 = _assum({"BRD": 0.2, "MIL": 0.05})
    a2 = _assum({"BRD": 0.1, "MIL": 0.1})
    assert a1 != a2  # different carbon shares ⇒ different provenance
    # Full per-sector vectors recorded, not scalars.
    a3 = _assum({"BRD": 0.2, "MIL": 0.05}, armington_elast=np.array([2.0, 4.0]))
    a4 = _assum({"BRD": 0.2, "MIL": 0.05}, armington_elast=np.array([2.0, 1.5]))
    assert a3["armington_elasticity"] == [2.0, 4.0]
    assert a3 != a4
    assert "va_elast" in a1 and "solver_backends" in a1


def test_open_recycling_income_identity_holds():
    """P2: recycling is solved in closed form; the household budget identity holds exactly (the old
    50-iteration loop could stop with the identity violated while the solver residual was ~0)."""
    cal = _cal()
    st = MO.derive_open_state(
        cal,
        np.ones(2),
        np.ones(2),
        np.ones(2),
        1.0,
        carbon_cost=np.array([0.2, 0.05]),
        recycling="lump_sum",
    )
    factor_income = float(np.dot(np.ones(2), cal.endowment))
    assert abs(st.income - (factor_income + st.carbon_revenue)) < 1e-12


def test_open_recycling_diverges_when_revenue_exceeds_income():
    """P2: a carbon cost so large that revenue ≥ income at the EQUILIBRIUM is rejected (k ≥ 1) in
    strict mode; in non-strict (solver-exploratory) mode it returns a finite clamped state instead
    of raising, so the residual function stays continuous (review round-2 P2)."""
    cal = _cal()
    cc = np.array([1.5, 0.375])
    # strict=True (final accepted equilibrium) → refuse.
    with pytest.raises(ValueError, match="k="):
        MO.derive_open_state(
            cal,
            np.ones(2),
            np.ones(2),
            np.ones(2),
            1.0,
            carbon_cost=cc,
            recycling="lump_sum",
            strict=True,
        )
    # strict=False (exploratory) → finite state, no exception (keeps the solve continuous).
    st = MO.derive_open_state(
        cal, np.ones(2), np.ones(2), np.ones(2), 1.0, carbon_cost=cc, recycling="lump_sum"
    )
    assert np.all(np.isfinite(st.Z))


def test_open_recycling_solver_survives_benchmark_start_on_k_ridge():
    """P2 (round 2): a shocked solve whose benchmark start sits on the k≈1 ridge still converges to
    the valid equilibrium (k<1) — the non-strict clamp + multi-start rescue it, not the guard
    aborting the solve at its initial point."""
    from cge.contracts.engine import registry

    eng = registry.get("cge_static")
    # cc=[1.4,0.35]: k≈1.006 at the benchmark start but ≈0.80 at the equilibrium (per the review).
    res = eng.run(
        data={"SAM": toy_open_sam(), "carbon_cost_share": {"BRD": 1.4, "MIL": 0.35}},
        shocks=[CarbonPrice(price=1.0)],
        years=[2020],
    )
    assert res.manifest.assumptions["solver_max_residual_norm"] < 1e-9


def test_open_residual_system_is_square():
    """P2: the residual vector has exactly n_unknowns rows (the tautological composite-market rows
    were removed), so the system is genuinely square — not 9×7 with two identically-zero rows."""
    cal = _cal()
    z = MO.initial_guess(cal)
    res = MO.residuals(cal, z)
    assert res.shape == (MO.n_unknowns(cal),)


def test_open_gdp_change_real_uses_cpi_weighted_expenditure():
    """P2: gdp_change_real is CPI-weighted expenditure pq·FD (the same contract as the closed
    model), not the unweighted Σ FD — so it equals the Cobb-Douglas welfare move at the CPI num."""
    eng = registry.get("cge_static")
    data = {"SAM": toy_open_sam(), "carbon_cost_share": {"BRD": 0.2, "MIL": 0.05}}
    res = eng.run(data=data, shocks=[CarbonPrice(price=0.1)], years=[2020])
    d = res.data
    gdp = d[(d["variable"] == "gdp_change_real")]["value"].iloc[0]
    welfare = d[(d["variable"] == "welfare_change")]["value"].iloc[0]
    # Under the CPI numéraire, real expenditure and CD utility move together (both = ΔI / CPI).
    assert abs(gdp - welfare) < 1e-6


@pytest.mark.parametrize("bad", [-1.0, 0.0])
def test_open_elasticity_must_be_positive(bad):
    """P2: negative and zero VA/Armington/CET elasticities are rejected (not silently used)."""
    with pytest.raises(ValueError, match="positive"):
        calibrate_open(toy_open_sam(), sectors=_SECTORS, factors=_FACTORS, arm_elast=bad)


def test_open_elasticity_length_one_vector_rejected():
    """P2: a length-1 elasticity vector is rejected, not silently broadcast to every sector."""
    with pytest.raises(ValueError, match="scalar or a length-2"):
        calibrate_open(
            toy_open_sam(), sectors=_SECTORS, factors=_FACTORS, arm_elast=np.array([3.0])
        )


def test_open_va_elast_length_one_rejected():
    """P2: a one-element VA vector raised a raw IndexError before; now a clear ValueError."""
    with pytest.raises(ValueError, match="scalar or a length-2"):
        calibrate_open(toy_open_sam(), sectors=_SECTORS, factors=_FACTORS, va_elast=np.array([0.8]))


def test_sensitivity_sweep_rejects_unordered_bands():
    """P2: an unordered (high, central, low) triple is rejected — it would mislabel the envelope."""
    from cge.engines.cge_static.engine import armington_sensitivity_sweep

    with pytest.raises(ValueError, match="ordered"):
        armington_sensitivity_sweep(
            {"SAM": toy_open_sam(), "carbon_cost_share": {"BRD": 0.2, "MIL": 0.05}},
            [CarbonPrice(price=0.1)],
            elasticities=(4.0, 2.0, 1.5),
        )
