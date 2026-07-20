"""Tests for the multi-region CGE (Phase 5.4 — true bilateral trade).

Pins the globally-balanced multi-region toy SAM, benchmark replication (including bilateral
imports/exports), homogeneity, and cross-region carbon leakage.
"""

import numpy as np
import pytest

from cge.contracts.shocks import CarbonPrice
from cge.data.sam.balance import is_balanced
from cge.data.sam.toy_multi import (
    REGIONS,
    SECTORS,
    SPARSE_REGIONS,
    SPARSE_SECTORS,
    toy_multi_sam,
    toy_multi_sparse_sam,
)
from cge.engines.cge_static import model_multi as MM
from cge.engines.cge_static.calibrate_multi import calibrate_multi
from cge.engines.cge_static.solver import solve

_FACTORS = ["CAP", "LAB"]


def _cal(**kw):
    return calibrate_multi(
        toy_multi_sam(), regions=REGIONS, sectors=SECTORS, factors=_FACTORS, **kw
    )


def _solve(cal, carbon_cost=None):
    sol = solve(
        lambda z: MM.residuals(cal, z, carbon_cost=carbon_cost, recycling="lump_sum"),
        MM.initial_guess(cal) * 1.03,
        prefer="scipy",
    )
    st = MM.unpack_state(cal, sol.x, carbon_cost=carbon_cost, recycling="lump_sum")
    return sol, st


def test_multi_sam_globally_balanced():
    sam = toy_multi_sam()
    assert is_balanced(sam.matrix, tol=1e-9)
    # Genuine two-way bilateral trade in both goods.
    m = sam.matrix
    assert m.loc["a_N_BRD", "c_S_BRD"] > 0  # North exports BRD to South
    assert m.loc["a_S_BRD", "c_N_BRD"] > 0  # South exports BRD to North


def test_multi_calibration_reproduces_benchmark_identities():
    cal = _cal()
    # Composite supply = domestic + imports; output = domestic + exports.
    assert np.allclose(cal.Q0, cal.D0 + cal.M0.sum(axis=2), atol=1e-12)
    assert np.allclose(cal.Z0, cal.D0 + cal.EX0.sum(axis=2), atol=1e-12)
    # Bilateral consistency: M[d,s,o] == EX[o,s,d].
    for d in range(cal.nr):
        for o in range(cal.nr):
            if d != o:
                assert np.allclose(cal.M0[d, :, o], cal.EX0[o, :, d], atol=1e-12)
    # Armington/CET shares sum to 1.
    assert np.allclose(cal.arm_share_d + cal.arm_share_m.sum(axis=2), 1.0)
    assert np.allclose(cal.cet_share_d + cal.cet_share_e.sum(axis=2), 1.0)


def test_multi_residual_system_is_square():
    cal = _cal()
    assert MM.residuals(cal, MM.initial_guess(cal)).shape == (MM.n_unknowns(cal),)


def test_multi_benchmark_replication():
    """The multi-region model replicates its benchmark to machine precision — every quantity,
    including bilateral imports and exports, returns to the SAM values at unit prices."""
    cal = _cal()
    sol, st = _solve(cal)
    assert np.allclose(sol.x, 1.0, atol=1e-7)  # all prices return to 1
    assert np.allclose(st.Z, cal.Z0, atol=1e-7)
    assert np.allclose(st.D, cal.D0, atol=1e-7)
    assert np.allclose(st.M, cal.M0, atol=1e-7)  # bilateral imports
    assert np.allclose(st.EX, cal.EX0, atol=1e-7)  # bilateral exports
    assert np.allclose(st.FD, cal.FD0, atol=1e-7)


def test_multi_numeraire_pins_every_region_not_just_region_zero():
    """P2 (review round 9): the numéraire is a single GLOBAL constraint (region 0's CPI = 1), but in
    a connected system that fixes the common nominal scale for EVERY region — every region's pq is
    fully determined at the solved equilibrium, not just region 0's. This pins the correct rationale
    against drifting back to the (wrong) claim that non-numéraire regions' prices are 'unpinned'."""
    cal = _cal()
    cc = np.zeros((cal.nr, cal.ns))
    cc[0, 0] = 0.3  # shock region 0 only, so other regions' prices actually move
    sol, st = _solve(cal, carbon_cost=cc)
    assert sol.residual_norm < 1e-8  # a genuine equilibrium was found — nothing was left free
    # Every region's pq is a single well-defined number (not a degenerate/free direction):
    # perturbing any region's solved pq away from its equilibrium value must violate a residual.
    for ri in range(cal.nr):
        for si in range(cal.ns):
            pd, pq, pe, w = MM._unpack(cal, sol.x)
            pq_perturbed = pq.copy()
            pq_perturbed[ri, si] *= 1.01
            # Re-pack: only pq changes, everything else stays at the solved point.
            z = sol.x.copy()
            o1 = cal.nr * cal.ns
            z[o1 : 2 * o1] = pq_perturbed.ravel()
            resid = np.abs(MM.residuals(cal, z, carbon_cost=cc)).max()
            assert resid > 1e-6, (
                f"region {ri} sector {si}: perturbing pq left residuals unchanged — its price "
                "is not actually pinned by the system"
            )


def test_multi_homogeneity():
    """Scaling both regions' endowments by k scales all real quantities by k, prices unchanged."""
    from dataclasses import replace

    cal = _cal()
    sol, st = _solve(cal)
    cal_k = replace(cal, endowment=cal.endowment * 1.4, foreign_savings=cal.foreign_savings * 1.4)
    sol_k, st_k = _solve(cal_k)
    assert np.allclose(sol.x, sol_k.x, atol=1e-6)  # prices unchanged
    assert np.allclose(st_k.Z, 1.4 * st.Z, atol=1e-6)  # real output scales


def test_multi_carbon_price_causes_cross_region_leakage():
    """The signature multi-region result: a carbon price on ONE region's dirty sector cuts that
    region's output, RAISES its imports of the good from the partner region (cross-region leakage),
    and RAISES the partner's output of that good."""
    cal = _cal()
    _b, base = _solve(cal)
    cc = np.zeros((cal.nr, cal.ns))
    cc[0, 0] = 0.3  # North (region 0), BRD (sector 0) — the taxed dirty sector
    _s, shk = _solve(cal, carbon_cost=cc)
    assert shk.Z[0, 0] < base.Z[0, 0]  # North BRD output falls
    assert shk.M[0, 0, 1] > base.M[0, 0, 1]  # North imports more BRD from South (leakage)
    assert shk.Z[1, 0] > base.Z[1, 0]  # South BRD output rises (production relocates)


def test_multi_bilateral_markets_clear_under_shock():
    """THE correctness test the earlier design failed: at the SHOCKED equilibrium every bilateral
    goods market clears (import demand M[d,s,o] = export supply EX[o,s,d]) — the equations the
    law-of-one-price reduction omitted, which let a machine-zero solver residual coexist with a 15%
    trade imbalance."""
    cal = _cal()
    cc = np.zeros((cal.nr, cal.ns))
    cc[0, 0] = 0.3
    _s, st = _solve(cal, carbon_cost=cc)
    max_disc = 0.0
    for d in range(cal.nr):
        for o in range(cal.nr):
            if d != o:
                max_disc = max(max_disc, float(np.max(np.abs(st.M[d, :, o] - st.EX[o, :, d]))))
    assert max_disc < 1e-8, f"bilateral market not cleared: {max_disc:.2e}"


def test_multi_all_factor_markets_clear_under_shock():
    """Every regional factor market clears at the shocked equilibrium (the one dropped by Walras
    included) — the earlier design left the dropped market ~0.7% off."""
    cal = _cal()
    cc = np.zeros((cal.nr, cal.ns))
    cc[0, 0] = 0.3
    _s, st = _solve(cal, carbon_cost=cc)
    for fi in range(cal.nf):
        for ri in range(cal.nr):
            gap = abs(float(st.F[fi, ri, :].sum()) - cal.endowment[fi, ri])
            assert gap < 1e-8, f"factor {fi} region {ri} market not cleared: {gap:.2e}"


@pytest.mark.parametrize("bad", [-2.0, 0.0])
def test_multi_trade_elasticity_must_be_positive(bad):
    with pytest.raises(ValueError, match="positive"):
        calibrate_multi(
            toy_multi_sam(), regions=REGIONS, sectors=SECTORS, factors=_FACTORS, arm_elast=bad
        )


def test_multi_trade_elasticity_one_rejected():
    with pytest.raises(ValueError, match="must be ≠ 1"):
        calibrate_multi(
            toy_multi_sam(), regions=REGIONS, sectors=SECTORS, factors=_FACTORS, arm_elast=1.0
        )


# -- engine dispatch ----------------------------------------------------------
def test_engine_dispatches_to_multi_region():
    from cge.contracts.engine import registry

    eng = registry.get("cge_static")
    res = eng.run(data={"SAM": toy_multi_sam()}, shocks=[CarbonPrice(price=0.0)], years=[2020])
    assert res.data["value"].abs().max() < 1e-6  # zero-shock replication
    assert "multi-region" in res.manifest.assumptions["model_variant"]
    assert sorted(res.data["region"].unique()) == ["N", "S"]
    # Per-region GDP and factor prices are emitted.
    assert (res.data["variable"] == "real_consumption_change").sum() == len(REGIONS)


def test_engine_multi_region_cross_region_leakage():
    from cge.contracts.engine import registry

    eng = registry.get("cge_static")
    res = eng.run(
        data={"SAM": toy_multi_sam(), "carbon_cost_share": {"N": {"BRD": 0.3}}},
        shocks=[CarbonPrice(price=1.0)],
        years=[2020],
    )
    d = res.data
    n_brd_vol = d[(d.region == "N") & (d.sector == "BRD") & (d.variable == "volume_change")]
    n_brd_imp = d[(d.region == "N") & (d.sector == "BRD") & (d.variable == "import_change")]
    s_brd_vol = d[(d.region == "S") & (d.sector == "BRD") & (d.variable == "volume_change")]
    assert n_brd_vol["value"].iloc[0] < 0  # North's dirty output falls
    assert n_brd_imp["value"].iloc[0] > 0  # North imports more from South (leakage)
    assert s_brd_vol["value"].iloc[0] > 0  # South's output rises


def test_engine_multi_region_rejects_unbalanced_sam():
    from cge.contracts.engine import registry

    eng = registry.get("cge_static")
    bad = toy_multi_sam()
    bad.matrix.iloc[0, 1] += 5.0  # break balance
    with pytest.raises(ValueError, match="not balanced"):
        eng.run(data={"SAM": bad}, shocks=[CarbonPrice(price=0.0)], years=[2020])


def test_engine_multi_region_carbon_share_validation():
    from cge.contracts.engine import registry

    eng = registry.get("cge_static")
    with pytest.raises(ValueError, match="regions not in the SAM"):
        eng.run(
            data={"SAM": toy_multi_sam(), "carbon_cost_share": {"Z": {"BRD": 0.1}}},
            shocks=[CarbonPrice(price=1.0)],
            years=[2020],
        )


def test_multi_zero_impact_run_does_not_overwrite_requested_recycling():
    """P2 regression (review round 9): same fix as the open path — a zero-impact multi-region run
    that explicitly requests revenue_recycling='none' must keep recycling_mode='none' in the
    manifest rather than being silently switched to lump_sum with recycling_defaulted_from_none
    reporting False."""
    from cge.contracts.engine import registry

    eng = registry.get("cge_static")
    res = eng.run(
        data={"SAM": toy_multi_sam()},
        shocks=[CarbonPrice(price=0.0, revenue_recycling="none")],
        years=[2020],
    )
    assert res.manifest.assumptions["recycling_mode"] == "none"
    assert res.manifest.assumptions["recycling_defaulted_from_none"] is False
    assert res.manifest.assumptions["emissions_priced"] is False


def test_multi_priced_run_still_defaults_none_to_lump_sum():
    """The positive-revenue multi-region case must still default: 'none' with a genuinely positive
    carbon cost switches to lump_sum AND records the flag."""
    from cge.contracts.engine import registry

    eng = registry.get("cge_static")
    res = eng.run(
        data={"SAM": toy_multi_sam(), "carbon_cost_share": {"N": {"BRD": 0.3}}},
        shocks=[CarbonPrice(price=1.0, revenue_recycling="none")],
        years=[2020],
    )
    assert res.manifest.assumptions["recycling_mode"] == "lump_sum"
    assert res.manifest.assumptions["recycling_defaulted_from_none"] is True
    assert res.manifest.assumptions["emissions_priced"] is True


# -- Sparse bilateral trade (review 2026-07): inactive routes must not be free unknowns ---------
def _sparse_cal(**kw):
    return calibrate_multi(
        toy_multi_sparse_sam(),
        regions=SPARSE_REGIONS,
        sectors=SPARSE_SECTORS,
        factors=_FACTORS,
        **kw,
    )


def test_multi_active_routes_excludes_zero_trade_routes():
    """Only routes with genuine benchmark trade are 'active'; the fixture's BRD N-E and S-E routes
    (structurally zero) must not appear."""
    cal = _sparse_cal()
    ns_brd = SPARSE_SECTORS.index("BRD")
    n_idx, s_idx, e_idx = (SPARSE_REGIONS.index(r) for r in ("N", "S", "E"))
    active = set(cal.active_routes)
    # BRD trades only N<->S.
    assert (n_idx, ns_brd, s_idx) in active
    assert (s_idx, ns_brd, n_idx) in active
    assert (n_idx, ns_brd, e_idx) not in active
    assert (e_idx, ns_brd, n_idx) not in active
    assert (s_idx, ns_brd, e_idx) not in active
    assert (e_idx, ns_brd, s_idx) not in active
    # 12 possible directed routes (3 regions x 2 partners x 2 sectors), only 8 active.
    assert len(cal.active_routes) == 8


def test_multi_sparse_topology_jacobian_is_full_rank():
    """THE regression for the review's P1: with the fix, only active routes get a price unknown, so
    the benchmark Jacobian is full rank — no free direction from an unpinned zero-trade route price
    (previously: rank fell by exactly the number of zero routes)."""
    cal = _sparse_cal()
    x0 = MM.initial_guess(cal)
    n = len(x0)
    eps = 1e-6
    r0 = MM.residuals(cal, x0)
    jac = np.zeros((n, n))
    for j in range(n):
        dz = x0.copy()
        dz[j] += eps
        jac[:, j] = (MM.residuals(cal, dz) - r0) / eps
    rank = np.linalg.matrix_rank(jac, tol=1e-8)
    assert rank == n, f"Jacobian rank {rank} < n_unknowns {n}: still rank-deficient"


def test_multi_sparse_topology_replicates_and_solves_uniquely():
    """The sparse-topology SAM replicates its benchmark and produces a genuine equilibrium: unlike
    the pre-fix design, perturbing an inactive route's price is not even possible (it no longer has
    an unknown), and the solve converges to machine precision with the bilateral markets cleared."""
    cal = _sparse_cal()
    sol, st = _solve(cal)
    assert sol.residual_norm < 1e-8
    assert np.allclose(st.Z, cal.Z0, atol=1e-6)
    assert np.allclose(st.M, cal.M0, atol=1e-6)
    assert np.allclose(st.EX, cal.EX0, atol=1e-6)
    assert MM.n_unknowns(cal) == 2 * cal.nr * cal.ns + len(cal.active_routes) + cal.nf * cal.nr
