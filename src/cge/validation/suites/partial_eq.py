"""Validation suite for Engine 2 (partial-equilibrium volume response).

Checks tied to docs/models/partial-equilibrium.md §7: sign, proportionality, band ordering,
zero-shock, price pass-through from Engine 1, and elasticity provenance.
"""

from __future__ import annotations

from cge.contracts.shocks import CarbonPrice
from cge.scenarios.loader import Scenario
from cge.validation.framework import check

SUITE = "partial_eq"


def _run(price=100.0):
    from cge.runner import run_scenario

    sc = Scenario(name="v", engine="partial_eq", years=[2020], shocks=[CarbonPrice(price=price)])
    return run_scenario(sc, data_source="toy")


@check(SUITE, "volume_sign_is_negative")
def _sign():
    """A carbon price raises prices; with ε ≤ 0 volumes fall — every central Δq/q ≤ 0."""
    df = _run().data
    vol = df[(df["variable"] == "volume_change") & (df["scenario"] == "central")]["value"]
    mx = float(vol.max())
    return mx <= 1e-12, f"max central Δq/q = {mx:.4f} (should be ≤ 0)", mx, 0.0


@check(SUITE, "proportional_to_price")
def _proportional():
    """Δq/q = ε·Δp exactly: doubling the price doubles the volume response (linearity)."""

    d1 = _run(100.0).data
    d2 = _run(200.0).data

    def central_vol(df):
        v = df[(df["variable"] == "volume_change") & (df["scenario"] == "central")]
        return v.set_index(["region", "sector"])["value"]

    v1, v2 = central_vol(d1), central_vol(d2)
    err = float((v2 - 2.0 * v1).abs().max())
    return err < 1e-9, f"max|Δq(2τ) − 2·Δq(τ)| = {err:.2e}", err, 1e-9


@check(SUITE, "band_ordering")
def _bands():
    """Low band (most elastic) gives the largest volume fall; high the smallest — per good."""
    df = _run().data
    vol = df[df["variable"] == "volume_change"]
    ok = True
    for _key, g in vol.groupby(["region", "sector"]):
        by = {row.scenario: row.value for row in g.itertuples()}
        # low is most negative ≤ central ≤ high ≤ 0
        if not (by["low"] <= by["central"] <= by["high"] <= 1e-12):
            ok = False
            break
    return ok, f"per-good bands ordered low ≤ central ≤ high ≤ 0: {ok}"


@check(SUITE, "prices_match_engine1")
def _price_passthrough():
    """The price_change rows are exactly Engine 1's — Engine 2 passes prices through."""

    from cge.runner import run_scenario

    shocks = [CarbonPrice(price=100.0)]
    pe = run_scenario(
        Scenario(name="pe", engine="partial_eq", years=[2020], shocks=shocks), data_source="toy"
    ).data
    io = run_scenario(
        Scenario(name="io", engine="io_price", years=[2020], shocks=shocks), data_source="toy"
    ).data

    def prices(df):
        p = df[df["variable"] == "price_change"]
        return p.set_index(["region", "sector"])["value"].sort_index()

    err = float((prices(pe) - prices(io)).abs().max())
    return err < 1e-12, f"max|PE price − Engine1 price| = {err:.2e}", err, 1e-12


@check(SUITE, "carries_elasticity_and_default_flag")
def _provenance():
    """Every good carries the elasticity used; the manifest flags how many used the default."""
    result = _run()
    df = result.data
    has_eps = (df["variable"] == "elasticity_used").sum() == 6  # 3 sectors × 2 regions
    has_flag = "n_goods_using_default_elasticity" in result.manifest.assumptions
    return has_eps and has_flag, f"elasticity rows & default-flag present = {has_eps and has_flag}"
