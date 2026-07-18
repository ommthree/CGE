"""Validation suite for Engine 2 (partial-equilibrium production-volume response).

Checks tied to docs/models/partial-equilibrium.md §7: production-volume sign, bounded finite-
change response, band ordering, zero-shock, price pass-through from Engine 1, Leontief
propagation (upstream output responds), and per-good elasticity provenance.
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
    """A carbon price raises prices; with ε ≤ 0 production volumes fall — central Δx/x ≤ 0."""
    df = _run().data
    vol = df[(df["variable"] == "volume_change") & (df["scenario"] == "central")]["value"]
    mx = float(vol.max())
    return mx <= 1e-9, f"max central Δx/x = {mx:.4f} (should be ≤ 0)", mx, 0.0


@check(SUITE, "volume_stays_above_minus_100pct")
def _bounded():
    """Finite-change form keeps production changes above −100% even at a large price (a units
    /linearisation bug produced impossible <−100% on live data — review)."""
    df = _run(price=500.0).data
    vol = df[df["variable"] == "volume_change"]["value"]
    mn = float(vol.min())
    return mn > -1.0, f"min Δx/x at €500/t = {mn:.4f} (must be > −1)", mn, -1.0


@check(SUITE, "leontief_propagation")
def _propagation():
    """Production propagates through the supply chain: aggregate output change should differ
    from the raw demand change (i.e. (I−A)⁻¹ actually did something), and volume ≠ final-demand
    for at least one good in a connected economy."""
    df = _run().data
    c = df[df["scenario"] == "central"]
    vol = c[c["variable"] == "volume_change"].set_index(["region", "sector"])["value"]
    dem = c[c["variable"] == "final_demand_change"].set_index(["region", "sector"])["value"]
    max_gap = float((vol - dem).abs().max())
    return max_gap > 1e-9, f"max|Δx/x − Δy/y| = {max_gap:.4f} (>0 ⇒ propagation occurred)", max_gap


@check(SUITE, "band_ordering")
def _bands():
    """Low band (most elastic) gives the largest volume fall; high the smallest — per good."""
    df = _run().data
    vol = df[df["variable"] == "volume_change"]
    ok = True
    for _key, g in vol.groupby(["region", "sector"]):
        by = {row.scenario: row.value for row in g.itertuples()}
        if not (by["low"] <= by["central"] <= by["high"] <= 1e-9):
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


@check(SUITE, "carries_per_good_elasticity_provenance")
def _provenance():
    """Every good carries the elasticity used; the manifest carries per-good source/confidence/
    default status and a content hash of the elasticity values (review: manifest must be
    reproducible and per-parameter sourced)."""
    result = _run()
    df = result.data
    a = result.manifest.assumptions
    has_eps = (df["variable"] == "elasticity_used").sum() == 6  # 3 sectors × 2 regions
    has_per_good = isinstance(a.get("elasticity_per_good"), dict) and bool(a["elasticity_per_good"])
    has_hash = bool(a.get("elasticity_set", {}).get("content_hash"))
    ok = has_eps and has_per_good and has_hash
    return ok, f"elasticity rows + per-good provenance + content hash present = {ok}"
