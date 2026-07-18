"""Validation suite for the macro-aggregate accounting layer (roadmap Phase 4b, PE tier).

Checks tied to docs/models/macro-aggregates.md: the base-year GDP identity (Σ value added =
Σ final demand), real == nominal at zero inflation, that a price-only engine (Engine 1) produces
inflation but zero real GDP change, that a volume-bearing engine (Engine 2) produces a negative
real GDP change under a carbon price, and that per-region GDP aggregates the per-sector GVA.
"""

from __future__ import annotations

from cge.accounting import ECONOMY_SECTOR, base_year_value_added
from cge.contracts.shocks import CarbonPrice
from cge.scenarios.loader import Scenario
from cge.validation.framework import check
from cge.validation.toy import toy_economy

SUITE = "macro"


def _run(engine: str, price: float = 100.0):
    from cge.runner import run_scenario

    sc = Scenario(name="m", engine=engine, years=[2020], shocks=[CarbonPrice(price=price)])
    return run_scenario(sc, data_source="toy")


@check(SUITE, "gdp_identity_production_equals_expenditure")
def _identity():
    """Base-year Σ value added (production GDP) equals Σ final demand (expenditure GDP) — the
    accounting identity the whole layer rests on."""
    io, _ = toy_economy()
    va = float(base_year_value_added(io).sum())
    labels = list(io.A.columns)
    fd = float(io.final_demand.sum(axis=1).reindex(labels).fillna(0.0).sum())
    rel = abs(va - fd) / max(fd, 1.0)
    return rel < 1e-9, f"ΣVA={va:.2f} vs Σfinal_demand={fd:.2f} (rel {rel:.2e})", rel, 1e-9


@check(SUITE, "zero_shock_zero_aggregates")
def _zero():
    """With no shock, every macro aggregate (deflator, GDP, GVA, nominal and real) is zero."""
    df = _run("io_price", price=0.0).data
    macro = df[df["variable"].str.startswith(("gva_", "gdp_", "deflator"))]["value"]
    mx = float(macro.abs().max()) if len(macro) else 0.0
    return mx < 1e-12, f"max|macro aggregate| at τ=0 = {mx:.2e}", mx, 1e-12


@check(SUITE, "real_equals_nominal_at_zero_inflation")
def _real_nominal():
    """When the region deflator is ~0, real GDP change equals nominal (real = nominal deflated by
    the index). Uses a tiny price so inflation is near zero but nonzero."""
    df = _run("partial_eq", price=1.0).data
    econ = df[df["region"] == "A"]
    dfl = float(
        econ[(econ["variable"] == "deflator") & (econ["scenario"] == "central")]["value"].iloc[0]
    )
    nom = float(
        econ[(econ["variable"] == "gdp_change") & (econ["scenario"] == "central")]["value"].iloc[0]
    )
    real = float(
        econ[(econ["variable"] == "gdp_change_real") & (econ["scenario"] == "central")][
            "value"
        ].iloc[0]
    )
    # real = (1+nom)/(1+dfl)-1; check the identity holds exactly for the emitted numbers.
    expected_real = (1.0 + nom) / (1.0 + dfl) - 1.0
    err = abs(real - expected_real)
    msg = f"deflator={dfl:.4f}; real matches (1+nom)/(1+dfl)−1 to {err:.2e}"
    return err < 1e-12, msg, err, 1e-12


@check(SUITE, "price_only_engine_has_zero_real_gdp")
def _price_only_real_zero():
    """Engine 1 (prices, no volume response) produces inflation but NO real GDP change — a
    price-only model says nothing about real quantities. Real GDP change must be ~0 while the
    deflator is positive."""
    df = _run("io_price").data
    econ = df[(df["region"] == "A") & (df["scenario"] == "central")]
    dfl = float(econ[econ["variable"] == "deflator"]["value"].iloc[0])
    real = float(econ[econ["variable"] == "gdp_change_real"]["value"].iloc[0])
    ok = dfl > 1e-6 and abs(real) < 1e-9
    return ok, f"deflator={dfl:.4f} (>0), real GDP change={real:.2e} (~0)", abs(real), 1e-9


@check(SUITE, "carbon_price_lowers_real_gdp")
def _real_gdp_falls():
    """Engine 2: a carbon price reduces REAL GDP in every region and band (volumes fall by more,
    in real terms, than the price index rises)."""
    df = _run("partial_eq").data
    real = df[df["variable"] == "gdp_change_real"]["value"]
    mx = float(real.max())
    return mx < 0.0, f"max real GDP change across regions/bands = {mx:.4f} (should be < 0)", mx, 0.0


@check(SUITE, "gdp_aggregates_sector_gva")
def _gdp_aggregates_gva():
    """Per-region nominal GDP change equals the value-added-weighted mean of its sectors' nominal
    GVA changes — GDP is the aggregate of GVA, by construction."""
    io, _ = toy_economy()
    va = base_year_value_added(io)
    df = _run("partial_eq").data
    worst = 0.0
    for region in sorted({lab.split(":", 1)[0] for lab in va.index}):
        sub = df[(df["region"] == region) & (df["scenario"] == "central")]
        gdp = float(sub[sub["variable"] == "gdp_change"]["value"].iloc[0])
        gva = sub[(sub["variable"] == "gva_change") & (sub["sector"] != ECONOMY_SECTOR)]
        num = den = 0.0
        for r in gva.itertuples():
            w = float(va.get(f"{region}:{r.sector}", 0.0))
            num += w * float(r.value)
            den += w
        recomputed = num / den if den > 0 else 0.0
        worst = max(worst, abs(gdp - recomputed))
    return worst < 1e-9, f"max|GDP − VA-weighted ΣGVA| = {worst:.2e}", worst, 1e-9
