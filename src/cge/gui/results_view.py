"""Shape a ResultSet into the tables/series the results page renders.

Pure pandas — no Streamlit — so the reshaping logic is unit-tested independently of the UI.
Engine 1 emits a main ``price_change`` variable plus ``price_change_<part>`` decomposition
rows; these helpers pivot those into per-good headline tables and waterfall series.
"""

from __future__ import annotations

import pandas as pd

from cge.contracts.results import ResultSet

# Decomposition parts emitted by io_price, in supply-chain order.
_DECOMP_ORDER = [
    "direct",
    "upstream_tier_1",
    "upstream_tier_2",
    "upstream_tier_3",
    "upstream_residual",
]


def headline_table(result: ResultSet, *, variable: str = "price_change") -> pd.DataFrame:
    """One row per good: the headline variable, sorted by impact (largest first)."""
    df = result.data
    out = (
        df[df["variable"] == variable]
        .loc[:, ["region", "sector", "year", "scenario", "value"]]
        .sort_values("value", ascending=False)
        .reset_index(drop=True)
    )
    return out


def available_variables(result: ResultSet) -> list[str]:
    return sorted(result.data["variable"].unique())


def waterfall(
    result: ResultSet, *, region: str, sector: str, year: int | None = None
) -> pd.DataFrame:
    """Decomposition of one good's price change into direct + upstream tiers + residual.

    Returns a tidy frame (part, value) in supply-chain order for a waterfall/bar chart.
    """
    df = result.data
    sel = (df["region"] == region) & (df["sector"] == sector)
    if year is not None:
        sel &= df["year"] == year
    rows = []
    for part in _DECOMP_ORDER:
        var = f"price_change_{part}"
        vals = df.loc[sel & (df["variable"] == var), "value"]
        if not vals.empty:
            rows.append({"part": part, "value": float(vals.mean())})
    return pd.DataFrame(rows)


def goods_with_decomposition(result: ResultSet) -> list[tuple[str, str]]:
    """(region, sector) pairs that have decomposition rows — the pickers for the waterfall."""
    df = result.data
    dec = df[df["variable"].str.startswith("price_change_")]
    pairs = dec[["region", "sector"]].drop_duplicates()
    return sorted((r.region, r.sector) for r in pairs.itertuples())


def summary_stats(result: ResultSet, *, variable: str = "price_change") -> dict[str, float]:
    # For banded variables (volume_change), summarise the central band only.
    df = result.data
    sel = df["variable"] == variable
    if "scenario" in df and (df.loc[sel, "scenario"] == "central").any():
        sel &= df["scenario"] == "central"
    vals = df.loc[sel, "value"]
    if vals.empty:
        return {}
    return {
        "goods": int(vals.shape[0]),
        "mean": float(vals.mean()),
        "max": float(vals.max()),
        "min": float(vals.min()),
    }


def has_volume(result: ResultSet) -> bool:
    return (result.data["variable"] == "volume_change").any()


def volume_envelope(result: ResultSet) -> pd.DataFrame:
    """Per good: low/central/high volume change (the uncertainty band), sorted by central."""
    df = result.data[result.data["variable"] == "volume_change"]
    wide = df.pivot_table(
        index=["region", "sector", "year"], columns="scenario", values="value"
    ).reset_index()
    cols = ["region", "sector", "year"] + [
        b for b in ("low", "central", "high") if b in wide.columns
    ]
    wide = wide[cols]
    if "central" in wide.columns:
        wide = wide.sort_values("central")  # biggest fall (most negative) first
    return wide.reset_index(drop=True)


# Economy-wide, region-level rows (GDP, deflator) carry this sentinel sector.
_ECONOMY_SECTOR = "__economy__"


_MACRO_GDP_VARS = ("gdp_change", "gdp_change_real", "gdp_change_nominal_wage", "deflator")


def has_macro(result: ResultSet) -> bool:
    # Either the PE-tier roll-up (gdp_change), the closed/open CGE's native real GDP
    # (gdp_change_real), or the multi-region CGE's consumption index (real_consumption_change —
    # review P2: this was previously missing, so the multi-region macro section never rendered).
    return (
        result.data["variable"]
        .isin(("gdp_change", "gdp_change_real", "real_consumption_change"))
        .any()
    )


def macro_gdp_table(result: ResultSet) -> pd.DataFrame:
    """Per region (central band): the GDP columns that are present. The PE tier reports nominal +
    real + a deflator; the closed/open CGE reports real (CPI-numéraire) + a wage-numéraire nominal
    reference and NO deflator (the CPI is its numéraire); the multi-region CGE instead reports
    ``real_consumption_change`` — a base-price household-consumption index, NOT production-side
    GDP (review P2: exposed here so it is visible on Results, not only in a download)."""
    df = result.data
    econ = df[(df["sector"] == _ECONOMY_SECTOR) & (df["scenario"] == "central")]
    wide = econ.pivot_table(index=["region", "year"], columns="variable", values="value")
    keep = [c for c in (*_MACRO_GDP_VARS, "real_consumption_change") if c in wide.columns]
    wide = wide[keep].reset_index()
    return wide.rename(
        columns={
            "gdp_change": "GDP Δ (nominal)",
            "gdp_change_real": "GDP Δ (real)",
            "gdp_change_nominal_wage": "GDP Δ (nominal, wage-numéraire)",
            "deflator": "deflator (inflation)",
            "real_consumption_change": "Real consumption Δ (multi-region; NOT GDP)",
        }
    )


# -- Trade, factor prices, welfare & carbon revenue (open/multi-region CGE) --------------------
_TRADE_VARS = ("import_change", "export_change", "exchange_rate_change")
_WELFARE_VARS = ("welfare_change", "carbon_revenue")


def has_trade(result: ResultSet) -> bool:
    return result.data["variable"].isin(_TRADE_VARS).any()


def trade_table(result: ResultSet) -> pd.DataFrame:
    """Per sector×region: import/export volume change (GE outputs — single value, not a banded
    envelope like Engine 2's volume_change)."""
    df = result.data
    sel = df["variable"].isin(("import_change", "export_change"))
    wide = df[sel].pivot_table(
        index=["region", "sector", "year"], columns="variable", values="value"
    )
    keep = [c for c in ("import_change", "export_change") if c in wide.columns]
    wide = wide[keep].reset_index()
    return wide.rename(columns={"import_change": "Imports Δ", "export_change": "Exports Δ"})


def exchange_rate_table(result: ResultSet) -> pd.DataFrame | None:
    """The economy-wide exchange-rate change, or ``None`` if the run has no such variable (the
    multi-region model has no exchange rate — trade is entirely among the build's own regions)."""
    df = result.data
    sel = df["variable"] == "exchange_rate_change"
    if not sel.any():
        return None
    out = df.loc[sel, ["region", "year", "value"]].rename(columns={"value": "Exchange rate Δ"})
    return out.reset_index(drop=True)


def has_factor_prices(result: ResultSet) -> bool:
    return (result.data["variable"] == "factor_price_change").any()


def factor_price_table(result: ResultSet) -> pd.DataFrame:
    """Per factor×region: the factor-price (wage / capital-rental-rate) change."""
    df = result.data[result.data["variable"] == "factor_price_change"]
    cols = [c for c in ("region", "sector", "year", "value") if c in df.columns]
    out = df.loc[:, cols].rename(columns={"sector": "factor", "value": "Factor price Δ"})
    return out.sort_values("Factor price Δ").reset_index(drop=True)


def has_welfare(result: ResultSet) -> bool:
    return result.data["variable"].isin(_WELFARE_VARS).any()


def welfare_table(result: ResultSet) -> pd.DataFrame:
    """Per region: welfare change and carbon revenue (share of THAT REGION's own benchmark GDP —
    review P2 round 10: the multi-region emitter previously divided by global benchmark GDP,
    understating the share for any region smaller than the whole economy) — the GE-specific
    outputs Engines 1-2 cannot produce (review P2: previously only reachable via download)."""
    df = result.data
    sel = df["variable"].isin(_WELFARE_VARS) & (df["sector"] == _ECONOMY_SECTOR)
    wide = df[sel].pivot_table(index=["region", "year"], columns="variable", values="value")
    keep = [c for c in _WELFARE_VARS if c in wide.columns]
    wide = wide[keep].reset_index()
    return wide.rename(
        columns={
            "welfare_change": "Welfare Δ",
            "carbon_revenue": "Carbon revenue (share of own region's GDP)",
        }
    )


def macro_gva_table(result: ResultSet) -> pd.DataFrame:
    """Per sector×region (central band): nominal & real GVA change, sorted by real (worst first)."""
    df = result.data
    sel = (
        df["variable"].isin(["gva_change", "gva_change_real"])
        & (df["sector"] != _ECONOMY_SECTOR)
        & (df["scenario"] == "central")
    )
    wide = df[sel].pivot_table(
        index=["region", "sector", "year"], columns="variable", values="value"
    )
    keep = [c for c in ("gva_change", "gva_change_real") if c in wide.columns]
    wide = wide[keep].reset_index()
    wide = wide.rename(columns={"gva_change": "GVA Δ (nominal)", "gva_change_real": "GVA Δ (real)"})
    if "GVA Δ (real)" in wide.columns:
        wide = wide.sort_values("GVA Δ (real)")
    return wide.reset_index(drop=True)
