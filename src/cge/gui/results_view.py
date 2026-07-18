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


def has_macro(result: ResultSet) -> bool:
    return (result.data["variable"] == "gdp_change").any()


def macro_gdp_table(result: ResultSet) -> pd.DataFrame:
    """Per region (central band): nominal & real GDP change and the deflator (inflation)."""
    df = result.data
    econ = df[(df["sector"] == _ECONOMY_SECTOR) & (df["scenario"] == "central")]
    wide = econ.pivot_table(index=["region", "year"], columns="variable", values="value")
    keep = [c for c in ("gdp_change", "gdp_change_real", "deflator") if c in wide.columns]
    wide = wide[keep].reset_index()
    return wide.rename(
        columns={
            "gdp_change": "GDP Δ (nominal)",
            "gdp_change_real": "GDP Δ (real)",
            "deflator": "deflator (inflation)",
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
