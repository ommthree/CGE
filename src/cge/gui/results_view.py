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
    vals = result.data.loc[result.data["variable"] == variable, "value"]
    if vals.empty:
        return {}
    return {
        "goods": int(vals.shape[0]),
        "mean": float(vals.mean()),
        "max": float(vals.max()),
        "min": float(vals.min()),
    }
