#!/usr/bin/env python
"""Extract the structural-trajectory source digest from the REAL published source files.

    python scripts/extract_structural_sources.py            # rebuild sources/inputs.json from raw/
    python scripts/extract_structural_sources.py --check     # non-zero if the digest is stale

Pipeline step 1a-DATA (see ``docs/structural-data-pipeline-plan.md``). Replaces the illustrative
headline figures in ``data/structural/sources/inputs.json`` with values EXTRACTED from the actual
source releases placed in ``data/structural/sources/raw/`` (see that README for the exact downloads
and filenames). Each source is parsed in its published format, mapped onto the model's
development/composition archetypes (N/S regions, BRD/MIL sectors) via the committed
``archetype_maps/``, and aggregated with the documented per-driver weights.

Because the raw files are large and separately licensed they are NOT committed; this script runs
only when they are present. The parsing logic is exercised on small byte-level FIXTURES in
``tests/test_structural_extractors.py`` (a 3-row PWT sheet, a toy WPP CSV), so extraction is tested
without shipping the multi-hundred-MB sources. Sources still missing from ``raw/`` keep their
current digest entry (the illustrative value) and are reported, so a partial extraction is explicit.

The functions here take an already-loaded DataFrame (tests feed a fixture frame directly); ``main``
wires them to the real files. All growth rates are DECIMAL FRACTIONS per year.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_SOURCES = _ROOT / "data" / "structural" / "sources"
_RAW = _SOURCES / "raw"
_MAPS = _SOURCES / "archetype_maps"
_DIGEST = _SOURCES / "inputs.json"


# --------------------------------------------------------------------------------------------------
# Archetype maps
# --------------------------------------------------------------------------------------------------
def _load_map(name: str, key: str) -> dict[str, str]:
    """Load an archetype map CSV (``# ``-commented) into ``{key_value: archetype}``."""
    df = pd.read_csv(_MAPS / name, comment="#")
    return {str(r[key]).strip(): str(r["archetype"]).strip() for _, r in df.iterrows()}


def country_archetype(iso3: str, cmap: dict[str, str]) -> str:
    """Region archetype for an ISO3 code; unlisted → S (conservative developing default)."""
    return cmap.get(iso3.strip().upper(), "S")


def industry_archetype(section: str, imap: dict[str, str]) -> str:
    """Sector archetype for an ISIC section letter; unlisted → MIL (services residual)."""
    return imap.get(section.strip().upper(), "MIL")


# --------------------------------------------------------------------------------------------------
# Aggregation helper
# --------------------------------------------------------------------------------------------------
def _weighted_mean(values: dict[str, float], weights: dict[str, float] | None) -> float:
    """Weighted mean of per-member ``values`` by ``weights`` (aligned on keys); unweighted mean when
    no usable weights. Members with no weight are dropped, not treated as zero."""
    if weights:
        num = sum(values[k] * weights[k] for k in values if weights.get(k, 0) > 0)
        den = sum(weights[k] for k in values if weights.get(k, 0) > 0)
        if den > 0:
            return num / den
    return sum(values.values()) / len(values) if values else 0.0


# --------------------------------------------------------------------------------------------------
# PWT 10.01 — aggregate TFP (rtfpna, an index → log-growth), per region archetype
# --------------------------------------------------------------------------------------------------
def extract_pwt_productivity(
    data: pd.DataFrame,
    cmap: dict[str, str],
    *,
    window: tuple[int, int] = (2010, 2019),
) -> dict[str, dict[int, float]]:
    """Per-archetype trend TFP growth from PWT's ``rtfpna`` index (cols ``countrycode``, ``year``,
    ``rtfpna``). For each country the trend growth over ``window`` is the annualised log-change of
    the index between the window endpoints; the archetype rate is the unweighted mean of members'
    trends. PWT 10.01 ends in 2019, so this is a HISTORICAL trend — the caller marks the forward
    knot as an assumption. Returns ``{archetype: {knot_year: rate}}`` keyed at the window's end."""
    lo, hi = window
    per_country: dict[str, float] = {}
    for code, g in data.dropna(subset=["rtfpna"]).groupby("countrycode"):
        gy = g.set_index("year")["rtfpna"]
        if lo in gy.index and hi in gy.index and gy[lo] > 0 and gy[hi] > 0:
            per_country[str(code)] = math.log(gy[hi] / gy[lo]) / (hi - lo)
    by_arch: dict[str, dict[str, float]] = {}
    for code, rate in per_country.items():
        by_arch.setdefault(country_archetype(code, cmap), {})[code] = rate
    # No GDP weights in PWT alone → unweighted archetype mean (documented confidence downgrade).
    out = {a: {hi: round(_weighted_mean(v, None), 4)} for a, v in by_arch.items()}
    if out:
        out["__all__"] = {hi: round(sum(r[hi] for r in out.values()) / len(out), 4)}
    return out


# --------------------------------------------------------------------------------------------------
# UN WPP 2024 — population growth per region archetype
# --------------------------------------------------------------------------------------------------
def extract_wpp_population(
    data: pd.DataFrame,
    cmap: dict[str, str],
    *,
    knots: tuple[int, ...] = (2025, 2035, 2050),
) -> dict[str, dict[int, float]]:
    """Per-archetype population growth from WPP totals (columns ``iso3``, ``year``, ``population``).
    For each knot year the growth rate is population[y+1]/population[y] − 1 per country, aggregated
    to the archetype weighted by that country's population (the per-driver weight for a
    population aggregate). Returns ``{archetype: {knot: rate}}``."""
    out: dict[str, dict[int, float]] = {}
    for y in knots:
        rates: dict[str, dict[str, float]] = {}
        pops: dict[str, dict[str, float]] = {}
        for code, g in data.groupby("iso3"):
            gy = g.set_index("year")["population"]
            if y in gy.index and (y + 1) in gy.index and gy[y] > 0:
                a = country_archetype(str(code), cmap)
                rates.setdefault(a, {})[str(code)] = gy[y + 1] / gy[y] - 1.0
                pops.setdefault(a, {})[str(code)] = float(gy[y])
        for a in rates:
            out.setdefault(a, {})[y] = round(_weighted_mean(rates[a], pops[a]), 4)
    if out:
        allk = {y: round(sum(out[a][y] for a in out if y in out[a]) / len(out), 4) for y in knots}
        out["__all__"] = allk
    return out


# --------------------------------------------------------------------------------------------------
# EU KLEMS 2023 — sector labour productivity, capital deepening, source labour share
# --------------------------------------------------------------------------------------------------
def extract_euklems(
    data: pd.DataFrame,
    imap: dict[str, str],
    *,
    knot: int = 2025,
) -> dict[str, dict]:
    """Per-sector-archetype series from EU KLEMS growth accounts. Expects columns ``isic_section``,
    ``lp_growth`` (labour productivity, output per hour), ``k_deepening`` (capital services per hour
    growth), ``labour_cost_share`` (the source-period labour cost share of value added), and
    ``va_weight`` (industry value added, for the output-weighted aggregate). Returns
    ``{"sector_productivity": {...}, "capital_deepening": {...}, "source_labour_shares": {...}}``,
    each ``{archetype: value_or_{knot: rate}}``, aggregated over each archetype's industries by
    value added."""
    lp: dict[str, dict[str, float]] = {}
    kd: dict[str, dict[str, float]] = {}
    sl: dict[str, dict[str, float]] = {}
    wt: dict[str, dict[str, float]] = {}
    for _, r in data.iterrows():
        a = industry_archetype(str(r["isic_section"]), imap)
        code = str(r["isic_section"])
        lp.setdefault(a, {})[code] = float(r["lp_growth"])
        kd.setdefault(a, {})[code] = float(r["k_deepening"])
        sl.setdefault(a, {})[code] = float(r["labour_cost_share"])
        wt.setdefault(a, {})[code] = float(r.get("va_weight", 1.0))

    def _agg(series: dict[str, dict[str, float]], rnd: int) -> dict[str, float]:
        return {a: round(_weighted_mean(series[a], wt[a]), rnd) for a in series}

    sp = {a: {knot: v} for a, v in _agg(lp, 4).items()}
    cd = {a: {knot: v} for a, v in _agg(kd, 4).items()}
    sls = _agg(sl, 3)
    for d, allrnd in ((sp, 4), (cd, 4)):
        if d:
            d["__all__"] = {knot: round(sum(x[knot] for x in d.values()) / len(d), allrnd)}
    if sls:
        sls["__all__"] = round(sum(sls.values()) / len(sls), 3)
    return {"sector_productivity": sp, "capital_deepening": cd, "source_labour_shares": sls}


# --------------------------------------------------------------------------------------------------
# NGFS — emissions intensity of GDP per sector archetype (pinned scenario tuple)
# --------------------------------------------------------------------------------------------------
def extract_ngfs_emissions(
    data: pd.DataFrame,
    *,
    model: str,
    scenario: str,
    knots: tuple[int, ...] = (2025, 2040),
) -> dict[str, dict[int, float]]:
    """Economy-wide emissions-intensity decarbonisation from an NGFS scenario-explorer export
    (long format: columns ``model``, ``scenario``, ``region``, ``variable``, ``year``, ``value``).
    Filters to the PINNED ``model``/``scenario`` + emissions-intensity variable, and returns the
    annualised decline over each knot interval as ``{"__all__": {knot: rate}}`` (negative =
    decarbonisation). A per-sector split needs a sectoral NGFS variable; the economy-wide path is
    the documented default until then."""
    sub = data[(data["model"] == model) & (data["scenario"] == scenario)]
    var = sub[sub["variable"].str.contains("Intensity", case=False, na=False)]
    if var.empty:
        var = sub[sub["variable"].str.contains("Emissions|CO2", case=False, na=False)]
    series = var.groupby("year")["value"].mean().sort_index()
    out: dict[int, float] = {}
    yrs = list(series.index)
    for y in knots:
        later = [yy for yy in yrs if yy > y]
        if y in series.index and later and series[y] > 0:
            y2 = later[0]
            out[y] = round((series[y2] / series[y]) ** (1.0 / (y2 - y)) - 1.0, 4)
    return {"__all__": out} if out else {}


# --------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------
def _rebuild_digest() -> tuple[dict, list[str]]:
    """Rebuild the digest from whatever raw files are present; return (digest, missing_sources)."""
    digest = json.loads(_DIGEST.read_text())
    missing: list[str] = []
    cmap = _load_map("country_to_archetype.csv", "iso3")
    imap = _load_map("industry_to_archetype.csv", "isic_section")

    pwt = _RAW / "pwt1001.xlsx"
    if pwt.exists():
        rates = extract_pwt_productivity(pd.read_excel(pwt, sheet_name="Data"), cmap)
        _apply_region(digest, "productivity", rates, source="PWT 10.01 rtfpna (extracted)")
    else:
        missing.append("pwt1001.xlsx (productivity)")

    wpp = _RAW / "wpp2024_population.csv"
    if wpp.exists():
        rates = extract_wpp_population(pd.read_csv(wpp), cmap)
        _apply_region(digest, "population", rates, source="UN WPP 2024 (extracted)")
    else:
        missing.append("wpp2024_population.csv (population)")

    klems = _RAW / "euklems_2023_growth_accounts.xlsx"
    if klems.exists():
        out = extract_euklems(pd.read_excel(klems), imap)
        _apply_sector(
            digest, "sector_productivity", out["sector_productivity"], "EU KLEMS 2023 (ext)"
        )
        _apply_sector(digest, "capital_deepening", out["capital_deepening"], "EU KLEMS 2023 (ext)")
        _apply_source_shares(digest, out["source_labour_shares"], "EU KLEMS 2023 (ext)")
    else:
        missing.append("euklems_2023_growth_accounts.xlsx (sector prod/deepening/labour share)")

    ngfs = _RAW / "ngfs_phase5.csv"
    if ngfs.exists():
        rates = extract_ngfs_emissions(
            pd.read_csv(ngfs), model="REMIND-MAgPIE 3.4-4.8", scenario="Net Zero 2050"
        )
        _apply_sector(
            digest, "emissions_intensity", {k: v for k, v in rates.items()}, "NGFS Phase 5 (ext)"
        )
    else:
        missing.append("ngfs_phase5.csv (emissions intensity)")

    return digest, missing


def _apply_region(digest: dict, driver: str, rates: dict, *, source: str) -> None:
    if not rates:
        return
    entry = digest["region_drivers"].setdefault(driver, {})
    for key, knots in rates.items():
        entry[key] = {
            "knots": {str(y): v for y, v in knots.items()},
            "confidence": "medium",
            "source": source,
        }


def _apply_sector(digest: dict, driver: str, rates: dict, source: str) -> None:
    if not rates:
        return
    entry = digest["sector_drivers"].setdefault(driver, {})
    for key, knots in rates.items():
        entry[key] = {
            "knots": {str(y): v for y, v in knots.items()},
            "confidence": "medium",
            "source": source,
        }


def _apply_source_shares(digest: dict, shares: dict, source: str) -> None:
    if not shares:
        return
    for key, val in shares.items():
        digest["source_labour_shares"][key] = {
            "value": val,
            "confidence": "medium",
            "source": source,
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="non-zero if the digest is stale vs raw/")
    args = ap.parse_args()
    if not _RAW.exists() or not any(_RAW.glob("*.xlsx")) and not any(_RAW.glob("*.csv")):
        print(
            f"No raw source files in {_RAW.relative_to(_ROOT)} — see its README for downloads. "
            "The committed digest keeps its current (illustrative) values."
        )
        return 0
    digest, missing = _rebuild_digest()
    if missing:
        print("Extracted from available raw files; STILL MISSING (kept illustrative):")
        for m in missing:
            print(f"  - {m}")
    rendered = json.dumps(digest, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if _DIGEST.read_text() != rendered:
            print(
                "inputs.json is OUT OF DATE vs the raw sources — re-run without --check + commit."
            )
            return 1
        print("inputs.json matches the extracted raw sources.")
        return 0
    _DIGEST.write_text(rendered)
    print(f"Wrote {_DIGEST}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
