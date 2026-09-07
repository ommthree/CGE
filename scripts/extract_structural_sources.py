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
# Raw-layout normalisers (review 2026-09-06 — accept the ACTUAL published downloads, not just the
# already-tidy fixture columns). Each maps the source's real column names / long-vs-wide shape /
# aggregate rows onto the tidy columns the extractor expects, so the user drops in the ORIGINAL
# file. A tidy-shape frame passes through unchanged (idempotent) — what the fixtures feed.
# --------------------------------------------------------------------------------------------------
def _rename_first_present(df: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    """Return ``df`` with columns renamed to each canonical name using the first matching alias
    present (case-insensitive). Canonical columns already present are left as-is."""
    lower = {c.lower(): c for c in df.columns}
    rename: dict[str, str] = {}
    for canonical, names in aliases.items():
        if canonical in df.columns:
            continue
        for a in names:
            if a.lower() in lower:
                rename[lower[a.lower()]] = canonical
                break
    return df.rename(columns=rename)


def normalise_wpp(df: pd.DataFrame) -> pd.DataFrame:
    """WPP 2024 "Total Population" export → tidy ``iso3``/``year``/``population``. The raw CSV has
    ``ISO3_code`` / ``Time`` / ``PopTotal``, carries a ``Variant`` column (keep ``Medium``), and
    mixes country + region rows (kept only when ``ISO3_code`` is a real 3-letter code)."""
    df = _rename_first_present(
        df, {"iso3": ["ISO3_code", "iso3_code"], "year": ["Time"], "population": ["PopTotal"]}
    )
    if "Variant" in df.columns:
        df = df[df["Variant"].astype(str).str.strip().str.lower() == "medium"]
    df = df[df["iso3"].notna()]
    df = df[df["iso3"].astype(str).str.fullmatch(r"[A-Za-z]{3}")]
    df = df.copy()
    df["year"] = df["year"].astype(int)
    return df[["iso3", "year", "population"]]


def normalise_euklems(
    df: pd.DataFrame,
    *,
    country: str | None = None,
    var_codes: dict[str, str] | None = None,
) -> pd.DataFrame:
    """EU KLEMS 2023 growth-accounts long export → tidy per-ISIC-section rows with ``isic_section``,
    ``lp_growth``, ``k_deepening``, ``labour_cost_share``, ``va_weight``. The raw file is long
    (``geo_code`` / ``nace_r2_code`` / ``var`` / ``year`` / ``value``, latest year); this picks one
    ``country`` (default: the first present), pivots the growth-account variables named in
    ``var_codes`` (defaults to the EU KLEMS statistical codes), maps the NACE industry code to its
    ISIC section letter, and averages within a section (VA-weighted if a VA variable is present).

    Because EU KLEMS variable codes vary by release, ``var_codes`` is overridable; if the expected
    variables are absent this raises with the codes it DID find, so the mismatch is explicit rather
    than silently producing empties. A frame already in the tidy fixture shape passes through."""
    if "isic_section" in df.columns and "lp_growth" in df.columns:
        return df  # already tidy (the fixture / a hand-prepared file)
    df = _rename_first_present(
        df,
        {
            "geo_code": ["geo_code", "geo", "country", "country_code"],
            "nace": ["nace_r2_code", "nace_r2", "nace", "industry", "code"],
            "var": ["var", "variable", "measure", "var_code"],
            "year": ["year", "time"],
            "value": ["value", "obs_value", "val"],
        },
    )
    codes = var_codes or {
        # EU KLEMS 2023 growth-accounts statistical variables (overridable per release).
        "lp_growth": "VA_QI_growth",  # value-added-volume-per-hour growth
        "k_deepening": "CAP_QI_growth",  # capital-services-per-hour growth
        "labour_cost_share": "LAB_share",  # labour compensation share of value added
        "va_weight": "VA_CP",  # value added at current prices (aggregation weight)
    }
    if country is None and "geo_code" in df.columns:
        country = str(df["geo_code"].iloc[0])
    if "geo_code" in df.columns:
        df = df[df["geo_code"].astype(str) == str(country)]
    if "year" in df.columns:  # keep the latest year of each (industry, variable)
        df = df.sort_values("year").groupby(["nace", "var"], as_index=False).last()
    present = set(df["var"].astype(str).unique())
    missing = [c for c in codes.values() if c not in present]
    if missing:
        raise ValueError(
            f"EU KLEMS export is missing expected variable code(s) {missing}; present codes are "
            f"{sorted(present)}. Pass var_codes={{...}} matching this release (see the README)."
        )
    wide = df.pivot_table(index="nace", columns="var", values="value", aggfunc="last")
    rows = []
    for nace, r in wide.iterrows():
        section = str(nace).strip()[0].upper() if str(nace).strip() else ""
        rows.append(
            {
                "isic_section": section,
                "lp_growth": r.get(codes["lp_growth"]),
                "k_deepening": r.get(codes["k_deepening"]),
                "labour_cost_share": r.get(codes["labour_cost_share"]),
                "va_weight": r.get(codes["va_weight"], 1.0),
            }
        )
    out = pd.DataFrame(rows).dropna(subset=["lp_growth", "k_deepening", "labour_cost_share"])
    # Collapse detailed industries that share a section into one VA-weighted row.
    agg = []
    for section, g in out.groupby("isic_section"):
        w = g["va_weight"].fillna(1.0)
        wsum = w.sum() or 1.0
        agg.append(
            {
                "isic_section": section,
                "lp_growth": float((g["lp_growth"] * w).sum() / wsum),
                "k_deepening": float((g["k_deepening"] * w).sum() / wsum),
                "labour_cost_share": float((g["labour_cost_share"] * w).sum() / wsum),
                "va_weight": float(wsum),
            }
        )
    return pd.DataFrame(agg)


def normalise_ngfs(df: pd.DataFrame) -> pd.DataFrame:
    """NGFS / IIASA scenario-explorer export → tidy long ``model``, ``scenario``, ``region``,
    ``variable``, ``year``, ``value``. Accepts the IIASA IAMC WIDE format (a column per year) by
    melting it, and normalises the capitalised column names. A long frame passes through."""
    df = _rename_first_present(
        df,
        {
            "model": ["Model"],
            "scenario": ["Scenario"],
            "region": ["Region"],
            "variable": ["Variable"],
            "value": ["Value"],
            "year": ["Year"],
        },
    )
    if "year" in df.columns and "value" in df.columns:
        return df  # already long
    id_cols = [c for c in ("model", "scenario", "region", "variable", "unit", "Unit") if c in df]
    year_cols = [c for c in df.columns if str(c).isdigit()]
    if not year_cols:
        return df  # nothing to melt; let the extractor validate downstream
    melted = df.melt(id_vars=id_cols, value_vars=year_cols, var_name="year", value_name="value")
    melted["year"] = melted["year"].astype(int)
    return melted


def normalise_ilostat(df: pd.DataFrame) -> pd.DataFrame:
    """ILOSTAT labour-force-participation-rate bulk CSV → tidy ``iso3``, ``year``, ``lfpr``. The raw
    file (indicator ``EAP_DWAP_SEX_AGE_RT``) uses ``ref_area`` / ``time`` / ``obs_value`` with a
    ``sex`` breakdown (keep the total ``SEX_T``) and a ``classif1`` age band. The headline
    working-age total is the **15+** aggregate band ``AGE_AGGREGATE_YGE15`` (ILOSTAT does NOT emit a
    ``_TOTAL`` age band — the age dimension is always a specific band), so we select exactly that
    band, falling back to ``AGE_YTHADULT_YGE15`` if the AGGREGATE variant is absent. Selecting ONE
    band avoids double-counting a country-year across overlapping bands (review 2026-09-06)."""
    df = _rename_first_present(
        df, {"iso3": ["ref_area", "iso3"], "year": ["time", "year"], "lfpr": ["obs_value", "lfpr"]}
    )
    if "sex" in df.columns:
        df = df[df["sex"].astype(str).str.upper() == "SEX_T"]
    if "classif1" in df.columns:
        c = df["classif1"].astype(str).str.upper()
        band = "AGE_AGGREGATE_YGE15" if (c == "AGE_AGGREGATE_YGE15").any() else "AGE_YTHADULT_YGE15"
        df = df[c == band]
    df = df[df["iso3"].astype(str).str.fullmatch(r"[A-Za-z]{3}")].copy()
    df["year"] = df["year"].astype(int)
    # One row per (country, year): if bands/sources still overlap, keep the last (latest source).
    return df[["iso3", "year", "lfpr"]].drop_duplicates(["iso3", "year"], keep="last")


# --------------------------------------------------------------------------------------------------
# PWT 10.01 — aggregate TFP (rtfpna, an index → log-growth), per region archetype
# --------------------------------------------------------------------------------------------------
def extract_pwt_productivity(
    data: pd.DataFrame,
    cmap: dict[str, str],
    *,
    window: tuple[int, int] = (2000, 2019),
) -> dict[str, dict[int, float]]:
    """Per-archetype trend TFP growth from PWT's ``rtfpna`` index (cols ``countrycode``, ``year``,
    ``rtfpna``, ``rgdpo``). For each country the trend growth over ``window`` is the annualised
    log-change of the index between the window endpoints; the archetype rate is the **GDP-weighted**
    mean of its members' trends (weight = the country's ``rgdpo`` at the window's end), so a large
    economy dominates and dozens of small volatile economies do not (review 2026-09-06). Countries
    NOT in the archetype map are EXCLUDED (not defaulted to S) so an unmapped economy cannot skew a
    bucket. PWT 10.01 ends in 2019, so this is a HISTORICAL trend — the caller marks the forward
    knot as an assumption. The default window is **2000–2019** (not a single decade): the 2010–2019
    window alone captures the post-GFC emerging-market slowdown and shows advanced>emerging TFP,
    whereas every window back to ~1990 shows the expected emerging>advanced convergence; 2000–2019
    spans full cycles and is the defensible default (review 2026-09-07). Returns
    ``{archetype: {knot_year: rate}}`` keyed at the window's end."""
    lo, hi = window
    per_country: dict[str, float] = {}
    gdp: dict[str, float] = {}
    has_gdp = "rgdpo" in data.columns
    for code, g in data.dropna(subset=["rtfpna"]).groupby("countrycode"):
        code = str(code)
        if code not in cmap:  # only aggregate mapped countries (no silent default-to-S, review P)
            continue
        gy = g.set_index("year")["rtfpna"]
        if lo in gy.index and hi in gy.index and gy[lo] > 0 and gy[hi] > 0:
            per_country[code] = math.log(gy[hi] / gy[lo]) / (hi - lo)
            if has_gdp:
                gr = g.set_index("year")["rgdpo"]
                if hi in gr.index and gr[hi] > 0:
                    gdp[code] = float(gr[hi])
    by_arch: dict[str, dict[str, float]] = {}
    by_gdp: dict[str, dict[str, float]] = {}
    for code, rate in per_country.items():
        a = country_archetype(code, cmap)
        by_arch.setdefault(a, {})[code] = rate
        by_gdp.setdefault(a, {})[code] = gdp.get(code, 0.0)
    # GDP-weighted archetype mean (falls back to unweighted in _weighted_mean if no GDP present).
    out = {a: {hi: round(_weighted_mean(v, by_gdp.get(a)), 4)} for a, v in by_arch.items()}
    if out:
        # World aggregate: GDP-weighted across ALL mapped countries, not a mean of the two buckets.
        allv = {c: r for a in by_arch for c, r in by_arch[a].items()}
        allw = {c: w for a in by_gdp for c, w in by_gdp[a].items()}
        out["__all__"] = {hi: round(_weighted_mean(allv, allw), 4)}
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
    population aggregate). Returns ``{archetype: {knot: rate}}``. Accepts the raw WPP export
    (``ISO3_code``/``Time``/``PopTotal`` + a Medium-variant filter) via :func:`normalise_wpp`."""
    data = normalise_wpp(data)
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
    value added. Accepts the raw EU KLEMS long export via :func:`normalise_euklems`."""
    data = normalise_euklems(data)
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
    the documented default until then. Accepts the IIASA IAMC WIDE (year-columns) export via
    :func:`normalise_ngfs`."""
    data = normalise_ngfs(data)
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
# ILOSTAT — labour-force participation growth per region archetype
# --------------------------------------------------------------------------------------------------
def extract_ilo_participation(
    data: pd.DataFrame,
    cmap: dict[str, str],
    *,
    knots: tuple[int, ...] = (2025, 2040),
    trend_window: int = 10,
) -> dict[str, dict[int, float]]:
    """Per-archetype labour-force-participation growth from ILOSTAT LFPR. The driver is a
    *proportional* annual growth rate of the participation RATE. ILOSTAT is historical (ends well
    before the knots), so rather than freeze a single year-on-year change we compute each country's
    **annualised trend over the most recent ``trend_window`` years available** — the log-change of
    the rate between the first and last year in the window, per year — far more stable than
    one noisy year-to-year step (review 2026-09-06). That single recent trend is held FORWARD at
    every knot (a documented no-further-information assumption). Countries are aggregated to their
    archetype weighted by the participation rate (a labour-force proxy; population weights aren't in
    this file), and only MAPPED countries are included (no silent default-to-S). Accepts the raw
    ILOSTAT CSV via ``normalise_ilostat``. Returns ``{archetype: {knot: rate}}``."""
    data = normalise_ilostat(data)
    trends: dict[str, dict[str, float]] = {}
    wts: dict[str, dict[str, float]] = {}
    for code, g in data.groupby("iso3"):
        code = str(code)
        if code not in cmap:  # mapped countries only (no silent default-to-S)
            continue
        gy = g.set_index("year")["lfpr"].sort_index()
        gy = gy[gy > 0]
        if len(gy) < 2:
            continue
        last = int(gy.index.max())
        recent = gy[gy.index >= last - trend_window]
        if len(recent) < 2:
            recent = gy
        y0, y1 = int(recent.index.min()), int(recent.index.max())
        trend = math.log(recent[y1] / recent[y0]) / (y1 - y0)
        a = country_archetype(code, cmap)
        trends.setdefault(a, {})[code] = trend
        wts.setdefault(a, {})[code] = float(recent[y1])  # participation-rate weight
    per_arch = {a: round(_weighted_mean(trends[a], wts[a]), 4) for a in trends}
    allv = {c: r for a in trends for c, r in trends[a].items()}
    allw = {c: w for a in wts for c, w in wts[a].items()}
    world = round(_weighted_mean(allv, allw), 4) if allv else None
    out: dict[str, dict[int, float]] = {}
    for a, rate in per_arch.items():
        out[a] = {y: rate for y in knots}  # single recent trend held forward at every knot
    if world is not None:
        out["__all__"] = {y: world for y in knots}
    return out


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
        # low_memory=False: the WPP CSV mixes numeric country rows with blank-field region rows, so
        # a chunked read infers mixed dtypes and warns; read it whole (it is ~85MB, fits in memory).
        rates = extract_wpp_population(pd.read_csv(wpp, low_memory=False), cmap)
        _apply_region(digest, "population", rates, source="UN WPP 2024 (extracted)")
    else:
        missing.append("wpp2024_population.csv (population)")

    ilo = _RAW / "ilostat_lfpr.csv"
    if ilo.exists():
        rates = extract_ilo_participation(pd.read_csv(ilo), cmap)
        _apply_region(digest, "labour_participation", rates, source="ILOSTAT LFPR (extracted)")
    else:
        missing.append("ilostat_lfpr.csv (labour participation)")

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
