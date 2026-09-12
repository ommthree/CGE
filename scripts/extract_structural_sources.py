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
current digest entry and are reported, so a partial extraction is explicit rather than silent.

The functions here take an already-loaded DataFrame (tests feed a fixture frame directly); ``main``
wires them to the real files. All growth rates are DECIMAL FRACTIONS per year.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import re
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_SOURCES = _ROOT / "data" / "structural" / "sources"
_RAW = _SOURCES / "raw"
_MAPS = _SOURCES / "archetype_maps"
_MANIFEST = _SOURCES / "raw_manifest.json"
_DIGEST = _SOURCES / "inputs.json"

# EXPLICIT NGFS selection tuple (review P1/P2 2026-09-09): scenario choice is a visible, validated
# configuration — not a value buried in the orchestration. The supplied export is REMIND-MAgPIE
# 3.3-4.8; the acquisition docs asked for Net Zero 2050, but the modelled baseline was chosen as
# "Below 2°C" (a less-abrupt transition) — that CHOICE is recorded here and the docs match it.
# Emissions intensity is derived as co2_var / gdp_var (the export has no ready-made intensity var).
# To change the scenario, edit this tuple (and the docs); extract_ngfs_emissions VALIDATES that each
# field resolves to exactly one series in the export.
_NGFS = {
    "model": "REMIND-MAgPIE 3.3-4.8",
    "scenario": "Below 2°C",
    "region": "World",
    "co2_var": "Emissions|CO2",
    "gdp_var": "GDP|PPP|Counterfactual without damage",
    # Per-sector emissions split (review-9 1d, 2026-09-16): each archetype's CO2 is the SUM of these
    # component variables, over the shared economy-wide GDP denominator (NGFS has no sectoral GDP).
    # BRD (goods) = industry energy demand + industrial processes + energy supply; MIL (services) =
    # transport + residential/commercial. AFOLU is EXCLUDED (land use, not production; net-negative
    # mid-century so it cannot be a multiplicative scale). These need the sectoral CSV vendored by
    # scripts/fetch_ngfs_sectoral.py; the economy-wide __all__ path works without them.
    "sector_bundles": {
        "BRD": [
            "Emissions|CO2|Energy|Demand|Industry",
            "Emissions|CO2|Industrial Processes",
            "Emissions|CO2|Energy|Supply",
        ],
        "MIL": [
            "Emissions|CO2|Energy|Demand|Transportation",
            "Emissions|CO2|Energy|Demand|Residential and Commercial",
        ],
    },
}


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
    """LEGACY long-format EU KLEMS normaliser → tidy per-ISIC-section rows with ``isic_section``,
    ``lp_growth``, ``k_deepening``, ``labour_cost_share``, ``va_weight``.

    NOTE (review P3 2026-09-15): the CURRENT MFP pipeline does NOT use this route — it reads the
    multi-sheet workbook via :func:`load_euklems_workbook` (which emits the LP1ConTFP-derived
    ``mfp``), and :func:`extract_euklems` REJECTS a frame that lacks ``mfp``. This function is
    retained only for the long-format ``VA_QI_growth``/``CAP_QI_growth`` export shape (and its own
    unit tests); its ``k_deepening`` output is not consumed. It is kept so a future long-format
    release can be adapted without re-deriving the parser.

    The raw long file is (``geo_code`` / ``nace_r2_code`` / ``var`` / ``year`` / ``value``, latest
    year); this picks one ``country`` (default: the first present), pivots the growth-account
    variables named in ``var_codes`` (defaults to the EU KLEMS statistical codes), maps the NACE
    industry code to its ISIC section letter, and averages within a section (VA-weighted if a VA
    variable is present). Because codes vary by release, ``var_codes`` is overridable; if the
    expected variables are absent this raises with the codes it DID find. A tidy frame passes
    through."""
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


def load_euklems_workbook(path, *, trend_window: int = 5) -> pd.DataFrame:
    """Read the REAL EU KLEMS 2024-release growth-accounts WORKBOOK into the tidy per-ISIC-section
    frame ``extract_euklems`` consumes (``isic_section``, ``mfp``, ``lp_growth``,
    ``labour_cost_share``, ``va_weight``). The workbook has ONE SHEET PER VARIABLE, each a matrix of
    ``nace_r2_code`` / ``geo_code`` / ``var`` + a column PER YEAR, reduced per single-letter ISIC
    section (the aggregates EU KLEMS already provides). Uses the CORRECT per-hour variables (review
    P1 2026-09-09 — the prior version used LP2_G = value added per PERSON and CAP_QI = the aggregate
    capital-services index, which is dimensionally inconsistent and reversed the BRD/MIL ordering):

    - ``LP1ConTFP`` — the per-hour-worked **TFP contribution** to value-added growth (p.p., a delta-
      log change). This IS the growth-accounting-identified MFP EU KLEMS computes, so it becomes the
      sourced ``mfp`` driver DIRECTLY (no hand decomposition, no capital-per-hour to reconstruct).
    - ``LP1_G`` — growth of value added **per hour worked** (delta log), the observed
      labour-productivity series → ``sector_productivity`` (a diagnostic; the wrapper uses ``mfp``
      when present).
    - ``LAB`` / ``VA_CP`` — labour compensation and value added at current prices → a
      ``labour_cost_share`` = ``LAB/VA_CP`` at the latest common year (an approximate source labour
      share for provenance; NOT an adjacent-period Törnqvist share — flagged as such).
    - ``VA_CP`` at the latest year → ``va_weight``.

    Both LP series are **delta-log percentage** changes, so a per-year value ``x`` p.p. is converted
    to the proportional annual rate the digest/wrapper use with ``expm1(mean(x)/100)`` (review P3
    2026-09-09 — the mean is over the recent ``trend_window`` years). Values are computed per
    (``geo_code``, section) and then VA-weighted across geographies per section (review P2
    2026-09-15), so a multi-country workbook is VA-weighted, not equal-averaged; the supplied file
    is Austria-only so each section has a single geo (the collapse is a no-op today)."""
    # (The reader's incidental stdout — openpyxl's stray "Legend" print-area line — is swallowed by
    # the caller _rebuild_digest, which wraps the whole build in a stdout redirect.)
    tfp = pd.read_excel(path, sheet_name="LP1ConTFP")
    lp = pd.read_excel(path, sheet_name="LP1_G")
    lab = pd.read_excel(path, sheet_name="LAB")
    va = pd.read_excel(path, sheet_name="VA_CP")

    def _year_cols(frame: pd.DataFrame) -> list[str]:
        return [c for c in frame.columns if str(c).isdigit()]

    def _sections(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame["nace_r2_code"].astype(str).str.fullmatch(r"[A-Za-z]")]

    tfp, lp, lab, va = (_sections(f) for f in (tfp, lp, lab, va))
    years = sorted(int(y) for y in _year_cols(lp))
    recent = [str(y) for y in years[-trend_window:]]
    last = str(years[-1])
    has_geo = "geo_code" in lp.columns

    # Compute values per (geo_code, section) FIRST, then VA-weight-collapse across geographies per
    # section (review P2 2026-09-15). Averaging across geographies before weighting would equal-
    # weight countries and discard the per-country VA the weighting needs — a silent corruption of a
    # future multi-country workbook. The supplied file is Austria-only, so each section has one geo
    # and the collapse is a no-op today, but the aggregation is now correct by construction.
    def _geo(frame: pd.DataFrame, sec: str, geo: str | None) -> pd.DataFrame:
        r = frame[frame["nace_r2_code"].astype(str) == sec]
        if geo is not None and "geo_code" in frame.columns:
            r = r[r["geo_code"].astype(str) == geo]
        return r

    def _val(frame: pd.DataFrame, cols: list[str], sec: str, geo: str | None) -> float | None:
        r = _geo(frame, sec, geo)
        vals = [
            float(pd.to_numeric(r[c], errors="coerce").mean())
            for c in cols
            if c in frame.columns and not pd.isna(pd.to_numeric(r[c], errors="coerce").mean())
        ]
        return sum(vals) / len(vals) if vals else None

    per_geo: dict[str, list[dict]] = {}  # section → list of per-geo rows (mfp/lp/share/va_weight)
    for section in sorted(lp["nace_r2_code"].astype(str).unique()):
        geos = (
            sorted(_geo(lp, section, None)["geo_code"].astype(str).unique()) if has_geo else [None]
        )
        for geo in geos:
            mfp_pp = _val(tfp, recent, section, geo)  # per-hour TFP contribution, delta-log p.p.
            lp_pp = _val(lp, recent, section, geo)  # per-hour VA growth, delta-log p.p.
            lab1, va1 = _val(lab, [last], section, geo), _val(va, [last], section, geo)
            if mfp_pp is None or lp_pp is None or not lab1 or not va1:
                continue
            per_geo.setdefault(section, []).append(
                {
                    # delta-log p.p. → proportional annual rate (review P3): expm1(mean_pp / 100).
                    "mfp": math.expm1(mfp_pp / 100.0),
                    "lp_growth": math.expm1(lp_pp / 100.0),
                    "labour_cost_share": max(min(lab1 / va1, 1.0), 1e-6),
                    "va_weight": va1,
                }
            )

    rows = []
    for section, gs in per_geo.items():
        wsum = sum(g["va_weight"] for g in gs) or 1.0
        rows.append(
            {
                "isic_section": section,
                # VA-weighted across geographies (populous/larger economies dominate).
                "mfp": sum(g["mfp"] * g["va_weight"] for g in gs) / wsum,
                "lp_growth": sum(g["lp_growth"] * g["va_weight"] for g in gs) / wsum,
                "labour_cost_share": (
                    sum(g["labour_cost_share"] * g["va_weight"] for g in gs) / wsum
                ),
                "va_weight": wsum,
            }
        )
    return pd.DataFrame(rows)


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
            # Annualised log-change → PROPORTIONAL annual rate (review P3 2026-09-09): the digest
            # and wrapper compound as (1+rate), so store the proportional rate, not the log rate.
            per_country[code] = math.expm1(math.log(gy[hi] / gy[lo]) / (hi - lo))
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
    Member populations are SUMMED per archetype (and globally) at each year, then each knot's rate
    is the **interval CAGR of the aggregate total** to the NEXT knot — ``(P[y_next]/P[y])**
    (1/(y_next−y))−1`` — so holding the knot's rate piecewise-constant reproduces the WPP population
    LEVEL at the next knot exactly (review P2 2026-09-15). The earlier instantaneous
    ``pop[y+1]/pop[y]−1`` held for a decade over-/under-shot the WPP levels (global ~+2.4% by 2050),
    and a per-country CAGR then start-weighted mean would only approximate it. Summing members IS
    the population weighting. The final knot has no successor, so it keeps the aggregate one-year
    rate. Returns ``{archetype: {knot: rate}}``. Accepts the raw WPP export (``ISO3_code``/``Time``/
    ``PopTotal`` + a Medium-variant filter) via :func:`normalise_wpp`."""
    data = normalise_wpp(data)
    data = data[data["iso3"].isin(cmap)]  # mapped countries only (as PWT; no default-to-S skew)
    knot_list = sorted(knots)
    # SUM member populations per archetype (and globally) at each year, THEN take the interval CAGR
    # of the aggregate totals (review P2 2026-09-15): this reproduces the WPP LEVEL at the next knot
    # exactly, and summing IS the population weighting (populous members dominate). A per-country
    # CAGR then start-year-weighted mean would only approximate it (fixed start weights ignore the
    # within-interval reweighting).
    archetype = data["iso3"].map(lambda c: country_archetype(str(c), cmap))
    totals: dict[str, pd.Series] = {}  # archetype (+ "__all__") → year → summed population
    for a in sorted(set(archetype)):
        totals[a] = data[archetype == a].groupby("year")["population"].sum().sort_index()
    totals["__all__"] = data.groupby("year")["population"].sum().sort_index()

    out: dict[str, dict[int, float]] = {}
    for i, y in enumerate(knot_list):
        y_next = knot_list[i + 1] if i + 1 < len(knot_list) else None
        for a, tot in totals.items():
            if y not in tot.index or tot[y] <= 0:
                continue
            if y_next is not None and y_next in tot.index and tot[y_next] > 0:
                # interval CAGR knot→next-knot on the aggregate total → reproduces the WPP level.
                rate = (tot[y_next] / tot[y]) ** (1.0 / (y_next - y)) - 1.0
            elif (y + 1) in tot.index and tot[y] > 0:
                rate = tot[y + 1] / tot[y] - 1.0  # final knot: instantaneous one-year rate
            else:
                continue
            out.setdefault(a, {})[y] = round(rate, 4)
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
    """Per-sector-archetype series from the EU KLEMS growth-accounts workbook (via
    :func:`load_euklems_workbook`). Expects columns ``isic_section``, ``mfp`` (per-hour TFP
    contribution, the growth-accounting-identified MFP — LP1ConTFP), ``lp_growth`` (per-hour
    value-added growth — LP1_G, a diagnostic), ``labour_cost_share`` (approximate source labour
    share), and ``va_weight`` (value added, the aggregation weight). Returns ``{"mfp": {...},
    "sector_productivity": {...}, "source_labour_shares": {...}}``, each ``{archetype:
    value_or_{knot: rate}}``, VA-weighted over each archetype's industries. The ``mfp`` series is
    the one the wrapper actually uses (review P1 2026-09-09 — replaces the dimensionally-wrong
    LP2_G/CAP_QI decomposition); ``sector_productivity`` is emitted as observed-LP context."""
    data = normalise_euklems(data)
    if "mfp" not in data.columns:
        # The real pipeline feeds load_euklems_workbook output (which carries the LP1ConTFP-derived
        # 'mfp'). The legacy long-format normalise_euklems path does not identify MFP; fail loudly
        # rather than KeyError deep inside the loop (review P1 2026-09-09).
        raise ValueError(
            "extract_euklems needs an 'mfp' column (per-hour TFP contribution, LP1ConTFP); pass "
            "the load_euklems_workbook output. The long-format normaliser does not identify MFP."
        )
    mfp: dict[str, dict[str, float]] = {}
    lp: dict[str, dict[str, float]] = {}
    sl: dict[str, dict[str, float]] = {}
    wt: dict[str, dict[str, float]] = {}
    for _, r in data.iterrows():
        a = industry_archetype(str(r["isic_section"]), imap)
        code = str(r["isic_section"])
        mfp.setdefault(a, {})[code] = float(r["mfp"])
        lp.setdefault(a, {})[code] = float(r["lp_growth"])
        sl.setdefault(a, {})[code] = float(r["labour_cost_share"])
        wt.setdefault(a, {})[code] = float(r.get("va_weight", 1.0))

    def _agg(series: dict[str, dict[str, float]], rnd: int) -> dict[str, float]:
        # VA-weighted per archetype; the world __all__ is VA-weighted over ALL members (review P2
        # 2026-09-09 — NOT a simple mean of the already-aggregated BRD/MIL archetypes).
        out = {a: round(_weighted_mean(series[a], wt[a]), rnd) for a in series}
        allv = {c: v for a in series for c, v in series[a].items()}
        allw = {c: w for a in wt for c, w in wt[a].items()}
        if allv:
            out["__all__"] = round(_weighted_mean(allv, allw), rnd)
        return out

    mfp_out = {a: {knot: v} for a, v in _agg(mfp, 4).items()}
    sp = {a: {knot: v} for a, v in _agg(lp, 4).items()}
    sls = _agg(sl, 3)
    return {"mfp": mfp_out, "sector_productivity": sp, "source_labour_shares": sls}


# --------------------------------------------------------------------------------------------------
# NGFS — emissions intensity of GDP per sector archetype (pinned scenario tuple)
# --------------------------------------------------------------------------------------------------
def extract_ngfs_emissions(
    data: pd.DataFrame,
    *,
    model: str,
    scenario: str,
    region: str = "World",
    co2_var: str = "Emissions|CO2",
    gdp_var: str = "GDP|PPP|Counterfactual without damage",
    start_year: int = 2025,
    sector_bundles: dict[str, list[str]] | None = None,
) -> dict[str, dict[int, float]]:
    """Emissions-intensity decarbonisation from an NGFS scenario-explorer export (columns ``model``,
    ``scenario``, ``region``, ``variable``, ``year``, ``value``).

    Selection is an EXPLICIT, VALIDATED tuple (review P1/P2 2026-09-09 — not loose substrings that
    could silently mix regions/variables/units): ``model`` and ``scenario`` match after stripping a
    trailing ``(version: n)`` suffix and normalising the ``°``→``?`` mangling, but must resolve to
    EXACTLY ONE value each (raises on none/ambiguous); ``region`` must equal ``region`` (default
    ``"World"``); and the CO2 and GDP series are the EXACT ``co2_var``/``gdp_var`` variable names,
    each required to be a single series (raises otherwise). No averaging across variables/regions.

    Intensity is DERIVED as ``co2_var / gdp_var`` per year. A knot is emitted at **every source year
    ≥ ``start_year``** and its rate is the CAGR over that source year → the NEXT source year (review
    P1 2026-09-15). Emitting all source years — rather than a sparse (2025, 2040) pair whose final
    knot then annualised the 2040→2100 tail and held that gentle average from 2040 (overstating 2050
    intensity ~30%) — makes the piecewise-constant trajectory reproduce the source intensity at
    every source year, so the 2030–2050 climate-risk horizon is faithful. The last source year holds
    flat (rate 0.0 — no further source information). The CO2/GDP endpoints are validated finite with
    positive GDP and positive intensity before the ratio/exponentiation, so a scenario with a zero
    or net-negative endpoint fails loudly rather than emitting inf/NaN (a multiplicative, non-
    negative intensity SCALE cannot represent net-negative emissions — e.g. Net Zero 2050 after
    ~2050 — and the engine forbids negative scales; such a scenario needs a gross-intensity +
    removals split, not this driver).

    PER-SECTOR SPLIT (review-9 1d, 2026-09-16): if ``sector_bundles`` is given (``{archetype:
    [co2_variable, ...]}``), each archetype gets its OWN intensity path — the SUM of its CO2
    component variables divided by the SAME economy-wide ``gdp_var`` (there is no sectoral GDP in
    the NGFS export, so the shared denominator is the honest choice; the split is on emissions).
    Each bundle is validated the same way (finite, positive). This lets goods vs services diverge
    at their own rates (BRD ≈ industry+processes+energy-supply falls faster than MIL ≈
    transport+buildings). ``__all__`` is always the economy-wide ``co2_var`` path.

    Returns ``{archetype: {year: rate}}`` (always ``__all__``; plus one key per ``sector_bundles``
    archetype). Negative = decarbonisation. Accepts the IAMC WIDE (year-columns) export via
    :func:`normalise_ngfs`."""
    data = normalise_ngfs(data)

    def _norm(x: str) -> str:
        # strip "(version: n)", normalise the °→? mangling and whitespace, lowercase.
        x = re.sub(r"\(version:[^)]*\)", "", str(x))
        return x.replace("?", "").replace("°", "").strip().lower()

    def _resolve(col: str, want: str) -> str:
        want_n = _norm(want)
        matches = sorted({v for v in data[col].dropna().unique() if _norm(v) == want_n})
        if len(matches) != 1:
            raise ValueError(
                f"NGFS {col} {want!r} did not resolve to exactly one value (got {matches}); "
                f"available: {sorted(data[col].dropna().unique())}."
            )
        return matches[0]

    model_r, scen_r = _resolve("model", model), _resolve("scenario", scenario)
    sub = data[
        (data["model"] == model_r) & (data["scenario"] == scen_r) & (data["region"] == region)
    ]
    if sub.empty:
        raise ValueError(
            f"NGFS: no rows for model={model_r!r}, scenario={scen_r!r}, region={region!r}."
        )

    def _series(var: str) -> pd.Series:
        v = sub[sub["variable"] == var]
        if v.empty:
            raise ValueError(
                f"NGFS variable {var!r} not present for the selected model/scenario/region; "
                f"available: {sorted(sub['variable'].unique())}."
            )
        if v["variable"].nunique() != 1 or v.groupby("year").size().max() > 1:
            raise ValueError(f"NGFS variable {var!r} is ambiguous (multiple series/units).")
        return v.groupby("year")["value"].first().sort_index()

    g = _series(gdp_var)

    def _check_finite(var: str, s: pd.Series) -> None:
        # Validate BEFORE dividing/exponentiating (review P1 2026-09-15): a zero/negative or
        # non-finite CO2 or GDP would give inf/NaN intensity that later becomes a non-positive (or
        # NaN) multiplicative scale the engine rejects. Fail loudly, naming the offending year.
        bad = s[~s.apply(lambda x: math.isfinite(x))]
        if len(bad):
            raise ValueError(f"NGFS {var!r} is non-finite at year(s) {sorted(bad.index)}.")

    def _intensity_path(co2: pd.Series, label: str) -> dict[int, float]:
        """Emissions-intensity knot rates for one CO2 series over the shared GDP denominator: a knot
        at every source year ≥ start_year, each the CAGR to the NEXT source year (last held flat),
        so the piecewise-constant path reproduces the source intensity at every source year."""
        common = co2.index.intersection(g.index)
        _check_finite(label, co2[common])
        _check_finite(gdp_var, g[common])
        nonpos_g = g[common][g[common] <= 0]
        if len(nonpos_g):
            raise ValueError(
                f"NGFS {gdp_var!r} is non-positive at year(s) {sorted(nonpos_g.index)}."
            )
        s = (co2[common] / g[common]).sort_index()
        nonpos_i = s[s <= 0]
        if len(nonpos_i):
            # e.g. Net Zero 2050's CO2 goes net-negative by ~2050 — a multiplicative intensity scale
            # cannot represent that. Reject rather than emit a negative/zero scale.
            raise ValueError(
                f"NGFS emissions intensity ({label}) is non-positive at year(s) "
                f"{sorted(nonpos_i.index)} (net-negative emissions cannot be a multiplicative "
                f"intensity scale; use a gross-intensity + removals split for that scenario)."
            )
        yrs = [y for y in s.index if y >= start_year]
        path: dict[int, float] = {}
        for i, y in enumerate(yrs):
            if i + 1 < len(yrs):
                y_end = yrs[i + 1]
                path[y] = round((s[y_end] / s[y]) ** (1.0 / (y_end - y)) - 1.0, 4)
            else:
                path[y] = 0.0  # final source year holds flat (no further source information)
        return path

    out: dict[str, dict[int, float]] = {}
    all_path = _intensity_path(_series(co2_var), co2_var)
    if all_path:
        out["__all__"] = all_path
    available = set(sub["variable"].unique())
    for archetype, variables in (sector_bundles or {}).items():
        # Sum the archetype's CO2 component variables (each a single series), then its own intensity
        # path over the shared economy-wide GDP denominator. If the sectoral variables are absent
        # (an economy-wide-only export), SKIP this archetype rather than raise — the __all__ path is
        # still emitted, so an older/leaner NGFS file degrades gracefully to economy-wide only.
        if not all(v in available for v in variables):
            continue
        # Sum on the intersection of years the components share (adding an empty-index Series as a
        # start would align to nothing and yield all-NaN — so reduce over the real series).
        comps = [_series(v) for v in variables]
        bundle = comps[0].copy()
        for s in comps[1:]:
            bundle = bundle.add(
                s, fill_value=None
            )  # aligns on year; NaN where a component is absent
        path = _intensity_path(bundle.dropna(), f"{archetype} bundle")
        if path:
            out[archetype] = path
    return out


# --------------------------------------------------------------------------------------------------
# ILOSTAT — labour-force participation growth per region archetype
# --------------------------------------------------------------------------------------------------
def extract_ilo_participation(
    data: pd.DataFrame,
    cmap: dict[str, str],
    *,
    knots: tuple[int, ...] = (2025, 2040),
    trend_window: int = 10,
    min_latest_year: int = 2015,
    clip: float = 0.01,
) -> dict[str, dict[int, float]]:
    """Per-archetype labour-force-participation growth from ILOSTAT LFPR — a LOW-CONFIDENCE driver
    (review P2 2026-09-09; see caveats below). The driver is a *proportional* annual growth rate of
    the participation RATE. ILOSTAT is historical, so each country's trend is the annualised log-
    change of its rate over the most recent ``trend_window`` years, converted to a proportional rate
    (``expm1``, review P3) and held forward at every knot (a no-further-information assumption).

    Hardening (review P2 2026-09-09, 2026-09-15):
    - **Common-vintage cutoff:** a country is used only if its latest observation is ≥
      ``min_latest_year`` — stale series (some ILOSTAT countries end in 1980/1991/2001) are dropped
      rather than projected forward decades identically.
    - **Recent-window requirement:** a country needs ≥2 observations INSIDE the recent
      ``trend_window`` — otherwise it is DROPPED, not silently back-filled from its entire history
      (which would project a decades-long trend under a "recent window" label).
    - **Outlier clip:** the per-country annual trend is clipped to ±``clip`` (default ±1%/yr), so a
      survey-break artefact (the raw data has trends near −8%/yr and +6%/yr) cannot dominate.
    - **Weighting:** an UNWEIGHTED archetype mean. The participation RATE is not a size measure (the
      earlier rate-weighting was indefensible) and labour-force / working-age-population weights are
      not in this file; equal weighting is the honest fallback until those weights are added.
    Only MAPPED countries are included. Accepts the raw ILOSTAT CSV via ``normalise_ilostat``.
    Returns ``{archetype: {knot: rate}}``."""
    data = normalise_ilostat(data)
    trends: dict[str, dict[str, float]] = {}
    for code, g in data.groupby("iso3"):
        code = str(code)
        if code not in cmap:  # mapped countries only (no silent default-to-S)
            continue
        gy = g.set_index("year")["lfpr"].sort_index()
        gy = gy[gy > 0]
        if len(gy) < 2:
            continue
        last = int(gy.index.max())
        if last < min_latest_year:  # stale vintage → drop (do not project a 1990s trend to 2040)
            continue
        recent = gy[gy.index >= last - trend_window]
        if len(recent) < 2:
            # Fewer than two observations INSIDE the recent window → DROP the country (review P2
            # 2026-09-15). The earlier fall-back to the entire history projected a decades-long
            # trend (e.g. 1973→2022) under a "recent ≤10-year window" label — dishonest; skip it.
            continue
        y0, y1 = int(recent.index.min()), int(recent.index.max())
        trend = math.expm1(math.log(recent[y1] / recent[y0]) / (y1 - y0))  # log→proportional (P3)
        trend = max(min(trend, clip), -clip)  # clip survey-break outliers
        trends.setdefault(country_archetype(code, cmap), {})[code] = trend
    per_arch = {a: round(_weighted_mean(trends[a], None), 4) for a in trends}  # unweighted mean
    allv = {c: r for a in trends for c, r in trends[a].items()}
    world = round(_weighted_mean(allv, None), 4) if allv else None
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
    """Rebuild the digest from whatever raw files are present; return (digest, missing_sources).

    All reads/computation only — no legitimate stdout — so we swallow any incidental stdout from
    the readers. openpyxl 3.1.x has a stray ``print()`` in ``print_settings.from_string`` that
    fires on workbooks carrying a named print-area (both the PWT and EU KLEMS files have one called
    "Legend"); left unguarded it pollutes the tool output and the ``--check`` diagnostics."""
    with contextlib.redirect_stdout(io.StringIO()):
        return _rebuild_digest_impl()


def _rebuild_digest_impl() -> tuple[dict, list[str]]:
    digest = json.loads(_DIGEST.read_text())
    missing: list[str] = []
    cmap = _load_map("country_to_archetype.csv", "iso3")
    imap = _load_map("industry_to_archetype.csv", "isic_section")

    pwt = _RAW / "pwt1001.xlsx"
    if pwt.exists():
        rates = extract_pwt_productivity(pd.read_excel(pwt, sheet_name="Data"), cmap)
        pwt_src = (
            "Penn World Table 10.01, sheet Data, variable rtfpna (TFP at constant national prices, "
            "index); annualised log-change over 2000–2019 (window end 2019 — PWT ends 2019), "
            "expm1→proportional; GDP-weighted (rgdpo, 2019) over mapped countries per N/S "
            "archetype. https://www.rug.nl/ggdc/productivity/pwt/"
        )
        _apply_region(digest, "productivity", rates, source=pwt_src)
    else:
        missing.append("pwt1001.xlsx (productivity)")

    wpp = _RAW / "wpp2024_population.csv"
    if wpp.exists():
        # low_memory=False: the WPP CSV mixes numeric country rows with blank-field region rows, so
        # a chunked read infers mixed dtypes and warns; read it whole (it is ~85MB, fits in memory).
        rates = extract_wpp_population(pd.read_csv(wpp, low_memory=False), cmap)
        wpp_src = (
            "UN World Population Prospects 2024, Total Population (both sexes), Medium variant; "
            "knots 2025/2035/2050, each the interval CAGR to the next knot (final knot: one-year "
            "rate) so holding it reproduces the WPP level at the next knot; population-weighted "
            "over mapped countries per N/S archetype. https://population.un.org/wpp/"
        )
        _apply_region(digest, "population", rates, source=wpp_src)
    else:
        missing.append("wpp2024_population.csv (population)")

    ilo = _RAW / "ilostat_lfpr.csv"
    if ilo.exists():
        rates = extract_ilo_participation(pd.read_csv(ilo), cmap)
        ilo_src = (
            "ILOSTAT indicator EAP_DWAP_SEX_AGE_RT (labour-force participation rate), SEX_T total, "
            "AGE_AGGREGATE_YGE15 (15+) band; per-country annualised log-trend over the recent "
            "≤10-year window (latest obs ≥2015), expm1→proportional, clipped ±1%/yr; UNWEIGHTED "
            "archetype mean, held forward. LOW confidence: no labour-force size weights, single "
            "recent trend projected flat. https://ilostat.ilo.org/"
        )
        # LOW confidence (review P2 2026-09-09): rate-weighting was indefensible, vintages vary, and
        # the trend is held flat — honestly downgraded until size weights + break handling land.
        _apply_region(digest, "labour_participation", rates, source=ilo_src, confidence="low")
    else:
        missing.append("ilostat_lfpr.csv (labour participation)")

    klems = _RAW / "euklems_2023_growth_accounts.xlsx"
    if klems.exists():
        out = extract_euklems(load_euklems_workbook(klems), imap)
        # Value-level provenance (review P2 2026-09-09): geography, release, sheet, years, window,
        # transform — not a bare "EU KLEMS 2023 (ext)".
        mfp_src = (
            "EU KLEMS & INTANProd 2024 release, growth accounts, Austria (AT — sole geo in the "
            "supplied workbook); sheet LP1ConTFP (per-hour-worked TFP contribution to VA growth, "
            "delta-log p.p.); mean of the 5 most recent years (2017–2021 — a COVID-affected window "
            "that can reverse the goods-vs-services ordering vs pre-COVID windows), "
            "expm1(mean/100) → proportional rate; aggregated to ISIC sections then VA-weighted "
            "(VA_CP). LOW "
            "confidence: single country, pandemic window. https://euklems-intanprod-llee.luiss.it/"
        )
        lp_src = mfp_src.replace(
            "LP1ConTFP (per-hour-worked TFP contribution to VA growth, ", "LP1_G ("
        )
        share_src = (
            "EU KLEMS & INTANProd 2024 release, Austria; latest-year LAB/VA_CP (labour "
            "compensation over value added at current prices) — an APPROXIMATE source labour "
            "share, NOT an adjacent-period Törnqvist share. "
            "https://euklems-intanprod-llee.luiss.it/"
        )
        # LOW confidence (review P2 2026-09-15): single-country (Austria) proxy on a COVID window
        # (2017–2021) that can flip the BRD/MIL ordering vs pre-COVID windows — too weak for medium.
        _apply_sector(digest, "mfp", out["mfp"], mfp_src, confidence="low")
        _apply_sector(
            digest, "sector_productivity", out["sector_productivity"], lp_src, confidence="low"
        )
        _apply_source_shares(digest, out["source_labour_shares"], share_src)
        # The old capital_deepening driver is retired for EU KLEMS (mfp is now sourced directly);
        # drop any stale entry so it does not linger with an illustrative value.
        digest["sector_drivers"].pop("capital_deepening", None)
    else:
        missing.append("euklems_2023_growth_accounts.xlsx (mfp/sector productivity/labour share)")

    ngfs = _RAW / "ngfs_phase5.csv"
    if ngfs.exists():
        rates = extract_ngfs_emissions(pd.read_csv(ngfs), **_NGFS)
        multi_sector = set(rates) - {"__all__"}
        sector_note = (
            (
                " Per-sector paths (BRD = industry energy demand + industrial processes + energy "
                "supply; MIL = transport + residential/commercial; AFOLU excluded) use each "
                "bundle's summed CO2 over the SAME economy-wide GDP (no sectoral GDP in NGFS)."
            )
            if multi_sector
            else ""
        )
        ngfs_src = (
            f"NGFS Phase 5 scenario explorer: model {_NGFS['model']}, scenario "
            f"{_NGFS['scenario']}, region {_NGFS['region']}; emissions intensity DERIVED as "
            f"{_NGFS['co2_var']} / {_NGFS['gdp_var']}; a knot at every source year (from 2025), "
            "each the CAGR to the next source year, so the path reproduces the source intensity at "
            f"every source year (last year held flat).{sector_note} "
            "https://data.ece.iiasa.ac.at/ngfs/"
        )
        _apply_sector(digest, "emissions_intensity", dict(rates), ngfs_src)
    else:
        missing.append("ngfs_phase5.csv (emissions intensity)")

    return digest, missing


def _apply_region(
    digest: dict, driver: str, rates: dict, *, source: str, confidence: str = "medium"
) -> None:
    if not rates:
        return
    # REPLACE the driver's whole entry (clear stale illustrative keys first) so an extracted driver
    # never leaves a mixed real+illustrative state under one source (review 2026-09-07).
    digest["region_drivers"][driver] = {
        key: {
            "knots": {str(y): v for y, v in knots.items()},
            "confidence": confidence,
            "source": source,
        }
        for key, knots in rates.items()
    }


def _apply_sector(
    digest: dict, driver: str, rates: dict, source: str, *, confidence: str = "medium"
) -> None:
    if not rates:
        return
    # REPLACE wholesale (see _apply_region): an economy-wide-only extraction (e.g. NGFS gives only
    # __all__) must not leave a stale illustrative per-sector key mixed under the real source.
    digest["sector_drivers"][driver] = {
        key: {
            "knots": {str(y): v for y, v in knots.items()},
            "confidence": confidence,
            "source": source,
        }
        for key, knots in rates.items()
    }


def _apply_source_shares(
    digest: dict, shares: dict, source: str, *, confidence: str = "medium"
) -> None:
    if not shares:
        return
    for key, val in shares.items():
        digest["source_labour_shares"][key] = {
            "value": val,
            "confidence": confidence,
            "source": source,
        }


def _raw_manifest() -> dict:
    """SHA-256 + byte size of each raw source file actually present, so the (git-ignored,
    separately-licensed) inputs behind the committed digest are AUDITABLE even though the files
    themselves are not redistributed (review P2 2026-09-15). A reviewer who acquires the same
    releases can confirm byte-identity; a changed source shows up as a changed hash here."""
    manifest: dict[str, dict] = {}
    for f in sorted(_RAW.glob("*")):
        if f.suffix.lower() not in (".xlsx", ".csv"):
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        manifest[f.name] = {"sha256": h, "bytes": f.stat().st_size}
    return {
        "_README": (
            "SHA-256 + size of the raw source files behind sources/inputs.json. The files are "
            "git-ignored (large, separately licensed); this manifest makes the raw→digest step "
            "auditable — re-acquire the cited releases (see raw/README.md) and compare hashes. "
            "Regenerated by scripts/extract_structural_sources.py."
        ),
        "files": manifest,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="non-zero if the digest is stale vs raw/")
    args = ap.parse_args()
    if not _RAW.exists() or not any(_RAW.glob("*.xlsx")) and not any(_RAW.glob("*.csv")):
        print(
            f"No raw source files in {_RAW.relative_to(_ROOT)} — see its README for downloads. "
            "The committed digest keeps its current values."
        )
        return 0
    digest, missing = _rebuild_digest()
    if missing:
        print("Extracted from available raw files; STILL MISSING (kept prior values):")
        for m in missing:
            print(f"  - {m}")
    rendered = json.dumps(digest, indent=2, ensure_ascii=False) + "\n"
    manifest_rendered = json.dumps(_raw_manifest(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        stale = _DIGEST.read_text() != rendered
        # The manifest records LOCAL raw-file hashes; only compare it when the file exists (a fresh
        # clone without the raw files can't reproduce hashes — don't fail CI's no-raw path on it).
        if stale:
            print(
                "inputs.json is OUT OF DATE vs the raw sources — re-run without --check + commit."
            )
            return 1
        print("inputs.json matches the extracted raw sources.")
        return 0
    _DIGEST.write_text(rendered)
    _MANIFEST.write_text(manifest_rendered)
    print(f"Wrote {_DIGEST}.")
    print(f"Wrote {_MANIFEST}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
