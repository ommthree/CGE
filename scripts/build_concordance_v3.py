#!/usr/bin/env python
"""Build concordance_v3.json — PER-DRIVER, TIME-VARYING block weights (pipeline step 1c).

    python scripts/build_concordance_v3.py           # (re)write concordance_v3.json
    python scripts/build_concordance_v3.py --check     # non-zero if stale vs v2 + raw sources

review-9 non-blocking ask 1c. concordance_v2 blends each coarse EXIOBASE block with ONE static
GDP-share weight table (IMF WEO 2024), applied to EVERY region driver (population, participation,
productivity) and held fixed over the horizon. v3 adds **driver-class-specific, year-indexed**
weights derived from the real sources already vendored for the trajectories:

  * ``population`` weights  ← UN WPP 2024 per-country population at each trajectory knot year
                              (genuinely time-varying: a block's 2050 blend differs from its 2025).
  * ``output`` weights      ← PWT 10.01 real GDP (``rgdpo``, 2019 — the latest PWT year), held flat.

The region drivers map to a class: ``population``→population weights, ``productivity``→output
weights, ``labour_participation``→ (no per-country labour series available) the static v2
``block_membership`` GDP weights, documented as a proxy. The loader picks the class per driver.

Only the named member countries carry real per-source weights; the EXIOBASE rest-of-region
aggregates (``WA``/``WE``/``WF``/``WL``/``WM``) have no single ISO3, so they RETAIN their v2
residual share (a documented emerging-economy proxy) and named members renormalise around it. Blocks
that are a single 1:1 country keep weight 1.0. v2 is preserved in the file for back-compat.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_STRUCT = _ROOT / "data" / "structural"
_V2 = _STRUCT / "concordance_v2.json"
_V3 = _STRUCT / "concordance_v3.json"
_RAW = _STRUCT / "sources" / "raw"

# EXIOBASE/World-Bank 2-letter member code → ISO3 (for the WPP/PWT lookup). The rest-of-region
# EXIOBASE aggregates (W*) have no single ISO3 and are handled separately (residual, held at v2).
_ALPHA2_TO_ISO3 = {
    "AT": "AUT",
    "AU": "AUS",
    "BE": "BEL",
    "BG": "BGR",
    "CA": "CAN",
    "CH": "CHE",
    "CY": "CYP",
    "CZ": "CZE",
    "DK": "DNK",
    "EE": "EST",
    "ES": "ESP",
    "FI": "FIN",
    "GR": "GRC",
    "HR": "HRV",
    "HU": "HUN",
    "ID": "IDN",
    "IE": "IRL",
    "KR": "KOR",
    "LT": "LTU",
    "LU": "LUX",
    "LV": "LVA",
    "MT": "MLT",
    "MX": "MEX",
    "NL": "NLD",
    "NO": "NOR",
    "PL": "POL",
    "PT": "PRT",
    "RO": "ROU",
    "SE": "SWE",
    "SI": "SVN",
    "SK": "SVK",
    "TR": "TUR",
    "TW": "TWN",
    "ZA": "ZAF",
}
_KNOTS = (2025, 2035, 2050)  # the region-driver knot years the weights index


def _normalise_wpp(df: pd.DataFrame) -> pd.DataFrame:
    import sys

    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from scripts.extract_structural_sources import normalise_wpp

    return normalise_wpp(df)


def _population_by_iso3_year() -> pd.DataFrame:
    wpp = _normalise_wpp(pd.read_csv(_RAW / "wpp2024_population.csv", low_memory=False))
    return wpp.set_index(["iso3", "year"])["population"]


def _gdp_by_iso3() -> dict[str, float]:
    # PWT rgdpo at 2019 (its latest year) — real output level, the output-weight basis.
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):  # swallow openpyxl's stray print-area line
        pwt = pd.read_excel(_RAW / "pwt1001.xlsx", sheet_name="Data")
    g = pwt[pwt["year"] == 2019].dropna(subset=["rgdpo"])
    return {str(r["countrycode"]): float(r["rgdpo"]) for _, r in g.iterrows()}


def _weights_for_block(members: dict, basis: dict, residual_keys: set[str]) -> dict[str, float]:
    """Reweight a block's members by ``basis`` (a {code: size} map for the named members); the W*
    residual aggregates keep their v2 share and the named members are renormalised around it, so the
    block still sums to 1. If no named member has a basis size, fall back to the v2 weights."""
    named = {c: w for c, w in members.items() if c not in residual_keys}
    residual = {c: w for c, w in members.items() if c in residual_keys}
    sizes = {c: basis.get(_ALPHA2_TO_ISO3.get(c, c)) for c in named}
    if not any(sizes.get(c) for c in named):
        return dict(members)  # no basis data → keep v2 weights (documented)
    residual_total = sum(residual.values())
    named_budget = 1.0 - residual_total
    size_sum = sum(s for s in sizes.values() if s) or 1.0
    out: dict[str, float] = {}
    for c in named:
        s = sizes.get(c) or 0.0
        out[c] = round(named_budget * s / size_sum, 4)
    for c, w in residual.items():
        out[c] = round(w, 4)
    # Fix rounding so the block sums to exactly 1 (adjust the largest weight).
    drift = round(1.0 - sum(out.values()), 4)
    if out:
        k = max(out, key=lambda c: out[c])
        out[k] = round(out[k] + drift, 4)
    return out


def _build() -> dict:
    v2 = json.loads(_V2.read_text())
    pop = _population_by_iso3_year()
    gdp = _gdp_by_iso3()
    residual_keys = {"WA", "WE", "WF", "WL", "WM", "WT"}

    block_weights: dict[str, dict] = {"population": {}, "output": {}}
    for block, members in v2["block_membership"].items():
        if len(members) == 1:  # 1:1 country block — weight 1.0 for every class/year
            block_weights["population"][block] = {str(y): dict(members) for y in _KNOTS}
            block_weights["output"][block] = dict(members)
            continue
        # population weights are year-indexed (WPP projects annually)
        by_year: dict[str, dict] = {}
        for y in _KNOTS:
            basis = {
                iso3: float(pop.get((iso3, y), 0.0))
                for iso3 in (_ALPHA2_TO_ISO3.get(c, c) for c in members)
            }
            by_year[str(y)] = _weights_for_block(members, basis, residual_keys)
        block_weights["population"][block] = by_year
        # output weights held flat at PWT-2019
        block_weights["output"][block] = _weights_for_block(members, gdp, residual_keys)

    v3 = dict(v2)
    v3["block_weights"] = block_weights
    # Map each region driver to a weight class; participation has no per-country basis → v2 proxy.
    v3["driver_weight_class"] = {
        "population": "population",
        "productivity": "output",
        "labour_participation": "block_membership",  # static v2 fallback (no labour-force weights)
    }
    v3["provenance"] = dict(v2["provenance"])
    v3["provenance"]["source"] = (
        "Structural-trajectory region concordance v3: country-level development archetypes (World "
        "Bank income classification, FY2025) aggregated to EXIOBASE coarse-v3 blocks with "
        "PER-DRIVER, TIME-VARYING weights — population shares (UN WPP 2024, per knot) and output "
        "shares (PWT 10.01 rgdpo 2019); v2's static IMF-2024 GDP weights are retained as the "
        "participation proxy and back-compat block_membership."
    )
    v3["provenance"]["source_version"] = "structural-concordance-v3"
    v3["provenance"]["retrieved"] = "2026-09-16"
    v3["provenance"]["v3_note"] = (
        "v3 (review-9 1c 2026-09-16): adds PER-DRIVER, TIME-VARYING block_weights. population "
        "weights = UN WPP 2024 per-country population at each knot (2025/2035/2050 — genuinely "
        "year-varying); output weights = PWT 10.01 rgdpo (2019, held flat). driver_weight_class "
        "maps population→population, productivity→output, labour_participation→the static v2 "
        "block_membership GDP weights (no per-country labour-force series available — documented "
        "proxy). The EXIOBASE rest-of-region aggregates (W*) have no single ISO3 and retain their "
        "v2 residual share, with named members renormalised around it. v2 block_membership is "
        "preserved for back-compat."
    )
    v3["sources"] = dict(v2["sources"])
    v3["sources"]["block_weights"] = (
        "population: UN WPP 2024 Total Population (Medium), per-country at each knot; output: PWT "
        "10.01 rgdpo (2019). Same raw files as the trajectories (data/structural/sources/raw/)."
    )
    return v3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="non-zero if concordance_v3 is stale")
    args = ap.parse_args()
    if not (_RAW / "wpp2024_population.csv").exists() or not (_RAW / "pwt1001.xlsx").exists():
        print("WPP/PWT raw files absent — cannot build v3; keeping committed concordance_v3.json.")
        return 0
    rendered = json.dumps(_build(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not _V3.exists() or _V3.read_text() != rendered:
            print("concordance_v3.json is OUT OF DATE — re-run without --check + commit.")
            return 1
        print("concordance_v3.json matches v2 + the raw sources.")
        return 0
    _V3.write_text(rendered)
    print(f"Wrote {_V3}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
