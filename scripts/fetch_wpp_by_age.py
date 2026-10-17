#!/usr/bin/env python
"""Fetch UN WPP 2024 working-age (15+) population and vendor it as a tidy CSV.

    # 1. Get a free token: register at https://population.un.org/dataportal/about/dataapi (the
    #    /data endpoints need `Authorization: Bearer <token>`; the public indicator list does not).
    # 2. export WPP_API_TOKEN=<your token>
    python scripts/fetch_wpp_by_age.py

Pipeline step 1a-DATA (review-12 P1#3). The labour-supply identity is labour force ≈ 15+ population
× 15+ LFPR; ILOSTAT's LFPR already uses the 15+ denominator, but the vendored WPP "Total Population"
file is total population — multiplying total-population growth by 15+-LFPR growth implicitly assumes
the 15+ SHARE of total population is constant to 2050 (weak for ageing advanced / young emerging
economies). This pulls the matching **15+ population** so the labour driver uses a consistent
denominator.

Source: UN WPP 2024 Data Portal API, indicator 70 ("Population by age and sex — broad age groups"),
Medium variant (variantId 4), both sexes (sexId 3). Indicator 70 is a bag of OVERLAPPING functional
age combinations (0-14, 15-24, 15-49, 15-64, 16+, 18+, 50+, …) AND the cumulative **"15+"** category
(ageId 26). We select exactly that cumulative category — summing the 15-prefixed bands would
double/triple count (review-13 P1). Writes ``data/structural/sources/raw/wpp2024_working_age.csv``
(tidy ``iso3``/``year``/``population``, the SAME shape ``normalise_wpp`` emits) so the extractor
consumes it with the existing machinery.

OFFLINE acquisition step (needs network + a token). The parsing is a pure function
(``_parse_records``) tested on a captured real-API response fixture; only ``_fetch_pages`` touches
the network. The /data endpoints are auth-gated (HTTP 401 without a Bearer token) — if you cannot
obtain a token, the labour driver falls back to total population with a documented caveat.
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "data" / "structural" / "sources" / "raw" / "wpp2024_working_age.csv"

API = "https://population.un.org/dataportalapi/api/v1"
INDICATOR = 70  # Population by age and sex — broad age groups
VARIANT_MEDIUM = 4  # WPP "Medium" variant id (variantId)
SEX_BOTH = 3  # both sexes combined (sexId)
AGE_15_PLUS_ID = 26  # the CUMULATIVE "15+" category in indicator 70 (NOT a 5-year band)
AGE_15_PLUS_LABEL = "15+"
PAGE_SIZE = 1000  # the portal caps pageSize at 1000; fewer pages = fewer round-trips
START_YEAR, END_YEAR = 2024, 2051  # cover the structural knots (2025/2035/2050) + a +1 for growth


def _base_url() -> str:
    # Explicit page controls: pageSize=PAGE_SIZE, pageNumber walked from 1 by _fetch_pages.
    return (
        f"{API}/data/indicators/{INDICATOR}/locations/all/"
        f"start/{START_YEAR}/end/{END_YEAR}"
        f"?variants={VARIANT_MEDIUM}&sexes={SEX_BOTH}&pageSize={PAGE_SIZE}&format=json"
    )


def _page_payload(obj):
    """Normalise one page into (records, next_page_url).

    The documented Data Portal response is an OBJECT ``{data, pages, pageNumber, nextPage, ...}``;
    some deployments return a bare list for a single page. Accept both, and never silently treat a
    non-dict/non-list as empty."""
    if isinstance(obj, dict):
        if "data" not in obj:
            raise SystemExit(
                "WPP API page is a JSON object without a 'data' key — the response contract "
                f"changed; got keys {sorted(obj)}."
            )
        return obj["data"], obj.get("nextPage")
    if isinstance(obj, list):
        return obj, None  # bare array ⇒ single, un-paginated page
    raise SystemExit(f"WPP API page is neither object nor array (got {type(obj).__name__}).")


def _fetch_pages(token: str):
    import requests

    headers = {"Authorization": f"Bearer {token}"}
    url = _base_url()
    rows: list[dict] = []
    seen_urls: set[str] = set()
    while url:
        if url in seen_urls:  # defend against a server that echoes its own page as nextPage
            break
        seen_urls.add(url)
        r = requests.get(url, headers=headers, timeout=120)
        if r.status_code == 401:
            raise SystemExit(
                "WPP data API returned 401 — set WPP_API_TOKEN to a valid Bearer token "
                "(register at https://population.un.org/dataportal/about/dataapi)."
            )
        r.raise_for_status()
        records, next_url = _page_payload(r.json())
        rows.extend(records)
        url = next_url  # the portal paginates; follow nextPage until exhausted
    return rows


def _field(rec: dict, *names):
    """First present value among camelCase / PascalCase field spellings the portal has used."""
    for n in names:
        if n in rec and rec[n] is not None:
            return rec[n]
    return None


def _parse_records(records: list[dict]) -> list[dict]:
    """Pure parse: keep ONLY country rows, Medium variant, both sexes, the cumulative 15+ category;
    reject duplicate (iso3, year) observations. Returns tidy dicts iso3/year/population.

    Selection is by the EXPLICIT 15+ identity (ageId 26 OR ageLabel '15+'), never a sum of bands —
    indicator 70's overlapping 15-prefixed groups (15-24, 15-49, 15-64, …) would multiply-count."""
    import re

    out: dict[tuple[str, int], float] = {}
    for rec in records:
        iso3 = _field(rec, "iso3", "Iso3", "iso3Code")
        if not isinstance(iso3, str) or not re.fullmatch(r"[A-Za-z]{3}", iso3):
            continue  # drop region/aggregate rows (non-3-letter ISO3)
        variant = _field(rec, "variantId", "VariantId")
        sex = _field(rec, "sexId", "SexId")
        if variant is not None and int(variant) != VARIANT_MEDIUM:
            continue
        if sex is not None and int(sex) != SEX_BOTH:
            continue
        age_id = _field(rec, "ageId", "AgeId")
        age_label = _field(rec, "ageLabel", "AgeLabel")
        is_15_plus = (age_id is not None and int(age_id) == AGE_15_PLUS_ID) or (
            isinstance(age_label, str) and age_label.strip() == AGE_15_PLUS_LABEL
        )
        if not is_15_plus:
            continue
        year = _field(rec, "timeLabel", "TimeLabel", "time", "Time")
        value = _field(rec, "value", "Value")
        if year is None or value is None:
            continue
        key = (iso3.upper(), int(float(year)))
        if key in out:
            raise SystemExit(
                f"duplicate 15+ observation for {key[0]} {key[1]} — the query returned more than "
                "one row for the cumulative 15+ category (check variant/sex filters)."
            )
        out[key] = float(value)
    if not out:
        raise SystemExit(
            "no 15+ (ageId 26 / ageLabel '15+') country observations found — indicator 70 may have "
            "changed its age categories, or the variant/sex filters excluded everything."
        )
    rows = [{"iso3": k[0], "year": k[1], "population": v} for k, v in out.items()]
    rows.sort(key=lambda d: (d["iso3"], d["year"]))
    return rows


def main() -> int:
    token = os.environ.get("WPP_API_TOKEN")
    if not token:
        raise SystemExit(
            "WPP_API_TOKEN is not set. Register (free) at "
            "https://population.un.org/dataportal/about/dataapi, then `export "
            "WPP_API_TOKEN=<token>` and re-run. (The /data endpoints require a Bearer token; there "
            "is no anonymous access.)"
        )
    import pandas as pd

    records = _fetch_pages(token)
    if not records:
        raise SystemExit("WPP API returned no rows for indicator 70.")
    out = pd.DataFrame(_parse_records(records))
    out.to_csv(_OUT, index=False)
    n = out["iso3"].nunique()
    print(f"Wrote {_OUT} — {n} countries, years {out['year'].min()}–{out['year'].max()}.")
    print("Re-run scripts/extract_structural_sources.py to rebuild the digest with 15+ population.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
