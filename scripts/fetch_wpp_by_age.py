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
(ageId 26). We select exactly that cumulative category (``ages=26`` on the request + a parse-side
guard) — summing the 15-prefixed bands would double/triple count (review-13 P1). Writes
``data/structural/sources/raw/wpp2024_working_age.csv`` (tidy ``iso3``/``year``/``population``, the
SAME shape ``normalise_wpp`` emits) so the extractor consumes it with the existing machinery.

API contract (per the UN Data Portal OpenAPI, review-14 P1#2): the FILTER endpoint takes
``variants``/``sexes``/``ages`` as query parameters (NOT the ``/start/.../end/...`` path); responses
page by an explicit ``pageNumber`` with ``pageSize`` ≤ 100, as either an object
``{data, nextPage, pages, …}`` or a bare array. ``_fetch_pages`` walks ``pageNumber`` until a short/
empty page — it never assumes the first page is the whole dataset — and ``_validate_coverage``
refuses to write a truncated file (too few countries, or a missing knot year).

OFFLINE acquisition step (needs network + a token). The parsing/pagination logic is pure
(``_parse_records``/``_page_records``) and unit-tested on captured response shapes; only
``_fetch_pages`` touches the network. The /data endpoints are auth-gated (HTTP 401 without a Bearer
token) — if you cannot obtain a token, the labour driver falls back to total population with a
documented caveat.
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
PAGE_SIZE = 100  # the OpenAPI caps pageSize at 100 (review-14 P1#2) — page explicitly by pageNumber
START_YEAR, END_YEAR = 2024, 2051  # cover the structural knots (2025/2035/2050) + a +1 for growth


def _page_url(page_number: int) -> str:
    # review-14 P1#2: variants/sexes/ages are supported on the FILTER (query-parameter) endpoint,
    # not the /start/.../end/... path; and pagination is by an EXPLICIT pageNumber (max pageSize is
    # 100). We walk pageNumber from 1 ourselves rather than assuming a single page.
    return (
        f"{API}/data/indicators/{INDICATOR}/locations/all"
        f"?startYear={START_YEAR}&endYear={END_YEAR}"
        f"&variants={VARIANT_MEDIUM}&sexes={SEX_BOTH}&ages={AGE_15_PLUS_ID}"
        f"&pageNumber={page_number}&pageSize={PAGE_SIZE}&format=json"
    )


def _page_records(obj) -> list[dict]:
    """The data records from one page response. The Data Portal returns either an OBJECT
    ``{data, pages, pageNumber, nextPage, ...}`` or a bare ARRAY (per the OpenAPI); accept both and
    never treat a non-dict/non-list as empty."""
    if isinstance(obj, dict):
        if "data" not in obj:
            raise SystemExit(
                "WPP API page is a JSON object without a 'data' key — the response contract "
                f"changed; got keys {sorted(obj)}."
            )
        return obj["data"] or []
    if isinstance(obj, list):
        return obj
    raise SystemExit(f"WPP API page is neither object nor array (got {type(obj).__name__}).")


def _fetch_pages(token: str, max_pages: int = 10000):
    """Page EXPLICITLY by pageNumber until exhausted (review-14 P1#2): a page shorter than PAGE_SIZE
    (or empty) is the last one. We do NOT assume a bare array is the whole dataset — that would
    silently accept only the first 100 records as the global total."""
    import requests

    headers = {"Authorization": f"Bearer {token}"}
    rows: list[dict] = []
    for page in range(1, max_pages + 1):
        r = requests.get(_page_url(page), headers=headers, timeout=120)
        if r.status_code == 401:
            raise SystemExit(
                "WPP data API returned 401 — set WPP_API_TOKEN to a valid Bearer token "
                "(register at https://population.un.org/dataportal/about/dataapi)."
            )
        r.raise_for_status()
        records = _page_records(r.json())
        rows.extend(records)
        if len(records) < PAGE_SIZE:  # a short/empty page ⇒ no more data
            break
    else:
        raise SystemExit(
            f"WPP API pagination did not terminate within {max_pages} pages — aborting rather "
            "than looping unbounded."
        )
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


# The structural trajectory knots + the +1 years the CAGR derivation needs on either side.
_REQUIRED_YEARS = frozenset({2025, 2026, 2035, 2036, 2050, 2051})
_MIN_COUNTRIES = 150  # WPP 2024 covers ~237 locations; a full country set is well above this floor


def _validate_coverage(rows: list[dict]) -> None:
    """Fail BEFORE writing the CSV if pagination was truncated or a knot year is missing (review-14
    P1#2): the single most dangerous failure mode is silently accepting page 1 (100 records ≈ a
    handful of countries) as the whole world. Require a plausible country count and every knot year
    present for enough countries to be usable."""
    countries = {r["iso3"] for r in rows}
    if len(countries) < _MIN_COUNTRIES:
        raise SystemExit(
            f"WPP 15+ coverage looks truncated: only {len(countries)} countries "
            f"(< {_MIN_COUNTRIES} expected). Pagination likely stopped early — do NOT write a "
            "partial file."
        )
    years = {r["year"] for r in rows}
    missing_years = sorted(_REQUIRED_YEARS - years)
    if missing_years:
        raise SystemExit(
            f"WPP 15+ data is missing required knot year(s) {missing_years} — the trajectory "
            "CAGR derivation needs 2025/26, 2035/36, 2050/51."
        )


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
    parsed = _parse_records(records)
    _validate_coverage(parsed)  # refuse to write a truncated/partial file (review-14 P1#2)
    out = pd.DataFrame(parsed)
    out.to_csv(_OUT, index=False)
    n = out["iso3"].nunique()
    print(f"Wrote {_OUT} — {n} countries, years {out['year'].min()}–{out['year'].max()}.")
    print("Re-run scripts/extract_structural_sources.py to rebuild the digest with 15+ population.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
