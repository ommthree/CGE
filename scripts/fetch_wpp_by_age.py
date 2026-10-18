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
double/triple count (review-13 P1). We narrow the payload with the documented ``startAge=15`` filter
but the CORRECTNESS gate is the parse-side ageId/ageLabel guard (review-15 P1). Writes
``data/structural/sources/raw/wpp2024_working_age.csv`` (tidy ``iso3``/``year``/``population``, the
SAME shape ``normalise_wpp`` emits) so the extractor consumes it with the existing machinery.

API contract (per the UN Data Portal OpenAPI, review-14 P1#2 / review-15 P1):
- the endpoint path is ``/data/indicators/{ind}/locations/{ids}/start/{y}/end/{y}``; locations are
  explicit NUMERIC location IDs (there is no ``locations/all``, no ``ages`` query parameter — the
  documented age filters are ``startAge``/``endAge``);
- the location inventory (``/locations``) gives the canonical COUNTRY ids + ISO3 — we query only
  those (``locationTypeId`` 4), which also fixes mistaking 3-letter AGGREGATE codes (WLD/ASS/…) for
  countries (review-15 P2#2);
- responses page by an explicit ``pageNumber`` (``pageSize`` ≤ 100) as an object
  ``{data, nextPage, pages, …}`` or a bare array; ``_fetch_pages`` walks pageNumber until a short/
  empty page (never assumes one page is the whole dataset);
- the API throttles to ~5 requests / 10 s — we PACE requests and honour ``Retry-After`` with bounded
  429 retries (review-15 P1), else a 150-country × 28-year pull aborts at raise_for_status().

``_validate_coverage`` refuses to write a partial file: it checks EVERY required knot year has
enough real mapped countries (review-15 P2#2), not just a global count.

OFFLINE acquisition step (needs network + a token). The parsing/pagination/validation logic is pure
and unit-tested; only ``_get`` touches the network. The /data endpoints are auth-gated (HTTP 401
without a Bearer token) — if you cannot obtain a token, the labour driver falls back to total
population with a documented caveat.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "data" / "structural" / "sources" / "raw" / "wpp2024_working_age.csv"

API = "https://population.un.org/dataportalapi/api/v1"
INDICATOR = 70  # Population by age and sex — broad age groups
VARIANT_MEDIUM = 4  # WPP "Medium" variant id (variantId)
SEX_BOTH = 3  # both sexes combined (sexId)
AGE_15_PLUS_ID = 26  # the CUMULATIVE "15+" category in indicator 70 (NOT a 5-year band)
AGE_15_PLUS_LABEL = "15+"
COUNTRY_LOCATION_TYPE = 4  # WPP locationTypeId for a country/area (vs region/SDG-group aggregates)
PAGE_SIZE = 100  # the OpenAPI caps pageSize at 100 (review-14 P1#2) — page explicitly by pageNumber
START_YEAR, END_YEAR = 2024, 2051  # cover the structural knots (2025/2035/2050) + a +1 for growth
_MIN_INTERVAL_S = 2.1  # ≤5 requests / 10 s ⇒ space calls ≥2.0s apart (review-15 P1); 2.1 for margin
_MAX_429_RETRIES = 6
_LOCATIONS_PER_CALL = 50  # batch country ids per data call to stay well within URL limits
# The structural trajectory knots + the +1 years the CAGR derivation needs on either side.
_REQUIRED_YEARS = frozenset({2025, 2026, 2035, 2036, 2050, 2051})
_MIN_COUNTRIES = 150  # WPP 2024 covers ~237 locations; a full country set is well above this floor
_MIN_COUNTRIES_PER_YEAR = 150  # review-15 P2#2: EACH required year needs this many real countries


def _field(rec: dict, *names):
    """First present value among camelCase / PascalCase field spellings the portal has used."""
    for n in names:
        if n in rec and rec[n] is not None:
            return rec[n]
    return None


class _Throttle:
    """Pace requests to ≤1 per ``_MIN_INTERVAL_S`` (the UN API allows ~5 / 10 s)."""

    def __init__(self):
        self._last = 0.0

    def wait(self):
        gap = _MIN_INTERVAL_S - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


def _get(url: str, token: str, throttle: _Throttle):
    """A single paced GET with bounded 429 retries honouring Retry-After (review-15 P1)."""
    import requests

    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(_MAX_429_RETRIES + 1):
        throttle.wait()
        r = requests.get(url, headers=headers, timeout=120)
        if r.status_code == 401:
            raise SystemExit(
                "WPP data API returned 401 — set WPP_API_TOKEN to a valid Bearer token "
                "(register at https://population.un.org/dataportal/about/dataapi)."
            )
        if r.status_code == 429:
            if attempt == _MAX_429_RETRIES:
                raise SystemExit(
                    f"WPP data API kept returning 429 after {_MAX_429_RETRIES} retries — the rate "
                    "limit (≈5 req / 10 s) is tighter than expected; increase _MIN_INTERVAL_S."
                )
            retry_after = r.headers.get("Retry-After")
            delay = (
                float(retry_after) if (retry_after or "").strip().isdigit() else 2.0 * (attempt + 1)
            )
            time.sleep(delay)  # honour Retry-After, else linear backoff
            continue
        r.raise_for_status()
        return r.json()
    raise SystemExit("unreachable")  # pragma: no cover


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


def _country_inventory(token: str, throttle: _Throttle) -> dict[str, str]:
    """{iso3 → location id} for the real COUNTRIES/areas only (locationTypeId 4), from the
    ``/locations`` endpoint (review-15 P1+P2#2). This (a) gives the explicit numeric ids the data
    endpoint requires — there is no ``locations/all`` — and (b) is the authoritative country set, so
    3-letter AGGREGATE codes (WLD/ASS/ANZ/…) are never mistaken for countries."""
    import re

    out: dict[str, str] = {}
    page = 1
    while True:
        obj = _get(
            f"{API}/locations?pageNumber={page}&pageSize={PAGE_SIZE}&format=json", token, throttle
        )
        recs = _page_records(obj)
        for rec in recs:
            type_id = _field(rec, "locationTypeId", "LocationTypeId")
            if type_id is not None and int(type_id) != COUNTRY_LOCATION_TYPE:
                continue
            iso3 = _field(rec, "iso3", "Iso3", "iso3Code")
            loc_id = _field(rec, "id", "Id", "locationId", "LocationId")
            if isinstance(iso3, str) and re.fullmatch(r"[A-Za-z]{3}", iso3) and loc_id is not None:
                out[iso3.upper()] = str(loc_id)
        if len(recs) < PAGE_SIZE:
            break
        page += 1
    if len(out) < _MIN_COUNTRIES:
        raise SystemExit(
            f"WPP /locations returned only {len(out)} countries (locationTypeId "
            f"{COUNTRY_LOCATION_TYPE}) — expected ≥{_MIN_COUNTRIES}; inventory looks incomplete."
        )
    return out


def _data_url(location_ids: list[str], page_number: int) -> str:
    # Documented path form: /data/indicators/{ind}/locations/{csv ids}/start/{y}/end/{y}. startAge
    # narrows the payload to 15+-ish bands (the parse-side ageId guard is the real selector).
    ids = ",".join(location_ids)
    return (
        f"{API}/data/indicators/{INDICATOR}/locations/{ids}/start/{START_YEAR}/end/{END_YEAR}"
        f"?variants={VARIANT_MEDIUM}&sexes={SEX_BOTH}&startAge=15"
        f"&pageNumber={page_number}&pageSize={PAGE_SIZE}&format=json"
    )


def _fetch_pages(token: str, max_pages: int = 100000):
    """Fetch 15+ data for the real-country inventory, batched over location ids and paged by an
    explicit pageNumber within each batch (review-14 P1#2 / review-15 P1). Paced + 429-aware via
    ``_get``; never assumes a bare array is the whole dataset."""
    throttle = _Throttle()
    inventory = _country_inventory(token, throttle)
    ids = sorted(inventory.values(), key=int)
    rows: list[dict] = []
    for i in range(0, len(ids), _LOCATIONS_PER_CALL):
        batch = ids[i : i + _LOCATIONS_PER_CALL]
        for page in range(1, max_pages + 1):
            records = _page_records(_get(_data_url(batch, page), token, throttle))
            rows.extend(records)
            if len(records) < PAGE_SIZE:  # a short/empty page ⇒ no more data for this batch
                break
        else:
            raise SystemExit(
                f"WPP pagination for a location batch did not terminate within {max_pages} pages."
            )
    return rows, set(inventory)  # records + the authoritative country-ISO3 set (review-15 P2#2)


def _parse_records(records: list[dict], valid_iso3: set[str] | None = None) -> list[dict]:
    """Pure parse: keep ONLY real-country rows, Medium variant, both sexes, the cumulative 15+
    category; reject duplicate (iso3, year) observations. Returns tidy dicts iso3/year/population.

    Selection is by the EXPLICIT 15+ identity (ageId 26 OR ageLabel '15+'), never a sum of bands —
    indicator 70's overlapping 15-prefixed groups (15-24, 15-49, 15-64, …) would multiply-count.

    review-15 P2#2: a 3-letter code is NOT sufficient to be a country — real aggregates have 3-char
    ids too (WLD/ASS/ANZ/…). When ``valid_iso3`` (the /locations country inventory) is supplied, a
    row whose iso3 is not in it is dropped; absent an inventory we fall back to the shape check (and
    the caller's coverage guard still catches a degenerate panel)."""
    import re

    out: dict[tuple[str, int], float] = {}
    for rec in records:
        iso3 = _field(rec, "iso3", "Iso3", "iso3Code")
        if not isinstance(iso3, str) or not re.fullmatch(r"[A-Za-z]{3}", iso3):
            continue  # drop obvious non-country rows (non-3-letter ISO3)
        iso3 = iso3.upper()
        if valid_iso3 is not None and iso3 not in valid_iso3:
            continue  # a 3-letter AGGREGATE code (WLD/ASS/…) is not in the country inventory
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
        key = (iso3, int(float(year)))
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


def _validate_coverage(rows: list[dict]) -> None:
    """Fail BEFORE writing the CSV if the panel is truncated or incomplete (review-14 P1#2;
    HARDENED review-15 P2#2). The dangerous failure is accepting a degenerate panel — e.g. 149
    countries with only 2025 plus one country supplying the other years — as complete. So require
    EACH required knot year to independently have ≥ _MIN_COUNTRIES_PER_YEAR real countries, which
    also implies a full overall country set."""
    by_year: dict[int, set[str]] = {}
    for r in rows:
        by_year.setdefault(r["year"], set()).add(r["iso3"])
    missing_years = sorted(_REQUIRED_YEARS - set(by_year))
    if missing_years:
        raise SystemExit(
            f"WPP 15+ data is missing required knot year(s) {missing_years} — the trajectory "
            "CAGR derivation needs 2025/26, 2035/36, 2050/51."
        )
    thin = {
        y: len(by_year[y]) for y in _REQUIRED_YEARS if len(by_year[y]) < _MIN_COUNTRIES_PER_YEAR
    }
    if thin:
        raise SystemExit(
            f"WPP 15+ coverage is incomplete per year: {thin} (each required year needs "
            f"≥{_MIN_COUNTRIES_PER_YEAR} real countries). Pagination likely truncated — do NOT "
            "write a partial file."
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

    records, valid_iso3 = _fetch_pages(token)
    if not records:
        raise SystemExit("WPP API returned no rows for indicator 70.")
    parsed = _parse_records(records, valid_iso3=valid_iso3)  # inventory rejects aggregate codes
    _validate_coverage(parsed)  # refuse to write a truncated/partial file (review-15 P2#2)
    out = pd.DataFrame(parsed)
    out.to_csv(_OUT, index=False)
    n = out["iso3"].nunique()
    print(f"Wrote {_OUT} — {n} countries, years {out['year'].min()}–{out['year'].max()}.")
    print("Re-run scripts/extract_structural_sources.py to rebuild the digest with 15+ population.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
