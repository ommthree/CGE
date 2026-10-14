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
Medium variant, both sexes, summed over the 15+ broad-age bands per country-year. Writes
``data/structural/sources/raw/wpp2024_working_age.csv`` (tidy ``iso3``/``year``/``population``, the
SAME shape ``normalise_wpp`` emits) so the extractor consumes it with the existing machinery.

OFFLINE acquisition step (needs network + a token). Not imported by the runtime; parsing is tested
on fixtures. The /data endpoints are auth-gated (HTTP 401 without a Bearer token) — if you cannot
obtain a token, the labour driver falls back to total population with an explicit documented caveat.
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "data" / "structural" / "sources" / "raw" / "wpp2024_working_age.csv"

API = "https://population.un.org/dataportalapi/api/v1"
INDICATOR = 70  # Population by age and sex — broad age groups
VARIANT_MEDIUM = 4  # WPP "Medium" variant id
START_YEAR, END_YEAR = 2024, 2051  # cover the structural knots (2025/2035/2050) + a +1 for growth
# The 15+ broad-age bands in indicator 70 (sum = working-age-and-older population). WPP labels these
# as 5-year-ish broad groups; we keep every band whose lower bound is ≥ 15 and sum them.
_MIN_AGE = 15


def _fetch_all(token: str):
    import requests

    headers = {"Authorization": f"Bearer {token}"}
    url = (
        f"{API}/data/indicators/{INDICATOR}/locations/all/"
        f"start/{START_YEAR}/end/{END_YEAR}?variants={VARIANT_MEDIUM}&sexes=3&format=json"
    )
    rows: list[dict] = []
    while url:
        r = requests.get(url, headers=headers, timeout=120)
        if r.status_code == 401:
            raise SystemExit(
                "WPP data API returned 401 — set WPP_API_TOKEN to a valid Bearer token "
                "(register at https://population.un.org/dataportal/about/dataapi)."
            )
        r.raise_for_status()
        payload = r.json()
        rows.extend(payload.get("data", []))
        url = payload.get("nextPage")  # the portal paginates; follow until exhausted
    return rows


def _lower_age_bound(age_label: str) -> int | None:
    """Lower bound of a WPP broad-age label like '15-19', '65+', '100+'; None if not age-like."""
    import re

    m = re.match(r"\s*(\d+)", str(age_label))
    return int(m.group(1)) if m else None


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

    df = pd.DataFrame(_fetch_all(token))
    if df.empty:
        raise SystemExit("WPP API returned no rows for indicator 70.")
    # Keep country rows (3-letter ISO3), the 15+ broad-age bands, both sexes; sum bands per country.
    iso = df.get("iso3") if "iso3" in df else df.get("Iso3")
    age = df.get("ageLabel") if "ageLabel" in df else df.get("AgeLabel")
    df = df.assign(iso3=iso, ageLabel=age)
    df = df[df["iso3"].astype(str).str.fullmatch(r"[A-Za-z]{3}")]
    df["_lo"] = df["ageLabel"].map(_lower_age_bound)
    df = df[df["_lo"].notna() & (df["_lo"] >= _MIN_AGE)]
    out = (
        df.groupby(["iso3", "timeLabel" if "timeLabel" in df else "time"])["value"]
        .sum()
        .reset_index()
    )
    out.columns = ["iso3", "year", "population"]
    out["year"] = out["year"].astype(int)
    out.to_csv(_OUT, index=False)
    n = out["iso3"].nunique()
    print(f"Wrote {_OUT} — {n} countries, years {out['year'].min()}–{out['year'].max()}.")
    print("Re-run scripts/extract_structural_sources.py to rebuild the digest with 15+ population.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
