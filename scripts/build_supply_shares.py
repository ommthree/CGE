"""Build the vendored EXIOBASE product→industry **supply-share** artifact from the raw MRSUT.

This is the *reproducible* generator for ``data/exiobase/supply_shares_2019.json`` — the observed
product→industry production shares the nature product-bridge uses instead of a code-prefix guess
(review P1-methodology round 7 2026-08-14). It is run once, offline, against the multi-GB EXIOBASE
MRSUT download; the small derived artifact is what the code and tests consume.

Input: the EXIOBASE 3 MRSUT ``supply.csv`` (industries × products, region-qualified on both axes;
cell V[product, industry] = monetary supply of the product by the industry).
Method: sum V over ALL regions → a global product × industry supply table; normalise each row
to shares summing to 1; drop shares < ``THRESHOLD`` and renormalise the kept subset (compact,
review-friendly, and immaterial to the averaged ENCORE weights). Products with no market supply
(recycling/treatment residuals, extra-territorial bodies) have an empty share and are recorded as
``zero_supply_products`` — the bridge falls back to the classification-prefix method for those.

Usage::

    python scripts/build_supply_shares.py \
        --mrsut downloads/exiobase/MRSUT_2019.zip \
        --out data/exiobase/supply_shares_2019.json
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile

import pandas as pd

THRESHOLD = 1e-4  # drop per-product industry shares below this, then renormalise
_MEMBER = "supply.csv"  # matched as a suffix so the MRSUT_<year>/ prefix is irrelevant


def _supply_member(zf: zipfile.ZipFile) -> str:
    for name in zf.namelist():
        if name.endswith(_MEMBER):
            return name
    raise FileNotFoundError(f"no {_MEMBER!r} in the MRSUT archive; has {zf.namelist()}")


def build_supply_shares(mrsut_zip: str, out_path: str, *, year: int = 2019) -> dict:
    with zipfile.ZipFile(mrsut_zip) as z, z.open(_supply_member(z)) as f:
        # Two-level headers on both axes: (region, sector). sector = INDUSTRY on columns, PRODUCT
        # on rows. Collapse regions on both axes by summing same-labelled rows/columns.
        df = pd.read_csv(f, sep="\t", header=[0, 1], index_col=[0, 1])
    df.columns = df.columns.get_level_values(1)
    df.index = df.index.get_level_values(1)
    prod_by_ind = df.groupby(level=0).sum().T.groupby(level=0).sum().T

    shares: dict[str, dict[str, float]] = {}
    zero_products: list[str] = []
    for product, row in prod_by_ind.iterrows():
        total = float(row.sum())
        if total <= 0:
            zero_products.append(str(product))
            continue
        kept = {ind: float(v) / total for ind, v in row.items() if v / total >= THRESHOLD}
        ssum = sum(kept.values())
        shares[str(product)] = {k: v / ssum for k, v in kept.items()}

    artifact = {
        "provenance": {
            "source": "EXIOBASE 3 MRSUT (monetary multi-regional supply-use table), supply matrix",
            "source_version": (
                f"EXIOBASE 3.8.2, MRSUT vers 20210125, year {year} (current-price Euro)"
            ),
            "doi": "10.5281/zenodo.5589597",
            "licence": "CC BY-SA 4.0 (EXIOBASE)",
            "retrieved": "2026-08-14",
            "method": (
                "product->industry production shares = sum over all regions of the supply matrix "
                f"V[product, industry], normalised per product to sum to 1; shares < {THRESHOLD:g} "
                "dropped and the remainder renormalised."
            ),
            "threshold": THRESHOLD,
            "n_products_with_supply": len(shares),
            "zero_supply_products": zero_products,
        },
        "shares": shares,
    }
    with open(out_path, "w") as fh:
        json.dump(artifact, fh, indent=0, sort_keys=True)
    return artifact


def export_product_bridge_audit(out_path: str) -> None:
    """Write the reviewable ``ProductBridgeAudit`` (per-product method / industry weights / fallback
    reason) to ``out_path``, using the vendored supply shares. Run after ``build_supply_shares`` and
    the ENCORE data are in place."""
    from cge.nature.real import real_encore_concordance_products

    _cmap, _uncovered, audit = real_encore_concordance_products(with_audit=True)
    out = {
        "summary": audit.summary(),
        "supply_shares_version": audit.supply_shares_version,
        "n_supply_share": audit.n_supply_share,
        "n_fallback": audit.n_fallback,
        "entries": {
            p: {
                "method": e.method,
                "fallback_reason": e.fallback_reason,
                "industry_weights": {
                    k: round(v, 6)
                    for k, v in sorted(e.industry_weights.items(), key=lambda kv: -kv[1])
                },
            }
            for p, e in sorted(audit.entries.items())
        },
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mrsut", default="downloads/exiobase/MRSUT_2019.zip")
    ap.add_argument("--out", default="data/exiobase/supply_shares_2019.json")
    ap.add_argument("--audit-out", default="data/exiobase/product_bridge_audit_2019.json")
    ap.add_argument("--year", type=int, default=2019)
    args = ap.parse_args()
    art = build_supply_shares(args.mrsut, args.out, year=args.year)
    prov = art["provenance"]
    print(f"wrote {args.out}")
    print(f"  products with supply: {prov['n_products_with_supply']}")
    print(f"  zero-supply products: {len(prov['zero_supply_products'])}")
    print(f"  size: {os.path.getsize(args.out) / 1e6:.2f} MB")
    try:
        export_product_bridge_audit(args.audit_out)
        print(f"wrote {args.audit_out} ({os.path.getsize(args.audit_out) / 1e6:.2f} MB)")
    except Exception as exc:  # ENCORE data may be absent in some checkouts
        print(f"  (skipped product-bridge audit: {exc})")


if __name__ == "__main__":
    main()
