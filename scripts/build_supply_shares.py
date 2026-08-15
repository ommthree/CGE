"""Build the vendored EXIOBASE product→industry **supply-share** artifact from the raw MRSUT.

This is the *reproducible* generator for ``data/exiobase/supply_shares_2019.json`` — the observed
product→industry production shares the nature product-bridge uses instead of a code-prefix guess
(review P1-methodology round 7 2026-08-14). It is run once, offline, against the multi-GB EXIOBASE
MRSUT download; the small derived artifact is what the code and tests consume.

Input: the EXIOBASE 3 MRSUT ``supply.csv`` — **product rows × industry columns**, region-qualified
on both axes; cell V[product, industry] = monetary supply of the product by the industry.
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


def _read_meta(zf: zipfile.ZipFile) -> dict:
    """Read the archive's own ``meta.json`` (``{year, currency, price, vers}``). The archive is the
    source of truth for the year/version — never a caller-supplied --year (review P2 round 8)."""
    for name in zf.namelist():
        if name.endswith("meta.json"):
            with zf.open(name) as f:
                return json.load(f)
    raise FileNotFoundError(f"no meta.json in the MRSUT archive; has {zf.namelist()}")


def build_supply_shares(mrsut_zip: str, out_path: str, *, year: int | None = None) -> dict:
    """Derive and write the supply-share artifact. The SUT year/version are read from the archive's
    own ``meta.json``; ``year`` is an optional CROSS-CHECK — if it disagrees with the archive year,
    this raises (so ``--year 2020`` against an MRSUT_2019 archive cannot mislabel the output; review
    P2 round 8 2026-08-14)."""
    with zipfile.ZipFile(mrsut_zip) as z:
        meta = _read_meta(z)
        archive_year = int(meta["year"])
        if year is not None and year != archive_year:
            raise ValueError(
                f"--year {year} disagrees with the MRSUT archive's meta.json year {archive_year} "
                f"({mrsut_zip}). The archive is authoritative; drop --year or pass the right one."
            )
        with z.open(_supply_member(z)) as f:
            # Two-level headers on both axes: (region, sector). sector = INDUSTRY on columns,
            # PRODUCT on rows. Collapse regions on both axes by summing same-labelled rows/columns.
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

    sut_vers = meta.get("vers", "unknown")
    currency = meta.get("currency", "?")
    price = meta.get("price", "?")
    artifact = {
        "provenance": {
            "source": "EXIOBASE 3 MRSUT (monetary multi-regional supply-use table), supply matrix",
            "source_version": (
                f"EXIOBASE 3.8.2, MRSUT vers {sut_vers}, year {archive_year} ({price}-price "
                f"{currency})"
            ),
            "sut_year": archive_year,
            "sut_version": sut_vers,
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


def export_product_bridge_audit(out_path: str, shares_path: str) -> None:
    """Write the reviewable ``ProductBridgeAudit`` (per-product method / industry weights / fallback
    reason) to ``out_path``, built from the supply shares at ``shares_path`` — the artifact THIS run
    just generated, NOT the vendored default (review P2 round 8 2026-08-14: the audit must not go
    stale against a freshly-built --out). Requires the ENCORE data to be present."""
    from cge.nature.concordance_build import bridge_to_products, pxp_to_ixi_industries
    from cge.nature.real import load_supply_shares, real_encore_concordance

    shares, prov = load_supply_shares(path=shares_path)  # validates the just-built artifact too
    industry_conc, _ = real_encore_concordance()
    _cmap, _uncovered, audit = bridge_to_products(
        industry_conc,
        pxp_to_ixi_industries(),
        supply_shares=shares,
        supply_shares_version=prov.get("source_version", ""),
    )
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mrsut", default="downloads/exiobase/MRSUT_2019.zip")
    ap.add_argument("--out", default=None, help="default: data/exiobase/supply_shares_{year}.json")
    ap.add_argument("--audit-out", default=None, help="default: product_bridge_audit_{year}.json")
    ap.add_argument("--year", type=int, default=None, help="optional cross-check vs archive year")
    ap.add_argument(
        "--skip-audit",
        action="store_true",
        help="skip the product-bridge audit (else its failure is fatal, review P2 round 8)",
    )
    args = ap.parse_args()

    # The archive's meta.json is authoritative for the year; derive output paths from it so a --out
    # can't mislabel the file (review P2 round 8 2026-08-14).
    with zipfile.ZipFile(args.mrsut) as z:
        year = int(_read_meta(z)["year"])
    out = args.out or f"data/exiobase/supply_shares_{year}.json"
    audit_out = args.audit_out or f"data/exiobase/product_bridge_audit_{year}.json"

    art = build_supply_shares(args.mrsut, out, year=args.year)
    prov = art["provenance"]
    print(f"wrote {out}  (SUT year {prov['sut_year']}, vers {prov['sut_version']})")
    print(f"  products with supply: {prov['n_products_with_supply']}")
    print(f"  zero-supply products: {len(prov['zero_supply_products'])}")
    print(f"  size: {os.path.getsize(out) / 1e6:.2f} MB")

    if args.skip_audit:
        print("  (product-bridge audit skipped by --skip-audit)")
        return 0
    # The audit is built from THIS run's --out and its failure is FATAL — a stale/absent audit must
    # not slip through with exit 0 (review P2 round 8). Absent ENCORE data is the one soft case.
    from cge.nature.real import encore_data_available

    if not encore_data_available():
        print("  ERROR: ENCORE data absent — cannot build the product-bridge audit.")
        print("  Re-run with --skip-audit if you intend to build only the supply shares.")
        return 1
    export_product_bridge_audit(audit_out, shares_path=out)
    print(f"wrote {audit_out} ({os.path.getsize(audit_out) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
