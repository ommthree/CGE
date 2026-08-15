"""Tests for the supply-share GENERATOR (scripts/build_supply_shares.py) — review P3 round 9.

These build a tiny SYNTHETIC MRSUT archive (a 2-region, few-product/industry supply.csv + meta.json)
so archive-metadata parsing, the mismatched-year guard, and audit-from-generated-file can be tested
without the multi-GB real MRSUT (which is gitignored and absent in CI)."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `scripts` importable
from scripts.build_supply_shares import (  # noqa: E402
    _read_meta,
    build_supply_shares,
    export_product_bridge_audit,
)

try:
    from cge.nature.real import encore_data_available

    _HAS_ENCORE = encore_data_available()
except Exception:  # pragma: no cover
    _HAS_ENCORE = False


def _tiny_mrsut(tmp_path: Path, *, year: int = 2019, vers: str = "20210125") -> str:
    """A minimal MRSUT_<year>.zip: supply.csv (product rows × industry cols, 2 regions) + meta."""
    regions = ["AA", "BB"]
    products = ["Prod X", "Prod Y"]
    industries = ["Ind P", "Ind Q"]
    row_idx = pd.MultiIndex.from_product([regions, products], names=["region", "sector"])
    col_idx = pd.MultiIndex.from_product([regions, industries], names=["region", "sector"])
    # Prod X made mostly by Ind P; Prod Y only by Ind Q.
    import numpy as np

    data = np.zeros((len(row_idx), len(col_idx)))
    for r in range(len(row_idx)):
        prod = row_idx[r][1]
        for c in range(len(col_idx)):
            ind = col_idx[c][1]
            if prod == "Prod X" and ind == "Ind P":
                data[r, c] = 90.0
            elif prod == "Prod X" and ind == "Ind Q":
                data[r, c] = 10.0
            elif prod == "Prod Y" and ind == "Ind Q":
                data[r, c] = 50.0
    supply = pd.DataFrame(data, index=row_idx, columns=col_idx)
    csv = supply.to_csv(sep="\t")
    zpath = tmp_path / f"MRSUT_{year}.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr(f"MRSUT_{year}/supply.csv", csv)
        z.writestr(
            f"MRSUT_{year}/meta.json",
            json.dumps({"year": year, "currency": "Euro", "price": "current", "vers": vers}),
        )
    return str(zpath)


def test_generator_reads_year_and_version_from_archive_meta(tmp_path):
    """The SUT year/version in the artifact come from the archive's meta.json, not a flag."""
    mrsut = _tiny_mrsut(tmp_path, year=2019, vers="20210125")
    with zipfile.ZipFile(mrsut) as z:
        assert _read_meta(z)["year"] == 2019
    out = tmp_path / "shares.json"
    art = build_supply_shares(mrsut, str(out))  # no --year passed
    prov = art["provenance"]
    assert prov["sut_year"] == 2019
    assert prov["sut_version"] == "20210125"
    # Prod X supply share: 90/100 Ind P, 10/100 Ind Q (summed over regions).
    assert art["shares"]["Prod X"]["Ind P"] == pytest.approx(0.9)
    assert art["shares"]["Prod X"]["Ind Q"] == pytest.approx(0.1)


def test_generator_rejects_year_disagreeing_with_archive(tmp_path):
    """--year that disagrees with the archive's meta.json year must raise, so a mislabelled output
    is impossible (the reviewer's `--year 2020` against MRSUT_2019 case, review P2 round 8)."""
    mrsut = _tiny_mrsut(tmp_path, year=2019)
    with pytest.raises(ValueError, match="disagrees with the MRSUT archive"):
        build_supply_shares(mrsut, str(tmp_path / "o.json"), year=2020)


def test_generator_year_crosscheck_passes_when_matching(tmp_path):
    """A matching --year is accepted (it is a cross-check, not a relabel)."""
    mrsut = _tiny_mrsut(tmp_path, year=2019)
    art = build_supply_shares(mrsut, str(tmp_path / "o.json"), year=2019)
    assert art["provenance"]["sut_year"] == 2019


@pytest.mark.skipif(not _HAS_ENCORE, reason="vendored ENCORE data (data/encore/) not present")
def test_export_product_bridge_audit_uses_the_supplied_shares_file(tmp_path):
    """export_product_bridge_audit must build the audit from the GIVEN shares file, not a hard-coded
    default (review P3 round 11 2026-08-15). Copy the real 2019 artifact to a temp path with a
    UNIQUE source_version marker; an implementation ignoring shares_path would not carry it."""
    marker = "UNIQUE-MARKER-r11-abc123"
    art = json.loads(Path("data/exiobase/supply_shares_2019.json").read_text())
    art["provenance"]["source_version"] = marker
    shares_path = tmp_path / "supply_shares_custom.json"
    shares_path.write_text(json.dumps(art))

    audit_out = tmp_path / "audit.json"
    export_product_bridge_audit(str(audit_out), shares_path=str(shares_path))
    data = json.loads(audit_out.read_text())
    # The audit's version came from the SUPPLIED file, proving shares_path was honoured.
    assert marker in data["supply_shares_version"]
    # And the bridge still resolved the real 2019 data: 184 supply-share + 16 fallbacks.
    assert data["n_supply_share"] == 184
    assert data["n_fallback"] == 16
    assert data["entries"]["Motor Gasoline"]["method"] == "supply-share"
