"""Tests for the structural-data extractors (pipeline step 1a-data).

The real PWT/WPP/EU KLEMS/NGFS files are large and separately licensed and are NOT committed, so the
parsing logic is exercised here on small SYNTHETIC frames in the published column layout. Each test
feeds a fixture DataFrame straight to the extractor (which takes an already-loaded frame), asserting
the archetype mapping, the growth-rate derivation, and the weighting are correct.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from scripts.extract_structural_sources import (
    country_archetype,
    extract_euklems,
    extract_ngfs_emissions,
    extract_pwt_productivity,
    extract_wpp_population,
    industry_archetype,
)

_CMAP = {"USA": "N", "DEU": "N", "CHN": "S", "IND": "S"}
_IMAP = {"C": "BRD", "D": "BRD", "G": "MIL", "J": "MIL"}


def test_archetype_maps_default_conservatively():
    assert country_archetype("USA", _CMAP) == "N"
    assert country_archetype("CHN", _CMAP) == "S"
    assert country_archetype("ZZZ", _CMAP) == "S"  # unlisted → developing default
    assert industry_archetype("C", _IMAP) == "BRD"
    assert industry_archetype("G", _IMAP) == "MIL"
    assert industry_archetype("Q", _IMAP) == "MIL"  # unlisted → services residual


def test_pwt_productivity_is_annualised_log_growth_of_rtfpna():
    """rtfpna is an index; the trend is the annualised log-change over the window, aggregated per
    archetype. USA index 1.0→1.10 over 2010–2019 → ln(1.1)/9; CHN 1.0→1.30 → ln(1.3)/9."""
    df = pd.DataFrame(
        {
            "countrycode": ["USA", "USA", "CHN", "CHN"],
            "year": [2010, 2019, 2010, 2019],
            "rtfpna": [1.0, 1.10, 1.0, 1.30],
        }
    )
    out = extract_pwt_productivity(df, _CMAP, window=(2010, 2019))
    assert out["N"][2019] == pytest.approx(round(math.log(1.10) / 9, 4))
    assert out["S"][2019] == pytest.approx(round(math.log(1.30) / 9, 4))
    assert "__all__" in out


def test_wpp_population_growth_is_population_weighted_over_archetype():
    """Population growth per country = pop[y+1]/pop[y]−1; the archetype rate is population-weighted.
    Two N countries: USA 1000→1010 (+1%), DEU 500→500 (0%); pop-wtd = (1000·0.01+500·0)/1500."""
    df = pd.DataFrame(
        {
            "iso3": ["USA", "USA", "DEU", "DEU"],
            "year": [2025, 2026, 2025, 2026],
            "population": [1000.0, 1010.0, 500.0, 500.0],
        }
    )
    out = extract_wpp_population(df, _CMAP, knots=(2025,))
    expected = round((1000 * 0.01 + 500 * 0.0) / 1500, 4)
    assert out["N"][2025] == pytest.approx(expected)


def test_euklems_aggregates_lp_deepening_and_source_share_by_va_weight():
    """Value-added-weighted aggregation over an archetype's industries. BRD={C,D}; va_weights
    3 and 1, LP {0.02,0.01} → (3·0.02+0.01)/4 = 0.0175; likewise deepening and the labour share."""
    df = pd.DataFrame(
        {
            "isic_section": ["C", "D", "G"],
            "lp_growth": [0.02, 0.01, 0.008],
            "k_deepening": [0.012, 0.008, 0.005],
            "labour_cost_share": [0.55, 0.60, 0.70],
            "va_weight": [3.0, 1.0, 5.0],
        }
    )
    out = extract_euklems(df, _IMAP, knot=2025)
    assert out["sector_productivity"]["BRD"][2025] == pytest.approx(round((3 * 0.02 + 0.01) / 4, 4))
    assert out["capital_deepening"]["BRD"][2025] == pytest.approx(round((3 * 0.012 + 0.008) / 4, 4))
    assert out["source_labour_shares"]["BRD"] == pytest.approx(round((3 * 0.55 + 0.60) / 4, 3))
    assert out["source_labour_shares"]["MIL"] == pytest.approx(0.70, abs=1e-9)  # single G industry


def test_ngfs_emissions_is_annualised_decline_of_the_pinned_scenario():
    """Filters to the pinned model/scenario + intensity variable, then annualises the decline over
    each knot interval. Intensity 100→60 over 2025→2040 → (60/100)^(1/15)−1 (negative)."""
    df = pd.DataFrame(
        {
            "model": ["M", "M", "OTHER"],
            "scenario": ["NZ", "NZ", "NZ"],
            "region": ["World", "World", "World"],
            "variable": [
                "Emissions|CO2 Intensity",
                "Emissions|CO2 Intensity",
                "Emissions|CO2 Intensity",
            ],
            "year": [2025, 2040, 2025],
            "value": [100.0, 60.0, 999.0],
        }
    )
    out = extract_ngfs_emissions(df, model="M", scenario="NZ", knots=(2025,))
    assert out["__all__"][2025] == pytest.approx(round((60 / 100) ** (1 / 15) - 1, 4))
    assert out["__all__"][2025] < 0  # decarbonisation


def test_extractor_no_op_when_no_raw_files(monkeypatch, capsys):
    """With no raw files present, ``main`` is a no-op that keeps the committed illustrative digest
    and says so (so a fresh checkout without downloads neither errors nor wipes the digest)."""
    import scripts.extract_structural_sources as ex

    monkeypatch.setattr(ex, "_RAW", ex._ROOT / "does_not_exist_raw")
    monkeypatch.setattr("sys.argv", ["extract_structural_sources.py"])
    rc = ex.main()
    assert rc == 0
    assert "No raw source files" in capsys.readouterr().out
