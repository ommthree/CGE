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
    extract_ilo_participation,
    extract_ngfs_emissions,
    extract_pwt_productivity,
    extract_wpp_population,
    industry_archetype,
    load_euklems_workbook,
    normalise_euklems,
    normalise_ilostat,
    normalise_wpp,
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


def test_pwt_productivity_is_gdp_weighted_annualised_log_growth():
    """rtfpna is an index; each country's trend is the annualised log-change over the window, and
    the archetype/world aggregate is GDP-WEIGHTED by rgdpo at the window end (2026-09-06). USA
    1.0→1.10 (N), CHN 1.0→1.30 (S), each alone in its bucket; the world figure is GDP-weighted over
    both. An unmapped country (ZZZ, not in _CMAP) is EXCLUDED, not defaulted to S."""
    df = pd.DataFrame(
        {
            "countrycode": ["USA", "USA", "CHN", "CHN", "ZZZ", "ZZZ"],
            "year": [2010, 2019, 2010, 2019, 2010, 2019],
            "rtfpna": [1.0, 1.10, 1.0, 1.30, 1.0, 0.5],  # ZZZ crashes but must be ignored
            "rgdpo": [200.0, 200.0, 100.0, 100.0, 999.0, 999.0],  # USA weight 2× CHN
        }
    )
    out = extract_pwt_productivity(df, _CMAP, window=(2010, 2019))
    assert out["N"][2019] == pytest.approx(round(math.log(1.10) / 9, 4))
    assert out["S"][2019] == pytest.approx(round(math.log(1.30) / 9, 4))
    # world = GDP-weighted (200·ln1.10 + 100·ln1.30)/(300·9); NOT the (−huge) ZZZ default, nor a
    # simple mean of the two buckets.
    world = (200 * math.log(1.10) + 100 * math.log(1.30)) / (300 * 9)
    assert out["__all__"][2019] == pytest.approx(round(world, 4))


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


# --- Raw-layout normalisers (accept the ACTUAL published downloads) -------------------------------


def test_normalise_wpp_maps_raw_columns_and_drops_aggregates():
    """The raw WPP export uses ISO3_code/Time/PopTotal, a Variant column, and mixes country rows
    with region aggregates (blank/non-3-letter ISO3). normalise_wpp yields tidy iso3/year/pop,
    Medium variant only, countries only — then extract_wpp_population works on it unchanged."""
    raw = pd.DataFrame(
        {
            "ISO3_code": ["USA", "USA", "", "USA"],  # blank row = a region aggregate → dropped
            "Time": [2025, 2026, 2025, 2025],
            "PopTotal": [1000.0, 1010.0, 9999.0, 500.0],
            "Variant": ["Medium", "Medium", "Medium", "High"],  # non-Medium → dropped
        }
    )
    tidy = normalise_wpp(raw)
    assert list(tidy.columns) == ["iso3", "year", "population"]
    assert set(tidy["iso3"]) == {"USA"}
    assert len(tidy) == 2  # the blank-ISO3 and the High-variant rows are gone
    out = extract_wpp_population(raw, {"USA": "N"}, knots=(2025,))
    assert out["N"][2025] == pytest.approx(0.01)


def test_normalise_euklems_pivots_long_export_to_isic_sections():
    """The raw EU KLEMS growth accounts are long (geo_code/nace_r2_code/var/year/value). normalise
    picks a country, keeps the latest year, pivots the growth-account variables, and maps NACE to
    its ISIC section. Two manufacturing industries (C10, C11) collapse to one VA-weighted row."""
    raw = pd.DataFrame(
        {
            "geo_code": ["DE"] * 8,
            "nace_r2_code": ["C10", "C10", "C10", "C10", "C11", "C11", "C11", "C11"],
            "var": ["VA_QI_growth", "CAP_QI_growth", "LAB_share", "VA_CP"] * 2,
            "year": [2020] * 8,
            "value": [0.02, 0.012, 0.55, 3.0, 0.01, 0.008, 0.60, 1.0],
        }
    )
    tidy = normalise_euklems(raw)
    assert set(tidy.columns) >= {"isic_section", "lp_growth", "k_deepening", "labour_cost_share"}
    row = tidy[tidy["isic_section"] == "C"].iloc[0]
    assert row["lp_growth"] == pytest.approx((3 * 0.02 + 1 * 0.01) / 4)  # VA-weighted
    assert row["labour_cost_share"] == pytest.approx((3 * 0.55 + 1 * 0.60) / 4)


def test_normalise_euklems_raises_on_missing_variable_codes():
    """If the release's variable codes differ, normalise_euklems raises naming the codes it found —
    an explicit failure, not silent empties."""
    raw = pd.DataFrame(
        {
            "geo_code": ["DE"],
            "nace_r2_code": ["C10"],
            "var": ["SOME_OTHER_CODE"],
            "year": [2020],
            "value": [0.02],
        }
    )
    with pytest.raises(ValueError, match="missing expected variable code"):
        normalise_euklems(raw)


def test_normalise_ngfs_melts_iamc_wide_format():
    """The IIASA IAMC WIDE export has a column per year; normalise_ngfs melts it to long and
    normalises the capitalised headers, so extract_ngfs_emissions consumes it directly."""
    wide = pd.DataFrame(
        {
            "Model": ["M"],
            "Scenario": ["NZ"],
            "Region": ["World"],
            "Variable": ["Emissions|CO2 Intensity"],
            "2025": [100.0],
            "2040": [60.0],
        }
    )
    out = extract_ngfs_emissions(wide, model="M", scenario="NZ", knots=(2025,))
    assert out["__all__"][2025] == pytest.approx(round((60 / 100) ** (1 / 15) - 1, 4))


def test_normalise_ilostat_keeps_only_total_sex_and_15plus_band():
    """normalise_ilostat keeps ONLY the SEX_T total and the 15+ aggregate band
    (AGE_AGGREGATE_YGE15 — ILOSTAT has no _TOTAL age band), so overlapping bands / other sexes on
    the same country-year are not double-counted (review 2026-09-06)."""
    raw = pd.DataFrame(
        {
            "ref_area": ["USA", "USA", "USA"],
            "time": [2020, 2020, 2020],
            "obs_value": [60.0, 40.0, 55.0],
            "sex": ["SEX_T", "SEX_M", "SEX_T"],  # SEX_M dropped
            "classif1": ["AGE_AGGREGATE_YGE15", "AGE_AGGREGATE_YGE15", "AGE_AGGREGATE_Y15-24"],
        }
    )
    tidy = normalise_ilostat(raw)
    assert list(tidy.columns) == ["iso3", "year", "lfpr"]
    assert len(tidy) == 1 and float(tidy["lfpr"].iloc[0]) == 60.0  # only SEX_T + 15+ band


def test_ilo_participation_is_recent_trend_held_forward():
    """extract_ilo_participation computes each country's annualised log-trend of the LFPR over the
    recent window, then HOLDS it forward at every knot (review 2026-09-06 — more stable than a
    single year-on-year step frozen). USA rate 58→60 over 2018→2023 → ln(60/58)/5 per year, held at
    knots. Mapped countries only."""
    years = list(range(2018, 2024))
    rate = [58.0, 58.5, 59.0, 59.3, 59.7, 60.0]
    raw = pd.DataFrame(
        {
            "ref_area": ["USA"] * 6,
            "time": years,
            "obs_value": rate,
            "sex": ["SEX_T"] * 6,
            "classif1": ["AGE_AGGREGATE_YGE15"] * 6,
        }
    )
    out = extract_ilo_participation(raw, {"USA": "N"}, knots=(2025, 2040))
    expected = round(math.log(60.0 / 58.0) / 5, 4)
    assert out["N"][2025] == pytest.approx(expected)
    assert out["N"][2040] == pytest.approx(expected)  # single recent trend held forward


def test_ngfs_derives_intensity_from_co2_over_gdp():
    """When the NGFS export has no ready-made intensity variable, extract_ngfs_emissions derives it
    as Emissions|CO2 / GDP per year (review 2026-09-07). Model/scenario match by SUBSTRING so
    'Below 2' matches 'Below 2?C (version: 1)'. CO2 100→50 and GDP 100→200 over 2025→2040 →
    intensity 1.0→0.25 → annualised (0.25)^(1/15)−1."""
    df = pd.DataFrame(
        {
            "Model": ["REMIND-MAgPIE 3.3-4.8"] * 4,
            "Scenario": ["Below 2?C (version: 1)"] * 4,
            "Region": ["World"] * 4,
            "Variable": ["Emissions|CO2", "Emissions|CO2", "GDP|PPP", "GDP|PPP"],
            "2025": [100.0, None, 100.0, None],
            "2040": [None, 50.0, None, 200.0],
        }
    )
    # melt-friendly: one row per (var, year); build long directly to avoid None cells
    df = pd.DataFrame(
        {
            "Model": ["REMIND-MAgPIE 3.3-4.8"] * 4,
            "Scenario": ["Below 2?C (version: 1)"] * 4,
            "Region": ["World"] * 4,
            "Variable": ["Emissions|CO2", "Emissions|CO2", "GDP|PPP", "GDP|PPP"],
            "year": [2025, 2040, 2025, 2040],
            "value": [100.0, 50.0, 100.0, 200.0],
        }
    )
    out = extract_ngfs_emissions(
        df, model="REMIND-MAgPIE 3.3-4.8", scenario="Below 2", knots=(2025,)
    )
    intensity_decline = (0.25) ** (1 / 15) - 1  # (50/200)/(100/100) = 0.25 over 15y
    assert out["__all__"][2025] == pytest.approx(round(intensity_decline, 4))
    assert out["__all__"][2025] < 0  # decarbonisation


def test_ngfs_raises_when_neither_intensity_nor_co2_and_gdp():
    """If the pinned scenario has neither an intensity variable nor both CO2 and GDP, the derivation
    raises rather than silently emitting nothing."""
    df = pd.DataFrame(
        {
            "model": ["M"],
            "scenario": ["S"],
            "region": ["World"],
            "variable": ["Population"],
            "year": [2025],
            "value": [100.0],
        }
    )
    with pytest.raises(ValueError, match="cannot derive"):
        extract_ngfs_emissions(df, model="M", scenario="S", knots=(2025,))


def test_load_euklems_workbook_reads_multisheet_layout(tmp_path):
    """The REAL EU KLEMS workbook has one sheet per variable (LP2_G/CAP_QI/LAB/VA_CP), each a matrix
    of nace_r2_code/geo_code/var + a column per year. load_euklems_workbook reads the section-letter
    rows and reduces them: LP2_G mean over the recent window /100; CAP_QI annualised index growth;
    labour_cost_share = LAB/VA_CP; va_weight = VA_CP (review 2026-09-07)."""
    import openpyxl  # noqa: F401 — ensures the Excel engine is present

    def sheet(var, val19, val18=None):
        return pd.DataFrame(
            {
                "nace_r2_code": ["C", "C10-C12"],  # section + a detailed row (detailed is ignored)
                "geo_code": ["AT", "AT"],
                "var": [var, var],
                "2018": [val18 if val18 is not None else val19, val18 or val19],
                "2019": [val19, val19],
            }
        )

    p = tmp_path / "euklems.xlsx"
    with pd.ExcelWriter(p) as w:
        sheet("LP2_G", 2.0, 1.0).to_excel(
            w, sheet_name="LP2_G", index=False
        )  # % → mean 1.5 → 0.015
        sheet("CAP_QI", 110.0, 100.0).to_excel(w, sheet_name="CAP_QI", index=False)  # +10%/1yr
        sheet("LAB", 55.0).to_excel(w, sheet_name="LAB", index=False)
        sheet("VA_CP", 100.0).to_excel(w, sheet_name="VA_CP", index=False)
    tidy = load_euklems_workbook(p, trend_window=2)
    row = tidy[tidy["isic_section"] == "C"].iloc[0]
    assert row["lp_growth"] == pytest.approx((2.0 + 1.0) / 2 / 100)  # mean of recent %, /100
    assert row["k_deepening"] == pytest.approx(110.0 / 100.0 - 1.0)  # 1-year index growth
    assert row["labour_cost_share"] == pytest.approx(0.55)  # LAB/VA_CP
    assert row["va_weight"] == pytest.approx(100.0)
    assert set(tidy["isic_section"]) == {"C"}  # only the single-letter section row
