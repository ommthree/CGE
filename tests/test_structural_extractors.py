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
    # PROPORTIONAL annual rate: expm1(annualised log-change) — the digest/wrapper compound (1+rate),
    # so the stored rate is proportional, not the log rate (review P3 2026-09-09).
    n_rate = math.expm1(math.log(1.10) / 9)
    s_rate = math.expm1(math.log(1.30) / 9)
    assert out["N"][2019] == pytest.approx(round(n_rate, 4))
    assert out["S"][2019] == pytest.approx(round(s_rate, 4))
    # world = GDP-weighted mean of the per-country PROPORTIONAL rates (weights 200 vs 100 at 2019);
    # NOT the (−huge) ZZZ default, nor a simple mean of the two buckets.
    world = (200 * n_rate + 100 * s_rate) / 300
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


def test_euklems_va_weights_mfp_lp_and_source_share_and_drops_capital_deepening():
    """Value-added-weighted aggregation over an archetype's industries. BRD={C,D}; va_weights 3 and
    1. The sourced ``mfp`` driver (LP1ConTFP, the per-hour TFP contribution) is VA-weighted:
    (3·0.006+0.004)/4. ``sector_productivity`` is the observed per-hour LP (LP1_G) for context.
    ``source_labour_shares`` is the VA-weighted labour share. capital_deepening is NO LONGER emitted
    (review P1 2026-09-09 — the LP2_G/CAP_QI decomposition was dimensionally wrong). The world
    ``__all__`` is VA-weighted over ALL members, not a mean of the archetypes."""
    df = pd.DataFrame(
        {
            "isic_section": ["C", "D", "G"],
            "mfp": [0.006, 0.004, 0.005],  # LP1ConTFP — the driver the wrapper uses
            "lp_growth": [0.02, 0.01, 0.008],  # LP1_G — observed per-hour LP, context only
            "labour_cost_share": [0.55, 0.60, 0.70],
            "va_weight": [3.0, 1.0, 5.0],
        }
    )
    out = extract_euklems(df, _IMAP, knot=2025)
    assert "capital_deepening" not in out
    assert out["mfp"]["BRD"][2025] == pytest.approx(round((3 * 0.006 + 0.004) / 4, 4))
    assert out["sector_productivity"]["BRD"][2025] == pytest.approx(round((3 * 0.02 + 0.01) / 4, 4))
    assert out["source_labour_shares"]["BRD"] == pytest.approx(round((3 * 0.55 + 0.60) / 4, 3))
    assert out["source_labour_shares"]["MIL"] == pytest.approx(0.70, abs=1e-9)  # single G industry
    # world __all__ over C,D,G VA-weighted (3,1,5), not mean of BRD/MIL:
    assert out["mfp"]["__all__"][2025] == pytest.approx(
        round((3 * 0.006 + 1 * 0.004 + 5 * 0.005) / 9, 4)
    )


# NGFS intensity is E_s / FinalEnergy_s (review-10 P1). spec = {arch: {co2:[...], activity:[...]}}.
def _spec(co2, activity):
    return {"co2": co2 if isinstance(co2, list) else [co2], "activity": activity}


def test_ngfs_emits_a_knot_per_source_year_and_reproduces_the_path():
    """A knot is emitted at every source year ≥ start_year, each the CAGR to the NEXT source year,
    so the piecewise-constant path reproduces the source E/FE intensity at every source year. CO2
    100→70→30, activity (final energy) flat at 100 → intensity 1.0→0.7→0.3; the 2030 knot spans
    2030→2040 (a decade). The final source year holds flat (rate 0.0)."""
    df = pd.DataFrame(
        {
            "model": ["M"] * 6,
            "scenario": ["NZ"] * 6,
            "region": ["World"] * 6,
            "variable": ["Emissions|CO2"] * 3 + ["Final Energy"] * 3,
            "year": [2025, 2030, 2040, 2025, 2030, 2040],
            "value": [100.0, 70.0, 30.0, 100.0, 100.0, 100.0],
        }
    )
    out = extract_ngfs_emissions(
        df,
        model="M",
        scenario="NZ",
        intensity_specs={"__all__": _spec("Emissions|CO2", ["Final Energy"])},
    )["__all__"]
    assert sorted(out) == [2025, 2030, 2040]  # a knot at every source year
    assert out[2025] == pytest.approx(round((0.70 / 1.0) ** (1 / 5) - 1, 4))  # 2025→2030
    assert out[2030] == pytest.approx(round((0.30 / 0.70) ** (1 / 10) - 1, 4))  # 2030→2040 (decade)
    assert out[2040] == 0.0  # final source year holds flat
    # holding piecewise-constant reproduces the source at 2040: 1.0·(1+r25)^5·(1+r30)^10 == 0.30
    assert (1 + out[2025]) ** 5 * (1 + out[2030]) ** 10 == pytest.approx(0.30, rel=1e-3)


def test_ngfs_per_sector_intensity_is_co2_over_sector_final_energy():
    """Each archetype's intensity = Σ(sector CO2) / Σ(sector Final Energy) — CO2 per unit sector
    energy, NOT CO2/GDP (review-10 P1). BRD CO2 A+B: 80→40, its activity FE_ind flat 100 → intensity
    halves. MIL CO2 C: 20→18, activity FE_tr+FE_rc = 40 flat → gentle. So BRD decarbonises faster,
    and each is independent of output share (activity, not GDP)."""
    rows = []

    def add(var, y2025, y2040):
        rows.extend(
            [
                {
                    "model": "M",
                    "scenario": "NZ",
                    "region": "World",
                    "variable": var,
                    "year": 2025,
                    "value": y2025,
                },
                {
                    "model": "M",
                    "scenario": "NZ",
                    "region": "World",
                    "variable": var,
                    "year": 2040,
                    "value": y2040,
                },
            ]
        )

    add("A", 60.0, 30.0)
    add("B", 20.0, 10.0)  # BRD CO2 = A+B: 80→40
    add("C", 20.0, 18.0)  # MIL CO2
    add("FE_ind", 100.0, 100.0)  # BRD activity (flat)
    add("FE_tr", 25.0, 25.0)
    add("FE_rc", 15.0, 15.0)  # MIL activity = 40 (flat)
    df = pd.DataFrame(rows)
    out = extract_ngfs_emissions(
        df,
        model="M",
        scenario="NZ",
        intensity_specs={
            "BRD": _spec(["A", "B"], ["FE_ind"]),
            "MIL": _spec(["C"], ["FE_tr", "FE_rc"]),
        },
    )
    assert set(out) == {"BRD", "MIL"}
    # BRD intensity 80/100→40/100 over 15y → CAGR (0.5)^(1/15)−1; MIL 20/40→18/40 → (0.9)^(1/15)−1.
    assert out["BRD"][2025] == pytest.approx(round((0.5) ** (1 / 15) - 1, 4))
    assert out["MIL"][2025] == pytest.approx(round((0.9) ** (1 / 15) - 1, 4))
    assert out["BRD"][2025] < out["MIL"][2025]  # goods decarbonise faster than services


def test_ngfs_missing_bundle_component_raises(monkeypatch):
    """review-10 P2#5: a configured component absent from the export must RAISE, not silently skip
    the archetype — else a future NGFS refresh could degrade to economy-wide with CI green."""
    df = pd.DataFrame(
        {
            "model": ["M"] * 4,
            "scenario": ["NZ"] * 4,
            "region": ["World"] * 4,
            "variable": ["Emissions|CO2", "Emissions|CO2", "Final Energy", "Final Energy"],
            "year": [2025, 2040, 2025, 2040],
            "value": [100.0, 50.0, 100.0, 100.0],
        }
    )
    with pytest.raises(ValueError, match="not present for the selected"):
        extract_ngfs_emissions(
            df,
            model="M",
            scenario="NZ",
            intensity_specs={
                "__all__": _spec("Emissions|CO2", ["Final Energy"]),
                "BRD": _spec(["MISSING_CO2"], ["Final Energy"]),  # component absent → raise
            },
        )


def test_ngfs_rejects_negative_emissions_and_bad_activity():
    """A multiplicative intensity SCALE cannot represent net-negative emissions or a zero/non-finite
    activity endpoint — the engine forbids non-positive scales. Each case raises before
    exponentiation, naming the offending year."""
    base = {
        "model": ["M"] * 4,
        "scenario": ["NZ"] * 4,
        "region": ["World"] * 4,
        "variable": ["Emissions|CO2", "Emissions|CO2", "Final Energy", "Final Energy"],
        "year": [2025, 2050, 2025, 2050],
    }
    spec = {"__all__": _spec("Emissions|CO2", ["Final Energy"])}
    neg = pd.DataFrame({**base, "value": [100.0, -20.0, 100.0, 100.0]})  # CO2 net-negative by 2050
    with pytest.raises(ValueError, match="intensity .* is non-positive"):
        extract_ngfs_emissions(neg, model="M", scenario="NZ", intensity_specs=spec)
    zero_a = pd.DataFrame({**base, "value": [100.0, 50.0, 100.0, 0.0]})  # activity → 0
    with pytest.raises(ValueError, match="non-positive"):
        extract_ngfs_emissions(zero_a, model="M", scenario="NZ", intensity_specs=spec)
    inf_a = pd.DataFrame({**base, "value": [100.0, 50.0, 100.0, float("inf")]})
    with pytest.raises(ValueError, match="non-finite"):
        extract_ngfs_emissions(inf_a, model="M", scenario="NZ", intensity_specs=spec)


def test_ngfs_excludes_wrong_region_rows_from_the_intensity():
    """A wrong-region row with a DIFFERENT value must not perturb the World intensity. World CO2
    100→50, activity flat; a Europe row with wildly different CO2 must be ignored."""
    df = pd.DataFrame(
        {
            "model": ["M"] * 5,
            "scenario": ["NZ"] * 5,
            "region": ["World", "World", "World", "World", "Europe"],
            "variable": [
                "Emissions|CO2",
                "Emissions|CO2",
                "Final Energy",
                "Final Energy",
                "Emissions|CO2",
            ],
            "year": [2025, 2040, 2025, 2040, 2025],
            "value": [100.0, 50.0, 100.0, 100.0, 5.0],  # Europe CO2=5 must not enter World
        }
    )
    out = extract_ngfs_emissions(
        df,
        model="M",
        scenario="NZ",
        region="World",
        intensity_specs={"__all__": _spec("Emissions|CO2", ["Final Energy"])},
    )["__all__"]
    assert out[2025] == pytest.approx(
        round((0.50 / 1.0) ** (1 / 15) - 1, 4)
    )  # unaffected by Europe


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
    # single knot 2025 (no successor) → the instantaneous one-year rate 1010/1000−1.
    out = extract_wpp_population(raw, {"USA": "N"}, knots=(2025,))
    assert out["N"][2025] == pytest.approx(0.01)


def test_wpp_knot_is_interval_cagr_that_reproduces_the_level():
    """Each knot's rate is the interval CAGR of the summed member total to the next knot, so holding
    it reproduces the WPP LEVEL at the next knot (review P2 2026-09-15). Two N countries; the N
    total goes 1500 (2025) → 1650 (2035), a decade → CAGR (1650/1500)^(1/10)−1; the __all__ over the
    same members reproduces the summed level. Aggregation is by SUMMING members (populous members
    dominate), not a mean of per-country rates."""
    raw = pd.DataFrame(
        {
            "iso3": ["USA", "USA", "DEU", "DEU"],
            "year": [2025, 2035, 2025, 2035],
            "population": [1000.0, 1100.0, 500.0, 550.0],  # N total 1500 → 1650 over 2025→2035
        }
    )
    out = extract_wpp_population(raw, {"USA": "N", "DEU": "N"}, knots=(2025, 2035))
    cagr = (1650.0 / 1500.0) ** (1 / 10) - 1
    assert out["N"][2025] == pytest.approx(round(cagr, 4))
    assert out["__all__"][2025] == pytest.approx(round(cagr, 4))
    # holding the 2025 rate for the decade reproduces the WPP level at 2035:
    assert 1500.0 * (1 + out["N"][2025]) ** 10 == pytest.approx(1650.0, rel=1e-3)


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
    normalises the capitalised headers, so extract_ngfs_emissions consumes it directly. CO2 100→60,
    GDP flat → intensity decline (60/100)^(1/15)−1."""
    wide = pd.DataFrame(
        {
            "Model": ["M", "M"],
            "Scenario": ["NZ", "NZ"],
            "Region": ["World", "World"],
            "Variable": ["Emissions|CO2", "Final Energy"],
            "2025": [100.0, 100.0],
            "2040": [60.0, 100.0],
        }
    )
    out = extract_ngfs_emissions(
        wide,
        model="M",
        scenario="NZ",
        intensity_specs={"__all__": _spec("Emissions|CO2", ["Final Energy"])},
    )
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


def test_ngfs_intensity_over_final_energy_nets_out_output_share():
    """The intensity is CO2 / Final Energy, NOT CO2 / GDP (review-10 P1). Even with GDP present and
    moving, the rate depends only on CO2 and activity (final energy). CO2 100→50, Final Energy flat
    → (0.5)^(1/15)−1; GDP doubling MUST NOT enter. Scenario/model still resolve strictly (the
    '(version: n)' suffix + °→? mangling are stripped)."""
    df = pd.DataFrame(
        {
            "Model": ["REMIND-MAgPIE 3.3-4.8"] * 6,
            "Scenario": ["Below 2?C (version: 1)"] * 6,
            "Region": ["World"] * 6,
            "Variable": [
                "Emissions|CO2",
                "Emissions|CO2",
                "Final Energy",
                "Final Energy",
                "GDP|PPP",
                "GDP|PPP",
            ],
            "year": [2025, 2040, 2025, 2040, 2025, 2040],
            "value": [100.0, 50.0, 100.0, 100.0, 100.0, 200.0],  # GDP doubles — must be ignored
        }
    )
    out = extract_ngfs_emissions(
        df,
        model="REMIND-MAgPIE 3.3-4.8",
        scenario="Below 2°C",
        intensity_specs={"__all__": _spec("Emissions|CO2", ["Final Energy"])},
    )
    # E/FE = 0.5 over 15y (GDP irrelevant); NOT E/GDP which would be 0.25.
    assert out["__all__"][2025] == pytest.approx(round((0.5) ** (1 / 15) - 1, 4))
    assert out["__all__"][2025] < 0  # decarbonisation


def test_ngfs_rejects_ambiguous_scenario():
    """The explicit tuple is strict: a scenario that normalises to two DISTINCT published names is
    ambiguous and rejected. Two raw scenarios 'Below 2°C (version: 1)' and '(version: 2)' both
    requested as 'Below 2°C' collapse to the same normalised key → not exactly one → reject."""
    df = pd.DataFrame(
        {
            "Model": ["M"] * 2,
            "Scenario": ["Below 2°C (version: 1)", "Below 2°C (version: 2)"],
            "Region": ["World"] * 2,
            "Variable": ["Emissions|CO2", "Emissions|CO2"],
            "year": [2025, 2025],
            "value": [100.0, 90.0],
        }
    )
    with pytest.raises(ValueError, match="did not resolve to exactly one"):
        extract_ngfs_emissions(
            df,
            model="M",
            scenario="Below 2°C",
            intensity_specs={"__all__": _spec("Emissions|CO2", ["Final Energy"])},
        )


def test_ngfs_raises_when_a_spec_variable_absent():
    """A configured co2/activity variable missing from the pinned scenario raises naming it, rather
    than silently emitting nothing (review-10 P2#5)."""
    df = pd.DataFrame(
        {
            "model": ["M"],
            "scenario": ["S"],
            "region": ["World"],
            "variable": ["Population"],  # neither CO2 nor Final Energy present
            "year": [2025],
            "value": [100.0],
        }
    )
    with pytest.raises(ValueError, match="not present for the selected"):
        extract_ngfs_emissions(
            df,
            model="M",
            scenario="S",
            intensity_specs={"__all__": _spec("Emissions|CO2", ["Final Energy"])},
        )


def test_load_euklems_workbook_reads_multisheet_layout(tmp_path):
    """The REAL EU KLEMS workbook has one sheet per variable (LP1ConTFP/LP1_G/LAB/VA_CP), each a
    matrix of nace_r2_code/geo_code/var + a column per year. load_euklems_workbook reads the
    section-letter rows and reduces them (review P1 2026-09-09): ``mfp`` = the per-hour TFP
    contribution LP1ConTFP (delta-log p.p.) → expm1(recent-window mean/100); ``lp_growth`` =
    VA growth LP1_G likewise; ``labour_cost_share`` = LAB/VA_CP; ``va_weight`` = VA_CP. NO CAP_QI /
    k_deepening (the old LP2_G/CAP_QI decomposition was dimensionally wrong)."""
    import math

    import openpyxl  # noqa: F401 — ensures the Excel engine is present

    def sheet(var, val19, val18):
        return pd.DataFrame(
            {
                "nace_r2_code": ["C", "C10-C12"],  # section + a detailed row (detailed is ignored)
                "geo_code": ["AT", "AT"],
                "var": [var, var],
                "2018": [val18, val18],
                "2019": [val19, val19],
            }
        )

    p = tmp_path / "euklems.xlsx"
    with pd.ExcelWriter(p) as w:
        sheet("LP1ConTFP", 0.6, 0.4).to_excel(w, sheet_name="LP1ConTFP", index=False)  # p.p. TFP
        sheet("LP1_G", 2.0, 1.0).to_excel(w, sheet_name="LP1_G", index=False)  # p.p. per-hour LP
        sheet("LAB", 55.0, 55.0).to_excel(w, sheet_name="LAB", index=False)
        sheet("VA_CP", 100.0, 100.0).to_excel(w, sheet_name="VA_CP", index=False)
    tidy = load_euklems_workbook(p, trend_window=2)
    row = tidy[tidy["isic_section"] == "C"].iloc[0]
    assert "k_deepening" not in tidy.columns
    assert row["mfp"] == pytest.approx(math.expm1((0.6 + 0.4) / 2 / 100))  # recent-mean p.p. → prop
    assert row["lp_growth"] == pytest.approx(math.expm1((2.0 + 1.0) / 2 / 100))
    assert row["labour_cost_share"] == pytest.approx(0.55)  # LAB/VA_CP
    assert row["va_weight"] == pytest.approx(100.0)
    assert set(tidy["isic_section"]) == {"C"}  # only the single-letter section row


def test_load_euklems_workbook_va_weights_across_geographies(tmp_path):
    """A multi-country workbook must VA-WEIGHT country growth per section, not equal-average it
    (review P2 2026-09-15). Section C in two countries: AT mfp 1.0 p.p. / VA 100, DE mfp 4.0 p.p. /
    VA 300 → VA-weighted mfp = (1·100+4·300)/400 = 3.25 p.p. (an equal average would give 2.5)."""
    import math

    import openpyxl  # noqa: F401

    def sheet(var, at_val, de_val):
        return pd.DataFrame(
            {
                "nace_r2_code": ["C", "C"],
                "geo_code": ["AT", "DE"],
                "var": [var, var],
                "2018": [at_val, de_val],
                "2019": [at_val, de_val],
            }
        )

    p = tmp_path / "euklems_multi.xlsx"
    with pd.ExcelWriter(p) as w:
        sheet("LP1ConTFP", 1.0, 4.0).to_excel(w, sheet_name="LP1ConTFP", index=False)
        sheet("LP1_G", 1.0, 4.0).to_excel(w, sheet_name="LP1_G", index=False)
        sheet("LAB", 50.0, 150.0).to_excel(w, sheet_name="LAB", index=False)  # share 0.5 both
        sheet("VA_CP", 100.0, 300.0).to_excel(w, sheet_name="VA_CP", index=False)  # weights 100/300
    tidy = load_euklems_workbook(p, trend_window=2)
    row = tidy[tidy["isic_section"] == "C"].iloc[0]
    assert row["mfp"] == pytest.approx(
        (math.expm1(1.0 / 100) * 100 + math.expm1(4.0 / 100) * 300) / 400
    )
    assert row["va_weight"] == pytest.approx(400.0)  # summed across geographies
