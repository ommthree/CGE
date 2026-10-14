# Raw source files for the structural-data extractors (pipeline step 1a-data)

Drop the downloaded source files here, then run `python scripts/extract_structural_sources.py`
to (re)build `data/structural/sources/inputs.json` from real data (replacing the illustrative
headline figures). The extractors parse the **actual published file formats** — see
`scripts/extract_structural_sources.py` for the exact sheet/column expectations and
`tests/test_structural_extractors.py` for byte-level fixtures.

**These raw files are NOT committed** (they are large and separately licensed); this directory is
git-ignored except for this README and the small `fixtures/` used by the tests. Each source below
lists the exact download and the filename the extractor expects.

## Sources, downloads, and expected filenames

Save each file in **this directory** (`data/structural/sources/raw/`) under the **exact expected
filename** below, then run `python scripts/extract_structural_sources.py`. The extractors accept the
**original published layout** — column renames, wide→long melts, and dropping region/aggregate rows
are handled automatically (`normalise_*` in the extractor), so you do **not** need to reshape by hand.

| Source | Download | Expected file | Licence |
|---|---|---|---|
| **PWT 10.01** (aggregate TFP, `rtfpna`) | <https://www.rug.nl/ggdc/productivity/pwt/> → "Download PWT 10.01" → the **Excel** workbook | `pwt1001.xlsx` (reads the `Data` sheet — the workbook's native sheet) | CC BY 4.0 |
| **UN WPP 2024** (total population growth) | <https://population.un.org/wpp/downloads> → **"Total Population - Both Sexes"** CSV, **Medium** variant | `wpp2024_population.csv` (raw `ISO3_code`/`Time`/`PopTotal`/`Variant` is fine) | CC BY 3.0 IGO |
| **UN WPP 2024** (15+ working-age population — labour identity, review-12 P1#3) | `pip install requests`; register for a free token at <https://population.un.org/dataportal/about/dataapi>; `export WPP_API_TOKEN=<token>`; `python scripts/fetch_wpp_by_age.py` (indicator 70, sums the 15+ bands) | `wpp2024_working_age.csv` (tidy `iso3`/`year`/`population`) — **OPTIONAL**: absent → labour falls back to total population with a documented caveat | CC BY 3.0 IGO |
| **ILOSTAT** (labour-force participation) | <https://ilostat.ilo.org/data/> → indicator **`EAP_DWAP_SEX_AGE_RT`** (LFPR) bulk CSV, or the ILOSTAT bulk-download portal | `ilostat_lfpr.csv` (raw `ref_area`/`time`/`obs_value`/`sex`/`classif1` is fine) | CC BY 4.0 |
| **EU KLEMS & INTANProd 2024** (sector MFP + per-hour LP + labour share) | <https://euklems-intanprod-llee.luiss.it/> → "Growth Accounts" release (registration/data agreement may apply) | `euklems_2023_growth_accounts.xlsx` — the supplied file is the **multi-sheet workbook** (one sheet per variable: `LP1ConTFP`/`LP1_G`/`LAB`/`VA_CP`), Austria only | CC BY 4.0 |
| **NGFS Phase 5** (per-channel emissions intensity) | Easiest: `pip install pyam-iamc` then `python scripts/fetch_ngfs_sectoral.py` (pulls each channel's CO₂ numerator + activity denominator for the pinned tuple). Or manually from <https://data.ece.iiasa.ac.at/ngfs/> → **model `REMIND-MAgPIE 3.3-4.8`**, **scenario `Below 2°C`**, region `World`, the CO₂ series (`Emissions\|CO2\|Energy\|Supply`, `\|Energy\|Supply\|Electricity`, `\|Energy\|Demand\|Industry`, `\|Industrial Processes`, `\|Energy\|Demand\|Transportation`, `\|Energy\|Demand\|Residential and Commercial`, plus `\|Energy`) **AND their activity denominators** (`Secondary Energy\|Electricity`, `Final Energy` + `\|Industry`/`\|Transportation`/`\|Residential and Commercial`, `Production\|Steel`, `Production\|Cement`) → Download CSV | `ngfs_phase5.csv` (IAMC long OR wide/year-columns both accepted) | see NGFS terms |

**Notes / caveats worth knowing before you download:**
- **PWT** is a clean drop-in (native `Data` sheet, `countrycode`/`year`/`rtfpna`).
- **WPP** — pick the *country-level* Total Population file (not the "by region" one); the extractor
  keeps only 3-letter-ISO3 country rows and the Medium variant, and needs consecutive years around
  each knot (2025/26, 2035/36, 2050/51 — WPP projects annually, so this is present).
- **EU KLEMS** — the supplied download is the **multi-sheet workbook** (one sheet per variable),
  read by `load_euklems_workbook`. The sourced MFP driver is **`LP1ConTFP`** (the workbook's own
  per-hour TFP contribution to VA growth, in delta-log p.p.), used directly as `mfp`; `LP1_G`
  (per-hour VA growth) is emitted as observed-LP context; `LAB`/`VA_CP` give the labour share and the
  VA aggregation weight. There is **no** capital-deepening / CAP_QI term (the earlier `LP2_G`/`CAP_QI`
  decomposition was dimensionally wrong — `LP2_G` is VA per *person*, not per hour). The file is
  **Austria only** (the sole geography present) — a single-country proxy for the archetypes, flagged
  as a breadth limitation in `docs/structural-data-pipeline-plan.md`. (The legacy long-format
  `normalise_euklems` path is kept for tidy exports but does not identify MFP.)
- **NGFS** — selection is an **explicit validated tuple** in `_NGFS`: model `REMIND-MAgPIE 3.3-4.8`,
  scenario `Below 2°C`, region `World`. Emissions intensity is built **per channel** (review-11 P1),
  each CO₂ over the activity that drives it: electricity = `CO2|Supply|Electricity / Secondary
  Energy|Electricity` (generation), industry = `CO2|Demand|Industry / FE|Industry` (combustion),
  process = `CO2|Industrial Processes / (Production|Steel + |Cement)`, transport / buildings over
  their final energy, `__all__` = (Energy + Industrial Processes) CO₂ / total Final Energy. This is
  **not CO₂/GDP** (double-counts the CGE-endogenous output-share change) and **not** one bundle over
  final energy (which mis-mapped electricity + divided process emissions by energy). Model/scenario
  are matched after stripping the `(version: n)` suffix and `°`→`?` mangling but must resolve to
  exactly one published value (raises on none/ambiguous); **every configured CO₂ and activity
  variable is MANDATORY** — a missing one **raises** (no silent degrade to economy-wide). KNOWN GAP:
  the FE/output (autonomous energy efficiency) term is not sourceable per sector from NGFS and the
  Leontief CGE does not reproduce it — the drivers are combustion/energy-emissions intensities, not
  full CO₂/output, hence **LOW confidence**. To change scenario/mapping, edit `_NGFS`.

After the files are in place, `python scripts/extract_structural_sources.py` rebuilds
`../inputs.json`; then `python scripts/build_structural_trajectories.py` regenerates
`../../trajectories_v1.json`. Both have a `--check` mode used by CI.

## Archetype mapping

Real countries/industries are mapped onto the model's development/composition **archetypes**
(`N` = advanced, `S` = emerging for regions; `BRD` = goods/broad-manufacturing, `MIL` = services/
mixed for sectors) by the committed concordances in `archetype_maps/`:

- `archetype_maps/country_to_archetype.csv` — ISO3 country → `N`/`S` (World Bank income groups).
- `archetype_maps/industry_to_archetype.csv` — source industry code → `BRD`/`MIL` (ISIC/EU KLEMS
  goods-vs-services split).

The extractor aggregates each source over the archetype using the documented per-driver weights
(population weights for population, labour-force weights for participation, output/GDP weights for
productivity and sector series). Missing weights fall back to an unweighted mean with a recorded
`confidence` downgrade.
