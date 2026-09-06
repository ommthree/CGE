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
| **UN WPP 2024** (population growth) | <https://population.un.org/wpp/downloads> → **"Total Population - Both Sexes"** CSV, **Medium** variant | `wpp2024_population.csv` (raw `ISO3_code`/`Time`/`PopTotal`/`Variant` is fine) | CC BY 3.0 IGO |
| **ILOSTAT** (labour-force participation) | <https://ilostat.ilo.org/data/> → indicator **`EAP_DWAP_SEX_AGE_RT`** (LFPR) bulk CSV, or the ILOSTAT bulk-download portal | `ilostat_lfpr.csv` (raw `ref_area`/`time`/`obs_value`/`sex`/`classif1` is fine) | CC BY 4.0 |
| **EU KLEMS 2023** (sector LP + capital deepening + labour cost share) | <https://euklems-intanprod-llee.luiss.it/> → "Growth Accounts" release (registration/data agreement may apply) | `euklems_2023_growth_accounts.xlsx` (raw long `geo_code`/`nace_r2_code`/`var`/`year`/`value`) | CC BY 4.0 |
| **NGFS Phase 5** (emissions intensity) | <https://data.ece.iiasa.ac.at/ngfs/> → filter **model `REMIND-MAgPIE 3.4-4.8`**, **scenario `Net Zero 2050`**, an emissions-intensity variable, region `World` → Download CSV | `ngfs_phase5.csv` (IAMC long OR wide/year-columns both accepted) | see NGFS terms |

**Notes / caveats worth knowing before you download:**
- **PWT** is a clean drop-in (native `Data` sheet, `countrycode`/`year`/`rtfpna`).
- **WPP** — pick the *country-level* Total Population file (not the "by region" one); the extractor
  keeps only 3-letter-ISO3 country rows and the Medium variant, and needs consecutive years around
  each knot (2025/26, 2035/36, 2050/51 — WPP projects annually, so this is present).
- **EU KLEMS** is the fiddliest: the growth-accounts release is a long table with per-release
  **variable codes**. The extractor defaults to `VA_QI_growth` / `CAP_QI_growth` / `LAB_share` /
  `VA_CP`; if your download uses different codes it will **raise and list the codes it found** so you
  can pass the right mapping (edit the `var_codes=` default or tell me the codes and I'll set them).
  It also picks one country by default (the first present) — tell me if you want a specific one or a
  multi-country aggregate.
- **NGFS** — the pinned `model`/`scenario` are hard-coded in `_rebuild_digest`; if you download a
  different tuple, tell me and I'll update the constant (or edit it there).

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
