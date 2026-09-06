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

| Source | Download | Expected file | Licence |
|---|---|---|---|
| **PWT 10.01** (aggregate TFP, `rtfpna`) | <https://www.rug.nl/ggdc/productivity/pwt/> → "Download PWT 10.01" (Stata/Excel) | `pwt1001.xlsx` (the `Data` sheet) | CC BY 4.0 |
| **UN WPP 2024** (population growth) | <https://population.un.org/wpp/downloads> → "Total Population" CSV (medium variant) | `wpp2024_population.csv` | CC BY 3.0 IGO |
| **ILO / World Bank** (participation) | ILOSTAT "Labour force participation rate" bulk CSV | `ilostat_lfpr.csv` | CC BY 4.0 |
| **EU KLEMS 2023** (sector LP + capital deepening + labour cost share) | <https://euklems-intanprod-llee.luiss.it/> → "Growth Accounts" release | `euklems_2023_growth_accounts.xlsx` | CC BY 4.0 |
| **NGFS Phase 5** (emissions intensity) | <https://data.ece.iiasa.ac.at/ngfs/> scenario explorer → download the pinned model/scenario/variable | `ngfs_phase5.csv` | see NGFS terms |

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
