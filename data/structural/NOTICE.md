# Structural trajectories — sources & licences

`trajectories_v1.json` holds **documented, sourced per-region annual growth trajectories** for the
Phase 7b.2 structural drivers (population, labour-force participation, labour productivity). Each
`(driver, region)` path carries its own citation and confidence in the artifact itself; this file
records the underlying sources and their licences. The figures are **headline trend rates** taken
from the published sources below — a compact, review-friendly artifact, not a re-distribution of the
full source databases. On a real EXIOBASE build these region labels map to actual country/region
trajectories through the same loader (`cge.data.structural.load_structural_trajectories`).

## Population — UN World Population Prospects 2024

- **Source:** United Nations, Department of Economic and Social Affairs, Population Division —
  *World Population Prospects 2024*, Medium variant. <https://population.un.org/wpp/>
- **Retrieved:** 2026-08-16.
- **Licence:** CC BY 3.0 IGO.

## Labour-force participation — ILOSTAT / World Bank

- **Source:** ILO modelled estimates (ILOSTAT) and World Bank World Development Indicators —
  labour-force participation rate. <https://ilostat.ilo.org/>
- **Retrieved:** 2026-08-16.
- **Licence:** CC BY 4.0.

## Labour productivity / TFP — Penn World Table 10.01

- **Source:** Feenstra, Robert C., Robert Inklaar and Marcel P. Timmer (2015), "The Next Generation
  of the Penn World Table", *American Economic Review* 105(10), 3150-3182 — *Penn World Table
  version 10.01* (`rtfpna`, TFP growth at constant national prices).
  <https://www.rug.nl/ggdc/productivity/pwt/>
- **Retrieved:** 2026-08-16.
- **Licence:** CC BY 4.0.

## Reproducing / updating

The vendored artifact is the small derived object the code and tests consume. To refresh it against
a new data vintage, edit `trajectories_v1.json` with the updated headline rates and their citations
(bump `source_version` and `retrieved`), or point the loader at a build-specific artifact. The
`StructuralTrajectory` contract validates every entry on load (finite rates in a plausible band,
known drivers, per-entry source + confidence), so a malformed or unsourced trajectory fails loudly.
