# ENCORE knowledge base — source & licence

The files in this directory are a **subset of the ENCORE knowledge base**, redistributed here under
the terms of the licence below. They are **not** produced by this project.

## Attribution (required by the licence)

> **ENCORE** (Exploring Natural Capital Opportunities, Risks and Exposure). Developed by the ENCORE
> Partners: **UNEP-WCMC, UNEP FI, and Global Canopy.** Updated as part of the Horizon Europe project
> *"Strengthening Understanding and Strategies of Business to Assess and Integrate Nature (SUSTAIN)."*

- **Version vendored here:** *Updated ENCORE knowledge base — May 2026.*
- **Source:** <https://www.encorenature.org/> (registration-gated download).
- **Retrieved:** 2026-08-09.

## Licence — CC BY-SA 4.0

> UNEP grants permission to use the information contained in the ENCORE knowledge base under the terms
> of a **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** licence, strictly
> subject to attribution being provided to the ENCORE Partners alongside original sources.

Full licence text: <https://creativecommons.org/licenses/by-sa/4.0/>

**ShareAlike caveat.** CC BY-SA 4.0 is a copyleft (share-alike) licence: a work that *adapts* this
data may itself have to be offered under CC BY-SA 4.0. For this research prototype that is fine.
**Before any commercial or closed-source distribution of a product that embeds or adapts this data,
confirm with legal counsel that the ShareAlike obligation is acceptable.** This NOTICE is a
developer's summary, not legal advice.

## What is vendored (and what is not)

Only the files this project consumes, plus the crosswalk needed for a future ENCORE↔EXIOBASE
concordance, are included — not the full knowledge base:

| File | Used for |
|---|---|
| `ENCORE files/06. Dependency mat ratings.csv` | production-process × ecosystem-service **dependency** materiality ratings (the core input) |
| `ENCORE files/07. Pressure mat ratings.csv` | process × pressure/impact-driver ratings (ingested and typed separately; not yet consumed by an engine) |
| `ENCORE files/02. Ecosystem services definitions.csv` | authoritative ecosystem-service vocabulary + definitions |
| `ENCORE files/04. Pressure definitions.csv` | pressure/impact-driver definitions |
| `ENCORE files/17. Explanatory notes.csv` | modelling caveats (e.g. the water-supply double-counting note) |
| `Crosswalk tables/EXIOBASE - NACE Rev. 2 - ISIC Rev. 4 - ISIC Rev. 5.csv` | ISIC↔EXIOBASE bridge for the (deferred) real concordance |
| `Crosswalk tables/GICS - ENCORE production processes - ISIC .xlsx` | GICS/ENCORE-process ↔ ISIC bridge |

## Modelling caveats from ENCORE's own documentation

- **Materiality ratings are indicators of *potential* significance**, derived qualitatively / via
  blended assessment — they are **not** calibrated fractions of output or TFP loss. This project's
  `severity × score → productivity` mapping is therefore a **transparent scenario assumption**, not
  an empirical elasticity (see `docs/models/nature-encore.md`).
- **`ND` = "No Data"** is distinct from a rated dependency and from a blank ("not applicable / no
  dependency"). The adapter keeps `ND` as an explicit *unknown*, never silently zero.
- **Water supply double-counts** other water-related services in the SEEA-EA categorisation; ENCORE
  advises users to consider excluding it to avoid duplication (Explanatory note #1).
