#!/usr/bin/env python
"""Fetch the NGFS Phase 5 SECTORAL CO2 series (+ GDP) and vendor them as an IAMC-wide CSV.

    pip install pyam-iamc            # one-off; NOT a runtime dependency of this project
    python scripts/fetch_ngfs_sectoral.py

Pipeline step 1d-DATA. The originally-supplied `ngfs_phase5.csv` carried only economy-wide
`Emissions|CO2` + `GDP|PPP|...`, so a per-sector emissions-intensity split was impossible from it
(review-9 flagged 1d as data-blocked). This script pulls the sector-resolved CO2 series from the
IIASA NGFS Phase 5 Scenario Explorer (`ngfs_phase_5` connection, via `pyam`) for the pinned tuple
and writes them, with economy-wide CO2 + GDP, to `data/structural/sources/raw/ngfs_phase5.csv` in
the same IAMC WIDE (Model/Scenario/Region/Variable + year columns) layout the extractor already
reads — so `extract_ngfs_emissions` consumes the richer file unchanged.

The extractor forms a per-archetype emissions INTENSITY = CO2 / Final Energy (review-10 P1 — CO2 per
unit of the sector's physical final energy, NOT CO2/GDP, which double-counted the sector's
output-share change the CGE determines endogenously):
  __all__ = (Energy + Industrial Processes) CO2  / total Final Energy
  BRD     = (Industry demand + Industrial Processes) CO2 / Final Energy|Industry            (goods)
  MIL     = (Transport + Res/Comm) CO2 / (Final Energy|Transport + Res+Comm)     (services)
Energy Supply (power/refining) is upstream, serving all end uses, so its emissions sit in __all__,
not a single sector bundle. AFOLU is EXCLUDED everywhere — land use, net-negative mid-century (a
multiplicative intensity scale cannot carry it), and out of scope for this gross-production driver.
All three intensities are strictly positive and monotone-declining across the horizon.

This is an OFFLINE data-acquisition step (needs network + the IIASA explorer). The parsing +
aggregation is tested on synthetic fixtures in tests/test_structural_extractors.py; this script is
not imported by the runtime and pyam is not a project dependency.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "data" / "structural" / "sources" / "raw" / "ngfs_phase5.csv"

MODEL = "REMIND-MAgPIE 3.3-4.8"
SCENARIO = "Below 2°C"
REGION = "World"

# The exact series to vendor. Emissions intensities are E_s / FinalEnergy_s (review-10 P1) — CO2 per
# unit of the sector's PHYSICAL FINAL ENERGY, which isolates fuel-switching and nets out the
# energy-per-output term the CGE determines endogenously (E/GDP over-counted that). So we vendor the
# CO2 numerators AND the matching Final Energy denominators. GDP + total-CO2 for reference only;
# AFOLU is NOT fetched (land use, net-negative — out of scope for this gross-production driver).
VARIABLES = [
    # --- reference / back-compat (not the intensity driver) ---
    "Emissions|CO2",  # total economy-wide CO2 (incl AFOLU) — reference only
    "GDP|PPP|Counterfactual without damage",  # reference denominator
    "Population",
    # --- CO2 numerators (energy + industrial-process, gross production; NO AFOLU) ---
    "Emissions|CO2|Energy",  # __all__ numerator (with Industrial Processes)
    "Emissions|CO2|Industrial Processes",
    "Emissions|CO2|Energy|Demand|Industry",  # BRD numerator (with Industrial Processes)
    "Emissions|CO2|Energy|Demand|Transportation",  # MIL numerator (with Res+Comm)
    "Emissions|CO2|Energy|Demand|Residential and Commercial",
    # --- Final Energy activity denominators (the E_s/FE_s intensity basis) ---
    "Final Energy",  # __all__ denominator
    "Final Energy|Industry",  # BRD denominator
    "Final Energy|Transportation",  # MIL denominator (with Res+Comm)
    "Final Energy|Residential and Commercial",
]


def main() -> int:
    try:
        import pyam
    except ImportError:
        raise SystemExit(
            "pyam is not installed — run `pip install pyam-iamc` first (it is a one-off "
            "data-acquisition dependency, not a runtime dependency of this project)."
        ) from None
    conn = pyam.iiasa.Connection("ngfs_phase_5")
    df = conn.query(model=MODEL, scenario=SCENARIO, region=REGION, variable=VARIABLES)
    data = df.data
    present = set(data["variable"].unique())
    missing = [v for v in VARIABLES if v not in present]
    if missing:
        raise SystemExit(f"NGFS Phase 5 did not return expected variable(s): {missing}")
    # IAMC WIDE: one row per (Model, Scenario, Region, Variable, Unit), a column per year.
    wide = data.pivot_table(
        index=["model", "scenario", "region", "variable", "unit"],
        columns="year",
        values="value",
    ).reset_index()
    wide = wide.rename(
        columns={
            "model": "Model",
            "scenario": "Scenario",
            "region": "Region",
            "variable": "Variable",
            "unit": "Unit",
        }
    )
    wide.columns = [str(c) for c in wide.columns]
    wide.to_csv(_OUT, index=False)
    print(f"Wrote {_OUT} — {len(wide)} series, years {min(data['year'])}–{max(data['year'])}.")
    print("Re-run scripts/extract_structural_sources.py to rebuild the digest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
