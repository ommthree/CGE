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

The extractor forms a PER-CHANNEL emissions INTENSITY (review-11 P1), each CO2 over the activity
that actually drives it (not one BRD/MIL bundle over final energy):
  electricity : CO2|Supply|Electricity / Secondary Energy|Electricity   (generation intensity)
  industry    : CO2|Demand|Industry    / Final Energy|Industry          (fuel combustion)
  process     : CO2|Industrial Processes / (Production|Steel + |Cement) (physical output driven)
  transport   : CO2|Demand|Transportation / Final Energy|Transportation
  buildings   : CO2|Demand|Residential and Commercial / FE|Res+Comm
  __all__     : (Energy + Industrial Processes) CO2 / total Final Energy (fallback)
The concordance maps each model sector to its channel (electricity→electricity, transport→transport,
industry-like→industry+process, services/buildings→buildings), so electricity no longer inherits
the industry combustion path. KNOWN GAP (documented, not fabricated): the energy-per-output term
FE/output (autonomous energy efficiency) is not sourceable per sector from NGFS (no sector output/VA
published) and the default Leontief CGE does not reproduce it, so these drivers are combustion/
energy-emissions intensities, NOT the full CO2/output the engine multiplier nominally represents —
confidence LOW. AFOLU is EXCLUDED everywhere (land use, net-negative — not a multiplicative scale).
Electricity intensity turns net-negative ~2090 (grid with net CCS); the extractor holds the last
knot flat rather than emit a non-positive scale.

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

# The exact series to vendor. review-11 P1: emissions intensity is built PER NGFS CHANNEL, each
# CO2 over its OWN activity denominator matched to that channel's emissions driver (not one crude
# BRD/MIL bundle over final energy). We vendor each channel's CO2 numerator + activity denominator:
#   electricity : CO2|Energy|Supply|Electricity / Secondary Energy|Electricity  (generation)
#   industry    : CO2|Energy|Demand|Industry     / Final Energy|Industry         (combustion)
#   process     : CO2|Industrial Processes        / (Production|Steel + |Cement)  (physical output)
#   transport   : CO2|Energy|Demand|Transportation / Final Energy|Transportation
#   buildings   : CO2|Energy|Demand|Residential and Commercial / FE|Res+Comm
#   __all__     : (CO2|Energy + CO2|Industrial Processes) / total Final Energy    (fallback)
# The energy-efficiency term FE/output (AEEI) is NOT sourceable per sector from NGFS (no sector
# output/VA published) and the default Leontief CGE does not reproduce it — a documented known gap;
# the drivers are combustion/energy-emissions intensities, NOT full CO2/output. AFOLU stays out
# (land use, net-negative). GDP + total CO2 kept for reference only.
VARIABLES = [
    # --- reference only (not a driver) ---
    "Emissions|CO2",
    "GDP|PPP|Counterfactual without damage",
    "Population",
    # --- CO2 numerators per channel (energy + industrial-process; gross production; NO AFOLU) ---
    "Emissions|CO2|Energy",  # __all__ numerator (with Industrial Processes)
    "Emissions|CO2|Industrial Processes",  # process channel numerator
    "Emissions|CO2|Energy|Supply|Electricity",  # electricity channel numerator
    "Emissions|CO2|Energy|Demand|Industry",  # industry-combustion numerator
    "Emissions|CO2|Energy|Demand|Transportation",  # transport numerator
    "Emissions|CO2|Energy|Demand|Residential and Commercial",  # buildings numerator
    # --- activity denominators, each matched to its channel's driver ---
    "Final Energy",  # __all__
    "Secondary Energy|Electricity",  # electricity generation activity
    "Final Energy|Industry",  # industry combustion activity
    "Production|Steel",  # process physical activity (major process emitters)
    "Production|Cement",
    "Final Energy|Transportation",  # transport activity
    "Final Energy|Residential and Commercial",  # buildings activity
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
