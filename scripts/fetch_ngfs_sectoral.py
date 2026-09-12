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

The BRD (goods) / MIL (services) archetype bundles the extractor aggregates from these series are:
  BRD = Industry energy demand + Industrial Processes + Energy Supply  (goods production + power)
  MIL = Transportation + Residential and Commercial                    (services + mobility)
AFOLU is deliberately EXCLUDED — it is land use, not goods/services production, and it goes
net-negative mid-century (a multiplicative intensity scale cannot carry that). The two bundles cover
~90-100% of economy-wide energy+process CO2 and stay strictly positive across the horizon.

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

# The exact series to vendor: economy-wide CO2 + GDP (the aggregate path) plus the sector components
# the extractor bundles into BRD/MIL. Population is kept for continuity with the prior file.
VARIABLES = [
    "Emissions|CO2",  # economy-wide (the __all__ path)
    "GDP|PPP|Counterfactual without damage",  # the intensity denominator
    "Population",
    # BRD (goods) bundle:
    "Emissions|CO2|Energy|Demand|Industry",
    "Emissions|CO2|Industrial Processes",
    "Emissions|CO2|Energy|Supply",
    # MIL (services) bundle:
    "Emissions|CO2|Energy|Demand|Transportation",
    "Emissions|CO2|Energy|Demand|Residential and Commercial",
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
