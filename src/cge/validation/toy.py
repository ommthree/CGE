"""A hand-built 3-sector / 2-region toy economy with known structure.

Small enough to check engine outputs by hand. Numbers are illustrative, not real:
the point is a stable, analytically tractable fixture, per roadmap task P0.5.

Sectors: agriculture, energy, manufacturing. Regions: A, B.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cge.contracts.data_objects import (
    Classification,
    IOSystem,
    Provenance,
    SatelliteAccount,
)

SECTORS = ["agriculture", "energy", "manufacturing"]
REGIONS = ["A", "B"]


def _labels() -> list[str]:
    return [f"{r}:{s}" for r in REGIONS for s in SECTORS]


def toy_economy() -> tuple[IOSystem, SatelliteAccount]:
    """Return a small IOSystem + GHG SatelliteAccount with a fixed, documented structure.

    The A-matrix is deliberately mildly interconnected and column-sums < 1 (so the
    Leontief inverse exists and is positive). Energy is emissions-intensive, which makes
    it the natural stress channel for carbon-price tests.
    """
    labels = _labels()
    n = len(labels)
    prov = Provenance(
        source="toy",
        source_version="1",
        licence="none",
        reference_year=2020,
        retrieved="2026-07-17",
        notes="Hand-built 3-sector/2-region fixture for engine tests.",
    )

    # Technical coefficients: column j = inputs required per unit of sector j output.
    # Kept simple and symmetric across regions with a small inter-region trade block.
    rng = np.random.default_rng(0)  # only to jitter tiny off-diagonals reproducibly
    base = np.array(
        [
            [0.10, 0.05, 0.15],  # agriculture inputs
            [0.08, 0.12, 0.20],  # energy inputs (energy used by all)
            [0.05, 0.05, 0.10],  # manufacturing inputs
        ]
    )
    A = np.zeros((n, n))
    for ri in range(len(REGIONS)):
        for rj in range(len(REGIONS)):
            block = base * (1.0 if ri == rj else 0.15)  # weaker cross-region linkage
            block = block + (rng.random(block.shape) * 0.005 if ri != rj else 0.0)
            A[ri * 3 : ri * 3 + 3, rj * 3 : rj * 3 + 3] = block
    A_df = pd.DataFrame(A, index=labels, columns=labels)

    final_demand = pd.Series([100, 60, 120, 90, 50, 110], index=labels, name="final_demand")

    io = IOSystem(
        provenance=prov,
        sectors=Classification(name="toy-sectors", kind="sector", labels=SECTORS),
        regions=Classification(name="toy-regions", kind="region", labels=REGIONS),
        A=A_df,
        final_demand=final_demand.to_frame(),
    )

    # GHG intensities (tCO2 per unit output): energy dirty, manufacturing mid, ag low.
    intensity = pd.Series([0.2, 2.5, 0.6, 0.2, 2.5, 0.6], index=labels, name="CO2")
    sat = SatelliteAccount(
        provenance=prov,
        name="GHG",
        units={"CO2": "tCO2/unit"},
        data=intensity.to_frame().T,  # 1 row (CO2) × n columns
    )
    return io, sat
