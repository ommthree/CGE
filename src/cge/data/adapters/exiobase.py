"""EXIOBASE 3 adapter (tasks 1.1 + 1.2).

Three layers:

- ``fetch_exiobase`` — download from the **live Zenodo source** via pymrio, with version
  pinning (DOI), a target year, and caching (skips if already present). This is the real
  data path (roadmap P1.1).
- ``parse_exiobase`` — parse a downloaded EXIOBASE folder into a pymrio ``IOSystem``.
- ``adapt_pymrio`` — map a parsed pymrio system into our harmonised ``IOSystem`` +
  ``SatelliteAccount``s. Source-format knowledge stops here.

``load_exiobase_test`` returns pymrio's tiny bundled test MRIO through the same adapter,
so the whole pipeline is exercised offline in CI without a multi-GB download.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pymrio

from cge.contracts.data_objects import (
    Classification,
    IOSystem,
    Provenance,
    SatelliteAccount,
)
from cge.data.metadata import GWP100_AR5

# EXIOBASE 3.8.2 on Zenodo. Pin the DOI so a build is reproducible; bump deliberately.
EXIOBASE_DOI = "10.5281/zenodo.5589597"
EXIOBASE_LICENCE = "CC BY-SA 4.0"


def _labels(index: pd.MultiIndex) -> list[str]:
    """pymrio (region, sector) MultiIndex -> our 'region:sector' flat labels."""
    return [f"{region}:{sector}" for region, sector in index]


def fetch_exiobase(
    storage_folder: str | Path,
    *,
    year: int = 2019,
    system: str = "pxp",
    doi: str = EXIOBASE_DOI,
    overwrite: bool = False,
) -> Path:
    """Download EXIOBASE 3 for ``year`` from the live Zenodo record into ``storage_folder``.

    ``system='pxp'`` selects product-by-product tables (roadmap decision). Returns the path
    to the **downloaded archive for this year** — the file ``parse_exiobase3`` expects, not
    the parent directory (which pymrio would not parse). Large (multi-GB); cached — re-runs
    skip existing files.
    """
    storage = Path(storage_folder)
    storage.mkdir(parents=True, exist_ok=True)
    pymrio.download_exiobase3(
        storage_folder=str(storage),
        years=[year],
        system=system,
        doi=doi,
        overwrite_existing=overwrite,
    )
    # pymrio writes one archive per (year, system), e.g. 'IOT_2019_pxp.zip'. Require the
    # SYSTEM to match — do NOT fall back to any archive for the year, or a request for pxp
    # could silently return an ixi (industry-by-industry) file (review).
    candidates = sorted(storage.glob(f"*{year}*{system}*.zip"))
    if not candidates:
        others = [p.name for p in storage.glob(f"*{year}*.zip")]
        raise FileNotFoundError(
            f"No EXIOBASE archive matching year {year}, system {system!r} in {storage}. "
            f"Other {year} archives present: {others or 'none'}. "
            f"Refusing to guess a different system."
        )
    return candidates[0]


def parse_exiobase(path: str | Path) -> pymrio.IOSystem:
    """Parse a downloaded EXIOBASE folder/zip into a pymrio IOSystem (coefficients computed)."""
    io = pymrio.parse_exiobase3(path=str(path))
    return io.calc_all()


def load_exiobase_test() -> pymrio.IOSystem:
    """pymrio's bundled small test MRIO (offline). Same shape as real EXIOBASE, tiny."""
    return pymrio.load_test().calc_all()


# Extension folder / attribute names that carry emission stressors, across EXIOBASE
# versions: 3.8.x uses 'emissions' in some products and 'satellite' in others; 'ghg' as a
# fallback. Order matters only for logging; all matching extensions are scanned.
_EMISSION_EXT_HINTS = ("satellite", "emiss", "ghg")

# Convert a stressor's native mass unit to tonnes. Carbon costs are €/tonne, so every gas
# flow must be normalised to tonnes before applying a carbon price (units defect fix).
_MASS_TO_TONNE = {
    "kg": 1e-3,
    "t": 1.0,
    "tonnes": 1.0,
    "tonne": 1.0,
    "Mt": 1e6,
    "kt": 1e3,
    "g": 1e-6,
}


def _emission_extension_names(pio: pymrio.IOSystem) -> list[str]:
    ext = list(pio.get_extensions())
    return [e for e in ext if any(h in e.lower() for h in _EMISSION_EXT_HINTS)]


def _stressor_unit_to_tonne(ext, stressor) -> float:
    """Return the multiplier taking this stressor's native unit to tonnes.

    Reads the extension's ``unit`` metadata (pymrio carries it). Raises if the unit is
    unrecognised — silently guessing a mass unit is exactly the class of error this fix
    exists to prevent.
    """
    unit_df = getattr(ext, "unit", None)
    if unit_df is None:
        raise ValueError(f"extension has no unit metadata; cannot convert stressor {stressor!r}")
    raw = unit_df.loc[stressor]
    unit = str(raw.iloc[0] if hasattr(raw, "iloc") else raw).strip()
    if unit not in _MASS_TO_TONNE:
        raise ValueError(
            f"unrecognised emission unit {unit!r} for stressor {stressor!r}; "
            f"known: {sorted(_MASS_TO_TONNE)}"
        )
    return _MASS_TO_TONNE[unit]


def _ghg_satellite(
    pio: pymrio.IOSystem,
    labels: list[str],
    provenance: Provenance,
    gas_aliases: dict[str, str] | None = None,
    monetary_unit: str = "MEUR",
) -> SatelliteAccount | None:
    """Build a GHG SatelliteAccount of **emission intensities in tonnes per M€ output**.

    For each GHG (CO2, CH4, N2O) matching stressors' flows are unit-normalised to tonnes
    (via the extension's ``unit`` metadata), summed, and divided by gross output. A combined
    ``CO2e`` row is added via GWP-100 (AR5). Per-gas rows are kept so the engine can honour a
    scenario's ``gases`` selection. Returns None if no emission extension/stressors are found.

    Units are **t/MEUR** (CO2e: **tCO2e/MEUR**). Because output is in M€, a carbon price in
    €/t applied to these intensities must still be scaled by 1e-6 (M€→€) to yield a
    dimensionless cost share — done in the engine, documented there.

    ``gas_aliases`` maps a substring of a stressor name to a gas symbol so the tiny pymrio
    test system (stressors ``emission_type1/2``) can stand in for real gases offline.
    """
    import numpy as np

    ext_names = _emission_extension_names(pio)
    if not ext_names:
        return None

    x = pio.x["indout"] if getattr(pio, "x", None) is not None else None
    if x is None:
        return None
    x_vec = x.reindex(pio.A.columns).to_numpy(dtype=float)

    def _matches(stressor, gas: str) -> bool:
        s = str(stressor[0] if isinstance(stressor, tuple) else stressor)
        if s.upper().startswith(gas):  # real EXIOBASE, e.g. 'CO2 - combustion - air'
            return True
        if gas_aliases:
            return any(sub in s and alias == gas for sub, alias in gas_aliases.items())
        return False

    # Accumulate tonne-normalised totals per gas across all emission extensions.
    gas_totals: dict[str, np.ndarray] = {}
    for ext_name in ext_names:
        ext = getattr(pio, ext_name)
        F = ext.F
        for gas in GWP100_AR5:
            matched = [s for s in F.index if _matches(s, gas)]
            if not matched:
                continue
            acc = gas_totals.setdefault(gas, np.zeros(len(labels)))
            for stressor in matched:
                factor = _stressor_unit_to_tonne(ext, stressor)
                flow = F.loc[[stressor]].sum(axis=0).reindex(pio.A.columns).to_numpy(dtype=float)
                acc += flow * factor  # now in tonnes

    if not gas_totals:
        return None

    gas_totals["CO2e"] = sum(GWP100_AR5[g] * gas_totals[g] for g in gas_totals if g in GWP100_AR5)

    data = {}
    for name, tonnes in gas_totals.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            data[name] = np.where(x_vec > 0, tonnes / x_vec, 0.0)  # t / M€
    df = pd.DataFrame(data, index=labels).T  # stressor × label

    # Denominator unit derives from the build's monetary unit (not hardcoded MEUR), so an
    # MUSD build gets t/MUSD, not a false t/MEUR (review).
    return SatelliteAccount(
        provenance=provenance,
        name="GHG",
        units={
            g: (f"tCO2e/{monetary_unit}" if g == "CO2e" else f"t/{monetary_unit}") for g in df.index
        },
        data=df,
    )


def adapt_pymrio(
    pio: pymrio.IOSystem,
    *,
    source: str,
    source_version: str,
    reference_year: int,
    licence: str = EXIOBASE_LICENCE,
    gas_aliases: dict[str, str] | None = None,
    currency: str = "EUR",
    monetary_unit: str = "MEUR",
) -> tuple[IOSystem, list[SatelliteAccount]]:
    """Map a parsed pymrio IOSystem into our harmonised IOSystem + satellites.

    ``currency``/``monetary_unit`` default to EXIOBASE 3's basic-price EUR/MEUR convention.
    They are parameters (not hardcoded) so a differently-denominated source is labelled
    honestly rather than silently relabelled EUR — the engine's exact-unit guard then trusts
    accurate metadata. (The bundled pymrio test fixture is Mill USD; the test build passes
    that through so the label reflects reality.)
    """
    labels = _labels(pio.A.index)
    provenance = Provenance(
        source=source,
        source_version=source_version,
        licence=licence,
        reference_year=reference_year,
        retrieved=date.today().isoformat(),
        notes="Adapted from pymrio IOSystem.",
    )

    # A must be square with identical row/column ordering; verify rather than assume, since a
    # silently mis-ordered source would be relabelled wrongly (review).
    if list(pio.A.index) != list(pio.A.columns):
        raise ValueError("pymrio A index and columns differ; cannot map to a square IOSystem")
    A = pd.DataFrame(pio.A.to_numpy(dtype=float), index=labels, columns=labels)

    # Reindex final demand onto A's product order explicitly (Y's row order need not match A);
    # a missing row would be a real alignment error, not a zero.
    y_by_product = pio.Y.sum(axis=1).reindex(pio.A.index)
    if y_by_product.isna().any():
        raise ValueError("final demand (Y) is not aligned with A's products after reindexing")
    final_demand = pd.DataFrame({"final_demand": y_by_product.to_numpy(dtype=float)}, index=labels)

    sectors, regions = [], []
    for region, sector in pio.A.index:
        if sector not in sectors:
            sectors.append(sector)
        if region not in regions:
            regions.append(region)

    io = IOSystem(
        provenance=provenance,
        sectors=Classification(name=f"{source}-sectors", kind="sector", labels=sectors),
        regions=Classification(name=f"{source}-regions", kind="region", labels=regions),
        price_basis="basic",
        currency=currency,
        unit=monetary_unit,
        A=A,
        final_demand=final_demand,
    )

    satellites: list[SatelliteAccount] = []
    ghg = _ghg_satellite(
        pio, labels, provenance, gas_aliases=gas_aliases, monetary_unit=monetary_unit
    )
    if ghg is not None:
        satellites.append(ghg)

    return io, satellites
