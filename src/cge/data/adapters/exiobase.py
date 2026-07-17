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
    to the downloaded archive folder. Large (multi-GB); cached — re-runs skip existing files.
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
    return storage


def parse_exiobase(path: str | Path) -> pymrio.IOSystem:
    """Parse a downloaded EXIOBASE folder/zip into a pymrio IOSystem (coefficients computed)."""
    io = pymrio.parse_exiobase3(path=str(path))
    return io.calc_all()


def load_exiobase_test() -> pymrio.IOSystem:
    """pymrio's bundled small test MRIO (offline). Same shape as real EXIOBASE, tiny."""
    return pymrio.load_test().calc_all()


def _ghg_satellite(
    pio: pymrio.IOSystem,
    labels: list[str],
    provenance: Provenance,
    gas_aliases: dict[str, str] | None = None,
) -> SatelliteAccount | None:
    """Build a GHG SatelliteAccount of **emission intensities** (per unit output).

    Looks for an emissions extension; sums stressors matching each GHG and adds a combined
    CO2e row via GWP-100 (AR5). Intensities = F / x (extension flow per unit gross output).
    Returns None if no emissions extension is present or no stressors match.

    ``gas_aliases`` maps a substring of a stressor name to a gas symbol, letting the tiny
    pymrio test system (whose stressors are ``emission_type1/2``) stand in for real gases
    so the offline pipeline carries a GHG account. Real EXIOBASE needs no aliases.
    """
    ext_name = next(
        (e for e in pio.get_extensions() if "emiss" in e.lower() or "ghg" in e.lower()),
        None,
    )
    if ext_name is None:
        return None
    ext = getattr(pio, ext_name)

    # Gross output x per (region, sector); align to labels.
    x = pio.x["indout"] if hasattr(pio, "x") and pio.x is not None else None
    if x is None:
        return None
    x = x.reindex(pio.A.columns)
    x_vec = x.to_numpy(dtype=float)

    F = ext.F  # stressor × (region, sector)

    def _matches(stressor: str, gas: str) -> bool:
        s = str(stressor)
        if s.upper().startswith(gas):  # real EXIOBASE, e.g. 'CO2 - combustion'
            return True
        if gas_aliases:  # alias substrings onto a gas (test system)
            return any(sub in s and alias == gas for sub, alias in gas_aliases.items())
        return False

    rows: dict[str, pd.Series] = {}
    for gas in GWP100_AR5:
        mask = [_matches(s, gas) for s in F.index]
        if not any(mask):
            continue
        gas_total = F.loc[mask].sum(axis=0).reindex(pio.A.columns).to_numpy(dtype=float)
        rows[gas] = pd.Series(gas_total, index=labels)

    if not rows:
        return None

    # Combined CO2e (GWP-weighted sum of available gases).
    co2e_total = sum(GWP100_AR5[g] * rows[g].to_numpy(dtype=float) for g in rows)
    rows["CO2e"] = pd.Series(co2e_total, index=labels)

    # Convert totals -> intensities (per unit output); guard divide-by-zero.
    import numpy as np

    data = {}
    for name, series in rows.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            intens = np.where(x_vec > 0, series.to_numpy(dtype=float) / x_vec, 0.0)
        data[name] = intens
    df = pd.DataFrame(data, index=labels).T  # stressor × label

    return SatelliteAccount(
        provenance=provenance,
        name="GHG",
        units={g: "tCO2e/MEUR" if g == "CO2e" else "t/MEUR" for g in df.index},
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
) -> tuple[IOSystem, list[SatelliteAccount]]:
    """Map a parsed pymrio IOSystem into our harmonised IOSystem + satellites."""
    labels = _labels(pio.A.index)
    provenance = Provenance(
        source=source,
        source_version=source_version,
        licence=licence,
        reference_year=reference_year,
        retrieved=date.today().isoformat(),
        notes="Adapted from pymrio IOSystem.",
    )

    A = pd.DataFrame(pio.A.to_numpy(dtype=float), index=labels, columns=labels)
    final_demand = pd.DataFrame(
        {"final_demand": pio.Y.sum(axis=1).to_numpy(dtype=float)}, index=labels
    )

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
        currency="EUR",
        unit="MEUR",
        A=A,
        final_demand=final_demand,
    )

    satellites: list[SatelliteAccount] = []
    ghg = _ghg_satellite(pio, labels, provenance, gas_aliases=gas_aliases)
    if ghg is not None:
        satellites.append(ghg)

    return io, satellites
