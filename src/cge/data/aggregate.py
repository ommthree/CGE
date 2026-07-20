"""Task 1.3 — aggregation machinery.

Aggregate an IO system from a fine classification (e.g. EXIOBASE 200 products × 49
regions) to a coarse one (the interactive "small build", ~40-60 sectors × ~10 regions),
driven by concordances.

**The one subtlety that matters:** technical coefficients ``A`` cannot be averaged. The
economically correct procedure aggregates *flows*, not coefficients [MillerBlair2009,
§4.3]:

1. Recover intermediate flows ``Z = A · x̂`` where ``x`` is gross output (``x̂`` its
   diagonal), and total output ``x``.
2. Aggregate flows and outputs with the bridge matrix ``B`` (target×source):
   ``Z' = B Z Bᵀ``, ``x' = B x``, ``f' = B f`` (final demand), ``e' = B e`` (satellite
   totals — extensive quantities add).
3. Recompute coefficients on the aggregated system: ``A' = Z' x̂'⁻¹``, and satellite
   *intensities* as ``e'/x'`` if intensities are wanted.

We store satellite accounts as **intensities** (per unit output), so we convert to totals
before aggregating and back to intensities after — otherwise the aggregate intensity would
be a meaningless sum of per-unit rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cge.contracts.data_objects import (
    Classification,
    ConcordanceMap,
    IOSystem,
    SatelliteAccount,
)
from cge.data.concordance import bridge_matrix
from cge.data.metadata import BuildMeta


def _combined_bridge(
    labels: list[str],
    sector_cmap: ConcordanceMap,
    region_cmap: ConcordanceMap,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a bridge over ``region:sector`` labels from separate sector and region
    concordances. Returns (B, target_labels) with B target×source."""
    sectors = []
    regions = []
    for label in labels:
        r, s = label.split(":", 1)
        if s not in sectors:
            sectors.append(s)
        if r not in regions:
            regions.append(r)

    Bs = bridge_matrix(sector_cmap, sectors)  # target_sector × source_sector
    Br = bridge_matrix(region_cmap, regions)  # target_region × source_region

    target_labels = [f"{tr}:{ts}" for tr in Br.index for ts in Bs.index]
    B = pd.DataFrame(0.0, index=target_labels, columns=labels)
    for label in labels:
        r, s = label.split(":", 1)
        for tr, wr in region_cmap.weights[r].items():
            for ts, ws in sector_cmap.weights[s].items():
                B.loc[f"{tr}:{ts}", label] = wr * ws
    return B, target_labels


def aggregate_io(
    io: IOSystem,
    satellites: list[SatelliteAccount],
    *,
    sector_cmap: ConcordanceMap,
    region_cmap: ConcordanceMap,
    meta: BuildMeta,
    new_build_id: str,
    aggregation_name: str,
    total_output: pd.Series | None = None,
) -> tuple[IOSystem, list[SatelliteAccount], BuildMeta]:
    """Aggregate ``io`` + ``satellites`` to the coarser classification.

    ``total_output`` (gross output per label) is needed to recover flows from coefficients;
    if not supplied it is derived from the Leontief identity ``x = (I-A)⁻¹ f`` using the
    system's final demand, which is exact for a balanced system.
    """
    labels = list(io.A.columns)
    A = io.A.to_numpy(dtype=float)
    n = A.shape[0]

    f = io.final_demand.sum(axis=1).reindex(labels).fillna(0.0).to_numpy(dtype=float)
    if total_output is None:
        # x = (I - A)^-1 f  (gross output consistent with final demand)
        x = np.linalg.solve(np.eye(n) - A, f)
    else:
        x = total_output.reindex(labels).fillna(0.0).to_numpy(dtype=float)

    Z = A * x[np.newaxis, :]  # Z = A x̂  (column j scaled by output j)

    B_df, target_labels = _combined_bridge(labels, sector_cmap, region_cmap)
    B = B_df.to_numpy(dtype=float)

    Z_agg = B @ Z @ B.T
    x_agg = B @ x
    f_agg = B @ f

    # Recompute aggregated coefficients; guard divide-by-zero for empty aggregates.
    with np.errstate(divide="ignore", invalid="ignore"):
        A_agg = np.where(x_agg[np.newaxis, :] > 0, Z_agg / x_agg[np.newaxis, :], 0.0)

    A_agg_df = pd.DataFrame(A_agg, index=target_labels, columns=target_labels)
    # Final demand: when the build carries the per-consuming-region split (review P1 — the open-SAM
    # builder needs it), aggregate BOTH axes: producing labels through B, consuming-region columns
    # through the region bridge. Otherwise keep the legacy single aggregate column.
    fd_region = io.fd_by_region()
    if fd_region is not None:
        source_regions = list(fd_region.columns)
        Br = bridge_matrix(region_cmap, source_regions)  # target_region × source_region
        F = fd_region.reindex(labels).fillna(0.0).to_numpy(dtype=float)
        F_agg = B @ F @ Br.to_numpy(dtype=float).T
        fd_agg_df = pd.DataFrame(F_agg, index=target_labels, columns=list(Br.index))
        fd_kind = "by_region"
    else:
        fd_agg_df = pd.DataFrame({"final_demand": f_agg}, index=target_labels)
        fd_kind = "aggregate"

    tr_sectors: list[str] = []
    tr_regions: list[str] = []
    for label in target_labels:
        r, s = label.split(":", 1)
        if s not in tr_sectors:
            tr_sectors.append(s)
        if r not in tr_regions:
            tr_regions.append(r)

    new_meta = meta.derived(
        build_id=new_build_id,
        aggregation=aggregation_name,
        notes=f"Aggregated from {meta.build_id}: {n} -> {len(target_labels)} labels.",
    ).model_copy(update={"final_demand_kind": fd_kind})  # explicit, not inherited from the source
    new_io = IOSystem(
        provenance=io.provenance,
        sectors=Classification(
            name=f"{aggregation_name}-sectors", kind="sector", labels=tr_sectors
        ),
        regions=Classification(
            name=f"{aggregation_name}-regions", kind="region", labels=tr_regions
        ),
        price_basis=io.price_basis,
        currency=io.currency,
        unit=io.unit,
        A=A_agg_df,
        final_demand=fd_agg_df,
        final_demand_kind=fd_kind,
    )

    # Satellites: intensities -> totals (× x) -> aggregate -> back to intensities (÷ x_agg).
    new_sats: list[SatelliteAccount] = []
    for sat in satellites:
        S = sat.data.reindex(columns=labels).fillna(0.0).to_numpy(dtype=float)  # stressor × label
        totals = S * x[np.newaxis, :]
        totals_agg = totals @ B.T
        with np.errstate(divide="ignore", invalid="ignore"):
            intens_agg = np.where(x_agg[np.newaxis, :] > 0, totals_agg / x_agg[np.newaxis, :], 0.0)
        new_sats.append(
            SatelliteAccount(
                provenance=sat.provenance,
                name=sat.name,
                units=sat.units,
                data=pd.DataFrame(intens_agg, index=sat.data.index, columns=target_labels),
            )
        )

    return new_io, new_sats, new_meta
