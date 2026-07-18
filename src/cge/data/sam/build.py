"""Raw SAM construction from an EXIOBASE build (roadmap Phase 5.1a).

Maps an ``IOSystem`` (multi-regional, from an aggregated EXIOBASE build) into the accounts of a
single-region **closed-economy** SAM — the calibration target for the CGE pilot. The pilot model
is closed (no Armington trade yet), so this collapses the MRIO's regions into one economy by
summing flows; inter-regional trade is folded into the domestic block until the open-economy
sub-phase adds a rest-of-world account (documented, not hidden).

Steps:
1. Gross output ``x = (I − A)⁻¹ · fd`` (Leontief), then intermediate flows ``Z = A · diag(x)``.
2. Aggregate Z, final demand and value added over regions → sector×sector, sector-vectors.
3. Value added per sector ``VA_i = x_i − Σ_j Z[j,i]`` (output minus intermediate purchases),
   split into capital/labour by a documented share (EXIOBASE's factor split is thin, so the
   split is an explicit assumption recorded in the SAM quality report).
4. Assemble the SAM: activities/commodities collapsed per sector, factors CAP/LAB, one household.

The result is passed to ``balance.py`` (RAS) and ``quality.py`` before the CGE calibrates on it.
Every fabricated cell / assumed share is recorded so a reviewer can see how much was "helped".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cge.contracts.data_objects import SAM, IOSystem, Provenance
from cge.contracts.quality import QualityReport
from cge.data.sam.balance import is_balanced, ras_balance
from cge.data.sam.quality import sam_quality_report

# Default capital share of value added when no factor split is available in the build. EXIOBASE's
# value-added detail is thin; 0.4 capital / 0.6 labour is a common macro default (documented, and
# recorded in the SAM quality report as an assumption, not silently applied).
DEFAULT_CAPITAL_SHARE = 0.4

FACTORS = ["CAP", "LAB"]
HOUSEHOLD = "HOH"


@dataclass(frozen=True)
class RawSAM:
    """A raw (pre-balancing) SAM plus the audit trail of how it was built."""

    sam: SAM
    sectors: list[str]
    capital_share: float
    # audit: source aggregates the SAM must preserve (checked by quality.py)
    source_gross_output: float
    source_final_demand: float
    source_value_added: float


def _gross_output(io: IOSystem) -> tuple[np.ndarray, list[str]]:
    """x = (I − A)⁻¹ · final_demand, per (region:sector) label."""
    labels = list(io.A.columns)
    A = io.A.to_numpy(dtype=float)
    fd = io.final_demand.sum(axis=1).reindex(labels).fillna(0.0).to_numpy(dtype=float)
    x = np.linalg.solve(np.eye(A.shape[0]) - A, fd)
    return x, labels


def _sector_of(label: str) -> str:
    return label.split(":", 1)[1]


def build_raw_sam(io: IOSystem, *, capital_share: float = DEFAULT_CAPITAL_SHARE) -> RawSAM:
    """Build a single-region closed-economy raw SAM from a (multi-regional) ``io``.

    Regions are summed into one economy. Sectors are the build's distinct sector labels. Value
    added is derived from the IO identity and split into capital/labour by ``capital_share``.
    """
    if not 0.0 < capital_share < 1.0:
        raise ValueError(f"capital_share must be in (0,1), got {capital_share}")

    x, labels = _gross_output(io)
    A = io.A.to_numpy(dtype=float)
    # Intermediate flows Z[(r,i),(s,j)] = A · diag(x); aggregate to sector×sector.
    Z = A * x[None, :]  # column j scaled by output x_j
    fd = io.final_demand.sum(axis=1).reindex(labels).fillna(0.0).to_numpy(dtype=float)

    sectors = sorted({_sector_of(lb) for lb in labels})
    s_index = {s: k for k, s in enumerate(sectors)}
    ns = len(sectors)

    # Aggregate intermediates and final demand over regions.
    Zagg = np.zeros((ns, ns))
    FDagg = np.zeros(ns)
    Xagg = np.zeros(ns)
    for a, lb_a in enumerate(labels):
        i = s_index[_sector_of(lb_a)]
        FDagg[i] += fd[a]
        Xagg[i] += x[a]
        for b, lb_b in enumerate(labels):
            j = s_index[_sector_of(lb_b)]
            Zagg[i, j] += Z[a, b]  # supply of sector i to sector j

    # Value added per sector = output − intermediate purchases (column sum of Z into j).
    VAagg = Xagg - Zagg.sum(axis=0)
    VAagg = np.clip(VAagg, 0.0, None)  # guard tiny negatives from rounding

    # Assemble the SAM (row = receipts, col = payments).
    accounts = sectors + FACTORS + [HOUSEHOLD]
    m = pd.DataFrame(0.0, index=accounts, columns=accounts)
    # Intermediates: sector i supplies sector j.
    for i, si in enumerate(sectors):
        for j, sj in enumerate(sectors):
            m.loc[si, sj] = Zagg[i, j]
    # Value added: factors paid by sectors (split by capital_share).
    for i, si in enumerate(sectors):
        m.loc["CAP", si] = capital_share * VAagg[i]
        m.loc["LAB", si] = (1.0 - capital_share) * VAagg[i]
    # Final demand: household buys commodities.
    for i, si in enumerate(sectors):
        m.loc[si, HOUSEHOLD] = FDagg[i]
    # Factor income to the household (closes the loop): all factor income flows to HOH.
    m.loc[HOUSEHOLD, "CAP"] = capital_share * VAagg.sum()
    m.loc[HOUSEHOLD, "LAB"] = (1.0 - capital_share) * VAagg.sum()

    prov = Provenance(
        source=io.provenance.source,
        source_version=io.provenance.source_version,
        licence=io.provenance.licence,
        reference_year=io.provenance.reference_year,
        retrieved=io.provenance.retrieved,
        build_id=io.provenance.build_id,
        generation=io.provenance.generation,
        notes=(
            f"single-region closed SAM from {io.provenance.build_id}; VA split "
            f"cap={capital_share} (assumption); regions summed (trade folded into domestic block)."
        ),
    )
    sam = SAM(provenance=prov, accounts=accounts, matrix=m)
    return RawSAM(
        sam=sam,
        sectors=sectors,
        capital_share=capital_share,
        source_gross_output=float(Xagg.sum()),
        source_final_demand=float(FDagg.sum()),
        source_value_added=float(VAagg.sum()),
    )


def build_sam(
    io: IOSystem, *, capital_share: float = DEFAULT_CAPITAL_SHARE, balance_tol: float = 1e-6
) -> tuple[SAM, QualityReport, list[str]]:
    """Build, balance (if needed) and quality-report a SAM from ``io``.

    Returns ``(sam, quality_report, sectors)``. The closed IO construction is balanced by
    construction; if a residual imbalance exceeds ``balance_tol`` (e.g. after fabricating cells on
    thinner data) it is RAS-balanced to a common row/column total per account, and the adjustment
    magnitude is recorded in the quality report. A build whose SAM cannot be balanced, or that
    fails aggregate preservation, produces a FAIL report (the caller must not calibrate on it)."""
    raw = build_raw_sam(io, capital_share=capital_share)
    m = raw.sam.matrix
    adjustment = None
    if not is_balanced(m, tol=balance_tol):
        # Target each account's total as the mean of its row and column sums (standard RAS target).
        target = (m.sum(axis=1) + m.sum(axis=0)) / 2.0
        balanced = ras_balance(m, target, tol=balance_tol)
        adjustment = m - balanced
        m = balanced
        raw.sam.matrix.loc[:, :] = m  # keep the SAM object's matrix in sync

    report = sam_quality_report(
        io.provenance.build_id or "sam",
        m,
        source_gross_output=raw.source_gross_output,
        source_final_demand=raw.source_final_demand,
        source_value_added=raw.source_value_added,
        sectors=raw.sectors,
        factors=FACTORS,
        household=HOUSEHOLD,
        capital_share=capital_share,
        adjustment=adjustment,
    )
    return raw.sam, report, raw.sectors
