"""SAM-specific quality report & adjustment audit (roadmap Phase 5.1c).

The credibility surface for the CGE: reviewers ask "how much did you make up, and how much did
balancing move the data?". This produces a ``QualityReport`` (the same contract the GUI renders)
capturing:

- **balance** — row-sum = column-sum for every account (fatal if not, after balancing);
- **aggregate preservation** — the SAM reproduces the source EXIOBASE aggregates (gross output,
  final demand, value added) within tolerance (the "conservation through a transform" gate);
- **adjustment magnitude** — how much RAS balancing moved each account (WARN past a threshold);
- **negative cells** — any negative SAM cell is flagged (a SAM should be non-negative);
- **assumptions** — the fabricated/assumed shares (e.g. the capital share) recorded explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cge.contracts.quality import QualityCheck, QualityReport, Severity
from cge.data.sam.balance import imbalance

# Balancing that moves more than this fraction of an account's total is a signal the raw data or
# assumptions are off — a WARN, not a silent "fixed" (roadmap 5.1 risk).
ADJUSTMENT_WARN_FRACTION = 0.05
# Aggregate preservation tolerance (relative).
AGGREGATE_TOL = 1e-6


def sam_quality_report(
    build_id: str,
    matrix: pd.DataFrame,
    *,
    source_gross_output: float,
    source_final_demand: float,
    source_value_added: float,
    sectors: list[str],
    factors: list[str],
    household: str,
    capital_share: float,
    adjustment: pd.DataFrame | None = None,
    value_added_clipped: float = 0.0,
) -> QualityReport:
    """Build the SAM ``QualityReport``. ``adjustment`` (raw − balanced) drives the adjustment-audit
    check when RAS moved cells; pass ``None`` when no balancing was needed."""
    report = QualityReport(build_id=build_id)

    # 1. Balance identity (fatal): row sum = column sum per account.
    imb = imbalance(matrix)
    worst_imb = float(imb.abs().max())
    report.add(
        QualityCheck(
            name="sam_balanced",
            severity=Severity.PASS if worst_imb < 1e-6 else Severity.FAIL,
            message=f"max |row−col| imbalance = {worst_imb:.3e}",
            value=worst_imb,
            tolerance=1e-6,
        )
    )

    # 2. Aggregate preservation: SAM reproduces source EXIOBASE aggregates.
    sam_fd = float(matrix.loc[sectors, household].sum())
    sam_va = float(matrix.loc[factors, sectors].to_numpy().sum())
    # Gross output = intermediate sales + final demand (row totals of the sector accounts).
    sam_x = float(matrix.loc[sectors, :].to_numpy().sum())
    for name, sam_val, src_val in (
        ("final_demand", sam_fd, source_final_demand),
        ("value_added", sam_va, source_value_added),
        ("gross_output", sam_x, source_gross_output),
    ):
        rel = abs(sam_val - src_val) / max(abs(src_val), 1.0)
        report.add(
            QualityCheck(
                name=f"preserves_{name}",
                severity=Severity.PASS if rel < AGGREGATE_TOL else Severity.FAIL,
                message=f"SAM {name} {sam_val:.4g} vs source {src_val:.4g} (rel {rel:.2e})",
                value=rel,
                tolerance=AGGREGATE_TOL,
            )
        )

    # 3. Adjustment audit: how much balancing moved each account (WARN past threshold).
    if adjustment is not None:
        totals = matrix.sum(axis=1).replace(0.0, np.nan)
        moved = adjustment.abs().sum(axis=1) / totals
        worst_moved = float(moved.max(skipna=True) or 0.0)
        report.add(
            QualityCheck(
                name="balancing_adjustment",
                severity=Severity.WARN if worst_moved > ADJUSTMENT_WARN_FRACTION else Severity.PASS,
                message=f"largest account moved {worst_moved:.1%} by balancing",
                value=worst_moved,
                tolerance=ADJUSTMENT_WARN_FRACTION,
                detail={a: float(moved.get(a, 0.0)) for a in matrix.index},
            )
        )

    # 4. Negative cells: a SAM must be non-negative, because calibration reads shares/coefficients
    # off the cells (a negative cell would produce an invalid share). Fatal (review: was a WARN).
    neg = int((matrix.to_numpy() < -1e-9).sum())
    report.add(
        QualityCheck(
            name="non_negative_cells",
            severity=Severity.PASS if neg == 0 else Severity.FAIL,
            message=f"{neg} negative SAM cell(s) (calibration requires non-negative cells)",
            value=float(neg),
        )
    )

    # 5. Negative value-added clip audit: how much negative derived VA was clipped to zero (a
    # non-productive column in the source data). WARN if it moved a material share of VA, so the
    # transformation is visible rather than hidden (review robustness note).
    clip_frac = value_added_clipped / max(source_value_added, 1.0)
    report.add(
        QualityCheck(
            name="value_added_clip",
            severity=Severity.WARN if clip_frac > 1e-6 else Severity.PASS,
            message=f"clipped {value_added_clipped:.4g} of negative value added ({clip_frac:.2%})",
            value=clip_frac,
        )
    )

    # 6. Assumptions recorded (informational PASS — the audit trail).
    report.add(
        QualityCheck(
            name="assumed_capital_share",
            severity=Severity.PASS,
            message=(
                f"value added split cap={capital_share:.2f}/lab={1 - capital_share:.2f} "
                f"(assumption — EXIOBASE factor detail is thin)"
            ),
            value=capital_share,
        )
    )
    return report
