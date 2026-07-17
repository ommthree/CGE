"""Pipeline consistency & plausibility checks.

Where `checks.py` verifies a *single* build in isolation, this module verifies invariants
that must hold *between* pipeline stages — the things that catch a broken adapter or a
wrong aggregation before bad data reaches an engine. These run inside the pipeline (not
only in tests), so a build that violates a hard invariant fails loudly at build time.

Two tiers:

- **Structural invariants** (``assert_structural``) — must-hold properties (finite values,
  square A, aligned labels, existing Leontief inverse). Violations raise; there is no
  meaningful build otherwise.
- **Plausibility checks** (``plausibility_checks``) — soft, quantitative sanity tests that
  become ``QualityCheck``s (non-negative intensities, final demand present, output positive).
  Violations warn, surfacing in the report rather than aborting.

The cross-stage aggregation check (``check_aggregation_conserves``) compares a coarse build
against the fine build it came from: total gross output and total final demand must be
preserved (the correctness property of flow-based aggregation, [MillerBlair2009 §4.3]).
"""

from __future__ import annotations

import numpy as np

from cge.contracts.data_objects import IOSystem, SatelliteAccount
from cge.contracts.quality import QualityCheck, QualityReport, Severity

# Relative tolerance for conservation identities (float32 storage + solve round-off).
CONSERVATION_RTOL = 1e-4


class ConsistencyError(ValueError):
    """A structural invariant was violated — the build is not usable."""


def total_output(io: IOSystem) -> np.ndarray:
    """Gross output x = (I - A)^-1 f, consistent with final demand."""
    A = io.A.to_numpy(dtype=float)
    f = io.final_demand.sum(axis=1).reindex(io.A.columns).fillna(0.0).to_numpy(dtype=float)
    return np.linalg.solve(np.eye(A.shape[0]) - A, f)


def assert_structural(io: IOSystem, satellites: list[SatelliteAccount]) -> None:
    """Raise ``ConsistencyError`` on any must-hold structural violation. Called by the
    build pipeline immediately after adapt and after aggregate."""
    A = io.A
    labels = list(A.columns)

    if A.shape[0] != A.shape[1]:
        raise ConsistencyError(f"A is not square: {A.shape}")
    if list(A.index) != labels:
        raise ConsistencyError("A row/column labels are not aligned")
    if not np.isfinite(A.to_numpy(dtype=float)).all():
        raise ConsistencyError("A contains non-finite values (NaN/inf)")

    fd_labels = list(io.final_demand.index)
    if fd_labels != labels:
        raise ConsistencyError("final_demand index is not aligned with A")

    # Leontief inverse must exist (productive economy).
    rho = float(np.max(np.abs(np.linalg.eigvals(A.to_numpy(dtype=float)))))
    if not rho < 1.0:
        raise ConsistencyError(f"ρ(A) = {rho:.4f} ≥ 1; Leontief inverse does not exist")

    for sat in satellites:
        if list(sat.data.columns) != labels:
            raise ConsistencyError(f"satellite {sat.name!r} columns are not aligned with A labels")
        if not np.isfinite(sat.data.to_numpy(dtype=float)).all():
            raise ConsistencyError(f"satellite {sat.name!r} contains non-finite values")


def plausibility_checks(io: IOSystem, satellites: list[SatelliteAccount]) -> list[QualityCheck]:
    """Soft, quantitative sanity checks appended to a build's QualityReport."""
    out: list[QualityCheck] = []

    x = total_output(io)
    n_nonpos = int(np.sum(x <= 0))
    out.append(
        QualityCheck(
            name="gross_output_positive",
            severity=Severity.PASS if n_nonpos == 0 else Severity.WARN,
            message=(
                "All sectors have positive gross output"
                if n_nonpos == 0
                else f"{n_nonpos} sectors have non-positive implied gross output"
            ),
            value=float(n_nonpos),
        )
    )

    fd_total = float(io.final_demand.sum().sum())
    out.append(
        QualityCheck(
            name="final_demand_present",
            severity=Severity.PASS if fd_total > 0 else Severity.FAIL,
            message=f"total final demand = {fd_total:.4g}",
            value=fd_total,
        )
    )

    for sat in satellites:
        neg = float((sat.data.to_numpy(dtype=float) < 0).mean())
        out.append(
            QualityCheck(
                name=f"satellite_nonneg_{sat.name}",
                # Emission intensities should be ≥ 0; a few negatives can occur but many
                # signal a mapping error.
                severity=Severity.PASS if neg < 0.01 else Severity.WARN,
                message=f"{neg:.3%} of '{sat.name}' intensities are negative",
                value=neg,
                tolerance=0.01,
            )
        )
    return out


def check_aggregation_conserves(
    fine: IOSystem, coarse: IOSystem, *, rtol: float = CONSERVATION_RTOL
) -> QualityReport:
    """Cross-stage check: a coarse build must conserve the fine build's totals.

    Returns a ``QualityReport`` (so the result is inspectable/stored); the pipeline treats a
    FAIL here as fatal because it means the aggregation is wrong, not merely low-quality.
    """
    report = QualityReport(build_id=f"{coarse.provenance.source}::agg-consistency")

    x_fine, x_coarse = total_output(fine).sum(), total_output(coarse).sum()
    ok_x = bool(np.isclose(x_fine, x_coarse, rtol=rtol))
    report.add(
        QualityCheck(
            name="aggregation_conserves_output",
            severity=Severity.PASS if ok_x else Severity.FAIL,
            message=f"total gross output {x_fine:.6g} -> {x_coarse:.6g}",
            value=abs(x_fine - x_coarse) / abs(x_fine) if x_fine else 0.0,
            tolerance=rtol,
        )
    )

    fd_fine = float(fine.final_demand.sum().sum())
    fd_coarse = float(coarse.final_demand.sum().sum())
    ok_fd = bool(np.isclose(fd_fine, fd_coarse, rtol=rtol))
    report.add(
        QualityCheck(
            name="aggregation_conserves_final_demand",
            severity=Severity.PASS if ok_fd else Severity.FAIL,
            message=f"total final demand {fd_fine:.6g} -> {fd_coarse:.6g}",
            value=abs(fd_fine - fd_coarse) / abs(fd_fine) if fd_fine else 0.0,
            tolerance=rtol,
        )
    )
    return report
