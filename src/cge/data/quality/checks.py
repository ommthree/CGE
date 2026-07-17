"""Quality checks producing a ``QualityReport``.

Checks implemented (roadmap task 1.4):

- **Productivity / well-posedness:** spectral radius ρ(A) < 1 — the precondition for the
  Leontief inverse to exist (Engine 1 depends on it; see io-price-model.md §4). FAIL if not.
- **Column-sum sanity:** input cost shares (column sums of A) should lie in [0, 1) for a
  productive economy; entries ≥ 1 are flagged.
- **Negative values:** EXIOBASE legitimately contains small negatives (stock changes,
  subsidies); we flag their share rather than clipping (roadmap P1 risk note).
- **RoW / imputation share:** share of activity in rest-of-world regions — high values mean
  results lean on the least-reliable part of the data.
- **Satellite coverage:** fraction of sectors with non-zero emission intensity.

Each check has a severity so the GUI can surface the worst issues first. Thresholds are
deliberately conservative and documented inline; tune as real EXIOBASE builds are inspected.
"""

from __future__ import annotations

import numpy as np

from cge.contracts.data_objects import IOSystem, SatelliteAccount
from cge.contracts.quality import QualityCheck, QualityReport, Severity


def build_quality_report(
    build_id: str,
    io: IOSystem,
    satellites: list[SatelliteAccount],
    *,
    row_regions: list[str] | None = None,
) -> QualityReport:
    report = QualityReport(build_id=build_id)
    A = io.A.to_numpy(dtype=float)
    labels = list(io.A.columns)

    # -- spectral radius (well-posedness of the Leontief inverse) --------------
    # Largest-magnitude eigenvalue; must be < 1 for a productive economy.
    try:
        rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    except np.linalg.LinAlgError:
        rho = float("nan")
    report.add(
        QualityCheck(
            name="spectral_radius",
            severity=Severity.PASS if rho < 1.0 else Severity.FAIL,
            message=(
                f"ρ(A) = {rho:.4f} < 1 (Leontief inverse exists)"
                if rho < 1.0
                else f"ρ(A) = {rho:.4f} ≥ 1 — not productive; Engine 1 will fail"
            ),
            value=rho,
            tolerance=1.0,
        )
    )

    # -- column sums (input cost shares) ---------------------------------------
    col_sums = A.sum(axis=0)
    n_over = int(np.sum(col_sums >= 1.0))
    max_col = float(col_sums.max()) if col_sums.size else 0.0
    report.add(
        QualityCheck(
            name="column_sums_lt_one",
            severity=Severity.PASS if n_over == 0 else Severity.WARN,
            message=(
                f"All {len(labels)} input cost shares < 1 (max {max_col:.3f})"
                if n_over == 0
                else f"{n_over} sectors have input cost share ≥ 1 (max {max_col:.3f})"
            ),
            value=max_col,
            tolerance=1.0,
        )
    )

    # -- negative values (flag, don't clip) ------------------------------------
    neg_share = float(np.mean(A < 0)) if A.size else 0.0
    report.add(
        QualityCheck(
            name="negative_coefficient_share",
            # small negatives are expected; only warn if pervasive.
            severity=Severity.PASS if neg_share < 0.02 else Severity.WARN,
            message=f"{neg_share:.3%} of A entries are negative (expected small; not clipped)",
            value=neg_share,
            tolerance=0.02,
        )
    )

    # -- RoW / imputation share ------------------------------------------------
    if row_regions:
        row_mask = np.array([lab.split(":", 1)[0] in row_regions for lab in labels])
        fd = io.final_demand.sum(axis=1).reindex(labels).fillna(0.0).to_numpy(dtype=float)
        total = float(fd.sum())
        row_share = float(fd[row_mask].sum() / total) if total > 0 else 0.0
        report.add(
            QualityCheck(
                name="rest_of_world_share",
                severity=Severity.PASS if row_share < 0.35 else Severity.WARN,
                message=f"{row_share:.1%} of final demand is in rest-of-world regions",
                value=row_share,
                tolerance=0.35,
            )
        )

    # -- satellite coverage ----------------------------------------------------
    for sat in satellites:
        data = sat.data.reindex(columns=labels).fillna(0.0)
        covered = float((data.abs().sum(axis=0) > 0).mean())
        report.add(
            QualityCheck(
                name=f"satellite_coverage_{sat.name}",
                severity=Severity.PASS if covered > 0.5 else Severity.WARN,
                message=f"{covered:.1%} of sectors have non-zero '{sat.name}' intensity",
                value=covered,
                tolerance=0.5,
            )
        )

    # -- plausibility checks (positive output, final demand present, non-neg intensities) --
    from cge.data.quality.consistency import plausibility_checks

    for check in plausibility_checks(io, satellites):
        report.add(check)

    return report


def drift_report(build_id: str, current: QualityReport, previous: QualityReport) -> QualityReport:
    """Compare two builds' numeric checks and flag material drift (build-over-build).

    Only checks present in both, with numeric values, are compared. A >10% relative change
    is flagged WARN — a cheap early warning that a data-source update changed something.
    """
    report = QualityReport(build_id=f"{build_id}::drift")
    prev = {c.name: c for c in previous.checks if c.value is not None}
    for c in current.checks:
        if c.value is None or c.name not in prev:
            continue
        p = prev[c.name].value
        if p is None or p == 0:
            continue
        rel = abs(c.value - p) / abs(p)
        report.add(
            QualityCheck(
                name=f"drift_{c.name}",
                severity=Severity.WARN if rel > 0.10 else Severity.PASS,
                message=f"{c.name}: {p:.4g} → {c.value:.4g} ({rel:+.1%})",
                value=rel,
                tolerance=0.10,
            )
        )
    return report
