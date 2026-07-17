"""Data-quality contract.

A ``QualityReport`` is a machine-readable verdict on a data build: the balance
identities that should hold, the coverage that exists, and the flags a user needs to
see before trusting a build. The GUI renders these directly (Phase 3), so the shape is
a contract, not an implementation detail.

Design: each check is a ``QualityCheck`` with a numeric value, an optional tolerance,
and a pass/warn/fail severity. Reports compose (a build has many checks) and are
comparable across builds (drift detection) because every check has a stable ``name``.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Severity(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class QualityCheck(BaseModel):
    """One named check with its result. ``value``/``tolerance`` are optional because
    some checks are categorical (a count, a boolean) rather than a tolerance test."""

    name: str
    severity: Severity
    message: str
    value: float | None = None
    tolerance: float | None = None
    detail: dict[str, float] = Field(
        default_factory=dict, description="optional per-entity breakdown, e.g. per region"
    )


class QualityReport(BaseModel):
    """The quality verdict for one data build."""

    build_id: str = Field(description="identifies the build these checks describe")
    checks: list[QualityCheck] = Field(default_factory=list)

    def add(self, check: QualityCheck) -> None:
        self.checks.append(check)

    @property
    def worst(self) -> Severity:
        order = {Severity.PASS: 0, Severity.WARN: 1, Severity.FAIL: 2}
        return max((c.severity for c in self.checks), key=lambda s: order[s], default=Severity.PASS)

    @property
    def passed(self) -> bool:
        return self.worst != Severity.FAIL

    def summary(self) -> dict[str, int]:
        out = {s.value: 0 for s in Severity}
        for c in self.checks:
            out[c.severity.value] += 1
        return out
