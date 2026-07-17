"""Validation framework.

A small, uniform structure for *model-correctness* validation — distinct from unit tests
(which check code) and from data `QualityReport`s (which check a build). A validation check
asserts that a model reproduces a known-answer or satisfies an economic identity, and
returns a structured result so the whole suite can be run and reported as one artefact
(the `validate` script / `cge validate` command).

Design mirrors the data quality contract: each check yields a `ValidationResult`
(pass/fail + measured value + tolerance + message); checks register into named suites;
the runner executes suites and aggregates. New engines add a suite; nothing else changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class ValidationResult:
    name: str
    passed: bool
    message: str
    value: float | None = None
    tolerance: float | None = None
    suite: str = ""
    error: str | None = None  # populated if the check itself raised


@dataclass
class Suite:
    """A named group of checks. A check is a callable returning a ValidationResult."""

    name: str
    checks: list[Callable[[], ValidationResult]] = field(default_factory=list)

    def add(self, fn: Callable[[], ValidationResult]) -> Callable[[], ValidationResult]:
        self.checks.append(fn)
        return fn


class Registry:
    """Process-wide registry of validation suites (engines populate it at import)."""

    def __init__(self) -> None:
        self._suites: dict[str, Suite] = {}

    def suite(self, name: str) -> Suite:
        return self._suites.setdefault(name, Suite(name))

    def names(self) -> list[str]:
        return sorted(self._suites)

    def run(self, only: list[str] | None = None) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        for name in self.names():
            if only and name not in only:
                continue
            suite = self._suites[name]
            for check in suite.checks:
                try:
                    r = check()
                    r.suite = name
                except Exception as exc:  # a check that raises is a failure, not a crash
                    r = ValidationResult(
                        name=getattr(check, "__name__", "check"),
                        passed=False,
                        message=f"check raised: {exc}",
                        suite=name,
                        error=repr(exc),
                    )
                results.append(r)
        return results


registry = Registry()


def check(suite_name: str, name: str):
    """Decorator: register a function as a named check in a suite. The function returns
    (passed, message, value?, tolerance?) and this wraps it into a ValidationResult."""

    def deco(fn: Callable) -> Callable[[], ValidationResult]:
        def wrapped() -> ValidationResult:
            out = fn()
            passed, message = out[0], out[1]
            value = out[2] if len(out) > 2 else None
            tol = out[3] if len(out) > 3 else None
            return ValidationResult(
                name=name, passed=passed, message=message, value=value, tolerance=tol
            )

        wrapped.__name__ = name
        registry.suite(suite_name).add(wrapped)
        return wrapped

    return deco


@dataclass
class RunSummary:
    results: list[ValidationResult]
    duration_s: float

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def failed(self) -> list[ValidationResult]:
        return [r for r in self.results if not r.passed]

    @property
    def passed(self) -> bool:
        return not self.failed

    def by_suite(self) -> dict[str, tuple[int, int]]:
        """suite -> (passed, total)."""
        out: dict[str, list[int]] = {}
        for r in self.results:
            agg = out.setdefault(r.suite, [0, 0])
            agg[1] += 1
            if r.passed:
                agg[0] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}


def run_all(only: list[str] | None = None) -> RunSummary:
    # Importing the suites package registers every suite via side effect.
    import cge.validation.suites  # noqa: F401

    t0 = perf_counter()
    results = registry.run(only=only)
    return RunSummary(results=results, duration_s=perf_counter() - t0)
