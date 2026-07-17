"""Render a validation RunSummary as text (for the CLI/script) or markdown (for docs/CI)."""

from __future__ import annotations

from cge.validation.framework import RunSummary


def format_text(summary: RunSummary) -> str:
    lines: list[str] = []
    for suite, (passed, total) in sorted(summary.by_suite().items()):
        lines.append(f"\n{suite}  ({passed}/{total})")
        for r in summary.results:
            if r.suite != suite:
                continue
            mark = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{mark}] {r.name}: {r.message}")
    n_fail = len(summary.failed)
    verdict = "ALL PASSED" if summary.passed else f"{n_fail} FAILED"
    n_pass = summary.total - n_fail
    lines.append(f"\n{verdict} — {n_pass}/{summary.total} checks in {summary.duration_s:.2f}s")
    return "\n".join(lines)


def format_markdown(summary: RunSummary) -> str:
    lines = ["# Validation report", ""]
    n_fail = len(summary.failed)
    lines.append(
        f"**{'✅ ALL PASSED' if summary.passed else f'❌ {n_fail} FAILED'}** — "
        f"{summary.total - n_fail}/{summary.total} checks in {summary.duration_s:.2f}s"
    )
    for suite, (passed, total) in sorted(summary.by_suite().items()):
        lines += [
            "",
            f"## {suite} ({passed}/{total})",
            "",
            "| check | result | detail |",
            "|---|---|---|",
        ]
        for r in summary.results:
            if r.suite != suite:
                continue
            lines.append(f"| {r.name} | {'✅' if r.passed else '❌'} | {r.message} |")
    return "\n".join(lines) + "\n"
