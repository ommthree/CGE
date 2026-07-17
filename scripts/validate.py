#!/usr/bin/env python
"""Run the model validation suite and report.

    python scripts/validate.py                 # all suites, text report
    python scripts/validate.py --suite io_price
    python scripts/validate.py --markdown docs/validation-report.md
    python scripts/validate.py --strict        # exit non-zero if any check fails (CI)

This is the standing model-correctness audit — distinct from `pytest` (which gates code
changes). Both share the underlying checks; this produces a human-readable report and a
CI-friendly exit code. Equivalent to `cge validate`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cge.validation import run_all
from cge.validation.report import format_markdown, format_text


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the CGE model validation suite")
    p.add_argument("--suite", action="append", help="limit to named suite(s); repeatable")
    p.add_argument("--markdown", metavar="PATH", help="also write a markdown report to PATH")
    p.add_argument("--strict", action="store_true", help="exit non-zero if any check fails")
    args = p.parse_args(argv)

    summary = run_all(only=args.suite)
    print(format_text(summary))

    if args.markdown:
        Path(args.markdown).write_text(format_markdown(summary))
        print(f"\nMarkdown report written to {args.markdown}")

    if args.strict and not summary.passed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
