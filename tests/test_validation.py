"""Run the whole model-validation suite inside pytest, so CI fails if any model-correctness
check regresses (not just the code-level unit tests). Mirrors `cge validate --strict`."""

from cge.validation import run_all


def test_validation_suite_all_pass():
    summary = run_all()
    failed = [f"{r.suite}::{r.name} — {r.message}" for r in summary.failed]
    assert summary.passed, "validation failures:\n" + "\n".join(failed)
    # sanity: the suites we expect are actually present and non-empty
    by_suite = summary.by_suite()
    assert "io_price" in by_suite and by_suite["io_price"][1] >= 8
    assert "data_layer" in by_suite and by_suite["data_layer"][1] >= 4
