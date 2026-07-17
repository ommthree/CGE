"""Data-quality checks (task 1.4).

Turn a built IO system + satellites into a ``QualityReport``: balance identities,
negative-value flags, RoW share, satellite coverage, and (optionally) drift vs a previous
build. The report is data the GUI renders; the checks encode what "trust this build" means.
"""

from cge.data.quality.checks import build_quality_report, drift_report
from cge.data.quality.consistency import (
    ConsistencyError,
    assert_structural,
    check_aggregation_conserves,
    plausibility_checks,
)

__all__ = [
    "build_quality_report",
    "drift_report",
    "assert_structural",
    "plausibility_checks",
    "check_aggregation_conserves",
    "ConsistencyError",
]
