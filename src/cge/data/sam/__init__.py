"""Social Accounting Matrix construction & balancing for the CGE (roadmap Phase 5.1).

The SAM is the CGE's calibration target: a square, balanced matrix over named accounts. This
package builds one (from a hand-checkable toy, and later from an EXIOBASE build), balances it,
and reports SAM-specific quality. The ``SAM`` contract lives in ``cge.contracts.data_objects``.
"""

from cge.data.sam.build import build_open_raw_sam, build_open_sam, build_raw_sam, build_sam
from cge.data.sam.toy import toy_sam
from cge.data.sam.toy_multi import toy_multi_sam
from cge.data.sam.toy_open import toy_open_sam

__all__ = [
    "toy_sam",
    "toy_open_sam",
    "toy_multi_sam",
    "build_sam",
    "build_raw_sam",
    "build_open_sam",
    "build_open_raw_sam",
]
