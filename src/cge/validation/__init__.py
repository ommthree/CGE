"""Validation assets: toy economies with known analytic answers, and identity checks.

Every future engine is tested against ``toy_economy()`` (P0.5): a hand-built system
small enough to reason about by hand, so engine tests can assert exact numbers rather
than 'looks plausible'.
"""

from cge.validation.toy import toy_economy

__all__ = ["toy_economy"]
