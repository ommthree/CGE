"""Contract-level tests: the schemas enforce their invariants."""

import pytest
from pydantic import TypeAdapter

from cge.contracts import CONTRACTS_VERSION
from cge.contracts.data_objects import SAM, ConcordanceMap, Provenance
from cge.contracts.provenance import content_hash
from cge.contracts.shocks import AnyShock, CarbonPrice


def _prov():
    return Provenance(
        source="t", source_version="1", licence="none", reference_year=2020, retrieved="2026-07-17"
    )


def test_contracts_version_is_semver():
    assert CONTRACTS_VERSION.count(".") == 2


def test_concordance_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        ConcordanceMap(
            provenance=_prov(),
            from_classification="a",
            to_classification="b",
            weights={"x": {"p": 0.5, "q": 0.4}},  # sums to 0.9
        )
    # valid one passes
    ConcordanceMap(
        provenance=_prov(),
        from_classification="a",
        to_classification="b",
        weights={"x": {"p": 0.5, "q": 0.5}},
    )


def test_sam_must_be_square_when_populated():
    import pandas as pd

    with pytest.raises(ValueError):
        SAM(provenance=_prov(), accounts=["a", "b"], matrix=pd.DataFrame([[1, 2, 3]]))


def test_shock_discriminated_union_roundtrips():
    adapter = TypeAdapter(AnyShock)
    payload = {"type": "carbon_price", "price": 50.0}
    shock = adapter.validate_python(payload)
    assert isinstance(shock, CarbonPrice)
    assert shock.price == 50.0


def test_shock_coverage_filtering():
    s = CarbonPrice(price=10, coverage_sectors=["energy"], coverage_regions=["A"])
    assert s.applies_to("energy", "A")
    assert not s.applies_to("energy", "B")
    assert not s.applies_to("agriculture", "A")


def test_content_hash_is_stable_and_order_independent():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})
