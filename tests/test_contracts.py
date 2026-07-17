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


# -- review-round-2 hardening -------------------------------------------------
def test_classification_rejects_duplicate_labels():
    from cge.contracts.data_objects import Classification

    with pytest.raises(ValueError, match="duplicate labels"):
        Classification(name="c", kind="sector", labels=["a", "b", "a"])


def test_concordance_rejects_negative_weights_that_sum_to_one():
    with pytest.raises(ValueError, match="negatives"):
        ConcordanceMap(
            provenance=_prov(),
            from_classification="s",
            to_classification="d",
            weights={"a": {"X": 1.5, "Y": -0.5}},  # sums to 1 but has a negative
        )


def test_naturestress_severity_bounded():
    from cge.contracts.shocks import NatureStress

    NatureStress(service="pollination", severity=0.3)  # ok
    with pytest.raises(ValueError):
        NatureStress(service="pollination", severity=2.0)


def test_resultset_rejects_string_and_duplicate_and_bad_band():
    import pandas as pd

    from cge.contracts.provenance import RunManifest
    from cge.contracts.results import RESULT_COLUMNS, ResultSet

    m = RunManifest.build(
        engine_name="e",
        engine_version="1",
        data_source="d",
        scenario={},
        assumptions={"x": 1},
    )
    # numeric-looking string value rejected
    df = pd.DataFrame([["price", "s", "r", 2020, "central", "1.5"]], columns=RESULT_COLUMNS)
    with pytest.raises(ValueError, match="numeric dtype"):
        ResultSet(data=df, manifest=m).validate_schema()
    # duplicate rows rejected
    row = ["price", "s", "r", 2020, "central", 1.0]
    dup = pd.DataFrame([row, row], columns=RESULT_COLUMNS)
    with pytest.raises(ValueError, match="duplicate"):
        ResultSet(data=dup, manifest=m).validate_schema()
    # invalid band rejected
    bad = pd.DataFrame([["price", "s", "r", 2020, "bogus", 1.0]], columns=RESULT_COLUMNS)
    with pytest.raises(ValueError, match="band labels"):
        ResultSet(data=bad, manifest=m).validate_schema()


def test_manifest_rejects_empty_assumptions():
    from cge.contracts.provenance import RunManifest

    with pytest.raises(ValueError, match="empty assumptions"):
        RunManifest.build(
            engine_name="e",
            engine_version="1",
            data_source="d",
            scenario={},
            assumptions={},
        )


# -- review-round-3 hardening -------------------------------------------------
def test_concordance_rejects_nan_weights():
    with pytest.raises(ValueError, match="not finite"):
        ConcordanceMap(
            provenance=_prov(),
            from_classification="s",
            to_classification="d",
            weights={"a": {"X": float("nan")}},
        )


def test_manifest_rejects_empty_assumptions_at_construction():
    from cge.contracts.provenance import RunManifest

    with pytest.raises(ValueError, match="mandatory"):
        RunManifest(
            engine_name="e", engine_version="1", data_source="d", scenario_hash="h", assumptions={}
        )


def test_resultset_rejects_string_year():
    import pandas as pd

    from cge.contracts.provenance import RunManifest
    from cge.contracts.results import RESULT_COLUMNS, ResultSet

    m = RunManifest.build(
        engine_name="e", engine_version="1", data_source="d", scenario={}, assumptions={"x": 1}
    )
    df = pd.DataFrame([["price", "s", "r", "2020", "central", 1.0]], columns=RESULT_COLUMNS)
    with pytest.raises(ValueError, match="year"):
        ResultSet(data=df, manifest=m).validate_schema()
