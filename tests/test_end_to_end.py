"""The Phase 0 definition-of-done test: a registered engine runs end-to-end on the toy
economy via a YAML scenario and emits a schema-valid ResultSet with provenance."""

from pathlib import Path

import pytest

from cge.contracts.engine import registry
from cge.contracts.results import RESULT_COLUMNS
from cge.runner import run_scenario
from cge.scenarios.loader import Scenario, load_scenario

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "carbon_price_toy.yaml"


def test_dummy_engine_is_registered():
    assert "dummy" in registry.names()


def test_end_to_end_from_yaml():
    scenario = load_scenario(EXAMPLE)
    result = run_scenario(scenario, data_source="toy")
    result.validate_schema()

    # Schema and content.
    assert list(result.data.columns) == RESULT_COLUMNS
    assert len(result.data) == 6  # 3 sectors × 2 regions × 1 year
    # Provenance is present and mandatory.
    assert result.manifest.engine_name == "dummy"
    assert result.manifest.scenario_hash
    assert "warning" in result.manifest.assumptions


def test_energy_is_most_carbon_exposed():
    """Sanity on the toy fixture: energy is the emissions-intensive sector, so under a
    carbon price its (placeholder) cost impact should exceed agriculture's."""
    scenario = load_scenario(EXAMPLE)
    df = run_scenario(scenario).data
    energy = df[df.sector == "energy"].value.mean()
    agri = df[df.sector == "agriculture"].value.mean()
    assert energy > agri > 0


def test_engine_rejects_unsupported_shock():
    from cge.contracts.shocks import NatureStress

    scenario = Scenario(
        name="bad",
        engine="dummy",
        years=[2020],
        shocks=[NatureStress(service="pollination", severity=0.3)],
    )
    with pytest.raises(ValueError, match="does not support"):
        run_scenario(scenario)
