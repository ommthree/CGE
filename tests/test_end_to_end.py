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


@pytest.mark.parametrize(
    "source,variant",
    [
        ("toy_cge", "single region"),
        ("toy_cge_open", "open economy"),
        ("toy_cge_multi", "multi-region"),
    ],
)
def test_toy_cge_data_sources_run(source, variant):
    """The built-in CGE toy SAMs (closed / open / multi-region) run through the runner without a
    data build — the seam the GUI uses to demo the CGE variants — and price the dirty sector."""
    from cge.contracts.shocks import CarbonPrice

    sc = Scenario(name="t", engine="cge_static", years=[2020], shocks=[CarbonPrice(price=50.0)])
    res = run_scenario(sc, data_source=source)
    res.validate_schema()
    d = res.data
    dirty = d[(d["variable"] == "volume_change") & (d["sector"] == "BRD")]
    # The TAXED dirty sector's output falls. For multi-region the cost is on North only, so restrict
    # to the taxed region (South's BRD rises via cross-region leakage — not a regression).
    if source == "toy_cge_multi":
        dirty = dirty[dirty["region"] == "N"]
    assert (dirty["value"] < 0).all()


def test_toy_cge_data_overrides_change_result():
    """Engine parameters passed via ``data_overrides`` (the GUI's CGE elasticity controls) reach the
    engine: a different Armington elasticity changes the open economy's carbon-leakage magnitude."""
    from cge.contracts.shocks import CarbonPrice

    sc = Scenario(name="t", engine="cge_static", years=[2020], shocks=[CarbonPrice(price=50.0)])
    lo = run_scenario(sc, data_source="toy_cge_open", data_overrides={"armington_elast": 1.5})
    hi = run_scenario(sc, data_source="toy_cge_open", data_overrides={"armington_elast": 5.0})

    def _imp(res):
        d = res.data
        return d[(d["variable"] == "import_change") & (d["sector"] == "BRD")]["value"].iloc[0]

    assert _imp(hi) > _imp(lo) > 0  # higher Armington elasticity → more import leakage


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
