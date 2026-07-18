"""Tests for the Streamlit-free GUI layers: the service façade and results-view reshaping.

The Streamlit pages themselves are thin renderers over these; keeping the logic here means
it is covered without a running server."""

import numpy as np

from cge.contracts.shocks import CarbonPrice
from cge.data.build import build_test
from cge.data.store import DataStore
from cge.gui import results_view as rv
from cge.gui.service import GuiService
from cge.scenarios.loader import Scenario


def _service(tmp_path) -> GuiService:
    store = DataStore(tmp_path)
    build_test(store=store)
    return GuiService(store=store)


def test_service_catalogue_and_frames(tmp_path):
    svc = _service(tmp_path)
    cat = svc.catalogue()
    assert not cat.empty
    build_id = svc.build_ids()[0]

    frames = svc.frames(build_id)
    # A-matrix and final demand always present; GHG satellite present for the test build.
    assert any(name.startswith("A ") for name in frames)
    assert any("Satellite" in name for name in frames)
    a_view = next(v for n, v in frames.items() if n.startswith("A "))
    assert a_view.df.shape[0] == a_view.df.shape[1]  # square


def test_service_run_produces_result(tmp_path):
    # io_price is EUR-only; run on the EUR toy economy (the USD test build is correctly
    # refused by the engine's currency guard).
    svc = _service(tmp_path)
    scenario = Scenario(
        name="t", engine="io_price", years=[2020], shocks=[CarbonPrice(price=100.0)]
    )
    result = svc.run(scenario, data_source="toy")
    assert (result.data["variable"] == "price_change").any()


def test_engines_listed(tmp_path):
    svc = _service(tmp_path)
    names = [e.name for e in svc.engines()]
    assert "io_price" in names


def test_service_sectors_for_toy_and_build(tmp_path):
    """The energy-carrier picker on the Run page needs distinct sectors for 'toy' and a build."""
    svc = _service(tmp_path)
    toy_sectors = svc.sectors("toy")
    assert "energy" in toy_sectors  # the toy fixture's energy carrier
    build_sectors = svc.sectors(svc.build_ids()[0])
    # sector labels, not region:sector composites
    assert build_sectors and all(":" not in s for s in build_sectors)


# -- results_view -------------------------------------------------------------
def _toy_result():
    from cge.runner import run_scenario

    scenario = Scenario(
        name="t", engine="io_price", years=[2020], shocks=[CarbonPrice(price=100.0)]
    )
    return run_scenario(scenario, data_source="toy")


def test_headline_table_sorted_descending():
    table = rv.headline_table(_toy_result())
    vals = table["value"].to_numpy()
    assert len(table) == 6
    assert np.all(np.diff(vals) <= 0)  # sorted descending


def test_waterfall_parts_sum_to_headline():
    result = _toy_result()
    table = rv.headline_table(result)
    top = table.iloc[0]
    wf = rv.waterfall(result, region=top["region"], sector=top["sector"])
    # direct + upstream tiers + residual reconstruct the headline value
    assert np.isclose(wf["value"].sum(), top["value"], atol=1e-6)
    # parts are in supply-chain order, direct first
    assert wf.iloc[0]["part"] == "direct"


def test_goods_with_decomposition_covers_all_goods():
    result = _toy_result()
    pairs = rv.goods_with_decomposition(result)
    assert len(pairs) == 6  # 3 sectors × 2 regions


def test_summary_stats():
    stats = rv.summary_stats(_toy_result())
    assert stats["goods"] == 6
    assert stats["max"] >= stats["mean"] >= stats["min"]
