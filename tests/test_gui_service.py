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


# -- P2 (review round 9): GUI must expose trade / factor / welfare / multi-region consumption ---
def _toy_open_result():
    from cge.runner import run_scenario

    scenario = Scenario(
        name="open", engine="cge_static", years=[2020], shocks=[CarbonPrice(price=50.0)]
    )
    return run_scenario(scenario, data_source="toy_cge_open")


def _toy_multi_result():
    from cge.runner import run_scenario

    scenario = Scenario(
        name="multi", engine="cge_static", years=[2020], shocks=[CarbonPrice(price=50.0)]
    )
    return run_scenario(scenario, data_source="toy_cge_multi")


def test_has_macro_recognises_multi_region_real_consumption_change():
    """THE P2 regression: has_macro() previously checked only gdp_change/gdp_change_real, so the
    multi-region CGE's real_consumption_change never triggered the macro section at all."""
    result = _toy_multi_result()
    assert (result.data["variable"] == "real_consumption_change").any()
    assert rv.has_macro(result)
    gdp = rv.macro_gdp_table(result)
    assert "Real consumption Δ (multi-region; NOT GDP)" in gdp.columns


def test_has_trade_and_trade_table_open_economy():
    """The open-economy CGE emits import_change/export_change per sector; the GUI must expose it."""
    result = _toy_open_result()
    assert rv.has_trade(result)
    table = rv.trade_table(result)
    assert {"region", "sector", "year", "Imports Δ", "Exports Δ"} <= set(table.columns)
    assert len(table) > 0


def test_exchange_rate_table_present_for_open_absent_for_multi():
    """The single-region open economy has an exchange rate; the multi-region model does not (trade
    is entirely among the build's own regions, no external ROW)."""
    open_result = _toy_open_result()
    multi_result = _toy_multi_result()
    open_er = rv.exchange_rate_table(open_result)
    multi_er = rv.exchange_rate_table(multi_result)
    assert open_er is not None and len(open_er) > 0
    assert multi_er is None


def test_has_factor_prices_and_factor_price_table():
    result = _toy_open_result()
    assert rv.has_factor_prices(result)
    table = rv.factor_price_table(result)
    assert {"factor", "year", "Factor price Δ"} <= set(table.columns)
    assert len(table) > 0


def test_has_welfare_and_welfare_table():
    result = _toy_open_result()
    assert rv.has_welfare(result)
    table = rv.welfare_table(result)
    cols = {"region", "year", "Welfare Δ", "Carbon revenue (share of own region's GDP)"}
    assert cols <= set(table.columns)
    assert len(table) > 0


def test_welfare_table_multi_region_has_one_row_per_region():
    result = _toy_multi_result()
    table = rv.welfare_table(result)
    assert set(table["region"]) == {"N", "S"}


def test_multi_region_carbon_revenue_is_share_of_own_regional_gdp():
    """THE P2 regression (review round 10): multi-region carbon_revenue must be a share of that
    REGION's own benchmark GDP, not global benchmark GDP — on the toy fixture North has ~53.16%
    of global GDP, so a bug dividing by global GDP instead would understate North's true revenue
    share by exactly that factor. Recompute st.carbon_revenue directly and compare against both
    denominators to pin which one the engine actually used."""
    from cge.data.sam.toy_multi import REGIONS, SECTORS, toy_multi_sam
    from cge.engines.cge_static import model_multi as MM
    from cge.engines.cge_static.calibrate_multi import calibrate_multi
    from cge.engines.cge_static.engine import CGEStaticEngine
    from cge.engines.cge_static.solver import solve

    cal = calibrate_multi(toy_multi_sam(), regions=REGIONS, sectors=SECTORS, factors=["CAP", "LAB"])
    cc = np.zeros((cal.nr, cal.ns))
    cc[0, 0] = 0.3  # carbon price on North's dirty sector only
    sol = solve(
        lambda z: MM.residuals(cal, z, carbon_cost=cc, recycling="lump_sum"),
        MM.initial_guess(cal) * 1.03,
        prefer="scipy",
    )
    st = MM.unpack_state(cal, sol.x, carbon_cost=cc, recycling="lump_sum")

    north_gdp0 = float(cal.F0[:, 0, :].sum())
    global_gdp0 = cal.gdp0
    assert north_gdp0 < global_gdp0  # sanity: North is not the whole economy

    expected_share_of_own_gdp = st.carbon_revenue[0] / north_gdp0
    share_of_global_gdp = st.carbon_revenue[0] / global_gdp0
    # The two denominators must actually differ on this fixture, or the test is vacuous.
    assert abs(expected_share_of_own_gdp - share_of_global_gdp) > 1e-6

    result = CGEStaticEngine().run(
        data={"SAM": toy_multi_sam(), "carbon_cost_share": {"N": {"BRD": 0.3}}},
        shocks=[CarbonPrice(price=1.0)],
        years=[2020],
    )
    d = result.data
    emitted = d[(d["variable"] == "carbon_revenue") & (d["region"] == "N")]["value"].iloc[0]
    assert abs(emitted - expected_share_of_own_gdp) < 1e-6
