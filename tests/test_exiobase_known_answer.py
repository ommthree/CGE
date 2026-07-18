"""Live EXIOBASE known-answer tests (the roadmap P1/P2 DoD gate).

These require a real EXIOBASE archive and are therefore **opt-in**: they are skipped unless
``CGE_EXIOBASE_ARCHIVE`` points at a downloaded ``IOT_YYYY_pxp.zip``. Offline CI stays green;
run them after ``cge build --exiobase`` (or set the env var to a cached archive).

    CGE_EXIOBASE_ARCHIVE=downloads/exiobase/IOT_2019_pxp.zip \
        pytest tests/test_exiobase_known_answer.py

What they pin (validated 2026-07 on 2019 pxp):
- the adapter preserves EXIOBASE's global CO2 total exactly (intensity×output == raw F);
- that total is in the known EXIOBASE magnitude (~30 Gt for 2019 production accounting);
- units come through as t/MEUR (not kg), so a €100/t run gives fractional price changes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ARCHIVE = os.environ.get("CGE_EXIOBASE_ARCHIVE")
pytestmark = pytest.mark.skipif(
    not (ARCHIVE and Path(ARCHIVE).exists()),
    reason="set CGE_EXIOBASE_ARCHIVE to a real IOT_YYYY_pxp.zip to run live known-answer tests",
)


def _year_from_archive(path: str) -> int:
    """Parse the year from an EXIOBASE archive filename like 'IOT_2019_pxp.zip'."""
    import re

    m = re.search(r"IOT_(\d{4})_", Path(path).name)
    return int(m.group(1)) if m else 0


@pytest.fixture(scope="module")
def adapted():
    from cge.data.adapters.exiobase import adapt_pymrio, parse_exiobase

    year = _year_from_archive(ARCHIVE)
    pio = parse_exiobase(ARCHIVE)
    io, sats = adapt_pymrio(
        pio, source="EXIOBASE", source_version=f"3-pxp-{year}", reference_year=year
    )
    return pio, io, sats


def test_full_mrio_shape_and_extension(adapted):
    pio, io, sats = adapted
    assert io.A.shape == (9800, 9800)  # 200 products × 49 regions
    assert io.unit == "MEUR" and io.currency == "EUR"
    ghg = next(s for s in sats if s.name == "GHG")
    # the real EXIOBASE emission extension is named 'satellite' — the round-1 fix that made
    # the adapter recognise it is exercised here on real data.
    assert {"CO2", "CO2e"} <= set(ghg.data.index)
    assert ghg.units["CO2"] == "t/MEUR"


def test_adapter_preserves_global_co2_total(adapted):
    """Adapter intensity×output must equal pymrio's raw satellite total (proves the kg→t
    conversion and intensity=F/x construction on real data)."""
    from cge.data.adapters.exiobase import _stressor_unit_to_tonne

    pio, io, sats = adapted
    ghg = next(s for s in sats if s.name == "GHG")
    labels = list(io.A.columns)
    x = pio.x["indout"].reindex(pio.A.columns).to_numpy(float)

    adapter_total = float(np.sum(ghg.data.loc["CO2"].reindex(labels).to_numpy(float) * x))

    ext = pio.satellite
    raw_total = 0.0
    for s in ext.F.index:
        if str(s).upper().startswith("CO2"):
            raw_total += float(ext.F.loc[[s]].sum(axis=0).sum()) * _stressor_unit_to_tonne(ext, s)

    # Same quantity computed two ways; the only difference is summation order over ~9800
    # float64 terms, so a modest relative tolerance is the honest bound (not 1e-9).
    assert np.isclose(adapter_total, raw_total, rtol=1e-6)


def test_global_co2_is_plausible_magnitude(adapted):
    """~30 Gt for EXIOBASE 2019 production accounting (global fossil CO2 was ~35 Gt)."""

    pio, io, sats = adapted
    ghg = next(s for s in sats if s.name == "GHG")
    x = pio.x["indout"].reindex(pio.A.columns).to_numpy(float)
    total_gt = (
        float(np.sum(ghg.data.loc["CO2"].reindex(list(io.A.columns)).to_numpy(float) * x)) / 1e9
    )
    assert 20.0 < total_gt < 45.0, f"global CO2 = {total_gt:.1f} Gt, outside plausible range"


def test_engine1_on_coarse_real_build(adapted, tmp_path):
    """End-to-end known-answer: aggregate the real MRIO to a coarse EUR build and run Engine 1;
    energy sectors (coal / coal-fired electricity) must be most exposed, with fractional impacts."""
    from cge.contracts.shocks import CarbonPrice
    from cge.data.aggregate import aggregate_io
    from cge.data.build import _coarse_region, _coarse_sector
    from cge.data.concordance.concordance import one_to_one
    from cge.data.metadata import BuildMeta
    from cge.data.store import DataStore
    from cge.runner import run_scenario
    from cge.scenarios.loader import Scenario

    pio, io, sats = adapted
    scm = one_to_one(
        {s: _coarse_sector(s) for s in io.sectors.labels},
        from_classification=io.sectors.name,
        to_classification="cs",
        provenance=io.provenance,
    )
    rcm = one_to_one(
        {r: _coarse_region(r) for r in io.regions.labels},
        from_classification=io.regions.name,
        to_classification="cr",
        provenance=io.provenance,
    )
    year = io.provenance.reference_year  # parsed from the archive filename by the fixture
    meta = BuildMeta(
        build_id="exio-coarse",
        source="EXIOBASE",
        source_version=f"live-{year}",
        reference_year=year,
        licence="CC BY-SA 4.0",
        retrieved="2026-07-18",
    )
    s_io, s_sats, s_meta = aggregate_io(
        io,
        sats,
        sector_cmap=scm,
        region_cmap=rcm,
        meta=meta,
        new_build_id="exio-coarse",
        aggregation_name="coarse",
    )
    assert s_io.A.shape[0] < 500  # runnable under the dense cap

    store = DataStore(tmp_path)
    store.save(meta=s_meta, io=s_io, satellites=s_sats)
    result = run_scenario(
        Scenario(name="ka", engine="io_price", years=[year], shocks=[CarbonPrice(price=100.0)]),
        data_source="exio-coarse",
        store=store,
    )
    dp = result.data[result.data["variable"] == "price_change"]
    # Energy sectors (coal and coal-fired electricity — the most emissions-intensive per € of
    # output) dominate the most-exposed set. The robust qualitative known answer.
    top = dp.nlargest(5, "value")["sector"].tolist()
    assert any(("coal" in s or "electricity" in s) for s in top), (
        f"no energy sector in top-5: {top}"
    )
    # Impacts are fractional (percent-scale), not the ~1e3–1e9 a units bug would give.
    assert 0.0 < dp["value"].max() < 5.0
