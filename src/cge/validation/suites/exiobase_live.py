"""Live EXIOBASE known-answer validation suite (opt-in).

Registers checks into the standing ``cge validate`` suite **only when** a real EXIOBASE
archive is available (``CGE_EXIOBASE_ARCHIVE`` points at an ``IOT_YYYY_pxp.zip``). Offline —
the default — this suite registers nothing, so ``cge validate`` stays fast and green. This is
the "the standing suite tightens when live data exists" behaviour promised in docs/validation.md.

The heavy parse is done once, lazily, and cached across checks.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from cge.validation.framework import check

SUITE = "exiobase_live"

_ARCHIVE = os.environ.get("CGE_EXIOBASE_ARCHIVE")
_ENABLED = bool(_ARCHIVE and Path(_ARCHIVE).exists())


def _archive_year() -> int:
    import re

    m = re.search(r"IOT_(\d{4})_", Path(_ARCHIVE).name) if _ARCHIVE else None
    return int(m.group(1)) if m else 0


@lru_cache(maxsize=1)
def _adapted():
    from cge.data.adapters.exiobase import adapt_pymrio, parse_exiobase

    year = _archive_year()
    pio = parse_exiobase(_ARCHIVE)
    io, sats = adapt_pymrio(
        pio, source="EXIOBASE", source_version=f"live-{year}", reference_year=year
    )
    return pio, io, sats


if _ENABLED:  # register checks only when a real archive is present

    @check(SUITE, "adapter_reproduces_full_mrio")
    def _shape():
        _, io, sats = _adapted()
        ok = io.A.shape == (9800, 9800) and any(s.name == "GHG" for s in sats)
        return (
            ok,
            f"MRIO shape {io.A.shape}, GHG account present={any(s.name == 'GHG' for s in sats)}",
        )

    @check(SUITE, "adapter_preserves_global_co2")
    def _preserve():
        import numpy as np

        from cge.data.adapters.exiobase import _stressor_unit_to_tonne

        pio, io, sats = _adapted()
        ghg = next(s for s in sats if s.name == "GHG")
        x = pio.x["indout"].reindex(pio.A.columns).to_numpy(float)
        adapter = float(np.sum(ghg.data.loc["CO2"].reindex(list(io.A.columns)).to_numpy(float) * x))
        ext = pio.satellite
        raw = sum(
            float(ext.F.loc[[s]].sum(axis=0).sum()) * _stressor_unit_to_tonne(ext, s)
            for s in ext.F.index
            if str(s).upper().startswith("CO2")
        )
        rel = abs(adapter - raw) / raw
        return rel < 1e-6, f"adapter vs raw global CO2 rel diff = {rel:.2e}", rel, 1e-6

    @check(SUITE, "global_co2_plausible_magnitude")
    def _magnitude():
        import numpy as np

        pio, io, sats = _adapted()
        ghg = next(s for s in sats if s.name == "GHG")
        x = pio.x["indout"].reindex(pio.A.columns).to_numpy(float)
        gt = (
            float(np.sum(ghg.data.loc["CO2"].reindex(list(io.A.columns)).to_numpy(float) * x)) / 1e9
        )
        return 20.0 < gt < 45.0, f"global CO2 = {gt:.1f} Gt (plausible 20-45)", gt

    @check(SUITE, "engine_end_to_end_on_coarse_build")
    def _engine_e2e():
        """Aggregate the real MRIO to a coarse EUR build and run Engine 1: fractional price
        changes with coal among the most exposed (the qualitative live known answer)."""
        import tempfile

        from cge.contracts.shocks import CarbonPrice
        from cge.data.aggregate import aggregate_io
        from cge.data.build import _coarse_region, _coarse_sector
        from cge.data.concordance.concordance import one_to_one
        from cge.data.metadata import BuildMeta
        from cge.data.store import DataStore
        from cge.runner import run_scenario
        from cge.scenarios.loader import Scenario

        _, io, sats = _adapted()
        year = _archive_year()
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
        meta = BuildMeta(
            build_id="exio-coarse-val",
            source="EXIOBASE",
            source_version=f"live-{year}",
            reference_year=year,
            licence="CC BY-SA 4.0",
            retrieved="live",
        )
        s_io, s_sats, s_meta = aggregate_io(
            io,
            sats,
            sector_cmap=scm,
            region_cmap=rcm,
            meta=meta,
            new_build_id="exio-coarse-val",
            aggregation_name="coarse",
        )
        store = DataStore(tempfile.mkdtemp())
        store.save(meta=s_meta, io=s_io, satellites=s_sats)
        res = run_scenario(
            Scenario(name="v", engine="io_price", years=[year], shocks=[CarbonPrice(price=100.0)]),
            data_source="exio-coarse-val",
            store=store,
        )
        dp = res.data[res.data["variable"] == "price_change"]
        top = dp.nlargest(5, "value")["sector"].tolist()
        mx = float(dp["value"].max())
        # Energy sectors (coal and coal-fired electricity) dominate — the robust qualitative
        # known answer (both are the most emissions-intensive per € of output).
        energy_on_top = any(("coal" in s or "electricity" in s) for s in top)
        ok = energy_on_top and 0.0 < mx < 5.0
        return ok, f"max Δp={mx:.1%}, energy sector in top-5={energy_on_top}: {top[:3]}"
