"""Live EXIOBASE known-answer validation suite (opt-in).

Registers checks into the standing ``cge validate`` suite **only when** a real EXIOBASE
archive is available (``CGE_EXIOBASE_ARCHIVE`` points at an ``IOT_YYYY_pxp.zip``). Offline —
the default — this suite registers nothing, so ``cge validate`` stays fast and green. This is
the "the standing suite tightens when live data exists" behaviour promised in docs/validation.md.

Checks: the adapter reproduces the full MRIO shape and preserves/plausibly-sizes global CO2; a
coarse EUR build runs Engine 1 end to end with energy sectors most exposed; and (review 7b.2
2026-08-30) the recursive-dynamic wrapper runs on a REAL coarse EXIOBASE CGE build — emissions
intensity engine-derived from the satellite, not a supplied carbon_cost_share — so a decarbonisation
trajectory drives covered emissions down through the whole IO→SAM→CGE→capital-carry path.

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

    @lru_cache(maxsize=1)
    def _coarse_eur_build():
        """Aggregate the real MRIO to a coarse single-region EUR build and store it, carrying the
        final-demand institution split so the derived SAM has a SAVINV account (dynamic-capable).
        Cached — the recursive emissions check below runs the CGE on it."""
        import tempfile

        from cge.data.aggregate import aggregate_io
        from cge.data.build import _coarse_region, _coarse_sector
        from cge.data.concordance.concordance import one_to_one
        from cge.data.metadata import BuildMeta
        from cge.data.store import DataStore

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
            build_id="exio-coarse-dyn",
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
            new_build_id="exio-coarse-dyn",
            aggregation_name="coarse",
        )
        store = DataStore(tempfile.mkdtemp())
        store.save(meta=s_meta, io=s_io, satellites=s_sats)
        return store, "exio-coarse-dyn", year

    @check(SUITE, "recursive_decarbonisation_on_real_io_build")
    def _recursive_io_emissions():
        """The standing "real IO-backed recursive emissions" gate (review 7b.2 2026-08-30). Run the
        recursive-dynamic wrapper on a REAL coarse EXIOBASE CGE build — where emissions intensity
        is DERIVED IN THE ENGINE from the satellite, NOT a supplied ``carbon_cost_share`` —
        with a 5%/yr per-sector decarbonisation trajectory. Covered emissions must fall well below
        the base year by the horizon, exercising the IO→SAM→CGE→capital-carry→intensity path end to
        end on live data (the toy dynamics suite only exercises the supplied-SAM path)."""
        from cge.contracts.data_objects import Provenance, StructuralTrajectory
        from cge.contracts.shocks import CarbonPrice
        from cge.dynamics import DynamicConfig, run_recursive
        from cge.scenarios.loader import Scenario

        store, build_id, year = _coarse_eur_build()
        # A probe run tells us the build's real sectors, so the decarbonisation trajectory can drive
        # every one of them (an IO build's sector labels are the coarse keyword groups).
        from cge.runner import run_scenario

        probe = run_scenario(
            Scenario(name="p", engine="cge_static", years=[year], shocks=[]),
            data_source=build_id,
            store=store,
        )
        vc = probe.data.loc[probe.data["variable"] == "volume_change", "sector"]
        sectors = sorted(vc.unique())
        traj = StructuralTrajectory(
            provenance=Provenance(
                source="validation decarbonisation path",
                source_version="v",
                licence="n/a",
                reference_year=year,
                retrieved="2026-08-30",
            ),
            sector_rates={"emissions_intensity": {s: {year: -0.05} for s in sectors}},
            sources={f"emissions_intensity:{s}": "validation" for s in sectors},
            confidence={f"emissions_intensity:{s}": "low" for s in sectors},
        )
        sc = Scenario(
            name="dyn",
            engine="cge_static",
            years=[year, year + 20],
            shocks=[CarbonPrice(price=50.0)],
        )
        path = run_recursive(
            sc, config=DynamicConfig(structural=traj), data_source=build_id, store=store
        )
        d = path.result.data
        ce = d[
            (d["variable"] == "covered_emissions_change")
            & (d["scenario"] == "central")
            & (d["year"] == year + 20)
        ]
        val = float(ce["value"].iloc[0])
        return (
            val < -0.3,
            "IO-backed recursive 5%/yr decarbonisation → covered emissions well below base year",
            val,
        )
