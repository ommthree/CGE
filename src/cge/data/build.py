"""Build orchestration — the "one command builds full + small datasets" of the P1 DoD.

Ties the adapter, aggregation, quality and store together:

    fetch (live) -> parse -> adapt -> quality -> save  (full build)
                                   -> aggregate -> quality -> save  (small build)

``build_from_pymrio`` is source-agnostic (takes an already-parsed pymrio system), so the
same path serves the live EXIOBASE download and the offline test system. ``build_exiobase``
is the live convenience wrapper.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pymrio

from cge.contracts.data_objects import IOSystem
from cge.contracts.quality import Severity
from cge.data.adapters.exiobase import (
    adapt_pymrio,
    fetch_exiobase,
    load_exiobase_test,
    parse_exiobase,
)
from cge.data.aggregate import aggregate_io
from cge.data.concordance.concordance import one_to_one
from cge.data.metadata import BuildMeta
from cge.data.quality import (
    ConsistencyError,
    assert_structural,
    build_quality_report,
    check_aggregation_conserves,
)
from cge.data.store import DataStore, default_store


def _region_row_labels(io: IOSystem) -> list[str]:
    """EXIOBASE rest-of-world region codes (start with 'W'); empty for the test system."""
    return [r for r in io.regions.labels if r.upper().startswith("W")]


def build_from_pymrio(
    pio: pymrio.IOSystem,
    *,
    source: str,
    source_version: str,
    reference_year: int,
    build_id: str,
    store: DataStore | None = None,
    make_small: bool = True,
    small_sector_map: dict[str, str] | None = None,
    small_region_map: dict[str, str] | None = None,
    gas_aliases: dict[str, str] | None = None,
    currency: str = "EUR",
    monetary_unit: str = "MEUR",
) -> dict[str, str]:
    """Adapt, quality-check, store a full build and (optionally) a derived small build.

    Returns a dict of {'full': build_id, 'small': build_id?} actually written.
    """
    store = store or default_store()
    io, satellites = adapt_pymrio(
        pio,
        source=source,
        source_version=source_version,
        reference_year=reference_year,
        gas_aliases=gas_aliases,
        currency=currency,
        monetary_unit=monetary_unit,
    )
    # Consistency gate 1: the adapted build must be structurally sound before we store it.
    assert_structural(io, satellites)

    meta = BuildMeta(
        build_id=build_id,
        source=source,
        source_version=source_version,
        reference_year=reference_year,
        licence=io.provenance.licence,
        currency=currency,
        monetary_unit=monetary_unit,
        retrieved=date.today().isoformat(),
    )
    quality = build_quality_report(build_id, io, satellites, row_regions=_region_row_labels(io))
    store.save(meta=meta, io=io, satellites=satellites, quality=quality)
    written = {"full": build_id}

    if make_small and small_sector_map and small_region_map:
        small_id = f"{build_id}-small"
        sector_cmap = one_to_one(
            small_sector_map,
            from_classification=io.sectors.name,
            to_classification="small-sectors",
            provenance=io.provenance,
        )
        region_cmap = one_to_one(
            small_region_map,
            from_classification=io.regions.name,
            to_classification="small-regions",
            provenance=io.provenance,
        )
        s_io, s_sats, s_meta = aggregate_io(
            io,
            satellites,
            sector_cmap=sector_cmap,
            region_cmap=region_cmap,
            meta=meta,
            new_build_id=small_id,
            aggregation_name="small",
        )
        # Consistency gate 2: the aggregate must be structurally sound AND conserve the
        # fine build's totals — a wrong aggregation is fatal, not merely low quality.
        assert_structural(s_io, s_sats)
        agg_check = check_aggregation_conserves(io, s_io)
        if not agg_check.passed:
            failed = [c.message for c in agg_check.checks if c.severity != Severity.PASS]
            raise ConsistencyError(f"Aggregation to {small_id} broke conservation: {failed}")

        s_quality = build_quality_report(
            small_id, s_io, s_sats, row_regions=_region_row_labels(s_io)
        )
        # Fold the cross-stage conservation checks into the stored small-build report.
        for c in agg_check.checks:
            s_quality.add(c)
        store.save(meta=s_meta, io=s_io, satellites=s_sats, quality=s_quality)
        written["small"] = small_id

    return written


def build_exiobase(
    *,
    year: int = 2019,
    system: str = "pxp",
    download_dir: str | Path = "downloads/exiobase",
    store: DataStore | None = None,
    make_small: bool = True,
) -> dict[str, str]:
    """Live build: download EXIOBASE from Zenodo, parse, adapt, quality, store.

    Small build uses the default EXIOBASE aggregation maps (see ``default_maps``).
    """
    folder = fetch_exiobase(download_dir, year=year, system=system)
    pio = parse_exiobase(folder)
    src_version = f"3-{system}-{year}"
    sec_map, reg_map = default_maps(pio) if make_small else (None, None)
    return build_from_pymrio(
        pio,
        source="EXIOBASE",
        source_version=src_version,
        reference_year=year,
        build_id=f"exiobase-{src_version}",
        store=store,
        make_small=make_small,
        small_sector_map=sec_map,
        small_region_map=reg_map,
    )


def build_test(store: DataStore | None = None) -> dict[str, str]:
    """Offline build from pymrio's bundled test MRIO — the CI/dev path. Exercises the whole
    pipeline (adapt -> quality -> aggregate -> store) with no download."""
    pio = load_exiobase_test()
    # Trivial small map: fold the 8 test sectors into 3 groups, 6 regions into 2.
    sectors = list(pio.get_sectors())
    regions = list(pio.get_regions())
    sec_map = {s: ["primary", "energy", "manufacturing"][i % 3] for i, s in enumerate(sectors)}
    reg_map = {r: ("A" if i < len(regions) // 2 else "B") for i, r in enumerate(regions)}
    return build_from_pymrio(
        pio,
        source="EXIOBASE-test",
        source_version="test",
        reference_year=2011,
        build_id="exiobase-test",
        store=store,
        make_small=True,
        small_sector_map=sec_map,
        small_region_map=reg_map,
        # The test MRIO's stressors aren't real gases; alias one onto CO2 so the offline
        # build carries a GHG account for downstream engine tests. Real EXIOBASE: no alias.
        gas_aliases={"emission_type1": "CO2"},
        # The bundled pymrio fixture is Mill USD; label it honestly. io_price (EUR-only) will
        # correctly refuse to run on it — that is the intended behaviour, and engine tests use
        # the EUR toy economy. This build exists to exercise the data pipeline, not the engine.
        currency="USD",
        monetary_unit="MUSD",
    )


# Coarse sector grouping by keyword: maps each EXIOBASE product to one of ~14 broad sectors.
# Ordered most-specific-first (first match wins). This is a functional default so a real
# build is actually runnable under the engine's product cap; a curated 40-50 sector
# concordance remains the documented follow-up (roadmap P1.6 / P5).
_SECTOR_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("energy_coal", ("coal", "lignite", "peat")),
    ("energy_oil_gas", ("petroleum", "crude", "natural gas", "gas ", "gasoline", "fuel")),
    ("electricity", ("electricity", "power", "steam")),
    (
        "agriculture",
        (
            "cattle",
            "crop",
            "wheat",
            "cereal",
            "vegetable",
            "fruit",
            "animal",
            "farming",
            "agricultur",
            "forestry",
            "fishing",
            "paddy",
            "sugar",
            "oil seeds",
            "plant",
            "meat",
            "dairy",
            "food",
            "beverage",
            "tobacco",
        ),
    ),
    ("mining", ("mining", "ore", "quarry", "extraction", "metal ores")),
    ("chemicals", ("chemical", "plastic", "rubber", "pharmaceutic", "fertiliser")),
    ("metals", ("iron", "steel", "aluminium", "copper", "metal", "foundry")),
    ("minerals", ("cement", "glass", "ceramic", "concrete", "mineral")),
    (
        "manufacturing",
        (
            "machinery",
            "equipment",
            "vehicle",
            "motor",
            "transport equipment",
            "electronic",
            "textile",
            "wood",
            "paper",
            "furniture",
            "manufactur",
        ),
    ),
    ("construction", ("construction", "building")),
    ("transport", ("transport", "shipping", "aviation", "logistics", "railway", "pipeline")),
    ("water_waste", ("water", "waste", "sewage", "recycling", "sanitation")),
    ("trade", ("trade", "retail", "wholesale", "sale ")),
    (
        "services",
        (
            "service",
            "financ",
            "insurance",
            "real estate",
            "education",
            "health",
            "hotel",
            "communication",
            "research",
            "public admin",
            "recreation",
        ),
    ),
]

# Region folding: keep the largest economies distinct, fold the rest into continental blocks.
_KEY_REGIONS = {"US", "CN", "DE", "GB", "JP", "IN", "FR", "BR", "RU", "IT"}
_CONTINENT: dict[str, str] = {
    # EXIOBASE uses ISO2 country codes plus 5 W* rest-of-world regions; a light map to blocks.
    "WA": "RoW_Asia",
    "WL": "RoW_America",
    "WE": "RoW_Europe",
    "WF": "RoW_Africa",
    "WM": "RoW_MiddleEast",
}
_EUROPE = {
    "AT",
    "BE",
    "BG",
    "CY",
    "CZ",
    "DK",
    "EE",
    "ES",
    "FI",
    "GR",
    "HR",
    "HU",
    "IE",
    "LT",
    "LU",
    "LV",
    "MT",
    "NL",
    "NO",
    "PL",
    "PT",
    "RO",
    "SE",
    "SI",
    "SK",
    "CH",
    "TR",
}


def _coarse_sector(name: str) -> str:
    low = str(name).lower()
    for target, keywords in _SECTOR_KEYWORDS:
        if any(k in low for k in keywords):
            return target
    return "other"


def _coarse_region(code: str) -> str:
    c = str(code)
    if c in _KEY_REGIONS:
        return c
    if c in _CONTINENT:
        return _CONTINENT[c]
    if c in _EUROPE:
        return "RoW_Europe"
    return "RoW_Other"


def default_maps(pio: pymrio.IOSystem) -> tuple[dict[str, str], dict[str, str]]:
    """Default EXIOBASE→small-build aggregation.

    Groups the 200 EXIOBASE products into ~14 broad sectors (keyword match) and the 49 regions
    into ~10-15 economies/continental blocks, giving a build of a few hundred products —
    runnable under the engine's dense cap. A curated, analytically-precise 40-50 sector
    concordance remains the documented follow-up (roadmap P1.6/P5); this is a functional
    default, not that.
    """
    sectors = list(pio.get_sectors())
    regions = list(pio.get_regions())
    sec_map = {s: _coarse_sector(s) for s in sectors}
    reg_map = {r: _coarse_region(r) for r in regions}
    return sec_map, reg_map
