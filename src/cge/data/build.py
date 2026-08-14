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
from cge.contracts.provenance import content_hash
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
    """Rest-of-world region labels — both raw EXIOBASE codes ('W*') and the aggregated
    'RoW_*' blocks the coarse map produces (review: the latter were missed, so the small
    build's RoW quality check was silently omitted)."""
    return [r for r in io.regions.labels if r.upper().startswith("W") or r.startswith("RoW")]


class NatureAttachError(RuntimeError):
    """Raised when nature attachment is *required* (or *auto* and the data is present-but-broken)
    but the ENCORE data cannot be loaded / does not cover the build's sectors."""


def _nature_for_sectors(sector_labels, *, policy: str = "auto"):
    """Return ``(EncoreDependencies, ConcordanceMap)`` covering **every** sector in
    ``sector_labels``, or ``(None, None)`` when nature is legitimately absent.

    Coverage is COMPLETE-or-nothing (review P1 round 6 2026-08-14): the previous logic persisted a
    concordance if *any* sector matched, so a full pxp build got a 13-sector map and then failed the
    other 187 at run time. Here the build's sectors must be either all EXIOBASE **industry** labels
    OR all EXIOBASE **product** (pxp) labels the product bridge covers; a partial match is a hard
    error under ``required``/``auto`` rather than a silent partial persist.

    ``policy`` (review P2 round 6):
      * ``off``    — never attach (returns ``(None, None)``).
      * ``auto``   — attach when the ENCORE data is present AND the labels are EXIOBASE-covered;
                     skip silently only when the data is genuinely ABSENT or the labels aren't
                     EXIOBASE at all (e.g. the offline test MRIO). Data present-but-broken, or a
                     PARTIAL EXIOBASE match, raises — that is a real defect, not "optional data".
      * ``required`` — like ``auto`` but any failure to attach (absent data included) raises."""
    if policy == "off":
        return None, None
    labels = list(sector_labels)
    try:
        from cge.nature.real import encore_data_available
    except ImportError as exc:  # pragma: no cover - nature module optional
        if policy == "required":
            raise NatureAttachError(
                "nature module unavailable but attach policy is 'required'"
            ) from exc
        return None, None
    if not encore_data_available():
        if policy == "required":
            raise NatureAttachError(
                "ENCORE data absent (data/encore/) but attach policy is 'required'"
            )
        return None, None
    # Data is present. Under 'auto' a load/validation failure now is a genuine defect (corrupt CSV,
    # schema drift) — surface it, don't disguise it as "optional data absent" (review P2 round 6).
    from cge.nature.real import (
        real_encore_concordance_industries,
        real_encore_concordance_products,
        real_encore_dependencies,
    )

    try:
        dep = real_encore_dependencies()
        # Completed 163-industry concordance (crosswalk-missing residual filled via NACE siblings),
        # so a direct system="ixi" build attaches over the FULL classification (review P2 round 7).
        industry_cmap, _filled = real_encore_concordance_industries()
    except Exception as exc:  # present-but-broken → always an error (auto and required)
        raise NatureAttachError(f"ENCORE data present but failed to load: {exc}") from exc

    # Pick the label space that covers the build BEST: the industry-keyed concordance or the product
    # (pxp) bridge. A few names appear in both spaces, so choose by coverage count, not "any hit" —
    # a pxp build has ~13 incidental industry-name collisions but is fully covered by the product
    # bridge (review P1 round 6).
    industry_hits = sum(1 for s in labels if s in industry_cmap.weights)
    product_cmap, _uncovered = real_encore_concordance_products()
    product_hits = sum(1 for s in labels if s in product_cmap.weights)
    if industry_hits == 0 and product_hits == 0:
        # No EXIOBASE labels at all (e.g. the offline test MRIO's 'sector1'…) — nature simply
        # doesn't apply. That is legitimate absence, not a defect.
        if policy == "required":
            raise NatureAttachError(
                "build sectors are not EXIOBASE labels (industry or product); cannot attach "
                "nature under policy 'required'"
            )
        return None, None
    full_cmap = product_cmap if product_hits >= industry_hits else industry_cmap

    missing = [s for s in labels if s not in full_cmap.weights]
    if missing:  # PARTIAL EXIOBASE coverage — the P1 defect. Fail loud, never persist a subset.
        raise NatureAttachError(
            f"nature concordance covers only {len(labels) - len(missing)}/{len(labels)} build "
            f"sectors; {len(missing)} uncovered (e.g. {missing[:5]}). A partial concordance would "
            "fail at run time. Extend the concordance/bridge, or set the attach policy to 'off'."
        )
    from cge.contracts.data_objects import ConcordanceMap

    restricted = ConcordanceMap(
        provenance=full_cmap.provenance,
        from_classification=full_cmap.from_classification,
        to_classification=full_cmap.to_classification,
        weights={s: full_cmap.weights[s] for s in labels},
    )
    return dep, restricted


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
    concordance_id: str = "custom",
    gas_aliases: dict[str, str] | None = None,
    currency: str = "EUR",
    monetary_unit: str = "MEUR",
    attach_nature: str = "auto",
) -> dict[str, str]:
    """Adapt, quality-check, store a full build and (optionally) a derived small build.

    Returns a dict of {'full': build_id, 'small': build_id?} actually written.

    ``attach_nature`` is an ``auto|required|off`` **policy** (review P2 round 6 2026-08-14; replaces
    the earlier bool that swallowed every failure). When nature attaches, the ENCORE dependency
    ratings + a COMPLETE EXIOBASE↔ENCORE concordance are persisted alongside the build so a
    NatureStress scenario runs from ``run_scenario(data_source=build_id)`` with no manual assembly;
    the SMALL (aggregated) build gets an **aggregation-aware** concordance derived from the sector
    map. A pxp (product) build is bridged product→industry→ENCORE (review P1 round 6).
      * ``auto``     — attach if ENCORE data is present and the labels are fully EXIOBASE-covered;
                       skip silently only for genuine absence (no data, or non-EXIOBASE labels like
                       the offline MRIO). Present-but-broken data, or PARTIAL coverage, raises.
      * ``required`` — any failure to attach raises (use in CI to assert nature is wired).
      * ``off``      — never attach.
    Accepts ``True``/``False`` for back-compatibility (→ ``auto``/``off``)."""
    if attach_nature is True:
        attach_nature = "auto"
    elif attach_nature is False:
        attach_nature = "off"
    if attach_nature not in ("auto", "required", "off"):
        raise ValueError(f"attach_nature must be auto|required|off, got {attach_nature!r}")
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
        final_demand_kind=io.final_demand_kind,
        retrieved=date.today().isoformat(),
    )
    quality = build_quality_report(build_id, io, satellites, row_regions=_region_row_labels(io))
    # Nature: derive the ENCORE concordance for the FULL build's (real EXIOBASE) sector labels if
    # they are covered, so the build carries nature. None when labels aren't EXIOBASE or ENCORE data
    # is absent — nature is optional (review P1 round 5).
    full_encore, full_conc = _nature_for_sectors(io.sectors.labels, policy=attach_nature)
    store.save(
        meta=meta,
        io=io,
        satellites=satellites,
        quality=quality,
        encore=full_encore,
        concordance=full_conc,
    )
    written = {"full": build_id}

    if make_small and small_sector_map and small_region_map:
        # ALWAYS hash the actual maps (including the default 'custom' id) so a changed
        # concordance yields a different build id/aggregation — a caller changing a custom map
        # must not silently overwrite a numerically different build under the same id (review).
        cmap_hash = content_hash(
            {"conc": concordance_id, "sec": small_sector_map, "reg": small_region_map}
        )[:8]
        agg_name = f"{concordance_id}-{cmap_hash}"
        small_id = f"{build_id}-{agg_name}"
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
            aggregation_name=agg_name,
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
        # Nature for the SMALL build: an AGGREGATION-AWARE concordance, composing the full build's
        # concordance with the sector map (fine EXIOBASE sector → coarse group), so the aggregated
        # build runs nature too (review P1 round 5).
        small_encore, small_conc = (None, None)
        if full_encore is not None and full_conc is not None:
            from cge.nature.concordance_build import aggregate_concordance

            small_encore = full_encore
            # Under 'required' the aggregated concordance must cover every member of every group,
            # too — a silently renormalised covered-subset is the same partial-coverage defect
            # (review P1 round 6). 'auto' tolerates the subset average (records it in provenance).
            small_conc = aggregate_concordance(
                full_conc, small_sector_map, require_complete=(attach_nature == "required")
            )
        store.save(
            meta=s_meta,
            io=s_io,
            satellites=s_sats,
            quality=s_quality,
            encore=small_encore,
            concordance=small_conc,
        )
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
        concordance_id=DEFAULT_CONCORDANCE_VERSION,
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


# Version of the default coarse concordance. Bump when the sector/region maps change so old
# and new small builds are distinguishable (review: changing the maps silently reused the same
# build id and manifest, despite different numbers).
DEFAULT_CONCORDANCE_VERSION = "coarse-v3"

# Coarse sector grouping by keyword: maps each EXIOBASE product to one of ~14 broad sectors.
# Ordered most-specific-first (first match wins). This is a functional default so a real
# build is actually runnable under the engine's product cap; a curated 40-50 sector
# concordance remains the documented follow-up (roadmap P1.6 / P5).
_SECTOR_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    # Exceptions FIRST (first match wins), so specific products aren't captured by the broad
    # energy keywords below (review named these false positives):
    #   'oil seeds' → agriculture (not oil/gas); 'nuclear fuel' → electricity (not 'fuel');
    #   'electricity … petroleum' → electricity; motor-'fuel' retail → trade; biogasification
    #   waste treatment → waste (not 'biogas').
    # Exceptions catch specific products before the broad energy keywords capture them. Ordered
    # most-specific-first. Review named several substring collisions now handled here:
    #   'Other Hydrocarbons' must NOT hit 'hydro' (energy, not electricity);
    #   'Manure (biogas treatment)' must NOT hit 'biogas' (waste, not oil/gas);
    #   animal/grain foods (pigs, poultry, milk, rice) were falling into 'other'.
    ("agriculture", ("oil seed", "seed")),
    ("water_waste", ("manure", "waste", "sewage", "sanitation", "biogasification")),
    ("electricity", ("nuclear", "electricity", "power generation")),
    ("trade", ("retail", "wholesale", "trade")),
    ("energy_coal", ("coal", "lignite", "peat", "anthracite", "coke", "coking", "patent fuel")),
    (
        "energy_oil_gas",
        (
            "petroleum",
            "crude",
            "natural gas",
            "gas ",
            "gasoline",
            "diesel",
            "kerosene",
            "naphtha",
            "refinery",
            "biogas",
            "biofuel",
            "ethanol",
            "motor spirit",
            "lubricant",
            "bitumen",
            "fuel oil",
            "hydrocarbon",
            # refined-product / gas labels the review found unmatched → 'other':
            "ethane",
            "white spirit",
            "paraffin",
            "blast furnace gas",
        ),
    ),
    # 'hydroelectric'/'hydro power' only — NOT 'hydro' alone (it matches 'Hydrocarbons').
    ("electricity", ("power", "steam", "hydroelectric", "hydro power", "wind", "solar")),
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
            # animal & grain products the review found falling into 'other':
            "pig",
            "poultry",
            "cattle",
            "milk",
            "rice",
            "grain",
            "livestock",
            "fish",
            "vegetable",
            "grape",
        ),
    ),
    ("mining", ("mining", "ore", "quarry", "extraction", "metal ores")),
    ("chemicals", ("chemical", "plastic", "rubber", "pharmaceutic", "fertiliser")),
    ("metals", ("iron", "steel", "aluminium", "copper", "metal", "foundry")),
    ("minerals", ("cement", "glass", "ceramic", "concrete", "mineral", "stone", "sand and clay")),
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


# Remaining EXIOBASE country codes folded to their continent (so RoW_Other isn't a grab-bag
# of major economies like CA/KR/MX/AU/TW/ID/ZA — review).
_ASIA = {"KR", "TW", "ID"}
_AMERICA = {"CA", "MX"}
_OCEANIA = {"AU"}
_AFRICA = {"ZA"}


def _coarse_region(code: str) -> str:
    c = str(code)
    if c in _KEY_REGIONS:
        return c
    if c in _CONTINENT:
        return _CONTINENT[c]
    if c in _EUROPE:
        return "RoW_Europe"
    if c in _ASIA or c in _OCEANIA:
        return "RoW_Asia"
    if c in _AMERICA:
        return "RoW_America"
    if c in _AFRICA:
        return "RoW_Africa"
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
