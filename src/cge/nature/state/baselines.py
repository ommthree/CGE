"""The shipped **physical state channels** with documented, cited baselines (Phase 6b.1).

Five channels for the best-sourced ecosystem services (the roadmap's recommended first set +
renewable-resource stocks): **water availability**, **pollination**, **soil quality**, **forestry
stock**, and **fisheries stock**. Each binds a physical state variable to the ENCORE service(s) it
degrades, with a baseline and a state→severity sensitivity.

**Index convention.** Each state variable is expressed as an **index with baseline = 100** (100 =
the reference-year condition from the cited account). This keeps the *baseline* a clean, transparent
reference and puts the empirical content in (a) the degradation/restoration **pathway** (Phase 6b.2,
scenario-driven) and (b) the **sensitivity** of service degradation to the state shortfall. A
scenario
saying "the renewable water stock falls to 70% of its baseline" is then unambiguous, and the
translation to a ``NatureStress`` severity is documented and swappable.

**Sourcing honesty.** The baselines are the reference condition of a *named published account*; the
sensitivities are transparent central estimates (default: proportional, ``sensitivity = 1.0``, i.e.
a
30% state shortfall → 30% service degradation before any exposure weighting) unless a channel has a
published reason to differ. Like the ENCORE materiality ramp, these are **illustrative-of-method**,
not calibrated risk — see ``docs/models/nature-state.md`` for each citation and caveat.
"""

from __future__ import annotations

from cge.contracts.data_objects import Provenance
from cge.nature.state.channels import ServiceStateChannel, StateResponse

_INDEX_UNIT = "index (baseline = 100)"


def _prov(source: str, version: str, ref_year: int, retrieved: str) -> Provenance:
    return Provenance(
        source=source,
        source_version=version,
        licence="see docs/models/nature-state.md (published account, cited)",
        reference_year=ref_year,
        retrieved=retrieved,
        notes=(
            "Physical ecosystem-service state baseline (Phase 6b). Index baseline = 100 = the "
            "reference-year condition of the cited account. Sensitivity is a documented central "
            "estimate, illustrative-of-method, not calibrated risk."
        ),
    )


# --- Water availability -----------------------------------------------------------------------
# Renewable freshwater stock relative to a reference-year condition. A shortfall drives water-supply
# and water-regulation services down (agriculture, and water-intensive industry, feel it via
# exposure).
_WATER = ServiceStateChannel(
    channel_id="water_availability",
    mechanism="water_availability",
    # ENCORE Explanatory note #1: "Water supply" is a COMBINED service that duplicates its
    # components; the Phase-6 translate layer rejects stressing the combined + a component together.
    # So this channel drives the COMPONENT services (flow regulation + purification), not the
    # combined one — the physically-meaningful, non-double-counting choice.
    services=("Water flow regulation", "Water purification"),
    state_variable="renewable freshwater stock (surface + accessible groundwater)",
    unit=_INDEX_UNIT,
    response=StateResponse(baseline=100.0, sensitivity=1.0),
    provenance=_prov(
        "AQUASTAT / SEEA-Water renewable-water-resources accounts (FAO / UN)",
        "SEEA-Water 2012 framework; AQUASTAT renewable water resources",
        2020,
        "2026-08-15",
    ),
    source_note=(
        "Baseline = reference-year total renewable water resources (AQUASTAT / SEEA-Water). "
        "Proportional sensitivity: a fractional stock shortfall degrades water-provisioning and "
        "-regulation services proportionally before exposure weighting."
    ),
)

# --- Pollination ------------------------------------------------------------------------------
# Wild-pollinator abundance index. IPBES: ~75% of global food-crop TYPES depend to some degree on
# animal pollination, but only ~5-8% of crop PRODUCTION VOLUME is at risk if pollinators vanish, so
# a
# pollinator-abundance shortfall maps to crop output with a sub-proportional sensitivity by default.
_POLLINATION = ServiceStateChannel(
    channel_id="pollination",
    mechanism="pollination",
    services=("Pollination",),
    state_variable="wild-pollinator abundance index",
    unit=_INDEX_UNIT,
    response=StateResponse(baseline=100.0, sensitivity=0.6),
    provenance=_prov(
        "IPBES Assessment of Pollinators, Pollination and Food Production",
        "IPBES 2016 pollination assessment",
        2016,
        "2026-08-15",
    ),
    source_note=(
        "Baseline = reference wild-pollinator abundance. IPBES: ~75% of crop types are pollinator-"
        "dependent but the production-volume at risk is far smaller; sensitivity 0.6 is a "
        "documented sub-proportional central estimate (the exposure layer then targets "
        "pollinator-dependent crops)."
    ),
)

# --- Soil quality -----------------------------------------------------------------------------
_SOIL = ServiceStateChannel(
    channel_id="soil_quality",
    mechanism="soil_quality",
    services=("Soil quality regulation", "Soil and sediment retention"),
    state_variable="soil-quality / organic-carbon index of agricultural land",
    unit=_INDEX_UNIT,
    response=StateResponse(baseline=100.0, sensitivity=0.7),
    provenance=_prov(
        "FAO Status of the World's Soil Resources / Global Soil Partnership",
        "FAO SWSR 2015",
        2015,
        "2026-08-15",
    ),
    source_note=(
        "Baseline = reference agricultural soil-quality index (FAO SWSR). Sensitivity 0.7: soil "
        "degradation reduces yield sub-proportionally over the near term (buffering by inputs), a "
        "documented central estimate."
    ),
)

# --- Forestry stock (renewable resource) ------------------------------------------------------
_FORESTRY = ServiceStateChannel(
    channel_id="forestry_stock",
    mechanism="forestry_stock",
    services=("Biomass provisioning",),
    state_variable="standing timber / forest growing-stock volume",
    unit=_INDEX_UNIT,
    response=StateResponse(baseline=100.0, sensitivity=1.0),
    provenance=_prov(
        "FAO Global Forest Resources Assessment (FRA) growing-stock accounts",
        "FAO FRA 2020",
        2020,
        "2026-08-15",
    ),
    source_note=(
        "Baseline = reference forest growing stock (FAO FRA). Proportional sensitivity: a stock "
        "shortfall constrains sustainable timber supply proportionally (forestry-sector input "
        "availability)."
    ),
)

# --- Fisheries stock (renewable resource) -----------------------------------------------------
_FISHERIES = ServiceStateChannel(
    channel_id="fisheries_stock",
    mechanism="fisheries_stock",
    services=("Biomass provisioning",),
    state_variable="fish biomass relative to that supporting maximum sustainable yield",
    unit=_INDEX_UNIT,
    response=StateResponse(baseline=100.0, sensitivity=1.0),
    provenance=_prov(
        "FAO State of World Fisheries and Aquaculture (SOFIA) stock-status accounts",
        "FAO SOFIA 2022",
        2022,
        "2026-08-15",
    ),
    source_note=(
        "Baseline = reference exploitable fish biomass (FAO SOFIA). Proportional sensitivity: a "
        "biomass shortfall constrains sustainable catch proportionally (fishing-sector input "
        "availability)."
    ),
)


# The shipped registry, keyed by channel_id.
SHIPPED_CHANNELS: dict[str, ServiceStateChannel] = {
    ch.channel_id: ch for ch in (_WATER, _POLLINATION, _SOIL, _FORESTRY, _FISHERIES)
}


def shipped_channels() -> dict[str, ServiceStateChannel]:
    """A fresh copy of the shipped physical state-channel registry (Phase 6b.1)."""
    return dict(SHIPPED_CHANNELS)


# --- Toy channels ------------------------------------------------------------------------------
# The shipped channels above map to the REAL ENCORE service vocabulary. For the OFFLINE tutorial /
# CI path — which runs against the synthetic ENCORE fixture (`cge.nature.fixture`) with simplified
# service labels — a parallel toy registry maps to that fixture's vocabulary. Same math,
# illustrative labels; used by examples/nature_state_water.yaml on `--data toy`.
def _toy_prov() -> Provenance:
    return Provenance(
        source="synthetic toy state channel (illustrative)",
        source_version="toy v1",
        licence="illustrative fixture",
        reference_year=2020,
        retrieved="2026-08-15",
        notes="Toy state channel for the offline tutorial; maps to the synthetic ENCORE fixture.",
    )


_TOY_WATER = ServiceStateChannel(
    channel_id="toy_water",
    mechanism="water_availability",
    services=("surface_water",),  # the toy fixture's simplified label
    state_variable="toy renewable water stock",
    unit=_INDEX_UNIT,
    response=StateResponse(baseline=100.0, sensitivity=1.0),
    provenance=_toy_prov(),
    source_note="Illustrative toy channel: proportional water-stock -> surface-water degradation.",
)

_TOY_POLLINATION = ServiceStateChannel(
    channel_id="toy_pollination",
    mechanism="pollination",
    services=("pollination",),
    state_variable="toy pollinator abundance",
    unit=_INDEX_UNIT,
    response=StateResponse(baseline=100.0, sensitivity=0.6),
    provenance=_toy_prov(),
    source_note="Illustrative toy channel: sub-proportional pollinator -> pollination degradation.",
)

TOY_CHANNELS: dict[str, ServiceStateChannel] = {
    ch.channel_id: ch for ch in (_TOY_WATER, _TOY_POLLINATION)
}


def toy_channels() -> dict[str, ServiceStateChannel]:
    """A fresh copy of the illustrative toy state channels (offline tutorial / CI)."""
    return dict(TOY_CHANNELS)


# All channels resolvable by id: real (shipped) + illustrative (toy).
_ALL_CHANNELS: dict[str, ServiceStateChannel] = {**SHIPPED_CHANNELS, **TOY_CHANNELS}


def get_channel(channel_id: str) -> ServiceStateChannel:
    """The channel with ``channel_id`` — real (shipped) or illustrative (toy) — or a ``KeyError``
    naming the valid ids."""
    try:
        return _ALL_CHANNELS[channel_id]
    except KeyError:
        raise KeyError(
            f"unknown state channel {channel_id!r}; known channels are {sorted(_ALL_CHANNELS)}"
        ) from None
