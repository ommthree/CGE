"""A small, explicitly-illustrative ENCORE dependency fixture (Phase 6.1).

This is **not** the full ENCORE knowledge base — it is a hand-entered **synthetic/expert-designed**
subset so the exposure engine and nature scenarios run offline and are testable without the
(registration-gated) ENCORE download. The ratings are informed by the *qualitative direction* of the
central-bank literature but are NOT substantiated cell-by-cell against a published table (review P1
2026-08-07): treat them as illustrative of the method, not as calibrated dependency values:

- van Toor et al. (2020), *Indebted to nature* (DNB/PBL) — agriculture and food production depend
  very highly on pollination, water and soil/nutrient services; water supply depends very highly on
  water-flow maintenance and climate regulation; manufacturing/other sectors have low direct
  dependency but inherit exposure through their agri-food inputs (the upstream channel the exposure
  engine computes).

The processes use short toy labels aligned with the toy economy's sectors so a fixture run threads
end-to-end. Replace this object with ``load_encore_csv(<full ENCORE export>)`` for real analysis —
the contract is identical, so nothing downstream changes.

Sources: [vanToor2020], [ENCORE] — see docs/references.md and docs/models/nature-encore.md.
"""

from __future__ import annotations

import pandas as pd

from cge.contracts.data_objects import ConcordanceMap, Provenance
from cge.nature.encore import EncoreDependencies

# Ecosystem services (an ENCORE-style short list).
SERVICES = [
    "pollination",
    "surface_water",
    "soil_quality",
    "climate_regulation",
    "flood_control",
]

# (process, service, materiality-class) — illustrative, sourced ratings. Direction/magnitude follow
# van Toor 2020: agri/food very-highly water/pollination/soil dependent; water utility very-highly
# water/climate dependent; manufacturing largely independent DIRECTLY (its exposure is upstream).
_RATINGS: list[tuple[str, str, str]] = [
    # Agriculture — the archetypal nature-dependent sector.
    ("agriculture", "pollination", "VH"),
    ("agriculture", "surface_water", "VH"),
    ("agriculture", "soil_quality", "VH"),
    ("agriculture", "climate_regulation", "H"),
    ("agriculture", "flood_control", "M"),
    # Food processing — depends on its agricultural inputs and on water.
    ("food", "surface_water", "H"),
    ("food", "climate_regulation", "M"),
    ("food", "soil_quality", "L"),
    # Water supply/utility — depends on water-flow and climate regulation.
    ("water_supply", "surface_water", "VH"),
    ("water_supply", "climate_regulation", "H"),
    ("water_supply", "flood_control", "H"),
    # Manufacturing — low DIRECT dependency (exposure comes through inputs, computed upstream).
    ("manufacturing", "surface_water", "L"),
    ("manufacturing", "climate_regulation", "L"),
    # Energy — moderate water dependency (cooling), else low.
    ("energy", "surface_water", "M"),
    ("energy", "climate_regulation", "L"),
]


def encore_fixture() -> EncoreDependencies:
    """The illustrative ENCORE dependency object (see module docstring for sourcing)."""
    df = pd.DataFrame(_RATINGS, columns=["process", "service", "materiality"])
    prov = Provenance(
        source="ENCORE (SYNTHETIC illustrative fixture, expert-designed; not the ENCORE export)",
        source_version="fixture-v1",
        licence="synthetic/illustrative — not the licensed ENCORE export",
        reference_year=2020,
        # A FIXED authored date, not date.today(): this is static synthetic data, so its provenance
        # must not look freshly retrieved every day (review P2 2026-08-07).
        retrieved="2026-08-01",
        notes=(
            "Hand-entered SYNTHETIC/expert-designed subset for offline testing, informed by the "
            "qualitative pattern of the central-bank literature but NOT substantiated cell-by-cell "
            "against a published table; NOT the full ENCORE knowledge base. Replace via "
            "load_encore_csv for real analysis. See docs/models/nature-encore.md."
        ),
    )
    return EncoreDependencies(provenance=prov, ratings=df, kind="dependency")


def toy_encore_concordance() -> ConcordanceMap:
    """Map the toy economy's sectors (``agriculture``/``energy``/``manufacturing``) to the ENCORE
    fixture's processes. Each maps one-to-one to the like-named process (weight 1) — the simplest
    reviewable case. A real concordance splits an EXIOBASE sector across several ENCORE processes
    with documented weights; here the toy sectors line up by name, so the weights are trivial and
    the mapping stays hand-checkable."""
    prov = Provenance(
        source="ENCORE↔toy concordance (illustrative)",
        source_version="fixture-v1",
        licence="illustrative",
        reference_year=2020,
        retrieved="2026-08-01",  # fixed authored date, not date.today() (review P2 2026-08-07)
        notes="Toy sector → ENCORE process, one-to-one by name. See docs/models/nature-encore.md.",
    )
    return ConcordanceMap(
        provenance=prov,
        from_classification="toy-sectors",
        to_classification="ENCORE-processes",
        weights={
            "agriculture": {"agriculture": 1.0},
            "energy": {"energy": 1.0},
            "manufacturing": {"manufacturing": 1.0},
        },
    )
