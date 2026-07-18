"""The default demand-elasticity set, keyed to the coarse sector classification.

Own-price demand elasticities (ε ≤ 0). Values are indicative ranges drawn from the applied
IO/CGE literature (GTAP parameter documentation, USDA/energy demand studies, and demand-system
meta-analyses); they are deliberately *ranges*, not point estimates, because volume results
are sensitive to them and precise sector-matched values require a curated concordance. Each
carries a (low, central, high) triple, a source tag, and a confidence tag.

**These are functional defaults for a screening tool, not a calibrated elasticity database.**
Treat the volume envelope as indicative. Replacing this with curated, sector-matched values is
the natural follow-up (roadmap P4.1 / P5 elasticities).
"""

from __future__ import annotations

from datetime import date

from cge.contracts.data_objects import ElasticitySet, Provenance

# Fallback own-price demand elasticity for any good without an explicit value: mild inelastic
# response, wide band. Tagged 'default' so results flag it.
DEFAULT_DEMAND_ELASTICITY: tuple[float, float, float] = (-0.8, -0.5, -0.2)

# Coarse-sector own-price demand elasticities: (low, central, high), all ≤ 0.
# 'low' = most elastic (most negative), 'high' = least elastic — so |Δq| spans low..high.
_COARSE_DEMAND: dict[str, tuple[float, float, float, str, str]] = {
    # sector: (low, central, high, source, confidence)
    "energy_coal": (
        -0.9,
        -0.5,
        -0.3,
        "energy demand meta-analyses (coal is substitutable)",
        "medium",
    ),
    "energy_oil_gas": (-0.8, -0.4, -0.2, "energy demand literature", "medium"),
    "electricity": (-0.5, -0.3, -0.1, "electricity demand studies (inelastic)", "medium"),
    "agriculture": (-0.6, -0.4, -0.2, "food demand systems (inelastic staples)", "medium"),
    "mining": (-0.9, -0.6, -0.3, "intermediate-input demand (derived)", "low"),
    "chemicals": (-1.0, -0.7, -0.4, "GTAP intermediate elasticities", "low"),
    "metals": (-1.1, -0.8, -0.5, "GTAP intermediate elasticities", "low"),
    "minerals": (-0.9, -0.6, -0.3, "GTAP intermediate elasticities", "low"),
    "manufacturing": (-1.3, -0.9, -0.6, "manufactured-goods demand (elastic)", "medium"),
    "construction": (-0.7, -0.5, -0.3, "investment-good demand", "low"),
    "transport": (-1.0, -0.6, -0.3, "transport demand studies", "medium"),
    "water_waste": (-0.4, -0.2, -0.1, "utility demand (very inelastic)", "low"),
    "trade": (-1.2, -0.8, -0.5, "trade-service demand", "low"),
    "services": (-1.0, -0.7, -0.4, "services demand", "medium"),
    "other": DEFAULT_DEMAND_ELASTICITY + ("residual bucket", "default"),  # type: ignore[operator]
}


def default_demand_set() -> ElasticitySet:
    """The default own-price demand ``ElasticitySet`` over the coarse sector classification."""
    prov = Provenance(
        source="literature (assembled)",
        source_version="coarse-default-v1",
        licence="see per-value sources",
        reference_year=0,
        retrieved=date.today().isoformat(),
        notes="Indicative demand elasticities for a screening tool; not a calibrated database.",
    )
    values: dict[str, tuple[float, float, float]] = {}
    sources: dict[str, str] = {}
    confidence: dict[str, str] = {}
    for sector, row in _COARSE_DEMAND.items():
        low, central, high, source, conf = row
        values[sector] = (low, central, high)
        sources[sector] = source
        confidence[sector] = conf
    return ElasticitySet(
        provenance=prov,
        kind="demand",
        classification="coarse-sectors",
        values=values,
        sources=sources,
        confidence=confidence,
    )
