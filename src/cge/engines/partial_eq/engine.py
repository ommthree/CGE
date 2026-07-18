"""Engine 2 — partial-equilibrium production-volume response.

Implements ``docs/models/partial-equilibrium.md``. A carbon price raises prices (Engine 1);
demand elasticities turn those price changes into **final-demand** changes; those propagate
through the Leontief quantity system to **gross-output (production) volume** changes:

    Δy_i/y_i = (1 + Δp_i)^{ε_i} − 1          finite-change constant-elasticity demand response
    x = (I − A)^{-1} y                        production follows demand (Leontief quantity model)
    Δx = (I − A)^{-1} Δy   ⇒   report Δx/x    gross-output (production) volume change

The finite-change form (not the linear ε·Δp) keeps the response bounded and physically valid
even for the large price changes real carbon-price runs produce (review: linear ε·Δp gave an
impossible −142% on live data). Production propagation means a fall in final demand for a
downstream good pulls its upstream suppliers' output down too — the thing a per-good own-demand
response misses (review: that is the core "not actually production volume" defect).

Evaluated across low/central/high demand-elasticity bands → an uncertainty envelope. Prices
come from Engine 1 (single source of truth). Own-price demand only; the Armington
domestic/import substitution nest is specified in the model doc but **not implemented** (v1).

Emitted variables (long-format ``ResultSet``):
- ``price_change`` — from Engine 1 (band ``central``);
- ``final_demand_change`` — Δy/y per good, per band (the direct demand response);
- ``volume_change`` — Δx/x per good, per band (**production**, via Leontief propagation);
- ``elasticity_used`` — the ε applied (band ``central``).
The manifest records, per-parameter: each good's full applied (low, central, high) triple with
source/confidence/default-status; a content hash of the explicit elasticity table AND of the
*effective* triples actually applied (so a changed fallback band moves the manifest); and an
identity + content hash for every substantive input (IO system, satellite, elasticity set).
"""

from __future__ import annotations

import numpy as np

from cge.contracts.data_objects import ElasticitySet, IOSystem
from cge.contracts.engine import Capability, EngineMeta, registry
from cge.contracts.provenance import (
    RunManifest,
    content_hash,
    data_source_id,
    input_identity,
)
from cge.contracts.results import ResultSet
from cge.contracts.shocks import Shock
from cge.data.elasticities import DEFAULT_DEMAND_ELASTICITY, default_demand_set
from cge.engines.io_price.engine import ASSUMPTIONS as IO_ASSUMPTIONS
from cge.engines.io_price.engine import (
    MAX_DENSE_PRODUCTS,
    IOPriceEngine,
    _assert_productive,
    _df_fingerprint,
)

VERSION = "0.3.1"

# Classifications this engine can apply a demand set against by NAME (no formal concordance yet).
# The default set is on 'coarse-sectors'; a built system's aggregated sectors are named
# '<aggregation>-sectors' on the same coarse family. Anything else is rejected rather than
# silently matched by coincidental sector labels (review P2: an unrelated classification was
# accepted). A curated ConcordanceMap is the follow-up that generalises this.
_COMPATIBLE_ELASTICITY_CLASSIFICATIONS = frozenset({"coarse-sectors"})


def _classification_compatible(eset_classification: str, build_sector_name: str) -> bool:
    """True if a demand set on ``eset_classification`` may be applied by name to a build whose
    sector classification is ``build_sector_name``. Accepts the coarse-sector family (the default
    set and any 'coarse-*-sectors' aggregation) and an exact name match; rejects the rest."""
    if eset_classification in _COMPATIBLE_ELASTICITY_CLASSIFICATIONS:
        return True
    if eset_classification == build_sector_name:
        return True  # exact same classification — trivially compatible
    # A coarse aggregation build ('coarse-...-sectors') paired with a coarse-family set.
    return eset_classification.startswith("coarse") and "coarse" in build_sector_name


ASSUMPTIONS = {
    "model": (
        "partial-equilibrium production volume: finite-change demand response "
        "Δy/y=(1+Δp)^ε−1, propagated through Leontief x=(I−A)⁻¹y to give Δx/x (production)"
    ),
    "scope": "own-price demand only; NO Armington substitution (v1); no income effect, no GE",
    "prices_from": "Engine 1 (io_price), taken as exogenous",
    "uncertainty": "low/central/high demand-elasticity bands → volume envelope",
    "finite_change": "(1+Δp)^ε−1 used (not linear ε·Δp) so responses stay bounded > −100%",
    "interpretation": (
        "INDICATIVE production-volume response, not precise: magnitudes depend on assembled "
        "elasticities (no clean open database). Screening/stress use; cross-check with the CGE."
    ),
    "reference": "Leontief quantity model x=(I−A)⁻¹y [MillerBlair2009]; demand theory for Δy/y",
}


def _elasticity_row(sector: str, eset: ElasticitySet):
    """Return (low, central, high, source, confidence, is_default) for a sector."""
    if sector in eset.values:
        lo, ce, hi = eset.values[sector]
        return (
            lo,
            ce,
            hi,
            eset.sources.get(sector, "unspecified"),
            eset.confidence.get(sector, "medium"),
            False,
        )
    lo, ce, hi = DEFAULT_DEMAND_ELASTICITY
    return lo, ce, hi, "default (no assembled value)", "default", True


def _finite_demand_response(dp: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Δy/y = (1+Δp)^ε − 1, the finite-change constant-elasticity form. Bounded below by −1
    (a price rise cannot destroy more than 100% of demand), unlike linear ε·Δp."""
    # (1+Δp) is a price ratio ≥ 0 for any Δp ≥ −1; carbon-price Δp ≥ 0, so this is safe.
    return np.power(1.0 + dp, eps) - 1.0


class PartialEqEngine:
    """Partial-equilibrium production-volume response. Satisfies the ``Engine`` protocol."""

    meta = EngineMeta(
        name="partial_eq",
        version=VERSION,
        description="Partial-equilibrium production volume: demand response via Leontief.",
        capabilities=[Capability.PRICES, Capability.VOLUMES],
        supported_shocks=["carbon_price", "energy_price"],
        required_data=["IOSystem", "SatelliteAccount"],
    )

    def run(self, *, data: dict, shocks: list[Shock], years: list[int]) -> ResultSet:
        io: IOSystem = data["IOSystem"]
        labels = list(io.A.columns)

        # Dense-only: enforce the product cap FIRST — before to_numpy / eigvals / solve — so a
        # full ~9800² MRIO can't start an O(n³) eigendecomposition/solve the guard exists to
        # prevent (review P1b: the guard was reached only inside Engine 1, too late).
        if len(labels) > MAX_DENSE_PRODUCTS:
            raise ValueError(
                f"partial_eq is dense-only and limited to ≤{MAX_DENSE_PRODUCTS} products; this "
                f"build has {len(labels)}. Use a small/aggregated build."
            )

        A = io.A.to_numpy(dtype=float)
        _assert_productive(A)  # Leontief inverse must exist (same precondition as Engine 1)
        m = np.eye(A.shape[0]) - A  # (I − A); we solve against it rather than invert

        # Baseline final demand y and baseline gross output x = (I−A)⁻¹ y (via a linear solve).
        y0 = io.final_demand.sum(axis=1).reindex(labels).fillna(0.0).to_numpy(dtype=float)
        # Reject negative row-total final demand: the demand-ratio response is only meaningful on
        # positive behavioural demand; a negative entry (inventory/stat. adjustment) flips sign
        # when quantity "falls" (review P2). Also require positive baseline output for a ratio.
        if float(np.min(y0)) < 0.0:
            raise ValueError(
                "final demand has negative row totals; partial_eq's demand-ratio response is "
                "defined only for non-negative behavioural final demand (exclude inventory / "
                "statistical-adjustment categories)."
            )
        x0 = np.linalg.solve(m, y0)
        if float(np.min(x0)) <= 0.0:
            raise ValueError(
                "baseline gross output has non-positive entries; cannot form a volume ratio."
            )

        # Prices from Engine 1 (single source of truth). Keyed by (year, region, sector).
        price_df = IOPriceEngine().run(data=data, shocks=shocks, years=years).data
        prices = price_df[price_df["variable"] == "price_change"]

        # Demand elasticities: explicit set in ``data`` wins; else the default. Validated below
        # (kind, and classification-compatibility against this build's sector classification).
        eset: ElasticitySet = data.get("ElasticitySet") or default_demand_set()
        _validate_demand_elasticities(eset, io.sectors.name)

        # Resolve per-label elasticity rows + provenance once.
        rows = {lab: _elasticity_row(lab.split(":", 1)[1], eset) for lab in labels}
        default_goods = sorted(lab for lab, r in rows.items() if r[5])
        # The *effective* elasticity triple actually applied to each distinct sector — includes
        # values that came from the fallback. Hashing this (not just eset.values) means a changed
        # fallback band moves the manifest even though no explicit value changed (review P2).
        effective_triples = {
            lab.split(":", 1)[1]: (r[0], r[1], r[2])
            for lab, r in rows.items()  # rows keyed by full label; collapse to distinct sector
        }

        records: list[dict] = []
        for year in sorted(prices["year"].unique()):
            pyr = prices[prices["year"] == year]
            dp = {f"{r.region}:{r.sector}": float(r.value) for r in pyr.itertuples()}
            dp_vec = np.array([dp.get(lab, 0.0) for lab in labels])

            for band, idx in (("low", 0), ("central", 1), ("high", 2)):
                eps_vec = np.array([rows[lab][idx] for lab in labels])
                dy_frac = _finite_demand_response(dp_vec, eps_vec)  # Δy/y per good
                y_new = y0 * (1.0 + dy_frac)
                x_new = np.linalg.solve(m, y_new)  # production follows the new demand (solve)
                dx_frac = (x_new - x0) / x0  # Δx/x (production); x0 > 0 verified above

                for i, lab in enumerate(labels):
                    region, sector = lab.split(":", 1)
                    records.append(
                        _rec("final_demand_change", sector, region, year, band, dy_frac[i])
                    )
                    records.append(_rec("volume_change", sector, region, year, band, dx_frac[i]))
                    if band == "central":
                        records.append(
                            _rec("price_change", sector, region, year, "central", dp_vec[i])
                        )
                        records.append(
                            _rec("elasticity_used", sector, region, year, "central", rows[lab][1])
                        )

        manifest = RunManifest.build(
            engine_name=self.meta.name,
            engine_version=self.meta.version,
            data_source=data_source_id(io.provenance),
            scenario={"shocks": [s.model_dump(mode="json") for s in shocks], "years": years},
            assumptions={
                **ASSUMPTIONS,
                "price_model": IO_ASSUMPTIONS["model"],
                "price_interpretation": IO_ASSUMPTIONS["interpretation"],
                # Full elasticity provenance so results are reproducible from the manifest.
                "elasticity_set": {
                    "source": eset.provenance.source,
                    "source_version": eset.provenance.source_version,
                    "classification": eset.classification,
                    # Hash the explicit table AND the *effective* triples actually applied (which
                    # include fallback values) — so changing the fallback band, not just an
                    # explicit value, moves the manifest (review P2).
                    "content_hash": content_hash(
                        {k: list(v) for k, v in sorted(eset.values.items())}
                    ),
                    "effective_content_hash": content_hash(
                        {k: list(v) for k, v in sorted(effective_triples.items())}
                    ),
                },
                # Per-good elasticity applied: the FULL (low, central, high) triple (not just the
                # central), its source, confidence, and whether it is the default (review P2).
                "elasticity_per_good": {
                    sector: {
                        "low": lo,
                        "central": ce,
                        "high": hi,
                        "source": rows[next(la for la in labels if la.endswith(":" + sector))][3],
                        "confidence": rows[next(la for la in labels if la.endswith(":" + sector))][
                            4
                        ],
                        "default": rows[next(la for la in labels if la.endswith(":" + sector))][5],
                    }
                    for sector, (lo, ce, hi) in sorted(effective_triples.items())
                },
                "goods_using_default_elasticity": [g.split(":", 1)[1] for g in default_goods],
                "n_sectors_using_default": len({g.split(":", 1)[1] for g in default_goods}),
                "data_build_id": io.provenance.build_id,
                "data_generation": io.provenance.generation,
                # Every substantive input identified with a content hash (review P1): IO system,
                # satellite (drives the prices via Engine 1), and the elasticity set.
                "inputs": _pe_inputs(io, data.get("SatelliteAccount"), eset),
            },
        )
        return ResultSet.from_records(records, manifest)


def _validate_demand_elasticities(eset: ElasticitySet, build_sector_name: str) -> None:
    """The engine needs a *demand* set whose classification is compatible with the build it will
    be applied to. Value-level checks (finite, band-ordered, ≤0 for demand, per-value metadata)
    are enforced at ElasticitySet construction (contract, review P2). Here we check the kind and
    reject a classification that is not name-compatible with the build's sectors — otherwise an
    unrelated set is matched purely by coincidental sector labels (review P2)."""
    if eset.kind != "demand":
        raise ValueError(f"partial_eq needs a demand ElasticitySet, got kind={eset.kind!r}")
    if not _classification_compatible(eset.classification, build_sector_name):
        raise ValueError(
            f"ElasticitySet classification {eset.classification!r} is not compatible with this "
            f"build's sector classification {build_sector_name!r}; refusing to match elasticities "
            f"by coincidental sector name. Provide a set on a coarse-sector classification or add "
            f"a ConcordanceMap (follow-up). Accepted: "
            f"{sorted(_COMPATIBLE_ELASTICITY_CLASSIFICATIONS)} or a 'coarse*' classification."
        )


def _pe_inputs(io: IOSystem, sat, eset: ElasticitySet) -> list[dict]:
    """Reproducibility descriptors for every substantive input to a partial_eq run: the IO
    system, the satellite (prices come from Engine 1, which is driven by it), and the elasticity
    set. Each carries a content hash so a change in any of them moves the manifest (review P1).
    The IO fingerprint covers BOTH A and final demand — final demand sets the baseline y0 that
    drives the volume response, so a changed final demand must move the manifest (review P1)."""
    io_content = {"A": _df_fingerprint(io.A), "final_demand": _df_fingerprint(io.final_demand)}
    inputs = [input_identity("IOSystem", io.provenance, content=io_content)]
    if sat is not None:
        inputs.append(
            input_identity("SatelliteAccount", sat.provenance, content=_df_fingerprint(sat.data))
        )
    inputs.append(
        input_identity(
            "ElasticitySet",
            eset.provenance,
            content={k: list(v) for k, v in sorted(eset.values.items())},
        )
    )
    return inputs


def _rec(variable: str, sector: str, region: str, year: int, scenario: str, value: float) -> dict:
    return {
        "variable": variable,
        "sector": sector,
        "region": region,
        "year": year,
        "scenario": scenario,
        "value": float(value),
    }


registry.register(PartialEqEngine())
