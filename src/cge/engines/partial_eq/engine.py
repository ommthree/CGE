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
Per-good elasticity source/confidence/default-status and a content hash of the elasticity
values go into the manifest (per-parameter provenance).
"""

from __future__ import annotations

import numpy as np

from cge.contracts.data_objects import ElasticitySet, IOSystem
from cge.contracts.engine import Capability, EngineMeta, registry
from cge.contracts.provenance import RunManifest, content_hash
from cge.contracts.results import ResultSet
from cge.contracts.shocks import Shock
from cge.data.elasticities import DEFAULT_DEMAND_ELASTICITY, default_demand_set
from cge.engines.io_price.engine import ASSUMPTIONS as IO_ASSUMPTIONS
from cge.engines.io_price.engine import IOPriceEngine, _assert_productive

VERSION = "0.2.0"

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
        supported_shocks=["carbon_price"],
        required_data=["IOSystem", "SatelliteAccount"],
    )

    def run(self, *, data: dict, shocks: list[Shock], years: list[int]) -> ResultSet:
        io: IOSystem = data["IOSystem"]
        labels = list(io.A.columns)
        A = io.A.to_numpy(dtype=float)
        _assert_productive(A)  # Leontief inverse must exist (same precondition as Engine 1)
        leontief = np.linalg.inv(np.eye(A.shape[0]) - A)

        # Baseline final demand y and baseline gross output x = (I−A)⁻¹ y.
        y0 = io.final_demand.sum(axis=1).reindex(labels).fillna(0.0).to_numpy(dtype=float)
        x0 = leontief @ y0

        # Prices from Engine 1 (single source of truth). Keyed by (year, region, sector).
        price_df = IOPriceEngine().run(data=data, shocks=shocks, years=years).data
        prices = price_df[price_df["variable"] == "price_change"]

        # Demand elasticities: explicit set in ``data`` wins; else the default. Validated below.
        eset: ElasticitySet = data.get("ElasticitySet") or default_demand_set()
        _validate_demand_elasticities(eset)

        # Resolve per-label elasticity rows + provenance once.
        rows = {lab: _elasticity_row(lab.split(":", 1)[1], eset) for lab in labels}
        default_goods = sorted(lab for lab, r in rows.items() if r[5])

        records: list[dict] = []
        for year in sorted(prices["year"].unique()):
            pyr = prices[prices["year"] == year]
            dp = {f"{r.region}:{r.sector}": float(r.value) for r in pyr.itertuples()}
            dp_vec = np.array([dp.get(lab, 0.0) for lab in labels])

            for band, idx in (("low", 0), ("central", 1), ("high", 2)):
                eps_vec = np.array([rows[lab][idx] for lab in labels])
                dy_frac = _finite_demand_response(dp_vec, eps_vec)  # Δy/y per good
                y_new = y0 * (1.0 + dy_frac)
                x_new = leontief @ y_new  # production follows the new demand
                with np.errstate(divide="ignore", invalid="ignore"):
                    dx_frac = np.where(x0 != 0, (x_new - x0) / x0, 0.0)  # Δx/x (production)

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
            data_source=io.provenance.build_id
            or f"{io.provenance.source} {io.provenance.source_version}",
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
                    # Canonical content hash: two different tables → different manifests.
                    "content_hash": content_hash(
                        {k: list(v) for k, v in sorted(eset.values.items())}
                    ),
                },
                # Per-good elasticity used, its source, confidence, and whether it is the default.
                "elasticity_per_good": {
                    lab.split(":", 1)[1]: {
                        "central": rows[lab][1],
                        "source": rows[lab][3],
                        "confidence": rows[lab][4],
                        "default": rows[lab][5],
                    }
                    for lab in labels
                    # one entry per distinct sector (labels repeat sectors across regions)
                    if lab == next(x for x in labels if x.split(":", 1)[1] == lab.split(":", 1)[1])
                },
                "goods_using_default_elasticity": [g.split(":", 1)[1] for g in default_goods],
                "n_sectors_using_default": len({g.split(":", 1)[1] for g in default_goods}),
                "data_build_id": io.provenance.build_id,
            },
        )
        return ResultSet.from_records(records, manifest)


def _validate_demand_elasticities(eset: ElasticitySet) -> None:
    """The engine needs a *demand* set. Value-level checks (finite, band-ordered, ≤0 for
    demand, per-value metadata) are enforced at ElasticitySet construction (contract, review
    P2); here we only check the kind is right for this engine."""
    if eset.kind != "demand":
        raise ValueError(f"partial_eq needs a demand ElasticitySet, got kind={eset.kind!r}")


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
