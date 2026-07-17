"""Engine 1 — Leontief carbon-cost price model.

Implements the method specified to equation level in ``docs/models/io-price-model.md``.
The core result is that doc's equation (5):

    Δp = (I − Aᵀ)⁻¹ · τ · e

computed as a linear solve (not an explicit inverse), where ``e`` is direct emission
intensity and ``τ`` the carbon price. Decomposition into direct-vs-upstream uses the
Neumann series (equation 6). Assumptions emitted in the manifest match the doc's §3
verbatim, per the documentation standard.

Numerics: the dense NumPy solve is used (core dependency, exact for the small build).
For the full ~9800² MRIO a sparse ``scipy.sparse.linalg.spsolve`` is the drop-in path;
gated behind the ``[cge]`` extra so the core install stays light (see ADR-0003).
"""

from __future__ import annotations

import numpy as np

from cge.contracts.data_objects import IOSystem, SatelliteAccount
from cge.contracts.engine import Capability, EngineMeta, registry
from cge.contracts.provenance import RunManifest
from cge.contracts.results import ResultSet
from cge.contracts.shocks import CarbonPrice, Shock

VERSION = "0.1.0"

# Assumptions from io-price-model.md §3 — printed on every result (GUI credibility).
ASSUMPTIONS = {
    "model": "Leontief price model (cost-push), full supply-chain pass-through",
    "fixed_technology": "A held at base year; no input substitution",
    "full_cost_pass_through": "producers pass 100% of cost increases downstream",
    "price_formation": "cost-push (Leontief price dual), not demand-driven",
    "carbon_cost_basis": "per-unit cost on direct (scope-1) emissions by default",
    "linearity": "price system is linear; independent shocks add",
    "interpretation": "UPPER BOUND on cost impact; NO volume effects (see Engine 2)",
    "reference": "Miller & Blair (2009) §2.3-2.6",
}

# Emission-intensity row to use, in priority order. Real EXIOBASE builds expose 'CO2e'
# (GWP-weighted); the toy fixture and some builds only have 'CO2'.
_INTENSITY_ROWS = ("CO2e", "CO2")


def _intensity_vector(sat: SatelliteAccount, labels: list[str]) -> tuple[np.ndarray, str]:
    """Return (intensity per label, row name used). Raises if no usable GHG row exists."""
    for row in _INTENSITY_ROWS:
        if row in sat.data.index:
            series = sat.data.loc[row].reindex(labels).fillna(0.0)
            return series.to_numpy(dtype=float), row
    raise ValueError(
        f"Satellite {sat.name!r} has no emission-intensity row in {_INTENSITY_ROWS}; "
        f"available: {list(sat.data.index)}"
    )


def _effective_price(shocks: list[CarbonPrice], labels: list[str]) -> np.ndarray:
    """Per-label carbon price τ. Multiple carbon shocks add where they overlap (linearity).

    A shock with no coverage applies everywhere; with coverage, only to matching labels.
    """
    tau = np.zeros(len(labels), dtype=float)
    for i, label in enumerate(labels):
        region, sector = label.split(":", 1)
        tau[i] = sum(s.price for s in shocks if s.applies_to(sector, region))
    return tau


def price_change(
    A: np.ndarray, carbon_cost: np.ndarray, *, check_productive: bool = True
) -> np.ndarray:
    """Solve (I − Aᵀ) Δp = c for Δp — equation (5), as a linear solve.

    ``carbon_cost`` is c = τ·e per unit output. Raises if the system is not productive
    (ρ(A) ≥ 1): the Leontief inverse then does not exist as a non-negative matrix, so the
    solve — even if numerically successful — is meaningless (spec §4). The data-layer build
    gate should already prevent this; the explicit check makes the engine safe standalone.
    """
    if check_productive:
        rho = float(np.max(np.abs(np.linalg.eigvals(A))))
        if not rho < 1.0:
            raise ValueError(
                f"economy not productive: ρ(A) = {rho:.4f} ≥ 1; Leontief inverse does not exist"
            )
    n = A.shape[0]
    M = np.eye(n) - A.T
    try:
        return np.linalg.solve(M, carbon_cost)
    except np.linalg.LinAlgError as exc:  # singular ⇒ ρ(A) = 1 exactly
        raise ValueError("(I − Aᵀ) is singular; economy not productive (ρ(A) = 1)") from exc


def decompose(A: np.ndarray, carbon_cost: np.ndarray, tiers: int = 3) -> dict[str, np.ndarray]:
    """Neumann-series decomposition (equation 6): direct term plus ``tiers`` upstream tiers,
    and the residual tail. Returns per-label vectors; they sum to the full Δp."""
    out: dict[str, np.ndarray] = {}
    term = carbon_cost.copy()  # tier 0: direct = τ·e
    out["direct"] = term.copy()
    AT = A.T
    cumulative = term.copy()
    for t in range(1, tiers + 1):
        term = AT @ term
        out[f"upstream_tier_{t}"] = term.copy()
        cumulative = cumulative + term
    full = price_change(A, carbon_cost, check_productive=False)  # already checked by caller
    out["upstream_residual"] = full - cumulative  # everything beyond the truncation
    return out


class IOPriceEngine:
    """Leontief carbon-cost price model. Satisfies the ``Engine`` protocol."""

    meta = EngineMeta(
        name="io_price",
        version=VERSION,
        description="Leontief carbon-cost pass-through: Δprice of every good under a carbon price.",
        capabilities=[Capability.PRICES],
        supported_shocks=["carbon_price"],
        required_data=["IOSystem", "SatelliteAccount"],
    )

    def run(self, *, data: dict, shocks: list[Shock], years: list[int]) -> ResultSet:
        io: IOSystem = data["IOSystem"]
        sat: SatelliteAccount = data["SatelliteAccount"]
        labels = list(io.A.columns)
        A = io.A.to_numpy(dtype=float)

        carbon_shocks = [s for s in shocks if isinstance(s, CarbonPrice)]
        intensity, intensity_row = _intensity_vector(sat, labels)
        tau = _effective_price(carbon_shocks, labels)
        carbon_cost = tau * intensity  # c = τ·e per label

        dp = price_change(A, carbon_cost)
        parts = decompose(A, carbon_cost, tiers=3)

        records = []
        for year in years:
            for i, label in enumerate(labels):
                region, sector = label.split(":", 1)
                records.append(
                    {
                        "variable": "price_change",
                        "sector": sector,
                        "region": region,
                        "year": year,
                        "scenario": "central",
                        "value": float(dp[i]),
                    }
                )
                # Decomposition rows so the GUI can build a waterfall per good.
                for part_name, vec in parts.items():
                    records.append(
                        {
                            "variable": f"price_change_{part_name}",
                            "sector": sector,
                            "region": region,
                            "year": year,
                            "scenario": "central",
                            "value": float(vec[i]),
                        }
                    )

        manifest = RunManifest.build(
            engine_name=self.meta.name,
            engine_version=self.meta.version,
            data_source=f"{io.provenance.source} {io.provenance.source_version}",
            scenario={"shocks": [s.model_dump(mode="json") for s in shocks], "years": years},
            assumptions={
                **ASSUMPTIONS,
                "intensity_row_used": intensity_row,
                "n_products": len(labels),
                "carbon_shocks": len(carbon_shocks),
                "decomposition_tiers": 3,
            },
        )
        return ResultSet.from_records(records, manifest)


registry.register(IOPriceEngine())
