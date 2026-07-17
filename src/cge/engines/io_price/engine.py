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

from cge.constants import GWP100_AR5
from cge.contracts.data_objects import IOSystem, SatelliteAccount
from cge.contracts.engine import Capability, EngineMeta, registry
from cge.contracts.provenance import RunManifest
from cge.contracts.results import ResultSet
from cge.contracts.shocks import CarbonPrice, Shock

VERSION = "0.2.0"

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

# Monetary base is million EUR (EXIOBASE, and the toy fixture by convention); intensities
# are tonnes per M€. A carbon price in €/tonne must therefore be scaled by 1e-6 (M€ → €) so
# that τ·e is a dimensionless cost share (fraction of unit value). This is the units fix.
MEUR_TO_EUR = 1e-6

# Combined-gas row name used when a scenario selects all/default gases.
_CO2E_ROW = "CO2e"


def _intensity_for_gases(
    sat: SatelliteAccount, labels: list[str], gases: list[str]
) -> tuple[np.ndarray, str]:
    """Return (GWP-weighted intensity per label in tCO2e/M€, description) for the selected
    gases. Honours the scenario's ``gases``: sums the named per-gas rows with GWP-100 weights.

    Falls back to the combined ``CO2e`` row only when the requested gases aren't individually
    present but CO2e is (e.g. the toy fixture). Raises if nothing usable is found — silently
    zeroing would hide a data/scenario mismatch.
    """
    available = list(sat.data.index)

    # A product missing from the satellite is an alignment error, not a zero-emission
    # product; refuse to silently zero-fill it (review).
    missing_labels = [x for x in labels if x not in sat.data.columns]
    if missing_labels:
        raise ValueError(
            f"Satellite {sat.name!r} is missing {len(missing_labels)} product columns present "
            f"in the IO system, e.g. {missing_labels[:5]}; intensities are not aligned."
        )

    present = [g for g in gases if g in available]
    if present:
        acc = np.zeros(len(labels))
        for g in present:
            weight = GWP100_AR5.get(g, 1.0)  # per-gas rows are physical tonnes of that gas
            acc += weight * sat.data.loc[g].reindex(labels).to_numpy(dtype=float)
        return acc, f"gases={present} (GWP-100 weighted)"

    if _CO2E_ROW in available:
        series = sat.data.loc[_CO2E_ROW].reindex(labels)
        return series.to_numpy(
            dtype=float
        ), f"{_CO2E_ROW} (requested {gases} not individually present)"

    raise ValueError(
        f"Satellite {sat.name!r} has none of the requested gases {gases} nor a {_CO2E_ROW} "
        f"row; available: {available}"
    )


def _effective_price(shocks: list[CarbonPrice], labels: list[str], year: int) -> np.ndarray:
    """Per-label carbon price τ (currency/tonne) in ``year``. Multiple carbon shocks add
    where they overlap (linearity); each shock reads its own time path via ``price_at``.

    A shock with no coverage applies everywhere; with coverage, only to matching labels.
    """
    tau = np.zeros(len(labels), dtype=float)
    for i, label in enumerate(labels):
        region, sector = label.split(":", 1)
        tau[i] = sum(s.price_at(year) for s in shocks if s.applies_to(sector, region))
    return tau


# Tolerance for the "negative coefficient" admissibility check. EXIOBASE can carry tiny
# negatives (rounding, stock changes); anything past this magnitude breaks the non-negative
# pass-through guarantee and is rejected (see review: a negative A entry lets a positive tax
# *lower* another sector's price).
NEG_COEFF_TOL = 1e-9


def _assert_productive(A: np.ndarray) -> None:
    """Raise unless A is *admissible* for the Leontief price model:

    1. Effectively non-negative (A ≥ −NEG_COEFF_TOL). The non-negativity of the Leontief
       inverse — and hence the "pass-through only adds cost" guarantee — requires A ≥ 0.
    2. Productive: ρ(A) < 1, so (I − Aᵀ)⁻¹ exists.

    Both are preconditions of the model doc; checking them here makes the engine safe on any
    data, not just data that happened to pass the build gate.
    """
    min_entry = float(A.min()) if A.size else 0.0
    if min_entry < -NEG_COEFF_TOL:
        raise ValueError(
            f"A has negative entries (min {min_entry:.3e}); the non-negative Leontief "
            f"pass-through guarantee does not hold. Reject or preprocess this build."
        )
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    if not rho < 1.0:
        raise ValueError(
            f"economy not productive: ρ(A) = {rho:.4f} ≥ 1; Leontief inverse does not exist"
        )


def price_change(
    A: np.ndarray, carbon_cost: np.ndarray, *, check_productive: bool = True
) -> np.ndarray:
    """Solve (I − Aᵀ) Δp = c for Δp — equation (5), as a linear solve.

    ``carbon_cost`` is the dimensionless direct cost share per product. With ``check_productive``
    (default), asserts A is admissible (non-negative and productive) first — see
    ``_assert_productive``. Callers that already checked pass ``check_productive=False``.
    """
    if check_productive:
        _assert_productive(A)
    n = A.shape[0]
    M = np.eye(n) - A.T
    try:
        return np.linalg.solve(M, carbon_cost)
    except np.linalg.LinAlgError as exc:  # singular ⇒ ρ(A) = 1 exactly
        raise ValueError("(I − Aᵀ) is singular; economy not productive (ρ(A) = 1)") from exc


def decompose(
    A: np.ndarray, carbon_cost: np.ndarray, tiers: int = 3, *, check_productive: bool = True
) -> dict[str, np.ndarray]:
    """Neumann-series decomposition (equation 6): direct term plus ``tiers`` upstream tiers,
    and the residual tail. Returns per-label vectors; they sum to the full Δp.

    These are *aggregate tier contributions*, not enumerated supply-chain paths — full
    structural path analysis (path-level enumeration) is a separate, heavier method.
    """
    if check_productive:
        _assert_productive(A)
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
        # Productivity is a per-run precondition; check once, then skip in the year loop.
        _assert_productive(A)

        carbon_shocks = [s for s in shocks if isinstance(s, CarbonPrice)]
        # Gas selection is the union of gases named across carbon shocks (default CO2).
        gases = sorted({g for s in carbon_shocks for g in s.gases}) or ["CO2"]
        intensity, intensity_desc = _intensity_for_gases(sat, labels, gases)

        # revenue_recycling has no meaning in a pure cost-push price model (no household /
        # government budget). Reject it rather than silently ignoring a declared control.
        recycling = {s.revenue_recycling for s in carbon_shocks} - {"none"}
        if recycling:
            raise ValueError(
                f"io_price does not model revenue recycling ({sorted(recycling)}); it is a "
                f"cost-push price model. Use a CGE engine (Phase 5) for revenue recycling."
            )

        records = []
        for year in years:
            tau = _effective_price(carbon_shocks, labels, year)  # €/tonne, per year
            carbon_cost = tau * intensity * MEUR_TO_EUR  # dimensionless cost share
            dp = price_change(A, carbon_cost, check_productive=False)
            parts = decompose(A, carbon_cost, tiers=3, check_productive=False)
            for i, label in enumerate(labels):
                region, sector = label.split(":", 1)
                records.append(_rec("price_change", sector, region, year, dp[i]))
                for part_name, vec in parts.items():
                    records.append(_rec(f"price_change_{part_name}", sector, region, year, vec[i]))

        manifest = RunManifest.build(
            engine_name=self.meta.name,
            engine_version=self.meta.version,
            data_source=io.provenance.build_id
            or f"{io.provenance.source} {io.provenance.source_version}",
            scenario={"shocks": [s.model_dump(mode="json") for s in shocks], "years": years},
            assumptions={
                **ASSUMPTIONS,
                "gases_selected": gases,
                "intensity_source": intensity_desc,
                "monetary_unit": io.unit,
                "unit_scaling": "τ(€/t)·e(t/MEUR)·1e-6 → dimensionless cost share",
                "value_meaning": "Δp is a fractional change in unit price index (base p₀=1)",
                "n_products": len(labels),
                "carbon_shocks": len(carbon_shocks),
                "decomposition_tiers": 3,
                "data_build_id": io.provenance.build_id,
                "reference_year": io.provenance.reference_year,
            },
        )
        return ResultSet.from_records(records, manifest)


def _rec(variable: str, sector: str, region: str, year: int, value: float) -> dict:
    return {
        "variable": variable,
        "sector": sector,
        "region": region,
        "year": year,
        "scenario": "central",
        "value": float(value),
    }


registry.register(IOPriceEngine())
