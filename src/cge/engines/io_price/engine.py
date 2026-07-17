"""Engine 1 — Leontief carbon-cost price model.

Implements the method specified to equation level in ``docs/models/io-price-model.md``.
The core result is that doc's equation (5):

    Δp = (I − Aᵀ)⁻¹ · τ · e

computed as a linear solve (not an explicit inverse), where ``e`` is direct emission
intensity and ``τ`` the carbon price. Decomposition into direct-vs-upstream uses the
Neumann series (equation 6). Assumptions emitted in the manifest match the doc's §3
verbatim, per the documentation standard.

Numerics: the dense NumPy solve is used (core dependency, exact for the small build). This
engine is **dense-only** and enforces a product cap (``MAX_DENSE_PRODUCTS``); the full
~9800² MRIO is NOT supported. A sparse path is intended but not implemented (see the model
doc §5). Run on a small/aggregated build.
"""

from __future__ import annotations

import numpy as np

from cge.constants import GWP100_AR5
from cge.contracts.data_objects import IOSystem, SatelliteAccount
from cge.contracts.engine import Capability, EngineMeta, registry
from cge.contracts.provenance import RunManifest
from cge.contracts.results import ResultSet
from cge.contracts.shocks import CarbonPrice, Shock

VERSION = "0.3.0"

# Assumptions from io-price-model.md §3 — printed on every result (GUI credibility). Kept in
# sync with the model doc; the doc references this dict as the source of truth (not a
# separate verbatim copy).
ASSUMPTIONS = {
    "model": "Leontief price model (cost-push), full supply-chain pass-through",
    "fixed_technology": "A held at base year; no input substitution",
    "full_cost_pass_through": "producers pass 100% of cost increases downstream",
    "price_formation": "cost-push (Leontief price dual), not demand-driven",
    "carbon_cost_basis": "per-unit cost on emissions in the selected GHG account (scope-1)",
    "linearity": "price system is linear; independent shocks add",
    "interpretation": (
        "Cost-pass-through price change under FIXED technology and FULL pass-through; no "
        "input substitution, demand response, or volume effect. Because substitution would "
        "let firms avoid some cost, this is expected to OVER-state the cost impact relative "
        "to a model with substitution — but it is not a proven upper bound over every model."
    ),
    "reference": "Miller & Blair (2009) §2.3-2.6",
}

# A carbon price in €/tonne is scaled by 1e-6 (M€ → €) so τ·e is a dimensionless cost share.
# Only valid when the monetary base is million-EUR (M€) and intensities are t/M€ — the engine
# asserts BOTH exactly (not just a '/MEUR' suffix: 'kg/MEUR' would be 1000× wrong) plus the
# currency, before applying it.
MEUR_TO_EUR = 1e-6
_MONETARY_UNIT = "MEUR"
_CURRENCY = "EUR"
# Exact intensity units accepted, per gas row. A physical gas must be tonnes/M€; the combined
# row must be tCO2e/M€. Anything else (kg/M€, g/M€, missing) is rejected.
_UNIT_PHYSICAL_GAS = "t/MEUR"
_UNIT_CO2E = "tCO2e/MEUR"

# Dense solve + dense eigvals are O(n³) and O(n²) memory; cap products so a full MRIO can't
# OOM. The small build (~40-60 sectors × ~10 regions) is well under this.
MAX_DENSE_PRODUCTS = 2000

# Combined-gas row name — used ONLY when a scenario explicitly selects it (or the default
# ["CO2e"]); never as a silent fallback for an unrecognised gas (that would tax all gases).
_CO2E_ROW = "CO2e"


def _assert_units(io: IOSystem, sat: SatelliteAccount) -> None:
    """Reject anything that would make the 1e-6 M€→€ cost-share scaling wrong.

    Requires: monetary unit exactly MEUR, currency exactly EUR, and every satellite row's
    unit exactly t/MEUR (physical gas) or tCO2e/MEUR (the CO2e row). A '/MEUR' *suffix* is
    insufficient — 'kg/MEUR' passes a suffix test but is 1000× off (review).
    """
    if io.unit != _MONETARY_UNIT:
        raise ValueError(
            f"io_price requires a {_MONETARY_UNIT} monetary base; build unit is {io.unit!r}."
        )
    if io.currency != _CURRENCY:
        raise ValueError(
            f"io_price applies euro-specific scaling; build currency is {io.currency!r}, "
            f"expected {_CURRENCY!r}."
        )
    if not sat.units:
        raise ValueError(f"satellite {sat.name!r} has no unit metadata; cannot verify t/MEUR.")
    for row in sat.data.index:
        unit = sat.units.get(row)
        expected = _UNIT_CO2E if row == _CO2E_ROW else _UNIT_PHYSICAL_GAS
        if unit != expected:
            raise ValueError(
                f"satellite row {row!r} has unit {unit!r}, expected {expected!r}; the M€→€ "
                f"cost-share scaling assumes exactly this unit."
            )


def _assert_labels_aligned(sat: SatelliteAccount, labels: list[str]) -> None:
    missing = [x for x in labels if x not in sat.data.columns]
    if missing:
        raise ValueError(
            f"Satellite {sat.name!r} is missing {len(missing)} product columns present in the "
            f"IO system, e.g. {missing[:5]}; intensities are not aligned."
        )


def _validate_gases(gases: list[str]) -> None:
    """Reject malformed gas selections (review): empty, duplicates, or mixing the aggregate
    CO2e row with component gases (which would double-count)."""
    if not gases:
        raise ValueError("gases must be a non-empty list")
    if len(set(gases)) != len(gases):
        raise ValueError(f"gases contains duplicates: {gases} (would multiply-count a gas)")
    if _CO2E_ROW in gases and len(gases) > 1:
        raise ValueError(
            f"cannot mix {_CO2E_ROW} with component gases {gases}: {_CO2E_ROW} already "
            f"aggregates them (double-counting)."
        )


def _gas_intensity(sat: SatelliteAccount, labels: list[str], gases: list[str]) -> np.ndarray:
    """CO2e intensity (tCO2e/M€) per label for exactly the requested ``gases``.

    Each named gas MUST be present as its own row (weighted by GWP-100), OR the request must
    be exactly the combined row ``["CO2e"]``. Malformed selections (empty, duplicate, or
    CO2e-mixed-with-components) and unknown/partially-missing gases all RAISE rather than
    silently misinterpreting (review).
    """
    _validate_gases(gases)
    available = set(sat.data.index)

    if gases == [_CO2E_ROW]:
        if _CO2E_ROW not in available:
            raise ValueError(
                f"Satellite {sat.name!r} has no {_CO2E_ROW} row; available: {sorted(available)}"
            )
        return sat.data.loc[_CO2E_ROW].reindex(labels).to_numpy(dtype=float)

    missing_gases = [g for g in gases if g not in available]
    if missing_gases:
        raise ValueError(
            f"Requested gases {missing_gases} not in satellite {sat.name!r} "
            f"(available: {sorted(available)}). Refusing to substitute the aggregate row."
        )
    acc = np.zeros(len(labels))
    for g in gases:
        weight = GWP100_AR5.get(g, 1.0)  # per-gas rows are physical tonnes of that gas
        acc += weight * sat.data.loc[g].reindex(labels).to_numpy(dtype=float)
    return acc


def _coverage_mask(shock: CarbonPrice, labels: list[str]) -> np.ndarray:
    """1.0 where the shock applies, 0.0 elsewhere (per label)."""
    mask = np.zeros(len(labels), dtype=float)
    for i, label in enumerate(labels):
        region, sector = label.split(":", 1)
        if shock.applies_to(sector, region):
            mask[i] = 1.0
    return mask


def carbon_cost_vector(
    shocks: list[CarbonPrice], sat: SatelliteAccount, labels: list[str], year: int
) -> tuple[np.ndarray, list[str]]:
    """Dimensionless direct carbon cost share per product for ``year``.

    Each shock contributes independently: its own price(year) × its own gases' intensity ×
    its own coverage. Contributions are then summed. This is the correct composition — the
    earlier version unioned gases and summed prices first, which cross-multiplied one shock's
    price against another shock's gas (review bug). The 1e-6 (M€→€) scaling makes the result a
    dimensionless share.
    """
    _assert_labels_aligned(sat, labels)
    cost = np.zeros(len(labels))
    descs: list[str] = []
    for s in shocks:
        intensity = _gas_intensity(sat, labels, s.gases)  # tCO2e/M€ (gases validated here)
        mask = _coverage_mask(s, labels)
        price = s.price_at(year)  # €/tonne
        cost += price * intensity * mask * MEUR_TO_EUR
        descs.append(f"€{price:g}/t on {s.gases}")
    return cost, descs


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

        # Dense-only: enforce the size cap FIRST — before to_numpy / eigvals / solve — so a
        # full ~9800² MRIO can't incur the OOM/runtime the guard exists to prevent.
        if len(labels) > MAX_DENSE_PRODUCTS:
            raise ValueError(
                f"io_price is dense-only and limited to ≤{MAX_DENSE_PRODUCTS} products; this "
                f"build has {len(labels)}. Use a small/aggregated build (a sparse path is not "
                f"yet implemented — see the model doc)."
            )

        # The 1e-6 cost-share scaling is valid ONLY for a million-EUR base in EUR with exact
        # t/MEUR (or tCO2e/MEUR) intensities. Check exactly — a '/MEUR' suffix is not enough
        # ('kg/MEUR' would be 1000× wrong); missing units are rejected too.
        _assert_units(io, sat)

        A = io.A.to_numpy(dtype=float)
        _assert_productive(A)  # per-run precondition; check once, then skip in the year loop

        carbon_shocks = [s for s in shocks if isinstance(s, CarbonPrice)]

        # revenue_recycling has no meaning in a pure cost-push price model (no household /
        # government budget). Reject it rather than silently ignoring a declared control.
        recycling = {s.revenue_recycling for s in carbon_shocks} - {"none"}
        if recycling:
            raise ValueError(
                f"io_price does not model revenue recycling ({sorted(recycling)}); it is a "
                f"cost-push price model. Use a CGE engine (Phase 5) for revenue recycling."
            )

        records = []
        intensity_desc: list[str] = []
        for year in years:
            carbon_cost, intensity_desc = carbon_cost_vector(carbon_shocks, sat, labels, year)
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
                "shock_contributions": intensity_desc,
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
