"""Engine 3 — static computable general equilibrium (roadmap Phase 5).

Wraps the calibrated pilot CGE (``calibrate`` + ``model`` + ``solver``) behind the ``Engine``
protocol, so the GUI/CLI pick it up via the registry with no changes. Given a benchmark ``SAM``,
it calibrates the model to reproduce the base year exactly, applies a ``CarbonPrice`` as a
per-unit emissions cost wedge, solves for the new equilibrium, and emits a ``ResultSet`` of price
and volume changes plus GE-specific outputs (factor prices, GDP, deflator).

**Pilot scope (single region, one household, `none` recycling):** the model is the small,
correctness-first pilot from `docs/phase-5-plan.md` §5.2a — Leontief intermediates, Cobb-Douglas
value added and household demand, CPI numéraire. It passes benchmark replication, homogeneity and
Walras (the `cge_static` validation suite). Armington trade, multiple regions, revenue recycling
and a full EXIOBASE SAM are the next sub-phases (5.1b/5.3); this engine is deliberately the
provable core, not yet the production model.

Data contract (``data`` dict): ``SAM`` (required) and an optional ``emission_intensity`` — a dict
``sector -> tCO2e per unit output`` (or a numpy vector aligned to the SAM sectors). Absent ⇒ zero
emissions ⇒ a carbon price has no cost wedge (and the engine says so in the manifest).
"""

from __future__ import annotations

import numpy as np

from cge.contracts.data_objects import SAM
from cge.contracts.engine import Capability, EngineMeta, registry
from cge.contracts.provenance import RunManifest, data_source_id, input_identity
from cge.contracts.results import ResultSet
from cge.contracts.shocks import CarbonPrice, Shock
from cge.engines.cge_static import model as M
from cge.engines.cge_static.calibrate import calibrate
from cge.engines.cge_static.solver import solve

VERSION = "0.1.0"

# Default factor accounts for the pilot SAM (capital, labour). The engine treats every SAM
# account that is neither a factor nor the single institution as a sector.
_DEFAULT_FACTORS = ("CAP", "LAB")

ASSUMPTIONS = {
    "model": (
        "static CGE pilot: Leontief intermediates + Cobb-Douglas value added and household "
        "demand; fixed factor endowments; CPI numéraire"
    ),
    "scope": (
        "single region, one representative household, `none` revenue recycling; Armington trade / "
        "multi-region / recycling are later sub-phases (5.1b/5.3)"
    ),
    "carbon_price": "per-unit emissions cost wedge τ·e[i] in the zero-profit condition",
    "closure": "savings-less pilot; fixed factor supply; numéraire = consumer price index (CPI=1)",
    "solver_rule": "non-optimal solve raises (well-posedness); backend + status recorded",
    "interpretation": (
        "GENERAL-EQUILIBRIUM price and volume response with factor-market feedback and input "
        "substitution via the CD value-added nest — the mechanism Engines 1-2 cannot capture. "
        "Indicative magnitudes (pilot calibration); brackets Engine 1 prices, same-sign Engine 2 "
        "volumes."
    ),
    "reference": "Hosoe, Gasawa & Hashimoto (2010), Textbook of CGE Modeling [Hosoe2010]",
}


def _emission_intensity(data: dict, sectors: list[str]) -> np.ndarray:
    """Per-sector emission intensity (tCO2e per unit output) aligned to ``sectors``. Reads a dict
    or vector from ``data['emission_intensity']``; defaults to zero (no carbon cost wedge)."""
    ei = data.get("emission_intensity")
    if ei is None:
        return np.zeros(len(sectors))
    if isinstance(ei, dict):
        return np.array([float(ei.get(s, 0.0)) for s in sectors])
    arr = np.asarray(ei, dtype=float)
    if arr.shape != (len(sectors),):
        raise ValueError(
            f"emission_intensity must have one value per sector ({len(sectors)}), got {arr.shape}"
        )
    return arr


class CGEStaticEngine:
    """Static CGE pilot. Satisfies the ``Engine`` protocol."""

    meta = EngineMeta(
        name="cge_static",
        version=VERSION,
        description="Static CGE pilot: GE price + volume response with factor-market feedback.",
        capabilities=[Capability.GENERAL_EQUILIBRIUM, Capability.PRICES, Capability.VOLUMES],
        supported_shocks=["carbon_price"],
        required_data=["SAM"],
    )

    def run(self, *, data: dict, shocks: list[Shock], years: list[int]) -> ResultSet:
        sam: SAM = data["SAM"]
        factors = [f for f in _DEFAULT_FACTORS if f in sam.accounts]
        sectors = data.get("sectors") or _infer_sectors(sam, factors)
        cal = calibrate(sam, sectors=sectors, factors=factors)
        e = _emission_intensity(data, sectors)

        carbon_shocks = [s for s in shocks if isinstance(s, CarbonPrice)]
        recycling = {s.revenue_recycling for s in carbon_shocks} - {"none"}
        if recycling:
            raise ValueError(
                f"cge_static pilot supports only `none` recycling; got {sorted(recycling)} "
                f"(lump_sum / labour_tax_cut land in sub-phase 5.3)."
            )

        # Benchmark solve (zero shock) — the replication point, and the base for % changes.
        ns = len(sectors)
        base_sol = _solve(cal, carbon_cost=np.zeros(ns))
        base = M.derive_state(cal, base_sol.x[:ns], base_sol.x[ns:])

        records: list[dict] = []
        backends: set[str] = {base_sol.backend}
        for year in years:
            tau = sum(s.price_at(year) for s in carbon_shocks)  # €/tCO2e (pilot: sum of shocks)
            cc = tau * e
            sol = _solve(cal, carbon_cost=cc)
            backends.add(sol.backend)
            st = M.derive_state(cal, sol.x[:ns], sol.x[ns:])
            _emit(records, cal, base, st, year)

        manifest = RunManifest.build(
            engine_name=self.meta.name,
            engine_version=self.meta.version,
            data_source=data_source_id(sam.provenance),
            scenario={"shocks": [s.model_dump(mode="json") for s in shocks], "years": years},
            assumptions={
                **ASSUMPTIONS,
                "sectors": sectors,
                "factors": factors,
                "solver_backends": sorted(backends),
                "emissions_priced": bool(np.any(e != 0.0)),
                "benchmark_gdp": cal.gdp0,
                "inputs": [
                    input_identity("SAM", sam.provenance, content=_sam_fingerprint(sam)),
                ],
            },
        )
        return ResultSet.from_records(records, manifest)


def _infer_sectors(sam: SAM, factors: list[str]) -> list[str]:
    """Sectors = SAM accounts that are neither factors nor the single institution (household)."""
    non_factor = [a for a in sam.accounts if a not in factors]
    # The institution is the account with no value-added-style column into factors; simplest for
    # the pilot: assume exactly one institution and take it as the last non-sector. We identify it
    # as the account that receives from factors (a factor row pays it).
    institutions = [a for a in non_factor if any(sam.matrix.loc[a, f] != 0 for f in factors)]
    return [a for a in non_factor if a not in institutions]


def _solve(cal, *, carbon_cost):
    return solve(
        lambda z: M.residuals(cal, z, carbon_cost=carbon_cost),
        M.initial_guess(cal),
        prefer=None,
    )


def _emit(records, cal, base, st, year: int) -> None:
    """Append price/volume changes and GE outputs (relative to the benchmark) for one year."""
    for i, sector in enumerate(cal.sectors):
        records.append(_rec("price_change", sector, year, st.p[i] / base.p[i] - 1.0))
        records.append(_rec("volume_change", sector, year, st.X[i] / base.X[i] - 1.0))
    for f_idx, factor in enumerate(cal.factors):
        records.append(_rec("factor_price_change", factor, year, st.w[f_idx] / base.w[f_idx] - 1.0))
    # Real GDP = final demand valued at BENCHMARK prices (all 1) = Σ FD[i]; the CPI numéraire
    # makes this the real quantity index. Nominal GDP change and the deflator follow from prices.
    real_gdp = float(st.FD.sum())
    nom_gdp = float(np.dot(st.p, st.FD))
    deflator = nom_gdp / real_gdp - 1.0 if real_gdp > 0 else 0.0
    records.append(_rec("gdp_change_real", "__economy__", year, real_gdp / cal.gdp0 - 1.0))
    records.append(_rec("gdp_change", "__economy__", year, nom_gdp / cal.gdp0 - 1.0))
    records.append(_rec("deflator", "__economy__", year, deflator))


def _rec(variable: str, sector: str, year: int, value: float) -> dict:
    return {
        "variable": variable,
        "sector": sector,
        "region": "R",
        "year": year,
        "scenario": "central",
        "value": float(value),
    }


def _sam_fingerprint(sam: SAM) -> dict:
    m = sam.matrix
    return {
        "accounts": list(sam.accounts),
        "values": [round(float(v), 10) for v in m.to_numpy(dtype=float).ravel().tolist()],
    }


registry.register(CGEStaticEngine())
