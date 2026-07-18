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

from cge.contracts.data_objects import SAM, IOSystem
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
        # Accepts either a supplied SAM (toy pilot) or an IOSystem (a real build, from which the
        # SAM is built + quality-gated). Validated in _resolve_sam, so no hard required_data here.
        required_data=[],
    )

    def run(self, *, data: dict, shocks: list[Shock], years: list[int]) -> ResultSet:
        sam, sectors, sam_quality, e = _resolve_sam(data)
        factors = [f for f in _DEFAULT_FACTORS if f in sam.accounts]
        cal = calibrate(sam, sectors=sectors, factors=factors)

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
                "benchmark_gdp_normalised": cal.gdp0,
                # SAM credibility surface: worst quality severity + per-check summary, so a run
                # states how much the SAM data was helped (roadmap 5.1c). None when a SAM was
                # supplied directly (the toy pilot) rather than built here.
                "sam_quality": (
                    {"worst": sam_quality.worst.value, "summary": sam_quality.summary()}
                    if sam_quality is not None
                    else "supplied directly"
                ),
                "inputs": [
                    input_identity("SAM", sam.provenance, content=_sam_fingerprint(sam)),
                ],
            },
        )
        return ResultSet.from_records(records, manifest)


def _resolve_sam(data: dict):
    """Return ``(sam, sectors, sam_quality_or_None, emission_intensity)`` from ``data``.

    Two entry points: a ``SAM`` supplied directly (the toy pilot — no quality report, emission
    intensity read from ``data``), or an ``IOSystem`` (a real EXIOBASE build) from which the SAM is
    **built and quality-gated** here (5.1b) and per-sector emission intensities are derived from the
    satellite account. A SAM whose quality report FAILS (unbalanced / aggregates not preserved) is
    rejected — we do not calibrate on a bad SAM (mirrors the data-layer conservation gates)."""
    if "SAM" in data:
        sam: SAM = data["SAM"]
        factors = [f for f in _DEFAULT_FACTORS if f in sam.accounts]
        sectors = data.get("sectors") or _infer_sectors(sam, factors)
        return sam, sectors, None, _emission_intensity(data, sectors)

    io = data.get("IOSystem")
    if io is None:
        raise ValueError("cge_static needs a 'SAM' or an 'IOSystem' in data")
    from cge.data.sam import build_sam

    sam, quality, sectors = build_sam(io)
    if not quality.passed:
        failed = [c.name for c in quality.checks if c.severity.value == "fail"]
        raise ValueError(f"SAM quality gate failed for the build: {failed}; refusing to calibrate.")
    e = _emission_intensity_from_satellite(data.get("SatelliteAccount"), io, sectors)
    return sam, sectors, quality, e


def _emission_intensity_from_satellite(sat, io: IOSystem, sectors: list[str]) -> np.ndarray:
    """Per-sector emission intensity (emissions per unit gross output), aggregated over regions
    from the satellite's CO2 row. The satellite holds **intensities** (t per unit output), so the
    single-region intensity is the output-weighted mean of the regional intensities:
    ``Σ_r e[r,i]·x[r,i] / Σ_r x[r,i]``. Zero when no satellite/row is available (a carbon price then
    has no cost wedge, and the manifest's ``emissions_priced`` flag says so)."""
    if sat is None or sat.data.empty:
        return np.zeros(len(sectors))
    row = "CO2" if "CO2" in sat.data.index else sat.data.index[0]
    labels = list(io.A.columns)
    A = io.A.to_numpy(dtype=float)
    fd = io.final_demand.sum(axis=1).reindex(labels).fillna(0.0).to_numpy(dtype=float)
    x = np.linalg.solve(np.eye(A.shape[0]) - A, fd)  # gross output per label
    intensity = sat.data.loc[row].reindex(labels).fillna(0.0).to_numpy(dtype=float)  # per label
    s_index = {s: k for k, s in enumerate(sectors)}
    num = np.zeros(len(sectors))  # Σ e·x  (total emissions)
    den = np.zeros(len(sectors))  # Σ x    (total output)
    for lb, e_i, xi in zip(labels, intensity, x, strict=True):
        k = s_index[lb.split(":", 1)[1]]
        num[k] += e_i * xi
        den[k] += xi
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


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
