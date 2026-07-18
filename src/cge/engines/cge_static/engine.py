"""Engine 3 — static computable general equilibrium (roadmap Phase 5).

Wraps the calibrated pilot CGE (``calibrate`` + ``model`` + ``solver``) behind the ``Engine``
protocol, so the GUI/CLI pick it up via the registry with no changes. Given a benchmark ``SAM``,
it calibrates the model to reproduce the base year exactly, applies a ``CarbonPrice`` as a
per-unit emissions cost wedge, solves for the new equilibrium, and emits a ``ResultSet`` of price
and volume changes plus GE-specific outputs (factor prices, GDP, deflator).

**Pilot scope (single region, one household):** the model is the small, correctness-first pilot
from `docs/phase-5-plan.md` §5.2a — Leontief intermediates, Cobb-Douglas value added and household
demand, CPI numéraire, with **revenue recycling** (lump_sum / labour_tax_cut). It passes benchmark
replication, homogeneity, Walras and the recycling-effect checks (the `cge_static` validation
suite). Armington trade and multiple regions are the next sub-phases; this engine is the provable
core, not yet the production model.

Data contract (``data`` dict): either a ``SAM`` supplied directly (validated: aligned, finite,
non-negative, balanced) with an optional per-sector dimensionless ``carbon_cost_share``, OR an
``IOSystem`` (+ ``SatelliteAccount``) — a real build — from which the SAM is built + quality-gated
and the carbon cost is derived the SAME way as Engine 1 (gases, coverage, and the 1e-6 M→currency
scaling). Emission provenance (satellite identity + effective cost-share hash) is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cge.contracts.data_objects import SAM, IOSystem, SatelliteAccount
from cge.contracts.engine import Capability, EngineMeta, registry
from cge.contracts.provenance import RunManifest, content_hash, data_source_id, input_identity
from cge.contracts.results import ResultSet
from cge.contracts.shocks import CarbonPrice, Shock
from cge.engines.cge_static import model as M
from cge.engines.cge_static.calibrate import calibrate
from cge.engines.cge_static.solver import solve

VERSION = "0.3.0"

# Default factor accounts for the pilot SAM (capital, labour). The engine treats every SAM
# account that is neither a factor nor the single institution as a sector.
_DEFAULT_FACTORS = ("CAP", "LAB")

ASSUMPTIONS = {
    "model": (
        "static CGE pilot: Leontief intermediates + Cobb-Douglas value added and household "
        "demand; fixed factor endowments; CPI numéraire"
    ),
    "scope": (
        "single region, one representative household; revenue recycling supported "
        "(none/lump_sum/labour_tax_cut, the last two equivalent with one household); "
        "Armington trade / multi-region are later sub-phases"
    ),
    "carbon_price": "per-unit emissions cost wedge τ·e[i] in the zero-profit condition",
    "revenue_recycling": (
        "carbon revenue R = Σ τ·e[i]·X[i]; none = revenue leaves the economy; "
        "lump_sum/labour_tax_cut = returned to the household (offsets the welfare loss)"
    ),
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


def _carbon_cost_share(data: dict, sectors: list[str]) -> np.ndarray | None:
    """Per-sector **dimensionless carbon cost-share** supplied directly (the toy pilot path).

    Reads ``data['carbon_cost_share']`` — the cost added to a sector's unit price *per €1 of carbon
    price*, i.e. already the τ=1 cost wedge (dimensionless, 1e-6-scaled if it came from real data).
    The engine multiplies it by the scenario's τ. This replaces the old raw ``emission_intensity``
    (t/MEUR), which was applied *without* the M€→€ conversion and so was ~1e6 too large (review P0).
    Returns None when absent (a real-build run derives the cost from the satellite via Engine 1)."""
    ei = data.get("carbon_cost_share")
    if ei is None:
        return None
    if isinstance(ei, dict):
        return np.array([float(ei.get(s, 0.0)) for s in sectors])
    arr = np.asarray(ei, dtype=float)
    if arr.shape != (len(sectors),):
        raise ValueError(
            f"carbon_cost_share must have one value per sector ({len(sectors)}), got {arr.shape}"
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
        inp = _resolve_inputs(data)
        sam, sectors, factors = (
            inp.sam,
            inp.sectors,
            [f for f in _DEFAULT_FACTORS if f in inp.sam.accounts],
        )
        cal = calibrate(sam, sectors=sectors, factors=factors)
        ns = len(sectors)

        carbon_shocks = [s for s in shocks if isinstance(s, CarbonPrice)]
        # One government ⇒ one recycling rule; a scenario cannot mix modes.
        modes = {s.revenue_recycling for s in carbon_shocks} or {"none"}
        if len(modes) > 1:
            raise ValueError(
                f"cge_static needs a single revenue_recycling mode across carbon shocks; "
                f"got {sorted(modes)}."
            )
        recycling = modes.pop()

        # Per-year carbon cost share (dimensionless, gas/coverage/units handled like Engine 1).
        cc_by_year = {y: _carbon_cost_by_sector(inp, carbon_shocks, y) for y in years}
        emissions_priced = any(np.any(cc != 0.0) for cc, _ in cc_by_year.values())

        # A closed CGE cannot destroy carbon revenue (it breaks Walras' law). When a positive
        # carbon price would raise revenue but the scenario left recycling at the default `none`,
        # default to `lump_sum` (the standard closed-economy choice) and record it, rather than
        # solve a non-closing model. Engine 1 gives the pure price-side / no-recycling view.
        recycling_defaulted = False
        if recycling == "none" and emissions_priced:
            recycling = "lump_sum"
            recycling_defaulted = True

        # Benchmark solve (zero shock) — the replication point, and the base for % changes.
        base_sol = _solve(cal, carbon_cost=np.zeros(ns), recycling="none")
        base = M.derive_state(cal, base_sol.x[:ns], base_sol.x[ns:])

        records: list[dict] = []
        backends: set[str] = {base_sol.backend}
        statuses: set[str] = {base_sol.status}
        for year in years:
            cc, _prov = cc_by_year[year]
            sol = _solve(cal, carbon_cost=cc, recycling=recycling)
            backends.add(sol.backend)
            statuses.add(sol.status)
            st = M.derive_state(cal, sol.x[:ns], sol.x[ns:], carbon_cost=cc, recycling=recycling)
            _emit(records, cal, base, st, year)

        # Emissions provenance: the effective aligned cost-share vector per year + the satellite
        # identity, so a changed satellite / gas selection / doubled emissions moves the manifest
        # (review P1). The cost share already folds in gases, coverage and the 1e-6 scaling.
        emissions_inputs = _emissions_provenance(inp, cc_by_year, sectors)
        manifest = RunManifest.build(
            engine_name=self.meta.name,
            engine_version=self.meta.version,
            data_source=data_source_id(sam.provenance),
            scenario={"shocks": [s.model_dump(mode="json") for s in shocks], "years": years},
            assumptions={
                **ASSUMPTIONS,
                "sectors": sectors,
                "factors": factors,
                "recycling_mode": recycling,
                "recycling_defaulted_from_none": recycling_defaulted,
                "solver_backends": sorted(backends),
                "solver_statuses": sorted(statuses),
                "emissions_priced": emissions_priced,
                "carbon_cost_path": cc_by_year[years[0]][1].get("path"),
                "benchmark_gdp_normalised": cal.gdp0,
                # SAM credibility surface: worst quality severity + per-check summary, so a run
                # states how much the SAM data was helped (roadmap 5.1c). None when a SAM was
                # supplied directly (validated separately, review P1).
                "sam_quality": (
                    {"worst": inp.sam_quality.worst.value, "summary": inp.sam_quality.summary()}
                    if inp.sam_quality is not None
                    else "supplied directly (validated: aligned, finite, non-negative, balanced)"
                ),
                "inputs": [
                    input_identity("SAM", sam.provenance, content=_sam_fingerprint(sam)),
                    *emissions_inputs,
                ],
            },
        )
        return ResultSet.from_records(records, manifest)


@dataclass(frozen=True)
class _Inputs:
    sam: SAM
    sectors: list[str]
    sam_quality: object  # QualityReport | None
    io: IOSystem | None  # present on the real-build path (drives the carbon cost)
    sat: SatelliteAccount | None
    cost_share: np.ndarray | None  # per-sector τ=1 cost share, supplied-SAM path only


def _resolve_inputs(data: dict) -> _Inputs:
    """Resolve the CGE's inputs from ``data``, validating both entry paths.

    - ``SAM`` supplied directly: **validated** (account alignment, finite, non-negative, balanced;
      review P1) before use; the carbon cost comes from a supplied per-sector ``carbon_cost_share``.
    - ``IOSystem`` (a real build): the SAM is built + quality-gated (a failing SAM is rejected), and
      the carbon cost is computed from the satellite the SAME way as Engine 1 (units, gases,
      coverage, and the 1e-6 M€→€ scaling), aggregated to sectors."""
    if "SAM" in data:
        sam: SAM = data["SAM"]
        factors = [f for f in _DEFAULT_FACTORS if f in sam.accounts]
        sectors = data.get("sectors") or _infer_sectors(sam, factors)
        _validate_supplied_sam(sam, sectors, factors)
        return _Inputs(sam, sectors, None, None, None, _carbon_cost_share(data, sectors))

    io = data.get("IOSystem")
    if io is None:
        raise ValueError("cge_static needs a 'SAM' or an 'IOSystem' in data")
    from cge.data.sam import build_sam

    sam, quality, sectors = build_sam(io)
    if not quality.passed:
        failed = [c.name for c in quality.checks if c.severity.value == "fail"]
        raise ValueError(f"SAM quality gate failed for the build: {failed}; refusing to calibrate.")
    return _Inputs(sam, sectors, quality, io, data.get("SatelliteAccount"), None)


def _validate_supplied_sam(sam: SAM, sectors: list[str], factors: list[str]) -> None:
    """Gate a directly-supplied SAM (review P1: a supplied SAM bypassed every check). Requires the
    named sector/factor accounts to exist, all cells finite and non-negative, and the matrix
    balanced (row sum = column sum per account). The engine will not calibrate on a bad SAM."""
    from cge.data.sam.balance import is_balanced

    m = sam.matrix
    missing = [a for a in sectors + factors if a not in sam.accounts]
    if missing:
        raise ValueError(f"supplied SAM is missing named accounts: {missing}")
    arr = m.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("supplied SAM has non-finite cells")
    if float(arr.min()) < -1e-9:
        raise ValueError("supplied SAM has negative cells; a SAM must be non-negative")
    if not is_balanced(m, tol=1e-6):
        from cge.data.sam.balance import imbalance

        worst = float(imbalance(m).abs().max())
        raise ValueError(
            f"supplied SAM is not balanced (max |row−col| = {worst:.3e} > 1e-6); "
            f"the CGE calibrates only on a balanced SAM."
        )


def _emissions_provenance(inp: _Inputs, cc_by_year: dict, sectors: list[str]) -> list[dict]:
    """Reproducibility records for the carbon-cost inputs (review P1: emissions were unrecorded).

    Records the satellite identity + content hash (real-build path) and a content hash of the
    **effective aligned cost-share vector** per year — so a changed satellite, doubled emissions, or
    a different gas/coverage selection all move the manifest. Empty when no carbon cost applies."""
    effective = {
        str(y): [round(float(v), 12) for v in cc.tolist()] for y, (cc, _p) in cc_by_year.items()
    }
    if not any(any(v != 0.0 for v in row) for row in effective.values()):
        return []  # no carbon cost priced; nothing substantive to fingerprint
    out = [
        {
            "name": "EffectiveCarbonCostShare",
            "sectors": sectors,
            "content_hash": content_hash(effective),
        }
    ]
    if inp.sat is not None:
        from cge.engines.io_price.engine import _df_fingerprint

        out.append(
            input_identity(
                "SatelliteAccount", inp.sat.provenance, content=_df_fingerprint(inp.sat.data)
            )
        )
    return out


def _assert_cge_units(io: IOSystem, sat: SatelliteAccount) -> None:
    """Currency-flexible unit gate for the CGE carbon cost. The 1e-6 M→unit scaling in
    ``carbon_cost_vector`` is valid for any *millions*-denominated currency, so we require the
    monetary unit to be ``M<CUR>`` matching the build currency, and every satellite row to be
    ``t/M<CUR>`` (physical gas) or ``tCO2e/M<CUR>`` (the CO2e row). A ``kg/…`` unit is 1000× off and
    rejected. The carbon price is then interpreted in ``<CUR>``/tonne."""
    from cge.engines.io_price.engine import _CO2E_ROW

    cur = io.currency
    if io.unit != f"M{cur}":
        raise ValueError(
            f"cge_static carbon cost needs a millions-denominated monetary base 'M{cur}'; "
            f"build unit is {io.unit!r} (currency {cur!r}). Aggregate/convert the build first."
        )
    if not sat.units:
        raise ValueError(f"satellite {sat.name!r} has no unit metadata; cannot verify t/M{cur}.")
    for row in sat.data.index:
        expected = f"tCO2e/M{cur}" if row == _CO2E_ROW else f"t/M{cur}"
        if sat.units.get(row) != expected:
            raise ValueError(
                f"satellite row {row!r} has unit {sat.units.get(row)!r}, expected {expected!r}; "
                f"the M→{cur} carbon cost-share scaling assumes exactly this unit."
            )


def _carbon_cost_by_sector(
    inp: _Inputs, carbon_shocks: list[CarbonPrice], year: int
) -> tuple[np.ndarray, dict]:
    """Per-sector **dimensionless** carbon cost share for ``year``, plus a provenance dict.

    Real-build path: reuse Engine 1's ``carbon_cost_vector`` (which honours gases, coverage, per-gas
    GWP, and the 1e-6 M€→€ scaling — fixing review P0/P1) on the multi-regional labels, then
    aggregate to sectors output-weighted. Supplied-SAM path: ``cost_share`` × Σ τ (the caller
    supplies the dimensionless τ=1 wedge). Returns ``(cc_sector, provenance)``; provenance feeds the
    manifest so a changed satellite / gas selection moves the result's identity (review P1)."""
    ns = len(inp.sectors)
    if inp.io is None:
        # Supplied-SAM (toy) path: dimensionless cost share × total carbon price.
        share = inp.cost_share if inp.cost_share is not None else np.zeros(ns)
        tau = sum(s.price_at(year) for s in carbon_shocks)
        return tau * share, {"path": "supplied_cost_share"}

    from cge.engines.io_price.engine import carbon_cost_vector

    io, sat = inp.io, inp.sat
    if sat is None:
        return np.zeros(ns), {"path": "no_satellite", "emissions_priced": False}
    # Reject a build whose units make the 1e-6 M-currency→currency scaling wrong. Unlike Engine 1
    # (euro-specific), the CGE accepts any millions-denominated currency, requiring the satellite
    # intensity to be t / M<currency> (or tCO2e / M<currency> for the CO2e row) so the 1e-6 in
    # carbon_cost_vector is exact and the carbon price is in <currency>/tonne (review P0/P2).
    _assert_cge_units(io, sat)
    labels = list(io.A.columns)
    # Per-label direct carbon cost share (dimensionless): honours gases + coverage + 1e-6 scaling.
    cost, descs = carbon_cost_vector(carbon_shocks, sat, labels, year)
    # Aggregate to sectors, output-weighted (cost is a per-unit-output share; weight by output).
    A = io.A.to_numpy(dtype=float)
    fd = io.final_demand.sum(axis=1).reindex(labels).fillna(0.0).to_numpy(dtype=float)
    x = np.linalg.solve(np.eye(A.shape[0]) - A, fd)
    s_index = {s: k for k, s in enumerate(inp.sectors)}
    num = np.zeros(ns)
    den = np.zeros(ns)
    for lb, c_i, xi in zip(labels, cost, x, strict=True):
        k = s_index[lb.split(":", 1)[1]]
        num[k] += c_i * xi
        den[k] += xi
    cc = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    prov = {
        "path": "engine1_carbon_cost_vector",
        "contributions": descs,
        "gases": sorted({g for s in carbon_shocks for g in s.gases}),
        "emissions_priced": bool(np.any(cc != 0.0)),
    }
    return cc, prov


def _infer_sectors(sam: SAM, factors: list[str]) -> list[str]:
    """Sectors = SAM accounts that are neither factors nor the single institution (household)."""
    non_factor = [a for a in sam.accounts if a not in factors]
    # The institution is the account with no value-added-style column into factors; simplest for
    # the pilot: assume exactly one institution and take it as the last non-sector. We identify it
    # as the account that receives from factors (a factor row pays it).
    institutions = [a for a in non_factor if any(sam.matrix.loc[a, f] != 0 for f in factors)]
    return [a for a in non_factor if a not in institutions]


def _solve(cal, *, carbon_cost, recycling="none"):
    # prefer='scipy' explicitly: the CGE model residual is numeric-only (it evaluates the Leontief
    # inverse and Cobb-Douglas cost functions with numpy), so it cannot build a symbolic Pyomo
    # model. Auto-selecting IPOPT when its binary is present would therefore FAIL (review P1). A
    # symbolic residual to enable IPOPT is a documented follow-up; scipy solves the small model.
    return solve(
        lambda z: M.residuals(cal, z, carbon_cost=carbon_cost, recycling=recycling),
        M.initial_guess(cal),
        prefer="scipy",
    )


def _emit(records, cal, base, st, year: int) -> None:
    """Append price/volume changes and GE outputs (relative to the benchmark) for one year."""
    for i, sector in enumerate(cal.sectors):
        records.append(_rec("price_change", sector, year, st.p[i] / base.p[i] - 1.0))
        records.append(_rec("volume_change", sector, year, st.X[i] / base.X[i] - 1.0))
    for f_idx, factor in enumerate(cal.factors):
        records.append(_rec("factor_price_change", factor, year, st.w[f_idx] / base.w[f_idx] - 1.0))
    # GDP (expenditure = Σ p_i·FD_i). Nominal change vs benchmark; real change deflates it by the
    # **exact Cobb-Douglas consumer price index** P_cd = Π p_i^γ_i (the household's true cost of
    # living), so real = nominal / P_cd. Reported ``deflator`` is that CD price index change
    # (review P1/P2: the earlier arithmetic Σ γ·p numéraire and Paasche ratio were inconsistent
    # with the CD household). ``gdp_change_real`` is thus the CD-deflated real expenditure change.
    nom_gdp = float(np.dot(st.p, st.FD))
    nom_gdp_base = float(np.dot(base.p, base.FD))
    cpi = float(np.prod(np.power(st.p, cal.gamma)))  # exact CD price index (base = 1)
    real_gdp = nom_gdp / cpi
    real_gdp_base = nom_gdp_base  # base CPI = 1
    records.append(_rec("gdp_change", "__economy__", year, nom_gdp / nom_gdp_base - 1.0))
    records.append(_rec("gdp_change_real", "__economy__", year, real_gdp / real_gdp_base - 1.0))
    records.append(_rec("deflator", "__economy__", year, cpi - 1.0))
    # Welfare: the change in Cobb-Douglas utility U = Π FD_i^γ_i (the correct welfare measure for a
    # CD household — review P1: the earlier Σ FD sum is not utility). Equivalent to real income
    # (nominal income deflated by the CD price index).
    u = float(np.prod(np.power(st.FD, cal.gamma)))
    u_base = float(np.prod(np.power(base.FD, cal.gamma)))
    records.append(_rec("welfare_change", "__economy__", year, u / u_base - 1.0))
    records.append(_rec("carbon_revenue", "__economy__", year, st.carbon_revenue / cal.gdp0))


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
