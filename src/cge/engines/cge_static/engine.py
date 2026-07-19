"""Engine 3 — static computable general equilibrium (roadmap Phase 5).

Wraps the calibrated pilot CGE (``calibrate`` + ``model`` + ``solver``) behind the ``Engine``
protocol, so the GUI/CLI pick it up via the registry with no changes. Given a benchmark ``SAM``,
it calibrates the model to reproduce the base year exactly, applies a ``CarbonPrice`` as a
per-unit emissions cost wedge, solves for the new equilibrium, and emits a ``ResultSet`` of price
and volume changes plus GE outputs (factor prices, real GDP, welfare, carbon revenue). The CPI is
the numéraire, so there is no separate deflator/inflation output.

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
        "carbon revenue R = Σ τ·e[i]·X[i] is returned to the household (lump_sum/labour_tax_cut). "
        "Established: CD welfare falls only slightly under a recycled carbon price, and at those "
        "prices the transfer raises utility. NOT established: a full GE welfare comparison against "
        "a valid no-recycling closure (the `none` mode does not close — it violates Walras' law — "
        "so it is not a valid counterfactual; a government/external account is a follow-up)."
    ),
    "closure": (
        "savings-less pilot; fixed factor supply; numéraire = the exact Cobb-Douglas consumer "
        "price index (Π p_i^γ_i = 1). The CPI is the unit of account, so no inflation/deflator is "
        "reported; outputs are real quantities and relative (CPI-unit) prices."
    ),
    "solver_rule": (
        "non-optimal solve raises (well-posedness); solver backend, termination status, and "
        "max residual norm recorded in the manifest"
    ),
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
        # Keys must belong to the declared sectors — a typo would otherwise be silently dropped
        # (review P1). Values must be finite and non-negative (a negative share is an undocumented
        # subsidy: it lowers the dirty-sector price and generates negative revenue).
        unknown = [k for k in ei if k not in sectors]
        if unknown:
            raise ValueError(f"carbon_cost_share keys not in the SAM sectors {sectors}: {unknown}")
        arr = np.array([float(ei.get(s, 0.0)) for s in sectors])
    else:
        arr = np.asarray(ei, dtype=float)
        if arr.shape != (len(sectors),):
            raise ValueError(
                f"carbon_cost_share must have one value per sector ({len(sectors)}), "
                f"got shape {arr.shape}"
            )
    if not np.isfinite(arr).all():
        raise ValueError("carbon_cost_share values must be finite")
    if float(arr.min()) < 0.0:
        raise ValueError(
            "carbon_cost_share values must be non-negative (a negative share is a carbon subsidy, "
            "not a price; it would lower the dirty-sector price and generate negative revenue)"
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
        # Open economy (Armington/CET) when the SAM carries a rest-of-world account; otherwise the
        # closed pilot. The open path has its own calibration/model (activity+commodity accounts).
        supplied_sam = data.get("SAM")
        if supplied_sam is not None and "ROW" in supplied_sam.accounts:
            return _run_open(self.meta, data, shocks, years)

        inp = _resolve_inputs(data)
        sam, sectors, factors = (
            inp.sam,
            inp.sectors,
            [f for f in _DEFAULT_FACTORS if f in inp.sam.accounts],
        )
        cal = calibrate(sam, sectors=sectors, factors=factors, va_elast=data.get("va_elast", 1.0))
        ns = len(sectors)

        carbon_shocks = [s for s in shocks if isinstance(s, CarbonPrice)]
        _validate_cge_shock_controls(inp, carbon_shocks)
        # One government ⇒ one recycling rule; a scenario cannot mix modes.
        modes = {s.revenue_recycling for s in carbon_shocks} or {"none"}
        if len(modes) > 1:
            raise ValueError(
                f"cge_static needs a single revenue_recycling mode across carbon shocks; "
                f"got {sorted(modes)}."
            )
        recycling = modes.pop()

        # A positive carbon price needs an emissions input, or it silently becomes a zero-impact
        # run. Require the input whenever ANY year has a nonzero effective carbon price; tolerate a
        # missing input only for a genuine zero-price baseline (review P1).
        positive_price = any(s.price_at(y) > 0 for s in carbon_shocks for y in years)
        if positive_price:
            if inp.io is not None and inp.sat is None:
                raise ValueError(
                    "a positive carbon price on an IO-backed CGE run requires a 'SatelliteAccount' "
                    "(emission intensities); none supplied — the run would be silently zero-impact."
                )
            if inp.io is None and inp.cost_share is None:
                raise ValueError(
                    "a positive carbon price on a supplied-SAM CGE run requires a "
                    "'carbon_cost_share'; none supplied — the run would be silently zero-impact."
                )

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
        resid_max: float = base_sol.residual_norm
        for year in years:
            cc, _prov = cc_by_year[year]
            sol = _solve(cal, carbon_cost=cc, recycling=recycling)
            backends.add(sol.backend)
            statuses.add(sol.status)
            resid_max = max(resid_max, sol.residual_norm)
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
                # The strongest numerical convergence evidence: max ‖F(x)‖∞ over all solves. The
                # solver already re-verifies this < tol before returning (else it raises), so this
                # records HOW converged the equilibrium is (review P2).
                "solver_max_residual_norm": resid_max,
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
    from cge.engines.io_price.engine import assert_io_aligned

    assert_io_aligned(io)  # boundary guard before the SAM is built from raw A (review P2)
    sam, quality, sectors = build_sam(io)
    if not quality.passed:
        failed = [c.name for c in quality.checks if c.severity.value == "fail"]
        raise ValueError(f"SAM quality gate failed for the build: {failed}; refusing to calibrate.")
    return _Inputs(sam, sectors, quality, io, data.get("SatelliteAccount"), None)


def _validate_supplied_sam(sam: SAM, sectors: list[str], factors: list[str]) -> None:
    """Gate a directly-supplied SAM (review P1: a supplied SAM bypassed every check). Requires the
    named sector/factor accounts to exist, the **matrix axes to be unique and aligned to the
    accounts** used (review P2: a renamed-axis matrix passed the name check then raised a raw
    KeyError during calibration), all cells finite and non-negative, and the matrix balanced (row
    sum = column sum per account). The engine will not calibrate on a bad SAM."""
    from cge.data.sam.balance import is_balanced

    m = sam.matrix
    missing = [a for a in sectors + factors if a not in sam.accounts]
    if missing:
        raise ValueError(f"supplied SAM is missing named accounts: {missing}")
    # Axis alignment: the matrix index and columns must be unique and **equal** the declared
    # accounts (review P2: containment is not enough — an extra balanced account passed the check,
    # was silently ignored by calibration, yet the manifest called the SAM 'aligned'). A renamed
    # axis would also raise a raw KeyError during calibration.
    idx, cols = list(m.index), list(m.columns)
    if len(set(idx)) != len(idx) or len(set(cols)) != len(cols):
        raise ValueError("supplied SAM matrix has duplicate row or column labels")
    accts = set(sam.accounts)
    if set(idx) != accts or set(cols) != accts:
        extra = sorted((set(idx) | set(cols)) - accts)
        missing_axis = sorted(accts - (set(idx) & set(cols)))
        raise ValueError(
            f"supplied SAM matrix axes must equal the declared accounts exactly; "
            f"extra axis labels not in accounts: {extra}; accounts missing from an axis: "
            f"{missing_axis}."
        )
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


def _validate_cge_shock_controls(inp: _Inputs, carbon_shocks: list[CarbonPrice]) -> None:
    """Reject shock controls the CGE cannot honour, rather than silently ignore them (review P1).

    - **IO-backed path:** coverage labels (sectors/regions) must exist in the build, exactly as
      Engine 1 requires — a typo would otherwise give a silent zero-impact scenario.
    - **Supplied-SAM path:** the dimensionless ``carbon_cost_share`` cannot express per-gas or
      spatial coverage, so any non-default ``gases`` / ``coverage_sectors`` / ``coverage_regions``
      is **rejected** (not applied to a global vector as if honoured)."""
    if inp.io is not None:
        from cge.engines.io_price.engine import _assert_coverage_labels

        _assert_coverage_labels(carbon_shocks, inp.io)
        return
    # Supplied-SAM path: reject controls that this path structurally cannot apply.
    for s in carbon_shocks:
        if s.gases != ["CO2"]:
            raise ValueError(
                f"the supplied-SAM CGE path applies a single dimensionless carbon_cost_share and "
                f"cannot select gases; got gases={s.gases}. Use an IOSystem+satellite build for "
                f"gas selection, or leave gases at the default ['CO2']."
            )
        if s.coverage_sectors or s.coverage_regions:
            raise ValueError(
                "the supplied-SAM CGE path cannot apply sector/region coverage (the cost share is "
                "already per-sector and single-region); set coverage via carbon_cost_share values, "
                "or use an IOSystem+satellite build."
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
    # GDP. The **numéraire is the household's exact CD price index** P_cd = Π p_i^γ_i, pinned to 1
    # (see model.residuals). So all prices are expressed in CPI units and there is **no separate
    # deflator to report** — the CPI is fixed to 1 by definition, and a "deflator" derived from it
    # would be a mechanical numéraire artifact, not inflation (review P1). We therefore report:
    #   • gdp_change_real — the real (CPI-numéraire) change in GDP = Σ p·FD (prices in CPI units,
    #     so this expenditure aggregate is already real);
    #   • gdp_change_nominal_in_factor_units — the same aggregate valued with a factor price as the
    #     unit of account, so a "money" magnitude is still available for readers who want one.
    real_gdp = float(np.dot(st.p, st.FD))
    real_gdp_base = float(np.dot(base.p, base.FD))
    records.append(_rec("gdp_change_real", "__economy__", year, real_gdp / real_gdp_base - 1.0))
    # A factor-price-numéraire nominal GDP (unit of account = labour), for a "nominal" reference
    # that is NOT mechanically tied to the CPI numéraire.
    lab = cal.factors.index("LAB") if "LAB" in cal.factors else 0
    nom_gdp = real_gdp / st.w[lab]
    nom_gdp_base = real_gdp_base / base.w[lab]
    records.append(
        _rec("gdp_change_nominal_wage", "__economy__", year, nom_gdp / nom_gdp_base - 1.0)
    )
    # Welfare: the change in Cobb-Douglas utility U = Π FD_i^γ_i (the correct welfare measure for a
    # CD household — review P1: the earlier Σ FD sum is not utility).
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


def _run_open(meta, data: dict, shocks: list[Shock], years: list[int]) -> ResultSet:
    """Open-economy (Armington/CET) CGE run. The SAM has ``a_<s>``/``c_<s>`` activity/commodity
    accounts, factors, a household and a ``ROW`` account. Carbon cost is a supplied per-sector
    dimensionless ``carbon_cost_share`` (an IOSystem→SAM open build is a follow-up)."""
    from cge.engines.cge_static import model_open as MO
    from cge.engines.cge_static.calibrate_open import calibrate_open

    sam: SAM = data["SAM"]
    sectors = [a[2:] for a in sam.accounts if a.startswith("a_")]
    factors = [f for f in _DEFAULT_FACTORS if f in sam.accounts]
    ns = len(sectors)

    carbon_shocks = [s for s in shocks if isinstance(s, CarbonPrice)]
    modes = {s.revenue_recycling for s in carbon_shocks} or {"none"}
    if len(modes) > 1:
        raise ValueError(f"cge_static needs a single recycling mode; got {sorted(modes)}.")
    recycling = modes.pop()
    if recycling == "none":
        recycling = "lump_sum"  # the open economy also circulates carbon revenue to the household

    share = _carbon_cost_share(data, sectors)
    positive_price = any(s.price_at(y) > 0 for s in carbon_shocks for y in years)
    if positive_price and share is None:
        raise ValueError(
            "a positive carbon price on the open-economy CGE requires a 'carbon_cost_share'."
        )
    share = share if share is not None else np.zeros(ns)

    cal = calibrate_open(
        sam,
        sectors=sectors,
        factors=factors,
        va_elast=data.get("va_elast", 1.0),
        arm_elast=data.get("armington_elast", 2.0),
        cet_elast=data.get("cet_elast", 2.0),
    )

    def _solve_year(cc):
        sol = solve(
            lambda z: MO.residuals(cal, z, carbon_cost=cc, recycling=recycling),
            MO.initial_guess(cal),
            prefer="scipy",
        )
        st = MO.derive_open_state(
            cal,
            sol.x[:ns],
            sol.x[ns : 2 * ns],
            sol.x[2 * ns : 2 * ns + len(factors)],
            float(sol.x[-1]),
            carbon_cost=cc,
            recycling=recycling,
        )
        return sol, st

    _bsol, base = _solve_year(np.zeros(ns))
    records: list[dict] = []
    resid_max = _bsol.residual_norm
    for year in years:
        tau = sum(s.price_at(year) for s in carbon_shocks)
        cc = tau * share
        sol, st = _solve_year(cc)
        resid_max = max(resid_max, sol.residual_norm)
        _emit_open(records, cal, base, st, year)

    manifest = RunManifest.build(
        engine_name=meta.name,
        engine_version=meta.version,
        data_source=data_source_id(sam.provenance),
        scenario={"shocks": [s.model_dump(mode="json") for s in shocks], "years": years},
        assumptions={
            **ASSUMPTIONS,
            "model_variant": "open economy (Armington imports + CET exports; small-open, fixed "
            "world prices and foreign savings; exchange rate endogenous)",
            "sectors": sectors,
            "factors": factors,
            "recycling_mode": recycling,
            "armington_elasticity": float(cal.arm_elast[0]),
            "cet_elasticity": float(cal.cet_elast[0]),
            "solver_max_residual_norm": resid_max,
            "emissions_priced": bool(np.any(share != 0.0) and positive_price),
            "inputs": [input_identity("SAM", sam.provenance, content=_sam_fingerprint(sam))],
        },
    )
    return ResultSet.from_records(records, manifest)


def _emit_open(records, cal, base, st, year: int) -> None:
    """Emit open-economy results: activity output (volume), domestic/import/export volumes, the
    composite price, factor prices, exchange rate, real GDP, welfare and carbon revenue."""
    for i, sector in enumerate(cal.sectors):
        records.append(_rec("price_change", sector, year, st.pq[i] / base.pq[i] - 1.0))
        records.append(_rec("volume_change", sector, year, st.Z[i] / base.Z[i] - 1.0))
        records.append(_rec("import_change", sector, year, _ratio(st.M[i], base.M[i])))
        records.append(_rec("export_change", sector, year, _ratio(st.E[i], base.E[i])))
    for f_idx, factor in enumerate(cal.factors):
        records.append(_rec("factor_price_change", factor, year, st.w[f_idx] / base.w[f_idx] - 1.0))
    records.append(_rec("exchange_rate_change", "__economy__", year, st.er / base.er - 1.0))
    # Real GDP = real absorption at benchmark composite prices (CPI numéraire): Σ FD (household) is
    # the real consumption index; report it as the real-GDP proxy for the pilot.
    records.append(_rec("gdp_change_real", "__economy__", year, st.FD.sum() / base.FD.sum() - 1.0))
    u = float(np.prod(np.power(st.FD, cal.gamma)))
    u_base = float(np.prod(np.power(base.FD, cal.gamma)))
    records.append(_rec("welfare_change", "__economy__", year, u / u_base - 1.0))
    records.append(_rec("carbon_revenue", "__economy__", year, st.carbon_revenue / cal.gdp0))


def _ratio(x: float, x0: float) -> float:
    return float(x / x0 - 1.0) if x0 > 0 else 0.0


def armington_sensitivity_sweep(
    data: dict,
    shocks: list[Shock],
    year: int = 2020,
    *,
    elasticities: tuple[float, float, float] = (1.5, 2.0, 4.0),
):
    """Run the open-economy CGE across low/central/high **Armington** elasticities and return a
    tidy DataFrame of the response **envelope** (Phase 5.3 sensitivity sweep). Volume responses are
    elasticity-sensitive, so — as with Engine 2's demand bands — the band is a first-class output.

    Returns columns ``sector, variable, low, central, high`` for the per-sector volume/import/export
    responses (the leakage channel is the most elasticity-sensitive). Requires an open SAM."""
    import pandas as pd

    sam = data.get("SAM")
    if sam is None or "ROW" not in sam.accounts:
        raise ValueError("armington_sensitivity_sweep needs an open SAM (with a ROW account)")

    bands = {"low": elasticities[0], "central": elasticities[1], "high": elasticities[2]}
    per_band: dict[str, pd.Series] = {}
    for band, elast in bands.items():
        res = CGEStaticEngine().run(
            data={**data, "armington_elast": elast}, shocks=shocks, years=[year]
        )
        d = res.data
        d = d[d["variable"].isin(("volume_change", "import_change", "export_change"))]
        per_band[band] = d.set_index(["sector", "variable"])["value"]

    out = pd.DataFrame(per_band).reset_index()
    return out.sort_values(["variable", "sector"]).reset_index(drop=True)


registry.register(CGEStaticEngine())
