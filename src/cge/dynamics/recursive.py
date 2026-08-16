"""The recursive-dynamic wrapper (Phase 7.1).

Steps a static ``general_equilibrium`` CGE forward year-by-year to a horizon, carrying the **capital
stock** between solves via the Phase-5d.3 accumulation identity, and stepping optional **exogenous
labour** (demographics) and **productivity** (TFP trend) endowments. Each year is a normal static
solve of the SAME calibrated model, re-scaled to that year's capital/labour endowment — *bookkeeping
between solves*, not perfect foresight and not a new solution concept (roadmap 7.1).

**The loop.** Starting from the benchmark stock ``K0`` (read from the CGE manifest's stock–flow
bridge, `benchmark_capital`):

1. solve year *t* with the endowment scaled to ``K_t`` (capital) and ``L_t`` (labour);
2. read that year's **investment** ``INV_t`` (the CGE's savings-investment outcome);
3. ``K_{t+1} = capital_next(K_t, INV_t, δ, retirement_t)`` (5d.3 perpetual inventory + optional
   premature retirement);
4. step ``L_{t+1}`` and the productivity index by their exogenous trends; advance.

**Why the endowment scale.** The CGE's capital endowment is the capital-services flow, proportional
to the stock, so scaling the stock by ``K_{t+1}/K0`` scales the services endowment by the same
factor — the `factor_endowment_scale` hook the engine exposes. Labour scales the same way.
Productivity enters as a Hicks-neutral endowment-equivalent scale on both factors (a documented
simplification; a genuine sector-level TFP term is a follow-up).

Results are reported per year **relative to the original benchmark**, so capital accumulation and
the trends are VISIBLE in the level path (a growing stock raises output vs the benchmark). The
wrapper adds ``capital_stock`` and ``capital_growth`` result rows (one per region).

**Scope: all three CGE variants.** The closed/gov SAM and the open economy (Armington/CET + rest of
world) carry one aggregate capital stock; the multi-region CGE carries a **per-region capital path**
— each region's stock steps by its own investment, ``K_{t+1,r}=(1−δ)(1−r)K_{t,r}+INV_{t,r}``, and
the ``factor_endowment_scale`` hook moves each region's capital independently. Any variant needs a
savings-investment account to be dynamic-capable: ``toy_cge_gov`` (closed), ``toy_cge_open_gov``
(open), ``toy_cge_multi_gov`` (multi). Labour and productivity trends are exogenous and applied
uniformly across regions (a documented simplification; region-specific trends are a follow-up).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cge.contracts.data_objects import StructuralTrajectory
from cge.contracts.results import ResultSet
from cge.engines.cge_static.capital import DEFAULT_DEPRECIATION_RATE, capital_next
from cge.runner import run_scenario
from cge.scenarios.loader import Scenario


@dataclass
class DynamicConfig:
    """Configuration for a recursive-dynamic run (Phase 7.1 / 7b.2).

    Trends default to **flat** (no growth), so a zero-trend run is transparent bookkeeping over the
    static solves. Supplying ``structural`` (a documented, sourced :class:`StructuralTrajectory`,
    Phase 7b.2) replaces the flat ``labour_growth``/``productivity_growth`` scalars with per-region,
    per-year sourced trajectories: labour-supply growth = population × participation, and TFP
    growth, each compounded from the sourced annual rates. The flat scalars remain the fallback when
    no trajectory is given."""

    # Flat fallback trends (used when ``structural`` is None): applied uniformly across regions.
    depreciation: float = DEFAULT_DEPRECIATION_RATE  # δ per year (5d.3 default 5%)
    labour_growth: float = 0.0  # labour-force growth per year
    productivity_growth: float = 0.0  # Hicks-neutral TFP trend per year
    # Per-year premature capital retirement fraction (5d.3 stranded assets), e.g. {2030: 0.1}. A
    # year absent → 0. Applied to the OPENING stock in that year's accumulation step.
    retirement: dict[int, float] = field(default_factory=dict)
    # Documented, sourced per-region structural trajectories (Phase 7b.2). When set, it drives the
    # labour and productivity trends per region instead of the flat scalars above.
    structural: StructuralTrajectory | None = None

    def __post_init__(self) -> None:
        for name in ("depreciation", "labour_growth", "productivity_growth"):
            v = getattr(self, name)
            if not np.isfinite(v):
                raise ValueError(f"DynamicConfig.{name} must be finite; got {v!r}")
        if not (0.0 <= self.depreciation <= 1.0):
            raise ValueError(f"depreciation δ must be in [0, 1]; got {self.depreciation}")
        for y, r in self.retirement.items():
            if not np.isfinite(r) or not (0.0 <= r <= 1.0):
                raise ValueError(f"retirement[{y}] must be a fraction in [0, 1]; got {r!r}")


@dataclass
class DynamicPath:
    """The result of a recursive-dynamic run: the per-year ``ResultSet`` plus the capital path.

    The path dicts are keyed by year. For the single-region variants (closed/gov/open — one
    aggregate capital stock) the values are **floats** (back-compatible). For the multi-region CGE
    they are 1-D ``numpy`` arrays, one entry per region (ordered as ``recursive_dynamics
    ['capital_regions']`` in the result manifest)."""

    result: ResultSet
    capital_stock: dict[int, float | np.ndarray]  # end-of-year stock K by year
    investment: dict[int, float | np.ndarray]  # nominal investment (share of benchmark GDP) by year
    growth: dict[int, float | np.ndarray]  # capital growth rate K_{t+1}/K_t − 1 by year


def _manifest_capital(manifest) -> tuple[np.ndarray, list[str]]:
    """The benchmark capital stock and the region labels it is ordered by, from the CGE manifest's
    stock–flow bridge. Returns ``(K0, regions)`` where ``K0`` is a 1-D array (one entry per capital
    region) and ``regions`` are the matching labels — ``["R"]`` for the single-region variants
    (closed/gov/open, one aggregate stock), or the model's regions for the multi-region CGE."""
    cd = manifest.assumptions.get("capital_dynamics", {})
    if not cd.get("available"):
        raise ValueError(
            "recursive dynamics need a CGE with a capital factor AND a savings-investment account "
            f"(the benchmark stock–flow bridge is unavailable: {cd.get('reason', 'unknown')}). "
            "Use a SAM with a SAVINV account, e.g. toy_cge_gov (or toy_cge_multi_gov)."
        )
    k0 = np.asarray(cd["benchmark_capital_stock"], dtype=float)
    # The multi-region CGE stamps a 'regions' list ordered the same as the capital vector; the
    # single-region variants (one aggregate stock) don't, and use the canonical "R" region label.
    regions = list(manifest.assumptions.get("regions") or ["R"])
    if len(regions) != len(k0):
        raise ValueError(
            f"capital regions ({len(k0)}) do not match manifest regions {regions}; cannot map the "
            "capital path to regions."
        )
    return k0, regions


def _year_investment(res: ResultSet, year: int, regions: list[str]) -> np.ndarray:
    """That year's nominal investment (share of benchmark GDP) per region, ordered by ``regions``.

    The single-region variants report one ``investment`` row with region label ``R``; the multi CGE
    reports one per region. Returns a 1-D array aligned to ``regions``."""
    d = res.data
    inv = d[(d["variable"] == "investment") & (d["scenario"] == "central") & (d["year"] == year)]
    if inv.empty:
        raise ValueError(f"no investment result for year {year}; the CGE reported none.")
    by_region = dict(zip(inv["region"], inv["value"], strict=False))
    missing = [r for r in regions if r not in by_region]
    if missing:
        raise ValueError(
            f"no investment result for year {year} region(s) {missing}; got {sorted(by_region)}."
        )
    return np.array([float(by_region[r]) for r in regions], dtype=float)


def run_recursive(
    scenario: Scenario,
    *,
    config: DynamicConfig | None = None,
    data_source: str = "toy_cge_gov",
    store=None,
) -> DynamicPath:
    """Run ``scenario`` recursively-dynamically over its ``years``, carrying capital forward.

    The scenario's ``years`` are the solve years (sorted); its shocks apply per year exactly as in a
    static run. Returns a ``DynamicPath`` whose ``result`` is the concatenated per-year ResultSet
    (with added ``capital_stock``/``capital_growth`` rows) and whose dicts give the capital path."""
    config = config or DynamicConfig()
    years = sorted(scenario.years)

    # K0 (per capital region) from the benchmark stock–flow bridge (a cheap no-shock probe run at
    # the first year), plus the region labels the vector is ordered by.
    probe = run_scenario(
        scenario.model_copy(update={"shocks": [], "years": [years[0]]}),
        data_source=data_source,
        store=store,
    )
    k0, regions = _manifest_capital(probe.manifest)
    multi = regions != ["R"]  # per-region capital path vs one aggregate stock

    frames: list[pd.DataFrame] = []
    capital_stock: dict[int, np.ndarray] = {}  # end-of-year stock K per region, by year
    investment: dict[int, np.ndarray] = {}  # nominal investment (share of benchmark GDP) per region
    growth: dict[int, np.ndarray] = {}  # capital growth rate K_{t+1}/K_t − 1 per region

    base_year = years[0]
    k_t = k0.copy()  # 1-D array, one entry per capital region
    for year in years:
        # Cumulative labour-supply and productivity scales from the base year to this year, PER
        # REGION. With a StructuralTrajectory (7b.2) these compound the sourced per-year rates over
        # the actual year gaps (solve years may be spaced apart); without one they fall back to the
        # flat DynamicConfig scalars, uniform across regions (Phase 7.1 back-compat).
        labour_scale = _trend_scale(config, base_year, year, regions, "labour")
        tfp_scale = _trend_scale(config, base_year, year, regions, "productivity")
        # Capital and labour endowments scale to this year's stock/force; TFP is applied Hicks-
        # neutrally as an equivalent scale on both primary factors (documented simplification).
        # Capital scale is PER REGION (each region carries its own stock).
        cap_scale = (k_t / k0) * tfp_scale  # 1-D array per region
        lab_scale = labour_scale * tfp_scale  # 1-D array per region
        overrides = {"factor_endowment_scale": _factor_scale(cap_scale, lab_scale, regions, multi)}

        res = run_scenario(
            scenario.model_copy(update={"years": [year]}),
            data_source=data_source,
            store=store,
            data_overrides=overrides,
        )
        frames.append(res.data)

        inv_share = _year_investment(res, year, regions)  # per-region nominal investment / GDP0
        # Investment is a GDP-share flow; convert to the same units as K (the user-cost stock is in
        # capital-income units = GDP-normalised too, since gdp0 = benchmark income). INV in stock
        # units = inv_share · gdp0 / gdp0-scale — but K0 is already in those units, so inv_share is
        # directly comparable to K0 (both GDP-normalised). Step each region's stock:
        r_t = float(config.retirement.get(year, 0.0))
        k_next = capital_next(k_t, inv_share, depreciation=config.depreciation, retirement=r_t)

        investment[year] = inv_share
        capital_stock[year] = k_next
        growth[year] = k_next / k_t - 1.0
        k_t = k_next

    # Append the capital path as result rows (one per region) so it flows through the ResultSet like
    # any variable.
    data = pd.concat(frames, ignore_index=True)
    extra = []
    for year in years:
        for ri, region in enumerate(regions):
            extra.append(_rec("capital_stock", year, float(capital_stock[year][ri]), region))
            extra.append(_rec("capital_growth", year, float(growth[year][ri]), region))
    data = pd.concat([data, pd.DataFrame(extra)], ignore_index=True)

    # Reuse the last year's manifest; stamp the dynamic configuration onto it.
    manifest = res.manifest
    manifest.assumptions["recursive_dynamics"] = {
        "mode": "recursive_dynamic (bookkeeping between static solves; no perfect foresight)",
        "horizon_years": years,
        "capital_regions": regions,
        "depreciation_rate": config.depreciation,
        "retirement": {int(k): float(v) for k, v in config.retirement.items()},
        "benchmark_capital_stock": [round(float(x), 12) for x in k0],
        "capital_stock_path": {
            int(y): [round(float(x), 12) for x in capital_stock[y]] for y in years
        },
        # Trend provenance: either the flat fallback scalars (Phase 7.1) or the sourced structural
        # trajectory (Phase 7b.2), so a run records exactly which drove its trends.
        "trend_source": _trend_provenance(config),
        "note": (
            "Capital carried forward via K_{t+1}=(1−δ)(1−r)K_t+INV_t (Phase 5d.3); labour and TFP "
            "are exogenous trends applied as endowment scales. Single aggregate capital stock for "
            "the closed/gov/open variants; a per-region capital path for the multi-region CGE."
        ),
    }
    result = ResultSet(data=data, manifest=manifest)
    result.validate_schema()
    # For the single-region variants, expose the path dicts as scalars (back-compatible with the
    # closed/open callers and tests); for multi, expose the per-region arrays.
    return DynamicPath(
        result=result,
        capital_stock=_unwrap(capital_stock, multi),
        investment=_unwrap(investment, multi),
        growth=_unwrap(growth, multi),
    )


def _factor_scale(
    cap_scale: np.ndarray, lab_scale: np.ndarray, regions: list[str], multi: bool
) -> dict:
    """Build the ``factor_endowment_scale`` override for one year. Both scales are per-region 1-D
    arrays aligned to ``regions``. Single-region: scalar factor scales (``{"CAP": s, "LAB": s}``).
    Multi-region: per-region scales (``{"CAP": {region: s}, "LAB": {region: s}}``) so each region's
    capital and labour supply move independently."""
    if not multi:
        return {"CAP": float(cap_scale[0]), "LAB": float(lab_scale[0])}
    return {
        "CAP": {r: float(cap_scale[ri]) for ri, r in enumerate(regions)},
        "LAB": {r: float(lab_scale[ri]) for ri, r in enumerate(regions)},
    }


def _trend_scale(
    config: DynamicConfig, base_year: int, year: int, regions: list[str], kind: str
) -> np.ndarray:
    """Cumulative endowment scale from ``base_year`` to ``year`` for a per-region trend, as a 1-D
    array aligned to ``regions``.

    ``kind`` is ``"labour"`` (labour-supply growth = population × participation) or
    ``"productivity"`` (TFP growth). With a :class:`StructuralTrajectory` the sourced per-year rates
    are **compounded year by year** over the actual gap (solve years may be spaced apart), so a 5-yr
    step compounds 5 annual rates; the rate for each intervening year is the trajectory's
    piecewise-constant value. Without a trajectory it falls back to the flat ``DynamicConfig``
    scalar, applied uniformly across regions (Phase 7.1 behaviour)."""
    traj = config.structural
    if traj is None:
        flat = config.labour_growth if kind == "labour" else config.productivity_growth
        scale = (1.0 + flat) ** (year - base_year)
        return np.full(len(regions), scale, dtype=float)

    out = np.ones(len(regions), dtype=float)
    for ri, region in enumerate(regions):
        acc = 1.0
        for y in range(base_year, year):  # compound each annual step up to (not incl.) target year
            if kind == "labour":
                rate = traj.rate("population", region, y) + traj.rate(
                    "labour_participation", region, y
                )
            else:
                rate = traj.rate("productivity", region, y)
            acc *= 1.0 + rate
        out[ri] = acc
    return out


def _trend_provenance(config: DynamicConfig) -> dict:
    """Record how the labour/productivity trends were set — the flat fallback scalars (Phase 7.1) or
    a sourced :class:`StructuralTrajectory` (Phase 7b.2), with its provenance and per-entry cites —
    so a run's manifest is self-documenting about which drove it."""
    traj = config.structural
    if traj is None:
        return {
            "kind": "flat",
            "labour_growth": config.labour_growth,
            "productivity_growth": config.productivity_growth,
            "note": "flat uniform trends (Phase 7.1 fallback); no sourced structural trajectory.",
        }
    return {
        "kind": "structural_trajectory",
        "provenance": {
            "source": traj.provenance.source,
            "source_version": traj.provenance.source_version,
            "licence": traj.provenance.licence,
            "retrieved": traj.provenance.retrieved,
        },
        "drivers": sorted(traj.rates),
        "sources": dict(traj.sources),
        "confidence": dict(traj.confidence),
        "note": (
            "Phase 7b.2 sourced trajectories: labour-supply growth = population × participation, "
            "and productivity growth, compounded per region from the cited annual rates."
        ),
    }


def _unwrap(path: dict[int, np.ndarray], multi: bool) -> dict[int, float | np.ndarray]:
    """Scalarise a single-region path (one capital region) for back-compatible float dict values;
    leave the multi-region per-region arrays as-is."""
    if multi:
        return dict(path)
    return {year: float(v[0]) for year, v in path.items()}


def _rec(variable: str, year: int, value: float, region: str) -> dict:
    return {
        "variable": variable,
        "sector": "__economy__",
        "region": region,
        "year": int(year),
        "scenario": "central",
        "value": float(value),
    }
