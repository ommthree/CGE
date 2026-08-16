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
wrapper adds ``capital_stock`` and ``capital_growth`` result rows. Scope: any **single
capital-region** variant — the closed/gov SAM and the **open** economy (Armington/CET + rest of
world), both of which carry one aggregate capital stock (matching 5d.3). A dynamic-capable open SAM
needs a savings-investment account (``toy_cge_open_gov``). Multi-region capital accumulation (a
per-region capital path) is the remaining follow-up (roadmap 7.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cge.contracts.results import ResultSet
from cge.engines.cge_static.capital import DEFAULT_DEPRECIATION_RATE, capital_next
from cge.runner import run_scenario
from cge.scenarios.loader import Scenario


@dataclass
class DynamicConfig:
    """Configuration for a recursive-dynamic run (Phase 7.1). All trends default to **flat** (no
    growth), so a zero-trend run is transparent bookkeeping over the static solves."""

    depreciation: float = DEFAULT_DEPRECIATION_RATE  # δ per year (5d.3 default 5%)
    labour_growth: float = 0.0  # exogenous labour-force growth per year (demographics)
    productivity_growth: float = 0.0  # exogenous Hicks-neutral TFP trend per year
    # Per-year premature capital retirement fraction (5d.3 stranded assets), e.g. {2030: 0.1}. A
    # year absent → 0. Applied to the OPENING stock in that year's accumulation step.
    retirement: dict[int, float] = field(default_factory=dict)

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
    """The result of a recursive-dynamic run: the per-year ``ResultSet`` plus the capital path."""

    result: ResultSet
    capital_stock: dict[int, float]  # end-of-year stock K by year
    investment: dict[int, float]  # nominal investment (share of benchmark GDP) by year
    growth: dict[int, float]  # capital growth rate K_{t+1}/K_t − 1 by year


def _manifest_capital_stock(manifest) -> float:
    cd = manifest.assumptions.get("capital_dynamics", {})
    if not cd.get("available"):
        raise ValueError(
            "recursive dynamics need a CGE with a capital factor AND a savings-investment account "
            f"(the benchmark stock–flow bridge is unavailable: {cd.get('reason', 'unknown')}). "
            "Use a SAM with a SAVINV account, e.g. toy_cge_gov."
        )
    k0 = cd["benchmark_capital_stock"]
    if len(k0) != 1:
        raise ValueError(
            "recursive dynamics currently support the single capital-region CGE (closed/gov and "
            f"open — one aggregate capital stock); this model has {len(k0)} capital regions. "
            "Multi-region capital accumulation is a documented follow-up (roadmap 7.1)."
        )
    return float(k0[0])


def _year_investment(res: ResultSet, year: int) -> float:
    """That year's nominal investment (share of benchmark GDP) from a single-year ResultSet."""
    d = res.data
    inv = d[(d["variable"] == "investment") & (d["scenario"] == "central") & (d["year"] == year)]
    if inv.empty:
        raise ValueError(f"no investment result for year {year}; the CGE reported none.")
    return float(inv["value"].iloc[0])


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

    # K0 from the benchmark stock–flow bridge (a cheap no-shock probe run at the first year).
    probe = run_scenario(
        scenario.model_copy(update={"shocks": [], "years": [years[0]]}),
        data_source=data_source,
        store=store,
    )
    k0 = _manifest_capital_stock(probe.manifest)

    frames: list[pd.DataFrame] = []
    capital_stock: dict[int, float] = {}
    investment: dict[int, float] = {}
    growth: dict[int, float] = {}

    k_t = k0
    for i, year in enumerate(years):
        labour_scale = (1.0 + config.labour_growth) ** i
        tfp_scale = (1.0 + config.productivity_growth) ** i
        # Capital and labour endowments scale to this year's stock/force; TFP is applied Hicks-
        # neutrally as an equivalent scale on both primary factors (documented simplification).
        cap_scale = (k_t / k0) * tfp_scale
        lab_scale = labour_scale * tfp_scale
        overrides = {"factor_endowment_scale": {"CAP": cap_scale, "LAB": lab_scale}}

        res = run_scenario(
            scenario.model_copy(update={"years": [year]}),
            data_source=data_source,
            store=store,
            data_overrides=overrides,
        )
        frames.append(res.data)

        inv_share = _year_investment(res, year)  # nominal investment / benchmark GDP
        # Investment is a GDP-share flow; convert to the same units as K (the user-cost stock is in
        # capital-income units = GDP-normalised too, since gdp0 = benchmark income). INV in stock
        # units = inv_share · gdp0 / gdp0-scale — but K0 is already in those units, so inv_share is
        # directly comparable to K0 (both GDP-normalised). Step the stock:
        r_t = float(config.retirement.get(year, 0.0))
        k_next = float(
            capital_next(k_t, inv_share, depreciation=config.depreciation, retirement=r_t)
        )

        investment[year] = inv_share
        capital_stock[year] = k_next
        growth[year] = k_next / k_t - 1.0
        k_t = k_next

    # Append the capital path as result rows so it flows through the ResultSet like any variable.
    data = pd.concat(frames, ignore_index=True)
    extra = []
    for year in years:
        extra.append(_rec("capital_stock", year, capital_stock[year]))
        extra.append(_rec("capital_growth", year, growth[year]))
    data = pd.concat([data, pd.DataFrame(extra)], ignore_index=True)

    # Reuse the last year's manifest; stamp the dynamic configuration onto it.
    manifest = res.manifest
    manifest.assumptions["recursive_dynamics"] = {
        "mode": "recursive_dynamic (bookkeeping between static solves; no perfect foresight)",
        "horizon_years": years,
        "depreciation_rate": config.depreciation,
        "labour_growth": config.labour_growth,
        "productivity_growth": config.productivity_growth,
        "retirement": {int(k): float(v) for k, v in config.retirement.items()},
        "benchmark_capital_stock": k0,
        "capital_stock_path": {int(k): round(v, 12) for k, v in capital_stock.items()},
        "note": (
            "Capital carried forward via K_{t+1}=(1−δ)(1−r)K_t+INV_t (Phase 5d.3); labour and TFP "
            "are exogenous trends applied as endowment scales. Single capital-region scope "
            "(closed/gov and open); multi-region capital accumulation is a follow-up."
        ),
    }
    result = ResultSet(data=data, manifest=manifest)
    result.validate_schema()
    return DynamicPath(
        result=result, capital_stock=capital_stock, investment=investment, growth=growth
    )


def _rec(variable: str, year: int, value: float) -> dict:
    return {
        "variable": variable,
        "sector": "__economy__",
        "region": "R",
        "year": int(year),
        "scenario": "central",
        "value": float(value),
    }
