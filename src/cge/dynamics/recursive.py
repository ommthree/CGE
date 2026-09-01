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
Aggregate productivity enters as a Hicks-neutral endowment-equivalent scale on both factors (a
documented simplification). Per-SECTOR structural change enters separately as a genuine
LABOUR-augmenting term on each sector's labour input (a ``ProductivityShock(mechanism=
"labour_augmenting")`` driven by the StructuralTrajectory's ``sector_productivity`` rates), so
capital deepening in the sector labour-productivity source is not double-counted against the model's
own capital accumulation (7b.2 review 2026-08-29).

Results are reported per year **relative to the original benchmark**, so capital accumulation and
the trends are VISIBLE in the level path (a growing stock raises output vs the benchmark). The
wrapper adds ``capital_stock`` and ``capital_growth`` result rows (one per region).

**Scope: all three CGE variants.** The closed/gov SAM and the open economy (Armington/CET + rest of
world) carry one aggregate capital stock; the multi-region CGE carries a **per-region capital path**
— each region's stock steps by its own investment, ``K_{t+1,r}=(1−δ)(1−r)K_{t,r}+INV_{t,r}``, and
the ``factor_endowment_scale`` hook moves each region's capital independently. Any variant needs a
savings-investment account to be dynamic-capable: ``toy_cge_gov`` (closed), ``toy_cge_open_gov``
(open), ``toy_cge_multi_gov`` (multi). Labour and productivity trends are exogenous. With the flat
``DynamicConfig`` scalars they are applied uniformly across regions; with a sourced
:class:`StructuralTrajectory` (Phase 7b.2) they are **per-region** (and per-sector for the
sectoral-productivity and emissions-intensity drivers), so region-specific trends exist.

**Capital is stepped every calendar year** between the first and last requested year, not only on
the (possibly sparse) requested years: the requested ``years`` are the *reporting* years and the
wrapper solves every intervening year internally so investment and depreciation accumulate annually
(a sparse ``[2025, 2030]`` reports the same 2030 stock as the full annual horizon). The stock is
stepped with the engine's ``investment_volume`` (a benchmark-price REAL flow in the same
GDP-normalised units as the stock), not the nominal ``investment`` share.
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
    # Whether to accept a structural run in which EVERY model region/sector falls through to the
    # global ``__all__`` path (no country/sector differentiation). Default False: such a run is
    # REJECTED (review P2 2026-08-29 — "unmapped label fails loudly" must hold at the RUN boundary,
    # not just be recorded after the fact). A caller that deliberately wants the uniform ``__all__``
    # trajectory on a real build opts in explicitly by setting this True. When the trajectory DOES
    # differentiate the model's labels (e.g. a concordance-mapped build trajectory), the gate is a
    # no-op.
    allow_uniform_fallback: bool = False

    def __post_init__(self) -> None:
        for name in ("depreciation", "labour_growth", "productivity_growth"):
            v = getattr(self, name)
            if not np.isfinite(v):
                raise ValueError(f"DynamicConfig.{name} must be finite; got {v!r}")
        if not (0.0 <= self.depreciation <= 1.0):
            raise ValueError(f"depreciation δ must be in [0, 1]; got {self.depreciation}")
        # Flat growth rates below −100 %/yr (≤ −1) drive the endowment to zero-or-negative and only
        # fail LATER inside the solve with an opaque zero-factor error; above +100 %/yr is
        # implausible for an annual labour/TFP trend. Reject at construction so the config is honest
        # (review P2 2026-08-28): a −1 growth previously passed here and blew up mid-run.
        for name in ("labour_growth", "productivity_growth"):
            v = getattr(self, name)
            if not (-1.0 < v < 1.0):
                raise ValueError(
                    f"DynamicConfig.{name} must be a per-year growth rate in (−1, 1); got {v} "
                    "(a rate ≤ −100%/yr zeroes the endowment; > +100%/yr is implausible)."
                )
        # Coerce and validate the retirement keys ONCE here, so the numerical loop (which looks up
        # integer years) and the manifest agree on exactly which years carry a retirement (review P2
        # 2026-08-28): a string "2025" or float 2025.5 key was silently ignored by the int-keyed
        # loop yet coerced to {2025: …} in the manifest, FALSELY reporting a retirement that never
        # ran. We accept only keys that are exactly integer-valued and rebind the dict to int keys.
        coerced: dict[int, float] = {}
        for y, r in self.retirement.items():
            if isinstance(y, bool) or not isinstance(y, (int, float)):
                raise ValueError(
                    f"retirement year key {y!r} must be an integer calendar year; "
                    f"got {type(y).__name__}."
                )
            if isinstance(y, float) and not y.is_integer():
                raise ValueError(
                    f"retirement year key {y!r} is not an integer calendar year; the accumulation "
                    "loop steps whole years, so a fractional year would be silently ignored."
                )
            yi = int(y)
            if yi in coerced:
                raise ValueError(f"retirement has duplicate year key {yi} after int coercion")
            if not np.isfinite(r) or not (0.0 <= r <= 1.0):
                raise ValueError(f"retirement[{y}] must be a fraction in [0, 1]; got {r!r}")
            coerced[yi] = float(r)
        self.retirement = coerced


@dataclass
class DynamicPath:
    """The result of a recursive-dynamic run: the per-year ``ResultSet`` plus the capital path.

    The path dicts are keyed by year. For the single-region variants (closed/gov/open — one
    aggregate capital stock) the values are **floats** (back-compatible). For the multi-region CGE
    they are 1-D ``numpy`` arrays, one entry per region (ordered as ``recursive_dynamics
    ['capital_regions']`` in the result manifest)."""

    result: ResultSet
    capital_stock: dict[int, float | np.ndarray]  # end-of-year stock K by year
    investment: dict[int, float | np.ndarray]  # real investment VOLUME (share of benchmark GDP) /yr
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


def _probe_sectors(probe: ResultSet) -> list[str]:
    """The model's sectors, read off the probe run's per-sector ``volume_change`` rows (the sector
    labels the engine actually solved on) — the authoritative sector list the sectoral-productivity
    driver enumerates so every model sector is driven, even by an ``__all__``-only trajectory or one
    whose keys do not match real sector names (review P1)."""
    d = probe.data
    vc = d[d["variable"] == "volume_change"]
    # Preserve first-seen order; drop the economy-wide placeholder used by non-sector rows.
    seen: list[str] = []
    for s in vc["sector"]:
        if s != "__economy__" and s not in seen:
            seen.append(str(s))
    return seen


def _probe_va_shares(manifest, regions: list[str], sectors: list[str]) -> dict:
    """The per-sector benchmark **value-added share** v_i = VA_i / Σ_j VA_j the engine stamped, used
    as the weights for the geometric-mean denominator the labour-augmenting sectoral-productivity
    drift is measured against (``_sector_productivity_shocks``), so a large sector's drift is
    weighted by its economic SIZE and the biases net out per region (review P1 2026-08-31: the
    earlier code used the labour COMPOSITION s_L here, weighting small and large sectors alike).

    Returns the flat ``{sector: v_i}`` map for the single-region variants and the nested
    ``{region: {sector: v_i}}`` map for multi. When the engine exposed no shares (e.g. a model with
    no LAB factor), returns an empty map — the caller then falls back to equal weighting, preserving
    the prior behaviour rather than failing."""
    lvs = manifest.assumptions.get("labour_va_shares", {})
    if not lvs.get("available"):
        return {}
    return lvs.get("va_share", {})


def _year_investment(res: ResultSet, year: int, regions: list[str]) -> np.ndarray:
    """That year's REAL investment VOLUME (share of benchmark GDP) per region, ordered by regions.

    Reads ``investment_volume`` — the investment demand valued at BENCHMARK prices — NOT the nominal
    ``investment`` share (review P1): the capital stock K is a real quantity, so its accumulation
    flow must be a real volume, else investment-price movements masquerade as capital formation. For
    the multi CGE ``investment_volume`` is normalised by GLOBAL GDP, matching the global-normalised
    K (the nominal ``investment`` share uses each region's OWN GDP, which does NOT match K).

    The single-region variants report one ``investment_volume`` row with region label ``R``; the
    multi CGE reports one per region. Returns a 1-D array aligned to ``regions``."""
    d = res.data
    inv = d[
        (d["variable"] == "investment_volume") & (d["scenario"] == "central") & (d["year"] == year)
    ]
    if inv.empty:
        raise ValueError(
            f"no investment_volume result for year {year}; the CGE reported none (a dynamic run "
            "needs a savings-investment account)."
        )
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
    data_overrides: dict | None = None,
) -> DynamicPath:
    """Run ``scenario`` recursively-dynamically over its ``years``, carrying capital forward.

    **The capital path is stepped every calendar year** between the first and last requested year,
    NOT only on the (possibly sparse) requested years (review P1): investment and depreciation are
    annual flows, so skipping 2026–2029 in a ``[2025, 2030]`` scenario would omit four years of
    accumulation and report the wrong 2030 stock. The wrapper therefore solves the FULL consecutive
    annual horizon internally and reports only the requested years — the requested ``years`` are the
    REPORTING years, the internal solve years are every year in ``[min, max]``.

    ``data_overrides`` are static override entries merged into EVERY per-year solve (on top of the
    per-year capital/labour/emissions/productivity overrides the wrapper builds). This is how a
    physical ``nature_state`` scenario runs end-to-end on a SAM-only dynamic source: the exposure
    triple — ``nature_iosystem`` (the exposure IOSystem), ``EncoreDependencies`` and
    ``ConcordanceMap`` — is injected here so NatureStress translates into per-sector
    ProductivityShocks each year, while the toy CGE SAM supplies the capital-carrying economic core
    (review 7b.2 2026-08-30). These are nature CONTROL keys, stripped before the strict CGE engine.

    Returns a ``DynamicPath`` whose ``result`` is the concatenated per-REPORTING-year ResultSet
    (with added ``capital_stock``/``capital_growth`` rows) and whose dicts give the capital path at
    the reporting years."""
    config = config or DynamicConfig()
    data_overrides = dict(data_overrides or {})
    report_years = sorted(scenario.years)
    base_year = report_years[0]
    # Solve EVERY calendar year from the first to the last requested year, so investment and
    # depreciation accumulate annually; report only the requested years (review P1 — sparse-year
    # path).
    solve_years = list(range(base_year, report_years[-1] + 1))
    report_set = set(report_years)

    # K0 (per capital region) from the benchmark stock–flow bridge (a cheap no-shock probe run at
    # the first year), plus the region labels the vector is ordered by.
    probe = run_scenario(
        scenario.model_copy(update={"shocks": [], "years": [base_year], "nature_state": []}),
        data_source=data_source,
        store=store,
    )
    k0, regions = _manifest_capital(probe.manifest)
    multi = regions != ["R"]  # per-region capital path vs one aggregate stock
    # The model's sectors, read off the probe's per-sector output rows — needed so the sectoral-
    # productivity driver can emit a shock for EVERY sector (review P1: enumerating the model's
    # sectors, not just the trajectory's keys, so an ``__all__``-only trajectory or one whose keys
    # do
    # not match real sector names still drives every sector).
    sectors = _probe_sectors(probe)
    # Per-region per-sector benchmark VALUE-ADDED SHARE v_i = VA_i/ΣVA_j (7b.2 review 2026-08-31),
    # read off the probe manifest — the weights for the geometric-mean denominator the labour-
    # augmenting sectoral-productivity DRIFT is measured against, so a large sector's drift is
    # weighted by its economic SIZE and the per-region biases net out. (Was wrongly the labour
    # composition s_L before — review P1 2026-08-31.)
    va_shares = _probe_va_shares(probe.manifest, regions, sectors)

    # "Unmapped label fails loudly" at the RUN boundary (review P2 2026-08-29): if a structural
    # trajectory is supplied but EVERY model region/sector falls through to the global ``__all__``
    # path — i.e. the trajectory does not differentiate THIS build's labels at all — reject the run
    # unless the caller explicitly opted into the uniform fallback. This is what forces a real
    # coarse-v3 build to be run with a concordance-mapped trajectory
    # (``structural_trajectories_for_build``) rather than the bare archetype trajectory silently
    # collapsing to ``__all__``.
    if config.structural is not None and not config.allow_uniform_fallback:
        cov = _structural_coverage(config.structural, regions, sectors)
        if cov["all_fallthrough"]:
            raise ValueError(
                "structural trajectory does not differentiate any of this build's labels — every "
                f"model region {cov['model_regions']} and sector {cov['model_sectors']} falls "
                "through to the global '__all__' path (no country/sector differentiation). On a "
                "real EXIOBASE build, map the trajectory to the build's labels first with "
                "cge.data.structural.structural_trajectories_for_build(regions, sectors); or set "
                "DynamicConfig(allow_uniform_fallback=True) to run the uniform __all__ trajectory "
                "deliberately."
            )

    # Expand any Phase-6b nature_state pathways ONCE over the FULL annual horizon (review P1): the
    # rate-form start year and recovery hysteresis depend on the complete year sequence, so slicing
    # the pathway one year at a time (as a per-year runner call would) resets the start year and
    # erases the preceding states. Expanding here yields NatureStress shocks carrying the full
    # multi-year severity `path`; each internal solve then reads its own year off that path. The
    # per-year scenario copies clear `nature_state` so the runner never re-expands (and re-slices)
    # it.
    base_nature_shocks = scenario.expanded_shocks(solve_years) if scenario.nature_state else None

    frames: list[pd.DataFrame] = []
    capital_stock: dict[int, np.ndarray] = {}  # end-of-year stock K per region, by year
    investment: dict[
        int, np.ndarray
    ] = {}  # real investment volume (share of benchmark GDP) /region
    growth: dict[int, np.ndarray] = {}  # capital growth rate K_{t+1}/K_t − 1 per region
    child_hashes: dict[int, str] = {}  # per solve-year child scenario_hash (review P2a 2026-08-28)

    # Whether an emissions-intensity trajectory is active (7b.2). We no longer read the benchmark
    # carbon_cost_share (review P1a 2026-08-28): a real IO/satellite build has NONE — its intensity
    # is derived inside the engine — so pre-scaling a supplied share silently skipped the
    # trajectory. Instead we build a PRICE-INDEPENDENT per-sector emissions_intensity_scale the
    # engine applies to whatever intensity it computed (supplied OR IO-derived).
    has_emissions = _has_emissions_traj(config)

    emissions_reference: np.ndarray | None = None  # base-year covered emissions per region (7b.2)
    k_t = k0.copy()  # 1-D array, one entry per capital region
    for year in solve_years:
        # Cumulative labour-supply and productivity scales from the base year to this year, PER
        # REGION. With a StructuralTrajectory (7b.2) these compound the sourced per-year rates over
        # the actual year gaps; without one they fall back to the flat DynamicConfig scalars,
        # uniform
        # across regions (Phase 7.1 back-compat).
        labour_scale = _trend_scale(config, base_year, year, regions, "labour")
        tfp_scale = _trend_scale(config, base_year, year, regions, "productivity")
        # Capital and labour endowments scale to this year's stock/force; TFP is applied Hicks-
        # neutrally as an equivalent scale on both primary factors (documented simplification).
        # Capital scale is PER REGION (each region carries its own stock).
        cap_scale = (k_t / k0) * tfp_scale  # 1-D array per region
        lab_scale = labour_scale * tfp_scale  # 1-D array per region
        overrides = {
            **data_overrides,
            "factor_endowment_scale": _factor_scale(cap_scale, lab_scale, regions, multi),
        }
        # Emissions-intensity driver (7b.2, per sector): a PRICE-INDEPENDENT decarbonisation scale
        # the engine multiplies into BOTH the physical intensity (so covered emissions fall) and the
        # priced wedge (so a decarbonising sector faces a smaller cost). Built from the trajectory's
        # cumulative ∏(1+rate) factor per (region,) sector — works on real IO builds where no
        # carbon_cost_share exists (review P1a 2026-08-28). The base-year covered-emissions
        # reference is fed back so covered_emissions_change is measured against the BASE YEAR (a
        # within-year uniform scale otherwise cancels in the same-year ratio, hiding the physical
        # decarbonisation) — a PER-REGION reference vector (multi needs a reference per region, not
        # a summed scalar).
        if has_emissions:
            overrides["emissions_intensity_scale"] = _emissions_intensity_scale_override(
                config, base_year, year, sectors, regions, multi
            )
            if emissions_reference is not None:
                overrides["covered_emissions_reference"] = _emissions_reference_override(
                    emissions_reference, regions, multi
                )

        # Sectoral-productivity drift (7b.2, structural change): synthesize per-sector
        # labour-augmenting ProductivityShocks (the engine's labour-augmenting channel scales that
        # sector's labour input), so sector-biased productivity shifts the output mix endogenously.
        # Composed with the scenario's shocks (and any pre-expanded nature shocks). nature_state is
        # cleared — expanded above. In multi mode the shocks are PER (region, sector) so each region
        # uses its OWN VA-share-weighted mean (review P1b 2026-08-28 / P1 2026-08-31).
        year_shocks = list(base_nature_shocks or scenario.shocks) + _sector_productivity_shocks(
            config, base_year, year, sectors, regions, multi, va_shares
        )

        res = run_scenario(
            scenario.model_copy(
                update={"years": [year], "shocks": year_shocks, "nature_state": []}
            ),
            data_source=data_source,
            store=store,
            data_overrides=overrides,
        )
        # Record each solve-year's child scenario hash so the dynamic manifest can carry the full
        # per-year provenance chain, not just the last static solve (review P2a 2026-08-28).
        child_hashes[year] = res.manifest.scenario_hash
        # Capture the base-year covered emissions PER REGION as the reference for later years
        # (7b.2).
        if has_emissions and year == base_year:
            emissions_reference = _year_covered_emissions_by_region(res, year, regions)

        inv_share = _year_investment(res, year, regions)  # per-region real investment volume / GDP0
        # investment_volume is a benchmark-price real flow in the SAME GDP-normalised units as K0
        # (review P1): both are shares of benchmark GDP, so the flow steps the stock directly.
        r_t = float(config.retirement.get(year, 0.0))
        k_next = capital_next(k_t, inv_share, depreciation=config.depreciation, retirement=r_t)

        # Only the REQUESTED reporting years contribute result rows and path entries; the
        # intervening years exist purely to accumulate capital (review P1 — sparse vs annual).
        if year in report_set:
            frames.append(res.data)
            investment[year] = inv_share
            capital_stock[year] = k_next
            growth[year] = k_next / k_t - 1.0
        k_t = k_next

    years = report_years  # the reported horizon (kept name for the manifest/row-building below)
    # Append the capital path as result rows (one per region) so it flows through the ResultSet like
    # any variable.
    data = pd.concat(frames, ignore_index=True)
    extra = []
    for year in years:
        for ri, region in enumerate(regions):
            extra.append(_rec("capital_stock", year, float(capital_stock[year][ri]), region))
            extra.append(_rec("capital_growth", year, float(growth[year][ri]), region))
    data = pd.concat([data, pd.DataFrame(extra)], ignore_index=True)

    # Reuse the last year's manifest as the base, then stamp the dynamic configuration AND fix its
    # identity (review P2a 2026-08-28): a dynamic run's manifest previously reported only the LAST
    # static solve, so its scenario_hash was that of the final year's scenario — unchanged when the
    # dynamic depreciation/retirement/trends changed, and missing the original scenario's physical
    # nature_state block. We overwrite the scenario_hash with a hash of the ORIGINAL scenario plus
    # the normalized DynamicConfig, re-stamp the physical nature_state block, and record each solve
    # year's child hash.
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
        "trend_source": _trend_provenance(config, regions, sectors),
        # Per-solve-year child scenario hashes (review P2a): the full provenance chain of the
        # internal static solves the dynamic result is composed of.
        "child_run_hashes": {int(y): h for y, h in child_hashes.items()},
        "note": (
            "Capital carried forward via K_{t+1}=(1−δ)(1−r)K_t+INV_t (Phase 5d.3); labour and TFP "
            "are exogenous trends applied as endowment scales. Single aggregate capital stock for "
            "the closed/gov/open variants; a per-region capital path for the multi-region CGE."
        ),
    }
    # Re-stamp the ORIGINAL scenario's physical nature_state provenance (review P2a): the per-year
    # scenario copies cleared nature_state (it was pre-expanded once above), so the reused per-year
    # manifest lost the physical-state block — a dynamic physical-state run and an equivalent
    # hand-written NatureStress dynamic run would otherwise be indistinguishable.
    if scenario.nature_state:
        from cge.runner import _nature_state_manifest

        manifest.assumptions["nature_state"] = _nature_state_manifest(scenario)
    # Overwrite the scenario identity so it reflects the DYNAMIC run: the original scenario (with
    # its nature_state intact) PLUS the normalized dynamic configuration. A changed depreciation,
    # retirement schedule, or trajectory now moves the manifest's scenario_hash (review P2a).
    manifest.scenario_hash = _dynamic_scenario_hash(scenario, config)
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
                # Labour-SUPPLY growth = population growth compounded WITH participation growth
                # (both
                # proportional annual rates): (1+pop)(1+part), the exact multiplicative step, NOT
                # the
                # additive pop+part approximation (review P2 2026-08-23). The two are proportional
                # growth rates (fractions/yr), so this multiplies the whole labour-supply index.
                step = (1.0 + traj.rate("population", region, y)) * (
                    1.0 + traj.rate("labour_participation", region, y)
                )
            else:
                step = 1.0 + traj.rate("productivity", region, y)
            acc *= step
        out[ri] = acc
    return out


def _has_emissions_traj(config: DynamicConfig) -> bool:
    """Whether the config carries an emissions-intensity sector driver (7b.2)."""
    return bool(config.structural and config.structural.sector_rates.get("emissions_intensity"))


def _cumulative_sector_level(traj, driver: str, sector: str, base_year: int, year: int) -> float:
    """Cumulative multiplier level from ``base_year`` to ``year`` for a per-sector driver, minus 1
    (so 0.0 = no change) — the sourced annual sector rates compounded over the actual gap, the same
    piecewise-constant compounding as the per-region trends. Returns the *fractional* cumulative
    change, i.e. ∏(1+rate) − 1."""
    acc = 1.0
    for y in range(base_year, year):
        acc *= 1.0 + traj.sector_rate(driver, sector, y)
    return acc - 1.0


def _sector_productivity_shocks(
    config: DynamicConfig,
    base_year: int,
    year: int,
    sectors: list[str],
    regions: list[str],
    multi: bool,
    va_shares: dict,
) -> list:
    """Synthesize LABOUR-AUGMENTING ProductivityShocks for the sectoral-drift driver (7b.2
    structural change), for EVERY model sector in ``sectors`` (an ``__all__``-only trajectory, or
    one whose keys do not match real EXIOBASE names, still drives every sector).

    **A HEURISTIC structural-drift parameter, not an identified technology series (review P1
    2026-08-31).** The sourced sector series is observed *labour-productivity* (output-per-hour)
    growth, which mixes true labour-augmenting technology with capital deepening, utilisation
    and labour-composition effects (g_{Y/L}=g_A+s_K·g_{K/L}+s_L·g_φ). This driver does NOT decompose
    those out — it uses the series as a transparent, illustrative knob for how the OUTPUT MIX drifts
    when one sector's measured productivity outpaces another's. It is implemented on the *labour*
    input (``mechanism="labour_augmenting"``) rather than as Hicks-neutral TFP so it stays a
    composition lever rather than re-imposing an aggregate level — but we do NOT claim it identifies
    the technology term or that it removes capital-deepening double-counting; a growth-accounting
    decomposition (needing sector K/L data the project does not vendor) is the documented follow-up.

    **Structural DRIFT, not a level.** The aggregate productivity level is ALREADY carried by the
    TFP endowment scale, so this driver contributes only the sector COMPOSITION drift. Each sector's
    cumulative labour-productivity level is expressed RELATIVE to the **VA-share-weighted** geo
    mean of all sectors' levels — weighted by each sector's SHARE OF VALUE ADDED v_i=VA_i/ΣVA_j
    (review P1 2026-08-31: was wrongly the labour composition s_L, which weighted a 1%-of-economy
    sector the same as a 99% one). So a large sector's drift dominates the mean and small sectors do
    not swing it: a faster sector gets φ>1, a slower one φ<1, and the VA-weighted geometric mean of
    the biases is 1 by construction. NOTE this cost-neutrality of the *mean* is geometric, not the
    aggregate model cost effect (which for CES is not φ^{−s_L}); a transparent normalisation, not
    an exact general-equilibrium neutrality claim. The mean is a within-region quantity, computed
    per region — shocks are per (region, sector) in multi mode (engine honours coverage_regions).

    Empty when the config has no ``sector_productivity`` driver, so a run without it is
    byte-identical to Phase 7.1."""
    traj = config.structural
    if not (traj and traj.sector_rates.get("sector_productivity")):
        return []
    from cge.contracts.shocks import ProductivityShock

    shocks = []
    for region in regions:
        share_by_sector = va_shares.get(region, {}) if multi else va_shares
        # Cumulative labour-productivity LEVEL per sector (∏(1+rate)); the sector_rate lookup falls
        # back to the ``__all__`` sector path so a global-only trajectory drives every sector.
        levels = {
            s: _cumulative_sector_level(traj, "sector_productivity", s, base_year, year) + 1.0
            for s in sectors
        }
        # VA-SHARE-weighted GEOMETRIC mean of the sector levels — the "average" the drift is
        # measured against, so a large sector's drift dominates and no aggregate level is re-imposed
        # (review P1 2026-08-31). Weights are each sector's SHARE of regional value added
        # v_i=VA_i/ΣVA_j (engine-stamped); a sector with no stamped weight falls to equal weighting.
        weights = np.array([max(float(share_by_sector.get(s, 0.0)), 0.0) for s in sectors])
        if weights.sum() <= 0:
            weights = np.ones(len(sectors))
        weights = weights / weights.sum()
        log_mean = float(np.sum(weights * np.log([levels[s] for s in sectors])))
        mean_level = float(np.exp(log_mean))
        for sector in sectors:
            # RELATIVE labour-productivity deviation from the weighted mean → the labour-augmenting
            # factor φ. A sector at the mean gets φ=1 (no drift). delta = φ − 1, floored at −1.
            phi = levels[sector] / mean_level if mean_level > 0 else levels[sector]
            delta = max(phi - 1.0, -1.0)
            kwargs = {
                "delta": delta,
                "coverage_sectors": [sector],
                "mechanism": "labour_augmenting",
            }
            if multi:
                kwargs["coverage_regions"] = [region]
            shocks.append(ProductivityShock(**kwargs))
        if not multi:
            break  # single-region: one pass over the sole "R" region is enough
    return shocks


def _emissions_intensity_scale_override(
    config: DynamicConfig,
    base_year: int,
    year: int,
    sectors: list[str],
    regions: list[str],
    multi: bool,
) -> dict:
    """The per-(region,)sector PRICE-INDEPENDENT emissions-intensity decarbonisation scale for
    ``year`` (Phase 7b.2, review P1a 2026-08-28), fed to the engine's ``emissions_intensity_scale``
    hook. Each sector's factor is the cumulative intensity multiplier ∏(1+rate) from ``base_year``:
    a negative (decarbonising) rate gives a factor < 1, so the engine shrinks BOTH the sector's
    physical emission intensity (covered emissions fall) AND its priced carbon wedge.

    Enumerates every MODEL sector (the ``sector_rate`` lookup falls back to ``__all__``), so an
    ``__all__``-only trajectory decarbonises every sector — the same completeness fix as the
    sectoral-productivity driver. Single-region: ``{sector: factor}``. Multi: ``{region: {sector:
    factor}}`` (each region gets the same sector rates unless a region-keyed sector rate is added
    later; the shape lets the engine apply it per (region, sector))."""
    traj = config.structural

    def factor(sector: str) -> float:
        acc = 1.0
        for y in range(base_year, year):
            acc *= 1.0 + traj.sector_rate("emissions_intensity", sector, y)
        return acc

    flat = {s: factor(s) for s in sectors}
    if not multi:
        return flat
    return {r: dict(flat) for r in regions}


def _year_covered_emissions_by_region(
    res: ResultSet, year: int, regions: list[str]
) -> np.ndarray | None:
    """The base-year absolute covered emissions PER REGION the engine emitted for ``year`` (Phase
    7b.2), aligned to ``regions``, used as the reference the intensity driver measures later years
    against. Per REGION (not a summed scalar) because the multi variant reports and needs a per-
    region reference; the single-region variants report one row on region ``R`` and this returns a
    length-1 array. None if the engine emitted none (no priced/covered sector)."""
    d = res.data
    ce = d[
        (d["variable"] == "covered_emissions_benchmark")
        & (d["scenario"] == "central")
        & (d["year"] == year)
    ]
    if ce.empty:
        return None
    by_region = dict(zip(ce["region"], ce["value"], strict=False))
    # The single-region engines emit the row on the economy region label the engine uses; the multi
    # engine emits one row per region. Map onto `regions` (fall back to the sole value when a
    # single-region engine used a different economy label than "R").
    if len(by_region) == 1 and len(regions) == 1:
        return np.array([float(next(iter(by_region.values())))], dtype=float)
    # A region with NO covered-emissions row (its sectors carry zero intensity / no coverage) gets a
    # ZERO reference — the engine then emits no covered-emissions change for it (a zero reference
    # yields None in _covered_emissions_change), leaving the COVERED regions' references intact
    # (review P1a 2026-08-28: the old code discarded the WHOLE reference if any region was
    # uncovered, so a shipped multi SAM with one uncovered region wiped out the decarbonisation
    # signal for the covered region too — the decarbonising run then reported HIGHER emissions).
    ref = np.array([float(by_region.get(r, 0.0)) for r in regions], dtype=float)
    if not np.any(ref > 0.0):
        return None  # no region has any covered emissions → no reference at all (within-year)
    return ref


def _emissions_reference_override(
    reference: np.ndarray, regions: list[str], multi: bool
) -> float | dict:
    """Build the ``covered_emissions_reference`` override from the per-region base-year vector. The
    single-region engines take a scalar (the sole region's reference); the multi engine takes a
    ``{region: reference}`` map so each region's covered-emissions change is measured against its
    OWN base year (review P1 — a summed scalar reference under-weights per-region decarb)."""
    if not multi:
        return float(reference[0])
    return {r: float(reference[ri]) for ri, r in enumerate(regions)}


def _dynamic_scenario_hash(scenario: Scenario, config: DynamicConfig) -> str:
    """A content hash of the DYNAMIC run's identity: the ORIGINAL scenario (still carrying its
    nature_state) plus the NORMALIZED DynamicConfig (review P2a 2026-08-28).

    A recursive run's identity is the scenario AND the dynamic configuration (depreciation, trends,
    retirement, trajectory) — not just the last year's static solve. Hashing both means a changed
    depreciation, retirement schedule, or structural trajectory moves the manifest's scenario_hash,
    and a physical-state dynamic run hashes differently from an equivalent bare-NatureStress one."""
    from cge.contracts.provenance import content_hash

    traj = config.structural
    normalized_config = {
        "depreciation": config.depreciation,
        "labour_growth": config.labour_growth,
        "productivity_growth": config.productivity_growth,
        # Retirement is already int-keyed and validated in __post_init__; sort for a stable hash.
        "retirement": {int(y): float(r) for y, r in sorted(config.retirement.items())},
        # The structural trajectory's numeric rate tables (not just its provenance) — a changed rate
        # must move the hash, the same discipline as _trend_provenance's rate_tables_hash.
        "structural": (
            None if traj is None else {"rates": traj.rates, "sector_rates": traj.sector_rates}
        ),
    }
    return content_hash(
        {"scenario": scenario.model_dump(mode="json"), "dynamic_config": normalized_config}
    )


def _structural_coverage(traj, regions: list[str], sectors: list[str]) -> dict:
    """Diagnose whether the trajectory actually DIFFERENTIATES the model's labels or every one falls
    through to the global ``__all__`` default (review P1c 2026-08-28).

    For each driver, count how many model regions/sectors have an EXPLICIT keyed path vs how many
    fall through to ``__all__``. ``all_fallthrough`` is True when NO model label matched an explicit
    key on ANY driver — i.e. the run claims sourced structural detail but every region/sector got
    the same uniform trajectory (no country differentiation, no sectoral composition drift). The
    manifest stamps this so a supposedly-real run with no differentiation is visible, rather than
    silently overclaiming (the concordance-mapped trajectory binds real labels explicitly, so a
    genuine real build reports ``all_fallthrough = False``)."""
    region_hits = 0
    sector_hits = 0
    per_driver: dict = {}
    for driver, by_key in traj.rates.items():
        keyed = [r for r in regions if r in by_key]
        region_hits += len(keyed)
        per_driver[driver] = {
            "explicit": sorted(keyed),
            "fell_through_to_all": sorted(r for r in regions if r not in by_key),
        }
    for driver, by_key in traj.sector_rates.items():
        keyed = [s for s in sectors if s in by_key]
        sector_hits += len(keyed)
        per_driver[driver] = {
            "explicit": sorted(keyed),
            "fell_through_to_all": sorted(s for s in sectors if s not in by_key),
        }
    return {
        "model_regions": sorted(regions),
        "model_sectors": sorted(sectors),
        "explicit_region_matches": region_hits,
        "explicit_sector_matches": sector_hits,
        "all_fallthrough": region_hits == 0 and sector_hits == 0,
        "per_driver": per_driver,
    }


def _trend_provenance(config: DynamicConfig, regions: list[str], sectors: list[str]) -> dict:
    """Record how the labour/productivity trends were set — the flat fallback scalars (Phase 7.1) or
    a sourced :class:`StructuralTrajectory` (Phase 7b.2), with its provenance and per-entry cites —
    so a run's manifest is self-documenting about which drove it. Also records a structural-coverage
    diagnostic (review P1c) so a run that claims sourced detail but sees every model label fall
    through to ``__all__`` (no differentiation) is visible in the manifest."""
    traj = config.structural
    if traj is None:
        return {
            "kind": "flat",
            "labour_growth": config.labour_growth,
            "productivity_growth": config.productivity_growth,
            "note": "flat uniform trends (Phase 7.1 fallback); no sourced structural trajectory.",
        }
    from cge.contracts.provenance import content_hash

    # Content hash of the ACTUAL numeric rate tables (review P2 2026-08-23): the citations and
    # confidence alone do not pin the numbers, so a changed rate would leave an identical manifest.
    # Hashing the rate tables makes any edit to a rate move the manifest; the full tables are also
    # stamped so a run is reconstructible without the source file.
    rate_tables = {"rates": traj.rates, "sector_rates": traj.sector_rates}
    return {
        "kind": "structural_trajectory",
        "provenance": {
            "source": traj.provenance.source,
            "source_version": traj.provenance.source_version,
            "licence": traj.provenance.licence,
            "retrieved": traj.provenance.retrieved,
        },
        "region_drivers": sorted(traj.rates),
        "sector_drivers": sorted(traj.sector_rates),
        "drivers": sorted(traj.rates) + sorted(traj.sector_rates),
        "sources": dict(traj.sources),
        "confidence": dict(traj.confidence),
        "rate_tables": rate_tables,
        "rate_tables_hash": content_hash(rate_tables),
        # Review P1c 2026-08-28: prove the trajectory differentiates the model's OWN labels (via the
        # concordance) rather than every label collapsing to __all__.
        "coverage": _structural_coverage(traj, regions, sectors),
        "note": (
            "Phase 7b.2 sourced trajectories: per-region labour-supply (population×participation) "
            "and productivity growth as endowment scales; per-sector productivity drift as a "
            "HEURISTIC labour-augmenting composition lever (VA-share-weighted zero-mean, not an "
            "identified technology series) and emissions-intensity decarbonisation scaling the "
            "per-sector intensity, all compounded from the cited annual rates. 'coverage' reports "
            "whether the model's labels are explicitly keyed (differentiated) or fall through to "
            "__all__ (see the structural concordance)."
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
