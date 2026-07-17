"""Validation suite for Engine 1 (Leontief carbon-cost price model).

These are model-correctness checks tied to the equations in docs/models/io-price-model.md
§7. They overlap with the pytest unit tests deliberately: the pytest versions gate CI on
code changes; these run through the ``validate`` script as a standing, human-readable audit
of whether the *model* still reproduces its known answers and identities. Each references
the specific property it guards.
"""

from __future__ import annotations

import numpy as np

from cge.contracts.shocks import CarbonPrice
from cge.engines.io_price.engine import decompose, price_change
from cge.scenarios.loader import Scenario
from cge.validation.framework import check
from cge.validation.toy import toy_economy

SUITE = "io_price"


def _toy_arrays():
    io, sat = toy_economy()
    labels = list(io.A.columns)
    A = io.A.to_numpy(dtype=float)
    e = sat.data.loc["CO2"].reindex(labels).to_numpy(dtype=float)
    return labels, A, e


@check(SUITE, "analytic_matches_explicit_inverse")
def _analytic():
    """Eq (5): the linear-solve Δp equals the explicit (I−Aᵀ)⁻¹ τe to machine precision."""
    _, A, e = _toy_arrays()
    c = 100.0 * e
    dp = price_change(A, c)
    dp_ref = np.linalg.inv(np.eye(A.shape[0]) - A.T) @ c
    err = float(np.max(np.abs(dp - dp_ref)))
    return err < 1e-9, f"max|Δp − reference| = {err:.2e}", err, 1e-9


@check(SUITE, "zero_shock_zero_change")
def _zero():
    """τ = 0 ⇒ Δp = 0 exactly (no spurious price movement)."""
    _, A, e = _toy_arrays()
    dp = price_change(A, 0.0 * e)
    m = float(np.max(np.abs(dp)))
    return m == 0.0, f"max|Δp| at τ=0 is {m:.2e}", m, 0.0


@check(SUITE, "linearity_in_price")
def _linearity():
    """Linearity (assumption 5): doubling τ doubles Δp."""
    _, A, e = _toy_arrays()
    dp1 = price_change(A, 100.0 * e)
    dp2 = price_change(A, 200.0 * e)
    err = float(np.max(np.abs(dp2 - 2.0 * dp1)))
    return err < 1e-9, f"max|Δp(2τ) − 2·Δp(τ)| = {err:.2e}", err, 1e-9


@check(SUITE, "pass_through_adds_cost")
def _passthrough():
    """With A ≥ 0 and e ≥ 0, full Δp ≥ direct cost everywhere (upstream never subtracts)."""
    _, A, e = _toy_arrays()
    c = 100.0 * e
    dp = price_change(A, c)
    gap = float(np.min(dp - c))
    return gap >= -1e-9, f"min(Δp − direct) = {gap:.2e} (should be ≥ 0)", gap, 0.0


@check(SUITE, "decomposition_sums_to_total")
def _decomp():
    """Eq (6): direct + upstream tiers + residual reconstruct Δp exactly."""
    _, A, e = _toy_arrays()
    c = 100.0 * e
    parts = decompose(A, c, tiers=3)
    recon = sum(parts.values())
    err = float(np.max(np.abs(recon - price_change(A, c))))
    return err < 1e-9, f"max|Σparts − Δp| = {err:.2e}", err, 1e-9


@check(SUITE, "energy_most_exposed")
def _energy():
    """Plausibility: the emissions-intensive sector (energy) has the largest cost impact."""
    labels, A, e = _toy_arrays()
    dp = price_change(A, 100.0 * e)
    top = labels[int(np.argmax(dp))]
    ok = "energy" in top
    return ok, f"largest Δp is {top!r} (expected an energy sector)"


@check(SUITE, "coverage_filtering")
def _coverage():
    """A carbon price restricted to region A leaves region-B *direct* costs at zero
    (upstream can still leak via trade, so we check the direct term)."""
    from cge.engines.io_price.engine import _effective_price

    labels, _, _ = _toy_arrays()
    shock = CarbonPrice(price=100.0, coverage_regions=["A"])
    tau = _effective_price([shock], labels, 2020)
    b_direct = {lab: t for lab, t in zip(labels, tau, strict=True) if lab.startswith("B:")}
    ok = all(t == 0.0 for t in b_direct.values())
    return ok, f"region-B direct carbon price all zero under A-only coverage: {ok}"


@check(SUITE, "well_posedness_guard")
def _wellposed():
    """A non-productive economy (ρ(A) ≥ 1) is rejected rather than silently returning
    garbage (spec §4 precondition)."""
    _, A, e = _toy_arrays()
    bad = A * 10.0  # push spectral radius past 1
    try:
        price_change(bad, e)
        return False, "non-productive economy did NOT raise"
    except ValueError:
        return True, "non-productive economy correctly rejected"


# Independent hand-derived known-answer: full pipeline (τ·e·1e-6 through the Leontief
# inverse) on the toy at €100/t. Computed once, checked in; NOT the same formula the engine
# uses to self-check — this pins the *units and orientation*, not just the linear algebra.
_TOY_EXPECTED_100 = {
    "A:agriculture": 0.05931359921488204,
    "A:energy": 0.2467073520635376,
    "A:manufacturing": 0.14494885261298424,
    "B:agriculture": 0.05849474073408029,
    "B:energy": 0.24746703896761568,
    "B:manufacturing": 0.1452329848050412,
}


@check(SUITE, "known_answer_full_pipeline")
def _known_answer():
    """Full engine run on the toy at €100/t matches the checked-in hand-derived vector —
    an independent check of units, orientation and scaling, not just the solve."""

    from cge.runner import run_scenario

    scenario = Scenario(
        name="ka", engine="io_price", years=[2020], shocks=[CarbonPrice(price=100.0)]
    )
    df = run_scenario(scenario, data_source="toy").data
    main = df[df["variable"] == "price_change"]
    got = {f"{r.region}:{r.sector}": r.value for r in main.itertuples()}
    err = max(abs(got[k] - v) for k, v in _TOY_EXPECTED_100.items())
    return err < 1e-9, f"max|got − hand-derived| = {err:.2e}", err, 1e-9


@check(SUITE, "units_plausible_magnitude")
def _units_plausible():
    """Units sanity: a €100/t price on the toy yields fractional (percent-scale) price
    changes, not the ~1e3–1e9 values a missing unit conversion would produce."""
    from cge.runner import run_scenario

    scenario = Scenario(
        name="u", engine="io_price", years=[2020], shocks=[CarbonPrice(price=100.0)]
    )
    df = run_scenario(scenario, data_source="toy").data
    mx = float(df[df["variable"] == "price_change"]["value"].max())
    ok = 0.0 < mx < 1.0
    return ok, f"max Δp = {mx:.4f} (expected 0 < Δp < 1 for €100/t)", mx, 1.0


@check(SUITE, "gas_selection_distinct")
def _gas_selection():
    """gases=[CO2] and gases=[CH4] must give different results when both are in the data
    (the review found them bit-identical because gas selection was ignored)."""
    import numpy as np
    import pandas as pd

    from cge.contracts.data_objects import Provenance, SatelliteAccount
    from cge.engines.io_price.engine import _intensity_for_gases

    prov = Provenance(
        source="t", source_version="1", licence="x", reference_year=2020, retrieved="2026-07-17"
    )
    sat = SatelliteAccount(
        provenance=prov,
        name="GHG",
        units={"CO2": "t/MEUR", "CH4": "t/MEUR", "CO2e": "tCO2e/MEUR"},
        data=pd.DataFrame({"L0": [100.0, 10.0, 380.0]}, index=["CO2", "CH4", "CO2e"]),
    )
    co2, _ = _intensity_for_gases(sat, ["L0"], ["CO2"])
    ch4, _ = _intensity_for_gases(sat, ["L0"], ["CH4"])
    both, _ = _intensity_for_gases(sat, ["L0"], ["CO2", "CH4"])
    ok = not np.allclose(co2, ch4) and np.allclose(both, co2 + ch4)
    return ok, f"CO2={co2[0]}, CH4(×GWP)={ch4[0]}, combined additive={ok}"


@check(SUITE, "time_path_varies_by_year")
def _time_path():
    """A price path {2020:0, 2030:200} must produce year-varying results (the review found
    one shock vector copied to every year)."""
    from cge.runner import run_scenario

    scenario = Scenario(
        name="p",
        engine="io_price",
        years=[2020, 2025, 2030],
        shocks=[CarbonPrice(price=0.0, path={2020: 0.0, 2030: 200.0})],
    )
    df = run_scenario(scenario, data_source="toy").data
    by_year = df[df["variable"] == "price_change"].groupby("year")["value"].sum()
    ok = by_year[2020] < by_year[2025] < by_year[2030]
    return ok, f"totals by year: {by_year.round(4).to_dict()} (should increase)"


@check(SUITE, "engine_end_to_end")
def _end_to_end():
    """Full path: runner → registered engine → schema-valid ResultSet with assumptions."""
    from cge.runner import run_scenario

    scenario = Scenario(
        name="validate",
        engine="io_price",
        years=[2020],
        shocks=[CarbonPrice(price=100.0)],
    )
    result = run_scenario(scenario, data_source="toy")
    df = result.data
    has_price = (df["variable"] == "price_change").any()
    has_assumptions = "interpretation" in result.manifest.assumptions
    ok = has_price and has_assumptions and len(df) > 0
    return ok, f"end-to-end rows={len(df)}, price rows & assumptions present={ok}"
