"""Capital accumulation identity (Phase 5d.3 — the mechanism Phase 7.1 will call).

A **standalone, stateless** perpetual-inventory update: given a capital stock, this period's
investment, a depreciation rate, and an optional premature-retirement fraction, return next
period's stock. Deliberately **not** wired into the equilibrium solve — 5d.3's scope is the
*identity*, unit-tested in isolation, ready for the recursive-dynamic wrapper (roadmap Phase 7.1)
to call between static solves. The wrapper (a multi-year loop that re-solves the CGE each year
with the updated stock) is Phase 7.1's job, not this module's.

**The identity** [perpetual-inventory method, OECD2009 ch. 5]:

    K_{t+1} = (1 − δ) · (1 − r) · K_t + INV_t

where ``δ`` is the depreciation rate (fraction of the stock that wears out per period) and ``r``
is an optional **premature-retirement** fraction — an exogenous, scenario-specified write-off of
capital *before* its natural depreciation (e.g. fossil capital stranded by a carbon shock). Both
apply to the *opening* stock; investment adds the new vintage. With ``r = 0`` this is the textbook
law of motion.

**Granularity is the caller's choice.** The stock/investment/retirement arrays are elementwise
aligned and can be any shape — a scalar aggregate, per-region, or per-region-sector — because the
identity is elementwise. 5d.3 recommends and Phase 7.1 will use **region-level** capital (matching
the single aggregate capital factor per region in the CGE's ``factors``); sector-specific vintage
capital needs a capital-mobility-across-sectors assumption that is a documented future extension,
not modelled here.

**Out of scope (documented limitations):**
- **Endogenous stranding** — capital exiting because its expected return fell below a threshold.
  Retirement here is an *exogenous* scenario input, not a modelled investment decision.
- **The multi-year loop itself** — this returns one step; Phase 7.1 owns the iteration,
  demographics, and productivity trend between solves.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# Documented default depreciation rate. 5%/yr is the standard applied central value for aggregate
# fixed capital (OECD capital-measurement manual [OECD2009]); overridable per scenario, the same
# "central default + documented override, not per-cell guesswork" discipline used for elasticities.
DEFAULT_DEPRECIATION_RATE = 0.05


def capital_next(
    k_t: ArrayLike,
    investment: ArrayLike,
    *,
    depreciation: float | ArrayLike = DEFAULT_DEPRECIATION_RATE,
    retirement: float | ArrayLike = 0.0,
) -> np.ndarray:
    """Next-period capital stock ``K_{t+1} = (1 − δ)(1 − r)·K_t + INV_t`` (Phase 5d.3).

    All array arguments are elementwise-aligned and broadcast together, so the identity works at
    any granularity (scalar, per-region, per-region-sector) — see the module docstring.

    - ``k_t`` — opening capital stock (must be finite and ≥ 0).
    - ``investment`` — this period's gross investment (must be finite and ≥ 0; it is 5d.2's
      investment level, aggregated to the stock's granularity by the caller).
    - ``depreciation`` (δ) — fraction of the stock wearing out per period, in [0, 1].
    - ``retirement`` (r) — premature-retirement fraction of the *opening* stock, in [0, 1]
      (default 0). An exogenous stranded-asset write-off.

    Inputs are validated at the boundary (rejected, not silently clamped — mirroring the
    ``ElasticitySet`` validator), so a mis-specified scenario fails loudly rather than producing a
    negative or NaN stock. Returns a float array of the broadcast shape."""
    k = np.asarray(k_t, dtype=float)
    inv = np.asarray(investment, dtype=float)
    delta = np.asarray(depreciation, dtype=float)
    r = np.asarray(retirement, dtype=float)

    if not (np.all(np.isfinite(k)) and np.all(np.isfinite(inv))):
        raise ValueError("capital stock and investment must be finite")
    if np.any(k < 0):
        raise ValueError("capital stock K_t must be non-negative")
    if np.any(inv < 0):
        raise ValueError("investment must be non-negative (gross investment)")
    if not np.all(np.isfinite(delta)) or np.any(delta < 0) or np.any(delta > 1):
        raise ValueError(f"depreciation rate δ must be in [0, 1]; got {delta.tolist()}")
    if not np.all(np.isfinite(r)) or np.any(r < 0) or np.any(r > 1):
        raise ValueError(f"retirement fraction r must be in [0, 1]; got {r.tolist()}")

    # δ, r ∈ [0,1] and K, INV ≥ 0 ⇒ the surviving stock (1−δ)(1−r)·K ≥ 0 and INV ≥ 0, so the
    # result is non-negative by construction — the boundary validation above is what guarantees it
    # (a retirement fraction > 1 would otherwise drive it negative; rejected rather than clamped).
    return (1.0 - delta) * (1.0 - r) * k + inv


# Convention: the capital factor's account name in the CGE's ``factors`` list.
_CAPITAL_FACTOR = "CAP"

# Documented default NET rate of return on capital (the real return *net* of depreciation). With
# the depreciation rate δ it forms the user cost of capital u = net_return + δ, the price of one
# unit of capital SERVICES per period. 4%/yr is a standard applied central value for the real net
# return in CGE/growth calibrations [e.g. the 4% "world interest rate" convention]; overridable per
# scenario — the same "central default + documented override" discipline as the elasticities/δ.
DEFAULT_NET_RETURN = 0.04


def _validate_capital_rates(*, net_return: float, depreciation: float) -> None:
    """Shared finite-range validation for the capital-bridge rates (review P2 round 13). The
    depreciation δ must be a finite scalar in [0, 1] — the SAME range ``capital_next`` enforces, so
    a negative δ cannot pass ``benchmark_capital`` merely because ``net_return + δ`` stays positive.
    The net return must be a finite scalar ≥ 0."""
    d = float(depreciation)
    if not np.isfinite(d) or d < 0.0 or d > 1.0:
        raise ValueError(
            f"depreciation rate δ must be a finite value in [0, 1]; got {depreciation}"
        )
    nr = float(net_return)
    if not np.isfinite(nr) or nr < 0.0:
        raise ValueError(f"net_return must be a finite value ≥ 0; got {net_return}")


def benchmark_capital(
    cal,
    *,
    net_return: float = DEFAULT_NET_RETURN,
    depreciation: float = DEFAULT_DEPRECIATION_RATE,
) -> np.ndarray:
    """Region-level benchmark capital **stock** K_0 from any calibrated CGE model (Phase 5d.3).

    The clean entry point for Phase 7.1's recursive-dynamic wrapper: the initial stock the
    accumulation identity steps forward from.

    **Stock–flow bridge (review remediation 2026-07-27).** The ``CAP`` factor's benchmark income
    (= its endowment at unit benchmark prices) is a *flow*: the annual payment for capital
    **services**, NOT a capital **stock**. Feeding that flow into ``capital_next`` (which adds gross
    investment in stock units) is dimensionally unsupported and inflates the implied I/K ratio (a
    services flow is roughly an order of magnitude smaller than the stock that yields it). This
    converts the flow to a stock via the **user cost of capital** [Jorgenson1963]

        capital_income = u · K_0,   u = net_return + δ   (rental rate = net return + depreciation)
        ⇒  K_0 = capital_income / (net_return + δ)

    the textbook Jorgensonian user-cost identity. ``net_return`` (the real return net of
    depreciation, default 4%/yr [KingRebelo1999]) and ``depreciation`` (δ, default 5%/yr [OECD2009])
    are documented, overridable parameters; a build carrying an observed capital stock should pass
    it directly rather than infer it here (a documented future extension).

    **This is NOT a steady-state calibration** (review P2, 2026-07-27). The user-cost identity pins
    the stock from the *income* flow; it does NOT assume the *investment* flow is at steady-state
    replacement level. The calibrated benchmark investment ``INV0`` is whatever the SAM records, so
    the implied growth rate ``g = INV0/K0 − δ`` (see ``implied_growth_rate``) is generally NONZERO —
    typically negative here (investment below δ·K, so the stock would contract). Callers
    building a Phase-7.1 dynamic path should read that implied ``g`` and decide whether to accept a
    contracting benchmark or re-anchor the stock to a target growth rate — this function does not
    silently impose ``I/K = g + δ``.

    Returns a 1-D array indexed by region: length 1 for the closed/open single-region variants,
    length ``nr`` for multi-region. Region-level (not region-sector) — matching the single
    aggregate capital factor per region, per 5d.3's recommended granularity.

    Raises if the model has no ``CAP`` factor, or if the user cost ``net_return + δ`` is not
    strictly positive (the conversion would be undefined/negative)."""
    factors = list(cal.factors)
    if _CAPITAL_FACTOR not in factors:
        raise ValueError(
            f"model has no {_CAPITAL_FACTOR!r} factor; capital accumulation needs a capital "
            f"factor to track (factors are {factors})."
        )
    # Shared finite-range validation (review P2 round 13): δ ∈ [0,1] and net_return finite ≥ 0,
    # the SAME range capital_next enforces on δ — so a negative depreciation cannot slip through
    # benchmark_capital and into the implied-growth calculation just because the SUM is positive.
    _validate_capital_rates(net_return=net_return, depreciation=depreciation)
    user_cost = float(net_return) + float(depreciation)
    if not np.isfinite(user_cost) or user_cost <= 0:
        raise ValueError(
            f"user cost of capital (net_return + depreciation) must be > 0; got "
            f"{net_return} + {depreciation} = {user_cost}."
        )
    fi = factors.index(_CAPITAL_FACTOR)
    endowment = np.asarray(cal.endowment, dtype=float)
    # Closed/open: endowment is [f]; capital is a single aggregate — return a length-1 array so the
    # caller always gets a per-region vector. Multi-region: endowment is [f, r]; the capital row is
    # per-region capital income. Either way it is an income FLOW, converted to a STOCK below.
    capital_income = np.array([endowment[fi]]) if endowment.ndim == 1 else endowment[fi, :].copy()
    return capital_income / user_cost


def implied_growth_rate(
    cal,
    *,
    net_return: float = DEFAULT_NET_RETURN,
    depreciation: float = DEFAULT_DEPRECIATION_RATE,
) -> np.ndarray:
    """The benchmark's IMPLIED capital growth rate ``g = INV0/K0 − δ`` per region (review P2,
    2026-07-27). ``K0`` is the user-cost stock from ``benchmark_capital``; ``INV0`` is the model's
    ACTUAL calibrated benchmark investment (nominal, at unit prices), NOT a fabricated replacement
    flow. The perpetual-inventory step ``K_{t+1}=(1−δ)K_t+INV`` gives next-period growth
    ``K_{t+1}/K_t − 1 = INV/K − δ = g``; a negative ``g`` means the benchmark investment is below
    the replacement level ``δ·K`` and the stock would contract if stepped forward unchanged.

    Reported so a caller can SEE the benchmark's dynamic implication rather than the code silently
    assuming a steady state. Raises if the model has no savings-investment account (no ``INV0`` to
    compare) or no ``CAP`` factor."""
    if not getattr(cal, "has_investment", False):
        raise ValueError(
            "implied_growth_rate needs a savings-investment account (INV0) to compare against the "
            "derived stock; this model has none."
        )
    k0 = benchmark_capital(cal, net_return=net_return, depreciation=depreciation)
    inv0 = np.asarray(cal.INV0, dtype=float)
    # INV0 is [i] (closed/open) or [r, i] (multi); aggregate to the region level to match K0.
    inv_by_region = np.array([inv0.sum()]) if inv0.ndim == 1 else inv0.sum(axis=1)
    return inv_by_region / k0 - float(depreciation)
