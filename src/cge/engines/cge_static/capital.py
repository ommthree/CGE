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
    converts the flow to a stock via the **user cost of capital**

        capital_income = u · K_0,   u = net_return + δ   (rental rate = net return + depreciation)
        ⇒  K_0 = capital_income / (net_return + δ)

    the textbook Jorgensonian user-cost identity, equivalent at a steady state to the
    ``I/K = g + δ`` relation. ``net_return`` (the real return net of depreciation) and
    ``depreciation`` (δ) are documented parameters (defaults ``DEFAULT_NET_RETURN`` /
    ``DEFAULT_DEPRECIATION_RATE``), overridable per scenario; a build carrying an observed capital
    stock should pass it directly rather than infer it here (a documented future extension).

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
