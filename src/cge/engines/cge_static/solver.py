"""Solver abstraction for the static CGE (roadmap Phase 5.0).

A CGE equilibrium is a **square system of nonlinear equations** F(z) = 0 (equations = unknowns
after imposing the numéraire and dropping one market-clearing equation by Walras' law), with the
variables bounded to the economically meaningful orthant (prices, quantities > 0).

This module solves such a system, **trying a real NLP solver (IPOPT via pyomo) when its binary is
available and falling back to scipy** otherwise. The fallback keeps the whole engine — and its CI
tests — runnable anywhere without a solver binary, exactly as the roadmap's 5.0 gate requires.

The single non-negotiable rule (the Engine-1 well-posedness lesson): a non-converged solve
**raises**; it never returns numbers. The chosen backend and its termination status are returned
on the ``Solution`` so the engine can record them in the run manifest.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


class SolveError(RuntimeError):
    """Raised when no backend converged the system. Carries the backend tried and the residual
    norm so the failure is diagnosable (never silently returns numbers)."""


@dataclass(frozen=True)
class Solution:
    """The result of a converged solve. ``x`` is the equilibrium variable vector; ``backend`` and
    ``status`` are recorded in the run manifest so a result states exactly how it was solved."""

    x: np.ndarray
    backend: str  # 'ipopt' or 'scipy'
    status: str  # solver-reported termination (e.g. 'optimal', 'converged')
    residual_norm: float  # ‖F(x)‖∞ at the solution — the honesty number


# Default convergence tolerance on the residual infinity-norm. Tight enough that a CGE benchmark
# replicates to well within the validation tolerances, loose enough for scipy on the small model.
DEFAULT_TOL = 1e-9


def ipopt_available() -> bool:
    """True iff a usable IPOPT binary is wired into pyomo on this machine. Solver-dependent
    validation checks gate on this (the same pattern as the live-EXIOBASE suite)."""
    try:
        import pyomo.environ as pe

        return bool(pe.SolverFactory("ipopt").available(exception_flag=False))
    except Exception:
        return False


def solve(
    residual: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    lower: np.ndarray | None = None,
    tol: float = DEFAULT_TOL,
    prefer: str | None = None,
) -> Solution:
    """Solve the square system ``residual(x) == 0`` from the initial guess ``x0``.

    ``lower`` is an optional per-variable lower bound (defaults to a small positive floor, since
    CGE variables are prices/quantities that must stay positive). ``prefer`` forces a backend
    ('ipopt' or 'scipy') — used by tests to exercise the fallback; by default IPOPT is used when
    available, else scipy.

    Raises ``SolveError`` if the chosen backend does not converge to ‖F(x)‖∞ < ``tol``.
    """
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    if lower is None:
        lower = np.full(n, 1e-9)  # positivity floor for prices/quantities
    lower = np.asarray(lower, dtype=float)

    backend = prefer or ("ipopt" if ipopt_available() else "scipy")
    if backend == "ipopt":
        sol = _solve_ipopt(residual, x0, lower, tol)
    elif backend == "scipy":
        sol = _solve_scipy(residual, x0, lower, tol)
    else:
        raise ValueError(f"unknown solver backend {backend!r}; use 'ipopt' or 'scipy'")

    # Final honesty gate: whatever the backend *claimed*, verify the residual ourselves.
    resid = float(np.max(np.abs(residual(sol.x))))
    if not np.isfinite(resid) or resid >= tol:
        raise SolveError(
            f"{backend} did not converge: ‖F(x)‖∞ = {resid:.3e} ≥ tol {tol:.1e} "
            f"(status {sol.status!r}). Refusing to return non-equilibrium numbers."
        )
    return Solution(x=sol.x, backend=backend, status=sol.status, residual_norm=resid)


def _solve_scipy(
    residual: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, lower: np.ndarray, tol: float
) -> Solution:
    """Root-find with scipy. Solves the system in log-space for the positivity-bounded variables
    so the iterate cannot cross zero (a common CGE failure mode), via a least-squares solve that
    is robust for a square system. ``z = log(x − lower)`` keeps x > lower for any real z."""
    from scipy.optimize import least_squares

    def _f(z: np.ndarray) -> np.ndarray:
        x = lower + np.exp(z)
        return residual(x)

    z0 = np.log(np.maximum(x0 - lower, 1e-12))
    res = least_squares(_f, z0, method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=10000)
    x = lower + np.exp(res.x)
    status = "converged" if res.success else f"scipy:status={res.status}"
    # Record the actual residual norm here (not NaN); ``solve`` re-verifies it against ``tol``.
    resid_norm = float(np.max(np.abs(residual(x)))) if x.size else 0.0
    return Solution(x=x, backend="scipy", status=status, residual_norm=resid_norm)


def _solve_ipopt(
    residual: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, lower: np.ndarray, tol: float
) -> Solution:
    """Solve the square system as an IPOPT feasibility problem: minimise 0 subject to
    F(x) = 0, x ≥ lower. pyomo builds the model; IPOPT solves it. Used only when the binary is
    available (checked by the caller); otherwise the scipy path runs."""
    import pyomo.environ as pe

    n = x0.size
    m = pe.ConcreteModel()
    m.I = pe.RangeSet(0, n - 1)
    m.x = pe.Var(m.I, initialize={i: float(x0[i]) for i in range(n)})
    for i in range(n):
        m.x[i].setlb(float(lower[i]))

    # Constraints F_k(x) = 0. residual is evaluated on a pyomo-var vector; the CGE residual is
    # built from pyomo-compatible algebra (it must avoid numpy-only ops on the var vector).
    def _con(model, k):
        vec = np.array([model.x[i] for i in range(n)], dtype=object)
        return residual(vec)[k] == 0

    m.F = pe.Constraint(m.I, rule=_con)
    m.obj = pe.Objective(expr=0.0)  # feasibility only

    result = pe.SolverFactory("ipopt").solve(m, tee=False)
    tc = str(result.solver.termination_condition)
    x = np.array([pe.value(m.x[i]) for i in range(n)], dtype=float)
    return Solution(x=x, backend="ipopt", status=tc, residual_norm=float("nan"))
