"""Tests for the CGE solver abstraction (roadmap Phase 5.0).

Proves the solver is wired and available (scipy fallback), returns known optima, records its
backend/status, and — the non-negotiable rule — RAISES on non-convergence rather than returning
non-equilibrium numbers. IPOPT is exercised only when its binary is present (gated).
"""

import numpy as np
import pytest

from cge.engines.cge_static.solver import (
    Solution,
    SolveError,
    ipopt_available,
    solve,
)


def test_solves_linear_system_known_answer():
    """A trivial 2x2 linear system has a unique known root; the fallback must find it."""

    # x + y = 3, x - y = 1  → x = 2, y = 1
    def residual(z):
        x, y = z[0], z[1]
        return np.array([x + y - 3.0, x - y - 1.0])

    sol = solve(residual, np.array([1.0, 1.0]), prefer="scipy")
    assert isinstance(sol, Solution)
    assert np.allclose(sol.x, [2.0, 1.0], atol=1e-8)
    assert sol.residual_norm < 1e-9
    assert sol.backend == "scipy"


def test_solves_nonlinear_system_known_answer():
    """A nonlinear system with a known positive root (a mini 'market clearing')."""

    # x^2 = 4, x*y = 6  → x = 2, y = 3 (positive branch)
    def residual(z):
        x, y = z[0], z[1]
        return np.array([x * x - 4.0, x * y - 6.0])

    sol = solve(residual, np.array([1.0, 1.0]), prefer="scipy")
    assert np.allclose(sol.x, [2.0, 3.0], atol=1e-7)


def test_positivity_is_respected():
    """The log-space solve keeps variables strictly above their lower bound."""

    def residual(z):
        return np.array([z[0] - 5.0, z[1] - 0.001])

    sol = solve(residual, np.array([1.0, 1.0]), lower=np.array([1e-9, 1e-9]), prefer="scipy")
    assert (sol.x > 0).all()
    assert np.allclose(sol.x, [5.0, 0.001], atol=1e-7)


def test_non_convergence_raises():
    """An inconsistent system (no root) must raise SolveError — never return numbers."""

    # x = 1 and x = 2 simultaneously: no solution.
    def residual(z):
        return np.array([z[0] - 1.0, z[0] - 2.0])

    with pytest.raises(SolveError, match="did not converge"):
        solve(residual, np.array([1.0]), prefer="scipy")


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="unknown solver backend"):
        solve(lambda z: z, np.array([1.0]), prefer="nope")


def test_ipopt_availability_is_boolean():
    """The availability probe never raises; it returns a bool used to gate solver checks."""
    assert isinstance(ipopt_available(), bool)


@pytest.mark.skipif(not ipopt_available(), reason="IPOPT binary not installed")
def test_ipopt_solves_known_answer():
    """When IPOPT is present, it solves the same known system (proves the real solver path)."""

    def residual(z):
        return np.array([z[0] + z[1] - 3.0, z[0] - z[1] - 1.0])

    sol = solve(residual, np.array([1.0, 1.0]), prefer="ipopt")
    assert np.allclose(sol.x, [2.0, 1.0], atol=1e-6)
    assert sol.backend == "ipopt"
