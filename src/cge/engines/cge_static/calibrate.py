"""Calibration: a benchmark SAM → static-CGE parameters (roadmap Phase 5.2b).

Standard CGE calibration [Hosoe2010, ch. 4-6]: read the benchmark flows from the SAM and back out
the structural parameters (Leontief input-output coefficients, Cobb-Douglas value-added and
household share parameters, and scale constants) so that **the model reproduces the benchmark SAM
exactly at the benchmark prices (all prices = 1)**. Calibration is deterministic and unit-tested
against the hand-computed pilot parameters; it is the foundation of the replication test.

Model structure calibrated here (the pilot; see docs/models/cge-static.md):

- **Production** — Leontief in intermediates, Cobb-Douglas in value added (the KLEM-with-Leontief-M
  baseline). For sector ``i``:
    intermediate demand   INT[j,i] = ax[j,i] · X[i]           (Leontief)
    value added           VA[i]    = av[i] · Π_f F[f,i]^β[f,i] (Cobb-Douglas, Σ_f β[f,i] = 1)
    zero profit           px[i]    = Σ_j ax[j,i]·p[j] + pv[i], pv[i] the VA unit cost.
- **Household** — Cobb-Douglas over commodities: budget shares γ[i] from benchmark final demand.
- **Factors** — fixed endowments FF[f] = total benchmark factor income.

At the benchmark all prices are 1, so the coefficients read straight off the SAM money flows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cge.contracts.data_objects import SAM
from cge.engines.cge_static.calibrate_open import _elast_vector


@dataclass(frozen=True)
class CalibratedModel:
    """Benchmark parameters for the static CGE. Vectors/matrices are indexed by the ``sectors`` /
    ``factors`` order. All benchmark prices are 1 by construction."""

    sectors: list[str]
    factors: list[str]
    ax: np.ndarray  # [j, i] Leontief intermediate coefficients (input j per unit output i)
    va_share: np.ndarray  # [i] value added per unit output (Leontief VA requirement)
    beta: np.ndarray  # [f, i] Cobb-Douglas value-added share of factor f in sector i (cols sum→1)
    av: np.ndarray  # [i] value-added scale constant (CD or CES, so unit VA cost = 1 at benchmark)
    gamma: np.ndarray  # [i] household Cobb-Douglas budget share (sums→1)
    endowment: np.ndarray  # [f] fixed factor endowment (benchmark factor income)
    X0: np.ndarray  # [i] benchmark gross output
    F0: np.ndarray  # [f, i] benchmark factor demand
    Z0: np.ndarray  # [j, i] benchmark intermediate flows
    FD0: np.ndarray  # [i] benchmark household final demand
    # Value-added nest elasticity of substitution between factors. σ = 1 ⇒ Cobb-Douglas (uses
    # ``beta``); σ ≠ 1 ⇒ CES (uses ``va_ces_share``). Per sector.
    va_elast: np.ndarray  # [i] VA substitution elasticity σ_va
    va_ces_share: np.ndarray  # [f, i] CES factor share δ_{fi} (cols sum→1); used when σ ≠ 1

    @property
    def gdp0(self) -> float:
        """Benchmark GDP = total value added = total final demand."""
        return float(self.F0.sum())


def calibrate(
    sam: SAM,
    *,
    sectors: list[str],
    factors: list[str],
    va_elast: float | np.ndarray = 1.0,
) -> CalibratedModel:
    """Calibrate the pilot CGE from a balanced ``sam``. ``sectors`` and ``factors`` name the SAM
    accounts to treat as activities/commodities and factors; the remaining institution account is
    the household (final demand + factor income). Assumes benchmark prices = 1.

    ``va_elast`` is the value-added substitution elasticity σ_va (scalar or per-sector). σ = 1 is
    Cobb-Douglas (the default, preserving the pilot); σ ≠ 1 is CES, calibrated so the VA unit cost
    is still 1 at benchmark."""
    m = sam.matrix
    hoh = [a for a in sam.accounts if a not in sectors and a not in factors]
    if len(hoh) != 1:
        raise ValueError(f"pilot calibration expects exactly one institution account, got {hoh}")
    household = hoh[0]

    # Benchmark flows straight off the SAM (prices = 1 ⇒ money flow = quantity).
    Z0 = np.array([[m.loc[j, i] for i in sectors] for j in sectors], dtype=float)  # [j,i]
    F0 = np.array([[m.loc[f, i] for i in sectors] for f in factors], dtype=float)  # [f,i]
    FD0 = np.array([m.loc[i, household] for i in sectors], dtype=float)  # [i]
    # Normalise all levels by GDP so magnitudes are O(1): a CGE is homogeneous of degree zero, so
    # the *level* scale is arbitrary and results are relative changes. Real EXIOBASE flows are
    # ~1e9, where absolute solver residuals never reach a 1e-9 tolerance; unit-scaling fixes that
    # without changing any calibrated ratio or reported change (it cancels in every % result).
    scale = float(F0.sum())  # benchmark GDP (= total value added)
    if scale <= 0:
        raise ValueError("SAM has non-positive total value added; cannot calibrate")
    Z0, F0, FD0 = Z0 / scale, F0 / scale, FD0 / scale
    X0 = Z0.sum(axis=0) + F0.sum(axis=0)  # output = Σ intermediate cost + Σ value added

    # Reject degenerate benchmarks before dividing by them (review robustness): a zero-output
    # sector, a sector with no value added (β/av undefined), or non-positive final demand would
    # otherwise produce NaN/inf shares.
    if float(X0.min()) <= 0:
        bad = [s for s, x in zip(sectors, X0, strict=True) if x <= 0]
        raise ValueError(f"SAM has zero/negative gross output for sectors {bad}; cannot calibrate")
    VA0_check = F0.sum(axis=0)
    if float(VA0_check.min()) <= 0:
        bad = [s for s, v in zip(sectors, VA0_check, strict=True) if v <= 0]
        raise ValueError(f"SAM has zero value added for sectors {bad}; Cobb-Douglas VA undefined")
    if float(FD0.min()) < 0 or float(FD0.sum()) <= 0:
        raise ValueError("SAM household final demand must be non-negative with a positive total")

    # Leontief intermediate coefficients: input j per unit output i.
    ax = Z0 / X0[None, :]
    # Value added per unit output (Leontief VA requirement): VA0[i]/X0[i]. With unit VA cost 1 at
    # benchmark, the zero-profit price Σ_j ax[j,i] + va_share[i] = 1 exactly (checked in tests).
    VA0 = F0.sum(axis=0)  # [i]
    va_share = VA0 / X0
    # Cobb-Douglas factor shares within value added (columns sum to 1).
    beta = F0 / VA0[None, :]  # [f,i]
    ns = len(sectors)
    # Validate σ_va: scalar or exactly (ns,), finite, strictly positive — a length-1 vector used to
    # raise a raw IndexError and a non-positive value was silently used (review P2).
    sigma_va = _elast_vector(va_elast, ns, "va_elast")
    # CES factor share δ (used when σ ≠ 1): from the benchmark factor mix, δ_f/δ_g = (F_f/F_g)^{1/σ}
    # at unit factor prices, normalised to sum to 1 per sector. (For σ = 1 δ = β, harmless.)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(F0 > 0, np.power(F0, 1.0 / sigma_va[None, :]), 0.0)
        va_ces_share = weights / weights.sum(axis=0)[None, :]
    # Scale ``av`` so the VA unit cost is exactly 1 at benchmark (w = 1). CD: av = Π (1/β)^β; CES:
    # av = [Σ_f δ_f^σ]^{1/(1-σ)}. Chosen per sector by whether σ = 1.
    av = np.empty(ns)
    for i in range(ns):
        if abs(sigma_va[i] - 1.0) < 1e-12:
            b = beta[:, i]
            av[i] = np.prod(np.power(np.where(b > 0, 1.0 / np.where(b > 0, b, 1.0), 1.0), b))
        else:
            s = sigma_va[i]
            d = va_ces_share[:, i]
            av[i] = np.power(np.sum(d**s), 1.0 / (1.0 - s))
    # Household Cobb-Douglas budget shares (of total final demand).
    gamma = FD0 / FD0.sum()
    # Fixed factor endowments = total benchmark factor income.
    endowment = F0.sum(axis=1)  # [f]

    return CalibratedModel(
        sectors=list(sectors),
        factors=list(factors),
        ax=ax,
        va_share=va_share,
        beta=beta,
        av=av,
        gamma=gamma,
        endowment=endowment,
        X0=X0,
        F0=F0,
        Z0=Z0,
        FD0=FD0,
        va_elast=sigma_va,
        va_ces_share=va_ces_share,
    )
