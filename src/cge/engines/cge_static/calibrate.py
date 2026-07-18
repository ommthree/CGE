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


@dataclass(frozen=True)
class CalibratedModel:
    """Benchmark parameters for the static CGE. Vectors/matrices are indexed by the ``sectors`` /
    ``factors`` order. All benchmark prices are 1 by construction."""

    sectors: list[str]
    factors: list[str]
    ax: np.ndarray  # [j, i] Leontief intermediate coefficients (input j per unit output i)
    va_share: np.ndarray  # [i] value added per unit output (Leontief VA requirement)
    beta: np.ndarray  # [f, i] Cobb-Douglas value-added share of factor f in sector i (cols sum→1)
    av: np.ndarray  # [i] value-added scale constant (Cobb-Douglas efficiency)
    gamma: np.ndarray  # [i] household Cobb-Douglas budget share (sums→1)
    endowment: np.ndarray  # [f] fixed factor endowment (benchmark factor income)
    X0: np.ndarray  # [i] benchmark gross output
    F0: np.ndarray  # [f, i] benchmark factor demand
    Z0: np.ndarray  # [j, i] benchmark intermediate flows
    FD0: np.ndarray  # [i] benchmark household final demand

    @property
    def gdp0(self) -> float:
        """Benchmark GDP = total value added = total final demand."""
        return float(self.F0.sum())


def calibrate(sam: SAM, *, sectors: list[str], factors: list[str]) -> CalibratedModel:
    """Calibrate the pilot CGE from a balanced ``sam``. ``sectors`` and ``factors`` name the SAM
    accounts to treat as activities/commodities and factors; the remaining institution account is
    the household (final demand + factor income). Assumes benchmark prices = 1."""
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

    # Leontief intermediate coefficients: input j per unit output i.
    ax = Z0 / X0[None, :]
    # Value added per unit output (Leontief VA requirement): VA0[i]/X0[i]. With unit VA cost 1 at
    # benchmark, the zero-profit price Σ_j ax[j,i] + va_share[i] = 1 exactly (checked in tests).
    VA0 = F0.sum(axis=0)  # [i]
    va_share = VA0 / X0
    # Cobb-Douglas factor shares within value added (columns sum to 1).
    beta = F0 / VA0[None, :]  # [f,i]
    # Cobb-Douglas scale so the VA unit cost is exactly 1 at benchmark prices (w = 1):
    # pv[i] = (1/av[i])·Π_f (1/β[f,i])^{β[f,i]} = 1  ⇒  av[i] = Π_f (1/β[f,i])^{β[f,i]}.
    av = np.prod(
        np.power(np.where(beta > 0, 1.0 / np.where(beta > 0, beta, 1.0), 1.0), beta), axis=0
    )
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
    )
