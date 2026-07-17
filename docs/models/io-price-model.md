# Model description: Engine 1 — Leontief carbon-cost price model

- **Implements:** `cge.engines.io_price` (Phase 2; not yet built — this doc is its spec)
- **Roadmap phase:** 2
- **Capabilities:** prices
- **Status:** draft (specification ahead of implementation)

> This is the worked reference example for the [documentation standard](../documentation-standard.md):
> it shows the equation-level detail and citation discipline every model doc must meet.

## 1. Purpose & scope

Compute the change in the price of every good (in every region) caused by a carbon price,
accounting for **full supply-chain cost pass-through**. Answers: *"if a carbon price of
$\tau$ per tonne is imposed, how much more expensive does good $i$ become, once the cost
increase in all of its direct and indirect inputs has propagated through?"*

**In scope:** cost pass-through through the fixed IO technology; decomposition into direct
vs upstream contributions.

**Explicitly not modelled:** no substitution between inputs (technology is fixed), no
demand response, no change in production volumes, no factor-market effects. The result is
therefore an **upper bound on the cost impact** and says nothing about quantities. Volume
response is Engine 2 (Phase 4); relaxing fixed technology is the CGE (Phase 5). This
boundary is the single most important thing to state when presenting results.

## 2. Notation

| Symbol | Meaning | Units / shape |
|---|---|---|
| $n$ | number of sector×region entries (products) | scalar |
| $\mathbf{A}$ | technical-coefficient matrix; $A_{ij}$ = value of $i$ needed per unit value of $j$ | $n\times n$ |
| $\mathbf{p}$ | vector of product prices (index; baseline $\mathbf{p}=\mathbf{1}$) | $n\times 1$ |
| $\mathbf{v}$ | value added per unit output (primary-input cost share) | $n\times 1$ |
| $\mathbf{e}$ | direct emission intensity: tonnes CO₂e per unit output of each product | $n\times 1$ |
| $\tau$ | carbon price | currency / tonne CO₂e |
| $\mathbf{c}$ | direct carbon cost per unit output, $\mathbf{c}=\tau\,\mathbf{e}$ | $n\times 1$ |
| $\mathbf{I}$ | identity matrix | $n\times n$ |
| $\Delta\mathbf{p}$ | change in prices vs baseline | $n\times 1$ |

Emission intensities $\mathbf{e}$ come from the `SatelliteAccount`; $\mathbf{A}$ from the
`IOSystem`. Both are harmonised data objects (contract 1).

## 3. Assumptions

1. **Fixed technology.** $\mathbf{A}$ is constant — no input substitution in response to
   price changes. (Encoded by using the base-year $\mathbf{A}$ unchanged.)
2. **Full cost pass-through.** Every producer passes 100% of cost increases downstream;
   margins are constant. (No pass-through parameter < 1 in v1.)
3. **Cost-push price formation.** Prices are set by unit cost (Leontief price model), not
   by demand — consistent with the fixed-quantity assumption.
4. **Carbon cost enters as a per-unit cost on direct (scope-1) emissions** by default;
   an option adds emissions embodied in purchased energy explicitly.
5. **Linearity.** The price system is linear, so impacts of independent shocks add.

Assumptions 1–5 must be reproduced verbatim in the engine's `RunManifest.assumptions`.

## 4. Derivation

The **Leontief price model** (cost-push dual of the quantity model) states that the unit
price of each product equals the cost of its intermediate inputs plus its value added
[MillerBlair2009, §2.3–2.4]:

$$
\mathbf{p} = \mathbf{A}^{\!\top}\mathbf{p} + \mathbf{v}. \tag{1}
$$

Here $\mathbf{A}^{\!\top}\mathbf{p}$ is the intermediate-input cost per unit output (column
$j$ of $\mathbf{A}$ lists the inputs to product $j$, so the transpose contracts prices
against $j$'s input requirements), and $\mathbf{v}$ is primary-input cost (value added)
per unit output. Solving (1):

$$
\mathbf{p} = (\mathbf{I} - \mathbf{A}^{\!\top})^{-1}\,\mathbf{v}. \tag{2}
$$

At baseline, prices are normalised to unity, $\mathbf{p}_0 = \mathbf{1}$, which pins down
the implied baseline value added $\mathbf{v}_0 = (\mathbf{I}-\mathbf{A}^{\!\top})\mathbf{1}$.

**Introducing the carbon price.** A carbon price adds a direct per-unit cost
$\mathbf{c} = \tau\,\mathbf{e}$ to each product (a new primary-input cost). The shocked
price system is

$$
\mathbf{p}_1 = \mathbf{A}^{\!\top}\mathbf{p}_1 + \mathbf{v}_0 + \tau\,\mathbf{e}, \tag{3}
$$

with solution

$$
\mathbf{p}_1 = (\mathbf{I}-\mathbf{A}^{\!\top})^{-1}\left(\mathbf{v}_0 + \tau\,\mathbf{e}\right). \tag{4}
$$

Subtracting the baseline (2) and using linearity, the **price change** is driven entirely
by the carbon term:

$$
\boxed{\;\Delta\mathbf{p} = \mathbf{p}_1 - \mathbf{p}_0 = (\mathbf{I}-\mathbf{A}^{\!\top})^{-1}\,\tau\,\mathbf{e}\;} \tag{5}
$$

Equation (5) is the engine's core result: $\Delta p_i / p_{0,i}$ is the proportional cost
increase of good $i$ under carbon price $\tau$.

**Decomposition (direct vs upstream).** Expanding the Leontief inverse as its Neumann
series [MillerBlair2009, §2.5],

$$
(\mathbf{I}-\mathbf{A}^{\!\top})^{-1} = \mathbf{I} + \mathbf{A}^{\!\top} + (\mathbf{A}^{\!\top})^2 + \cdots, \tag{6}
$$

so $\Delta\mathbf{p} = \underbrace{\tau\mathbf{e}}_{\text{direct}} + \underbrace{\mathbf{A}^{\!\top}\tau\mathbf{e} + (\mathbf{A}^{\!\top})^2\tau\mathbf{e} + \cdots}_{\text{upstream, by tier}}$.
The first term is the good's own emissions; each subsequent term is one more tier up the
supply chain. Truncating after 2–3 terms gives the dominant supply-chain paths for the
result explorer (structural path analysis, [MillerBlair2009, Ch. 12]).

**Well-posedness.** $(\mathbf{I}-\mathbf{A}^{\!\top})^{-1}$ exists and is non-negative iff
the spectral radius $\rho(\mathbf{A}) < 1$. For a productive economy every column sum of
$\mathbf{A}$ (input cost share) is $< 1$, which guarantees $\rho(\mathbf{A})<1$ by the
Perron–Frobenius bound [MillerBlair2009, §2.6]. The toy fixture is constructed to satisfy
this; for EXIOBASE the engine asserts $\rho(\mathbf{A})<1$ as a precondition.

## 5. Algorithm

1. Assemble $\mathbf{A}$ (sector×region) and $\mathbf{e}$ from the data objects; align on
   the shared classification.
2. Form $\mathbf{c} = \tau\mathbf{e}$, zeroing entries outside the shock's coverage
   (sector/region filters from the `CarbonPrice` shock).
3. Solve $(\mathbf{I}-\mathbf{A}^{\!\top})\,\Delta\mathbf{p} = \mathbf{c}$ for
   $\Delta\mathbf{p}$ — a linear solve, **not** an explicit inverse (`scipy.linalg.solve`
   / sparse `spsolve`), which is more accurate and cheaper.
4. For decomposition, accumulate the first $k$ Neumann terms (matrix–vector products only).

**Complexity.** One $n\times n$ linear solve, $O(n^3)$ dense or far less sparse. On the
small build ($n\sim 450$–600) this is milliseconds; on the full MRIO ($n\sim 9800$) it is
seconds and runs sparse. This is a LAPACK/SciPy call, so the hot path is compiled (cf.
ADR-0003); no Python-level optimisation is needed here.

## 6. Calibration / parameters

- $\mathbf{A}$, $\mathbf{e}$: directly from EXIOBASE (Phase 1 data build); no free
  parameters.
- $\tau$, gas coverage, sector/region coverage, scope: from the `CarbonPrice` scenario.
- The only modelling choice is **scope** (scope-1 only vs including embodied energy);
  exposed as a scenario option and recorded in the manifest.

There are no fitted parameters — a strength of this engine, and why its *cost* answers are
the most defensible in the whole platform.

## 7. Validation

- **Analytic (toy economy):** on `validation.toy_economy()`, hand-compute $\Delta\mathbf{p}$
  from (5) for a known $\tau$ and assert equality to machine precision.
- **Zero shock:** $\tau=0 \Rightarrow \Delta\mathbf{p}=\mathbf{0}$.
- **Monotonicity/linearity:** doubling $\tau$ doubles $\Delta\mathbf{p}$.
- **Known-answer (real data):** reproduce published EXIOBASE carbon-footprint / CO₂
  multipliers for a few sectors within tolerance [Stadler2018].
- **Well-posedness guard:** assert $\rho(\mathbf{A})<1$ on the loaded build.

Tests: `tests/test_io_price.py` (Phase 2). Identity checks run in CI.

## 8. References

[MillerBlair2009] (Leontief price model §2.3–2.4, Neumann series §2.5, well-posedness
§2.6, SPA Ch. 12); [Leontief1970]; [Stadler2018] (EXIOBASE data & published multipliers).
Full entries in [`../references.md`](../references.md).
