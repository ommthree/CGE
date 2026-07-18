# Model description: Engine 1 — Leontief carbon-cost price model

- **Implements:** `cge.engines.io_price` (`IOPriceEngine`, v0.4.0)
- **Roadmap phase:** 2
- **Capabilities:** prices
- **Status:** implemented and validated — on the toy economy (internal identities +
  hand-derived known answer) **and against live EXIOBASE 3 (2019 pxp)**: the adapter
  preserves EXIOBASE's global CO₂ total (30.0 Gt), units come through correctly, and a
  €100/t run gives fractional price changes with energy sectors most exposed (see §7). Engine is
  dense-only / small-build (sparse full-MRIO not implemented). Remaining refinement is a
  tighter published-footprint comparison and a curated sector concordance.

> This is the worked reference example for the [documentation standard](../documentation-standard.md):
> it shows the equation-level detail and citation discipline every model doc must meet.

> **Units (post-review).** EXIOBASE emission flows `F` are **mass totals (kg)**; dividing by
> monetary output (M€) gives an intensity in **kg per M€**. The adapter
> converts to **tonnes per M€** using the source unit metadata. A carbon price in €/tonne is
> then scaled by 1e‑6 (M€→€) so that τ·e is a **dimensionless cost share**. Δp is therefore a
> **fractional change in the unit price index** (baseline p₀=1) — e.g. 0.06 = a 6% price rise.
> The engine asserts the exact units (t/MEUR, tCO2e/MEUR) and EUR/MEUR base before scaling.

## 1. Purpose & scope

Compute the change in the price of every good (in every region) caused by a carbon price,
accounting for **full supply-chain cost pass-through**. Answers: *"if a carbon price of
$\tau$ per tonne is imposed, how much more expensive does good $i$ become, once the cost
increase in all of its direct and indirect inputs has propagated through?"*

**In scope:** cost pass-through through the fixed IO technology; decomposition into direct
vs upstream contributions.

**Explicitly not modelled:** no substitution between inputs (technology is fixed), no
demand response, no change in production volumes, no factor-market effects. Because
substitution would let firms avoid part of the cost, this fixed-technology / full-pass-through
result is expected to **over-state the cost impact relative to a model with substitution** —
but it is *not* a proven upper bound over every possible model (supply constraints, market
power, or factor-market effects can push the true impact either way). It says nothing about
quantities. Volume response is Engine 2 (Phase 4); relaxing fixed technology is the CGE
(Phase 5). This boundary is the single most important thing to state when presenting results.

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
4. **Carbon cost enters as a per-unit cost on the scenario's selected GHG account** (the
   producer's own scope-1 emissions, priced by `gases`). A scope-2 (embodied-energy) option
   is **not implemented** — see the note below.
5. **Linearity.** The price system is linear, so impacts of independent shocks add.

> **On scope.** Equation (5) already propagates each supplier's own carbon cost through the
> Leontief inverse, so a downstream good already bears its suppliers' emission costs. Adding
> the buyer's *purchased-energy* emissions directly to its own cost vector would double-count
> unless framed as a distinct scope-2 policy liability. Because that distinction is a
> modelling decision not yet made, **no scope option exists in v0.4.0**; the engine prices
> scope-1 emissions of the selected gases only.

These assumptions are the single source of truth; the engine's `ASSUMPTIONS` dict (emitted
into every `RunManifest.assumptions`) restates them for machine consumption and must stay
consistent with this list — it is a paraphrase for the manifest, not a byte-for-byte copy.

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
The first term is the good's own emissions; each subsequent term is the *aggregate*
contribution of one more tier up the supply chain. The implementation reports these **tier
aggregates plus a residual**, which sum exactly to $\Delta\mathbf{p}$. These are *not*
enumerated individual supply-chain paths — full **structural path analysis** [MillerBlair2009,
Ch. 12] would enumerate paths and is a separate, heavier method not implemented here.

**Well-posedness.** Assume $\mathbf{A} \ge 0$ (non-negative technical coefficients). Then
$(\mathbf{I}-\mathbf{A}^{\!\top})^{-1}$ exists and **is non-negative iff** the spectral radius
$\rho(\mathbf{A}) < 1$ [MillerBlair2009, §2.6]. The $\mathbf{A}\ge 0$ condition is essential:
with negative entries the inverse can exist yet fail to be non-negative, so the
"pass-through only adds cost" guarantee breaks (a positive tax could lower another good's
price). The engine therefore asserts **both** $\mathbf{A}\ge 0$ (within a small tolerance for
rounding) **and** $\rho(\mathbf{A})<1$ as preconditions. All column sums of $\mathbf{A}$ being
$<1$ is a **sufficient** condition for $\rho(\mathbf{A})<1$ (by the Perron–Frobenius bound on
the column-sum norm), not a necessary one — a productive economy can have some column sums
$\ge 1$ while still satisfying $\rho(\mathbf{A})<1$, so the engine checks $\rho$ directly.

> **Implementation note.** `price_change` computes $\rho(\mathbf{A})$ explicitly and raises
> if $\ge 1$, rather than relying on the linear solve to fail. This matters: `np.linalg.solve`
> succeeds numerically for many non-productive matrices (it only errors on exact
> singularity), so a solve-only guard would silently return meaningless prices. The
> validation suite caught exactly this gap during Phase 2.

## 5. Algorithm

1. Assemble $\mathbf{A}$ (sector×region) and $\mathbf{e}$ from the data objects; align on
   the shared classification.
2. Form $\mathbf{c} = \tau\mathbf{e}$, zeroing entries outside the shock's coverage
   (sector/region filters from the `CarbonPrice` shock).
3. Convert the carbon price to a dimensionless cost share $\mathbf{c} = \tau\,\mathbf{e}\,
   \times 10^{-6}$ (M€→€; see the units note above), zeroing entries outside the shock's
   coverage. Multiple carbon shocks add; each reads its own time path per year.
4. Solve $(\mathbf{I}-\mathbf{A}^{\!\top})\,\Delta\mathbf{p} = \mathbf{c}$ for
   $\Delta\mathbf{p}$ — a linear solve (`np.linalg.solve`), **not** an explicit inverse.
5. For decomposition, accumulate the first $k$ Neumann terms (matrix–vector products only)
   and take the residual against the full solve so the parts sum exactly to $\Delta\mathbf{p}$.

**Complexity & scope of the current implementation.** The implementation is **dense**
(`np.linalg.solve` + a dense eigenvalue check for the admissibility guard), which is exact
and fast on the small build ($n\sim 450$–600: milliseconds). It is **not** yet suitable for
the full ~9800² MRIO: a dense float64 matrix is ~768 MB before LAPACK workspace, and the
solve/eigvals are $O(n^3)$. A sparse path (`scipy.sparse.linalg`, iterative or `spsolve`, and
a sparse spectral-radius estimate) is the intended drop-in but is **not implemented**. Until
it is, run the engine on the small build only. The engine seam (ADR-0002) makes a sparse
reimplementation a local change.

## 6. Calibration / parameters

- $\mathbf{A}$, $\mathbf{e}$: directly from EXIOBASE (Phase 1 data build); no free
  parameters.
- $\tau$ (price and optional time path), `gases`, and sector/region coverage: from the
  `CarbonPrice` scenario. There is **no scope option** in v0.4.0 (see the scope note in §3);
  the engine prices scope-1 emissions of the selected gases.

There are no fitted parameters — a strength of this engine, and why its *cost* answers are
the most defensible in the whole platform.

## 7. Validation

Implemented as the `io_price` **validation suite** (`cge.validation.suites.io_price`), run
by `scripts/validate.py` / `cge validate` and gated in CI via `tests/test_validation.py`.
Code-level unit tests live in `tests/test_io_price.py`. Current checks:

| Check | Property (equation) |
|---|---|
| `analytic_matches_explicit_inverse` | linear solve equals $(\mathbf{I}-\mathbf{A}^{\!\top})^{-1}\tau\mathbf{e}$ to machine precision (5) |
| `zero_shock_zero_change` | $\tau=0 \Rightarrow \Delta\mathbf{p}=\mathbf{0}$ |
| `linearity_in_price` | doubling $\tau$ doubles $\Delta\mathbf{p}$ (assumption 5) |
| `pass_through_adds_cost` | $\Delta\mathbf{p} \ge$ direct cost everywhere ($\mathbf{A},\mathbf{e}\ge 0$) |
| `decomposition_sums_to_total` | direct + upstream tiers + residual $= \Delta\mathbf{p}$ (6) |
| `energy_most_exposed` | emissions-intensive sector has the largest impact (plausibility) |
| `coverage_filtering` | region/sector-restricted shock zeroes direct cost elsewhere |
| `well_posedness_guard` | non-productive economy ($\rho(\mathbf{A})\ge 1$) is rejected |
| `known_answer_full_pipeline` | full run on the toy at €100/t matches a **hand-derived** vector (checks units + orientation, not just the solve) |
| `units_plausible_magnitude` | €100/t on the toy gives $0<\Delta p<1$ (fractional), not the ~$10^3$–$10^9$ a missing unit conversion would give |
| `gas_selection_distinct` | `gases=[CO2]` ≠ `gases=[CH4]`; combined is GWP-additive |
| `time_path_varies_by_year` | a price path produces year-varying results |
| `engine_end_to_end` | runner → registered engine → schema-valid `ResultSet` + assumptions |

Additional adversarial coverage in `tests/test_io_price.py`: negative-coefficient rejection,
missing-satellite-label rejection, revenue-recycling rejection, negative-price rejection.

**Live EXIOBASE known-answer (the P2.4/P1 DoD gate — now met).** `tests/test_exiobase_known_answer.py`
runs against a real EXIOBASE archive (opt-in via `CGE_EXIOBASE_ARCHIVE`; skipped in offline
CI). Validated on **EXIOBASE 3, 2019, pxp** (2026-07):

- the adapter reproduces the full 9800×9800 MRIO and detects the real `satellite` extension;
- global CO₂ from the adapter (intensity × output) equals pymrio's raw satellite total to
  `rtol 1e-6` — proving the kg→tonne conversion and `e = F/x` construction on real data;
- that total is **30.0 Gt**, the expected order of magnitude for global production-accounting
  fossil CO₂ (global fossil CO₂ was ~35 Gt in 2019 per the Global Carbon Project);
- a €100/t run on a coarse EUR build gives **fractional** price changes (not the ~$10^9$ a
  units bug would give), with **coal among the most carbon-exposed sectors** across regions —
  the qualitative known answer.

**Honest scope of this gate.** These are strong live *integration and sanity* checks — the
adapter is compared to the same archive's raw accounts (reusing the unit helper), the total is
checked against a broad plausibility band, and the engine's qualitative behaviour is verified.
They are **not** an independent numerical comparison against a *published* EXIOBASE
footprint/multiplier table, which is the roadmap's ultimate P2.4 requirement. ([Stadler2018]
documents EXIOBASE through 2011, so it cannot substantiate a 2019 numerical benchmark.) That
independent published comparison — plus a curated sector concordance (the current default is a
functional keyword grouping, not analytically precise) — remains the outstanding refinement.

## 8. References

[MillerBlair2009] (Leontief price model §2.3–2.4, Neumann series §2.5, well-posedness
§2.6, SPA Ch. 12); [Leontief1970]; [Stadler2018] (EXIOBASE data & published multipliers).
Full entries in [`../references.md`](../references.md).
