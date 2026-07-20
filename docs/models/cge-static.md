# Model description: Engine 3 — static CGE (pilot)

- **Implements:** `cge.engines.cge_static` (`CGEStaticEngine`, v0.5.2)
- **Roadmap phase:** 5 (pilot: 5.0 solver + 5.1 SAM build + 5.2a model + 5.3 revenue recycling;
  open economy Armington/CET + CES value added + elasticity sweeps)
- **Capabilities:** general_equilibrium, prices, volumes
- **Status:** implemented as the **correctness-first pilot** with **revenue recycling**, in three
  variants sharing one engine: a **closed** single-region economy, an **open** economy (Armington
  imports + CET exports + a rest-of-world account, CES value added, an endogenous exchange rate),
  and a **multi-region** economy with true bilateral trade among the build's own regions (§8a),
  selected automatically from the SAM's account structure — a `ROW` account selects open (§8);
  several region-tagged households and region-prefixed activities select multi-region. The closed
  model calibrates to a hand-checkable 2-sector SAM *and* to a SAM built from an aggregated
  **EXIOBASE-shaped** build (§5a); the open model calibrates to a hand-checkable open SAM *and* to
  one built from an EXIOBASE-shaped IOSystem; the multi-region model calibrates to a hand-checkable
  multi-region SAM (an IOSystem-driven multi-region build is a remaining sub-phase — §8a). All pass
  the standard CGE correctness battery (benchmark replication, homogeneity, Walras) plus the
  recycling-effect checks, with theory-consistent carbon-price responses; the open and multi-region
  models additionally exhibit **carbon leakage** (`cge_static` validation suite), and the
  multi-region model clears every bilateral trade route and factor market under shock, not just at
  the benchmark.
  **Honest scope note:** the automated build/validation uses the **offline pymrio *test* MRIO**
  (an EXIOBASE-shaped fixture), *not* live EXIOBASE — the live-data suite currently exercises only
  the adapter and Engine 1. A live-EXIOBASE CGE gate is a follow-up. The open CGE also builds its
  SAM from an EXIOBASE-shaped **IOSystem** (`build_open_sam`: a home region + rest-of-world, with
  Armington import / CET export / ROW accounts derived from the MRIO's inter-regional blocks) and
  replicates that built benchmark to machine precision — the `open_replicates_on_built_sam` gate.
  A **live-EXIOBASE** SAM build for all three variants, an **IOSystem-driven multi-region SAM
  build**, and **per-cell (rather than uniform) trade elasticities** are the remaining sub-phases;
  magnitudes are illustrative.

## 1. Purpose & scope

Compute the **general-equilibrium** response to a carbon price: prices, output volumes, and factor
prices when firms substitute inputs and factor markets clear — the feedback Engines 1–2
structurally cannot capture. The carbon price raises the cost of emitting sectors; the economy
re-allocates; the model reports the new equilibrium relative to the benchmark.

**In scope:** N sectors (activity/commodity), Leontief intermediates, a **CES value-added nest**
(capital + labour; elasticity σ_va, default 1 = Cobb-Douglas), Cobb-Douglas household demand, fixed
factor endowments, CPI numéraire, a carbon price as a per-unit emissions cost wedge, and
**carbon-revenue recycling** (lump-sum / labour-tax-cut). An **open-economy variant** (§8) adds
Armington imports and CET exports with a rest-of-world account — chosen automatically when the SAM
carries a `ROW` account.

**Not yet modelled:** an IOSystem-driven multi-region SAM build (§8a is supplied-SAM only today);
a **government/fiscal account** (carbon-tax revenue is collected and recycled in the same period —
there is no balance sheet, so it cannot run a deficit/surplus or fund non-recycled spending); a
**genuine energy nest** (KL–E–M — energy is a plain Leontief/CES intermediate today, not a
separable nest a carbon price can shift substitution within); **savings/investment** (no
capital-accumulation mechanism, so Phase 7's recursive-dynamic wrapper has nothing real to update
between years); heterogeneous households; and a distortionary labour-tax wedge (so the
"double-dividend" channel that would distinguish labour-tax-cut from lump-sum recycling in a
*single*-household model). Roadmap Phase 5.2 originally specified the government account,
investment, and energy nest — they were dropped rather than carried forward, and are now tracked as
**Phase 5d** (`roadmap.md` §Phase 5).

## 2. Notation

| Symbol | Meaning |
|---|---|
| $i, j$ | sectors / commodities |
| $f \in \{K, L\}$ | factors (capital, labour) |
| $p_i$ | commodity price (unknown) |
| $w_f$ | factor price (unknown) |
| $ax_{ji}$ | Leontief intermediate coefficient (input $j$ per unit output $i$) |
| $va_i$ | value added per unit output of $i$ |
| $\beta_{fi}$ | Cobb-Douglas share of factor $f$ in sector $i$'s value added ($\sum_f \beta_{fi}=1$) |
| $av_i$ | value-added scale constant (calibrated so unit VA cost = 1 at benchmark) |
| $\gamma_i$ | household Cobb-Douglas budget share ($\sum_i \gamma_i = 1$) |
| $FF_f$ | fixed factor endowment |
| $\tau$ | carbon price; $e_i$ emission intensity (tCO₂e per unit output) |
| $X_i$ | gross output; $FD_i$ household final demand; $F_{fi}$ factor demand |

## 3. Assumptions

1. **Leontief intermediates, CES value added** (KLEM with Leontief M) — the standard structure
   [Hosoe2010, ch. 4-6]. Substitution happens within the value-added nest, between capital and
   labour, with elasticity σ_va (`va_elast`; default 1 = Cobb-Douglas). A lower σ_va means factors
   are harder to substitute, so a carbon price that shifts the relative factor price produces a
   larger factor-price swing (validated). Intermediates are fixed-coefficient.
2. **Cobb-Douglas household demand** — constant budget shares $\gamma_i$.
3. **Fixed factor endowments** — labour and capital supply are inelastic; factor prices adjust to
   clear their markets.
4. **CPI numéraire** — the consumer price index is fixed to 1, pinning the price level (the model
   is homogeneous of degree zero in nominal prices; only relative prices are determined).
5. **Carbon price = a cost wedge** — a dimensionless per-sector carbon cost share $cc_i$ is added
   to sector $i$'s unit cost in the zero-profit condition. On a real build $cc_i$ is built the
   **same way as Engine 1** — via `carbon_cost_vector` (honouring the selected gases, coverage, and
   the $10^{-6}$ M-currency→currency scaling that makes $\tau e_i$ dimensionless) — then aggregated
   to sectors output-weighted. A currency-flexible unit gate requires the build to be
   millions-denominated with matching satellite units. The revenue is recycled (§4a).

## 4. The model (equation level)

Unknowns are the price vectors $(p, w)$; every quantity is a closed-form function of prices, so the
equilibrium is a small square system.

**Value-added unit cost** (Cobb-Douglas cost function, the default $\sigma_{va}=1$):
$$ pv_i = \frac{1}{av_i}\prod_f \left(\frac{w_f}{\beta_{fi}}\right)^{\beta_{fi}},\qquad
av_i = \prod_f \beta_{fi}^{-\beta_{fi}} \ \Rightarrow\ pv_i = 1 \text{ at } w=1. \tag{1} $$

**CES generalisation ($\sigma_{va}\neq 1$).** Value added is a CES nest over factors with per-sector
substitution elasticity $\sigma_{va}$ (the `va_elast` calibration input; $\sigma_{va}=1$ recovers
(1) as the Cobb-Douglas special case). The CES cost function and its Shephard factor demand are
$$ pv_i = \frac{1}{av_i}\Big[\textstyle\sum_f \delta_{fi}^{\,\sigma}\, w_f^{\,1-\sigma}\Big]^{\frac{1}{1-\sigma}},
\qquad F_{fi} = \frac{va_i}{pv_i}\,\frac{1}{av_i}\,(pv_i\,av_i)^{\sigma}\,\delta_{fi}^{\,\sigma}\,w_f^{-\sigma},
\tag{1'} $$
with the CES share $\delta_{fi}$ and scale $av_i$ calibrated so $pv_i=1$ and $F_{fi}=F^0_{fi}$ at the
benchmark ($w=1$). A **non-unitary** $\sigma_{va}$ lets firms substitute between capital and labour
as the carbon price shifts relative factor prices (factor substitution). Note this is *not* a
double-dividend model: there is no distortionary labour-tax wedge, and with one household
`labour_tax_cut` recycling is allocation-equivalent to `lump_sum` — the double-dividend channel
needs heterogeneous households or a labour-tax distortion (a documented follow-up; see §3.1/§9).
`va_elast` must be finite and strictly positive (scalar or one value per sector). The same nest is
used in the open model (§8).

**Zero-profit / price** (unit cost = price):
$$ p_i = \sum_j ax_{ji}\,p_j + va_i\,pv_i + \tau e_i. \tag{2} $$

**Household** — income from factor endowments **plus recycled carbon revenue** $R$, Cobb-Douglas
demand:
$$ I = \sum_f w_f FF_f + R, \qquad R = \sum_i \tau e_i X_i, \qquad FD_i = \gamma_i \frac{I}{p_i}. \tag{3} $$

**Goods market clearing** (output meets intermediate + final demand):
$$ X_i = \sum_j ax_{ij} X_j + FD_i \ \Rightarrow\ X = (I - ax)^{-1} FD. \tag{4} $$

Since $R$ depends on $X$ which depends on $I$ which depends on $R$, the fixed point is solved in
closed form: with $FD = I\,(\gamma/p)$ and $X=(I-ax)^{-1}FD$, we get $R = I\,k$ where
$k = (\tau e)^{\top}(I-ax)^{-1}(\gamma/p)$, so $I = \big(\sum_f w_f FF_f\big)/(1-k)$.

**Factor demand** (Shephard's lemma on the value-added cost) and **market clearing**:
$$ F_{fi} = \beta_{fi}\,\frac{va_i\,pv_i\,X_i}{w_f},\qquad \sum_i F_{fi} = FF_f. \tag{5} $$

**Closure / square system.** Unknowns: $N$ prices $p$ + $|F|$ factor prices $w$. Equations: $N$
zero-profit conditions (2), $|F|-1$ factor-clearing conditions (5) — **one is dropped by Walras'
law** — and 1 numéraire equation $\prod_i p_i^{\gamma_i} = 1$ (the household's exact Cobb-Douglas
price index = its cost of living, fixed to 1). Equations = unknowns, so the system is square. The
dropped factor market is confirmed to clear at the solution (the Walras check). Because the CPI is
the numéraire there is **no separate inflation/deflator output** — pinning the *arithmetic* index
while reporting the *geometric* one would force a spurious non-positive "deflator" by AM-GM (review
P1); the CPI-consistent numéraire avoids that. Outputs are real quantities and relative prices.

### 4a. Revenue recycling

The carbon tax collects $R=\sum_i \tau e_i X_i$. **In a closed economy the revenue must
circulate** — money cannot vanish, or the circular flow (and Walras' law) does not close. So the
household receives $R$ (equation 3):

- **`lump_sum`** — $R$ is returned as a lump-sum transfer.
- **`labour_tax_cut`** — $R$ rebates a labour tax. In this **single-household** pilot the household
  owns both factors, so a labour rebate and a lump-sum transfer give the *same* aggregate income
  and hence the same real allocation; the modes are equivalent here. They diverge only with
  heterogeneous households or a distortionary labour-tax wedge (the double-dividend channel) — a
  documented follow-up.
- **`none`** — revenue *not* returned. This does **not** close the economy (the dropped factor
  market fails to clear — Walras' law breaks by exactly the leaked revenue), so the engine rejects
  it: it defaults a positive-carbon-price `none` scenario to `lump_sum` (recorded in the manifest as
  `recycling_defaulted_from_none`) and points the user to Engine 1 for the pure price-side view.

**Revenue recycling — what is established, precisely.** Two results are validated: (i) under a
recycled carbon price the household's **Cobb-Douglas utility** falls only slightly (the remaining
loss is the relative-price distortion, since the revenue is returned); and (ii) *at the recycled
equilibrium prices*, adding the transfer raises utility versus not adding it. What is **not** yet
established is a full general-equilibrium welfare *comparison against a valid no-recycling closure*:
the `none` mode does not close (it violates Walras' law — the revenue leaks), so it is not a valid
counterfactual, and a proper one needs a government/external account (a documented follow-up). The
substitution signal itself is clear: output **reallocates** from the dirty to the clean sector.
Reported outputs: `welfare_change` — the change in CD utility $U=\prod_i FD_i^{\gamma_i}$ (the
correct welfare measure for the CD household); `carbon_revenue`; `gdp_change_real` (real GDP in
CPI-numéraire units) and `gdp_change_nominal_wage` (a wage-numéraire nominal reference — not tied
to the CPI numéraire); and per-factor `factor_price_change` (incl. the capital rental rate). There
is **no `deflator`** output (the CPI is the numéraire — see §4).

## 5. Calibration

At the benchmark all prices are 1, so parameters read straight off the balanced SAM
[Hosoe2010, ch. 6]: $ax_{ji} = Z^0_{ji}/X^0_i$; $va_i = VA^0_i/X^0_i$;
$\beta_{fi} = F^0_{fi}/VA^0_i$; $av_i = \prod_f \beta_{fi}^{-\beta_{fi}}$;
$\gamma_i = FD^0_i / \sum_k FD^0_k$; $FF_f = \sum_i F^0_{fi}$. By construction the **zero-profit
identity** $\sum_j ax_{ji} + va_i = 1$ holds exactly, which is why the benchmark replicates.

**Universal replication gate.** A balanced SAM can nonetheless carry flows *outside* the implemented
topology (e.g. an offsetting household↔commodity loop), which the structural validators accept but
the model cannot reproduce — every reported change would then be measured against a wrong benchmark.
So after calibration the engine derives the state at benchmark prices and **asserts every calibrated
quantity is reproduced** ($X/FD/F$ closed; $Z/D/E/M/Q/FD/F$ open) to $10^{-6}$, refusing the run
otherwise. The run manifest's SAM fingerprint is **canonicalised by account label** so two economies
with permuted axes but an identical numeric block get distinct identities.

**Scale normalisation.** All benchmark levels are divided by benchmark GDP before calibration, so
magnitudes are $O(1)$. A CGE is homogeneous of degree zero, so the level scale is arbitrary and
every reported result is a relative change — normalisation changes no ratio, share, or output. It
matters only for numerics: real EXIOBASE flows are $\sim 10^9$, where an *absolute* solver residual
never reaches a tight tolerance; unit-scaling makes the benchmark residual genuinely machine-zero.

## 5a. SAM construction from EXIOBASE (Phase 5.1)

The real-data calibration target is built by `cge.data.sam.build_sam` from an aggregated EXIOBASE
`IOSystem`:

1. **Gross output** $x = (I-A)^{-1}\,fd$ (Leontief), then intermediate flows $Z = A\,\mathrm{diag}(x)$.
2. **Collapse to one region** (the pilot is closed): sum $Z$, final demand and value added over
   regions to a sector-by-sector table. Inter-regional trade is folded into the domestic block
   until the open-economy sub-phase adds a rest-of-world account (documented, not hidden).
3. **Value added** $VA_i = x_i - \sum_j Z_{ji}$, **split** into capital/labour by a documented share
   (EXIOBASE's factor detail is thin; the split is an explicit assumption recorded in the SAM
   quality report, default 0.4 capital / 0.6 labour).
4. **Assemble** the SAM (sectors, factors CAP/LAB, one household) and **quality-gate** it.

**Balancing & quality.** A closed IO construction is balanced by construction; if a residual
imbalance remains (thin data, fabricated cells) it is **RAS**-balanced [MillerBlair2009, §7.4] and
the adjustment magnitude is recorded. The `cge.data.sam.quality` report checks: balance identity
(fatal), **aggregate preservation** (the SAM reproduces the source EXIOBASE gross output / final
demand / value added within $10^{-6}$), balancing-adjustment magnitude (WARN past 5%), negative
cells (**fatal** — calibration reads shares off the cells), the size of any negative-value-added
clip (recorded pre-clip so the transformation is visible), and the assumed capital share. A SAM
whose report **fails** is rejected; the engine will not calibrate on a bad SAM. The report's worst
severity and summary are surfaced in the run manifest (`sam_quality`).

**A directly-supplied SAM** (the toy pilot path) is separately **validated** — account alignment,
finite, non-negative, balanced — and rejected otherwise (it does not go through `build_sam`, so it
must be checked on its own; review P1). Calibration additionally rejects zero-output sectors, zero
value added, and invalid final-demand totals before dividing by them.

**Emission cost.** For a real build the per-sector carbon cost share is built by Engine 1's
`carbon_cost_vector` (gases, coverage, GWP, and the $10^{-6}$ scaling) on the multi-regional labels
and aggregated to sectors output-weighted — so the carbon wedge is unit-consistent with Engine 1
(review P0). The satellite identity and a content hash of the effective cost-share vector are
recorded in the manifest, so a changed satellite, doubled emissions, or a different gas/coverage
selection all move the result's identity (review P1).

## 6. Solver

The equilibrium is a square nonlinear system $F(z)=0$ solved by `cge.engines.cge_static.solver`
via a **pure-Python scipy root-find** (log-space, so positivity-bounded prices/quantities cannot
cross zero). A non-converged solve **raises** (`SolveError`) — never returns non-equilibrium numbers
— and the solver backend, its termination status, and the residual norm are recorded in the run
manifest (`solver_backends`, `solver_statuses`).

**IPOPT is not yet enabled for the CGE.** The solver abstraction *can* use IPOPT (and does for a
symbolic test program), but the CGE model residual is currently numeric-only (it evaluates the
Leontief inverse and Cobb-Douglas cost functions with numpy), so it cannot build a symbolic Pyomo
model. The engine therefore pins the scipy backend; a symbolic residual to enable IPOPT is a
documented follow-up (review P1). scipy solves the small model well; CI needs no solver binary.

## 7. Validation

Implemented as the `cge_static` **validation suite** (`cge.validation.suites.cge_static`), run by
`cge validate` and gated in CI via `tests/test_validation.py`; unit tests in
`tests/test_cge_static.py` and `tests/test_cge_solver.py`. Current checks (the standard CGE
battery, §7 of the phase-5 plan):

| Check | Property |
|---|---|
| `benchmark_replication` | zero shock ⇒ model reproduces the SAM to machine precision (**the** CGE correctness test) |
| `homogeneity_degree_zero` | scaling nominal size leaves prices unchanged, reals scale (no money illusion) |
| `walras_law` | the dropped factor market clears residually at the solution |
| `walras_holds_under_carbon_price_with_recycling` | under a recycled carbon price the dropped factor market still clears (a pure-loss `none` would not) |
| `carbon_price_reallocates_dirty_to_clean` | with recycling, output shifts from the dirty sector to the clean one |
| `carbon_price_raises_dirty_relative_price` | the dirty good's price rises relative to the clean good's |
| `recycled_carbon_price_welfare_is_small_and_negative` | under a recycled carbon price the CD utility change is small and negative (the distortion) |
| `recycling_improves_welfare_over_no_recycling` | at the recycled prices, adding the transfer raises CD utility (a valid fixed-price comparison, not the non-closing `none` equilibrium) |
| `replicates_on_built_sam` | the CGE calibrates on a SAM built from an EXIOBASE-shaped build (offline pymrio test MRIO, not live EXIOBASE) and replicates its benchmark to machine precision (the 5.1b gate) |

Plus solver checks (known-answer, non-convergence raises, IPOPT gated) and engine tests
(zero-shock replication, GE outputs emitted, cross-engine sign consistency with Engine 2, recycling
rejected in the pilot).

## 8. Open economy (Armington / CET)

When the SAM carries a rest-of-world (`ROW`) account, the engine runs the **open-economy variant**
(`cge.engines.cge_static.model_open`) — a small open economy with separate activity and commodity
accounts [Hosoe2010, ch. 7]:

- **Armington**: the composite commodity $Q_i$ used by intermediates and the household is a CES
  aggregate of the domestically-produced variety $D_i$ and imports $M_i$ (elasticity σ, `arm_elast`).
- **CET**: activity output $Z_i$ is transformed between domestic sales $D_i$ and exports $E_i$
  (elasticity Ω, `cet_elast`), a convex transformation frontier.
- World import/export prices are fixed (small-open-economy); foreign savings is fixed at its
  benchmark level; the **exchange rate is endogenous** and the value trade balance clears through
  relative prices. The CES/CET share and scale parameters are calibrated so both the composite
  price and the output price equal 1 at benchmark (verified against Shephard's/Hotelling's lemma).
  Value added uses the same CES nest as the closed model (§4, eq. 1′); a per-sector `va_elast`
  drives factor substitution. Structural zeros (a non-traded good, $M_i=0$; a non-exporter,
  $E_i=0$) are supported — the singular CES/CET price terms are masked, not evaluated.

**Square residual system.** The unknowns are $z=(pd, pq, w, er)$ — that is $2N + |F| + 1$ of them
($N$ domestic prices, $N$ composite prices, $|F|$ factor prices, one exchange rate); the output
price $pz$ is a *derived* CET dual of $pd$, not an independent unknown. The residuals are exactly
$2N + |F| + 1$: $N$ Armington-price identities, $N$ zero-profit identities, $|F|-1$ factor-market
clearings (one dropped by Walras), one trade balance, and the CD-CPI numéraire. **Composite-market
clearing $Q_i=\sum_j ax_{ij}Z_j+FD_i$ is *not* an independent residual** — it is solved
algebraically inside the quantity block ($Q=(I-ax\,\mathrm{diag}(\text{ratio}))^{-1}FD$,
$Z=\text{ratio}\cdot Q$), so the market clears by construction; adding it as a row would be
tautological and overdetermine the system on paper. The system is therefore genuinely square.

**Foreign-savings closure.** A **non-zero current account** is supported. Foreign savings
$S_f = \sum M - \sum E$ (the net capital inflow financing a trade deficit) is **fixed at its
benchmark level** — the standard small-open-economy closure — and the rest of the world runs a
matching capital account, recorded in the SAM as a **ROW→household transfer** of $S_f$. That transfer
enters household income valued at the exchange rate:
$$ I = \sum_f w_f FF_f + er\cdot S_f + R, $$
so the model replicates a non-zero-$S_f$ benchmark exactly. Calibration checks that the SAM's
ROW→household transfer equals $\sum M - \sum E$ (rejecting a mis-specified ROW capital account). The
trade-balance residual holds $S_f$ fixed: $\sum p^m_i M_i - \sum p^e_i E_i - er\cdot S_f = 0$. An
*endogenous*-$S_f$ closure (a savings-investment account with domestic investment demand) is a
documented follow-up.

It **replicates its benchmark to machine precision** and produces the signature open-economy result:
a carbon price on the dirty sector causes **carbon leakage** — its domestic output falls, its
**imports rise** (substitution to foreign supply) and its **exports fall** (lost competitiveness),
while the clean sector expands and exports more. The engine emits `import_change`, `export_change`
and `exchange_rate_change` alongside the usual outputs; `gdp_change_real` is CPI-weighted expenditure
$pq\cdot FD$ (the same real-GDP contract as the closed model). An **Armington elasticity sensitivity
sweep** (`armington_sensitivity_sweep`) returns the low/central/high leakage envelope. Validation:
`open_benchmark_replication`, `open_carbon_price_causes_leakage`.

## 8a. Multi-region (true bilateral trade)

When the SAM carries **several households** ``HOH_<r>`` and region-prefixed activities
``a_<r>_<s>``, the engine runs the **multi-region variant** (`cge.engines.cge_static.model_multi`) —
a closed global economy of ``R`` regions that trade bilaterally (`toy_multi_sam` is the hand-checkable
2-region × 2-sector target). Each region ``r`` has:

- an **Armington** composite ``Q[r,s]`` that is a CES over its **domestic variety** and **imports
  from every partner region** ``o≠r`` (region-of-origin substitution);
- a **CET** transform of output ``Z[r,s]`` into **domestic sales** and **exports to every partner**
  ``d≠r``;
- **region-specific, immobile** factors and its own household.

**Price convention.** Every trade route has its own **destination-specific price** ``pe[o,s,d]`` —
there is no law-of-one-price reduction. The Armington composite in destination ``d`` is a CES over
domestic ``pd[d,s]`` and imports at ``pe[o,s,d]`` for every origin ``o``; the CET transform in
origin ``o`` is a CES-dual over domestic ``pd[o,s]`` and exports at ``pe[o,s,d]`` for every
destination ``d``. Import demand ``M[d,s,o]`` and export supply ``EX[o,s,d]`` are computed
separately from their respective duals and reconciled by an **explicit bilateral market-clearing
residual** ``M[d,s,o]=EX[o,s,d]`` for every ``o≠d`` — the equation set an earlier reduction omitted,
which let a machine-zero solver residual coexist with a double-digit-percent trade imbalance. The
unknowns are ``pd[r,s]``, ``pq[r,s]`` (composite price), the packed bilateral route prices
``pe[o,s,d]`` — **one unknown per directed route with genuine benchmark trade**, own-region slot
fixed at 1 — and ``w[f,r]`` — a square system of ``2·nr·ns + n_active + nf·nr`` where
``n_active = len(cal.active_routes) ≤ nr·(nr−1)·ns`` (domestic-market clearing solved
algebraically; one global CPI numéraire; one factor market dropped by Walras). A route with zero
benchmark trade gets **no** price unknown and **no** clearing residual — packing one unconditionally
for every possible directed route (as an earlier version did) left the system rank-deficient by
exactly the number of zero-trade routes, since nothing in the Armington/CET duals reads a
zero-share route's price anyway (`cal.active_routes`, `model_multi.py`). Foreign savings per region
``Sf[r]=ΣM−ΣE`` is fixed at benchmark and globally zero-sum (financed by household capital
transfers); there is no exchange rate and no external rest-of-world — trade is entirely among the
build's own regions.

It **replicates its benchmark to machine precision** — every bilateral import and export returns to
the SAM values (`multi_region_benchmark_replication`, plus the universal post-calibration
replication gate shared with the closed/open variants) — and **clears every bilateral goods market
and every factor market under shock**, not just at the benchmark
(`multi_region_markets_clear_under_shock`). It produces the signature result: a carbon price in
**one** region cuts that region's output, **raises its imports from partner regions** (cross-region
carbon leakage) and **raises partners' output** of that good
(`multi_region_cross_region_leakage`). Results are **region-tagged**, and the manifest records the
hashed effective carbon-cost matrix (``EffectiveCarbonCostMatrix``) so two runs that differ only in
carbon shares or recycling mode produce distinct manifests — there is currently no
IOSystem-driven multi-region SAM build (§8a's "Remaining sub-phases", below), so unlike the
single-region open economy the multi-region manifest never carries a ``SatelliteAccount``
identity; that identity only appears when a satellite is actually consulted. The emitted
``real_consumption_change`` is a base-price (Laspeyres) household-consumption index, not
production-side real GDP. This is **not** because other regions' prices are "unpinned" — one
global numéraire (region 0's CPI) fixes the common nominal scale for every region, and every
region's ``pq`` is fully determined at the solved equilibrium. The actual reason ``pq·FD`` is
unsuitable is that it is **current-price nominal expenditure**: it moves with both the quantity
change and the composite-price change, so summing it conflates the two. Valuing ``FD`` at
**base** (benchmark) prices instead strips out the price move and isolates the real quantity
effect.

**Remaining sub-phases:** an IOSystem-driven multi-region SAM build (today the multi-region variant
requires a supplied SAM — see §5a for the single-region open economy, which does have an
IOSystem build) and per-cell (rather than uniform) trade elasticities.

## 9. Honest expectations

The pilot delivers a *provably correct* general-equilibrium core: it replicates its benchmark,
satisfies homogeneity and Walras, and moves in the theory-consistent direction under a carbon
price, with input substitution and factor-market feedback. It is **not** a GTAP-precision model,
and its magnitudes are illustrative until it runs on a real balanced EXIOBASE SAM (5.1b) with
literature elasticities. *Precise about structure, indicative about magnitudes, transparent about
assumptions* — every run prints its closures, sectors, factors, solver backend, and the SAM
identity it was calibrated to.

## 10. References

- **[Hosoe2010]** Hosoe, Gasawa & Hashimoto, *Textbook of Computable General Equilibrium Modeling*
  (Palgrave Macmillan) — the pilot's structure, calibration, closures, and correctness tests.
- **[Armington1969]** Armington — the trade nest (added in the open-economy sub-phase).
- **[Robinson2001]** Robinson et al. — cross-entropy SAM balancing (the real-data SAM path).
- Miller & Blair (2009), §7.4 — RAS balancing. See `docs/references.md`.
