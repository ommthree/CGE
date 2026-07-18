# Model description: Engine 3 — static CGE (pilot)

- **Implements:** `cge.engines.cge_static` (`CGEStaticEngine`, v0.2.0)
- **Roadmap phase:** 5 (pilot: 5.0 solver + 5.1 SAM build + 5.2a model + 5.3 revenue recycling)
- **Capabilities:** general_equilibrium, prices, volumes
- **Status:** implemented as the **correctness-first pilot** that now runs on **real EXIOBASE
  data**. The model calibrates to a hand-checkable 2-sector SAM *and* to a SAM built from an
  aggregated EXIOBASE build (§5a), and in both cases passes the standard CGE correctness battery
  (benchmark replication, homogeneity, Walras) with theory-consistent carbon-price responses
  (`cge_static` validation suite). It is deliberately **not** the full production model yet:
  Armington trade, multiple regions, and revenue recycling (5.3) are the next sub-phases.
  Magnitudes from the pilot are illustrative.

## 1. Purpose & scope

Compute the **general-equilibrium** response to a carbon price: prices, output volumes, and factor
prices when firms substitute inputs and factor markets clear — the feedback Engines 1–2
structurally cannot capture. The carbon price raises the cost of emitting sectors; the economy
re-allocates; the model reports the new equilibrium relative to the benchmark.

**In scope (pilot):** a closed, single-region economy with N sectors (activity = commodity),
Leontief intermediates, Cobb-Douglas value added (capital + labour) and household demand, fixed
factor endowments, CPI numéraire, a carbon price as a per-unit emissions cost wedge, and
**carbon-revenue recycling** (lump-sum / labour-tax-cut) — the headline general-equilibrium feature
Engines 1–2 cannot provide.

**Not yet modelled:** Armington imports / CET exports (open economy), multiple regions,
savings/investment dynamics, heterogeneous households, non-unitary substitution elasticities in the
value-added nest, and a distortionary labour-tax wedge (so the "double-dividend" channel that would
distinguish labour-tax-cut from lump-sum recycling). These are the documented next sub-phases; the
pilot is the provable core.

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

1. **Leontief intermediates, Cobb-Douglas value added** (KLEM with Leontief M) — the standard
   "toy but honest" pilot structure [Hosoe2010, ch. 4-6]. Substitution happens within value added
   (between capital and labour); intermediates are fixed-coefficient.
2. **Cobb-Douglas household demand** — constant budget shares $\gamma_i$.
3. **Fixed factor endowments** — labour and capital supply are inelastic; factor prices adjust to
   clear their markets.
4. **CPI numéraire** — the consumer price index is fixed to 1, pinning the price level (the model
   is homogeneous of degree zero in nominal prices; only relative prices are determined).
5. **Carbon price = a cost wedge** — $\tau e_i$ is added to sector $i$'s unit cost in the
   zero-profit condition, reusing the Engine-1 emission intensities so units stay consistent. The
   pilot uses `none` revenue recycling (the tax is a pure wedge).

## 4. The model (equation level)

Unknowns are the price vectors $(p, w)$; every quantity is a closed-form function of prices, so the
equilibrium is a small square system.

**Value-added unit cost** (Cobb-Douglas cost function):
$$ pv_i = \frac{1}{av_i}\prod_f \left(\frac{w_f}{\beta_{fi}}\right)^{\beta_{fi}},\qquad
av_i = \prod_f \beta_{fi}^{-\beta_{fi}} \ \Rightarrow\ pv_i = 1 \text{ at } w=1. \tag{1} $$

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
law** — and 1 numéraire equation $\sum_i \gamma_i p_i = 1$. Equations = unknowns, so the system is
square. The dropped factor market is confirmed to clear at the solution (the Walras check).

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

**The revenue-recycling effect** (a validated result): a carbon price *without* recycling reduces
real household consumption; returning the revenue offsets it. With full recycling the aggregate
economy is roughly preserved but output **reallocates** from the dirty sector to the clean one — the
substitution signal, which Engines 1–2 cannot show. Reported outputs include `welfare_change` (real
consumption index), `carbon_revenue`, and per-factor `factor_price_change`.

## 5. Calibration

At the benchmark all prices are 1, so parameters read straight off the balanced SAM
[Hosoe2010, ch. 6]: $ax_{ji} = Z^0_{ji}/X^0_i$; $va_i = VA^0_i/X^0_i$;
$\beta_{fi} = F^0_{fi}/VA^0_i$; $av_i = \prod_f \beta_{fi}^{-\beta_{fi}}$;
$\gamma_i = FD^0_i / \sum_k FD^0_k$; $FF_f = \sum_i F^0_{fi}$. By construction the **zero-profit
identity** $\sum_j ax_{ji} + va_i = 1$ holds exactly, which is why the benchmark replicates.

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
cells, and the assumed capital share — so a run states how much the data was "helped". A SAM whose
report **fails** is rejected; the engine will not calibrate on a bad SAM. The report's worst
severity and summary are surfaced in the run manifest (`sam_quality`).

**Emission intensities.** For a real build, per-sector emission intensity is the output-weighted
mean of the regional intensities from the satellite CO₂ row — reusing the same emissions data as
Engine 1, so the carbon-price wedge $\tau e_i$ is unit-consistent across engines.

## 6. Solver

The equilibrium is a square nonlinear system $F(z)=0$ solved by `cge.engines.cge_static.solver`:
**IPOPT via pyomo when its binary is present, else a pure-Python scipy root-find** (log-space, so
positivity-bounded prices/quantities cannot cross zero). A non-converged solve **raises**
(`SolveError`) — never returns non-equilibrium numbers — and the backend + termination status are
recorded in the run manifest. CI runs on the scipy fallback so it needs no solver binary.

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
| `revenue_recycling_offsets_welfare_loss` | lump-sum recycling restores real consumption a pure wedge destroys (the revenue-recycling effect) |
| `replicates_on_real_exiobase_sam` | the CGE calibrates on a SAM built from a real EXIOBASE build and replicates its benchmark to machine precision (the 5.1b gate) |

Plus solver checks (known-answer, non-convergence raises, IPOPT gated) and engine tests
(zero-shock replication, GE outputs emitted, cross-engine sign consistency with Engine 2, recycling
rejected in the pilot).

## 8. Honest expectations

The pilot delivers a *provably correct* general-equilibrium core: it replicates its benchmark,
satisfies homogeneity and Walras, and moves in the theory-consistent direction under a carbon
price, with input substitution and factor-market feedback. It is **not** a GTAP-precision model,
and its magnitudes are illustrative until it runs on a real balanced EXIOBASE SAM (5.1b) with
literature elasticities. *Precise about structure, indicative about magnitudes, transparent about
assumptions* — every run prints its closures, sectors, factors, solver backend, and the SAM
identity it was calibrated to.

## 9. References

- **[Hosoe2010]** Hosoe, Gasawa & Hashimoto, *Textbook of Computable General Equilibrium Modeling*
  (Palgrave Macmillan) — the pilot's structure, calibration, closures, and correctness tests.
- **[Armington1969]** Armington — the trade nest (added in the open-economy sub-phase).
- **[Robinson2001]** Robinson et al. — cross-entropy SAM balancing (the real-data SAM path).
- Miller & Blair (2009), §7.4 — RAS balancing. See `docs/references.md`.
