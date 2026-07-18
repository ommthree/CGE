# Phase 5 plan — Engine 3: a simple static CGE

**Status: PLAN ONLY — not started.** This is the design-and-test plan for the hardest phase.
The guiding principle from the roadmap holds: **"toy but honest."** A small, static,
single-region-first CGE that is *correct* and *transparent*, not a GTAP competitor. Its value
is general-equilibrium price **and** volume answers with revenue recycling, and a cross-check
on Engines 1–2 — not three-decimal precision.

Read alongside: `roadmap.md` §Phase 5, `docs/documentation-standard.md` (every model doc is
equation-level with citations), `docs/validation.md` (standing validation suite), and the
`review-2026-07-remediation.md` history (the bar for "validated" is high here).

---

## 0. Why this is the risky phase, and how the plan de-risks it

The roadmap already flagged the real risks. They are **not** the pyomo code; they are:

1. **The solver.** pyomo needs a nonlinear solver (IPOPT). It is *not currently installed*,
   and `idaes`/conda binaries or a scipy fallback must be chosen deliberately (§2).
2. **SAM construction & balancing.** EXIOBASE's value-added / tax / income detail is thinner
   than GTAP's; building a *balanced* SAM is a known rabbit hole (§4).
3. **Elasticities.** Volume results are elasticity-sensitive; there is no clean open database
   (Phase 4 already hit this).
4. **Credibility ceiling.** A solo 30–50-sector CGE gives *indicative, directionally sound*
   answers. The plan's job is to make that honest and *provable* (replication, homogeneity,
   Walras — §7), not to overclaim.

The de-risking strategy, in order: **solver first, then SAM, then a deliberately tiny model
that passes the standard correctness tests, then scale.** Never write the production model
before a 2-sector hand-checkable version passes replication.

---

## 1. Architecture & where it plugs in

Engine 3 is a new engine behind the existing `Engine` protocol (ADR-0002) — the GUI/CLI pick
it up via the registry with no changes, exactly as Engine 2 did. It reuses:

- the **`SAM` contract** (`cge.contracts.data_objects.SAM`: `accounts` + square `matrix`) —
  already exists;
- the **`ElasticitySet` contract** and the **elasticity library** from Phase 4;
- the **`CarbonPrice` / shock vocabulary** — the carbon price enters as it does in Engine 1,
  but now with substitution and factor markets responding;
- the **`ResultSet`** long-format output and the **validation framework**.

New packages/modules:

```
src/cge/
├── data/sam/            # NEW: SAM construction from an IOSystem + balancing
│   ├── build.py         #   IOSystem → raw SAM (accounts, flows, VA split, taxes)
│   ├── balance.py       #   RAS / cross-entropy balancing → balanced SAM
│   └── quality.py       #   SAM-specific QualityReport (imbalance, adjustments)
├── engines/cge_static/  # placeholder today → the model
│   ├── solver.py        #   solver abstraction (IPOPT via pyomo; scipy fallback)
│   ├── model.py         #   pyomo model: sets, params, vars, equations, closures
│   ├── calibrate.py     #   SAM → model parameters (share/scale calibration)
│   └── engine.py        #   Engine protocol wrapper; run() → ResultSet
└── validation/suites/cge_static.py   # replication, homogeneity, Walras, cross-checks
```

**Capabilities:** `Capability.GENERAL_EQUILIBRIUM` (+ `PRICES`, `VOLUMES`). The enum value
already exists.

**Sequencing decision (from the roadmap, confirmed): single region + rest-of-world FIRST.**
Multi-region only after the single-region pilot passes every correctness test in §7.

---

## 2. Sub-phase 5.0 — Solver & environment (NEW; 2–4 days) ⚠ do this first

The roadmap folded this into 5.2; experience says it deserves its own gate, because a CGE
that can't solve is worthless and solver setup is the classic time-sink.

- **Choose the solver.** Options, in recommended order:
  1. **IPOPT via `idaes get-extensions`** — a prebuilt IPOPT binary, pip-installable, no
     conda. Most portable; recommended default.
  2. **conda-forge `ipopt`** — if the environment is conda-based.
  3. **scipy `trust-constr` / SLSQP fallback** — pure-Python, no external binary; slower and
     less robust on large nonlinear systems, but guarantees the engine *runs anywhere* (e.g.
     CI without a solver binary). Keep as a documented fallback for the tiny model.
- **Solver abstraction (`solver.py`).** A thin interface: `solve(model) -> Solution` that
  tries IPOPT and falls back to scipy, records which solver ran and its termination status in
  the result manifest. Never let a silent non-convergence produce numbers (mirror the
  well-posedness lesson from Engine 1).
- **Add solver deps to `pyproject.toml`** `[cge]` extra (pyomo already declared; add the
  IPOPT-provider) and document the one-command install in the README/model doc.
- **Testing (5.0):**
  - a trivial nonlinear program with a known optimum solves and returns it (proves the solver
    is wired and available);
  - the fallback path is exercised (scipy) so CI without IPOPT still runs the tiny model;
  - `solver.available()` is surfaced so `cge validate` skips solver-dependent checks cleanly
    when no solver is present (same gated pattern as the live-EXIOBASE suite).
- **DoD 5.0:** `from cge.engines.cge_static.solver import solve` solves a known 2-var NLP to
  the analytic optimum on this machine; the fallback is tested; the install path is documented.

**Decision to make here:** IPOPT-required vs scipy-fallback-acceptable for the *shipped* small
model. Recommendation: IPOPT for real runs, scipy fallback kept green in CI on the 2-sector
toy so the test suite never depends on a solver binary.

---

## 3. Notation & the model, to equation level (drafted now, finalised in the model doc)

The model doc (`docs/models/cge-static.md`) is written **before** the model code, per the
documentation standard, and is the spec. Its skeleton (single region + RoW, one representative
household, government, one composite labour + capital factor set):

**Sets:** sectors/commodities $i,j$; factors $f\in\{L,K\}$.

**Production** — nested CES so a carbon price can shift the energy nest [Hosoe2010]:
$$
X_i = A_i \Big[\delta_i\,\mathrm{KLE}_i^{\rho_i} + (1-\delta_i)\,\mathrm{M}_i^{\rho_i}\Big]^{1/\rho_i},
\quad \mathrm{KLE}_i \text{ a nested CES of } K,L,\text{Energy}. \tag{P}
$$
Carbon pricing raises the price of the energy composite, inducing substitution *away* from
energy — the mechanism Engines 1–2 cannot capture.

**Armington** imports / **CET** exports [Armington1969]:
$$
Q_i = B_i[\ldots D_i^{..}+\ldots M_i^{..}]^{..},\qquad
X_i = \ldots \text{(domestic vs export split)}. \tag{A}
$$

**Household** — Cobb–Douglas first (LES if needed), maximising utility s.t. its budget from
factor income net of taxes and transfers. **Government** collects the carbon tax (and any
other taxes) and recycles: `none` / `lump_sum` / `labour_tax_cut`. **Investment / savings**
close the loop. **Market clearing** for every commodity and factor; **income balance** for
each institution; **carbon price** enters as a per-unit tax on sectoral emissions (reusing the
Engine-1 emission intensities, so units stay consistent — the 2026-07 units work carries
through).

**Closure (default, documented, switchable):** savings-driven investment; fixed trade balance;
numéraire = the consumer price index (CPI). Alternatives (fixed investment, flexible trade
balance) are config options tested for *homogeneity independence*.

**Well-posedness:** the model is **square** (equations = unknowns after imposing the numéraire
and dropping one redundant market-clearing equation by Walras' law). The plan requires an
explicit equation/variable count in the model doc — a standard CGE sanity check.

The full equation set with every symbol defined is the deliverable of 5.2's model doc; this
plan fixes the *structure* and *citations* so there is no ambiguity when coding starts.

---

## 4. Sub-phase 5.1 — SAM construction & balancing (2–4 wk; the rabbit hole)

**Goal:** a balanced Social Accounting Matrix from an aggregated EXIOBASE build.

- **5.1a Raw SAM (`data/sam/build.py`).** Map the coarse EXIOBASE build (Phase 4's ~14-sector
  or a curated ~30-sector aggregation, single region + RoW) into SAM accounts: activities,
  commodities, factors (L, K), household, government, savings/investment, RoW. Split value
  added into labour vs capital using EXIOBASE's factor-input extension where present; fill
  thin spots (net taxes, margins, income flows) with **documented, cited assumptions** (every
  fabricated cell recorded — see below).
- **5.1b Balancing (`data/sam/balance.py`).** A SAM must be square and row-sum = column-sum.
  Real data isn't. Balance with **RAS** first (simple, transparent) and **cross-entropy**
  [Robinson2001] as the principled alternative when RAS struggles. The balancing method and
  its adjustment magnitudes are recorded.
- **5.1c SAM quality (`data/sam/quality.py`).** A SAM-specific `QualityReport`: row/column
  imbalance *before* and *after* balancing, the magnitude of every adjustment (so a reviewer
  can see how much the data was "helped"), negative-cell flags, and an **audit trail** of
  fabricated cells with their source/assumption. This is the credibility surface for the CGE —
  reviewers will ask "how much did you make up," and the answer must be inspectable.

**Testing & validation (5.1) — this is where correctness is cheapest to establish:**
- **Balance identity:** balanced SAM has row-sum = column-sum for every account within `1e-6`
  (enforced, like the Phase 1 conservation gates).
- **Aggregate preservation:** the SAM's totals reproduce the source EXIOBASE aggregates
  (gross output, final demand, total emissions) within tolerance — the same "conservation
  through a transform" check that caught bugs in Phase 1.
- **Adjustment audit:** the total absolute adjustment made by balancing is reported and
  bounded (a WARN if balancing moved more than, say, 5% of any account — a signal the raw
  data or assumptions are off, not a silent "fixed").
- **Hand-checkable toy SAM:** a 2-sector SAM built by hand with a known balanced form, so the
  balancer is tested against an exact answer, not just "it converged."
- **DoD 5.1:** a balanced SAM from a real EXIOBASE small build, reproducing source aggregates
  within tolerance, with a stored SAM `QualityReport` and a full adjustment audit trail.

**Timebox discipline (roadmap risk):** SAM balancing can consume unlimited time chasing tiny
imbalances. The plan sets an explicit tolerance and *documents* residual imbalance rather than
perfecting it. "Balanced to 1e-6 with a documented 2% assumption on capital's VA share" is an
acceptable, honest outcome; a month lost to a 0.1% residual is not.

---

## 5. Sub-phase 5.2 — Model core (2–4 wk)

Build the model **smallest-first**:

- **5.2a The 2-sector pilot.** A hand-derived 2-sector, 1-household, 1-factor CGE calibrated to
  the hand-checkable toy SAM. Small enough to verify every calibrated share by hand. This
  pilot must pass replication + homogeneity + Walras (§7) *before* any scaling. This is the
  single most important discipline in the phase — a CGE that replicates its benchmark is
  almost certainly wired correctly; one that doesn't is silently wrong.
- **5.2b Calibration (`calibrate.py`).** Standard CGE calibration: derive CES share and scale
  parameters, tax rates, and household budget shares *from the benchmark SAM* so the model
  reproduces the base year exactly at the benchmark prices. Calibration is deterministic and
  unit-tested against the pilot's hand-computed parameters.
- **5.2c The full model (`model.py`).** Generalise the pilot to N sectors, the nested-CES
  energy structure, Armington/CET, government recycling, and switchable closures. Square-model
  and degrees-of-freedom checks assert equations == unknowns.
- **5.2d Engine wrapper (`engine.py`).** `run(data, shocks, years)` → calibrate → apply the
  carbon-price shock (as a per-unit emissions tax) → solve → emit `ResultSet` with
  `price_change`, `volume_change`, plus GE-specific outputs (factor prices, welfare, government
  revenue) and a manifest recording the SAM build id, closure, elasticities, solver + status.
- **5.2e Macro aggregates (the GE tier of Phase 4b).** Emit as **native** model variables, per
  region and time-step: **GVA per sector** (factor income by activity), **GDP per region**
  (Σ GVA / final-expenditure identity — report both and check they agree), the **GDP deflator and
  CPI**, and every nominal value alongside its **real** counterpart (deflated by the index; the
  CPI numéraire makes this exact and homogeneity-test-backed). Report the **capital rental rate**
  as the honest "interest-rate-like" factor price — a *monetary* interest rate is out of scope for
  the CGE core (see roadmap 4b.5). Cross-check these against the indicative PE-tier estimates
  (roadmap 4b.1–4b.2): same sign, GE typically smaller in magnitude.

**Decisions forced (make explicit in the model doc):**
- nesting structure (KL-E-M recommended so carbon pricing bites the energy nest);
- closure defaults (savings-driven, fixed trade balance, CPI numéraire recommended);
- Cobb-Douglas vs LES household demand (start CD);
- how much tax detail to model vs omit (start: carbon tax + a single output tax placeholder);
- single- vs multi-region (single first, non-negotiable).

**Testing (5.2):**
- calibration reproduces the pilot's hand-computed parameters exactly;
- the model is square (equation/variable count asserted);
- the model **instantiates and solves** from the SAM at the benchmark (feeds directly into
  §7's replication test);
- closures are switchable by config and each yields a solvable model.
- **DoD 5.2:** model solves from a real SAM; equation/variable count documented; closures
  switchable; the 2-sector pilot passes all §7 correctness tests.

---

## 6. Sub-phase 5.3 — Calibration credibility & the carbon-price experiment (2–4 wk)

Wire the carbon price through and establish that the *responses* are sensible, not just that
the benchmark replicates.

- **Carbon-price run** on the small build: report Δprices, Δvolumes, factor-price changes,
  welfare change, and government carbon revenue, under each recycling option
  (`none`/`lump_sum`/`labour_tax_cut`). Revenue recycling is a headline GE feature Engines 1–2
  cannot do — it is the reason the CGE exists.
- **Elasticity sensitivity sweeps:** run low/central/high elasticity bundles → response
  envelopes (reuse the Phase 4 band pattern), because volume results are elasticity-sensitive.
- **Documentation deliverable:** the model-description doc (§3) finalised — equations,
  closures, elasticities and sources. "This is what CGE-literate reviewers will ask for."

**DoD 5.3:** replication + homogeneity + Walras green in CI; a carbon-price experiment with
revenue recycling produces sensible, documented responses; the model doc is complete.

---

## 7. Testing & validation strategy (the core of this plan)

CGE correctness has a **standard, non-negotiable test battery**. These are not optional; a CGE
that fails any of them is wrong. They go in the `cge_static` validation suite (`cge validate`)
and the pytest gate, following the established framework.

### Tier 1 — the standard CGE correctness tests (must pass, in CI)

1. **Benchmark replication.** With **zero shocks**, the calibrated model reproduces the
   base-year SAM to **machine precision** (all quantities and prices return their benchmark
   values). This is *the* CGE correctness test — it proves calibration and the equation system
   are mutually consistent. Tolerance: `1e-8` on the toy, `1e-6` on the real small build.
2. **Homogeneity of degree zero in prices.** Scaling the numéraire (all nominal prices) by any
   factor leaves all **real** quantities and relative prices unchanged. Tests that the model
   has no money illusion. Run at ×2 and ×10.
3. **Walras' law.** The sum of all excess demands (value) is identically zero, so one market
   clears residually. Test by *dropping* one market-clearing equation and confirming it holds
   at the solution to `1e-6` — this also confirms the square-model count is right.

### Tier 2 — economic-sense and consistency checks

4. **Sign & direction under a carbon price:** dirty-sector output falls, its price rises,
   emissions fall; the energy nest substitutes away from energy. Directional, robust,
   published-consensus signs.
5. **Cross-engine consistency (the built-in advantage):** for a small carbon price, the CGE's
   price changes should **bracket** Engine 1's (substitution dampens the pure pass-through),
   and its volume changes should be **same-sign** as Engine 2's. This is a free, powerful
   sanity check — three independent methods on the same data must agree qualitatively. Encode
   it as a validation check.
6. **Revenue neutrality:** under `lump_sum` recycling, government's budget balances (carbon
   revenue = lump-sum transfer) to `1e-6`.
7. **Welfare direction:** a carbon price without recycling reduces household welfare; recycling
   partially offsets it (the standard "revenue-recycling effect"). Directional check.

### Tier 3 — robustness & documentation

8. **Elasticity sensitivity is bounded and monotone** where theory predicts monotonicity
   (e.g. higher substitution elasticity → larger volume response).
9. **Solver-status gate:** a non-optimal solver termination **raises**, never returns numbers
   (the Engine-1 well-posedness lesson). The manifest records solver + status.
10. **Model doc ↔ code consistency:** the manifest's assumptions match the model doc's stated
    closures/assumptions (as enforced for Engines 1–2).

### Known-answer & literature comparison

11. **Analytic toy known-answer:** the 2-sector pilot's response to a small tax is compared to
    a **hand-derived** (or textbook, e.g. Hosoe et al.) closed-form/known result — an
    *independent* check, not the model checking itself. This is the CGE analogue of Engine 1's
    hand-derived known-answer.
12. **Literature bracket (indicative):** the small build's carbon-price GDP/output responses
    are compared to *published* CGE carbon-price studies for similar economies — checked to be
    in the right **order of magnitude and direction**, not to a decimal. Documented honestly
    as a bracket, not a benchmark (the same honesty applied to the EXIOBASE live gate).

### Test infrastructure notes

- **CI stays solver-independent:** Tier-1 tests run on the 2-sector toy via the **scipy
  fallback** so they pass in CI with no IPOPT binary. IPOPT-only checks on the real small build
  are **gated** (like the live-EXIOBASE suite) and run locally / when a solver is present.
- **Every fabricated SAM cell and every default elasticity is flagged** in the run manifest, so
  no result hides an assumption.
- **Replication is the canary:** it runs first; if it fails, nothing else is trusted.

---

## 8. Effort, sequencing & dependencies

| Sub-phase | Deliverable | Effort |
|---|---|---|
| 5.0 Solver & env | IPOPT/scipy solver abstraction, install path, solver tests | 2–4 d |
| 5.1 SAM | balanced SAM + SAM QualityReport + audit trail + balance tests | 2–4 wk |
| 5.2 Model core | 2-sector pilot → full model, calibration, engine wrapper | 2–4 wk |
| 5.3 Credibility | carbon-price experiment + recycling, sensitivity, model doc | 2–4 wk |
| **Total** | **general-equilibrium price + volume + revenue recycling** | **~6–12 wk** |

**Dependencies:** Phase 1 (data), Phase 4 (elasticities + `ElasticitySet`), the `SAM` contract
(exists), a solver (5.0). **Unblocks:** the Phase 7 recursive-dynamic pathway stack, and
GE-mode nature runs in Phase 6.

**Critical path & gates (do not skip in order):**
`5.0 solver works` → `5.1 SAM balances & preserves aggregates` → `5.2a 2-sector pilot
replicates + homogeneity + Walras` → `5.2c scale to full model` → `5.3 carbon-price experiment`.
The 2-sector-pilot-replicates gate is the make-or-break checkpoint; do not build the full model
until it passes.

---

## 9. Honest expectations (carry into the docs, as for Engines 1–2)

- **What the CGE will genuinely deliver:** general-equilibrium price *and* volume responses to
  a carbon price, **with revenue recycling** — the thing Engines 1–2 structurally cannot do —
  validated to replicate its benchmark and satisfy homogeneity and Walras.
- **What it will *not* be:** a GTAP-precision model. A solo ~30–50-sector CGE gives
  *indicative, directionally sound* magnitudes. Reviewers who know CGE will probe closures and
  elasticities — so the plan documents both explicitly and prints them per run.
- **The phrase to hold onto** (from the roadmap): *precise about costs, indicative about
  volumes, transparent about assumptions.* The CGE adds general-equilibrium feedback and
  revenue recycling to that, at the same honesty bar.

## 10. New references needed (add to `docs/references.md` when work starts)

- **[Hosoe2010]** — already listed; the "toy but honest" static CGE reference the model core
  follows (nesting, calibration, closures).
- **[Robinson2001]** — already listed; cross-entropy SAM balancing.
- **[Armington1969]** — already listed; Armington trade.
- To add: a standard CGE textbook treatment of homogeneity/Walras/replication tests, and 1–2
  published carbon-price CGE studies for the Tier-3 literature bracket.
