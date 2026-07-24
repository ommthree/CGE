# Phase 5d plan — the macro closure (government, investment, energy nest, labour market)

**Status: 5d.1, 5d.2, 5d.3, 5d.4 COMPLETE; 5d.5 closed-variant COMPLETE** (engine v0.9.0). 5d.1: government
account in all three variants (`GOV` / `GOV_<r>` per region, balanced-budget closure, benchmark
direct tax as a rate on factor income, `fiscal_balance`/`gov_spending` outputs — `docs/models/
cge-static.md` §4b; `deficit_financed` reserved for 5d.7 as planned). 5d.2: savings-investment
account in all three variants (`SAVINV` / `SAVINV_<r>` per region, savings rate on disposable
income, `savings_driven` + `fixed_real` closures both switchable and tested, `investment`/`savings`
outputs — §4c). **The plan's anticipated +1 unknown/+1 equation never materialised**: every
closure substitutes the savings-investment identity in closed form, so all three residual systems
stay square — the §0.3 square-count risk was discharged trivially. **The genuine open/multi
closure change** (identified in the plan) landed too: foreign savings re-route from household
income into the investment pool (SAM: ROW↔`SAVINV` / inter-`SAVINV` capital transfers, not
ROW↔household / HOH↔HOH), so the open/multi identity is `p·ID = S + er·Sf` — investment and
savings differ by exactly the capital-account inflow. Mixing the two routes is rejected at
calibration. 5d.3: the capital-accumulation identity `K_{t+1}=(1−δ)(1−r)K_t+INV` as a standalone,
stateless, unit-tested module (`cge.engines.cge_static.capital`) — the Phase 7.1 entry point, with
a 5%/yr depreciation default [OECD2009], an exogenous premature-retirement (stranded-asset) term,
boundary validation, and a `benchmark_capital(cal)` adapter that extracts region-level $K_0$ from
any variant (§4d). Deliberately NOT wired into the solve — the multi-year loop is Phase 7.1.
5d.4: labour-market wage-floor closure (closed variant, §4e) — default flexible-wage/full-
employment plus an optional `labour_floor` via a REGIME-SWITCH (solve full-employment first; if
the unconstrained wage would fall below the floor, re-solve with the LAB clearing row replaced by
the wage pin `w_L = floor` — labour demand ≤ supply slack, gap reported as `unemployment`). No MCP
solver needed. The employed-labour income fixed point (household earns only employed labour → a
contraction reusing the six income branches) keeps it exact. Walras re-proved: the CAP market still
clears at the pinned-wage solution (the flagged Tier-1 check). Floor never applied to the benchmark
(full employment at wage 1); floor ≥ 1 rejected. A slack floor is byte-identical to full
employment. Wage-curve alternative + open/multi generalisation are documented follow-ups.
5d.5: the KL-E-M energy nest (closed variant, §4f) — the shared CES algebra in `energy_nest.py`
(unit-tested standalone first, like `capital.py`), opt-in via `energy_sectors`. Production goes
from flat Leontief to a 3-level CES (outer KLE-vs-materials, middle KL-vs-energy, inner
CES-over-energy-commodities); each layer calibrated δ∝v^{1/σ} so the benchmark replicates to
machine precision. Carbon cost attaches to the energy commodities (raising their in-nest price);
the fixed Leontief inverse becomes a price-responsive `(I−A(p))⁻¹`, with an effective per-output
carbon cost `ĉ_i=Σ_e cc_e·a_ei(p)` keeping the recycling/government fixed points linear/unchanged.
Tier-1 (replication/homogeneity/Walras) green; the Tier-2 DELIVERABLE proven: a fossil carbon
price shifts a sector's fossil/electricity ratio down, contracts fossil output, expands
electricity. Open/multi energy nest = remaining 5d.5 work. 5d.6–5d.7 not started. This is the
detailed implementation plan for Phase 5d, reopened Phase 5
debt (see `roadmap.md` §Phase 5 correction note and `docs/phase-5-plan.md`'s status header). Phase
5's original §2/§3 design called for a government/fiscal account, savings/investment with capital
accumulation, and a genuine KL-E-M energy nest; none of these existed in the implemented model
when this plan was drafted. This plan follows the same structure and rigor as
`docs/phase-5-plan.md` — read that first for the overall CGE architecture and the Tier 1/2/3
testing discipline this plan extends, not replaces.

Read alongside: `roadmap.md` §Phase 5d (task table 5d.1–5d.7, effort/sequencing, dependencies),
`docs/models/cge-static.md` (the current equation-level spec this plan amends), `docs/
documentation-standard.md`, `docs/validation.md`.

---

## 0. Why this phase is riskier than it looks, and how the plan de-risks it

Phase 5's pilot is correct *within its scope* — replication, homogeneity, and Walras all hold for
the closed/open/multi-region variants as built. The risk in 5d is not "does the CGE solve" (it
already does) but **"does adding new accounts and a new nest preserve every one of those
guarantees while genuinely changing behaviour."** Four specific risks:

1. **The SAM contract currently assumes exactly one non-factor institution.**
   `calibrate.py::calibrate()` does `hoh = [a for a in sam.accounts if a not in sectors and a not
   in factors]; if len(hoh) != 1: raise ValueError(...)`. The `SAM` contract's docstring
   (`data_objects.py`) already names `government`, `savings/investment`, and `rest-of-world` as
   accounts a SAM *may* carry — but no calibration code has ever read them. 5d.1/5d.2 must
   generalise this to N named institutions without breaking the existing single-household SAMs
   used by every current test fixture and the pilot toy SAM. **This is the load-bearing
   compatibility risk of the whole phase** — get the institution-account plumbing wrong and every
   downstream sub-task (5d.2–5d.7) inherits the bug.
2. **The recycling fixed point must generalise, not get replaced.** The existing carbon-revenue
   recycling (`_safe_denom`-guarded fixed point `income = factor_income / (1 - k)`, strict/non-strict
   modes) is a *working, tested* pattern across all three engine variants. A real government account
   changes *what* accumulates revenue and *how* it's spent, but the underlying fixed-point-with-
   divergence-guard shape is exactly what a government budget constraint under deficit financing
   will need too. Reuse the pattern; do not invent a parallel mechanism.
3. **Investment demand changes the goods-market closure everyone already depends on.** Today,
   `_quantities()` in `model.py`/`model_open.py`/`model_multi.py` solves `Q = (I - ax·ratio)⁻¹ · FD`
   with `FD` being pure household demand. Adding investment demand means `FD` becomes
   `FD_household + FD_investment`, with the latter's *sectoral composition* and *total* determined
   by a savings-investment identity that is itself a new unknown in the residual system. This
   changes the square-count (equations == unknowns) for every engine variant — the single highest
   risk of silently breaking Walras/homogeneity if done carelessly.
4. **The energy nest is the item a CGE-literate reviewer will check first.** Roadmap 5d.5 is
   explicitly flagged as "the single highest-value item... most likely to be asked about... if
   skipped." It is also the most invasive change: production goes from flat Leontief-intermediate
   + Cobb-Douglas/CES-VA (today: energy is just another row in `ax`, no different from steel or
   services) to a genuine nested structure. Every calibration formula in `calibrate.py`/
   `calibrate_open.py`/`calibrate_multi.py` that reads `ax`/`va_share`/`beta` flat must be revisited.

**De-risking strategy, in order:** government account first (5d.1) on the *existing* single-region
closed model only, proved against Tier 1 before touching open/multi-region — then savings/
investment (5d.2, same discipline) — then the energy nest (5d.5) as its own gated sub-phase, since
it is orthogonal to the institution-account work and can be developed and tested in parallel by a
second workstream if one exists. Labour market (5d.4) and the closure alternatives (5d.7) are
switch-only additions layered on top once 5d.1–5d.2 are solid. Do not attempt all seven sub-tasks
simultaneously in one branch — each must independently pass Tier 1 before the next is layered on,
exactly as Phase 5's own "2-sector-pilot-replicates gate" was the non-negotiable checkpoint before
scaling.

---

## 1. Architecture & where it plugs in

No new engine, no new `Engine` protocol implementation — 5d extends the existing `cge_static`
engine (closed/open/multi-region variants) in place. It touches:

```
src/cge/
├── contracts/data_objects.py       # SAM: no contract change (accounts already generic);
│                                    #   IOSystem/BuildMeta: unaffected
├── engines/cge_static/
│   ├── calibrate.py                #   AMEND: generalise institutions beyond one household
│   ├── calibrate_open.py           #   AMEND: same, open-economy variant
│   ├── calibrate_multi.py          #   AMEND: same, multi-region variant
│   ├── model.py                    #   AMEND: government + investment demand + labour closure
│   ├── model_open.py               #   AMEND: same
│   ├── model_multi.py              #   AMEND: same, per-region government/investment
│   ├── energy_nest.py              #   NEW: KL-E-M nest calibration + price/quantity functions,
│   │                                #     shared by all three model variants
│   └── engine.py                   #   AMEND: emit fiscal_balance, investment, wage/employment,
│                                    #     energy-nest substitution as native result variables
└── validation/suites/cge_static.py # AMEND: extend Tier 1/2 checks for the new accounts/nest
```

**No new packages.** Every sub-task is an amendment to an existing, well-tested module — this is
what "reopened debt" means in practice: the shape of the system (contract, engine wrapper, result
set, validation suite) was already built to hold this; only the model equations were incomplete.

**Institution-account generalisation (the shared plumbing all of 5d.1–5d.4 depends on):**
`calibrate()`'s current `hoh = [...]; if len(hoh) != 1: raise` becomes an explicit, named split:

```python
institutions: dict[str, str]  # role -> SAM account name, e.g. {"household": "hoh", "government": "gov"}
```

with `"household"` the only required role (preserves every existing single-institution SAM/test
fixture unchanged — they simply don't populate `"government"` or `"investment"`, and the model
falls back to today's behaviour exactly, which **is** the backward-compatibility test). This is a
**parameter to `calibrate()`**, not a SAM contract change — the contract's `accounts: list[str]`
already supports arbitrary named accounts; only the calibration code's hard-coded assumption that
there is exactly one non-factor account needs to go.

---

## 2. Sub-phase 5d.1 — Government / fiscal account (1–2 wk)

**Goal:** a tracked government balance sheet, not a same-period pass-through.

**Current state (confirmed by reading `model.py`/`model_open.py`/`model_multi.py`):** carbon
revenue is collected and redistributed to the household *in the same equation*, within the income
fixed point (`income = factor_income / (1 - k)` for closed; `base_income = factor_income +
er·foreign_savings` for open, then the same `/(1-k)` recycling). There is no persistent government
account, no government spending line, no other tax instrument, and no `fiscal_balance` variable in
any `ResultSet`.

**Design:**
- Add a `government` institution (SAM account, optional — see §1's `institutions` parameter).
  When absent, calibration behaves exactly as today (single-household, pass-through recycling) —
  this is the regression-safety net, not a special case to special-case around.
- When present, government income = carbon-tax revenue + any other modelled tax instrument
  (initially: carbon tax only — "other tax instruments" in the roadmap task description is a
  documented placeholder, not built out in 5d.1) + a calibrated benchmark government-consumption
  share of GDP (so the government's *benchmark* size is read off the SAM like every other account,
  not assumed).
- Government spending: at benchmark, a Cobb-Douglas or fixed-coefficient demand vector over
  commodities (mirrors household `gamma`), calibrated from the SAM's government row. Post-shock,
  government demand responds to government income under the chosen closure (see below).
- **Closure choice (first-class, documented, switchable — this is the actual deliverable the
  roadmap task names):**
  - `balanced_budget` (default): government spending is a residual that exactly exhausts
    government income each period — no deficit, no accumulation. This is the simplest generalisation
    of "recycling" and should be provably equivalent to today's `lump_sum` household recycling in
    the degenerate case (no separate government demand vector, 100% transferred to household) —
    **this equivalence is itself a regression test**.
  - `deficit_financed` (alternative, 5d.7's concern but declared here): government spending is
    fixed at its benchmark real level regardless of revenue; the gap is financed by a `fiscal_balance`
    that is reported, not closed. Requires deciding what absorbs a persistent deficit/surplus in a
    *static* (single-period) model — the honest answer is "nothing yet" (this is not a multi-period
    debt-accumulation model; that's Phase 7.1's job once 5d.3's capital identity exists). Document
    explicitly: `fiscal_balance` is a *reported* imbalance under this closure, not a modelled
    financing mechanism. This is the single biggest "don't overclaim" trap in 5d.1 — write it into
    the model doc in the same words used here.
- **New result variable:** `fiscal_balance` (government income − government spending; identically
  zero under `balanced_budget`, generally nonzero and reported under `deficit_financed`).
- **Reuse, don't replace:** the `_safe_denom` smooth-floor pattern extends directly — government
  income under `balanced_budget` recycling is structurally the same fixed-point shape as today's
  household recycling (`income = base / (1 - k)`), just with the recipient split between two
  institutions instead of one. The `strict`/non-strict divergence guard carries over unchanged.

**Decisions forced:**
- Government demand composition: Cobb-Douglas from SAM benchmark (recommended, mirrors household)
  vs. a simpler fixed-real-consumption assumption. Recommendation: Cobb-Douglas — no new
  behavioural assumption beyond what's already used for households.
- Whether "other tax instruments" (an output tax, an income tax) are in scope for 5d.1 or deferred.
  Recommendation: **defer** — carbon tax only in 5d.1; a generic ad-valorem output/sales tax is a
  natural 5d.7-or-later extension once the institution-account plumbing is proven, not a
  precondition for it.

**Testing (5d.1):**
- **Regression equivalence:** with no separate government demand vector and `balanced_budget`
  closure, results are bit-identical (to solver tolerance) to today's pre-5d.1 `lump_sum` recycling
  on every existing fixture — the degenerate-case proof that this is a strict generalisation.
- **Balance identity:** `fiscal_balance == 0` to `1e-9` under `balanced_budget` on every run,
  including under a carbon shock.
- **Deficit reporting:** under `deficit_financed`, `fiscal_balance` moves in the expected direction
  (revenue changes, spending fixed → nonzero, correctly signed balance) and is *documented*, not
  silently absorbed into another account (would violate Walras if it were).
- **Walras still holds** with the government account added — the extra institution must not
  introduce a hidden money leak. This is a Tier 1 non-negotiable re-run, not optional.
- **DoD 5d.1:** government account calibrates from a SAM that has one (optionally, falls back
  identically without one); `balanced_budget` and `deficit_financed` both switchable by config;
  `fiscal_balance` in every `ResultSet`; Tier 1 (replication/homogeneity/Walras) green on both
  closures.

---

## 3. Sub-phase 5d.2 — Savings and investment (1–2 wk)

**Goal:** investment as a genuine final-demand component with its own sectoral composition, not
folded into household consumption.

**Current state (confirmed):** `FD = cal.gamma * income / pq` is the *entire* final-demand vector
in every variant — there is no investment line. The "closed economy" identity `X = leontief @ FD`
treats 100% of final demand as household consumption.

**Design:**
- Add a `savings_investment` institution (SAM account — again, optional; SAM contract already
  names it generically). Benchmark investment demand = the SAM's savings/investment row, with its
  own sectoral composition (typically capital-goods-heavy, unlike household consumption) —
  calibrated as a fixed-coefficient or Cobb-Douglas vector exactly like government's, from the SAM.
- **Savings-investment identity (the closure, default per Phase 5's original spec):**
  total investment = household savings + government savings (`fiscal_balance` if positive,
  under whichever government closure is active) + foreign savings (`Sf`, already modelled as
  `cal.foreign_savings` in the open/multi-region variants). This is **savings-driven investment**:
  investment adjusts to whatever savings the model generates, not the reverse.
  - **Alternative closure (5d.7's concern, declared here):** `fixed_investment` — investment is
    held at its benchmark real level; households' savings rate is instead the adjusting residual.
    Standard CGE closure-alternative pairing (savings-driven vs. investment-driven), both must be
    switchable and tested.
- **Household savings:** currently 100% of factor income is spent (`gamma` sums to 1 over
  commodities, nothing held back). Introduce a household savings rate `s` (calibrated from the SAM
  if a savings/investment account exists and the household row shows a residual; otherwise `s = 0`,
  which is exactly today's behaviour — again the regression-safety fallback). Household consumption
  becomes `(1-s)·income`, with `s·income` flowing to the savings-investment account.
- **Goods-market closure change (the risk flagged in §0.3):** `FD` in `_quantities()` becomes
  `FD_household + FD_investment`, where `FD_investment`'s *total* is pinned by the savings-investment
  identity (a new equation) and its *composition* is the calibrated fixed vector. This adds exactly
  one new unknown (aggregate investment level; call it `INV`) and one new equation (the
  savings-investment identity) per model variant — the square-count must be re-derived and
  re-documented in `docs/models/cge-static.md`, mirroring §"well-posedness" from `phase-5-plan.md`.

**Decisions forced:**
- Savings-driven (default) vs. fixed-investment closure — both required, per Phase 5's original
  spec and roadmap 5d.7.
- Whether household savings rate is itself a calibrated constant (start here — simplest, consistent
  with Cobb-Douglas-everywhere-else) or elastic in income (defer — no clean elasticity source
  exists yet, same caution Phase 4 already flagged for demand elasticities generally).

**Testing (5d.2):**
- **Regression equivalence:** with no savings-investment account in the SAM (`s = 0`,
  `FD_investment = 0`), results are unchanged from pre-5d.2 behaviour — same fallback discipline as
  5d.1.
- **Savings-investment balance:** total investment = total savings (household + government +
  foreign) to `1e-9`, under the savings-driven closure, on every run.
- **Square-model re-check:** equation count == unknown count re-asserted (the specific new risk
  from §0.3) for closed, open, and multi-region variants independently.
- **Walras + homogeneity re-run** — Tier 1, non-negotiable, on all three variants with the new
  account active.
- **DoD 5d.2:** investment demand is a distinct `ResultSet` line with its own sectoral breakdown;
  both closures switchable; Tier 1 green on closed/open/multi-region.

---

## 4. Sub-phase 5d.3 — Capital accumulation, depreciation, premature retirement (1 wk, shared with 7.1)

**Goal:** the actual mechanism Phase 7.1's recursive-dynamic wrapper needs to update capital
between static solves — today there is nothing real to call.

**Design:**
- A capital stock `K_t` per region-sector (or per region, if sector-level capital detail isn't
  warranted — recommend region-level to start, matching the existing single aggregate capital
  factor in `factors`), updated by:
  $$K_{t+1} = (1-\delta)\,K_t + INV_t$$
  where `INV_t` is 5d.2's investment level (by sector, aggregated to whatever granularity `K`
  is tracked at) and `δ` is a calibrated or assumed depreciation rate (documented default, e.g.
  5%/yr, cited).
- **This sub-phase is deliberately thin within 5d itself** — it is the *identity*, not the dynamic
  wrapper (that's Phase 7.1's job). 5d.3's scope is: the identity exists, is unit-tested in
  isolation (given `K_t`, `INV_t`, `δ`, produces `K_{t+1}` correctly), and is exposed as a function
  Phase 7.1 can call — not a full multi-year simulation loop.
- **Premature retirement (stranded-asset) option:** a documented, switchable extra term — a shock
  can mark a fraction of `K_t` as retired before its natural depreciation (e.g. a carbon shock
  stranding fossil capital). Modelled as an exogenous scenario input (a retirement fraction per
  sector per period), not an endogenous investment decision — endogenous stranding (capital exits
  because its expected return falls below a threshold) is out of scope here and should be named as
  a documented limitation, not silently assumed away.

**Decisions forced:**
- Capital tracked at region level or region-sector level. Recommendation: region level first
  (matches the current single aggregate capital factor per region) — region-sector capital
  requires a capital-mobility-across-sectors assumption that's a bigger design decision than 5d.3's
  budget warrants; defer to a future extension if a reviewer asks for sector-specific vintage capital.
- Depreciation rate: single global default vs. per-region/sector. Recommendation: single documented
  default (e.g. 5%), overridable per scenario — consistent with how elasticities are handled
  (central default + documented override, not per-sector guesswork).

**Testing (5d.3):**
- **Identity correctness:** given known `K_t`, `INV_t`, `δ`, `K_{t+1}` matches the hand-computed
  value exactly (analytic known-answer, same discipline as Phase 5's 2-sector pilot).
- **Non-negativity:** `K_{t+1} ≥ 0` is enforced or explicitly flagged if a retirement fraction would
  drive it negative (a scenario-input validation, not a silent clamp — mirrors the `ElasticitySet`
  validator's "reject invalid inputs at the boundary" pattern).
- **Retirement option exercised:** a test scenario with nonzero retirement fraction produces a
  strictly lower `K_{t+1}` than the same scenario without it, all else equal.
- **DoD 5d.3:** `K_{t+1} = f(K_t, INV_t, δ, retirement)` is a standalone, unit-tested function;
  documented depreciation default with citation; Phase 7.1 can call it directly (no redesign needed
  when 7.1 starts).

---

## 5. Sub-phase 5d.4 — Labour market: employment/unemployment and wages (1–2 wk)

**Goal:** a documented closure alternative to today's "full employment, wage adjusts" default.

**Current state:** factor markets clear with a fixed endowment (`cal.endowment`) and a flexible
wage — full employment by construction, for every factor including labour, in every variant.

**Design:**
- **Default (unchanged):** flexible wage, full employment — this remains the baseline closure and
  needs no new code, only explicit documentation that it *is* a closure choice, not a structural
  fact about the labour market.
- **New alternative: wage-floor / wage-curve closure.** Labour supply is still fixed (no
  demographic labour-supply response — that's out of scope), but the wage is bounded below (a
  floor, e.g. a minimum-wage-like constraint) or follows a wage curve (wage responds to
  unemployment with a calibrated elasticity, [Blanchflower-Oswald]-style). When the floor binds,
  labour demand falls short of supply — the gap is reported as `unemployment` (a new `ResultSet`
  variable), and the labour market-clearing equation is replaced by a complementarity condition
  (wage = floor when demand < supply; wage floats above the floor and clears exactly when demand ≥
  supply at the floor wage).
- **Implementation approach:** a complementarity condition changes the residual system's structure
  (it's no longer a smooth equation everywhere) — the practical approach is a documented
  **regime-switch**: solve under full employment first; if the resulting wage would fall below the
  floor, re-solve with wage fixed at the floor and labour demand (not supply) determining employment,
  reporting the shortfall as unemployment. This avoids a genuine mixed-complementarity solver
  (out of scope — the existing scipy/least-squares solver doesn't support MCP natively) while still
  producing the right economics for the documented case.

**Decisions forced:**
- Wage floor (simple, one parameter) vs. wage curve (calibrated elasticity, more literature
  grounding but another elasticity-sourcing problem Phase 4 already flagged as hard). Recommendation:
  **wage floor first** — the regime-switch approach above is exact and simple for a floor; a wage
  curve can be layered on later as a documented extension once the floor mechanics are proven.
- Whether unemployment applies per-region-factor (multi-region) independently. Recommendation: yes
  — each region's labour market is a separate market in the existing model structure; the floor/
  curve closure is a per-region-factor config choice, not global.

**Testing (5d.4):**
- **Regression equivalence:** with no floor configured (or floor set below the market-clearing
  wage), results are identical to today's full-employment closure.
- **Floor binds correctly:** a scenario constructed so the equilibrium wage would fall below a
  configured floor produces `unemployment > 0` and wage exactly at the floor, to solver tolerance.
- **Walras re-check under the regime switch:** the dropped-equation accounting must still balance
  when the labour market is not a smooth clearing condition — this is the trickiest Tier 1 re-proof
  in 5d.4 and should get its own dedicated test, not an assumption that Walras "still obviously holds."
- **DoD 5d.4:** wage-floor closure switchable by config; `unemployment` in `ResultSet` when active;
  Tier 1 green under both the default and floor-active closures.

---

## 6. Sub-phase 5d.5 — A genuine energy nest, KL-E-M (2–3 wk)

**Goal:** energy separated from other intermediates so a carbon price can shift substitution
*within* the energy bundle (e.g. toward electricity, away from fossil fuels), not just across
sectors. Flagged by the roadmap as **the single highest-value item** in 5d, and the piece Phase 7b
needs to harmonize NGFS energy-transition pathways against.

**Current state (confirmed by reading `calibrate.py`):** production is flat Leontief in
intermediates (`ax[j,i]`, energy is just one more row like any other sector) and Cobb-Douglas/CES
in value-added only (`beta`/`va_ces_share` over factors, no energy). There is no nest.

**Design — nested nomenclature, following [Hosoe2010]'s standard KL-E-M structure:**

$$
X_i = A_i\Big[\delta_i\,\mathrm{KLE}_i^{\rho_i} + (1-\delta_i)\,\mathrm{M}_i^{\rho_i}\Big]^{1/\rho_i}
\tag{outer nest: value-added-plus-energy vs. other materials}
$$
$$
\mathrm{KLE}_i = \Big[\delta^{KL}_i\,\mathrm{KL}_i^{\rho^{KL}_i} + (1-\delta^{KL}_i)\,E_i^{\rho^{KL}_i}\Big]^{1/\rho^{KL}_i}
\tag{middle nest: capital-labour composite vs. energy composite}
$$
$$
\mathrm{KL}_i = A^{KL}_i\prod_f F_{f,i}^{\beta_{f,i}}, \qquad
E_i = A^{E}_i\Big[\textstyle\sum_{e} \delta_{e,i}\,E_{e,i}^{\rho^{E}_i}\Big]^{1/\rho^{E}_i}
\tag{innermost: KL Cobb-Douglas/CES as today; E a CES over energy commodities}
$$

- **Which commodities are "energy"** must be declared per build (a config-level tag on the
  `sectors` list — e.g. `energy_sectors: list[str]` passed to `calibrate()`), not inferred from
  names. This is the same discipline as the existing `institutions` role-mapping in §1: explicit,
  not string-matched.
- **Calibration:** at benchmark (all prices = 1), the nest's scale/share parameters are backed out
  exactly as `va_share`/`beta`/`av` are today, just with an extra layer — the energy-vs-materials
  split share `δ_i` from the SAM's energy-commodity intermediate flows vs. non-energy, and the
  within-energy CES shares `δ_{e,i}` from the relative energy-commodity flows. New module
  `energy_nest.py` holds this calibration logic and the corresponding price/quantity dual functions
  (unit cost of `KLE`, unit cost of `E`, Shephard demand for each energy commodity), called from
  `calibrate.py`/`calibrate_open.py`/`calibrate_multi.py` rather than duplicated three times.
- **Carbon price mechanism:** the carbon cost `cc[i]` currently enters as a flat per-unit add-on to
  the zero-profit price (`pz = intermediate_cost + va_cost + cc`). With the energy nest, carbon cost
  should attach to the *energy* commodities specifically (it's an emissions tax on fossil energy
  use), raising the price of `E_i` and inducing substitution toward `KL_i` (away from energy) and
  within `E_i` toward lower-carbon energy commodities — this is the entire point of 5d.5 and must be
  demonstrated as a Tier 2 sign/direction test, not just claimed.

**Decisions forced:**
- Nest elasticities (`ρ_i`, `ρ^{KL}_i`, `ρ^E_i`): no clean open elasticity database exists for
  these (Phase 4 already flagged this general problem for demand/Armington elasticities). Start
  with published central-case defaults from [Hosoe2010] or similar CGE literature, documented and
  overridable, exactly as the existing Armington/CET elasticities are handled today (`ElasticitySet`
  with low/central/high bands and per-value sourcing).
- Whether the outer nest is materials-vs-KLE or a different grouping (e.g. some CGEs nest energy
  directly with capital, "KE-L" instead of "KL-E"). Recommendation: **KL-E-M** as specified — it's
  the standard in the cited literature and is what the roadmap explicitly names.
- Backward compatibility: builds/SAMs with no `energy_sectors` declared must fall back to today's
  flat-Leontief behaviour exactly (same regression-safety discipline as every other 5d sub-task).

**Testing (5d.5):**
- **Regression equivalence:** with no `energy_sectors` declared, results are unchanged from
  pre-5d.5 behaviour.
- **Nest calibration replication:** with the nest active and zero shocks, benchmark replication
  still holds to machine precision (Tier 1, non-negotiable) — the new nest must reproduce the exact
  same benchmark SAM flows it was calibrated from.
- **Sign/direction under a carbon price (Tier 2, the actual deliverable):** a carbon shock reduces
  energy's share of the KLE composite (substitution away from energy) and, within the energy
  composite, shifts share toward the lowest-carbon-intensity energy commodity — both directional,
  published-consensus signs, encoded as a validation check exactly like the existing "dirty-sector
  output falls" Tier 2 test in `docs/phase-5-plan.md` §7.
- **Homogeneity and Walras re-run** with the nest active, on closed/open/multi-region.
- **DoD 5d.5:** `energy_nest.py` calibrates and prices the KL-E-M structure; `energy_sectors`
  config is required to activate it (opt-in, defaulting to flat-Leontief); Tier 1 green; the
  carbon-price substitution-toward-non-energy sign test passes and is documented in
  `docs/models/cge-static.md`.

---

## 7. Sub-phase 5d.6 — Adaptation and transition investment (1 wk)

**Goal:** a documented shock/response channel where a scenario can specify adaptation or
transition capital expenditure that competes with 5d.2's investment demand, reported as its own
line item.

**Design:**
- A new scenario-input type: `adaptation_investment: dict[sector, float]` (or a scalar plus a
  default sectoral allocation) — an exogenous addition to investment demand, distinct from 5d.2's
  endogenous savings-driven investment.
- **Competes with, not adds to:** under the savings-driven closure, total investment is pinned by
  total savings (§5d.2). If a scenario specifies adaptation investment, it must be financed from
  the same savings pool — i.e., it *crowds out* other investment unless savings also rise (e.g.
  from a shock that raises government revenue, which could be earmarked). This crowding-out
  mechanic is the actual economics 5d.6 needs to get right, not just "add a number to investment."
  Document the default assumption explicitly (recommend: full crowding-out under savings-driven
  closure — a scenario reports both the adaptation-investment line and the resulting reduction in
  other investment, not a free lunch).
- **Reported separately:** `ResultSet` gets an `adaptation_investment` line distinct from total
  investment, so a report can show "X was spent on adaptation, and Y of ordinary investment was
  displaced" rather than a single blended number.

**Decisions forced:**
- Full crowding-out (recommended, honest default) vs. an assumption that adaptation investment is
  "additional" (implicitly assumes new financing from outside the model — a bigger claim that
  should require an explicit closure choice, not a default).

**Testing (5d.6):**
- **Crowding-out identity:** total investment (adaptation + other) still equals total savings to
  `1e-9` under the savings-driven closure — this is 5d.2's identity, just with a second demand
  component, and must not be allowed to violate it.
- **Reported separately:** the two investment lines are distinct in `ResultSet` and sum to the
  total.
- **DoD 5d.6:** adaptation investment is a scenario input; crowds out other investment under the
  documented default closure; reported as its own line item.

---

## 8. Sub-phase 5d.7 — Alternative external-balance and fiscal closures (1 wk)

**Goal:** beyond the fixed-trade-balance / lump-sum-recycling defaults, a flexible-trade-balance
option and a deficit-financed government closure (already designed in 5d.1), each documented and
switchable, with the standard correctness battery re-run under each.

**Design:**
- **Flexible trade balance** (open/multi-region variants only): today `Sf` (foreign savings) is
  fixed exogenously. The alternative lets the real exchange rate `er` adjust to a target trade
  balance instead — a standard CGE closure swap (fixed-Sf/flexible-er vs. flexible-Sf/fixed-er).
  This is a genuinely new unknown/equation pair per open/multi-region variant and needs its own
  square-count re-derivation, same discipline as 5d.2.
- **Deficit-financed government** — already specified in 5d.1; this sub-task is where it gets its
  full correctness battery re-run (5d.1 declares the closure exists; 5d.7 is where "each documented
  and switchable... with the standard correctness battery re-run under each" from the roadmap task
  description is actually discharged for both new closures together).

**Testing (5d.7):**
- **Flexible-trade-balance square-count:** re-derived and asserted for open/multi-region.
- **Both closures independently pass Tier 1** (replication/homogeneity/Walras) — this sub-task's
  entire job is proving the *alternatives* are as correct as the defaults, not just switchable.
- **DoD 5d.7:** flexible-trade-balance and deficit-financed closures both switchable by config;
  Tier 1 green under each; `docs/models/cge-static.md` documents all four closure combinations
  (trade balance × government) as a 2×2 table of what's implemented and defaulted.

---

## 9. What is explicitly out of scope for 5d (don't overclaim)

- **An absolute price level / inflation.** Already corrected in the roadmap (round-10 remediation):
  none of 5d.1/5d.2/5d.5 introduce a money-demand equation, a central-bank rule, or a fixed money
  stock — the CPI-numéraire closure has no mechanism for an absolute price level. 5d can report a
  **relative cost-of-living / GDP-deflator index**, never "inflation."
  - **Interaction with 5d.5 (energy nest):** the energy nest introduces new relative-price movement
    *within* the CPI basket (energy commodities move relative to non-energy), which changes the
    deflator's decomposition (how much of the deflator's movement is "energy" vs. "everything
    else") but still does not create an absolute price level — the numéraire is still the CPI, no
    new degree of freedom is added. State this explicitly in the model doc so a reviewer doesn't
    mistake "the energy nest changes relative prices" for "the model now has inflation."
- **Multi-period debt/capital dynamics.** 5d.3 provides the *identity*; the actual recursive
  wrapper that calls it year-over-year is Phase 7.1, not 5d. A `deficit_financed` government closure
  (5d.1/5d.7) reports an imbalance but does not accumulate debt across periods within 5d's own scope.
- **Endogenous labour supply or migration.** 5d.4's labour market closure is about
  employment/wages given a *fixed* labour endowment; demographic or migration-driven labour-supply
  change is out of scope (as it already is for factor endowments generally).
- **Endogenous capital stranding.** 5d.3's premature-retirement option is an exogenous scenario
  input, not a modelled investment decision responding to expected returns.
- **Sector-specific energy-technology detail** (e.g. a full electricity-generation-mix
  sub-model). 5d.5 nests energy as a CES composite over energy *commodities* already present in
  the SAM's sector list — it does not add new technology-level detail beyond what Phase 4's build
  already carries. That is Phase 7b.5's job (energy-mix/technology-cost pathways), which explicitly
  depends on 5d.5 existing first.

---

## 10. Testing & validation strategy (extends Phase 5's Tier 1/2/3)

Every sub-task above names its own DoD tests; this section is the cross-cutting discipline that
ties them together, following `docs/phase-5-plan.md` §7's structure exactly.

### Tier 1 — non-negotiable, re-run after every sub-task, on closed/open/multi-region independently

1. **Benchmark replication** — still machine precision, with the new account/nest active and
   zero shocks.
2. **Homogeneity of degree zero** — still holds with government, investment, and the energy nest
   in the model.
3. **Walras' law** — the hardest re-proof in this phase (§0.3), because two sub-tasks (5d.2's new
   investment unknown, 5d.4's regime-switch labour closure) change the equation/unknown count or
   smoothness. Each must get its own explicit re-derivation, documented in `docs/models/
   cge-static.md`, not an assumption that it "should still hold."

### Tier 2 — economic sense, specific to what 5d adds

4. **Revenue neutrality under `balanced_budget`** — `fiscal_balance == 0` to `1e-9`.
5. **Savings-investment balance** — total investment == total savings to `1e-9`.
6. **Energy-nest substitution direction** — carbon price shifts share away from energy within KLE,
   and toward lower-carbon energy commodities within E.
7. **Wage-floor/unemployment direction** — a binding floor produces positive, correctly-signed
   unemployment, never negative.
8. **Crowding-out identity (5d.6)** — adaptation investment displaces other investment 1-for-1
   under the default closure.

### Tier 3 — robustness, regression, documentation

9. **Every sub-task's fallback is a byte-for-byte regression test against pre-5d.n behaviour** —
   this is 5d's version of Phase 5's "hand-checkable toy" discipline: instead of a hand-derived
   known-answer, the known-answer *is* the already-validated pre-5d model, reached by construction
   when the new institution/nest/closure is inactive. This is unusually strong regression coverage
   available specifically because 5d is additive to an already-correct model — use it fully.
10. **Model doc ↔ code consistency** — `docs/models/cge-static.md` updated per sub-task (not
    batched to the end), each closure/nest/institution's assumptions matching the manifest exactly,
    as enforced for every engine so far.
11. **Solver-status gate** — unchanged discipline; a non-optimal termination still raises, never
    returns numbers, regardless of which sub-task added complexity.

---

## 11. Effort, sequencing & dependencies

| Sub-phase | Deliverable | Effort |
|---|---|---|
| 5d.1 Government/fiscal account | institution-account plumbing, balanced/deficit closures, `fiscal_balance` | 1–2 wk |
| 5d.2 Savings and investment | investment demand, savings-driven/fixed closures, square-count re-derivation | 1–2 wk |
| 5d.3 Capital accumulation | depreciation/investment identity, premature retirement, shared with 7.1 | 1 wk |
| 5d.4 Labour market | wage-floor/curve closure, unemployment reporting, regime-switch Walras re-proof | 1–2 wk |
| 5d.5 Energy nest (KL-E-M) | `energy_nest.py`, calibration + carbon-price substitution mechanism | 2–3 wk |
| 5d.6 Adaptation/transition investment | crowding-out mechanic, separate reporting line | 1 wk |
| 5d.7 Alternative closures | flexible trade balance, deficit-financed re-proof, 2×2 closure table | 1 wk |
| **Total** | **the full macro closure: government, investment, capital, labour, energy nest** | **8–12 wk solo FTE** |

**Critical path (do not skip in order):** `5d.1 government account (Tier 1 green)` → `5d.2
savings/investment (Tier 1 green, square-count re-derived)` → `5d.3 capital identity (unit-tested,
standalone)` in parallel with `5d.5 energy nest (Tier 1 + Tier 2 sign test green)` → `5d.4 labour
market` → `5d.6 adaptation investment (needs 5d.2's investment demand to compete with)` → `5d.7
alternative closures (needs 5d.1 + 5d.2's default closures to have alternatives to)`.

5d.3 and 5d.5 are the two sub-tasks with no dependency on each other or on 5d.4/5d.6/5d.7 — genuine
parallel-workstream candidates if a second person is available (this compresses **elapsed** time
only; total solo-FTE work is unchanged, per roadmap.md §2's FTE-vs-elapsed-time convention).

**Depends on:** Phase 5 (5.0–5.3, done — this plan amends that model, doesn't rebuild it).
**Unblocks:** Phase 7.1 (needs 5d.3's capital-accumulation identity to update capital between
recursive-dynamic solves — currently has nothing real to call); Phase 7b (needs 5d.1's government
account and 5d.5's energy nest to harmonize NGFS fiscal/energy-transition pathways against).

---

## 12. Honest expectations

- **What 5d will genuinely deliver:** the macro closure the roadmap has been carrying as debt since
  Phase 5 was marked complete — a real government account with a documented, switchable financing
  closure; investment as a modelled demand component with a savings-investment identity; the
  capital-accumulation identity Phase 7.1 needs; a labour-market closure alternative to permanent
  full employment; and the energy nest that makes carbon-price substitution economically credible
  rather than a flat pass-through. The standard scenario output set (GDP/GVA/consumption/
  investment/employment/wages/fiscal-balance/capital-returns) named in the roadmap's Phase 5d entry
  becomes real, native model output rather than aspirational.
- **What it will not be:** a multi-period dynamic model (that's Phase 7.1, built on 5d.3's identity)
  or a model with an absolute price level / genuine inflation (no mechanism for that exists or is
  added here — see §9). It will also not have sector-specific energy-technology detail beyond what
  the SAM's commodity list already carries (that's Phase 7b.5).
- **The phrase to hold onto, extended:** Phase 5's "precise about costs, indicative about volumes,
  transparent about assumptions" now also means *transparent about closures* — every one of 5d's
  new mechanisms (government financing, investment closure, labour market, trade balance) is a
  documented, named, switchable assumption, not a hidden default. A reviewer should be able to read
  one table (the 2×2 closure table from 5d.7, plus the energy-nest activation flag) and know exactly
  which macro-closure variant produced a given result.

## 13. New references needed (add to `docs/references.md` when work starts)

- **[Hosoe2010]** — already listed; the KL-E-M nest structure and standard closure-alternative
  pairing (savings-driven vs. fixed investment; fixed vs. flexible trade balance) both follow this
  text directly.
- **To add:** a standard reference for wage-curve/wage-floor labour-market closures in CGE models
  (e.g. Blanchflower-Oswald-style wage curve) for 5d.4, if the wage-curve alternative (not just the
  floor) is pursued.
- **To add:** 1–2 published CGE studies with an explicit government/fiscal account and energy nest,
  for the Tier-3 literature-bracket comparison once 5d.1/5d.5 are built (extending Phase 5's own
  Tier-3 literature-bracket discipline to the new accounts).
