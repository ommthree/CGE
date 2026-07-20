# Plan — energy-price inputs & temperature-target back-solving

**Status: Feature 1 (energy prices) IMPLEMENTED; Feature 2 (temperature-target back-solve) is a
Phase-7 plan.** Two requested features. The energy-price input is built (see the Feature 1 section
below for the as-built note); the temperature back-solve is a well-shaped Phase 7 feature. Both
slot into the contract seams (the typed shock vocabulary and the module slots, see
`docs/overview.md` §2) without changing the existing engines' internals.

---

## Feature 1 — Country-level energy prices as optional inputs (near-term; low risk) — ✅ IMPLEMENTED

**Status: implemented** in Engine 1 (`io_price`) and available to Engine 2 (`partial_eq`) via the
shared price step, and **exposed in the GUI Run page**. Interpretation (1) — a rise in the carrier's
*output price* — is built as an **exogenous price pin** (the carrier's Δp equals the request
exactly; a +30% run gives +30% on the carrier, not the +35.9% an earlier cost-wedge form produced);
downstream users still pick up the propagated price. Interpretation (2) is documented as
available-if-needed but not implemented. See `docs/models/io-price-model.md` §5a for the
equation-level method, `examples/energy_price_io.yaml` for a runnable scenario, and the
`energy_price_pins_carrier_exactly` / `carbon_energy_pinned_carrier` validation checks. Below is the
original design; note the cost-wedge framing there has been superseded by the price-pin form above.

### What it is
An **exogenous energy-price change**, per country and energy carrier (coal / oil-gas /
electricity), added *optionally and additionally* to a scenario. E.g. "in addition to a €100/t
carbon price, apply a +30% electricity price in Germany and +15% in France."

### Why it's easy — it's the same shape as a carbon cost
Engine 1 already turns a carbon price into a per-unit cost on a sector's output and propagates
it through the supply chain via the Leontief inverse: `Δp = (I − Aᵀ)⁻¹ · c`, where `c = τ·e`.
An energy-price change is *another cost vector* `c` — a proportional cost increase on the energy
carriers' output — that adds to the carbon term and propagates identically. The whole
propagation machinery is reused; the only new code is assembling the energy cost vector.

This is standard IO price-side pass-through of a cost shock [MillerBlair2009, §2.4]; the broader
economics of how energy-price shocks propagate is surveyed by Kilian [Kilian2008]. Cost-push
pass-through gives an *upper bound* on the price effect (as for the carbon price) — the CGE
relaxes it with substitution.

### Design
- **New shock type `EnergyPrice`** in the shock vocabulary (contract 2), e.g.:
  ```yaml
  - type: energy_price
    carrier: electricity        # coal | oil_gas | electricity
    change: 0.30                # +30% (fractional) — or an absolute €/unit form
    coverage_regions: [DE, FR]  # per-country; empty = all
    path: {2025: 0.0, 2035: 0.30}   # optional time path, like every shock
  ```
- **Cost-vector assembly (Engine 1 / Engine 2).** For each `EnergyPrice`, add to the carbon
  cost vector the term "energy carrier's baseline cost share × change × coverage mask." Because
  the engines already sum independent shock contributions before the Leontief solve, this
  composes with `CarbonPrice` for free — a scenario can carry both.
- **The one modelling decision to pin down: scope.** Two distinct things, and the doc/UI must be
  explicit about which:
  1. **A rise in the energy *sector's output price*** — i.e. electricity/oil-gas becomes more
     expensive to buy. This propagates to every good in proportion to how much energy it uses
     (directly and upstream). This is the natural, well-defined interpretation and the one to
     build first.
  2. **A rise in *every sector's energy input cost*** — a targeted cost shock scaled by each
     sector's energy purchase share. Overlaps with (1) if applied to the same carrier; only
     needed if energy-input taxes differ from output-price changes. Defer unless required.
  Recommendation: implement (1); document (2) as available-if-needed.
- **In the CGE (Phase 5), it's richer:** an energy-price change shifts substitution in the
  KL-E-M production nest (firms use less of the now-costlier carrier). Same shock, but the CGE
  gives a substitution response the IO engines cannot. No new shock type — the CGE just
  interprets `EnergyPrice` through its nest.

### Data
Energy carriers are already identifiable in EXIOBASE (the coarse map has `energy_coal`,
`energy_oil_gas`, `electricity`). The "baseline cost share" of each carrier per sector comes
straight from the A-matrix column — no new data needed for interpretation (1).

### Testing & validation
- Sign/proportionality: a +x% carrier price raises energy-intensive goods' prices, more so for
  higher energy use; doubling the change doubles the effect (linearity).
- Composition: a scenario with `CarbonPrice` + `EnergyPrice` equals the sum of the two run
  separately (independent shocks add — the same property tested for multi-gas carbon shocks).
- Coverage: a country-restricted energy price leaves other countries' direct costs untouched.
- Known-answer on the toy economy, as for every engine feature.

### Effort & placement
**~3–5 days**, folded into the engines as an incremental feature (a new shock + cost-assembly
branch + validation checks). Best done alongside or just after any Engine-1/2 revisit; it does
not block the CGE. Fits entirely within the existing architecture.

---

## Feature 2 — Temperature target → back-solve the carbon price (Phase 7)

### What it is (Interpretation A, confirmed)
Fix a **temperature (or cumulative-emissions) target** and **invert** the forward model to find
the **carbon-price path** consistent with it — then run the engines forward along that path to
report the sector cost/volume impacts.

```
Target: stay under 1.75°C by 2050
    ↓  back-solve (invert the forward chain)
Implied carbon-price path: €40 (2030) → €180 (2050)
    ↓  forward run (engines)
Sector cost / volume impacts along that path
```

This is the **credible, buildable** version of "make it IAM-ish": the model traces the sectoral
consequences of a target, using a standard climate emulator for the emissions→temperature link.
It is *not* the contested damage-feedback problem (see the caveat at the end).

**How this relates to competing methodologies.** Full process-based IAMs — GCAM [GCAM], REMIND
[REMIND], MESSAGE — solve for cost-optimal transition pathways with explicit energy systems and
(often) intertemporal optimisation. This platform deliberately does **not** do that; it consumes
the pathways such models produce (via NGFS [NGFS]) and adds sector/supply-chain resolution.
Target-driven back-solving is the *inverse* of what those IAMs do internally, but done as a
light outer loop on our forward model rather than a full optimisation — a standard trade-off in
financial-sector "IAM-based" tools. The carbon-budget form of the target rests on the well-
established near-linearity of warming in cumulative CO₂ [MatthewsCarbonBudget; IPCC_AR6_WG3].

### The forward chain to invert
1. **Carbon price → emissions.** The engines already produce sectoral outputs under a carbon
   price; emissions follow from output × emission intensity. (Engine 2's volume response makes
   this behavioural — a higher price lowers dirty output, lowering emissions. The CGE makes it
   general-equilibrium with substitution.)
2. **Emissions → temperature.** Via **FaIR** [Leach2021] — an open, IPCC-used reduced-form
   climate emulator — behind the existing **climate module slot** (contract 5, already defined:
   `ClimateModule: emissions → temperature`). This is the one-way coupling already planned for
   Phase 7.3.

The forward chain is **plausibly monotone in aggregate** — a uniformly higher carbon price tends
to lower emissions, which lowers temperature — but this is an empirical property to check per
scenario, not a guarantee: revenue recycling, cross-region leakage, sector substitution, and
rebound effects can all push a *specific* price change in a direction that isn't simply "more
price, less emissions" once general-equilibrium responses are in play. That distinction matters
for what "the inversion" can honestly mean.

### The inversion (the new bit) — corrected 2026-07 (review P1)

**Two target forms, both a single scalar constraint:**
- **peak/end temperature** (e.g. "≤ 1.75°C in 2050"): scale the price path so the FaIR-projected
  temperature at that year (or its peak) stays at or under the cap;
- **carbon budget** (cumulative emissions ≤ B): often cleaner numerically, since temperature is
  near-linear in cumulative CO₂ [MatthewsCarbonBudget; IPCC_AR6_WG3].

**A single terminal temperature (or carbon-budget) target is ONE scalar constraint. A full
multi-year carbon-price path has one degree of freedom per period.** Recovering an entire path
from one target is underdetermined — infinitely many price paths can hit the same terminal
temperature, because per-year temperatures are **dynamically coupled through cumulative
emissions and climate inertia** (FaIR's temperature in year *t* depends on the whole emissions
history up to *t*, not on that year's price in isolation). Treating this as "a 1-D root-find per
period" — as an earlier version of this plan and the roadmap summary both said — is not a
well-posed decomposition: the per-period equations are not independent scalar roots to solve one
at a time.

There are exactly two honest ways to make this a scalar root-find, and the design must pick one
explicitly rather than leave the shape unstated:

1. **Scale a predetermined path shape by one scalar.** Fix the *shape* of the price path from an
   external source (e.g. an NGFS scenario's own price trajectory, or a documented functional form
   like a constant real growth rate), and solve for a single multiplier that scales the whole
   shape so the resulting terminal temperature (or cumulative-emissions budget) hits the target.
   This **is** a genuine 1-D root-find — one scalar, one constraint — and the shape choice is a
   first-class, documented assumption (which shape, and why) surfaced in the scenario's
   provenance, not an implementation detail.
2. **Constrained optimisation with an explicit objective**, if the path's *shape* should also be
   chosen by the solver rather than fixed exogenously: minimise a stated objective (least
   cumulative cost, path smoothness, proximity to a reference NGFS trajectory, a cap on
   maximum year-over-year price change, or some documented combination) subject to hitting the
   temperature/budget target. This is a genuinely harder numerical problem than a root-find — it
   needs a real optimiser (not bisection/Brent), an explicit objective the user can see and the
   GUI can label, and its own convergence/feasibility diagnostics.

**Recommendation: build (1) first.** It is the buildable, honestly-scoped version — a real 1-D
root-find, cheap, and consistent with this platform's "toy but honest" discipline. (2) is a
documented future extension, not part of this task's scope, and should not be described as "a
1-D root-find" anywhere it's mentioned.

- **Monotonicity is now an empirical precondition, checked per scenario, not assumed.** Before
  accepting a scaling-factor root as the answer, the back-solve must verify the forward map
  (scale factor → terminal temperature) is monotone over the search bracket for *this* scenario's
  recycling mode, leakage pattern, and elasticities — if it isn't, the root-finder's bracket
  invariant doesn't hold and the driver must say so rather than silently accepting whatever
  root-finder output it gets.
- The outer loop calls the (recursive-dynamic, Phase 7.1) engine forward each iteration, scaling
  the fixed-shape path by the trial multiplier. Cost: a handful of forward solves — cheap on the
  small build.
- **Guardrails (carry the platform's discipline):** if no scale factor within a sane bound hits
  the target (e.g. the target is infeasible given the economy's structure, or the empirical
  monotonicity check fails), the solver **reports infeasibility (or non-monotonicity) clearly**
  rather than returning a garbage price — the same "never return meaningless numbers" rule as
  Engine 1's well-posedness guard.

### Design placement
- **Depends on:** Phase 7.1 (recursive-dynamic wrapper) and 7.3 (FaIR climate module). The
  climate module slot already exists as a contract; FaIR is its first implementation.
- **New component:** a `TemperatureTarget` scenario option carrying **both** the target (peak/end
  temperature or cumulative-emissions budget) **and** a required `path_shape` reference (an NGFS
  scenario's own price trajectory, or a documented parametric shape — e.g. constant real growth
  rate — with its parameters) — the shape is not optional or inferred, since it's what makes the
  problem a 1-D root-find at all. A `back_solve` driver wraps the forward run in the root-finder,
  scaling `path_shape` by the trial multiplier each iteration, and runs the empirical
  monotonicity check before accepting a root. Emits, alongside the usual `ResultSet`, the
  **implied scale factor and resulting carbon-price path** and the **resulting temperature
  path** — both provenance-tagged, including which `path_shape` was used and why.
- **Reuses everything else:** the engines, the shock vocabulary (the back-solved price becomes a
  `CarbonPrice` time path), the result schema.

### Testing & validation
- **Round-trip identity (the key test):** back-solve a scale factor for target T* against a fixed
  `path_shape`, then run forward with the resulting price path — the forward temperature must
  equal T* to tolerance. This is the CGE-replication analogue: it proves the inverter and the
  forward model are mutually consistent.
- **Empirical monotonicity, checked not assumed:** verify (and test, per scenario configuration —
  recycling mode, elasticities) that scaling `path_shape` up strictly lowers the resulting
  terminal temperature over the search bracket; the test suite should include at least one
  documented case (e.g. a recycling mode or elasticity setting) where monotonicity is checked and
  holds, so the guard is exercised, not just present in code.
- **Feasibility / non-monotonicity guard:** an impossible target, or a scenario configuration
  where the monotonicity check fails, raises/reports clearly and never returns a fabricated
  scale factor.
- **Climate emulator sanity:** FaIR reproduces a known emissions→temperature response (e.g. a
  published SSP scenario's temperature) within tolerance — the climate module's own known-answer.
- **NGFS cross-check:** the implied price path for a target should be in the ballpark of NGFS
  scenarios reaching similar temperatures (a directional bracket, not a benchmark — the same
  honesty as the EXIOBASE live gate).

### Effort & difficulty
**~2–3 wk on top of Phase 7.1 + 7.3** (revised from ~1–2 wk — review P1: the corrected design
needs a documented `path_shape` mechanism, an empirical monotonicity check with its own test
coverage, and infeasibility/non-monotonicity reporting, none of which existed in the original
"just a root-find loop" framing). The scale-factor root-find itself is still small; the added
work is making the shape choice and the monotonicity precondition explicit and tested rather than
assumed. The prerequisites (recursive dynamics + FaIR) remain the larger cost and were already in
the Phase 7 plan. **Difficulty: medium.** The constrained-optimisation alternative (letting the
solver also choose the path shape, not just scale a fixed one) is out of scope for this task —
documented above as a distinct, harder follow-up, not a variant of this one.

### The caveat — what this is NOT (and why)
This back-solves the **price to hit a target**. It does **not** feed temperature back onto the
economy through a damage function (temperature → productivity loss → GDP). That
(Interpretation B) is genuinely harder — and hard for *scientific*, not engineering, reasons:
damage functions are the most contested object in climate economics, with order-of-magnitude
disagreement for the same warming (contrast DICE's process form [Nordhaus2017] with the
econometric estimates of Burke, Hsiang & Miguel [Burke2015] — they differ substantially). It
stays where the roadmap already put it: an **optional, clearly-labelled-illustrative** extension
(Phase 7.4) using those published functions, never a headline result. Interpretation A needs
none of that — it only uses the emissions→temperature direction, which is well-established.

---

## Summary

| Feature | Fits where | Difficulty | New pieces |
|---|---|---|---|
| **Energy-price inputs** | near-term, in Engines 1/2 (richer in CGE) | low | `EnergyPrice` shock + cost-assembly branch |
| **Temperature-target back-solve** | Phase 7 (needs 7.1 dynamics + 7.3 FaIR) | medium | root-find inverter + `TemperatureTarget` |
| **Temperature → damage feedback** | Phase 7.4, optional/illustrative only | high (scientific) | published damage function; caveated |

Both requested features are sound and fit the architecture. Energy prices are a cheap, high-value
near-term add. The temperature back-solve is the credible way to make the platform target-driven
and IAM-ish — and it's a Phase 7 feature, not a bolt-on, because it needs the climate module and
dynamics that phase provides.
