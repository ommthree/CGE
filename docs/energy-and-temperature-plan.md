# Plan — energy-price inputs & temperature-target back-solving

**Status: PLAN ONLY — not started.** Two requested features, captured here so the design is not
lost. One is a small near-term addition; the other is a well-shaped Phase 7 feature. Neither
requires changes to the existing engines' internals — both slot into the contract seams (the
typed shock vocabulary and the module slots, see `docs/overview.md` §2).

---

## Feature 1 — Country-level energy prices as optional inputs (near-term; low risk) — ✅ IMPLEMENTED

**Status: implemented** in Engine 1 (`io_price`) and available to Engine 2 (`partial_eq`) via the
shared price step, and **exposed in the GUI Run page** (carbon + any number of energy-carrier price
shocks, composed into one scenario). Interpretation (1) — a rise in the carrier's *output price* —
is built; interpretation (2) is documented as available-if-needed but not implemented. See
`docs/models/io-price-model.md` §5a for the equation-level method, `examples/energy_price_io.yaml`
for a runnable scenario, and the `energy_price_direct_share_and_propagation` /
`carbon_energy_additive` validation checks. Below is the design as built.

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

The forward chain is **monotone**: a higher carbon price ⇒ lower emissions ⇒ lower temperature.
Monotonicity is what makes the inversion easy.

### The inversion (the new bit)
- Because the chain is monotone, back-solving is a **1-D root find per period** (or over a
  cumulative-emissions budget): "find the carbon price such that the resulting temperature path
  hits the target." A bisection / Brent solver on top of the forward model — a well-behaved,
  standard numerical problem, not a bespoke optimisation.
- Two target forms, both simple:
  - **peak/end temperature** (e.g. "≤ 1.75°C in 2050"): solve the price path so temperature
    stays under the cap;
  - **carbon budget** (cumulative emissions ≤ B): often cleaner numerically, since temperature
    is near-linear in cumulative CO₂.
- The outer loop calls the (recursive-dynamic, Phase 7.1) engine forward each iteration. Cost:
  a handful of forward solves per period — cheap on the small build.
- **Guardrails (carry the platform's discipline):** if no carbon price within a sane bound hits
  the target (e.g. the target is infeasible given the economy's structure), the solver **reports
  infeasibility clearly** rather than returning a garbage price — the same "never return
  meaningless numbers" rule as Engine 1's well-posedness guard.

### Design placement
- **Depends on:** Phase 7.1 (recursive-dynamic wrapper) and 7.3 (FaIR climate module). The
  climate module slot already exists as a contract; FaIR is its first implementation.
- **New component:** a `TemperatureTarget` scenario option + a `back_solve` driver that wraps
  the forward run in the root-finder. Emits, alongside the usual `ResultSet`, the **implied
  carbon-price path** and the **resulting temperature path** — both provenance-tagged.
- **Reuses everything else:** the engines, the shock vocabulary (the back-solved price becomes a
  `CarbonPrice` time path), the result schema.

### Testing & validation
- **Round-trip identity (the key test):** back-solve a price path for target T*, then run
  forward with that exact path — the forward temperature must equal T* to tolerance. This is the
  CGE-replication analogue: it proves the inverter and the forward model are mutually consistent.
- **Monotonicity:** a stricter target ⇒ a higher implied carbon price, everywhere.
- **Feasibility guard:** an impossible target raises/reports infeasible, never returns numbers.
- **Climate emulator sanity:** FaIR reproduces a known emissions→temperature response (e.g. a
  published SSP scenario's temperature) within tolerance — the climate module's own known-answer.
- **NGFS cross-check:** the implied price path for a target should be in the ballpark of NGFS
  scenarios reaching similar temperatures (a directional bracket, not a benchmark — the same
  honesty as the EXIOBASE live gate).

### Effort & difficulty
**~1–2 wk on top of Phase 7.1 + 7.3.** The inversion itself is small (a root-find loop); the
prerequisites (recursive dynamics + FaIR) are the real work, and were already in the Phase 7
plan. **Difficulty: medium.**

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
