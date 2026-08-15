# Model description: Physical ecosystem-service state (Phase 6b)

- **Implements:** `cge.nature.state` (`channels.py`, `baselines.py`, `pathways.py`,
  `translate_state.py`, `double_count.py`)
- **Roadmap phase:** 6b (tasks 6b.1–6b.5)
- **Sits upstream of Phase 6.** Phase 6 (`docs/models/nature-encore.md`) answers *how exposed* a
  sector is to an ecosystem service. Phase 6b models the **physical state** of the service itself, so
  a scenario specifies a *physical degradation/restoration pathway* (e.g. "the renewable water stock
  falls to 70% of baseline by 2040") rather than a bare `severity` number fed straight into 6.4.
- **Status: BUILT, EXPERIMENTAL.** The state variables, baselines and translation functions are
  documented and cited, but — exactly as with the ENCORE materiality ramp and the equal-weighted
  concordance — the numbers are **illustrative of the method, not calibrated risk**. Do not use for
  consulting without channel-by-channel empirical calibration.

## 1. What 6b adds and where it plugs in

A Phase-6 nature scenario carries a `NatureStress(service=…, severity=…)`: the author asserts the
fractional degradation directly. Phase 6b replaces that assertion with a **modelled chain**:

```
physical state pathway  →  state→severity response  →  NatureStress(service, path)  →  [Phase 6.4]
(6b.2, scenario)           (6b.1/6b.4, this phase)      (6b.3, this phase)             exposure→engine
```

Everything downstream of the `NatureStress` is Phase 6, unchanged. 6b only produces the `severity`
(as a per-year **path**) that the scenario author used to write by hand.

## 2. State variables and baselines (6b.1)

Each **channel** binds a physical state variable to the ENCORE service(s) it degrades. State is an
**index with baseline = 100** (100 = the reference-year condition of the cited account), so a scenario
speaks in fractions of baseline and the empirical content lives in the *pathway* and the *sensitivity*.
Five channels ship (`cge.nature.state.baselines.SHIPPED_CHANNELS`); the model is open — a user can
define their own `ServiceStateChannel`.

| Channel | Mechanism | ENCORE service(s) driven | State variable | Baseline source | Sensitivity |
|---|---|---|---|---|---|
| `water_availability` | water_availability | Water flow regulation, Water purification | renewable freshwater stock | AQUASTAT / SEEA-Water renewable-water accounts (FAO/UN) | 1.0 |
| `pollination` | pollination | Pollination | wild-pollinator abundance index | IPBES 2016 pollination assessment | 0.6 |
| `soil_quality` | soil_quality | Soil quality regulation, Soil and sediment retention | soil-quality / organic-carbon index | FAO *Status of the World's Soil Resources* 2015 | 0.7 |
| `forestry_stock` | forestry_stock | Biomass provisioning | forest growing-stock volume | FAO *Global Forest Resources Assessment* 2020 | 1.0 |
| `fisheries_stock` | fisheries_stock | Biomass provisioning | fish biomass vs. MSY | FAO *SOFIA* 2022 | 1.0 |

Notes on the two load-bearing choices:

- **Water drives the COMPONENT services, not "Water supply".** ENCORE Explanatory note #1 flags "Water
  supply" as a *combined* service that duplicates its components; Phase 6's translate layer rejects
  stressing the combined service alongside a component. So the water channel drives *Water flow
  regulation* + *Water purification* (the physically meaningful, non-double-counting choice).
- **Pollination sensitivity is sub-proportional (0.6).** IPBES: ~75% of crop *types* are
  pollinator-dependent, but the crop *production volume* at risk if pollinators vanish is far smaller;
  a proportional map would overstate the hit. The exposure layer then targets pollinator-dependent
  crops via the concordance.

Each channel carries a `Provenance` and a human-readable `source_note` — reviewable data, not opaque
constants.

## 3. State → severity response (6b.1 / 6b.4)

For a physical state level `s` with baseline `b`, the **shortfall** is the fraction of baseline lost:

$$\text{shortfall} = \operatorname{clamp}\!\left(\frac{b - s}{b},\ 0,\ 1\right)$$

The **default response is linear** — transparent, no hidden nonlinearity:

$$\text{severity} = \operatorname{clamp}(\text{sensitivity} \times \text{shortfall},\ 0,\ 1)$$

Two **opt-in, documented** nonlinearities are first-class parameters (never applied silently):

- **`threshold`** (a tipping point). Below a critical state fraction `x_crit` of baseline, the response
  turns **convex**: an extra penalty `(1 − severity) · depth^exponent` where `depth` is how far past
  `x_crit` the state has fallen (0 at the threshold, 1 at total collapse). Above `x_crit` the response
  is exactly linear. `exponent = 2` (quadratic) by default.
- **recovery hysteresis** (`recovery_rate`, in the pathway layer). See §4.

Severity is always clamped to `[0, 1]`.

## 4. Degradation / restoration pathways (6b.2) and recovery (6b.4)

A `StatePathway` gives one channel's physical state trajectory over the scenario years, in one of two
forms:

- explicit **`states`** — `{2030: 80, 2040: 60}`, piecewise-linear between points, held at baseline
  before the first point and flat-extrapolated after the last;
- **`degradation_rate`** — index points of baseline lost per year from `start_year` (negative =
  restoration).

**Recovery hysteresis.** Restoration of a physical state does not instantly restore the *service*.
When `recovery_rate` is set, the **effective** state the severity response sees may fall freely
(damage is prompt) but rise by at most `recovery_rate` index points per year — so a scenario that
restores the stock overnight still shows the service lagging. The default (`recovery_rate = None`)
applies no lag; nothing nonlinear is assumed silently.

## 5. Translation to NatureStress (6b.3)

`state_to_nature_stresses(channel, pathway, years)` runs the effective state path through the
channel's response and emits **one `NatureStress` per ENCORE service** the channel drives, each
carrying the derived per-year severity **path** (the scalar `severity` is the path's peak). Coverage
(`coverage_sectors` / `coverage_regions`) passes through so a physical pathway can be restricted to,
say, one region. A pathway that never degrades the service emits nothing — byte-identical to an
unstressed run. `build_state_scenario([...])` composes several channels into the list a scenario feeds
to `build_nature_shocks`.

## 6. Double-counting reconciliation vs climate damages (6b.5)

Some physical mechanisms are claimed by **both** a 6b nature channel and a Phase-7c
physical-climate-risk channel — e.g. heat/drought-driven water stress is both a `water_availability`
nature mechanism and a climate physical-risk pathway; soil degradation likewise. Applying both would
count the same physical effect on the economy twice.

Phase 7c does not exist yet, so 6b ships the **reconciliation rule** (`cge.nature.state.double_count`)
plus an automated conflict check the runner will call once 7c lands:

- `water_availability` and `soil_quality` are **shared** mechanisms. The *physical-climate* portion is
  assigned to the 7c channel (it models the hazard); the *non-climate* portion (aquifer
  over-abstraction, tillage practice) to the 6b channel. A scenario may run **either** as a
  mechanism's owner, **not both** on the same mechanism.
- `pollination`, `forestry_stock`, `fisheries_stock` are **nature-owned** — 7c has no counterpart, so
  no conflict arises.

`check_double_counting(nature_mechanisms, climate_mechanisms)` returns a report; a scenario driving a
shared mechanism through both is **rejected** with a message naming the mechanism and both claimants —
it is never silently summed. This is a Definition-of-Done criterion, not an afterthought.

## 7. Scope & honesty

- Baselines are the reference condition of a named published account; sensitivities are documented
  central estimates (proportional unless a published reason to differ).
- The state→severity map, the nonlinearity shapes, and the recovery rate are **scenario assumptions**,
  exactly like 6.4's severity→productivity choice — results are **illustrative of the method**.
- The pathway is deterministic bookkeeping over the run years: no stochastic content, no optimisation.

## References

- **[AQUASTAT / SEEA-Water]** FAO AQUASTAT renewable water resources; UN *System of Environmental-
  Economic Accounting for Water* (SEEA-Water, 2012).
- **[IPBES2016]** IPBES (2016). *Assessment Report on Pollinators, Pollination and Food Production.*
- **[FAO-SWSR]** FAO & ITPS (2015). *Status of the World's Soil Resources.*
- **[FAO-FRA]** FAO (2020). *Global Forest Resources Assessment.*
- **[FAO-SOFIA]** FAO (2022). *The State of World Fisheries and Aquaculture.*

See [`docs/references.md`](../references.md) for full citations and
[`docs/models/nature-encore.md`](nature-encore.md) for the Phase-6 exposure layer this feeds.
