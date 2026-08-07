# Model description: Nature exposure via ENCORE (Phase 6.1–6.5)

- **Implements:** `cge.nature` (`encore.py`, `concord.py`, `exposure.py`, `translate.py`,
  `fixture.py`); `ProductivityShock` consumption in `cge.engines.partial_eq` (Engine 2) and in all
  three variants of `cge.engines.cge_static` (Engine 3, the GE tier — closed, open, multi-region)
- **Roadmap phase:** 6 (tasks 6.1–6.4)
- **Capabilities:** ecosystem-service dependency exposure (direct + upstream) per good; a
  `NatureStress` degradation scenario run end-to-end through an economic engine, in both the
  partial-equilibrium (Engine 2) and general-equilibrium (Engine 3, all variants) tiers
- **Status:** implemented and tested on the toy economy with an **illustrative, published-sourced
  ENCORE fixture**. The real ENCORE knowledge base (registration-gated) drops into the same
  `EncoreDependencies` contract via `load_encore_csv` with **no code change**. **Phase 6 is
  complete** (6.1–6.5): ingestion, concordance, exposure, the nature→productivity translation
  consumed by Engine 2 and all three CGE variants, and the nature GUI page (`gui/pages/nature.py`:
  dependency heatmap, supply-chain drill-down, nature-scenario runner). A curated full
  ENCORE↔EXIOBASE concordance (replacing the illustrative fixture) is the only remaining nature
  enhancement, tracked under Phase 6b.

> **Honest scope.** Every dependency rating shipped here is a small hand-entered subset seeded from
> the central-bank literature ([vanToor2020], [ENCORE]) and **labelled illustrative** in its
> provenance. It is enough to exercise and test the mechanism end-to-end; it is **not** an
> analytical result. Nature-scenario numbers are illustrative until run on the licensed ENCORE data,
> exactly as climate-damage numbers are labelled illustrative in Phase 7.

## 1. Purpose & scope

Answer, for any good in any region: *which ecosystem services does it depend on — directly, and
through its supply chain — and how materially?* ENCORE rates the **direct** dependency of a
production process on each ecosystem service (pollination, surface water, soil quality, climate
regulation, flood control, …) as an ordinal **materiality class**. This module turns those ordinal
ratings into numeric scores, maps them onto the economy's sectors, and **propagates them upstream
through the input–output structure**, so a good that has no direct dependency but buys
water-dependent agricultural inputs still shows the inherited exposure.

**In scope:** the materiality→numeric scale; ENCORE ingestion as provenance-carrying data; the
ENCORE↔economy concordance as reviewable weighted data; and the exposure engine (direct scores +
two upstream-propagation rules).

**Explicitly not modelled here:** the *economic effect* of a degradation (that is Phase 6.4 —
translating a `NatureStress` into `ProductivityShock`s scaled by these exposure scores, fed to the
economic engines); the physical *state* of the services themselves (Phase 6b); and region-specific
dependency (dependency is treated as a per-unit-of-output intensity, identical across regions, until
region-specific ENCORE data is available — a documented follow-up).

## 2. The materiality → numeric scale (the load-bearing choice)

ENCORE ratings are ordinal: Very High / High / Medium / Low / Very Low. Every downstream number
depends on how those classes become numbers, so the mapping is a **named, cited, review-visible
constant** (`encore.MATERIALITY_SCALE`), not a value buried in code:

| Class | VH | H | M | L | VL |
|---|---|---|---|---|---|
| Score | 1.0 | 0.8 | 0.6 | 0.4 | 0.2 |

A linear 0.2-step ramp, the mapping used by DNB's *Indebted to nature* ([vanToor2020]) and later
central-bank studies to turn ENCORE classes into a [0, 1] dependency weight. The only load-bearing
property is the **strict ordering** (a more-material class must score higher); the exact spacing is a
modelling choice. To make only the highest classes bite, swap in a convex ramp (e.g.
VH=1.0, H=0.5, M=0.25, L=0.1, VL=0.0) — one constant, no other change.

## 3. Ingestion (6.1)

`EncoreDependencies` is a first-class data object (`_DataObject`, so it carries `Provenance`). Its
`ratings` is a long table `(process, service, materiality)`; the validator rejects an unknown
materiality class (so the numeric scale is total) and a duplicated `(process, service)` pair (which
would double-count or silently override). `score_matrix()` applies the scale to give a dense
**process × service** matrix in [0, 1] (an unrated pair is 0 — *no* rated dependency). `load_encore_csv`
is the real ingestion path; `fixture.encore_fixture()` builds the same object in memory for tests.

## 4. Concordance (6.2)

ENCORE speaks in its own process taxonomy; the economy speaks EXIOBASE-shaped **sectors** labelled
`<region>:<sector>`. `concord.sector_scores` maps process scores onto sectors through a Phase-1
`ConcordanceMap` — each economy sector maps to one or more ENCORE processes with **weights summing to
1**, so a sector blended from several processes is a documented weighted average. This is the single
biggest credibility surface, so it is **data with a cited source, reviewed weight by weight**, never
a code constant. Every sector must be covered; an unmapped sector is rejected (it would otherwise get
silent-zero dependency). `broadcast_to_goods` then gives every region the same per-sector dependency
intensity (dependency is per unit of output; region-specific dependency needs region-specific ENCORE
data — a follow-up).

## 5. The exposure engine (6.3)

Let $D$ be the goods × service **direct** dependency matrix (from §3–4) and $A$ the
technical-coefficient matrix ($A_{ij}$ = units of good $i$ used to make one unit of good $j$; column
$j$ sums to good $j$'s intermediate-input share $\rho_j < 1$, the remainder being value added).

Exposure is a **risk** measure with a defining invariant: a good is *at least* as exposed as its own
direct dependency, and its supply chain can only *add* exposure — so $E \ge D$ everywhere, and
$E \in [0, 1]$. Two aggregation rules combine the upstream contribution (both exposed as the `rule`
parameter, per the roadmap):

**`weighted_mean` (noisy-OR).** A good's own direct dependency, raised by an input-intensity-weighted
contribution from its inputs' exposure, scaled by the remaining headroom:

$$ E_{jk} = D_{jk} + (1 - D_{jk})\sum_i A_{ij}\,E_{ik}. $$

Because $\sum_i A_{ij} = \rho_j < 1$ (value added is a genuine "leak" each tier) the map is a
contraction, so the fixed-point iteration converges to a unique $E \in [0, 1]$. A small dirty input
contributes a little; a large one, more.

**`max` (conservative screen).** A good is as exposed as the **most-exposed** thing anywhere in its
supply chain, regardless of that input's share (any $A_{ij} > 0$ counts as "uses"):

$$ E_{jk} = \max\!\Big(D_{jk},\ \max_{i:\,A_{ij}>0} E_{ik}\Big). $$

Not linear, so solved by fixed-point iteration (monotone non-decreasing, bounded by 1 → converges).
Defensible for risk screening ("is this good exposed *at all*?").

Both return a goods × service matrix in [0, 1]; the direct scores are returned alongside so a GUI can
show "direct vs total". A non-productive economy (some $\rho_j \ge 1$, i.e. no value added) is
rejected — the propagation would not damp.

**Why not $(I - \tilde A)^{-1}D$ on input shares.** Column-normalising $A$ to input shares makes each
column sum to exactly 1, so $(I - \tilde A^{\top})$ is singular and the "standard Leontief" inverse
diverges. Anchoring the recursion on the actual coefficients (whose columns sum to $\rho_j < 1$) is
what keeps it bounded and convergent — this is a real subtlety, checked by the non-productive-economy
guard and the [0, 1] bound tests.

## 6. Worked result (toy economy)

On the toy economy (`agriculture`/`energy`/`manufacturing` × regions A, B), with the illustrative
fixture, manufacturing has **low direct** surface-water dependency (L = 0.4) but buys agricultural and
energy inputs that are highly water-dependent. Its **total** exposure therefore rises above direct:
`weighted_mean` lifts it to ≈ 0.64, and the conservative `max` screen to 1.0 (its chain reaches
agriculture, VH = 1.0). This upstream inheritance — invisible to a direct-only reading — is the whole
point of the exposure engine, and it is exactly the supply-chain channel the Leontief price model
(Engine 1) uses for cost, reused here for dependency.

## 7. From degradation to economic effect (6.4)

Exposure (§5) is *potential* — which goods depend on which services. A **scenario** says a service
actually degrades, and `nature.translate` turns that into the economic hit the engines consume, via
the shock vocabulary (the architectural seam: the nature module *emits shocks*, it does not talk to
engines).

A `NatureStress(service=k, severity=σ)` — σ ∈ [0, 1], the fraction of service *k* lost — reduces good
*j*'s output-producing capacity in proportion to its exposure $E_{jk}$:

$$ \text{loss}_{jk} = \sigma_k\,E_{jk}, \qquad
\text{surviving productivity of } j = \prod_k (1 - \sigma_k E_{jk}), \qquad
\text{delta}_j = \prod_k (1 - \sigma_k E_{jk}) - 1 \le 0. $$

Multiple degraded services compose multiplicatively (independent proportional hits). The mapping is
deliberately **linear in the exposure score** — the score already carries materiality and
supply-chain propagation, so no second non-linearity is baked in; a convex severity response is an
optional documented layer, not a default. `nature_to_productivity` emits one `ProductivityShock` per
affected good (carrying its region/sector), rejecting a stress that names a service absent from the
exposure matrix (it would silently do nothing). `build_nature_shocks` runs the whole
ENCORE→concordance→exposure→translation chain in one call.

**Shock incidence — direct vs. total (avoids double-counting upstream).** The exposure score $E_{jk}$
already embeds the *upstream* dependence a good inherits through its supply chain (§5). Whether the
productivity shock should carry that upstream part depends on the consuming engine, so
`build_nature_shocks` takes an `incidence` argument:

- **`direct`** — each good is shocked only for its *own direct* dependency; upstream propagation is
  left to the engine's own supply-chain transmission. This is the default for the **CGE** (and the
  IO price engine): they already propagate a shocked sector's price and input requirements through
  the input–output network, so applying the *total* exposure there would count the upstream channel
  **twice** (the score embeds it once, the GE network transmits it again).
- **`total`** — each good is shocked for its full direct + upstream exposure. This is the default for
  **`partial_eq`**, which has no endogenous supply transmission, so the reduced-form total is the
  only way upstream dependence reaches the good.

`nature.INCIDENCE_BY_ENGINE` records the per-engine default; the runner selects it automatically.
This follows the central-bank practice (e.g. DNB) of separating exposure screening from the shock
incidence actually applied, rather than compounding embedded exposure and then re-transmitting it
through the GE network.

**Engine 2 consumption (partial equilibrium).** Engine 2 (`partial_eq`) consumes a
`ProductivityShock` as a supply-side output multiplier $\prod(1+\text{delta})$, clipped at 0,
composed multiplicatively **on top of** the price-driven demand response:
$\Delta x/x = (1 + \Delta x_{\text{demand}})\cdot m_{\text{prod}} - 1$. A good with no productivity
shock has $m = 1$, so a pure carbon/energy run is byte-identical to before; a `productivity_change`
row is emitted only where the supply channel bites. On the toy economy a 40% surface-water
degradation cuts agriculture's output ~40% (fully water-dependent), manufacturing less (its exposure
is largely inherited upstream) — the nature-risk propagation the exposure engine exists to produce.

**Engine 3 consumption (general equilibrium — the GE tier).** All three static-CGE variants
(`cge_static`: **closed**, **open**, **multi-region**) consume the same `ProductivityShock`s as a
per-sector **Hicks-neutral productivity multiplier** $\theta_i$ ($\theta = 1 + \text{delta}$, so a
nature degradation gives $\theta < 1$). A sector with productivity $\theta_i$ needs $1/\theta_i$ of
its whole input bundle — intermediates, value added, and the per-output carbon wedge — per unit
output, so its zero-profit unit cost divides by $\theta_i$: $p_i = \text{unit cost}_i / \theta_i$.
Because that single scaling flows into the Leontief inverse, factor demand, and goods-market
clearing, the equilibrium **reallocates**: the degraded sector's price rises, its output falls, and
— unlike Engine 2's first-round quantity hit — relative prices and factor demands adjust across the
whole economy. $\theta = 1$ leaves the residual **bit-for-bit unchanged** in every variant, so
benchmark replication / homogeneity / Walras are untouched (proven over random price/wage points,
not just asserted).

- **Closed** (single region): a −20% hit on bread raises its price ~+11%, cuts its output ~−17%, and
  drags its upstream input supplier down too. Matched by sector (single-region, so a shock's region
  tag is ignored).
- **Open** (Armington/CET, home + ROW): the same hit raises the home price and cuts home output, but
  now demand shifts toward **imports** and the un-degraded domestic sector expands (Armington
  substitution) — a richer response than the closed variant. Matched by sector.
- **Multi-region** (bilateral trade): the shock's **region coverage is honoured** — a degradation on
  region N's bread cuts N's output while production **leaks** to the un-degraded region (S's bread
  output *rises*), the nature analogue of carbon leakage.

**Scope.** Engine 2 gives the direct/first-round supply hit through a fixed-technology quantity
system; all three CGE variants give the general-equilibrium response (single-region, open-economy
carbon-leakage-style, and true multi-region leakage).

**Standard pipeline & provenance.** A `NatureStress` scenario runs through the **standard runner**
(`run_scenario`), not a GUI-only path: the runner translates the stresses to `ProductivityShock`s at
the engine-appropriate incidence and stamps the manifest with the full nature provenance — the
ENCORE snapshot + concordance content hashes, the materiality scale, the exposure rule, and the
incidence mode. So a nature run is **reconstructible from its manifest**, and a YAML/CLI nature
scenario is a first-class citizen alongside a carbon-price scenario. (`EncoreDependencies` and
`ConcordanceMap` travel in the data source; the `toy` source ships the illustrative fixture, a real
build supplies them, and a source lacking them rejects a nature run with guidance.)

## 8. Validation

`tests/test_nature.py` covers: the materiality scale's monotonicity/bounds; ingestion validation
(unknown class, duplicate pair) and CSV round-trip; the concordance (sector mapping, uncovered-sector
rejection, region broadcast); the exposure engine — the $E \ge D$ invariant and [0, 1] bound for
both rules, upstream propagation (manufacturing's inherited water exposure), the conservative `max`
screen, the non-productive-economy rejection, and the unknown-rule rejection; and the 6.4
translation — severity scaled by exposure, unknown-service rejection, the end-to-end
`build_nature_shocks` chain, Engine 2's productivity-shock consumption (supply hit, region scoping,
byte-identical no-op without a shock), and a full `NatureStress` scenario producing a schema-valid
`ResultSet`. The **GE tier** is covered across the three variant test files: the closed CGE's
consumption (a −20% hit raises the degraded sector's price / lowers its output, schema-valid
`ResultSet`) and sector-only matching (`tests/test_cge_static.py`); the open economy's consumption
(`tests/test_cge_open.py`); and the multi-region **region-scoped leakage** — a hit on region N's
bread cuts N's output while S's rises (`tests/test_cge_multi.py`). Each variant's test file also
asserts the **exact byte-identical residual at $\theta = 1$** over random price/wage points — the
guarantee that replication/homogeneity/Walras are untouched. The **GUI** (§9) is covered by
`tests/test_gui_service.py` (the Streamlit-free `nature_exposure`/`run_nature` façade) and a headless
render smoke test in `tests/test_gui_pages.py`.

## 9. GUI (Phase 6.5)

The **Nature** page (`cge/gui/pages/nature.py`, within the P3 Streamlit framework) exposes all of the
above interactively, driven by `GuiService.nature_exposure`/`run_nature` (a Streamlit-free façade so
the logic is unit-tested independent of the UI):

- **Dependency heatmap** — the good × ecosystem-service exposure matrix (§5), toggling direct-only
  vs. direct+upstream and the `weighted_mean`/`max` aggregation rule, on a green→red gradient.
- **Supply-chain drill-down** — for one good, its direct dependency on each service vs. the total
  once inherited-from-inputs exposure is added, so the upstream channel is visible per good.
- **Nature-scenario runner** — pick services to degrade and by how much, choose the economic engine
  (partial-equilibrium or GE), and run the whole exposure→`NatureStress`→`ProductivityShock`→engine
  chain end-to-end, charting the per-good output response.

The page operates on the illustrative ENCORE fixture and labels it as such at the top.

## References

- **[ENCORE]** ENCORE Partners (Natural Capital Finance Alliance & UNEP-WCMC). *ENCORE: Exploring
  Natural Capital Opportunities, Risks and Exposure.* — The dependency/impact knowledge base.
- **[vanToor2020]** van Toor, J. et al. (2020). *Indebted to nature: Exploring biodiversity risks for
  the Dutch financial sector.* De Nederlandsche Bank / PBL. — Source of the ENCORE↔sector mapping
  approach and the materiality→numeric scale.

See [`docs/references.md`](../references.md) for full citations.
