# Model description: Nature exposure via ENCORE (Phase 6.1–6.3)

- **Implements:** `cge.nature` (`encore.py`, `concord.py`, `exposure.py`, `fixture.py`)
- **Roadmap phase:** 6 (tasks 6.1–6.3)
- **Capabilities:** ecosystem-service dependency exposure (direct + upstream) per good
- **Status:** implemented and tested on the toy economy with an **illustrative, published-sourced
  ENCORE fixture**. The real ENCORE knowledge base (registration-gated) drops into the same
  `EncoreDependencies` contract via `load_encore_csv` with **no code change**. The nature→shock
  translation (6.4), the GUI heatmaps (6.5), and a curated full ENCORE↔EXIOBASE concordance are the
  remaining Phase-6 sub-tasks.

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

## 7. Validation

`tests/test_nature.py` covers: the materiality scale's monotonicity/bounds; ingestion validation
(unknown class, duplicate pair) and CSV round-trip; the concordance (sector mapping, uncovered-sector
rejection, region broadcast); and the exposure engine — the $E \ge D$ invariant and [0, 1] bound for
both rules, upstream propagation (manufacturing's inherited water exposure), the conservative `max`
screen, the non-productive-economy rejection, and the unknown-rule rejection.

## References

- **[ENCORE]** ENCORE Partners (Natural Capital Finance Alliance & UNEP-WCMC). *ENCORE: Exploring
  Natural Capital Opportunities, Risks and Exposure.* — The dependency/impact knowledge base.
- **[vanToor2020]** van Toor, J. et al. (2020). *Indebted to nature: Exploring biodiversity risks for
  the Dutch financial sector.* De Nederlandsche Bank / PBL. — Source of the ENCORE↔sector mapping
  approach and the materiality→numeric scale.

See [`docs/references.md`](../references.md) for full citations.
