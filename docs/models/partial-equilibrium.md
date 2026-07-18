# Model description: Engine 2 — partial-equilibrium volume response

- **Implements:** `cge.engines.partial_eq` (`PartialEqEngine`, v0.1.0)
- **Roadmap phase:** 4
- **Capabilities:** prices, volumes
- **Status:** implemented (`PartialEqEngine` v0.1.0); validated on the toy economy and internal
  identities (`partial_eq` validation suite: sign, proportionality, band ordering, price
  pass-through, elasticity provenance). Volume magnitudes depend on assembled elasticities and
  are **indicative, not precise** (see §6). The Armington nest (equation 2) is specified but
  not yet implemented — v1 applies the own-price response (1) only.

## 1. Purpose & scope

Estimate the **first-order change in production volume** of each good under a carbon price,
by applying demand elasticities to the price changes Engine 1 computes. Answers: *"if a
carbon price raises the price of good X by Δp, roughly how much does its produced quantity
fall?"* — with an explicit uncertainty range.

**In scope:** own-price demand response per good, propagated to production volume through the
Leontief quantity system; low/central/high elasticity envelopes.

**Not modelled:** Armington domestic/import substitution (specified in §4, **not implemented**
in v1); income effects, cross-price substitution, factor-market clearing, or general-equilibrium
feedback (that is the CGE, Phase 5). This is a **partial-equilibrium** estimate: it holds
everything except the priced good's own demand
fixed. It is deliberately simpler than the CGE and remains useful as a cross-check on it.

## 2. Notation

| Symbol | Meaning | Units |
|---|---|---|
| $\Delta p_i$ | fractional price change of good $i$ (from Engine 1) | dimensionless |
| $\varepsilon_i$ | own-price elasticity of demand for good $i$ (≤ 0) | dimensionless |
| $\Delta q_i / q_i$ | fractional change in quantity of good $i$ | dimensionless |
| $\sigma_i$ | Armington elasticity (domestic↔import substitution) for $i$ | dimensionless |
| $\Delta p^{d}_i,\ \Delta p^{m}_i$ | domestic vs import price change of $i$ | dimensionless |

## 3. Assumptions

1. **First-order / log-linear.** Responses are linear in the (small) price change:
   $\Delta q_i/q_i = \varepsilon_i\,\Delta p_i$. Valid for modest price changes; large shocks
   need the CGE.
2. **Own-price only** by default. Cross-price effects enter only through the optional
   Armington nest.
3. **Prices are taken from Engine 1** (the corrected, unit-consistent price model) and are
   exogenous to this engine — no feedback from quantity back to price.
4. **Elasticities are uncertain.** Each is a (low, central, high) triple with a cited source;
   the engine propagates all three, and results carry the band.
5. **No income effect / budget rebalancing.** A pure demand-curve movement.

## 4. Derivation

**Step 1 — Final-demand response.** By the constant-elasticity definition of own-price demand,
final demand $y_i$ responds to the price change $\Delta p_i$ (from Engine 1) by the
**finite-change** form:

$$
\frac{\Delta y_i}{y_i} = (1 + \Delta p_i)^{\varepsilon_i} - 1. \tag{1}
$$

This is used rather than the linear approximation $\varepsilon_i\,\Delta p_i$ because carbon-
price runs produce large $\Delta p_i$ on real data (a live EXIOBASE example gave $\Delta p =
2.37$); the linear form there yields impossible responses below $-100\%$, whereas (1) is bounded
below by $-1$ (a price rise cannot destroy more than 100% of demand). For small $\Delta p_i$,
$(1+\Delta p_i)^{\varepsilon_i}-1 \approx \varepsilon_i\,\Delta p_i$, recovering the textbook
first-order form. With $\varepsilon_i \le 0$, a price rise gives a demand fall — the expected sign.

**Step 2 — Production follows demand (Leontief quantity model).** Production volume is *not*
the per-good demand response — a fall in demand for a good pulls its upstream suppliers' output
down too. Gross output $x$ is tied to final demand $y$ by the Leontief quantity model
[MillerBlair2009, §2.2]:

$$
x = (I - A)^{-1}\,y, \qquad\text{so}\qquad
\frac{\Delta x_i}{x_i} = \frac{\big[(I-A)^{-1}\,\Delta y\big]_i}{x_i}. \tag{2}
$$

**Equation (2) is the engine's production-volume result.** A fall in final demand for a
downstream good propagates through $(I-A)^{-1}$ to reduce every upstream sector that supplies
it. (An earlier version applied (1) per good and reported it directly as "volume," which
omitted this propagation — a fall in a final good's demand left its suppliers unchanged. That
was an own-demand response, not production volume, and is corrected here.)

**Uncertainty.** Steps (1)–(2) are evaluated three times — low, central, high elasticity — for
a $\Delta x/x$ **envelope**. Because volume answers are elasticity-sensitive (roadmap P4 risk),
the band is a first-class output.

**Armington substitution — specified, NOT implemented in v1.** When domestic and imported
varieties of good $i$ are imperfect substitutes [Armington1969], their relative demand responds
to their relative price with elasticity $\sigma_i$:
$\Delta\ln(q^{d}_i/q^{m}_i) = -\sigma_i(\Delta p^{d}_i - \Delta p^{m}_i)$ (carbon leakage, in
miniature). This term is **not implemented** — v1 applies the own-price response (1) and its
Leontief propagation (2) only. Adding it requires a build with a domestic/import price split.

## 5. Algorithm

1. Run Engine 1 to get $\Delta p_i$ for the scenario (reuse, don't reimplement).
2. Load the demand `ElasticitySet` (validated: finite, band-ordered, $\le 0$, sourced). For
   goods with no assembled value, use a documented default and **tag it** `default`.
3. Precompute the Leontief inverse $(I-A)^{-1}$ and baseline $x_0 = (I-A)^{-1} y_0$.
4. For each band $b$: $\Delta y_i/y_i = (1+\Delta p_i)^{\varepsilon_i^{(b)}}-1$ (1); form
   $y_{\text{new}} = y_0(1+\Delta y/y)$; propagate $x_{\text{new}} = (I-A)^{-1} y_{\text{new}}$;
   report $\Delta x_i/x_i = (x_{\text{new},i}-x_{0,i})/x_{0,i}$ (2).
5. Emit into the `ResultSet`, per good per band: `final_demand_change` (Δy/y), `volume_change`
   (Δx/x, production), plus `price_change` and `elasticity_used` (central). The manifest carries
   a content hash of the elasticity values and per-good source/confidence/default status, so a
   result is reproducible and its elasticity provenance auditable.

Note: elasticities are matched to build sectors by **name** (with a flagged default for
unmatched goods), not via a formal `ConcordanceMap` — a proper concordance is a follow-up.

No iteration is needed for the first-order form; the "fixed-point between quantity weights and
prices" in the roadmap is only required if prices are made to respond to quantities, which is
out of scope for this PE engine (it is what the CGE does).

**Complexity.** One Engine-1 solve plus $O(n)$ elementwise products per band — negligible.

## 6. Calibration / parameters

- **Demand elasticities** $\varepsilon_i$: assembled from the literature (published GTAP
  parameter papers, USDA, meta-analyses), stored in an `ElasticitySet` with per-value source
  and (low, central, high). There is **no clean open elasticity database**, so coverage is
  partial and defaults are used and flagged. This is the dominant uncertainty in the volume
  answer — the reason the band is mandatory.
- **Armington elasticities** $\sigma_i$: from the same literature; optional.
- **This engine has no other free parameters**; prices come from Engine 1.

## 7. Validation

`cge.validation.suites.partial_eq` (run by `cge validate`), plus `tests/test_partial_eq.py`:

- **Sign:** a positive price change with $\varepsilon<0$ gives a negative production-volume
  change.
- **Bounded:** even at a large price (€500/t) the finite-change form keeps $\Delta x/x > -100\%$
  (the linear form produced impossible values below $-100\%$ on live data).
- **Leontief propagation:** production change $\Delta x/x$ differs from the raw demand change
  $\Delta y/y$ for connected goods — i.e. $(I-A)^{-1}$ actually propagated (a 2-sector
  network test confirms an upstream supplier's output falls when a downstream good's demand does).
- **Band ordering:** low ≤ central ≤ high volume envelopes, per good.
- **Zero shock:** $\Delta p=0 \Rightarrow \Delta x=0$.
- **Price cross-check vs Engine 1:** price rows match Engine 1 exactly (Engine 2 reuses, not
  recomputes).
- **Elasticity provenance & reproducibility:** per-good source/confidence/default in the
  manifest, and a content hash of the elasticity values (two different tables → different
  manifests). Invalid elasticity sets (positive, unordered, unsourced) are rejected at
  construction.

**Remaining (needs live data / literature):** magnitude comparison against published
carbon-price incidence/volume studies for real sectors, a formal elasticity concordance, and
the Armington nest — all indicative/deferred until a curated elasticity set and a published
benchmark are in place.

## 8. References

[Armington1969] (Armington substitution); demand-elasticity sources are recorded per value in
the `ElasticitySet` and keyed into [`../references.md`](../references.md) as they are added.
Cross-reference the Engine 1 doc [`io-price-model.md`](io-price-model.md) for the price inputs.
