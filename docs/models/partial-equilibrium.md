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

**In scope:** own-price demand response per good; optional Armington-style substitution
between domestic and imported varieties; low/central/high elasticity envelopes.

**Not modelled:** income effects, cross-price substitution beyond Armington, factor-market
clearing, or general-equilibrium feedback (that is the CGE, Phase 5). This is a **first-order
partial-equilibrium** estimate: it holds everything except the priced good's own demand
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

**Own-price demand response.** By definition of the own-price elasticity of demand,

$$
\varepsilon_i = \frac{\partial \ln q_i}{\partial \ln p_i}
\;\Longrightarrow\;
\frac{\Delta q_i}{q_i} = \varepsilon_i\,\frac{\Delta p_i}{p_i} = \varepsilon_i\,\Delta p_i, \tag{1}
$$

since Engine 1 reports $\Delta p_i$ already as a fractional change (baseline $p_0=1$). With
$\varepsilon_i \le 0$, a price rise ($\Delta p_i>0$) gives a quantity fall — the expected
sign. Equation (1) is the engine's core result.

**Armington substitution (optional).** When domestic and imported varieties of good $i$ are
imperfect substitutes [Armington1969], their relative demand responds to their relative price
with elasticity $\sigma_i$:

$$
\Delta\ln\!\left(\frac{q^{d}_i}{q^{m}_i}\right) = -\,\sigma_i\,\Delta\ln\!\left(\frac{p^{d}_i}{p^{m}_i}\right)
= -\,\sigma_i\,(\Delta p^{d}_i - \Delta p^{m}_i). \tag{2}
$$

A carbon price that raises the domestic price more than the import price ($\Delta p^{d}_i >
\Delta p^{m}_i$) shifts demand toward imports (carbon leakage, in miniature). Equation (2)
composes with (1): total demand moves by the own-price response, and its domestic/import
split moves by the Armington response. In v1 the Armington term is available where import and
domestic prices differ by region; where the build has no domestic/import split it is skipped
and only (1) applies.

**Uncertainty.** Equations (1)–(2) are evaluated three times — with the low, central and high
elasticity of each good — producing a $\Delta q/q$ **envelope**. Because volume answers are
sensitive to elasticities (roadmap P4 risk), the band is a first-class output, not a footnote.

## 5. Algorithm

1. Run Engine 1 to get $\Delta p_i$ for the scenario (reuse, don't reimplement).
2. Load the demand `ElasticitySet`; concord it onto the build's classification; for goods
   with no elasticity, use a documented default and **tag it** as `default` confidence.
3. For each band $b\in\{$low, central, high$\}$: $\Delta q_i/q_i = \varepsilon_i^{(b)}\Delta p_i$.
4. If an Armington set and a domestic/import price split are present, add the substitution
   term (2).
5. Emit `price_change` (passed through from Engine 1) and `volume_change` per good per band,
   plus the elasticity used, into the `ResultSet`.

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

- **Sign:** a positive price change with $\varepsilon<0$ gives a negative volume change.
- **Proportionality:** $\Delta q/q = \varepsilon\,\Delta p$ exactly on the toy.
- **Band ordering:** low ≤ central ≤ high (in magnitude) volume envelopes.
- **Zero shock:** $\Delta p=0 \Rightarrow \Delta q=0$.
- **Elasticity provenance:** every result row carries the elasticity/source used.
- **Cross-check vs Engine 1:** price rows match Engine 1 exactly (Engine 2 passes them
  through, doesn't recompute).

**Remaining (needs live data / literature):** magnitude comparison against published
carbon-price incidence/volume studies for real sectors — indicative until an elasticity set
for a real build and a published benchmark are in place.

## 8. References

[Armington1969] (Armington substitution); demand-elasticity sources are recorded per value in
the `ElasticitySet` and keyed into [`../references.md`](../references.md) as they are added.
Cross-reference the Engine 1 doc [`io-price-model.md`](io-price-model.md) for the price inputs.
