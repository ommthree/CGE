# Model description: Macroeconomic aggregates (GVA, GDP, deflators; real vs nominal)

- **Implements:** `cge.accounting` (`augment_with_macro_aggregates`); applied by `cge.runner`
- **Roadmap phase:** 4b (PE tier)
- **Capabilities:** post-processing accounting layer over any price/volume `ResultSet`
- **Status:** implemented (PE tier); validated on the toy economy (`macro` validation suite). These
  are **indicative** aggregates derived by arithmetic on the IO engines' outputs — they inherit the
  engines' caveats (fixed technology; Engine-2 elasticity uncertainty). The CGE (Phase 5) will emit
  GVA/GDP/CPI as **native equilibrium variables** (the GE tier), cross-checked against this layer.

## 1. Purpose & scope

Roll the per-good price and volume responses up into the aggregates a macro reader expects, **per
country and time-step**: **gross value added (GVA) per sector**, **GDP per country**, and an
aggregate **deflator** (inflation) — each reported in both **nominal** and **real** terms.

**In scope:** GVA per sector/region; GDP per region; a GDP deflator per region; the real/nominal
split. **Not in this layer:** a monetary **interest rate** (the CGE yields a *capital rental rate*,
a real factor price; a nominal policy rate needs a macro-financial closure — an optional,
explicitly-illustrative overlay, see roadmap 4b.5); **GDP growth over time** (needs the
recursive-dynamic wrapper, roadmap 7.1 — static runs give a level per year, not a growth rate).

## 2. Notation

| Symbol | Meaning | Units |
|---|---|---|
| $\mathbf{A}$ | technical coefficients (input $j$ per unit output $i$) | dimensionless |
| $\mathbf{y}$ | final demand per good | money (base year) |
| $\mathbf{x}=(\mathbf{I}-\mathbf{A})^{-1}\mathbf{y}$ | gross output per good | money |
| $\text{VA}_i$ | base-year value added of good $i$ | money |
| $\Delta p_i$ | fractional output-price change of good $i$ (from Engine 1) | dimensionless |
| $\Delta x_i$ | fractional production-volume change of good $i$ (Engine 2; 0 if none) | dimensionless |
| $g^{\text{nom}}_i$ | nominal GVA change of good $i$ | dimensionless |
| $D_r$ | GDP deflator (inflation) for region $r$ | dimensionless |
| $G^{\text{nom}}_r,\ G^{\text{real}}_r$ | nominal / real GDP change for region $r$ | dimensionless |

## 3. Assumptions

1. **Value added from the IO identity.** Base-year value added is derived from the IO system, not a
   separate table: $\text{VA}_i = x_i\,(1-\sum_j A_{ji})$ — gross output minus intermediate use. This
   is the standard national-accounts identity [MillerBlair2009, §2.2].
2. **Fixed-technology valuation (PE tier).** Value added per unit output moves with the output price
   (full pass-through, no factor substitution), and quantity with output volume. This is the same
   fixed-technology assumption as Engine 1; the CGE relaxes it.
3. **Value-added weights for aggregation.** GDP and the deflator are value-added-weighted sums of
   sector figures (production-approach GDP / GDP deflator).
4. **Real = nominal deflated by the index.** Given the deflator, real change $=(1+\text{nominal})/(1+D)-1$.
   This makes the real/nominal split exact and, at zero inflation, real $=$ nominal (validated).

## 4. Derivation (equation level)

**Base-year value added.** With $\mathbf{x}=(\mathbf{I}-\mathbf{A})^{-1}\mathbf{y}$,

$$ \text{VA}_i = x_i\left(1-\sum_j A_{ji}\right). \tag{1} $$

The column sum $\sum_j A_{ji}$ is the intermediate-input share of a unit of good $i$'s output, so
$1-\sum_j A_{ji}$ is its value-added share. Summing (1) gives the accounting identity

$$ \sum_i \text{VA}_i \;=\; \sum_i y_i, \tag{2} $$

i.e. **production-approach GDP equals expenditure-approach GDP** (checked to machine precision).

**Nominal GVA change per good.** Under fixed technology the value-added price moves with the output
price and the quantity with output volume, so for a price change $\Delta p_i$ and volume change
$\Delta x_i$,

$$ g^{\text{nom}}_i = (1+\Delta p_i)(1+\Delta x_i) - 1. \tag{3} $$

When the engine emits no volume response (Engine 1), $\Delta x_i=0$ and $g^{\text{nom}}_i=\Delta p_i$.

**Region deflator (inflation).** The GDP deflator for region $r$ is the value-added-weighted mean of
its sectors' price changes:

$$ D_r = \frac{\sum_{i\in r}\text{VA}_i\,\Delta p_i}{\sum_{i\in r}\text{VA}_i}. \tag{4} $$

**Region GDP change.** Nominal GDP change is the value-added-weighted mean of nominal GVA changes;
real GDP change deflates it by (4):

$$ G^{\text{nom}}_r = \frac{\sum_{i\in r}\text{VA}_i\,g^{\text{nom}}_i}{\sum_{i\in r}\text{VA}_i},
\qquad
G^{\text{real}}_r = \frac{1+G^{\text{nom}}_r}{1+D_r}-1. \tag{5} $$

Per-good real GVA change uses the same regional deflator:
$g^{\text{real}}_i=(1+g^{\text{nom}}_i)/(1+D_{r(i)})-1$.

**Reading the two engines.** With Engine 1 (prices only), $\Delta x_i=0$, so $G^{\text{nom}}_r=D_r$
and $G^{\text{real}}_r=0$: a price-only model produces **inflation but no real GDP movement** — the
honest statement that it says nothing about real quantities. With Engine 2, volumes fall, so
$G^{\text{real}}_r<0$ under a carbon price, with a low/central/high band inherited from the demand
elasticities.

## 5. Algorithm

1. Compute base-year $\text{VA}_i$ from the run's IO system via (1).
2. Read `price_change` (band `central`) and, if present, `volume_change` (per band) from the result.
3. For each year and band: form $g^{\text{nom}}_i$ (3), then per region the deflator (4) and GDP
   changes (5); emit real GVA/GDP by deflating.
4. Append the rows to the `ResultSet`. Region-level rows (GDP, deflator) use the sentinel sector
   `__economy__`. The step is engine-agnostic (applied in the runner), a no-op when a result has no
   price rows, and idempotent (won't double-add).

Emitted variables: `gva_change`, `gva_change_real` (per sector×region); `gdp_change`,
`gdp_change_real`, `deflator` (per region, sector `__economy__`). Uncertainty bands propagate.

## 6. Calibration / parameters

No new parameters: value added comes from the IO system; the responses come from Engines 1–2. The
only modelling choice is the aggregation weight (value added → a GDP deflator); a CPI (final-demand
weighted) is the natural GE-tier companion and is added with the CGE.

## 7. Validation

Implemented as the `macro` **validation suite** (`cge.validation.suites.macro`), run by
`cge validate` and gated in CI via `tests/test_validation.py`; unit tests in
`tests/test_accounting.py`. Current checks:

| Check | Property (equation) |
|---|---|
| `gdp_identity_production_equals_expenditure` | $\sum\text{VA}=\sum y$ at the base year (2) |
| `zero_shock_zero_aggregates` | no shock ⇒ every aggregate is 0 |
| `real_equals_nominal_at_zero_inflation` | real $=(1+\text{nom})/(1+D)-1$ holds exactly (5) |
| `price_only_engine_has_zero_real_gdp` | Engine 1 ⇒ $D_r>0$ but $G^{\text{real}}_r=0$ |
| `carbon_price_lowers_real_gdp` | Engine 2 ⇒ $G^{\text{real}}_r<0$ everywhere |
| `gdp_aggregates_sector_gva` | region GDP change $=$ VA-weighted mean of sector GVA changes (5) |

## 8. References

- Miller, R. E. & Blair, P. D. (2009). *Input–Output Analysis: Foundations and Extensions* (2nd ed.),
  §2.2 (value added from the IO accounts), §2.3–2.6 (Leontief quantity/price duals). See
  `docs/references.md`.
- National-accounts identity (production = expenditure GDP): standard SNA 2008 accounting.
