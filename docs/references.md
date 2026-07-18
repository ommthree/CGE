# References

Single source of truth for citations. Docs cite by the **key** in the left column, e.g.
"following [MillerBlair2009]". Keep entries in this format; add as phases introduce new
methods. Prefer peer-reviewed papers and standard textbooks; institutional methodology
reports are citable for applied choices.

## Input–output analysis

- **[MillerBlair2009]** Miller, R.E. & Blair, P.D. (2009). *Input–Output Analysis:
  Foundations and Extensions*, 2nd ed. Cambridge University Press. — The standard
  reference for the Leontief quantity and price models, structural path analysis, and
  environmentally-extended IO. Ch. 2 (quantity & price models), Ch. 12 (SPA), Ch. 10
  (energy/environment extensions).
- **[Leontief1970]** Leontief, W. (1970). Environmental repercussions and the economic
  structure: an input–output approach. *Review of Economics and Statistics*, 52(3),
  262–271. — Origin of the environmentally-extended IO framework.

## EXIOBASE / MRIO data

- **[Stadler2018]** Stadler, K. et al. (2018). EXIOBASE 3: Developing a time series of
  detailed environmentally extended multi-regional input–output tables. *Journal of
  Industrial Ecology*, 22(3), 502–515. — The EXIOBASE 3 database this project builds on.
- **[Wood2015]** Wood, R. et al. (2015). Global sustainability accounting—developing
  EXIOBASE for multi-regional footprint analysis. *Sustainability*, 7(1), 138–163.

## CGE / SAM (Phase 5)

- **[Hosoe2010]** Hosoe, N., Gasawa, K. & Hashimoto, H. (2010). *Textbook of Computable
  General Equilibrium Modelling: Programming and Simulations*. Palgrave Macmillan. — The
  "toy but honest" static CGE reference the P5 core follows.
- **[Robinson2001]** Robinson, S., Cattaneo, A. & El-Said, M. (2001). Updating and
  estimating a social accounting matrix using cross entropy methods. *Economic Systems
  Research*, 13(1), 47–64. — SAM balancing via cross-entropy (P5.1).
- **[Armington1969]** Armington, P.S. (1969). A theory of demand for products
  distinguished by place of production. *IMF Staff Papers*, 16(1), 159–178. — The
  Armington import-substitution assumption (P4, P5).

## Nature / ENCORE (Phase 6)

- **[ENCORE]** ENCORE Partners (Natural Capital Finance Alliance & UNEP-WCMC). *ENCORE:
  Exploring Natural Capital Opportunities, Risks and Exposure.* — The dependency/impact
  knowledge base. Cite the accessed version/date.
- **[vanToor2020]** van Toor, J. et al. (2020). *Indebted to nature: Exploring
  biodiversity risks for the Dutch financial sector.* De Nederlandsche Bank / PBL. — Source
  of the ENCORE↔economic-sector mapping approach seeded in P6.2.

## Climate / pathways (Phase 7)

- **[NGFS]** Network for Greening the Financial System. *NGFS Climate Scenarios* (and the
  IIASA scenario database). — Exogenous pathway inputs (P7.2). Cite phase/vintage used.
- **[Leach2021]** Leach, N.J. et al. (2021). FaIRv2.0.0: a generalized impulse response
  model for climate uncertainty and future scenario exploration. *Geoscientific Model
  Development*, 14(5), 3007–3036. — The FaIR climate emulator (P7.3).
- **[Nordhaus2017]** Nordhaus, W. (2017). Revisiting the social cost of carbon. *PNAS*,
  114(7), 1518–1523. — DICE damage function (P7.4, optional).
- **[Burke2015]** Burke, M., Hsiang, S.M. & Miguel, E. (2015). Global non-linear effect of
  temperature on economic production. *Nature*, 527, 235–239. — Econometric damage
  estimates, alternative to DICE (P7.4).
- **[MatthewsCarbonBudget]** Matthews, H.D. et al. (2009). The proportionality of global
  warming to cumulative carbon emissions. *Nature*, 459, 829–832. — The near-linear
  temperature ↔ cumulative-CO₂ relationship that makes the temperature-target back-solve a
  well-behaved 1-D inversion (see energy-and-temperature-plan.md).
- **[IPCC_AR6_WG3]** IPCC (2022). *Climate Change 2022: Mitigation of Climate Change* (AR6
  WG3). — Reference for carbon-budget / temperature-target framings and the scenario families.

## Energy prices & pass-through

- **[Kilian2008]** Kilian, L. (2008). The economic effects of energy price shocks. *Journal of
  Economic Literature*, 46(4), 871–909. — Survey of how energy-price shocks propagate through
  an economy; motivates the `EnergyPrice` shock (energy-and-temperature-plan.md, Feature 1).
- **[HasenzahlDietzenbacher]** Standard IO price-side pass-through of cost shocks (Ghosh /
  Leontief price models) is covered in [MillerBlair2009] §2.4 and §12 — the same machinery the
  `EnergyPrice` shock reuses. (Cite Miller & Blair for the method.)

## Competing methodologies & model families (context, not dependencies)

Where this platform sits relative to established approaches. Cited so the docs are honest about
what is being approximated and by whom.

- **[GTAP]** Aguiar, A. et al. (2019). The GTAP Data Base version 10. *Journal of Global
  Economic Analysis*, 4(1). — The standard multi-region CGE database + framework. **Licensed,
  not open** — the reason this project builds its SAM from EXIOBASE instead (roadmap P5). The
  precision benchmark a solo CGE will not match.
- **[GCAM]** Calvin, K. et al. (2019). GCAM v5.1: representing the linkages between energy,
  water, land, climate, and economic systems. *Geoscientific Model Development*, 12(2). — A
  process-based integrated assessment model (explicit energy system, land use). The kind of
  "real IAM" this project deliberately does *not* attempt; instead it consumes such models'
  pathways (via NGFS) and adds sector resolution.
- **[REMIND]** Baumstark, L. et al. (2021). REMIND2.1: transformation and innovation dynamics
  of the energy-economic system. *Geoscientific Model Development*, 14(10). — Another
  process-based IAM with intertemporal optimisation; contrast with this project's
  recursive-dynamic, no-perfect-foresight approach (roadmap P7.1).
- **[DICE]** Nordhaus, W. (2017), see [Nordhaus2017] — a *cost–benefit* IAM: aggregate economy
  + damage function, no sectoral resolution. Contrast: this project has sector detail but (for
  Interpretation A) no damage feedback. DICE is the reference for the optional P7.4 damage path.
- **[E3ME]** Cambridge Econometrics. *E3ME technical manual.* — A large econometric/IO
  macro-econometric model used in EU policy analysis; a "non-equilibrium" alternative to CGE.
  Named as a methodological neighbour to the IO engines here.
- **[NGFS]** (see above) — the scenario set most financial-sector "IAM-based" tools consume, as
  this project does. The point of comparison for the temperature back-solve's implied price paths.
