# Model description: <engine / module name>

- **Implements:** `<module path, e.g. cge.engines.io_price>`
- **Roadmap phase:** <n>
- **Capabilities:** <prices | volumes | general_equilibrium | dynamic>
- **Status:** draft | implemented | validated

## 1. Purpose & scope

What question this answers. **What it deliberately does not model** (the boundary is as
important as the content — it is what stops the numbers being over-interpreted).

## 2. Notation

| Symbol | Meaning | Units |
|---|---|---|
| $n$ | number of sector×region entries | — |
| … | … | … |

## 3. Assumptions

1. …  (each falsifiable; must match the engine's `RunManifest.assumptions`)

## 4. Derivation

State the model to equation level. Number display equations and refer to them by number.
Give matrix dimensions and argue well-posedness (existence/uniqueness of the solution).
Cite where each step is standard, e.g. "following Author (Year, §x)".

## 5. Algorithm

How the equations are computed: solver / iteration / convergence criterion, complexity,
and the hot path. Note where performance work would go (cf. ADR-0003).

## 6. Calibration / parameters

Where each parameter comes from, with per-parameter sourcing and uncertainty.

## 7. Validation

The known-answer and identity tests that hold the code to the equations. Link to
`tests/…`. State tolerances and the published numbers checked against.

## 8. References

Cite by key into `docs/references.md`.
