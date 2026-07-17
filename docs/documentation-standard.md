# Documentation standard

Every engine, module, and non-trivial data transformation in this project ships with a
**model-description document** that explains the method *to equation level* and cites the
peer work it derives from. This is a definition-of-done criterion, not an optional extra:
a phase is not complete until its model doc exists.

The reason is credibility. This is a screening/stress tool whose numbers people may act
on; the single thing that makes it trustworthy is that any result can be traced to a
written-down method and a published source. "Precise about costs, indicative about
volumes, **transparent about assumptions**" (roadmap §6) is only true if the transparency
is actually written down.

## What every model-description doc must contain

1. **Purpose & scope** — the question it answers, and explicitly what it does *not* model.
2. **Notation** — every symbol defined once, in a table, with units.
3. **Assumptions** — stated as a numbered list, each one falsifiable and, where possible,
   linked to the parameter or data that encodes it. These must match the `assumptions`
   dump the engine writes into its `RunManifest` (see `contracts/provenance.py`).
4. **Derivation to equation level** — the actual equations, numbered, with the algebra
   that gets from inputs to outputs. Not prose gesturing at "a Leontief calculation" — the
   matrix expression, its dimensions, and why it is well-posed (e.g. existence of the
   inverse). A reader should be able to reimplement from the doc alone.
5. **Algorithm** — how the equations are actually computed (solver, iteration scheme,
   convergence criterion, complexity), and where the hot path is.
6. **Calibration / parameters** — where every number comes from, with per-parameter
   sourcing and uncertainty where relevant.
7. **Validation** — the known-answer and identity tests that hold the implementation to
   the equations (link to the tests in `tests/`).
8. **References** — full citations (see `docs/references.md`), keyed so the equations and
   assumptions cite them inline, e.g. "following Miller & Blair (2009, §2.3)".

## Equations

Write mathematics in LaTeX inside `$…$` (inline) and `$$…$$` (display) fences, GitHub
renders it. Number display equations `\tag{n}` and refer to them by number in the prose.
Keep symbol conventions consistent across docs (the notation table in each doc is the
contract; cross-doc symbols should not clash without saying so).

## Referencing peer work

- Prefer **peer-reviewed papers** and standard **textbooks** over blogs or vendor docs.
- Every non-obvious modelling choice cites where it is done in the literature — especially
  the load-bearing ones (SAM balancing method, Armington elasticities, ENCORE↔EXIOBASE
  concordance, damage functions). Reviewers who know this field *will* check these.
- When we deviate from a cited method, say so and say why.
- Central-bank / institutional methodology reports (NGFS, DNB, ECB/EIOPA, World Bank) are
  citable and often the right anchor for the applied choices — cite the report, not a
  press summary.
- All citations live in `docs/references.md`; docs cite by key. One source of truth.

## Docstrings vs model docs

Code docstrings explain *what a function does and why it exists* for someone reading the
code. Model docs explain *the method and its provenance* for someone deciding whether to
trust the numbers. Both are required; they are not substitutes. A docstring on the core
method should point to its model doc (`See docs/models/<engine>.md`).

## Template

New model docs start from [`models/_template.md`](models/_template.md).
The Engine 1 doc [`models/io-price-model.md`](models/io-price-model.md) is the worked
reference example of the standard.
