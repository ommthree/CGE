# ADR-0003: Python as the implementation language; a performance strategy, not a rewrite

- **Status:** accepted
- **Date:** 2026-07-17
- **Phase:** 0

## Context

A fair question was raised: is Python the right language if we might want to be *fast*?
Worth answering deliberately, because the language choice is expensive to reverse once
engines exist.

## Decision

Build in Python, and treat performance as an architecture question (where does time go,
and what handles the hot loops) rather than a language question.

The reasoning:

1. **The ecosystem is the moat.** `pymrio` (EXIOBASE handling), `pyomo` + IPOPT (the CGE
   solver), FaIR (climate), the entire scientific-data stack, and Streamlit all live in
   Python. In a faster language we would reimplement or FFI-bridge these — that cost
   dwarfs any raw-speed gain.

2. **The heavy math already runs in C/Fortran.** The Leontief inverse is a NumPy/SciPy
   LAPACK call; the CGE solve is IPOPT (compiled). Python is orchestration around
   compiled kernels. For a 40–60-sector build these solves are sub-second to seconds.

3. **The real cost is usually I/O and memory**, not CPU — parsing/reshaping the ~9800²
   MRIO tables. The fix is data engineering (parquet, DuckDB, float32, the pre-aggregated
   small build), which is language-independent.

**Performance strategy (in order of preference):**
1. Right-size the problem — interactive work runs on the small build, not the full MRIO.
2. Vectorise (NumPy/SciPy); never hand-loop matrix math in Python.
3. Push data ops into DuckDB/Arrow.
4. If a genuine Python-level hot loop survives all that (e.g. structural path analysis,
   fixed-point PE iteration), reach for Numba/Cython on *that function* — the contracts
   make an engine's internals a black box, so this is a local change.

**When Python would be the wrong call (not our case):** hard low-latency serving, or a
massive high-resolution intertemporal-optimisation IAM. We are explicitly building a
recursive-dynamic CGE consuming exogenous pathways (see roadmap P7), which does not need
that.

## Consequences

- We keep the whole open modelling ecosystem for free and stay readable/hackable for a
  solo maintainer.
- Performance work, when needed, is targeted and local (one engine function), never a
  language migration — the engine Protocol (ADR-0002) is the seam that makes a hot engine
  swappable, including for a compiled reimplementation behind the same interface.
- We accept that naive pure-Python loops are slow, and commit to vectorised/compiled
  kernels for anything hot.

## Alternatives considered

- **Julia:** genuinely fast with good numerics and a real CGE/JuMP story; the blocker is
  the data/nature ecosystem (no `pymrio`, no ENCORE tooling, thinner EXIOBASE support) —
  we'd spend the saved CPU time rebuilding data plumbing. Reasonable to revisit *only* if
  the CGE core ever becomes the bottleneck and the data layer has stabilised.
- **Rust/C++ core + Python bindings:** the right shape *if* a specific kernel proves too
  slow — and the architecture already permits it per-engine. Doing it up front optimises a
  cost we have not yet measured.
- **Keep the door open:** because engines sit behind a Protocol, a future hot engine can be
  a Rust/Julia implementation exposed to Python without touching data, scenarios, or GUI.
