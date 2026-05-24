---
description: Code audit against the AGENTS model ontology.
---

# Ontology conformance audit

## TL;DR

After correcting the zero-inflated formulas in `AGENTS.md`, the Rust
`PairDistribution`/provider formulas are mostly aligned. The largest remaining
code gaps are B zero-inflated fitters calling ME fitters, degree-events positive
weight parameters, missing unified Python routing, and partial B/W logic running
in Python instead of Rust.

## Conforming building blocks

| Area | Why it aligns |
|---|---|
| `crates/odme-core/src/distribution.rs` | Implements zero-truncated Poisson, Binomial, Geometric, and Negative-Binomial expectations. |
| `crates/odme-core/src/pairs.rs` | Strength-edges and strength-degree providers use the required $G_F(q)$ partition factors for ME/B/W. |
| Provider-backed generation/filtering | Reuses one Rust stream for sampling, p-values, and absent-edge scans. |
| B fixed-strength and B strength-cost | Use B-specific expected-weight equations. |
| W fixed-strength/cost/edge/degree APIs | Have separate W entry points rather than public ME aliases. |

## Non-conforming code and fixes

| Area | Non-conformance | Fix |
|---|---|---|
| `fit_strength_edges_binomial` | Calls `_odme.fit_strength_edges_poisson` and relabels the result as B. | Add Rust `fit_strength_edges_binomial` with B zero-inflated kernel; expose via PyO3; update Python wrapper. |
| `fit_strength_degree_binomial` | Calls `_odme.fit_strength_degree_poisson` and relabels the result as B. | Add Rust B strength-degree solver using B mean and B occupation channel. |
| `fit_degree_events_binomial` | Derives `q = 1 - E/T`, which is the geometric positive-mean inverse, not B. | Solve $M q(1+q)^{M-1}/((1+q)^M-1)=T/E$ and reject infeasible `T/E > M`. |
| `fit_degree_events_poisson` | Stores a `q` that is not the positive Poisson rate used by samplers/filters. | Store the zero-truncated Poisson rate solving $q/(1-e^{-q})=T/E`; let samplers read it from `DegreeEventsFit`. |
| Degree-events APIs | Some samplers/filters still take manual positive-weight parameters. | Make `DegreeEventsFit` carry all parameters and remove external rate/q arguments. |
| Python public API | No unified router for ensemble + family + constraint. | Add `fit_model(...)` or equivalent with enums and custom unsupported-case errors; keep family-specific endpoints underneath. |
| `src/odme/models/partial.py` | B/W coordinate partials build dense masks and assemble rate tables in Python. | Move excess, mask, full-solver call, and rate-table assembly to Rust for every family. |
| `crates/odme-core/src/fitting/partial.rs` | Some ME partial paths duplicate masked inner solver logic. | Make full solvers support a free-pair mask and call them from partial paths. |
| Cost handling across fitting modules | Cost providers are ad hoc; B sparse strength-cost treats sparse entries differently from ME/W missing-cost-as-zero semantics. | Introduce `CostProvider` and `FamilyKernel` traits/factories shared by ME/B/W, with one support/missing-cost policy. |
| Negative-binomial sample/filter defaults | Several negative-binomial APIs allow `layers=1`, which should be geometric. | Reject `layers <= 1` or route explicitly to geometric. |
| W no-self-loop fitting | AGENTS records non-convergence at realistic `N >= 50`. | Add adaptive damping/feasibility projection or an accelerated convex solver; test generated gravity-like networks. |
| B no-self-loop strength fitting | AGENTS records slow convergence at `N >= 200`. | Benchmark IPF residual path; add acceleration or better saturation projection. |
| Dense Python helpers | Coordinate partial helpers and complete cost triples can allocate `N x N` arrays. | Prefer Rust on-the-fly coordinate providers and sparse support iteration. |

## Red/green repair order

1. Add failing B strength-edges and B strength-degree recovery tests showing
   ME, B, and W differ on the same feasible generated constraints.
2. Add degree-events formula tests for ME, B, and W positive-weight parameters.
3. Implement B zero-inflated Rust kernels and route Python wrappers to them.
4. Add the unified Python router with unsupported-case errors.
5. Move partial B/W coordinate logic fully into Rust.
6. Fix W no-self-loop convergence and B no-self-loop performance regressions.
