---
description: Code audit against the MENoBiS model ontology.
---

# Ontology conformance audit

## TL;DR

MENoBiS now has separate ME, B, and W fitting paths for the main fixed-strength,
strength-cost, strength-edges, and strength-degree families. Remaining ontology
gaps are unified routing and dense Rust masks in partial solvers.

## Conforming building blocks

| Area | Why it aligns |
|---|---|
| `crates/menobis-core/src/distribution.rs` | Implements zero-truncated Poisson, Binomial, Geometric, and Negative-Binomial expectations. |
| `crates/menobis-core/src/pairs.rs` | Strength-edges and strength-degree providers use the required $G_F(q)$ partition factors for ME/B/W. |
| Provider-backed generation/filtering | Reuses one Rust stream for sampling, p-values, and absent-edge scans. |
| B fixed-strength/cost/edge/degree | Use B-specific expected-weight and zero-inflated kernels rather than ME relabeling. |
| W fixed-strength/cost/edge/degree | Have separate W entry points and diagnostics. |

## Non-conforming code and fixes

| Area | Non-conformance | Fix |
|---|---|---|
| Python public API | **Fixed.** Unified router `fit_model(...)` / `sample_model(...)` implemented with `Ensemble`, `Family`, `Constraint` enums and `UnsupportedModelCaseError`. |
| `crates/menobis-core/src/fitting/partial.rs` | Partial paths still use dense pair masks (`Vec<bool>` of `N*N`) and some ME paths duplicate masked inner solver logic. | Add reusable sparse free-pair providers and make full solvers support them. |
| Cost handling across fitting modules | Cost providers are ad hoc; sparse and coordinate paths are not one abstraction. | Introduce `CostProvider` and `FamilyKernel` traits/factories shared by ME/B/W. |
| W no-self-loop fitting | AGENTS records non-convergence at realistic `N >= 50`. | Add adaptive damping/feasibility projection or an accelerated convex solver; test generated gravity-like networks. |
| B no-self-loop strength fitting | AGENTS records slow convergence at `N >= 200`. | Benchmark IPF residual path; add acceleration or better saturation projection. |

## Red/green repair order

1. Add the unified Python router with unsupported-case errors.
2. Replace dense Rust partial masks with reusable sparse free-pair providers.
3. Fix W no-self-loop convergence and B no-self-loop performance regressions.
