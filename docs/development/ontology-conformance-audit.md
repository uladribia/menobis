---
description: Code audit against the MENoBiS model ontology.
---

# Ontology conformance audit

> TL;DR: MENoBiS conforms to the thesis model ontology. ME, B, and W have
> separate fitting/sampling/filtering paths sharing common infrastructure.
> Cost handling uses projected coordinates only. The **dense regime**
> (`average_degree = N/5`) is now the canonical benchmark regime and exercises
> the ontology correctly across all constraint types for ME and B. Remaining
> gaps are W solver performance for zero-inflated constraints.

## Conforming building blocks

| Area | Why it aligns |
|---|---|
| `crates/menobis-core/src/distribution.rs` | Implements Poisson, Binomial(M), Geometric, NegBin(M), and all zero-inflated variants with correct $G_F(q)$ partition factors. |
| `crates/menobis-core/src/pairs.rs` | `PairCostProvider` trait with `EuclideanCostProvider`; strength-edges/degree providers use thesis $G_F(q)$ formulas. |
| Provider-backed generation/filtering | One Rust stream for sampling, p-values, and absent-edge scans. |
| Conditional filtering p-values | `P(X≥w|X>0)` for observed edges eliminates selection bias. |
| B/W family-specific kernels | Each family has its own solver; no ME-relabeling. |
| Unified Python routing | `fit_model(ensemble, family, constraint)` with `UnsupportedModelCaseError`. |
| Coordinate-only costs | `EuclideanCostProvider` generates $d_{ij}$ on demand; no dense matrices. |
| Partial fitting | Uses sparse `PairMask` O(N+K), calls masked balance on excess sequences. |

## Regime conformance

The ontology prescribes three regimes that map to different solver behaviour:

| Regime | Ontology conformance | Empirical validation |
|---|---|---|
| **Dense** (`N/5`, 8.0 events/edge) | **Full** — moderate connectivity, heterogeneous degrees, no saturated nodes | ME + B converge at 100% across all 4 constraint types at N ≤ 1000. Memory: ~215 MB RSS at N=1000 with 2% known pairs. |
| Sparse (avg deg=3, 3 events/edge) | Partial — `k ≈ s` makes degree constraints ill-posed | B degree-overflow at N=5000; degree-constrained solvers converge but provide little added information over strength-only. |
| Saturated (avg deg=0.85(N−1), 8.0 events/edge) | Partial — `k ≈ N` means degree constraints are saturated | B strength-cost IPF plateaus at N ≥ 1000. Fast convergence but unrealistic scenario. |

## Remaining limitations

| Area | Status | Impact |
|---|---|---|
| W strength-edges | Never converges at any N | ZI W formula unstable; all q approaches the upper bound 1.0. |
| W strength-degree | Never converges at any N | Per-node ZI multipliers amplify numerical instability. |
| W strength-cost partial | Diverges at all N | Conic solver boundary: q → 1.0 for high-strength nodes. |
| W strength-cost `self_loops=False` at N≥500 | Slow (~500s at N=500) | Newton needs adaptive damping. |
| B strength-cost saturated N≥1000 | IPF plateau within 10k iters | Residual ~10⁻¹⁰ but never drops below tolerance. |
| B strength-degree sparse N=5000 | Overflow in z multipliers | z ~ 1e+262 exceeds float64 range. |
| Constraint-level code factoring | Not implemented | Internal quality; doesn't affect correctness. |
| Partial ≠ full with empty known set | Separate entry points | Same math, different functions; cosmetic. |