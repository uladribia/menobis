---
description: Code audit against the MENoBiS model ontology.
---

# Ontology conformance audit

## TL;DR

MENoBiS conforms to the thesis model ontology. ME, B, and W have separate
fitting/sampling/filtering paths sharing common infrastructure. Cost handling
uses projected coordinates only. Remaining gaps are solver performance at
scale for W no-self-loop cases.

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

## Resolved non-conformances

| Area | Resolution |
|---|---|
| Cost providers ad hoc | Replaced with `PairCostProvider` trait + `EuclideanCostProvider`. |
| Sparse cost triples in public API | Removed; coordinate-only. |
| Dense pair masks `Vec<bool>` N² | Replaced with sparse `HashSet`-backed `PairMask`. |
| No unified routing | Fixed: enum-based router for fit/sample/filter. |
| Filter FPR N-dependent | Fixed: conditional p-values. |

## Remaining limitations

| Area | Status | Impact |
|---|---|---|
| W strength-cost `self_loops=False` at N≥500 | ~500s fitting time | Usable but slow; needs Newton damping improvement. |
| B `self_loops=False` at N≥200 | Resolved: 4ms at N=300 | No longer reproducible. |
| Constraint-level code factoring | Not implemented | Internal quality; doesn't affect correctness. |
| Partial ≠ full with empty known set | Separate entry points | Same math, different functions; cosmetic. |
