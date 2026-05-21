---
description: Automatically peel saturated constraints for boundary convergence.
---

# 0008 Saturation peeling

## Status

Accepted and implemented (B strength + all degree-events families).

## Context

IPF solvers for degree-constrained and B(M) strength models diverge when a
node's constraint equals the theoretical maximum (capacity). The Lagrange
multiplier would need to be infinite, causing oscillation.

## Decision

Before entering the IPF loop, scan for saturated nodes:

- **Degree saturation**: `k_i ≥ capacity` (N with self-loops, N−1 without)
- **B strength saturation**: `s_i ≥ M × capacity`

For each saturated node, mark all its outgoing (or incoming) pairs as
deterministically known:

- Degree models: occupation = 1 (all edges present)
- B strength: weight = M (maximum binomial weight)

Compute excess constraints for the remaining free sub-problem, rebalance if
clamping introduced imbalance, then solve using the existing masked IPF solver.

Assign large multipliers (1e6) to saturated nodes in the output so downstream
sampling produces near-deterministic behavior (Option B).

## Affected solvers

| Solver | Saturation detected | Families benefiting |
|---|---|---|
| `balance_degree_bernoulli` | node degree = capacity | ME, B, W, Wnb degree-events |
| `balance_strength_binomial` | node strength = M × capacity | B strength-only |

## Not applied: strength-degree coupled models

Degree saturation in the coupled strength-degree case (ME, W, Wnb) requires
the strength x/y multiplier channel to remain active for per-pair weight
fitting even when occupation is forced to 1. This would need a mixed-constraint
solver where:

- Saturated-degree nodes have **strength-only** constraints (positive-conditional formula)
- Free nodes have **both** strength and degree constraints

This is architecturally different from simple peeling and remains a future
improvement. **Users encountering degree = capacity in strength-degree models
should clip degrees to `capacity - 1` or use the degree-events formulation.**

## Sampling consequences (Option B)

Large multipliers (1e6) produce near-deterministic sampling:

- B: `Binomial(M, p≈1)` → weight = M with P > 0.999999
- Bernoulli: `Bernoulli(p≈1)` → occupation = 1 with P > 0.999999
- Error: < 1e-6 per pair per sample — invisible in practice

No mask propagation to the sampler is needed.

## Consequences

| Benefit | Cost |
|---|---|
| Boundary B/degree-events inputs converge in few iterations | Slight complexity in support.rs |
| No more rejection of feasible boundary degrees | Rebalancing after clamping is approximate |
| Same public API, no user-facing changes | Not applicable to strength-degree coupled models |
