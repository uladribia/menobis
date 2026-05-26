---
description: Automatically peel saturated constraints for boundary convergence.
---

# 0008 Saturation peeling

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

# Strength-degree saturation handling

TL;DR: ME/W/Wnb strength-degree solvers now keep degree-saturated occupation multipliers fixed at a large value while continuing to fit strength multipliers.

## Context

In coupled strength-degree models, a node with degree equal to its candidate-pair capacity must occupy every admissible pair. The binary channel saturates, but the weight channel still has to fit node strengths.

## Decision

For ME, geometric W, and negative-binomial W strength-degree fitting:

| Saturated constraint | Solver behavior |
|---|---|
| `k_out[i] = capacity` | Fix `z_i` to a large multiplier |
| `k_in[j] = capacity` | Fix `w_j` to a large multiplier |
| Strength sequence | Continue updating `x_i` and `y_j` |
| Non-saturated degrees | Continue coordinate updates |

This represents the limiting zero-inflated equations where occupation tends to one while positive-support weights remain family-specific.

## Consequences

- Boundary feasible cases no longer fail solely because a degree reaches capacity.
- The fitted multipliers can be very large for saturated nodes.
- The implementation preserves separate ME/W/Wnb expectation formulas.

## Verification

`tests/test_menobis_strength_degree_saturation.py` covers ME, W, and Wnb saturated degree cases.

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
