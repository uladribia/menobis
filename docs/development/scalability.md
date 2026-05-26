---
description: Complexity, memory model, and practical scaling limits.
---

# Scalability

## TL;DR

MENoBiS avoids dense matrices for generation and filtering where possible. Fitting
some constraints is inherently all-pairs and scales as O(N²) times solver
iterations.

## Operation costs

| Operation | Typical complexity | Notes |
|---|---:|---|
| edge-list analysis | O(E) | single pass |
| strength fit | O(N) or cheap IPF | analytic for Poisson |
| degree-events fit | O(N² I) | usually few iterations |
| strength-edges fit | O(N² I) | expensive at N >= 1000 |
| strength-degree fit | O(N² I) | highest dense-IPF cost |
| strength-cost fit | O(N² I) plus cost lookup | cost search; costs should stream or compute on demand |
| generation | O(P + Es) | P candidate pairs, Es sampled edges |
| filtering | O(E) or O(P) | absent-edge filtering can use full support |

## Memory principles

| Data | Preferred representation |
|---|---|
| observed network | sparse edge list |
| sampled network | sparse edge list |
| custom probabilities | sparse triples |
| multipliers | O(N) arrays |
| costs | sparse triples or on-the-fly functions; avoid dense triples |
| dense support | avoided except all-pairs fitting |

## Current practical limits

| Solver | Comfortable range | Limiting factor |
|---|---:|---|
| strength Poisson | very large | O(N) |
| degree-events | 5000+ | O(N²), low iterations |
| ME/B strength-edges | 1000 | O(N² I) |
| ME/B strength-degree | 1000 | O(N² I), high constant |
| W strength-edges | 1000 | W zero-inflated kernels |
| W strength-degree | 1000 | W zero-inflated kernels |
| W strength/cost | 100-500 today | O(N² I), sensitive to heterogeneity and self-loop policy |

## Release-run lesson

The attempted all-case N=5000 benchmark timed out after B strength-degree. That
is expected from dense all-pair constraints: N=5000 means 25 million candidate
pairs per sweep, repeated hundreds of times.

## Future improvements

1. Incremental benchmark persistence.
2. Chunked benchmark presets.
3. On-the-fly or factorized cost providers for large strength-cost fits.
4. Sparse support fitting for user-provided masks.
5. Better stopping diagnostics for W coordinate solvers.
6. Parallel all-pairs sweeps where reproducibility is unaffected.
