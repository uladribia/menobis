# ADR-0014: Sparse Pair Mask

## Status

Accepted (complete)

## Context

The fitting module used dense `Vec<bool>` of size N² to represent excluded
pairs (self-loops + known/frozen pairs). At N=10000, this is 100MB per mask.
Most masks are extremely sparse: only N diagonal entries + K known pairs
(K << N²).

## Decision

Introduce `PairMask` struct with O(N+K) memory that stores per-row and
per-column lists of masked indices. For linear IPF sums (ME strength), this
enables O(K_i) per-row/column corrections instead of O(N) iteration:

```
free_col_sum(j) = full_sum(x) - sum(x[i] for i in masked_rows[j])
```

All non-partial (full) fitting functions now delegate to the corresponding
sparse-masked versions with a `PairMask::from_self_loops(n, self_loops)`:

| Non-partial function | Sparse delegate |
|---|---|
| `balance_strength_poisson` | `balance_sparse_masked_strength_poisson` |
| `balance_degree_bernoulli` | `balance_sparse_masked_degree_bernoulli` |
| `balance_strength_binomial` | `balance_sparse_masked_strength_binomial` |
| `balance_strength_degree_poisson` | `balance_sparse_masked_strength_degree_poisson` |

Dense masked functions (`balance_masked_*`) are retained as crate-internal
for equivalence tests but removed from the public API.

## Consequences

- Memory: O(N+K) instead of O(N²) for mask storage
- Computation: O(K_i) per node for linear sums (ME strength IPF)
- Nonlinear solvers (Bernoulli, strength-degree, binomial) still iterate
  all N candidates per node but use `is_masked(i, j)` HashSet lookup
- Unified code path: non-partial functions are now thin wrappers over
  the same sparse-masked implementations used by partial fitting
- `PairMask::from_dense()` bridges saturation peeling (which still
  produces dense masks internally)
