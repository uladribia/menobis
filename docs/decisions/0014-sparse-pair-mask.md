# ADR-0014: Sparse Pair Mask

## Status

Accepted (in progress)

## Context

The fitting module uses dense `Vec<bool>` of size N² to represent excluded
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

Migration is incremental: functions are converted one at a time, using
`to_dense()` as a bridge during transition.

## Consequences

- Memory: O(N+K) instead of O(N²) for mask storage
- Computation: O(K_i) per node for linear sums (ME strength IPF)
- Nonlinear solvers (Bernoulli, strength-degree) still iterate all N pairs
  but avoid dense allocation
- `fit_partial_strength` converted first as proof of concept
- Remaining functions to be converted in follow-up commits
