# PLAN.md — Partial Fitting Coverage

## Status

All partial fitting cases are now implemented. Every family (ME, B, W) supports
known pairs for every constraint type where the underlying solver can accept a
mask.

## Partial fitting matrix

| Constraint | ME | B | W |
|---|---|---|---|
| strength | ✓ | ✓ | ✓ |
| strength-cost (coordinates) | ✓ | ✓ | ✓ |
| strength-edges | ✓ | ✓ | ✓ |
| strength-degree | ✓ | ✓ | ✓ |

## Known convergence limitations

| Case | Behavior |
|---|---|
| W strength-edges (all) | Often fails to converge with heterogeneous inputs |
| W strength-degree (all) | Often fails to converge with heterogeneous inputs |
| ME/B strength-edges (sparse, near-binary) | L-BFGS oscillates when s/k ≈ 1 |

These are solver algorithm issues, not missing plumbing.

## Implementation notes

- W strength partial uses a new `balance_sparse_masked_strength_w` IPF solver
  in `crates/menobis-core/src/fitting/w.rs` that respects arbitrary `PairMask`.
- W strength-degree partial uses `fit_strength_degree_w_newton_masked` which
  accepts `&PairMask` and replaces `!self_loops && i == j` checks with
  `mask.is_masked(i, j)`.
- The `w_se_inner_solve` helper (used by strength-edges) still uses
  `self_loops: bool` internally — this is acceptable because known pairs for
  strength-edges are handled at the partial pipeline level (excess computation
  removes their contribution before calling the solver).
