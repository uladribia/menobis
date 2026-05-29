# PLAN.md — Partial Fitting: Missing Cases

## Status

The partial fitting plumbing (known pairs via `PairMask`) is now complete for
all cases where the underlying Rust solver accepts an external mask.

## Implemented partial fitting

| Constraint | ME | B | W |
|---|---|---|---|
| strength | ✓ | ✓ | ✗ |
| strength-cost (coordinates) | ✓ | ✓ | ✓ |
| strength-edges | ✓ | ✓ | ✓ |
| strength-degree | ✓ | ✓ | ✗ |

## Missing: W partial strength

**Root cause**: `fit_strength_w_newton` in `crates/menobis-core/src/fitting/w_lbfgs.rs`
takes `opts: &CostFitOptions` which only contains `self_loops: bool`. It internally
creates `PairMask::from_self_loops(n, opts.self_loops)`.

**Fix**: Add a `mask: &PairMask` parameter to `fit_strength_w_newton` (or create a
`fit_strength_w_newton_masked` variant), then wire through PyO3 + Python.

**Effort**: ~30 lines Rust + 20 lines PyO3 + 20 lines Python.

## Missing: W partial strength-degree

**Root cause**: `fit_strength_degree_w_newton` in `crates/menobis-core/src/fitting/w_lbfgs.rs`
takes `self_loops: bool` and internally constructs the mask via
`PairMask::from_self_loops(n, false)`.

**Fix**: Add a `mask: &PairMask` parameter (or create a masked variant), then wire
through PyO3 + Python.

**Effort**: ~30 lines Rust + 20 lines PyO3 + 20 lines Python.

## Priority

Low — these W solvers already have convergence issues with heterogeneous inputs
(documented in AGENTS.md). Fixing the convergence is higher priority than adding
partial support for non-convergent solvers.
