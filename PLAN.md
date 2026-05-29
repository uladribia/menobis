# PLAN.md — Outstanding Work

## 1. Router as Single Dispatch Point — COMPLETE

The unified router (`menobis.routing.{fit_model, sample_model, filter_model}`)
is the single entry point for all model operations. All 12 family x constraint
combinations work for fit, sample, and filter.

Public API: `fit_model`, `sample_model`, `filter_model`, plus enums
`ModelFamily`, `Constraint`, `Ensemble`, and `UnsupportedModelCaseError`.

The benchmark CLI (`benchmarks/cli.py`) still uses manual dispatch for sampling
and filtering because it needs to pass extra runtime context (network coordinates)
that aren't stored in the fit result. Once the benchmark is refactored to use
the router exclusively, ~100 lines of manual dispatch can be deleted.

---


## 3. Solver Convergence Issues

Documented in `docs/decisions/convergence-issues.md`. Summary:

| Case | Status | Proposed fix |
|---|---|---|
| W strength-edges | rarely converges | barrier + adaptive damping |
| W strength-degree | rarely converges | barrier + adaptive damping |
| ME/B strength-edges (very sparse) | fragile | scaled log-lambda + better init |

These are algorithm improvements, not missing plumbing.
