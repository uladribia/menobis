# PLAN.md — Outstanding Work

## 1. Router as Single Dispatch Point — COMPLETE

The unified router (`menobis.routing.{fit_model, sample_model, filter_model}`)
is the single entry point for all model operations:

- **fit_model**: All 12 family x constraint full fits + all 12 partial fits
  (via `known_source`/`known_target`/`known_rate` params)
- **sample_model**: All 12 family x constraint combinations
- **filter_model**: All 12 family x constraint combinations

Public API: `fit_model`, `sample_model`, `filter_model`, plus enums
`ModelFamily`, `Constraint`, `Ensemble`, and `UnsupportedModelCaseError`.

The benchmark CLI still uses manual dispatch for sampling and filtering to pass
extra runtime context. This can be refactored in a follow-up.

---


## 3. Solver Convergence Issues

Documented in `docs/decisions/convergence-issues.md`. Summary:

| Case | Status | Proposed fix |
|---|---|---|
| W strength-edges | rarely converges | barrier + adaptive damping |
| W strength-degree | rarely converges | barrier + adaptive damping |
| ME/B strength-edges (very sparse) | fragile | scaled log-lambda + better init |

These are algorithm improvements, not missing plumbing.
