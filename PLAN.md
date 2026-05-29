# PLAN.md — Outstanding Work

## 1. Router as Single Dispatch Point — MOSTLY COMPLETE

The unified router (`menobis.routing.{fit_model, sample_model, filter_model}`)
is the single entry point for all full-model operations. All 12 family x
constraint combinations work for fit, sample, and filter.

Public API: `fit_model`, `sample_model`, `filter_model`, plus enums
`ModelFamily`, `Constraint`, `Ensemble`, and `UnsupportedModelCaseError`.

### Remaining gap: partial fitting not routed

Partial fitting (known pairs) is not accessible through the router. Users must
import `fit_partial_*` functions directly from `menobis.models.partial`.

**What to do**:
1. Add `known_source`, `known_target`, `known_rate` optional parameters to
   `fit_model`.
2. When these are provided, dispatch to the corresponding `fit_partial_*`
   function instead of the full-fit function.
3. Return a `PartialFitResult` (which is already a subtype of the general
   result pattern).

**Effort**: ~40 lines in routing.py.

The benchmark CLI still uses manual dispatch for sampling, filtering, and
partial fitting because it needs extra runtime context. Once partial is routed,
the benchmark `_fit_partial` function (~80 lines) can be replaced with a single
`fit_model(...)` call.

---


## 3. Solver Convergence Issues

Documented in `docs/decisions/convergence-issues.md`. Summary:

| Case | Status | Proposed fix |
|---|---|---|
| W strength-edges | rarely converges | barrier + adaptive damping |
| W strength-degree | rarely converges | barrier + adaptive damping |
| ME/B strength-edges (very sparse) | fragile | scaled log-lambda + better init |

These are algorithm improvements, not missing plumbing.
