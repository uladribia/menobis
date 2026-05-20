---
description: Full public and private API coherence audit across ME, W, and B families.
---

# API Consistency Audit

## TL;DR

Full coherence audit of fitting, sampling, and filtering across all 5
constraints × 3 families (ME/W/B). Public API naming and parameter ordering are
largely consistent. Six categories of discrepancies are documented below with
recommended actions.

## Coverage matrix

### Fitting

| Constraint | Poisson (ME) | Geometric (W) | Binomial (B) | Neg. Binomial (W) |
|---|:---:|:---:|:---:|:---:|
| strength | ✅ | ✅ | ✅ | ✅ |
| strength-cost | ✅ | ✅ | ❌ missing | ✅ |
| strength-edges | ✅ | ✅ | ❌ missing | ✅ |
| strength-degree | ✅ | ✅ | ❌ missing | ✅ |
| degree-events | ❌ missing | ✅ | ❌ missing | ✅ |

Missing B fits (cost, edges, degree) exist as sampling/filtering but not as
fitting APIs. Missing ME `fit_degree_events_poisson` — the Poisson
degree-events model uses `fit_degree_bernoulli` for occupation and a separate
positive-weight-rate parameter; it is not exposed as a single
`fit_degree_events_poisson`.

### Sampling and filtering

Full coverage: all 5 constraints × 4 families have both sample and filter
functions. ✅

## Discrepancy 1: Result type fields

| Field | `FitResult` | `StrengthCostFit` | `StrengthEdgesFit` | `StrengthDegreeFit` | `DegreeEventsFit` |
|---|:---:|:---:|:---:|:---:|:---:|
| `node` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `x`, `y` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `converged` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `iterations` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `family` | ✅ | ✅ | ✅ | ✅ | ❌ missing |
| `layers` | ✅ | ✅ | ✅ | ✅ | ❌ missing |
| `self_loops` | ❌ missing | ✅ | ✅ | ✅ | ❌ missing |
| `diagnostics` | ✅ | ✅ | ✅ | ✅ | ❌ missing |

**Action required:** Add `family`, `layers`, `self_loops`, and `diagnostics` to
`DegreeEventsFit`. Add `self_loops` to `FitResult`.

## Discrepancy 2: `family` field not populated correctly

| Function | Current `family` | Correct `family` |
|---|---|---|
| `fit_strength_poisson` | `"poisson"` | ✅ correct |
| `fit_strength_geometric` | `"geometric"` | ✅ correct |
| `fit_strength_binomial` | `"poisson"` | ❌ should be `"binomial"` |
| `fit_degree_bernoulli` | `"poisson"` | ❌ should be `"bernoulli"` |
| `fit_strength_degree_poisson` | `"poisson"` | ✅ correct but no `diagnostics` |

**Action required:** Fix `fit_strength_binomial` → `family="binomial"` with
`layers=M`. Fix `fit_degree_bernoulli` → `family="bernoulli"`.

## Discrepancy 3: `diagnostics` not populated consistently

| Function | Has `diagnostics`? |
|---|:---:|
| ME strength (poisson) | ❌ |
| W strength (geometric/NB) | ✅ |
| ME strength-cost (poisson) | ✅ |
| W strength-cost (geometric/NB) | ✅ |
| ME strength-edges (poisson) | ✅ |
| W strength-edges (geometric/NB) | ✅ |
| ME strength-degree (poisson) | ❌ |
| W strength-degree (geometric/NB) | ✅ |
| B strength (binomial) | ❌ |
| B degree (bernoulli) | ❌ |
| W degree-events (geometric/NB) | ❌ (type lacks field) |

**Action required:** Populate `diagnostics` for all fits. For simple analytic
fits (ME Poisson with self-loops), use
`OptimizationDiagnostics(converged=True, status="solved", iterations=0)`. This
makes all results inspectable uniformly.

## Discrepancy 4: Sampling interface style

Independent-strength samplers accept raw `x`, `y` arrays:
- `sample_strength_poisson(x, y, ...)`
- `sample_strength_geometric(x, y, ...)`
- `sample_strength_binomial(x, y, ...)`
- `sample_strength_negative_binomial(x, y, ...)`

All other samplers accept a fit object:
- `sample_strength_cost_poisson(fit, cost_sources, cost_targets, cost_values, ...)`
- `sample_strength_edges_geometric(fit, ...)`

**Assessment:** This is **justified**. The independent-strength model has a
trivial relationship between fit result and sampler parameters (`x`, `y` are the
only parameters). Forcing a fit object would add unnecessary coupling for the
simplest case. The zero-inflated models need multiple parameters (`lam`, `z`,
`w`, `self_loops`) that are best bundled in the fit object. Force all cases to conform to the fit object.

However, the degree-events samplers accept `fit` but also require a separate
`positive_weight_rate` parameter that is available as `fit.q`. This is
**inconsistent** — the fit object should encode everything needed for sampling.

**Action required:** Degree-events samplers should derive `positive_weight_rate`
from the fit's `q` field internally, or the parameter should be removed and `q`
read from the fit. Unify all into the fit structure.

## Discrepancy 5: Keyword argument ordering

The standard ordering should be:
```
*, [layers,] self_loops, tolerance, verbose, max_iterations
```

Current violations:
- `fit_degree_events_*`: order is `self_loops, tolerance, max_iterations, verbose`
  (verbose after max_iterations).
- All others: `self_loops, tolerance, verbose, max_iterations`.

**Action required:** Reorder `fit_degree_events_*` to match:
`self_loops, tolerance, verbose, max_iterations`.

## Discrepancy 6: Default tolerance and max_iterations differ by family

| Constraint | Poisson tolerance | W tolerance | Poisson max_iter | W max_iter |
|---|---:|---:|---:|---:|
| strength | 1e-8 | 1e-8 | 10000 | 1000 |
| strength-cost | 1e-6 | 1e-8 | 5000 | 1000 |
| strength-edges | 1e-10 | 1e-8 | 50000 | 1000 |
| strength-degree | 1e-10 | 1e-8 | 50000 | 1000 |

**Assessment:** This is **partially justified**. ME IPF converges faster and
can use tighter tolerances. W conic/coordinate solvers have different iteration
semantics (Clarabel iterations ≠ IPF iterations). However, the user-facing
*meaning* of tolerance should be consistent: "how close are constraints
recovered."

**Action required:** Document that tolerance semantics differ by solver family.
Consider whether ME defaults should be relaxed to 1e-8 for consistency, or
whether W defaults should be tightened. Do not change without benchmarking.

## Missing fitting APIs

| Missing | Rust backend exists? | Sampling/filtering exists? | Action |
|---|:---:|:---:|---|
| `fit_strength_cost_binomial` | ✅ (ME IPF with layers) | ✅ / ✅ | Add Python wrapper |
| `fit_strength_edges_binomial` | ✅ (ME IPF with layers) | ✅ / ✅ | Add Python wrapper |
| `fit_strength_degree_binomial` | ✅ (ME IPF with layers) | ✅ / ✅ | Add Python wrapper |
| `fit_degree_events_binomial` | ✅ (Bernoulli IPF + rate) | ✅ / ✅ | Add Python wrapper |
| `fit_degree_events_poisson` | ✅ (Bernoulli IPF + rate) | ✅ / ✅ | Add unified wrapper |

**Assessment:** All 5 missing B constraint fits have working Rust backends and
complete sampling/filtering. The gap is purely in the Python fitting wrapper
layer. These should be added for full coherence — they are thin wrappers over
existing Rust code with `family="binomial"` and `layers=M`.

The ME `fit_degree_events_poisson` is the same: occupation is Bernoulli IPF,
positive-weight parameter is derived from `T/E`. A unified wrapper would match
the W degree-events API pattern.

## Private implementation discrepancies

### A. Rust result types: ME vs W

ME fits return minimal structs (`FitResult`, `StrengthCostFitResult`, etc.)
with only `x, y, [scalar], converged, iterations`. W fits return richer structs
with `status, objective, residuals, metrics`.

**Justified:** ME IPF is deterministic and cheap; richer diagnostics would be
noise. W solvers can fail in interesting ways that need reporting. However, DO unify them as close as possible by modifying B and ME returned info.

### B. PyO3 return tuples

ME: `(Vec<f64>, Vec<f64>, bool, usize)`.
W: larger tuples with nested diagnostic sub-tuples.

**Justified:** The Python wrapper unifies them into shared dataclasses. The
tuple boundary is internal. Returns should be equivalent in all cases, since they serve the same purpose with different models. DO unify them.

### C. Sampling: `self_loops` comes from fit object vs parameter

For zero-inflated models, `self_loops` is stored in the fit object and used
automatically. For degree-events samplers, `self_loops` is a separate parameter
even though it could come from the fit.

**Not justified:** If `DegreeEventsFit` carried `self_loops`, the sampler could
read it from the fit like other constraints do. This requires fixing
discrepancy 1 first. FIX it.

### D. Filtering: W absent-edge gaps

Absent-edge filtering for W geometric/NB is implemented for:
- ✅ strength (independent)
- ✅ strength-cost

But **not** for:
- ❌ strength-edges geometric/NB
- ❌ strength-degree geometric/NB
- ❌ degree-events geometric/NB

Rust native `absent_*` functions do not exist for these. Python wrappers reject
with clear error.

**Assessment:** Implementing requires adding W zero-inflated absent-pair
iteration in Rust for these providers. Not architecturally difficult but needs
test coverage. Lower priority than fitting consistency. DO IT.

## Summary of required actions (ordered by priority)

1. Fix `DegreeEventsFit`: add `family`, `layers`, `self_loops`, `diagnostics`.
2. Fix `FitResult`: add `self_loops`.
3. Fix `fit_strength_binomial` → `family="binomial"`, `layers=M`.
4. Fix `fit_degree_bernoulli` → `family="bernoulli"`.
5. Populate `diagnostics` in all fits (ME strength, ME degree, B strength,
   B degree, ME strength-degree). Unify ME/B Rust return types to carry the
   same diagnostic fields as W where practical.
6. Unify PyO3 return tuples: ME/B should return equivalent structure to W so
   the Python wrapper does not need separate unpacking paths per family.
7. Reorder `fit_degree_events_*` kwargs to match standard ordering.
8. Fix degree-events samplers: remove `positive_weight_rate` parameter, derive
   from fit `q` internally. Requires `DegreeEventsFit` to carry `self_loops`.
9. Add missing B constraint fitting wrappers: cost, edges, degree, degree-events.
10. Add missing ME `fit_degree_events_poisson` unified wrapper.
11. Add missing W absent-edge filtering (edges, degree, degree-events) in Rust
    and expose through PyO3/Python.
12. Document tolerance/iteration defaults as family-dependent with rationale.

