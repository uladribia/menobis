---
description: Public API consistency audit across ME, W, and B fitting families.
---

# API Consistency Audit

## TL;DR

Public fitting APIs are consistent in naming and parameter ordering across
ME/W/B families. Result types are constraint-oriented with shared diagnostics.
Six discrepancies remain in the result-type population and need review.

## Public API style (consistent)

All fitting functions follow:

```text
fit_{constraint}_{distribution}(
    constraint_params...,  # positional-or-keyword, required
    *,
    [layers: int,]        # keyword-only, default, only for NB/binomial
    self_loops: bool,     # keyword-only, default True
    tolerance: float,     # keyword-only, default
    verbose: int,         # keyword-only, default 0
    max_iterations: int,  # keyword-only, default
) -> ConstraintResult
```

Result types are constraint-oriented, not family-oriented:

| Constraint | Result type | Constraint-specific fields |
|---|---|---|
| strength | `FitResult` | — |
| strength-cost | `StrengthCostFit` | `gamma`, `self_loops` |
| strength-edges | `StrengthEdgesFit` | `lam`, `self_loops` |
| strength-degree | `StrengthDegreeFit` | `z`, `w`, `self_loops` |
| degree-events | `DegreeEventsFit` | `q`, `positive_mean` |
| degree | `FitResult` | — |

All constraint types (except `DegreeEventsFit`) share:
`family`, `layers`, `diagnostics: OptimizationDiagnostics | None`.

## Discrepancies found

### 1. `DegreeEventsFit` lacks `family`, `layers`, `self_loops`

| Field | Other constraint types | `DegreeEventsFit` |
|---|---|---|
| `family` | present | **missing** |
| `layers` | present | **missing** |
| `self_loops` | present on cost/edges/degree | **missing** |
| `diagnostics` | present | **missing** (has bare `converged`/`iterations`) |

**Impact:** Users cannot distinguish geometric from negative-binomial
degree-events fits programmatically; diagnostics are flat instead of nested.

### 2. `FitResult` lacks `self_loops`

`fit_strength_poisson`, `fit_strength_geometric`, `fit_strength_binomial`, and
`fit_degree_bernoulli` all return `FitResult` which does not carry `self_loops`.
Other constraint types (`StrengthCostFit`, `StrengthEdgesFit`,
`StrengthDegreeFit`) do carry it.

**Impact:** If a fit is passed to a sampler that needs `self_loops`, the user
must track it separately. Inconsistent with the other types.

### 3. `fit_strength_binomial` returns `family="poisson"` and `layers=None`

It should return `family="binomial"` and `layers=M`.

**Impact:** A binomial fit is indistinguishable from a Poisson fit by inspection.

### 4. `fit_degree_bernoulli` returns `family="poisson"` and `layers=None`

It should return `family="bernoulli"` and `layers=None` (or `layers=1`).

### 5. `fit_strength_degree_poisson` does not populate `diagnostics`

Other Poisson constraint fits (`strength_cost`, `strength_edges`) populate
`OptimizationDiagnostics` with `converged`/`status`/`iterations`. The ME
strength-degree fit returns bare `converged`/`iterations` only.

### 6. W absent-edge filtering not implemented for geometric/NB

Rust has absent-edge functions for:
- Poisson: all 5 constraints ✅
- Binomial: strength, cost, edges, degree, degree-events ✅
- Geometric: strength ✅, cost ✅ — but **not** edges, degree, degree-events
- Negative binomial: strength ✅, cost ✅ — but **not** edges, degree, degree-events

Python wrappers explicitly reject absent-edge detection for these W
zero-inflated models with a clear error.

## Private implementation discrepancies

### A. Rust result types are not unified

| Constraint | Rust type | Fields |
|---|---|---|
| strength ME | `FitResult` | x, y, converged, iterations |
| strength W | `WStrengthFitResult` | x, y, layers, status, objective, residuals, metrics |
| strength-cost ME | `StrengthCostFitResult` | x, y, gamma, converged, iterations |
| strength-cost W | `WStrengthCostFitResult` | x, y, gamma, layers, status, objective, residuals, metrics |
| strength-edges ME | `StrengthEdgesFitResult` | x, y, lam, converged, iterations |
| strength-edges W | `WStrengthEdgesFitResult` | x, y, lam, layers, status, objective, residuals, metrics |
| strength-degree ME | `StrengthDegreeFitResult` | x, y, z, w, converged, iterations |
| strength-degree W | `WStrengthDegreeFitResult` | x, y, z, w, layers, status, objective, residuals, metrics |
| degree-events W | `DegreeEventsFitResult` | x, y, q, positive_mean, converged, iterations |

**Assessment:** The ME types are minimal; W types carry solver diagnostics. This
split is **justified** because ME IPF is simple (converged + iterations is
enough) while W conic/coordinate solvers produce richer diagnostics (objective,
residuals, metrics). Forcing them into one Rust enum would add unnecessary
complexity to simple ME paths.

### B. PyO3 tuple return types differ per family

ME fits return `(Vec<f64>, Vec<f64>, bool, usize)`.
W fits return larger tuples with diagnostics and metrics sub-tuples.

**Assessment:** **Justified.** PyO3 does not support trait objects or enums as
return types cleanly. The Python wrapper layer already unifies them into shared
dataclasses. The tuple boundary is an implementation detail, not a public API.

### C. Validation is partially shared

Strength-edges and cost entries have shared validators. But strength-degree and
degree-events still validate inline.

**Assessment:** Minor inconsistency. Could be harmonized but low priority since
the validation logic is small and correct in each case.

## Recommendations (pending review)

1. Add `family`, `layers`, `self_loops`, and `diagnostics` to `DegreeEventsFit`.
2. Add `self_loops` to `FitResult`.
3. Fix `fit_strength_binomial` to return `family="binomial"`.
4. Fix `fit_degree_bernoulli` to return `family="bernoulli"`.
5. Populate `diagnostics` in `fit_strength_degree_poisson`.
6. W absent-edge for edges/degree/degree-events: implement in Rust or document
   as intentionally unsupported.
