---
description: Python API for ODME statistical filtering.
---

# Filtering API

## TL;DR

All filtering functions accept observed edges and a fitted null model, then
return a `FilterResult` with upper, lower, compatible, and absent-lower tables.

## Functions

| Function | Null model | Key parameters |
|----------|------------|----------------|
| `filter_fixed_strength_me` | Poisson, auto-fitted | edges |
| `filter_strength_cost_me` | Poisson with costs | edges, fit, cost arrays |
| `filter_strength_edges_me` | ZIP, pre-fitted | edges, fit |
| `filter_strength_degree_me` | ZIP, pre-fitted | edges, fit |
| `filter_degree_events_me` | ZIP, manual | edges, x, y, rate |
| `filter_custom_rates_poisson` | Poisson, user rates | edges, rates table |

## Common options

All functions accept:

| Parameter | Type | Default |
|-----------|------|---------|
| `alpha` | `float` | `0.05` |
| `tail` | `"upper"`, `"lower"`, `"two-sided"` | `"two-sided"` |
| `correction` | `"none"`, `"bonferroni"`, `"fdr"` | `"none"` |
| `detect_absent` | `bool` | `False` |
| `min_occupation` | `float` | `0.5` |
| `min_expected` | `float` | `0.0` |
| `max_absent` | `int \| None` | `None` |

## Result types

```python
@dataclass(frozen=True)
class FilteredEdges:
    edges: EdgeTable
    upper_pvalue: NDArray[np.float64]
    lower_pvalue: NDArray[np.float64]
    expected: NDArray[np.float64]
    occupation: NDArray[np.float64]

@dataclass(frozen=True)
class FilterResult:
    upper: FilteredEdges
    lower: FilteredEdges
    compatible: FilteredEdges
    absent_lower: FilteredEdges
    alpha: float
    tail: Tail
    correction: Correction
```

## Example

```python
from odme.filtering import filter_fixed_strength_me
from odme.data.io import read_edges

edges = read_edges("network.csv")
result = filter_fixed_strength_me(edges, alpha=0.05, detect_absent=True)

print(f"Upper: {len(result.upper.edges)} edges")
print(f"Lower: {len(result.lower.edges)} edges")
print(f"Compatible: {len(result.compatible.edges)} edges")
print(f"Absent: {len(result.absent_lower.edges)} pairs")
```
