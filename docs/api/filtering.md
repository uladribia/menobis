---
description: Python API for MENoBiS statistical filtering.
---

# Filtering API

## TL;DR

Filtering compares observed edges with a fitted independent null model and
returns upper, lower, compatible, and absent-edge tables. It should use the same
pair-provider math as generation.

## Function groups

| Group | Functions |
|---|---|
| ME strength/cost/binary | `filter_strength_poisson`, `filter_strength_cost_poisson`, `filter_strength_edges_poisson`, `filter_strength_degree_poisson`, `filter_degree_events_poisson` |
| B | `filter_strength_binomial`, `filter_strength_cost_binomial`, `filter_strength_edges_binomial`, `filter_strength_degree_binomial`, `filter_degree_events_binomial` |
| W | `filter_strength_geometric`, `filter_strength_negative_binomial`, `filter_strength_cost_geometric`, `filter_strength_cost_negative_binomial`, zero-inflated W filters |
| Custom/partial | `filter_custom_poisson` over sparse expected-rate tables |

Independent-strength filters accept raw `x`, `y` multipliers. Multi-parameter
models, including degree-events filters, accept fitted result objects so `family`,
`layers`, `self_loops`, and positive-weight parameters stay consistent.

## Common options

| Parameter | Type | Default |
|---|---|---|
| `alpha` | `float` | `0.05` |
| `tail` | `"upper"`, `"lower"`, `"two-sided"` | `"two-sided"` |
| `correction` | `"none"`, `"bonferroni"`, `"fdr"` | `"none"` |
| `detect_absent` | `bool` | `False` |
| `min_occupation` | `float` | `0.5` |
| `min_expected` | `float` | `0.0` |
| `max_absent` | `int | None` | `None` |

## Result types

```python
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

`FilteredEdges` stores the sparse `EdgeTable`, p-values, expected weights, and
occupation probabilities.

## Example

```python
from menobis.filtering import filter_strength_poisson
from menobis.data.io import read_edges

edges = read_edges("network.csv")
result = filter_strength_poisson(edges, alpha=0.05, detect_absent=True)

print(len(result.upper.edges), len(result.lower.edges))
```

## Conformance note

Degree-events generation and filtering read ME/B/W positive-weight parameters
from `DegreeEventsFit`. Low-level PyO3 bindings still expose primitive arrays for
internal wrappers; see the [Ontology conformance audit](../development/ontology-conformance-audit.md).
