---
description: Python filtering result types and options.
---

# Filtering API

## TL;DR

`filter_model` compares an observed sparse `EdgeTable` with a fitted independent
null model and returns sparse edge subsets with p-values and expectations.

## Entry point

```python
from menobis.filtering import filter_model

result = filter_model(
    edges,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    fit=fit,
    alpha=0.05,
    tail="two-sided",
)
```

Strength-cost filtering also needs coordinates:

```python
result = filter_model(
    edges,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH_COST,
    fit=fit,
    coord_x=x,
    coord_y=y,
)
```

## Options

| Option | Default | Meaning |
|---|---:|---|
| `alpha` | `0.05` | significance level |
| `tail` | `two-sided` | `upper`, `lower`, or `two-sided` |
| `correction` | `none` | `none`, `bonferroni`, or `fdr` |
| `detect_absent` | `False` | scan zero-weight candidate pairs |
| `min_occupation` | `0.5` | absent-pair occupation threshold |
| `min_expected` | `0.0` | absent-pair expected-weight threshold |
| `max_absent` | `None` | cap absent output |

## Result shape

```python
result.upper
result.lower
result.compatible
result.absent_lower
```

Each field is a `FilteredEdges` object with:

| Field | Meaning |
|---|---|
| `edges` | sparse edge table |
| `upper_pvalue` | upper-tail p-values |
| `lower_pvalue` | lower-tail p-values |
| `expected` | null expected occupation |
| `occupation` | null probability of positive occupation |

## Supported constraints

Filtering is for independent grand-canonical nulls: strength, strength-cost,
strength-edges, strength-degree, degree-events, and sparse custom/partial Poisson
rates through lower-level wrappers.
