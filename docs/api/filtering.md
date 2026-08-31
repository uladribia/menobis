---
description: Python filtering API — entry point, options, and result structure.
---

# Filtering API

## TL;DR

`filter_model` compares an observed sparse `EdgeTable` with a fitted
independent null model and returns sparse edge subsets with p-values and
expectations. Theory: [Filtering statistics](../science/filtering.md);
workflow: [Filter node pairs](../guide/filter-network.md).

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
    correction="fdr",
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
| `detect_absent` | `False` | scan zero-occupation candidate pairs |
| `min_occupation` | `0.5` | absent-pair occupation threshold |
| `min_expected` | `0.0` | absent-pair expected-occupation threshold |
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
| `upper_pvalue` | upper-tail p-values (positive-support conditioned) |
| `lower_pvalue` | lower-tail p-values (positive-support conditioned) |
| `expected` | null expected occupation |
| `occupation` | null probability of positive occupation |

## Supported constraints

Filtering applies to independent grand-canonical nulls: strength,
strength-cost, strength-edges, strength-degree, degree-events, and
edges-events (see [Supported models](../guide/supported-models.md)).