# Cost constraints

## TL;DR

MENoBiS currently supports cost-constrained models only through projected node
coordinates and Euclidean distance. It does not accept dense user-provided cost
matrices.

## Current scope

| Item | Status |
|------|--------|
| Euclidean distance from XY coordinates | Supported |
| Dense `N x N` cost matrices | Not supported |
| Sparse user-provided cost tables | Not public API |
| Custom distance functions | Extension point |

## Why coordinates are required

A complete pair-cost table has one value per candidate ordered pair. For large
networks this becomes a dense `N x N` object and conflicts with MENoBiS' memory
policy. Coordinates let Rust generate `d_ij` on demand while fitting,
generating, and filtering stream over candidate pairs.

For projected coordinates `(x_i, y_i)`, MENoBiS uses:

```text
d_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
```

This distance enters strength-cost models through:

```text
q_ij = x_i y_j exp(-gamma d_ij)
```

## Custom costs

If a workflow needs road distance, travel time, or another non-Euclidean cost,
implement a Rust cost provider and model wrapper instead of materializing a
dense matrix. The provider should compute one pair cost at a time and integrate
with `PairCostProvider` in `menobis-core`.
