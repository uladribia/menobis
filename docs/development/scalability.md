# Scalability

ODME is designed to handle large networks efficiently by exploiting the
**node-factorized** nature of maximum-entropy models.

## Key insight

For the fixed-strength multi-edge model:

```
E[t_ij] = x_i * y_j
```

The expected weight between any pair (i, j) depends only on per-node Lagrange
multipliers `x` (length N) and `y` (length N). The full NxN matrix is never
needed for fitting, sampling, or most statistics.

## Memory complexity by operation

| Operation | Memory | Notes |
|-----------|--------|-------|
| Fitting (Lagrange multipliers) | O(N) | Only x and y vectors |
| Factorized Poisson sampling | O(E) | E = non-zero edges in sample |
| Factorized multinomial sampling | O(E) | Row-by-row allocation |
| Directed strengths/degrees | O(E) | Rust single-pass kernel |
| Y2, k_nn, s_nn | O(E) | Rust single-pass kernel |
| Clustering coefficient | O(E) | Via rustworkx graph |
| Dense expected matrix | O(N²) | **Avoid for large N** |

## Practical thresholds

| N (nodes) | Dense NxN | Recommendation |
|-----------|-----------|----------------|
| < 3,000 | ~72 MB | Dense API is fine |
| 3,000–10,000 | 72 MB–800 MB | Use factorized samplers |
| > 10,000 | > 800 MB | **Must** use factorized operations |

## API guidance

For small networks (N < 3,000):

```python
from odme.models import expected_multi_edge_weights, sample_poisson

expected = expected_multi_edge_weights(strengths)
sample = sample_poisson(expected, seed=42)
```

For large networks (N > 3,000):

```python
from odme.models import fit_fixed_strength_me, sample_poisson_factorized

multipliers = fit_fixed_strength_me(strengths)
x = multipliers.get_column("x").to_numpy()
y = multipliers.get_column("y").to_numpy()
sample = sample_poisson_factorized(x, y, seed=42)
```

The factorized samplers iterate over source nodes (O(N) outer loop), sampling
a target vector of length N per source but only storing non-zero results.
Peak memory per row is O(N); total stored output is O(E).

## Which models are node-factorized?

| Model | Factorized? | Notes |
|-------|:-----------:|-------|
| Fixed strength (ME) | ✓ | E[t_ij] = x_i * y_j |
| Fixed strength (W/AB/AW) | ✓ | Same structure, different link function |
| Fixed degree | ✓ | E[p_ij] = z_i * w_j / (1 + z_i * w_j) |
| Fixed strength + degree | ✓ | Four multipliers per node |
| Strength-cost / distance | ✓ | E[t_ij] = x_i * y_j * f(d_ij); metric avoids NxN storage |
| Custom p_ij | ✗ | Requires full NxN probability matrix |

All models except custom p_ij are node-factorized and scale to large N
without materializing dense matrices. The strength-cost model with a metric
function (e.g., Euclidean distance) is also scalable: `f(d_ij)` is evaluated
on-the-fly per row at O(N) peak memory, never storing the full NxN cost matrix.

## The strength-cost / distance-constrained model

The most general node-factorized model in the thesis is the strength-cost model:

```
p_ij = x_i * y_j * f(d_ij)
```

where `f(d_ij)` is a deterrence function of the cost/distance between nodes i and j.
This does NOT require a full NxN distance matrix if the cost function `f` can be
evaluated on-the-fly from node coordinates or a metric function.

For example, with exponential deterrence `f(d) = exp(-gamma * d)` and Euclidean
distance, the sampling loop for source i only needs:

1. The coordinates of node i.
2. The coordinates of all target nodes j.
3. Evaluate `d_ij = distance(i, j)` and `rate_ij = x_i * y_j * exp(-gamma * d_ij)`.

This is O(N) per source, O(N^2) total computation but O(N) peak memory per row.
The full NxN distance matrix is never stored.

When the distance matrix IS pre-computed (e.g., from a file or a non-Euclidean
metric), it is inherently O(N^2) in storage. In that case, ODME should support
sparse distance formats (only storing node pairs within a cutoff) or streaming
row-by-row access.

Summary of distance/cost handling strategies:

| Scenario | Memory | Approach |
|----------|--------|----------|
| Metric function (Euclidean, etc.) | O(N) per row | Compute d_ij on-the-fly |
| Sparse cost matrix (cutoff) | O(E_cost) | Only store pairs within cutoff |
| Full pre-computed distance matrix | O(N^2) | Avoid for N > 10,000 |

## The custom p_ij exception

The custom p_ij model requires an explicit probability for every node pair.
This is inherently O(N²) in memory and cannot be factorized. For this model,
ODME limits N to MAX_DENSE_NODES (3,000) by default, with an override flag
for users who have sufficient memory.
"""
