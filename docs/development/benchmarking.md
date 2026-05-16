---
description: Performance benchmarks and scaling characteristics.
---

# Benchmarking

## TL;DR

All ODME analysis and generation operations scale smoothly to $N = 10{,}000$
nodes. Fixed-strength fitting is $O(N)$ analytical. Iterative fitters scale
as $O(N^2)$ per iteration; degree-based fitters are practical to $N \approx 1000$.

## Scaling results

Benchmarks run on a 14-core x86\_64 machine with 15 GB RAM (release build),
using Pareto-distributed strength sequences with average strength 100 per node.

### Time vs N (log-log)

![Benchmark scaling](../figures/benchmark_scaling.png)

### Timing table

![Benchmark table](../figures/benchmark_table.png)

## Analysis (Rust kernels)

All analysis operations are $O(E)$ single-pass Rust kernels:

| Operation | N=1000 | N=10000 |
|-----------|--------|---------|
| `directed_strengths` | 4 ms | 71 ms |
| `directed_degrees` | 4 ms | 70 ms |
| `compute_all_stats` | 4 ms | 72 ms |

## Fitting

| Model | Method | Complexity | N=100 | N=1000 |
|-------|--------|------------|-------|--------|
| Fixed strength | Analytical | $O(N)$ | 0.1 ms | 0.2 ms |
| Fixed degree | IPF balancing | $O(N^2 \cdot I)$ | 1 s | 0.6 s |
| Strength + edges | IPF + bisection on $\lambda$ | $O(N^2 \cdot I \cdot 80)$ | 21 ms | 2.7 s |
| Strength + cost | IPF + adaptive search on $\gamma$ | $O(N^2 \cdot I \cdot K)$ | 13 ms | 0.95 s |
| Strength + degree | 4-variable IPF | $O(N^2 \cdot I)$ | 10.7 s | — |

**Preconditioning**: all fitters use informed initial guesses derived from the
homogeneous solution (e.g., $c = \sqrt{k_{avg} / (N - k_{avg})}$ for degree).

**Convergence**: iterative fitters check both multiplier convergence and
constraint satisfaction. The degree fitter may oscillate at small $N$ with
heterogeneous sequences; constraints are satisfied but multipliers keep
adjusting. This matches the original thesis code behavior.

**Warm start**: the strength-cost fitter reuses $x, y$ from the previous
$\gamma$ iteration as initial guess for the next.

## Generation (Rust kernels)

| Sampler | N=1000 | N=10000 | Memory |
|---------|--------|---------|--------|
| Poisson | 17 ms | 1.0 s | $O(E)$ sparse |
| Multinomial | 34 ms | 2.8 s | $O(E)$ sparse |
| Microcanonical | 13 ms | 0.2 s | $O(T)$ stubs |

All samplers produce sparse edge lists, never dense $N \times N$ matrices.
The microcanonical sampler is fastest because it shuffles $T$ stubs instead of
generating per-pair random numbers.

## Memory

ODME uses sparse edge-list representations throughout. Memory scales as
$O(E + N)$, not $O(N^2)$. Dense $N \times N$ matrices are only allocated
inside fitters that require pairwise summations.

## Running benchmarks

```bash
uv run maturin develop --release
uv run python benchmarks/bench_scaling.py
uv run python benchmarks/bench_quick.py
uv run pytest tests/test_odme_benchmark.py -v
```

## Performance regression tests

`tests/test_odme_benchmark.py` asserts at $N = 10{,}000$:

- Analysis < 1 second
- Fixed-strength fitting < 0.1 seconds
- Poisson generation < 5 seconds
- Microcanonical generation < 5 seconds
