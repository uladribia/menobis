---
description: Canonical benchmark pipeline and commands.
---

# Benchmarking

## TL;DR

A single Python CLI (`benchmarks/cli.py`) exercises the full MENoBiS pipeline
using PA-geographic networks. It measures time, memory, convergence, constraint
precision, and parallelism across all model families and constraint types.

## Benchmark matrix

| Dimension | Values |
|---|---|
| Node counts | 100, 500, 1000, 2000 |
| Families | ME, B, W |
| Constraints | strength, strength-cost, strength-edges, strength-degree |
| Self-loops | no-self-loops (default) |
| Known-pair fraction | 0%, 5%, 20% |
| Regime | sparse, saturated |

## Regimes

| Regime | Parameters | Character |
|---|---|---|
| Sparse | `average_degree=3.0, events_per_edge=3.0` | Low connectivity, s/k ~ 3 |
| Saturated | `average_degree=0.85*(N-1), events_per_edge=8.0` | k near N-1, dense, high weight |

## Metrics collected

| Metric | How measured |
|---|---|
| Wall time | `time.perf_counter()` |
| CPU time | `time.process_time()` |
| Parallelism factor | CPU / wall ratio (>1 = multi-threaded Rust) |
| Memory (Python) | `tracemalloc` peak delta |
| Memory (RSS) | `/proc/self/status` VmRSS delta |
| Convergence | `result.converged` + `result.iterations` |
| Constraint precision | Max absolute residual per constraint type |

## Commands

```bash
# Full matrix
uv run python -m benchmarks all --nodes 100,500,1000 --output benchmarks/results/full.json

# Quick CI smoke
uv run python -m benchmarks all --nodes 20 --families me --constraints strength \
  --regime sparse --known-pairs 0.0 --filter-samples 1 --no-memory

# Fit-only with partial
uv run python -m benchmarks fit --nodes 500,1000 --regime saturated --known-pairs 0.0,0.05,0.20

# Compare regimes
uv run python -m benchmarks compare --nodes 500 --families me,w --constraints strength-degree
```

## Options

| Option | Meaning |
|---|---|
| `--nodes N,N,...` | Node counts |
| `--families me,b,w` | Model families |
| `--constraints ...` | Constraint types |
| `--regime sparse,saturated` | Regime selection |
| `--known-pairs 0.0,0.05,0.20` | Known-pair fractions |
| `--filter-samples N` | Null samples for FPR |
| `--no-memory` | Skip memory profiling |
| `--json` | Machine-readable stdout |
| `--output PATH` | JSON results file |

## Output format

Results are JSON arrays. Each row contains stage, metrics, and precision fields.
See `benchmarks/results/` for stored runs.

## Partial fitting

Partial benchmarks freeze a fraction of observed edges as known pairs and fit
the remainder. Available for ME (all constraints) and all families
(strength-cost coordinates). Reports convergence and timing for the free
subproblem.

## Parallelism detection

The CPU/wall ratio reveals whether Rust rayon parallelism is active:

- ~1.0: single-threaded
- >1.5: multi-threaded computation
- Typically saturates around core count at N >= 500

## Removed: Rust benchmark

The former `crates/menobis-core/benches/me_strength_degree.rs` has been removed.
The Python benchmark captures Rust solver performance through PyO3 (overhead <1ms)
and additionally measures RSS (captures Rust heap allocations) and CPU/wall ratio
(captures rayon parallelism).
