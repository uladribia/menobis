---
description: Canonical benchmark pipeline and commands.
---

# Benchmarking

## TL;DR

A single Python CLI (`benchmarks/cli.py`) exercises the full MENoBiS pipeline
using PA-geographic networks. It measures time, memory, convergence, constraint
precision, and parallelism across all model families and constraint types.

The **dense regime** (`average_degree = N/5`, `events_per_edge = 8.0`) is the
recommended default — it avoids pathological solver behaviour seen in sparse
(ill-posed degree constraints) and saturated (k ≈ N) regimes.

## Benchmark matrix

| Dimension | Values |
|---|---|
| Node counts | 100, 1000 |
| Families | ME, B (W excluded due to ZI convergence failures) |
| Constraints | strength, strength-cost, strength-edges, strength-degree |
| Self-loops | no-self-loops (default), self-loops |
| Known-pair fraction | 0%, 2% |
| **Regime** | **dense** (default), sparse, saturated |

## Regimes

| Regime | Parameters | Character | Recommendation |
|---|---|---|---|
| Sparse | `average_degree=3.0, events_per_edge=3.0` | Low connectivity, s/k ~ 3 | ❌ Pathological for degree constraints (k ≈ s) |
| **Dense** | `average_degree=N/5, events_per_edge=8.0` | **Moderate connectivity, no node saturates** | **✅ Optimal — exercises solvers realistically** |
| Saturated | `average_degree=0.85*(N-1), events_per_edge=8.0` | k near N-1, dense, high weight | ❌ Pathological — all nodes near full connectivity |

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
# Default: dense regime, ME + B, N=100 + N=1000
uv run python -m benchmarks all --nodes 100,1000 --regime dense

# Full E2E with self-loops and known-pair partial fits
uv run python -m benchmarks all --nodes 1000 --families me,b \
  --regime dense --known-pairs 0.0,0.02 --self-loops

# Fit-only (skips sampling and filtering)
uv run python -m benchmarks fit --nodes 500 --regime dense

# Compare all three regimes side-by-side
uv run python -m benchmarks compare --nodes 500 --families me,b \
  --constraints strength-degree

# Quick CI smoke (minimum configuration)
uv run python -m benchmarks all --nodes 20 --families me \
  --constraints strength --regime dense --known-pairs 0.0 \
  --filter-samples 1 --no-memory
```

## Options

| Option | Meaning |
|---|---|
| `--nodes N,N,...` | Node counts |
| `--families me,b,w` | Model families (w excluded by default for ZI issues) |
| `--constraints ...` | Constraint types |
| `--regime sparse,dense,saturated` | Regime selection (default: `dense`) |
| `--known-pairs 0.0,0.02,0.05,0.20` | Known-pair fractions |
| `--self-loops/--no-self-loops` | Allow/forbid self-loops |
| `--filter-samples N` | Null samples for FPR estimation |
| `--no-memory` | Skip memory profiling |
| `--tol-dense` | Solver tolerance for dense regime |
| `--tol-sparse` | Solver tolerance for sparse regime |
| `--tol-saturated` | Solver tolerance for saturated regime |
| `--json` | Machine-readable stdout |
| `--output PATH` | JSON results file |

## Output format

Results are JSON arrays. Each row contains stage, metrics, and precision fields.
See `benchmarks/results/` for stored runs.

## Partial fitting

Partial benchmarks freeze a fraction of observed edges as known pairs and fit
the remainder. Available for ME (all constraints) and all families
(strength-cost coordinates). Reports convergence and timing for the free
subproblem. Known-pair fractions up to 2% are recommended for dense regime;
higher fractions increase memory footprint due to `O(N²)` lookup structures.

## Parallelism detection

The CPU/wall ratio reveals whether Rust rayon parallelism is active:

- ~1.0: single-threaded
- >1.5: multi-threaded computation
- Typically saturates around core count at N >= 500

## Testing recommendation

All tests and CI should use the **dense regime** by default. Sparse and
saturated regimes are reserved for regime-comparison benchmarks and edge-case
validation only.

```bash
# Run tests
uv run pytest

# Run benchmarks with recommended settings
uv run python -m benchmarks all --nodes 100,1000 --families me,b --regime dense
```
