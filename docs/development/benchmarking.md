---
description: Canonical benchmark pipeline and commands.
---

# Benchmarking

## TL;DR

The benchmark CLI exercises the MENoBiS pipeline with PA-geographic non-binary
networks: generate constraints, fit, sample, filter, and measure time, memory,
convergence, and precision.

!!! note "Canonical regime"
    Use the **dense** regime by default: `average_degree = N/5` and
    `events_per_edge = 8.0`. Sparse and saturated regimes are stress tests.

## Benchmark matrix

| Dimension | Values |
|---|---|
| Node counts | user-selected, commonly 100 and 1000 first |
| Families | ME, B, W; W zero-inflated routes need extra care |
| Constraints | strength, strength-cost, strength-edges, strength-degree |
| Self-loops | no-self-loops default; self-loops optional |
| Known-pair fraction | 0%, 2%, or explicit stress values |
| Regime | dense default, sparse, saturated |
| Microcanonical metrics | 11 per-stage metrics: construction time, repair time, repair steps, repair restarts, occupied pairs, MCMC proposals/sec, accepted/sec, gamma fit time, cost ESS, final sampling time, peak memory |

## Regimes

| Regime | Parameters | Use |
|---|---|---|
| sparse | `average_degree=3.0`, `events_per_edge=3.0` | stress nearly-binary occupations |
| dense | `average_degree=N/5`, `events_per_edge=8.0` | default scientific benchmark |
| saturated | `average_degree=0.85*(N-1)`, `events_per_edge=8.0` | stress boundary degrees |

## Metrics collected

| Metric | How measured |
|---|---|
| wall time | `time.perf_counter()` |
| CPU time | `time.process_time()` |
| parallelism factor | CPU / wall ratio |
| Python memory | `tracemalloc` peak delta |
| RSS memory | `/proc/self/status` VmRSS delta |
| convergence | `result.converged`, `status`, iterations |
| precision | constraint residuals |

## Commands

```bash
uv run python -m benchmarks matrix   # microcanonical fixed-strength matrix

uv run python -m benchmarks all --nodes 100,1000 --regime dense

uv run python -m benchmarks all --nodes 1000 --families me,b \
  --regime dense --known-pairs 0.0,0.02 --self-loops

uv run python -m benchmarks compare --nodes 500 --families me,b \
  --constraints strength-degree
```

## Interpreting results

| Observation | Meaning |
|---|---|
| `parallel_factor > 1.5` | Rust parallelism is active |
| `converged=False` | inspect residuals before scientific use |
| high residual, low iterations | likely feasibility or scaling issue |
| high iterations, low residual | tolerance may be stricter than needed |

## Conformance checks covered

| Area | Benchmark expectation |
|---|---|
| family separation | ME, B, and W use different expectation equations |
| coordinate costs | costs are computed on demand from projected XY |
| partial fitting | frozen pairs are subtracted before fitting free support |
| sparse memory | masks and outputs stay sparse; no public dense cost/rate matrix |
| filtering calibration | false-positive rates are measured on null samples |

## Microcanonical benchmarks

The `matrix` command runs the microcanonical fixed-strength matrix
(N=100/500/1000 × ME/B/W × sparse/dense × strength/strength-cost; 72 cells,
plus a quick smoke subset with `--nodes 100 --families me,b`). Stored results
are in JSON format in the local `benchmarks/results/` directory
(gitignored; regenerate with `uv run python -m benchmarks matrix`).
Each cell reports the 11 per-stage metrics listed above, including repair
steps/restarts, MCMC throughput, gamma fit time, cost ESS, and peak memory.

The fixed-(s,E) route has the same repair/MCMC metric shape; its per-stage
timings and counters (local acceptance, bridge departures/returns/timeouts)
are exposed by the Rust bench entry point
(`sample_fixed_strength_edges_bench`) and exercised by the `#[ignore]`
N=1000 scalability oracle (ME/B/W + fixed pairs) and the Python `--run-heavy`
gates, so no separate CLI matrix cell is required.

## Partial fitting

Partial benchmarks freeze a fraction of observed pairs as known and fit the free
subproblem. Reports include convergence and timing for the excess constraints.
Start with 2% known pairs; increase only after validating memory and runtime.

## Output format

Results are JSON arrays. Each row contains stage, metrics, and precision fields.
See `benchmarks/results/` for stored runs and [Scalability](scalability.md) for a
plot built from repository results.
