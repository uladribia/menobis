---
description: Benchmark CLI reference — all commands and flags.
---

# Benchmark CLI

## TL;DR

The `python -m benchmarks` CLI exercises the MENoBiS full pipeline
(PA-geographic generation, fitting, sampling, filtering) across node sizes,
families, regimes, and constraint types. All commands write JSON to a
results file; use `--json` to print to stdout.

## Commands

### `all` — Full E2E pipeline (fit → sample → filter)

```bash
uv run python -m benchmarks all --nodes 100,500,1000 --regime dense
```

Reports: wall/CPU time, memory (tracemalloc + RSS), convergence, precision
(constraint residuals), false-positive filtering rate.

Flags: `--nodes`, `--families`, `--constraints`, `--regime`, `--known-pairs`,
`--self-loops`, `--no-memory`, `--seed`, `--output`.

### `fit` — Fitting only (no sampling/filtering)

Same flags as `all`, stops after fit convergence.

### `compare` — Compare fit precision across families

Same flags, reports per-family precision tables.

### `micro` — Microcanonical sampling benchmarks

```bash
uv run python -m benchmarks micro --nodes 500,1000 --families me,b,w --regime sparse --constraint edges-events
```

Reports: wall/CPU time, memory, constraint recovery precision. Does not fit —
derives constraints from the PA-geographic generator and samples directly.

Flags: `--constraint` (edges-events, degree-events, strength, strength-degree, strength-cost),
`--burn-in-sweeps`, `--sweeps-per-sample`, gamma-fit tuning flags for
strength-cost.

### `matrix` — Microcanonical fixed-strength benchmark matrix

```bash
uv run python -m benchmarks matrix   # 72-cell default
```

Reports per-stage metrics: construction time, repair time, repair steps,
repair restarts, occupied pairs, MCMC proposals/sec, accepted/sec, gamma
fit time, cost ESS, final sampling time, peak memory.

Flags: `--nodes`, `--families`, `--regime`, `--constraints` (strength,
strength-cost), `--no-memory`.

## Output format

All commands write an array of `BenchmarkRow` objects to the output JSON file.
Each row carries dimension fields (N, family, constraint, regime...) plus
stage-specific metrics. See the [Benchmark interpretation]
(../development/benchmarking.md) page for field semantics.