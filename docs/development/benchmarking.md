---
description: Canonical benchmark pipeline and commands.
---

# Benchmarking

## TL;DR

Use the canonical PA geographic generator for every benchmark. It intentionally
creates networks outside the ODME null family, then fits ODME nulls to the
derived constraints.

## Canonical network

`generate_pa_geographic_network` creates directed weighted networks by:

1. sampling binary support with preferential attachment;
2. assigning projected XY coordinates;
3. scoring occupied edges with origin/destination support degree and
   `exp(-gamma * distance / distance_scale)`;
4. normalizing positive integer weights to an exact total event count.

This gives controlled density, exact total events, heterogeneous degrees,
heterogeneous strengths, and spatially damped weights.

## Pipeline

| Stage | Purpose |
|---|---|
| `generate` | Build PA geographic networks and derive constraints |
| `fit` | Fit requested ODME families/constraints to those constraints |
| `sample` | Sample fitted nulls and compare ensemble means to constraints |
| `filter` | Sample from fitted nulls, filter those null samples, assess FPR |

Filter calibration is measured on samples from the fitted null, not on the PA
network itself. The PA network supplies constraints only.

## Commands

```bash
uv run python -m benchmarks all --nodes 100,500 --samples 5 --filter-samples 5
uv run python -m benchmarks fit --nodes 500 --families me,w --constraints strength
uv run python -m benchmarks filter --nodes 100 --json --output benchmarks/results/run
uv run python -m benchmarks.legacy_compare --nodes 100,500
uv run python -m benchmarks.legacy_fit_compare --nodes 100,500 --families me,b
uv run python -m benchmarks.fit_memory_matrix --nodes 1000
uv run --with scipy python -m benchmarks.legacy_supported_fit_compare --nodes 1000
```

Options:

| Option | Meaning |
|---|---|
| `--nodes 100,500` | exact sizes |
| `--families me,b,w,wnb` | model families |
| `--constraints ...` | constraint families |
| `--average-degree 8` | PA binary support density control |
| `--density 0.05` | directed density override |
| `--events-per-edge 6` | total event control |
| `--samples 5` | sample-check ensemble size |
| `--filter-samples 5` | null-filter calibration samples |
| `--json` | machine-readable stdout |
| `--output DIR` | logs, networks, summary JSON |

## Legacy comparison

`benchmarks.legacy_compare` extracts the archived C analyzer from git history,
compiles it with `gcc`, runs the same generated input network through both
implementations, and reports JSON with:

- strength/degree/Y2 maximum absolute differences;
- modern Python process time and peak RSS;
- modern load+Rust-compute time excluding import/startup;
- legacy C process time and peak RSS.

It requires GSL development headers and libraries for the archived analyzer.

`benchmarks.legacy_fit_compare` extracts the archived Python strength fitter,
converts `fitter_s.py` to Python 3 in a temporary directory, and compares ME and
B fixed-strength no-self-loop solvers. `benchmarks.fit_memory_matrix` runs every
modern supported fit case in a fresh process and records peak RSS.

N=1000 results are stored in `benchmarks/results/`:

| Benchmark | Main result |
|---|---|
| `legacy_fit_n1000.json` | ME matches and modern is faster; B matches but modern is slower. |
| `legacy_supported_fit_n1000.json` | legacy-supported ME/B/W/degree cases compared or recorded as legacy failures/timeouts. |
| `fit_memory_n1000.json` | all 20 modern cases converged; per-process RSS was ~77-78 MB. |

Legacy-supported N=1000 comparisons show ME strength, W strength, ME
strength-cost, and degree observables match. Legacy strength-edges timed out or
failed on the PA fixture, and legacy strength-degree failed; modern ODME solved
those same cases. Slowest N=1000 modern fits were W strength-cost (~707 s), WNB
strength-cost (~212 s), WNB strength-edges (~162 s), and W strength-edges
(~118 s).

## Known skips

| Case | Status |
|---|---|
| B strength-edges | skipped: known P5 wrapper uses ME kernel |
| B strength-degree | skipped: known P5 wrapper uses ME kernel |
| W no-self-loop N≥50 | track convergence separately until damping/projection is fixed |
| B strength no-self-loop N≥200 | track wall time as a performance regression |

Do not compare the skipped B zero-inflated cases until family-specific Rust kernels are added.
