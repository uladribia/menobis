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

## Known skips

| Case | Status |
|---|---|
| B strength-edges | skipped: known P5 wrapper uses ME kernel |
| B strength-degree | skipped: known P5 wrapper uses ME kernel |
| W no-self-loop N≥50 | track convergence separately until damping/projection is fixed |
| B strength no-self-loop N≥200 | track wall time as a performance regression |

Do not compare the skipped B zero-inflated cases until family-specific Rust kernels are added.
