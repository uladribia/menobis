---
description: Benchmark commands, current scaling results, and practical limits.
---

# Benchmarking

## TL;DR

Use exact node-size runs for expensive fitting benchmarks. The all-case N=5000
run timed out before partial benchmarks, so large runs must be chunked and saved
incrementally.

## Commands

```bash
uv run maturin develop --release -m crates/odme-python/Cargo.toml
uv run python -m benchmarks fit --nodes 100 --max-n 100 \
  --known-fractions 0.05,0.40 --tolerance 1e-4 --verbose 2
```

Options:

| Option | Meaning |
|---|---|
| `--nodes 25,50` | exact sizes to run |
| `--max-n N` | maximum allowed size |
| `--known-fractions` | partial known-weight fractions |
| `--verbose 2` | model-level convergence logging |
| `--plot` | write scaling/residual figures |
| `--output DIR` | output directory |

## Latest N=100 validation

| Scope | Rows | Failures |
|---|---:|---:|
| full ME/B/W | 20 | 0 |
| partial ME/B/W, 5% known weights | 20 | 0 |
| partial ME/B/W, 40% known weights | 20 | 0 |

Partial benchmark inputs are known weighted pairs. Occupation contributions for
edge/degree constraints are inferred from positive known weights.

## Large release attempt

The command below timed out after six hours:

```bash
uv run python -m benchmarks fit --nodes 25,50,100,500,1000,5000 \
  --max-n 5000 --known-fractions 0.05,0.40 --tolerance 1e-4 \
  --verbose 2 --plot --output benchmarks/results/release-fit-25-5000
```

It reached N=5000 B strength-degree and did not save JSON because saving occurs
at the end. Add incremental saves before rerunning large suites.

## Observed full-fit timings

| Case | N=1000 | N=5000 before timeout |
|---|---:|---:|
| ME strength | ~0 s | ~0 s |
| ME strength-edges | 49 s | 1463 s |
| ME strength-degree | 231 s | 6237 s |
| B strength | 0.01 s | 0.28 s |
| B strength-edges | 50 s | 1416 s |
| B strength-degree | 236 s | 6224 s |
| W geometric strength-edges | 368 s | not reached |
| W geometric strength-degree | 117 s | not reached |
| W NB strength-edges | 340 s | not reached |
| W NB strength-degree | 184 s | not reached |

## Why timings differ

| Solver shape | Examples | Scaling driver |
|---|---|---|
| analytic | ME strength | O(N) |
| scalar + IPF | degree-events | O(N²) but few iterations |
| coordinate/IPF | strength-edges | O(N² I) |
| four-multiplier IPF | strength-degree | O(N² I), high constant |
| W coordinate | W edges/degree | O(N² I) with expensive kernels |
| conic | W strength/cost | interior point over O(N²) cones |

## Memory model

Generation streams pair distributions and stores only the sampled sparse edge
list. Fitting dense all-pair constraints usually sweeps N² candidate pairs and
can be CPU-bound before memory-bound. Custom sparse probability models remain
O(Ep), where Ep is support size.

## Recommendation

Do not run all N=5000 fitting cases in one process. Run chunks such as:

```bash
uv run python -m benchmarks fit --nodes 500,1000 --max-n 1000 --verbose 2
uv run python -m benchmarks fit --nodes 5000 --max-n 5000 --verbose 2
```

After incremental saving is implemented, use the full matrix again.
