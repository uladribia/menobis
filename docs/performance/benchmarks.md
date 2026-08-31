---
description: The MENoBiS benchmark suite — matrices, presets, provenance metadata, and rules for drawing conclusions.
---

# Benchmarks

## TL;DR

The benchmark CLI (`python -m benchmarks`) drives the full
generate → fit → sample → filter pipeline over a family × ensemble ×
constraint × regime matrix, writing JSON rows with full provenance. Results
are **measurements for stated hardware, commit, and configuration**, not
universal limits.

> **Benchmark note:** wall times on this site are measurements for the
> stated hardware, commit and benchmark configuration; they are not
> universal limits.

## The 72-cell microcanonical matrix

The `matrix` command benchmarks the microcanonical fixed-strength sampler
over

\[
3\ N \times 2\ \text{regimes} \times 3\ \text{families}
\times 2\ \text{constraints} \times 2\ \text{self-loop policies}
=72.
\]

- 3 node counts: \(N\in\{100,500,1000\}\);
- 2 regimes: sparse, dense (see [Practical scaling](scaling.md) for the
  exact generator parameters);
- 3 families: ME, B, W;
- 2 constraints: strength, strength-cost;
- 2 self-loop policies: with and without.

Run it with:

```bash
uv run python -m benchmarks matrix --self-loops \
    -o benchmarks/results/microcanonical-bench-matrix-sl.json
uv run python -m benchmarks matrix --no-self-loops \
    -o benchmarks/results/microcanonical-bench-matrix-nosl.json
```

A documentation test guards that the preset expands to 72 cells while the
preset remains unchanged.

## The E2E command

The `all` command runs the full generate → fit → sample → filter-FPR
pipeline per cell and is the tool for grand-canonical timing/quality
evidence:

```bash
uv run python -m benchmarks all \
    --regime dense --known-pairs 0.0,0.05,0.20 \
    -o benchmarks/results/e2e-modern.json
```

## Provenance metadata schema

Every benchmark result intended for public interpretation records:

```text
git_sha
date
cpu_model
physical_cores
logical_cores
ram
os
rust_profile
python_version
self_loops
N
family
ensemble
constraint
average_degree
support_density
T_over_E
burn_in_sweeps
sweeps_per_sample
number_of_samples
seed
wall_time
peak_rss
```

GC-specific: `fit_iterations`, `fit_converged`, `fit_residual`.
MCMC-specific when meaningful: `acceptance_rate`, `effective_move_rate`,
`ESS`. Do not expose an ESS field when the route does not calculate it.

## Reading results

Every conclusion from benchmark data must specify:

- family;
- constraint;
- \(N\);
- sparsity regime;
- observable.

Example wording (form only — replace with actual evidence):

> For ME with strength constraints in the tested \(\bar k=8\), \(T/E=8\)
> regime, the finite-sample mean \(Y_2\) values approach each other as \(N\)
> increases, while support metrics can remain visibly different.

Never turn an empirical trend into a theorem.

## Committed evidence

- `benchmarks/results/phase0-baseline/` — legacy low-resolution baseline
  (N=50, both self-loop policies), captured on `refactor/phase-0-foundation`
  commit `c2a2a39`; see its `summary.md` for the recorded behaviour.
- Current matrix/JSON outputs are git-ignored; regenerate them locally and
  treat them as evidence with the metadata above.

## Relation to the notebook

The planned practical GC-vs-MC comparison notebook
(`docs/examples/grand-vs-micro-practical.ipynb`) is the intended empirical
source for ensemble-comparison claims; this page documents how benchmark
evidence is produced so the notebook and the CLI stay consistent.