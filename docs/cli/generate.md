---
description: menobis generate — sample networks from fitted constraints or directly from microcanonical constraints.
---

# `menobis generate`

## TL;DR

Use `menobis generate` to emit one seeded synthetic `EdgeTable`. Most
commands fit constraints from the input network first; the microcanonical
command samples directly from derived constraints. Data goes to stdout
unless `--output` is set.

> The Python API is the authoritative full model interface. CLI commands
> expose a convenience subset for the most common routes and may retain
> command names that do not mirror the model ontology exactly.

## Commands

| Command | Ensemble | Family | Constraint | Notes |
|---|---|---|---|---|
| `strength-poisson` | grand-canonical | ME | strength | default route |
| `strength-multinomial` | canonical | ME | strength | fixed total \(T\), needs `--total-events` |
| `strength-edges-poisson` | grand-canonical | ME | strength-edges | optional `--target-edges` |
| `strength-degree-poisson` | grand-canonical | ME | strength-degree | |
| `degree-events-poisson` | grand-canonical | ME | degree-events | needs `--total-events` |
| `strength-cost-poisson` | grand-canonical | ME | strength-cost | needs `--coordinates` |
| `custom-poisson` | grand-canonical | ME | custom rates | needs `--total-events`, `--ensemble` |
| `strength-degree-mcmc` | microcanonical | ME | strength-degree | exact \((s,k)\); extras-first constructor + degree-fiber trace; no fit |

## Examples

```bash
menobis generate strength-poisson edges.csv --seed 42 -o sample.csv
menobis generate strength-multinomial edges.csv --total-events 1000 --json
menobis generate custom-poisson probabilities.csv --total-events 1000 --ensemble poisson
menobis generate strength-cost-poisson edges.csv --coordinates xy.csv --seed 7
menobis generate strength-edges-poisson edges.csv --target-edges 500
menobis generate strength-degree-poisson edges.csv --seed 99
menobis generate degree-events-poisson edges.csv --total-events 5000
menobis generate strength-degree-mcmc edges.csv --seed 99 --no-self-loops
```

## Options

| Option | Meaning |
|--------|---------|
| `--output`, `-o` | Write edge table |
| `--json` | Print JSON to stdout |
| `--quiet` | Suppress progress |
| `--seed`, `-s` | Random seed |
| `--self-loops/--no-self-loops` | Diagonal handling |
| `--total-events` | Total \(T\) (multinomial, custom, degree-events) |
| `--ensemble` | `poisson` or `multinomial` (custom only) |
| `--target-edges` | Target \(E\) (strength-edges) |
| `--coordinates` | Projected XY coordinate CSV (strength-cost) |
| `--burn-in-sweeps`, `--sweeps-per-sample` | MCMC settings (`strength-degree-mcmc`) |

## Microcanonical sampling

The CLI exposes the fixed-strength-degree microcanonical route
(`strength-degree-mcmc`), which samples directly from derived constraints
with no fitting step. Use the Python API
(`sample_model` with `ensemble=Ensemble.MICROCANONICAL`) for the complete
supported family × ensemble × constraint matrix — see
[Supported models](../guide/supported-models.md) and
[Microcanonical sampling](../guide/microcanonical.md).