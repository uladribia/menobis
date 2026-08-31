---
description: menobis fit — fit grand-canonical multipliers from an observed network.
---

# `menobis fit`

## TL;DR

Use `menobis fit` to fit grand-canonical multipliers from an observed
network and emit the fitted model. Data goes to stdout unless `--output` is
set.

> The Python API is the authoritative full model interface. CLI commands
> expose a convenience subset for the most common grand-canonical routes and
> may retain command names that do not mirror the model ontology exactly.

## Commands

| Command | Ensemble | Family | Constraint | Notes |
|---|---|---|---|---|
| `strength-poisson` | grand-canonical | ME | strength | default route |
| `strength-geometric` | grand-canonical | W | strength | geometric \(M=1\) |
| `strength-negative-binomial` | grand-canonical | W | strength | needs `--layers` |
| `degree-bernoulli` | grand-canonical | ME | degree-events | binary degree fit |
| `strength-degree-poisson` | grand-canonical | ME | strength-degree | zero-inflated |
| `strength-edges-poisson` | grand-canonical | ME | strength-edges | zero-inflated |
| `strength-cost-poisson` | grand-canonical | ME | strength-cost | needs coordinates |

## Examples

```bash
menobis fit strength-poisson edges.csv --seed 42
menobis fit strength-negative-binomial edges.csv --layers 4 --json
menobis fit strength-cost-poisson edges.csv --coordinates xy.csv
```

## Options

| Option | Meaning |
|--------|---------|
| `--output`, `-o` | Write the fitted model |
| `--json` | Print JSON to stdout |
| `--quiet` | Suppress progress |
| `--self-loops/--no-self-loops` | Diagonal handling |
| `--layers` | B/W layer parameter \(M\) (B/W routes) |
| `--coordinates` | Projected XY coordinate CSV (cost routes) |

The complete supported matrix is the Python API
[Supported models](../guide/supported-models.md).