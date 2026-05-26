---
description: Fit MENoBiS model parameters from edge tables.
---

# `menobis fit`

## TL;DR

Use `menobis fit` to solve Lagrange multipliers from an observed edge table.

## Commands

| Command | Model | Case | Output columns |
|---------|-------|------|----------------|
| `strength-poisson` | Strength Poisson | — | `node,x,y` |
| `strength-geometric` | Strength geometric W | W | `node,x,y,layers,status,...` |
| `strength-negative-binomial` | Strength negative-binomial W | W | `node,x,y,layers,status,...` |
| `degree-bernoulli` | Degree Bernoulli | 5 | `node,x,y` |
| `strength-cost-poisson` | Strength-cost Poisson | 2 | `node,x,y,gamma` |
| `strength-edges-poisson` | Strength-edges Poisson | 3 | `node,x,y,lambda` |
| `strength-degree-poisson` | Strength-degree Poisson | 4 | `node,x,y,z,w` |

## Examples

```bash
menobis fit strength-poisson edges.csv --json
menobis fit strength-poisson edges.csv --output fit.csv
menobis fit strength-geometric edges.csv --json
menobis fit strength-negative-binomial edges.csv --layers 3 --output fit.csv
menobis fit strength-cost-poisson edges.csv --costs costs.csv --target-cost 120.0
menobis fit strength-edges-poisson edges.csv --target-edges 500
menobis fit strength-degree-poisson edges.csv --output fit.csv
menobis fit degree-bernoulli edges.csv --json
```

## Options

| Option | Meaning |
|--------|---------|
| `--output`, `-o` | Write multipliers to CSV |
| `--json` | Print JSON to stdout |
| `--quiet` | Suppress progress |
| `--self-loops/--no-self-loops` | Diagonal handling |
| `--target-edges` | Target $E$ (strength-edges) |
| `--layers` | Negative-binomial layer count $M > 1$ |
| `--tolerance` | Solver tolerance |
| `--max-iterations` | Solver iteration cap |
| `--target-cost` | Target $C$ (strength-cost) |
| `--costs` | Cost CSV path (strength-cost) |
