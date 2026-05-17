---
description: Fit ODME model parameters from edge tables.
---

# `odme fit`

## TL;DR

Use `odme fit` to solve Lagrange multipliers from an observed edge table.

## Commands

| Command | Model | Case | Output columns |
|---------|-------|------|----------------|
| `strength-poisson` | Strength Poisson | — | `node,x,y` |
| `degree-bernoulli` | Degree Bernoulli | 5 | `node,x,y` |
| `strength-cost-poisson` | Strength-cost Poisson | 2 | `node,x,y,gamma` |
| `strength-edges-poisson` | Strength-edges Poisson | 3 | `node,x,y,lambda` |
| `strength-degree-poisson` | Strength-degree Poisson | 4 | `node,x,y,z,w` |

## Examples

```bash
odme fit strength-poisson edges.csv --json
odme fit strength-poisson edges.csv --output fit.csv
odme fit strength-cost-poisson edges.csv --costs costs.csv --target-cost 120.0
odme fit strength-edges-poisson edges.csv --target-edges 500
odme fit strength-degree-poisson edges.csv --output fit.csv
odme fit degree-bernoulli edges.csv --json
```

## Options

| Option | Meaning |
|--------|---------|
| `--output`, `-o` | Write multipliers to CSV |
| `--json` | Print JSON to stdout |
| `--quiet` | Suppress progress |
| `--self-loops/--no-self-loops` | Diagonal handling |
| `--target-edges` | Target $E$ (strength-edges) |
| `--target-cost` | Target $C$ (strength-cost) |
| `--costs` | Cost CSV path (strength-cost) |
