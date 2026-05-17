---
description: Generate sampled ODME networks from fitted constraints.
---

# `odme generate`

## TL;DR

Use `odme generate` to fit constraints from inputs and emit one
seeded synthetic edge table. Data goes to stdout unless `--output` is set.

## Commands

| Command | Model | Case | Required extra args |
|---------|-------|------|---------------------|
| `strength-poisson` | Strength Poisson | — | — |
| `strength-multinomial` | Strength multinomial | — | `--total-events` |
| `strength-poisson-multinomial` | Strength Poisson-multinomial | — | — |
| `custom-poisson` | Custom Poisson | 1 | `--total-events`, `--ensemble` |
| `strength-cost-poisson` | Strength-cost Poisson | 2 | `--costs` |
| `strength-edges-poisson` | Strength-edges Poisson (ZIP) | 3 | optional `--target-edges` |
| `strength-degree-poisson` | Strength-degree Poisson (ZIP) | 4 | — |
| `degree-events-poisson` | Degree-events Poisson (ZIP) | 5 | `--total-events` |

## Examples

```bash
odme generate strength-poisson edges.csv --seed 42 -o sample.csv
odme generate strength-multinomial edges.csv --total-events 1000 --json
odme generate custom-poisson probabilities.csv --total-events 1000 --ensemble poisson
odme generate strength-cost-poisson edges.csv --costs costs.csv --seed 7
odme generate strength-edges-poisson edges.csv --target-edges 500
odme generate strength-degree-poisson edges.csv --seed 99
odme generate degree-events-poisson edges.csv --total-events 5000
```

## Options

| Option | Meaning |
|--------|---------|
| `--output`, `-o` | Write edge table |
| `--json` | Print JSON to stdout |
| `--quiet` | Suppress progress |
| `--seed`, `-s` | Random seed |
| `--self-loops/--no-self-loops` | Diagonal handling |
| `--total-events` | Total $T$ (multinomial, custom, degree-events) |
| `--ensemble` | `poisson` or `multinomial` (custom only) |
| `--target-edges` | Target $E$ (strength-edges) |
| `--costs` | Cost CSV (strength-cost) |
