---
description: Generate sampled MENoBiS networks from fitted constraints.
---

# `menobis generate`

## TL;DR

Use `menobis generate` to fit constraints from inputs and emit one
seeded synthetic edge table. Data goes to stdout unless `--output` is set.

## Commands

| Command | Model | Case | Required extra args |
|---------|-------|------|---------------------|
| `strength-poisson` | Strength Poisson | — | — |
| `strength-multinomial` | Strength multinomial | — | `--total-events` |
| `strength-poisson-multinomial` | Strength Poisson-multinomial | — | — |
| `custom-poisson` | Custom Poisson | 1 | `--total-events`, `--ensemble` |
| `strength-cost-poisson` | Strength-cost Poisson | 2 | `--coordinates` |
| `strength-edges-poisson` | Strength-edges Poisson (zero-inflated) | 3 | optional `--target-edges` |
| `strength-degree-poisson` | Strength-degree Poisson (zero-inflated) | 4 | — |
| `degree-events-poisson` | Degree-events Poisson (zero-inflated) | 5 | `--total-events` |

## Examples

```bash
menobis generate strength-poisson edges.csv --seed 42 -o sample.csv
menobis generate strength-multinomial edges.csv --total-events 1000 --json
menobis generate custom-poisson probabilities.csv --total-events 1000 --ensemble poisson
menobis generate strength-cost-poisson edges.csv --coordinates xy.csv --seed 7
menobis generate strength-edges-poisson edges.csv --target-edges 500
menobis generate strength-degree-poisson edges.csv --seed 99
menobis generate degree-events-poisson edges.csv --total-events 5000
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
| `--coordinates` | Projected XY coordinate CSV (strength-cost) |
