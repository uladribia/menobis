---
description: Generate sampled ODME networks from fitted constraints.
---

# `odme generate`

## TL;DR

Use `odme generate` to fit implemented ME constraints from inputs and emit one
seeded synthetic edge table. Data goes to stdout unless `--output` is set.

## Commands

| Command | Model | Required extra args |
|---------|-------|---------------------|
| `poisson` | fixed-strength ME, GC Poisson | — |
| `multinomial` | fixed-strength ME, canonical multinomial | `--total-events` |
| `poisson-multinomial` | fixed-strength ME, Poisson total then multinomial | — |
| `degree-events-me` | fixed-degree ME with expected events | `--total-events` |
| `strength-degree-me` | fixed-strength + degree ME | — |
| `strength-edges-me` | fixed-strength + total binary edges ME | optional `--target-edges` |
| `custom-pij` | custom `p_ij` ME | `--total-events`, `--ensemble` |

## Examples

```bash
odme generate poisson edges.csv --seed 42 -o sample.csv
odme generate multinomial edges.csv --total-events 1000 --json
odme generate poisson-multinomial edges.csv --seed 42
odme generate degree-events-me edges.csv --total-events 1000
odme generate strength-degree-me edges.csv --seed 42
odme generate strength-edges-me edges.csv --target-edges 500
odme generate custom-pij probabilities.csv --total-events 1000 --ensemble poisson
```

## Options

| Option | Meaning |
|--------|---------|
| `--output`, `-o` | Write data to a file instead of stdout |
| `--json` | Emit JSON instead of CSV |
| `--quiet` | Suppress progress messages |
| `--seed`, `-s` | Reproducible random seed |
| `--self-loops/--no-self-loops` | Include or exclude diagonal constraints where applicable |
