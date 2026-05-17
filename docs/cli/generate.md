---
description: Generate sampled ODME networks from fitted constraints.
---

# `odme generate`

## TL;DR

Use `odme generate` to fit implemented constraints from inputs and emit one
seeded synthetic edge table. Data goes to stdout unless `--output` is set.

## Commands

| Command | ODME model | Thesis case | Required extra args |
|---------|------------|-------------|---------------------|
| `poisson` | Fixed-strength ME, grand-canonical | — | — |
| `multinomial` | Fixed-strength ME, canonical | — | `--total-events` |
| `poisson-multinomial` | Fixed-strength ME, Poisson total then multinomial | — | — |
| `custom-pij` | Custom probability ME | 1 | `--total-events`, `--ensemble` |
| `strength-cost-me` | Strength-cost ME | 2 | `--costs` |
| `strength-edges-me` | Strength-edges ME | 3 | optional `--target-edges` |
| `strength-degree-me` | Strength-degree ME | 4 | — |
| `degree-events-me` | Degree-events ME | 5 | `--total-events` |

## Examples

```bash
odme generate poisson edges.csv --seed 42 -o sample.csv
odme generate multinomial edges.csv --total-events 1000 --json
odme generate custom-pij probabilities.csv --total-events 1000 --ensemble poisson
odme generate strength-cost-me edges.csv --costs costs.csv --target-cost 120.0
odme generate strength-edges-me edges.csv --target-edges 500
odme generate strength-degree-me edges.csv --seed 42
odme generate degree-events-me edges.csv --total-events 1000
```

## Options

| Option | Applies to | Meaning |
|--------|------------|---------|
| `--output`, `-o` | all | Write data to a file instead of stdout |
| `--json` | all | Emit JSON instead of CSV |
| `--quiet` | all | Suppress progress messages |
| `--seed`, `-s` | all | Reproducible random seed |
| `--self-loops/--no-self-loops` | model commands where applicable | Include or exclude diagonal pairs |
| `--total-events` | multinomial, custom-pij, degree-events | Total events $T$ |
| `--target-edges` | strength-edges-me | Expected binary edge count; defaults to observed $E$ |
| `--costs` | strength-cost-me | CSV with `source,target,cost` columns |
| `--target-cost` | strength-cost-me | Expected total cost; defaults to observed $C$ |
| `--ensemble` | custom-pij | `poisson` or `multinomial` |

## Output

Output edge tables contain `source,target,weight` rows. Zero-weight sampled
edges are omitted.
