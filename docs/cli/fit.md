---
description: Fit ODME maximum-entropy model parameters from edge tables.
---

# `odme fit`

## TL;DR

Use `odme fit` to infer model multipliers from an observed weighted edge table.
Data goes to stdout unless `--output` is set; progress goes to stderr unless
`--quiet` or `--json` is used.

## Commands

| Command | ODME model | Thesis case | Output columns |
|---------|------------|-------------|----------------|
| `strengths` | Fixed-strength ME | — | `node,x,y` |
| `degrees` | Degree-events ME binary fit | 5 | `node,x,y` |
| `strength-cost-me` | Strength-cost ME | 2 | `node,x,y,gamma` |
| `strength-edges-me` | Strength-edges ME | 3 | `node,x,y,lambda` |
| `strength-degree-me` | Strength-degree ME | 4 | `node,x,y,z,w` |

## Examples

```bash
odme fit strengths edges.csv --output strength-fit.csv
odme fit degrees edges.csv --json
odme fit strength-cost-me edges.csv --costs costs.csv --target-cost 120.0
odme fit strength-edges-me edges.csv --target-edges 500 --json
odme fit strength-degree-me edges.csv --output strength-degree-fit.csv
```

## Options

| Option | Applies to | Meaning |
|--------|------------|---------|
| `--output`, `-o` | all | Write data to a file instead of stdout |
| `--json` | all | Emit JSON instead of CSV |
| `--quiet` | all | Suppress progress messages |
| `--self-loops/--no-self-loops` | all except `strengths` | Include or exclude diagonal pairs |
| `--target-edges` | `strength-edges-me` | Expected binary edge count; defaults to observed $E$ |
| `--costs` | `strength-cost-me` | CSV with `source,target,cost` columns |
| `--target-cost` | `strength-cost-me` | Expected total cost; defaults to observed $C$ |

## Cost files

For `strength-cost-me`, omitted cost pairs are treated as zero cost by the
current implementation. Use a complete cost table unless that is intentional.
