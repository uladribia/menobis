---
description: Fit ODME maximum-entropy model parameters from edge tables.
---

# `odme fit`

## TL;DR

Use `odme fit` to infer model multipliers from an observed weighted edge table.
Data goes to stdout unless `--output` is set.

## Commands

| Command | Model | Output columns |
|---------|-------|----------------|
| `strengths` | Fixed-strength multi-edge | `node,x,y` |
| `degrees` | Fixed-degree binary occupation | `node,x,y` |
| `strength-degree-me` | Exact ME fixed-strength + degree | `node,x,y,z,w` |
| `strength-edges-me` | Exact ME fixed-strength + total edges | `node,x,y,lambda` |

## Examples

```bash
odme fit strengths edges.csv --output strength-fit.csv
odme fit degrees edges.csv --json
odme fit strength-degree-me edges.csv --output strength-degree-fit.csv
odme fit strength-edges-me edges.csv --target-edges 500 --json
```

## Options

| Option | Applies to | Meaning |
|--------|------------|---------|
| `--output`, `-o` | all | Write data to a file instead of stdout |
| `--json` | all | Emit JSON instead of CSV |
| `--quiet` | all | Suppress progress messages |
| `--self-loops/--no-self-loops` | `degrees`, `strength-degree-me`, `strength-edges-me` | Include or exclude diagonal constraints |
| `--target-edges` | `strength-edges-me` | Expected binary edge count; defaults to observed count |
