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
| `degrees` | Fixed-degree binary | `node,x,y` |
| `strength-degree-zip` | Exact ME fixed-strength-degree ZIP | `node,x,y,z,w` |

## Examples

```bash
odme fit strengths edges.csv --output strength-fit.csv
odme fit degrees edges.csv --json
odme fit strength-degree-zip edges.csv --output zip-fit.csv
```

## Options

| Option | Applies to | Meaning |
|--------|------------|---------|
| `--output`, `-o` | all | Write data to a file instead of stdout |
| `--json` | all | Emit JSON instead of CSV |
| `--quiet` | all | Suppress progress messages |
| `--self-loops/--no-self-loops` | `degrees`, `strength-degree-zip` | Include or exclude diagonal constraints |
