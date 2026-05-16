---
description: Generate sampled ODME networks from fitted constraints.
---

# `odme generate`

## TL;DR

Use `odme generate` to fit a model from an observed edge table and emit one
seeded synthetic edge table sample.

## Commands

| Command | Model | Output columns |
|---------|-------|----------------|
| `poisson` | Fixed-strength Poisson multi-edge | `source,target,weight` |
| `strength-degree-zip` | Strength-degree zero-inflated shifted-Poisson | `source,target,weight` |

## Examples

```bash
odme generate poisson edges.csv --seed 42 --output sample.csv
odme generate strength-degree-zip edges.csv --seed 42 --json
```

## Options

| Option | Applies to | Meaning |
|--------|------------|---------|
| `--output`, `-o` | all | Write data to a file instead of stdout |
| `--json` | all | Emit JSON instead of CSV |
| `--quiet` | all | Suppress progress messages |
| `--seed`, `-s` | all | Reproducible random seed |
| `--self-loops/--no-self-loops` | `strength-degree-zip` | Include or exclude diagonal constraints |
