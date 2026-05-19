---
description: Filter observed edges against ODME null models.
---

# `odme filter`

## TL;DR

Use `odme filter` to classify observed edges as upper-significant,
lower-significant, or compatible with an independent ODME null model.

## Commands

| Command | Null model | Input |
|---------|------------|-------|
| `strength-poisson` | fit strength Poisson | edge table |
| `strength-cost-poisson` | fit strength-cost Poisson | edge table + costs CSV |
| `strength-edges-poisson` | fit strength-edges Poisson (zero-inflated) | edge table |
| `strength-degree-poisson` | fit strength-degree Poisson (zero-inflated) | edge table |
| `degree-events-poisson` | fit degree-events Poisson (zero-inflated) | edge table |
| `custom-poisson` | user-supplied Poisson rates | edge table + rates CSV |

## Examples

```bash
odme filter strength-poisson edges.csv --output-prefix filtered/
odme filter strength-edges-poisson edges.csv --target-edges 500 --output-prefix filtered/
odme filter strength-cost-poisson edges.csv --costs costs.csv --target-cost 1.5 --output-prefix filtered/
odme filter strength-degree-poisson edges.csv --output-prefix filtered/
odme filter degree-events-poisson edges.csv --output-prefix filtered/
odme filter custom-poisson edges.csv --rates rates.csv --output-prefix filtered/
```

## Outputs

`--output-prefix DIR` writes:

| File | Meaning |
|------|---------|
| `upper.csv` | observed edges heavier than expected |
| `lower.csv` | observed positive edges lighter than expected |
| `compatible.csv` | observed edges compatible with the null |
| `absent_lower.csv` | absent pairs that should exist under the null |

Each file contains p-values, expected weight, and occupation probability.

## Options

| Option | Meaning |
|--------|---------|
| `--alpha` | significance level |
| `--tail` | `upper`, `lower`, or `two-sided` |
| `--correction` | `none`, `bonferroni`, or `fdr` |
| `--detect-absent` | stream absent-pair candidates |
| `--min-occupation` | absent-edge occupation threshold |
| `--min-expected` | absent expected-weight threshold |
| `--max-absent` | cap absent output size |
| `--self-loops/--no-self-loops` | diagonal handling |
| `--target-edges` | target edge count (strength-edges-poisson) |
| `--target-cost` | target average cost (strength-cost-poisson) |
| `--costs` | cost CSV path (strength-cost-poisson) |
| `--rates` | rates CSV path (custom-poisson) |
