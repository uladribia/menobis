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
| `fixed-strength` | fit fixed-strength Poisson ME | edge table |
| `strength-cost` | fit strength-cost Poisson ME | edge table + costs CSV |
| `strength-edges` | fit strength-edges ZIP ME | edge table |
| `strength-degree` | fit strength-degree ZIP ME | edge table |
| `degree-events` | fit degree-events ZIP ME | edge table |
| `custom-rates` | user-supplied Poisson rates | edge table + rates CSV |

## Examples

```bash
odme filter fixed-strength edges.csv --output-prefix filtered/
odme filter strength-edges edges.csv --target-edges 500 --output-prefix filtered/
odme filter strength-cost edges.csv --costs costs.csv --target-cost 1.5 --output-prefix filtered/
odme filter strength-degree edges.csv --output-prefix filtered/
odme filter degree-events edges.csv --output-prefix filtered/
odme filter custom-rates edges.csv --rates rates.csv --output-prefix filtered/
```

The custom rates CSV must contain `source,target,rate`, where `rate` is the
occupation number $T p_{ij}$.

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
| `--target-edges` | target edge count (strength-edges) |
| `--target-cost` | target average cost (strength-cost) |
| `--costs` | cost CSV path (strength-cost) |
| `--rates` | rates CSV path (custom-rates) |
