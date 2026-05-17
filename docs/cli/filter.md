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
| `custom-rates` | user-supplied Poisson rates | edge table + rates CSV |

## Examples

```bash
odme filter fixed-strength edges.csv --output-prefix filtered/
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
| `--max-absent` | cap absent output size |
