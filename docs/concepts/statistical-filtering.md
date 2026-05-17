---
description: Statistical filtering against ODME null models.
---

# Statistical filtering

## TL;DR

Filtering compares observed edge weights against an independent ODME null model.
The default test is two-sided with split alpha. Rust streams provider-backed
pair distributions into generic observed and absent filter sinks.

## Tail rules

| Tail | Rule |
|------|------|
| upper | flag if $P(T \ge t) < \alpha$ |
| lower | flag if $P(T \le t) < \alpha$ |
| two-sided | flag upper/lower independently with $\alpha/2$ |

## Implementation pipeline

Filtering shares the same pair-distribution abstraction as generation:

```text
model parameters -> PairDistributionProvider -> PairDistribution -> filter sink
```

`PairDistribution` computes expectations, occupation probabilities, and tail
p-values. Providers expose either all pairs or sparse support, so filtering does
not materialize dense $N^2$ rate matrices.

## Supported nulls

Filtering supports all independent grand-canonical distributions:

| Model | Distribution |
|-------|--------------|
| fixed-strength ME | Poisson($x_i y_j$) |
| strength-cost ME | Poisson($x_i y_j e^{-\gamma d_{ij}}$) |
| custom rates | Poisson($\lambda_{ij}$), where $\lambda_{ij}=T p_{ij}$ |
| strength-edges ME | ZIP/ZTP with fitted occupation and rate $x_i y_j$ |
| strength-degree ME | ZIP/ZTP with fitted occupation and rate $x_i y_j$ |
| degree-events ME | ZIP/ZTP with binary degree occupation and shared rate |

| partial constraints | Poisson rates from combined known + free-pair rates |

Partial-constraint filtering uses `filter_custom_rates_poisson` with the
combined rate table from `PartialFitResult.as_probability_table()`.

Canonical multinomial filters are intentionally out of scope because pair tests
are coupled by the fixed total event count. Custom inputs therefore use
independent Poisson rates, not fixed-total multinomial probabilities.

## Absent edges

Absent-edge filtering is opt-in. It streams provider candidate pairs with
observed weight zero and keeps only pairs whose null occupation probability
satisfies:

$$
P(t_{ij}>0) \ge \texttt{min\_occupation}.
$$

For Poisson models, $P(t_{ij}>0)=1-e^{-\lambda_{ij}}$.

## Python API

```python
from odme.filtering import (
    filter_fixed_strength_me,
    filter_strength_cost_me,
    filter_strength_degree_me,
    filter_strength_edges_me,
    filter_degree_events_me,
    filter_custom_rates_poisson,
)

result = filter_fixed_strength_me(
    edges,
    alpha=0.05,
    tail="two-sided",
    detect_absent=True,
    min_occupation=0.5,
)
```

`result.upper`, `result.lower`, `result.compatible`, and
`result.absent_lower` contain sparse edge tables plus p-values.

## CLI

```bash
odme filter fixed-strength edges.csv --output-prefix filtered/
odme filter strength-edges edges.csv --target-edges 500 --output-prefix filtered/
odme filter strength-cost edges.csv --costs costs.csv --target-cost 1.5 --output-prefix filtered/
odme filter strength-degree edges.csv --output-prefix filtered/
odme filter degree-events edges.csv --output-prefix filtered/
odme filter custom-rates edges.csv --rates rates.csv --output-prefix filtered/
```

The custom rates file must contain `source,target,rate`, where `rate` is the
occupation number $T p_{ij}$.

## Calibration benchmark

![Filter calibration](../figures/filter_calibration.png)

The calibration is conservative because p-values are discrete and the plotted
fraction is measured over existing positive edges. Generate the calibration data
and figure with:

```bash
uv run python benchmarks/bench_filter_calibration.py
uv run python benchmarks/plot_filter_calibration.py
```
