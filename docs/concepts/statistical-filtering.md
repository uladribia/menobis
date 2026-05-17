---
description: Statistical filtering against ODME null models.
---

# Statistical filtering

## TL;DR

Filtering compares observed edge weights against an independent ODME null model.
The default test is two-sided with split alpha. Absent edges are reported
separately and are prefiltered by binary occupation probability.

## Tail rules

| Tail | Rule |
|------|------|
| upper | flag if $P(T \ge t) < \alpha$ |
| lower | flag if $P(T \le t) < \alpha$ |
| two-sided | flag upper/lower independently with $\alpha/2$ |

## Supported nulls

Initial filtering supports independent grand-canonical distributions:

| Model | Distribution |
|-------|--------------|
| fixed-strength ME | Poisson($x_i y_j$) |
| strength-edges ME | ZIP/ZTP with fitted occupation and rate $x_i y_j$ |

Canonical multinomial filters are intentionally out of scope because pair tests
are coupled by the fixed total event count.

## Absent edges

Absent-edge filtering is opt-in. It streams candidate pairs with observed
weight zero and keeps only pairs whose null occupation probability satisfies:

$$
P(t_{ij}>0) \ge \texttt{min\_occupation}.
$$

For Poisson models, $P(t_{ij}>0)=1-e^{-\lambda_{ij}}$.

## Python API

```python
from odme.filtering import filter_fixed_strength_me

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
