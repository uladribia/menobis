---
description: Taxonomy of ODME models, ensembles, and API mapping.
---

# Thesis cases and taxonomy

## TL;DR

ODME models are organized by **constraint** (what is fixed), **ensemble type**
(ME/W/B), and **distribution** (how weights are drawn). All public names follow
`{operation}_{constraint}_{distribution}`.

## Naming convention

```text
{operation}_{constraint}_{distribution}
```

| Token | Values |
|-------|--------|
| operation | `fit`, `sample`, `filter` |
| constraint | `strength`, `strength_edges`, `strength_degree`, `strength_cost`, `degree`, `custom` |
| distribution | `poisson`, `geometric`, `binomial`, `neg_binomial`, `multinomial`, `microcanonical`, `bernoulli` |

## Ensemble types

The distribution name implies the ensemble type:

| Ensemble | Legacy | Distributions | Weight support | Fitting |
|----------|--------|---------------|----------------|---------|
| ME (multi-edge) | `ME` | Poisson, Multinomial, Microcanonical | $\{0, 1, 2, \ldots\}$ | IPF |
| W (weighted) | `W` | Geometric, Negative binomial | $\{0, 1, 2, \ldots\}$ | Bounded optimization |
| B (binary layers) | `B` | Binomial, Bernoulli | $\{0, \ldots, M\}$ | IPF with correction |

## Constraint × distribution matrix

| Constraint | Poisson | Geometric | Binomial | Neg. binomial |
|------------|---------|-----------|----------|---------------|
| strength | ✅ | ✅ | ✅ | ✅ (no fit yet) |
| strength-cost | ✅ | — | — | — |
| strength-edges | ✅ (ZIP) | — | — | — |
| strength-degree | ✅ (ZIP) | — | — | — |
| degree | Bernoulli ✅ | — | — | — |
| custom | ✅ | — | — | — |

## Case mapping

| Case | Constraint | Equation | Fit | Sample |
|------|-----------|----------|-----|--------|
| — | strength | $\mathbb{E}[t_{ij}] = x_i y_j$ | `fit_strength_poisson` | `sample_strength_poisson` |
| 1 | custom | $\mathbb{E}[t_{ij}] = T p_{ij}$ | — | `sample_custom_poisson` |
| 2 | strength-cost | $\mathbb{E}[t_{ij}] = x_i y_j e^{-\gamma d_{ij}}$ | `fit_strength_cost_poisson` | `sample_strength_cost_poisson` |
| 3 | strength-edges | ZIP occupation + ZTP weight | `fit_strength_edges_poisson` | `sample_strength_edges_poisson` |
| 4 | strength-degree | ZIP occupation + ZTP weight | `fit_strength_degree_poisson` | `sample_strength_degree_poisson` |
| 5 | degree-events | Bernoulli occupation + ZTP weight | `fit_degree_bernoulli` | `sample_degree_events_poisson` |

Additional strength samplers for ensemble comparison:

| Sampler | Ensemble | Total weight |
|---------|----------|-------------|
| `sample_strength_multinomial` | canonical | exactly $T$ |
| `sample_strength_microcanonical` | microcanonical | exactly $T$ |
| `sample_strength_poisson_multinomial` | mixed | Poisson $T$ then multinomial |

## CLI mapping

| Constraint | `odme fit` | `odme generate` | `odme filter` |
|-----------|------------|-----------------|---------------|
| strength | `strength-poisson` | `strength-poisson` | `strength-poisson` |
| strength (B) | — | — | — |
| strength-cost | `strength-cost-poisson` | `strength-cost-poisson` | `strength-cost-poisson` |
| strength-edges | `strength-edges-poisson` | `strength-edges-poisson` | `strength-edges-poisson` |
| strength-degree | `strength-degree-poisson` | `strength-degree-poisson` | `strength-degree-poisson` |
| degree | `degree-bernoulli` | `degree-events-poisson` | `degree-events-poisson` |
| custom | — | `custom-poisson` | `custom-poisson` |

## Zero-inflated models

Cases 3, 4, and 5 use zero-inflated distributions: a Bernoulli occupation
draw followed by a zero-truncated positive-weight draw. The occupation
formula depends on the constraint; the positive-weight distribution is
currently always Poisson (ZTP).

## Reference

[1] Sagarra, O. *Statistical mechanics of multi-edge networks*.
    PhD thesis, Universitat de Barcelona, 2015.
    <https://hdl.handle.net/10803/400560>
