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
| distribution | `poisson`, `geometric`, `binomial`, `negative_binomial`, `multinomial`, `stub_matching`, `bernoulli` |

## Ensemble types

The distribution name implies the ensemble type:

| Ensemble | Legacy | Distributions | Weight support | Fitting |
|----------|--------|---------------|----------------|---------|
| ME (multi-edge) | `ME` | Poisson, Multinomial, Stub matching | $\{0, 1, 2, \ldots\}$ | IPF / scalar search |
| W (weighted) | `W` | Geometric, Negative binomial | $\{0, 1, 2, \ldots\}$ | conic / root-IPF by constraint |
| B (binary layers) | `B` | Binomial, Bernoulli | $\{0, \ldots, M\}$ | IPF with correction |

## Constraint × distribution matrix

| Constraint | Poisson | Geometric | Binomial | Neg. binomial |
|------------|---------|-----------|----------|---------------|
| strength | ✅ | ✅ | ✅ | ✅ |
| strength-cost | ✅ | ✅ | ✅ | ✅ |
| strength-edges | ✅ (zero-inflated) | ✅ (zero-inflated) | ✅ (zero-inflated) | ✅ (zero-inflated) |
| strength-degree | ✅ (zero-inflated) | sampling/filtering ✅ | ✅ (zero-inflated) | sampling/filtering ✅ |
| degree | Bernoulli ✅ | — | — | — |
| degree-events | ✅ (zero-inflated) | ✅ (zero-inflated) | ✅ (zero-inflated) | ✅ (zero-inflated) |
| custom | ✅ | — | — | — |

## Case mapping

| Case | Constraint | Equation | Fit | Sample |
|------|-----------|----------|-----|--------|
| — | strength | $\mathbb{E}[t_{ij}] = x_i y_j$ | `fit_strength_poisson` | `sample_strength_poisson` |
| 1 | custom | $\mathbb{E}[t_{ij}] = T p_{ij}$ | — | `sample_custom_poisson` |
| 2 | strength-cost | $\mathbb{E}[t_{ij}] = x_i y_j e^{-\gamma d_{ij}}$ | `fit_strength_cost_poisson` | `sample_strength_cost_poisson` |
| 3 | strength-edges | zero-inflated occupation + positive Poisson weight | `fit_strength_edges_poisson` | `sample_strength_edges_poisson` |
| 4 | strength-degree | zero-inflated occupation + positive Poisson weight | `fit_strength_degree_poisson` | `sample_strength_degree_poisson` |
| 5 | degree-events | Bernoulli occupation + positive Poisson weight | `fit_degree_bernoulli` | `sample_degree_events_poisson` |

W fixed-strength and strength-cost fitting use Clarabel exponential cones.
W degree-events uses a scalar `q` solve plus Bernoulli IPF. W strength-edges
currently uses an experimental monotone root/IPF solver over `lambda`. W fits
return constraint-oriented result types with `family`, optional `layers`, and
nested diagnostics; see [W Ensemble](w-ensemble.md).

Additional strength samplers for ensemble comparison:

| Sampler | Ensemble | Total weight |
|---------|----------|-------------|
| `sample_strength_multinomial` | canonical | exactly $T$ |
| `sample_strength_stub_matching` | stub_matching | exactly $T$ |
| `sample_strength_poisson_multinomial` | mixed | Poisson $T$ then multinomial |

## CLI mapping

| Constraint | `odme fit` | `odme generate` | `odme filter` |
|-----------|------------|-----------------|---------------|
| strength | `strength-poisson` | `strength-poisson` | `strength-poisson` |
| strength (W geometric) | `strength-geometric` | — | — |
| strength (W negative binomial) | `strength-negative-binomial` | — | — |
| strength (B) | — | — | — |
| strength-cost | `strength-cost-poisson` | `strength-cost-poisson` | `strength-cost-poisson` |
| strength-edges | `strength-edges-poisson` | `strength-edges-poisson` | `strength-edges-poisson` |
| strength-degree | `strength-degree-poisson` | `strength-degree-poisson` | `strength-degree-poisson` |
| degree | `degree-bernoulli` | `degree-events-poisson` | `degree-events-poisson` |
| custom | — | `custom-poisson` | `custom-poisson` |

## Zero-inflated models

Cases 3, 4, and 5 use zero-inflated distributions: a Bernoulli occupation
draw followed by a positive-weight conditional draw. The occupation formula
depends on the constraint; the positive-weight distribution follows the chosen
family (Poisson, geometric, negative binomial, or binomial where implemented).

## Reference

[1] Sagarra, O. *Statistical mechanics of multi-edge networks*.
    PhD thesis, Universitat de Barcelona, 2015.
    <https://hdl.handle.net/10803/400560>
