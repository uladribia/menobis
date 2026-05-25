---
description: Taxonomy of ODME model cases, ensembles, and public names.
---

# Thesis cases and taxonomy

## TL;DR

Public ODME names should make the thesis ontology explicit:
`{operation}_{constraint}_{distribution}`. The mathematical source of truth is
[Model ontology](model-ontology.md); this page maps that ontology to API and CLI
names.

## Naming convention

```text
{operation}_{constraint}_{distribution}
```

| Token | Values |
|---|---|
| operation | `fit`, `sample`, `filter` |
| constraint | `strength`, `strength_cost`, `strength_edges`, `strength_degree`, `degree_events`, `custom` |
| distribution | `poisson`, `binomial`, `geometric`, `negative_binomial`, `multinomial`, `stub_matching`, `bernoulli` |

## Ensemble support

| Ensemble | Families | Pair coupling | Public examples |
|---|---|---|---|
| Grand canonical | ME, B, W | independent pairs | `fit_strength_poisson`, `fit_strength_binomial`, `fit_strength_geometric` |
| Canonical | ME only | fixed total events | `sample_strength_multinomial` |
| Microcanonical | ME fixed strength only | exact strength sequence | `sample_strength_stub_matching` |

B and W must reject canonical or microcanonical requests until a supported
thesis formulation is implemented.

## Constraint families

| Constraint | Linear in $\mathbb{E}[t]$ | Uses $\mathbb{E}[\Theta(t>0)]$ | Notes |
|---|:---:|:---:|---|
| strength | ✅ | — | Total events are induced by strengths. |
| strength-cost | ✅ | — | Adds family-orthogonal cost provider. |
| degree-events | — | ✅ | Occupation + positive-weight family parameter. |
| strength-edges | ✅ | ✅ | Zero-inflated global edge-count constraint. |
| strength-degree | ✅ | ✅ | Zero-inflated node-degree constraint. |
| partial | depends on parent | depends on parent | Frozen pairs are deducted first. |

## ME thesis case map

| Case | Constraint | Fit | Sample |
|---|---|---|---|
| 1 | custom pair probabilities | — | `sample_custom_poisson`, `sample_custom_multinomial` |
| 2 | strength-cost | `fit_strength_cost_poisson` | `sample_strength_cost_poisson` |
| 3 | strength-edges | `fit_strength_edges_poisson` | `sample_strength_edges_poisson` |
| 4 | strength-degree | `fit_strength_degree_poisson` | `sample_strength_degree_poisson` |
| 5 | degree-events | `fit_degree_events_poisson` | `sample_degree_events_poisson` |

Strength-only ME uses `fit_strength_poisson`; canonical and microcanonical
strength samplers are `sample_strength_multinomial` and
`sample_strength_stub_matching`.

## Current CLI coverage

| CLI group | Implemented model names |
|---|---|
| `odme fit` | `strength-poisson`, `strength-geometric`, `strength-negative-binomial`, `degree-bernoulli`, `strength-cost-poisson`, `strength-edges-poisson`, `strength-degree-poisson` |
| `odme generate` | ME Poisson strength, cost, edges, degree, custom, multinomial variants |
| `odme filter` | ME Poisson strength, cost, edges, degree, custom |

The Python API exposes more B/W functions than the CLI. A unified Python router
for `{ensemble, family, constraint}` dispatch is still planned; direct typed
entry points are the supported API until then. Radiation and sequential gravity
are intentional gaps after the legacy archive; see
[Archive legacy thesis folders](../decisions/0011-archive-legacy-thesis-folders.md).

## Reference

Sagarra, O. *Statistical mechanics of multi-edge networks*. PhD thesis,
Universitat de Barcelona, 2015. <https://hdl.handle.net/10803/400560>
