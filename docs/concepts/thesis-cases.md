---
description: Mapping between thesis model cases and ODME API.
---

# Thesis model cases

## TL;DR

Each ODME model maps to a numbered case from the thesis (ref [1]).

## Case mapping

| Case | Constraint | Expectation | Fit API | Samplers |
|------|------------|-------------|---------|----------|
| 0 | Radiation / seq. gravity | — | — | not yet |
| 1 | Custom $p_{ij}$, total $T$ | $E[t_{ij}] = T \, p_{ij}$ | — | `sample_custom_pij_events_*` |
| 2 | Strength + cost | $E[t_{ij}] = x_i y_j e^{-\gamma d_{ij}}$ | `fit_strength_cost_me` | `sample_strength_cost_me` |
| 3 | Strength + edges $E$ | see below | `fit_strength_edges_me` | `sample_strength_edges_me` |
| 4 | Strength + degree | see below | `fit_strength_degree_me` | `sample_strength_degree_me` |
| 5 | Degree + total $T$ | see below | `fit_fixed_degree_binary` | `sample_fixed_degree_events_me` |
| — | Strength only | $E[t_{ij}] = x_i y_j$ | `fit_fixed_strength_me` | `sample_poisson`, `sample_multinomial`, `sample_poisson_multinomial` |

### Case 3: fixed strength + total binary edges

$$
E[t_{ij}] = \frac{\lambda \, x_i \, y_j \, e^{x_i y_j}}
{1 + \lambda(e^{x_i y_j} - 1)}.
$$

### Case 4: fixed strength + degree

$$
E[t_{ij}] = \frac{z_i \, w_j \, x_i \, y_j \, e^{x_i y_j}}
{1 + z_i \, w_j (e^{x_i y_j} - 1)}.
$$

Binary occupation probability:

$$
P(t_{ij} > 0) = \frac{z_i \, w_j (e^{x_i y_j} - 1)}
{1 + z_i \, w_j (e^{x_i y_j} - 1)}.
$$

### Case 5: fixed degree + total events

$$
p_{ij} = \frac{x_i \, y_j}{1 + x_i \, y_j}, \qquad
E[t_{ij}] = \frac{T}{\langle E \rangle} \, p_{ij}, \qquad
\langle E \rangle = \sum_{ij} p_{ij}.
$$

## CLI mapping

| Case | `odme fit` | `odme generate` |
|------|------------|-----------------|
| 1 | — | `custom-pij` |
| 2 | `strength-cost-me` | `strength-cost-me` |
| 3 | `strength-edges-me` | `strength-edges-me` |
| 4 | `strength-degree-me` | `strength-degree-me` |
| 5 | `degrees` | `degree-events-me` |
| — | `strengths` | `poisson`, `multinomial`, `poisson-multinomial` |

## Reference

[1] Sagarra, O. *Statistical mechanics of multi-edge networks*.
    PhD thesis, Universitat de Barcelona, 2015.
    <https://hdl.handle.net/10803/400560>
