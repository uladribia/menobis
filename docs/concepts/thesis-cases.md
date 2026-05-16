---
description: Mapping between thesis model cases and ODME API.
---

# Thesis model cases

## TL;DR

Each ODME model maps to a numbered case from the thesis (ref [1]).

## Case mapping

| Thesis case | Constraint | Expectation | Fit API | Samplers |
|---|---|---|---|---|
| 0 | Radiation / seq. gravity | — | — | not yet implemented |
| 1 | Custom `p_ij`, total `T` | `E[t_ij] = T p_ij` | — | `sample_custom_pij_events_poisson`, `sample_custom_pij_events_multinomial` |
| 2 | Strength + cost (`γ`) | `E[t_ij] = x_i y_j e^{-γ d_ij}` | `fit_gravity_me` | `sample_gravity_me` |
| 3 | Strength + total edges `E` | `E[t_ij] = λ x_i y_j e^{x_i y_j} / (1 + λ(e^{x_i y_j}-1))` | `fit_strength_edges_me` | `sample_strength_edges_me` |
| 4 | Strength + degree | `E[t_ij] = z_i w_j x_i y_j e^{x_i y_j} / (1 + z_i w_j(e^{x_i y_j}-1))` | `fit_strength_degree_me` | `sample_strength_degree_me` |
| 5 | Degree + total `T` | `p_ij = x_i y_j/(1+x_i y_j)`, `E[t_ij] = (T/<E>) p_ij` | `fit_fixed_degree_binary` | `sample_fixed_degree_events_me` |
| — | Strength only (ME) | `E[t_ij] = x_i y_j` | `fit_fixed_strength_me` | `sample_poisson`, `sample_multinomial`, `sample_poisson_multinomial` |

## CLI mapping

| Thesis case | `odme fit` command | `odme generate` command |
|---|---|---|
| 1 | — | `custom-pij` |
| 2 | `gravity-me` | `gravity-me` |
| 3 | `strength-edges-me` | `strength-edges-me` |
| 4 | `strength-degree-me` | `strength-degree-me` |
| 5 | `degrees` | `degree-events-me` |
| — (strength) | `strengths` | `poisson`, `multinomial`, `poisson-multinomial` |

## Reference

[1] Sagarra, O. *Statistical mechanics of multi-edge networks*.
    PhD thesis, Universitat de Barcelona, 2015.
    <https://hdl.handle.net/10803/400560>
