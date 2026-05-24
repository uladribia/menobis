---
description: W ensemble equations, supported constraints, and known solver gaps.
---

# W ensemble

## TL;DR

W is a grand-canonical family only. `M=1` is geometric; `M>1` is negative
binomial. It shares cost, support, and zero-inflation infrastructure with ME and
B, but its pair mean uses the W kernel and must not call ME as a substitute.

## Family kernel

For every candidate pair:

$$
q_{ij}=x_i y_j f_{ij}, \qquad 0<q_{ij}<1.
$$

The non-zero-inflated W expectation is:

$$
\mathbb{E}[t_{ij}] = \frac{M q_{ij}}{1-q_{ij}}.
$$

## Zero-inflated W

With raw binary multiplier $\ell_{ij}$, define:

$$
G_W(q_{ij})=(1-q_{ij})^{-M}-1.
$$

Then:

$$
\mathbb{E}[\Theta(t_{ij}>0)] =
\frac{\ell_{ij}G_W(q_{ij})}{1+\ell_{ij}G_W(q_{ij})},
$$

$$
\mathbb{E}[t_{ij}\mid t_{ij}>0] =
\frac{M q_{ij}(1-q_{ij})^{-M-1}}{(1-q_{ij})^{-M}-1}.
$$

The unconditional mean is occupation times the zero-truncated mean. See
[Model ontology](model-ontology.md) for the shared ME/B/W formula table.

## Constraint status

| Constraint | Geometric API | Negative-binomial API | Status |
|---|---|---|---|
| strength | `fit_strength_geometric` | `fit_strength_negative_binomial` | implemented; known no-self-loop convergence gap |
| strength-cost | `fit_strength_cost_geometric` | `fit_strength_cost_negative_binomial` | implemented; shares cost-provider target |
| degree-events | `fit_degree_events_geometric` | `fit_degree_events_negative_binomial` | implemented via scalar positive-weight solve + Bernoulli occupation |
| strength-edges | `fit_strength_edges_geometric` | `fit_strength_edges_negative_binomial` | implemented with experimental coordinate/root solver |
| strength-degree | `fit_strength_degree_geometric` | `fit_strength_degree_negative_binomial` | implemented with experimental coordinate solver |

Public negative-binomial fit APIs require `layers > 1`; use the geometric name
for `M=1`.

## Solver expectations

| Invariant | Required check |
|---|---|
| W domain | all fitted `q_ij < 1` |
| constraint recovery | strengths, degrees, edges, and cost recover targets within documented tolerance |
| family separation | W tests compare against ME and B on the same feasible input |
| no relabeling | W code may reuse infrastructure, but not ME fitted multipliers as the answer |

## Current known violations

- The W Newton solver does not reliably converge with `self_loops=False` at
  realistic `N >= 50` gravity-style inputs.
- Negative-binomial sampling/filtering APIs should reject `layers <= 1` or route
  to geometric for consistency with the ontology.

See [Ontology conformance audit](../development/ontology-conformance-audit.md)
for the actionable fix list.
