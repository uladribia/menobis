---
description: Maximum-entropy model constraints implemented by ODME.
---

# Maximum entropy models

## TL;DR

ODME implements directed multi-edge maximum-entropy models for all thesis
cases: fixed strength, fixed degree, fixed strength + cost, fixed strength +
edges, fixed strength + degree, and custom $p_{ij}$.

## Implemented models

| Model | Constraints | Python API |
|-------|-------------|------------|
| Fixed strength ME | $s^{out}$, $s^{in}$ | `fit_fixed_strength_me` |
| Fixed degree ME | $k^{out}$, $k^{in}$, $T$ | `fit_fixed_degree_binary` |
| Strength + cost ME | $s^{out}$, $s^{in}$, $C$ | `fit_strength_cost_me` |
| Strength + edges ME | $s^{out}$, $s^{in}$, $E$ | `fit_strength_edges_me` |
| Strength + degree ME | $s^{out}$, $s^{in}$, $k^{out}$, $k^{in}$ | `fit_strength_degree_me` |
| Custom $p_{ij}$ ME | $p_{ij}$, $T$ | `sample_custom_pij_events_*` |

All models support partial-constraint fitting (see
[Partial Constraints](partial-constraints.md)).

## Strength-degree invariant

For weighted integer networks:

$$
s_i^{out} \ge k_i^{out}, \quad s_i^{in} \ge k_i^{in}.
$$

## Fixed-strength ME

$$
E[t_{ij}] = x_i \, y_j = \frac{s_i^{out} \, s_j^{in}}{T}.
$$

```python
fit = fit_fixed_strength_me(s_out, s_in)
```

Three sampler variants:

- **Grand-canonical** (Poisson): `sample_poisson(fit.x, fit.y)`
- **Canonical** (multinomial, fixed $T$): `sample_multinomial(fit.x, fit.y, total_events=T)`
- **Microcanonical** (exact $s$, exact $T$): `sample_microcanonical(s_out, s_in)`

## Fixed-degree ME

Binary occupation probability:

$$
p_{ij} = \frac{x_i \, y_j}{1 + x_i \, y_j}.
$$

Expected weighted occupation with $T$ total events:

$$
E[t_{ij}] = \frac{T}{\langle E \rangle} \, p_{ij}, \quad
\langle E \rangle = \sum_{ij} p_{ij}.
$$

```python
fit = fit_fixed_degree_binary(k_out, k_in)
sample = sample_fixed_degree_events_me(fit, total_events=T, seed=42)
```

## Strength + cost ME

The doubly-constrained gravity model in exponential deterrence form:

$$
E[t_{ij}] = x_i \, y_j \, e^{-\gamma \, d_{ij}}.
$$

See [Spatial Costs](spatial-costs.md) for details.

```python
fit = fit_strength_cost_me(s_out, s_in, cost_src, cost_tgt, cost_val, C)
sample = sample_strength_cost_me(fit, cost_src, cost_tgt, cost_val, seed=42)
```

## Strength + total edges ME (Case 3)

$$
E[t_{ij}] = \frac{\lambda \, x_i \, y_j \, e^{x_i y_j}}
{1 + \lambda \left(e^{x_i y_j} - 1\right)}.
$$

```python
fit = fit_strength_edges_me(s_out, s_in, target_edges=E)
sample = sample_strength_edges_me(fit, seed=42)
```

## Strength + degree ME (Case 4)

$$
E[t_{ij}] = \frac{z_i \, w_j \, x_i \, y_j \, e^{x_i y_j}}
{1 + z_i \, w_j \left(e^{x_i y_j} - 1\right)}.
$$

Binary occupation probability:

$$
P(t_{ij} > 0) = \frac{z_i \, w_j \left(e^{x_i y_j} - 1\right)}
{1 + z_i \, w_j \left(e^{x_i y_j} - 1\right)}.
$$

```python
fit = fit_strength_degree_me(s_out, s_in, k_out, k_in)
sample = sample_strength_degree_me(fit, seed=42)
```

## Custom $p_{ij}$ ME (Case 1)

$$
E[t_{ij}] = T \, p_{ij}.
$$

```python
probabilities = normalize_probabilities(source, target, p)
sample = sample_custom_pij_events_multinomial(probabilities, total_events=T, seed=42)
sample = sample_custom_pij_events_poisson(probabilities, total_events=T, seed=42)
```

## Sampler summary

| Constraint | Canonical | Grand-canonical | Microcanonical |
|------------|-----------|-----------------|----------------|
| Strength | `sample_multinomial` | `sample_poisson` | `sample_microcanonical` |
| Strength + cost | — | `sample_strength_cost_me` | — |
| Strength + edges | — | `sample_strength_edges_me` | — |
| Strength + degree | — | `sample_strength_degree_me` | — |
| Degree + events | — | `sample_fixed_degree_events_me` | — |
| Custom $p_{ij}$ | `sample_custom_pij_events_multinomial` | `sample_custom_pij_events_poisson` | — |
