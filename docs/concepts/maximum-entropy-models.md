---
description: Maximum-entropy model constraints implemented by ODME.
---

# Maximum entropy models

## TL;DR

ODME implements thesis-aligned directed multi-edge models: fixed strength,
fixed degree, fixed strength + total edges, fixed strength + degree, and custom
`p_ij` generation.

## Implemented models

| Model | Constraint | Equation | Python API |
|-------|------------|----------|------------|
| Fixed-strength ME | `s_out`, `s_in` | `E[t_ij] = x_i y_j` | `fit_fixed_strength_me` |
| Fixed-degree ME | `k_out`, `k_in`, `T` | `E[t_ij] = T p_ij / <E>` | `sample_fixed_degree_events_me` |
| Strength + edges ME | `s_out`, `s_in`, `E` | thesis Case 3 | `fit_strength_edges_me` |
| Strength + degree ME | `s_out`, `s_in`, `k_out`, `k_in` | thesis Case 4 | `fit_strength_degree_me` |
| Custom `p_ij` ME | `p_ij`, `T` | `E[t_ij] = T p_ij` | `sample_custom_pij_events_*` |

## Invariant

Weighted integer networks require `s_i >= k_i` for every node and direction.
ODME rejects fractional weights and validates this before coupled fitting.

## Fixed-strength ME

With self loops:

\[
E[t_{ij}] = \frac{s_i^{out} s_j^{in}}{T} = x_i y_j.
\]

```python
fit = fit_fixed_strength_me(s_out, s_in)
```

## Fixed-degree ME

The binary occupation probability is:

\[
p_{ij} = \frac{x_i y_j}{1 + x_i y_j}.
\]

For total events `T`, the expected weighted occupation is:

\[
E[t_{ij}] = \frac{T}{\langle E \rangle}p_{ij}, \quad
\langle E \rangle = \sum_{ij} p_{ij}.
\]

```python
fit = fit_fixed_degree_binary(k_out, k_in)
sample = sample_fixed_degree_events_me(fit, total_events=10_000, seed=42)
```

## Fixed strength + total edges ME

Thesis Case 3 uses:

\[
E[t_{ij}] =
\frac{\lambda x_i y_j e^{x_i y_j}}
{1 + \lambda(e^{x_i y_j}-1)}.
\]

```python
fit = fit_strength_edges_me(s_out, s_in, target_edges=500.0)
```

## Fixed strength + degree ME

Thesis Case 4 uses:

\[
E[t_{ij}] =
\frac{z_i w_j x_i y_j e^{x_i y_j}}
{1 + z_i w_j(e^{x_i y_j}-1)}.
\]

The binary occupation probability is:

\[
P(t_{ij}>0) =
\frac{z_i w_j(e^{x_i y_j}-1)}
{1 + z_i w_j(e^{x_i y_j}-1)}.
\]

```python
fit = fit_strength_degree_me(s_out, s_in, k_out, k_in)
sample = sample_strength_degree_me(fit, seed=42)
```

## Custom `p_ij` ME

For thesis generator Case 1:

\[
E[t_{ij}] = T p_{ij}.
\]

```python
probabilities = normalize_probabilities(source, target, p)
sample = sample_custom_pij_events_multinomial(probabilities, total_events=T, seed=42)
sample = sample_custom_pij_events_poisson(probabilities, total_events=T, seed=42)
```

## Sampler variants

| Constraint | Canonical | Grand-canonical | Alternative GC |
|------------|-----------|-----------------|----------------|
| Fixed strength ME | `sample_multinomial` | `sample_poisson` | `sample_poisson_multinomial` |
| Strength + edges ME | — | `sample_strength_edges_me` | — |
| Strength + degree ME | — | `sample_strength_degree_me` | — |
| Fixed degree ME | — | `sample_fixed_degree_events_me` | — |
