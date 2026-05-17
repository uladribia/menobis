---
description: Multi-edge model constraints implemented by ODME.
---

# Multi-edge models

## TL;DR

ODME implements directed multi-edge models derived from maximum entropy
principles. ME stands for **multi-edge** throughout this project.
The taxonomy is documented in [Thesis Cases](thesis-cases.md).

## Implemented models

| Model | Case | Constraints | Fit API |
|-------|------|-------------|----------|
| Fixed strength | — | $s^{out}$, $s^{in}$ | `fit_strength_poisson` |
| Custom probability | 1 | $p_{ij}$, $T$ | `sample_custom_*` |
| Strength-cost | 2 | $s^{out}$, $s^{in}$, $C$ | `fit_strength_cost_poisson` |
| Strength-edges | 3 | $s^{out}$, $s^{in}$, $E$ | `fit_strength_edges_poisson` |
| Strength-degree | 4 | $s^{out}$, $s^{in}$, $k^{out}$, $k^{in}$ | `fit_strength_degree_poisson` |
| Degree-events | 5 | $k^{out}$, $k^{in}$, $T$ | `fit_degree_bernoulli` |

All models support partial-constraint fitting except custom probability
sampling; see [Partial Constraints](partial-constraints.md).

## Shared notation

| Symbol | Meaning |
|--------|---------|
| $t_{ij}$ | Integer event count from node $i$ to node $j$ |
| $T$ | Total events, $\sum_{ij} t_{ij}$ |
| $E$ | Binary edge count, $\sum_{ij}\Theta(t_{ij})$ |
| $s^{out}, s^{in}$ | Outgoing and incoming strength sequences |
| $k^{out}, k^{in}$ | Outgoing and incoming binary degree sequences |

For integer networks, ODME validates:

$$
s_i^{out} \ge k_i^{out}, \qquad s_i^{in} \ge k_i^{in}.
$$

## Fixed-strength ME

The fitted factors recover the analytical expectation:

$$
\mathbb{E}[t_{ij}] = x_i y_j = \frac{s_i^{out} s_j^{in}}{T}.
$$

```python
fit = fit_strength_poisson(s_out, s_in)
```

Sampler variants:

| Ensemble | Function | Exactly fixed |
|----------|----------|---------------|
| Grand-canonical | `sample_strength_poisson(fit.x, fit.y)` | nothing |
| Canonical | `sample_strength_multinomial(fit.x, fit.y, total_events=T)` | $T$ |
| Stub-matched | `sample_strength_microcanonical(s_out, s_in)` | $s^{out}$, $s^{in}$, $T$ |

## Degree-events ME, thesis case 5

Binary occupation probability:

$$
p_{ij} = \frac{x_i y_j}{1 + x_i y_j},
\qquad
\langle E \rangle = \sum_{ij} p_{ij}.
$$

Positive edges receive weights with common mean $T / \langle E \rangle$:

$$
\mathbb{E}[t_{ij}] = \frac{T}{\langle E \rangle} p_{ij}.
$$

```python
fit = fit_degree_bernoulli(k_out, k_in)
sample = sample_degree_events_poisson(fit, total_events=T, seed=42)
```

## Strength-cost ME, thesis case 2

$$
\mathbb{E}[t_{ij}] = x_i y_j e^{-\gamma d_{ij}}.
$$

```python
fit = fit_strength_cost_poisson(s_out, s_in, cost_src, cost_tgt, cost_val, C)
sample = sample_strength_cost_poisson(fit, cost_src, cost_tgt, cost_val, seed=42)
```

See [Spatial Costs](spatial-costs.md) for constraints and solver notes.

## Strength-edges ME, thesis case 3

Let $u_{ij}=x_i y_j$. Then:

$$
p_{ij}=\frac{\lambda(e^{u_{ij}}-1)}{1+\lambda(e^{u_{ij}}-1)},
\qquad
\mathbb{E}[t_{ij}]=\frac{\lambda u_{ij}e^{u_{ij}}}{1+\lambda(e^{u_{ij}}-1)}.
$$

```python
fit = fit_strength_edges_poisson(s_out, s_in, target_edges=E)
sample = sample_strength_edges_poisson(fit, seed=42)
```

## Strength-degree ME, thesis case 4

Let $u_{ij}=x_i y_j$ and $v_{ij}=z_i w_j$. Then:

$$
p_{ij}=\frac{v_{ij}(e^{u_{ij}}-1)}{1+v_{ij}(e^{u_{ij}}-1)},
\qquad
\mathbb{E}[t_{ij}]=\frac{v_{ij}u_{ij}e^{u_{ij}}}{1+v_{ij}(e^{u_{ij}}-1)}.
$$

```python
fit = fit_strength_degree_poisson(s_out, s_in, k_out, k_in)
sample = sample_strength_degree_poisson(fit, seed=42)
```

## Custom probability, thesis case 1

The supplied probabilities are normalized before sampling:

$$
\mathbb{E}[t_{ij}] = T \frac{p_{ij}}{\sum_{ab} p_{ab}}.
$$

```python
probabilities = normalize_probabilities(source, target, p)
sample = sample_custom_multinomial(probabilities, total_events=T, seed=42)
sample = sample_custom_poisson(probabilities, total_events=T, seed=42)
```
