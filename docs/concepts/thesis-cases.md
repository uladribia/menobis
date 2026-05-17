---
description: Mapping between numbered thesis cases, ODME model names, APIs, and equations.
---

# Thesis cases

## TL;DR

Use **thesis case** only for numbered cases from Sagarra [1]. Use **ODME
model** for implementation names such as `fixed-strength ME`. ODME implements
selected numbered thesis cases plus the unnumbered fixed-strength baseline.

## Naming rules

| Term | Meaning |
|------|---------|
| Thesis case | Numbered model family in the thesis and legacy docs |
| ODME model | Public implementation/API name |
| ME | Multi-edge directed weighted network with integer event counts |
| $T$ | Total events: $T = \sum_{ij} t_{ij}$ |
| $E$ | Binary edges: $E = \sum_{ij} \Theta(t_{ij})$ |

## Case mapping

| Thesis case | ODME model | Constraints | Main equation | Fit API | Sampler |
|-------------|------------|-------------|---------------|---------|---------|
| 0 | Radiation / sequential gravity | model-specific | not implemented | — | — |
| 1 | Custom probability ME | $p_{ij}$, $T$ | $\mathbb{E}[t_{ij}] = T p_{ij}$ | — | `sample_custom_pij_events_*` |
| 2 | Strength-cost ME | $s^{out}$, $s^{in}$, $C$ | $\mathbb{E}[t_{ij}] = x_i y_j e^{-\gamma d_{ij}}$ | `fit_strength_cost_me` | `sample_strength_cost_me` |
| 3 | Strength-edges ME | $s^{out}$, $s^{in}$, $E$ | ZIP, see below | `fit_strength_edges_me` | `sample_strength_edges_me` |
| 4 | Strength-degree ME | $s^{out}$, $s^{in}$, $k^{out}$, $k^{in}$ | ZIP, see below | `fit_strength_degree_me` | `sample_strength_degree_me` |
| 5 | Degree-events ME | $k^{out}$, $k^{in}$, $T$ | binary occupation + weighted events | `fit_fixed_degree_binary` | `sample_fixed_degree_events_me` |
| — | Fixed-strength ME | $s^{out}$, $s^{in}$ | $\mathbb{E}[t_{ij}] = x_i y_j$ | `fit_fixed_strength_me` | `sample_poisson`, `sample_multinomial`, `sample_microcanonical` |

## Case 3: strength-edges ME

Let $u_{ij} = x_i y_j$. The binary occupation probability is:

$$
p_{ij} = P(t_{ij} > 0) =
\frac{\lambda \left(e^{u_{ij}} - 1\right)}
{1 + \lambda \left(e^{u_{ij}} - 1\right)}.
$$

Conditional on occupation, ODME samples a zero-truncated Poisson with rate
$u_{ij}$. Therefore:

$$
\mathbb{E}[t_{ij}] =
\frac{\lambda u_{ij} e^{u_{ij}}}
{1 + \lambda \left(e^{u_{ij}} - 1\right)},
\qquad
E = \sum_{ij} p_{ij}.
$$

## Case 4: strength-degree ME

Let $u_{ij} = x_i y_j$ and $v_{ij} = z_i w_j$. The binary occupation
probability is:

$$
p_{ij} = P(t_{ij} > 0) =
\frac{v_{ij} \left(e^{u_{ij}} - 1\right)}
{1 + v_{ij} \left(e^{u_{ij}} - 1\right)}.
$$

The expected weight is:

$$
\mathbb{E}[t_{ij}] =
\frac{v_{ij} u_{ij} e^{u_{ij}}}
{1 + v_{ij} \left(e^{u_{ij}} - 1\right)}.
$$

## Case 5: degree-events ME

ODME first fits binary degree multipliers:

$$
p_{ij} = \frac{x_i y_j}{1 + x_i y_j},
\qquad
\langle E \rangle = \sum_{ij} p_{ij}.
$$

Positive edges receive zero-truncated Poisson weights with common mean
$T / \langle E \rangle$, so:

$$
\mathbb{E}[t_{ij}] = \frac{T}{\langle E \rangle} p_{ij}.
$$

## CLI mapping

| Thesis case | `odme fit` | `odme generate` |
|-------------|------------|-----------------|
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
