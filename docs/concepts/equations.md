---
description: Compact MENoBiS equations mapped to thesis terminology.
---

# Equations

## TL;DR

Grand-canonical MENoBiS models are independent over node pairs. Families share
multipliers but differ in the expected occupation equation.

## Pair parameter

For every allowed ordered pair:

$$q_{ij}=x_i y_j f_{ij}$$

where `x` and `y` are strength multipliers. Without cost, $f_{ij}=1$. With cost,
$f_{ij}=\exp(-\gamma d_{ij})$.

## Non-zero-inflated families

| Family | MENoBiS | Expected occupation | Domain |
|---|---|---|---|
| ME | Poisson | $\mathbb{E}[t_{ij}]=q_{ij}$ | $q>0$ |
| B | Binomial(M) | $\mathbb{E}[t_{ij}]=Mq_{ij}/(1+q_{ij})$ | $q>0$ |
| W | NegBin(M) | $\mathbb{E}[t_{ij}]=Mq_{ij}/(1-q_{ij})$ | $0<q<1$ |

For W, `M=1` is the geometric case.

!!! note "Event nature matters"
    The same strength or degree constraints generate different statistics when
    events are distinguishable, aggregated binary layers, or indistinguishable.

## Zero-inflated constraints

Strength-edges and strength-degree constraints also control binary occupation.
They use a raw binary multiplier $\ell_{ij}$ and a positive-support factor
$G_F(q)$:

| Family | $G_F(q)$ |
|---|---|
| ME | $e^q-1$ |
| B | $(1+q)^M-1$ |
| W | $(1-q)^{-M}-1$ |

The occupation probability is:

$$
\mathbb{E}[\Theta(t_{ij}>0)] =
\frac{\ell_{ij}G_F(q_{ij})}{1+\ell_{ij}G_F(q_{ij})}.
$$

The expected occupation is:

$$
\mathbb{E}[t_{ij}] =
\frac{\ell_{ij}q_{ij}G'_F(q_{ij})}{1+\ell_{ij}G_F(q_{ij})}.
$$

## Constraint map

| MENoBiS constraint | Matched expectation |
|---|---|
| `STRENGTH` | outgoing and incoming strengths |
| `STRENGTH_COST` | strengths plus total cost $\sum_{ij}\mathbb{E}[t_{ij}]d_{ij}$ |
| `STRENGTH_EDGES` | strengths plus total binary edges |
| `STRENGTH_DEGREE` | strengths plus in/out degrees |
| `DEGREE_EVENTS` | in/out degrees plus total events |
| partial variants | parent constraints after subtracting frozen pairs |

## Important rule

ME, B, and W are not interchangeable. B and W must use their own equations and
solvers; they must not call the ME solution and relabel the result.
