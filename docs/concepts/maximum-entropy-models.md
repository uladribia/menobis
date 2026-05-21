---
description: Mathematical overview of ODME fitting, generation, and filtering.
---

# Maximum entropy models

## TL;DR

ODME builds maximum-entropy null models by choosing pair distributions whose
expected sufficient statistics match requested constraints. Fitting finds
Lagrange multipliers; generation samples independent pair distributions;
filtering computes p-values under the fitted null.

## Families

| Family | Pair law | Parameter idea |
|---|---|---|
| ME | Poisson | unbounded integer weights |
| B | Binomial(M) | bounded layer count |
| W geometric | geometric | unbounded weighted edges |
| W negative-binomial | NB(M) | layered W model |

## Constraints

| Constraint | Matched statistic |
|---|---|
| strength | outgoing/incoming expected weights |
| degree-events | binary degrees plus total events |
| strength-edges | strengths plus expected occupied edges |
| strength-degree | strengths plus binary degrees |
| strength-cost | strengths plus expected cost |

## Fitting

Fitting solves Lagrange multiplier equations. For a directed pair `(i,j)`,
node multipliers usually appear as products such as `x_i y_j`. Additional
multipliers encode edge, degree, or cost constraints.

Examples:

| Model | Mean or occupation form |
|---|---|
| Poisson strength | `mu_ij = x_i y_j` |
| Binomial strength | `mu_ij = M x_i y_j / (1 + x_i y_j)` |
| W strength | `mu_ij = M q_ij / (1 - q_ij)` |
| zero-inflated | `pi_ij = v G(q) / (1 + v G(q))` |

The W family uses `q = exp(-r)` and kernels derived from
`G_M(r) = (1 - exp(-r))^{-M} - 1`.

## Solvers

| Solver | Used for |
|---|---|
| analytic | Poisson fixed strength |
| IPF/balancing | ME/B strength and degree equations |
| scalar root + IPF | degree-events |
| coordinate bisection | strength-edges and strength-degree |
| sparse conic Clarabel | W independent strength and strength-cost |

Partial known-weight fitting is the same problem on residual constraints: known
weighted pairs are subtracted from strengths/events/cost and removed from the
free support.

## Generation

After fitting, ODME constructs a provider for each candidate pair and samples
independently unless the model is explicitly multinomial or stub-matching.
Generation streams over candidate pairs and returns a sparse edge table.

## Filtering

Filtering evaluates observed weights under the fitted pair distributions. Upper
p-values mark unexpectedly large weights; lower p-values mark unexpectedly small
weights. Absent-edge filtering uses the model occupation probability.

## Numerical policy

- Use approximate residual checks for floating point fits.
- Expose convergence, iterations, and diagnostics.
- Prefer Rust kernels for all heavy numerical loops.
- Treat legacy code as scientific reference, not compatibility target.
