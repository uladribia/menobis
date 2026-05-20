---
description: Weighted W ensemble equations, constraints, and solver status.
---

# W ensemble

## TL;DR

W models use geometric (`M=1`) or negative-binomial (`M>1`) integer weights.
ODME currently fits W fixed strengths and strength-cost with Clarabel conic
models, fits degree-events with scalar root finding plus Bernoulli IPF, and fits
strength-edges with an experimental monotone coordinate/root solver.

## Pair distributions

Let `P` be the allowed directed pair support after self-loop filtering. For
geometric `M=1`; for negative binomial `M>1`. Define:

$$
q_{ij}=e^{-r_{ij}}, \qquad 0 \le q_{ij}<1.
$$

Independent W pairs have expected weight:

$$
\mu_{ij}=\frac{M q_{ij}}{1-q_{ij}}.
$$

Zero-inflated W pairs use occupation multiplier `v_ij` and:

$$
G_M(r)=(1-e^{-r})^{-M}-1,
$$

$$
Z_{ij}=1+v_{ij}G_M(r_{ij}),
$$

$$
\pi_{ij}=\frac{v_{ij}G_M(r_{ij})}{Z_{ij}},
$$

$$
\mu_{ij}=\frac{v_{ij}M q_{ij}(1-q_{ij})^{-M-1}}{Z_{ij}}.
$$

## Implemented fitting constraints

| Constraint | Geometric API | Negative-binomial API | Solver | Max N |
|------------|---------------|-----------------------|--------|------:|
| strengths | `fit_strength_geometric` | `fit_strength_negative_binomial` | Clarabel conic (sparse) | ~200 |
| strengths + cost | `fit_strength_cost_geometric` | `fit_strength_cost_negative_binomial` | Clarabel conic (sparse) | ~200 |
| degrees + events | `fit_degree_events_geometric` | `fit_degree_events_negative_binomial` | scalar `q` + Bernoulli IPF | 1000+ |
| strengths + edges | `fit_strength_edges_geometric` | `fit_strength_edges_negative_binomial` | monotone coordinate | ~200 |
| strengths + degrees | `fit_strength_degree_geometric` | `fit_strength_degree_negative_binomial` | monotone coordinate | ~200 |

`layers=1` is rejected by public negative-binomial fit APIs; use the geometric
spelling for `M=1`.

## Fixed strengths

The inverse/log variables are:

$$
r_{ij}=a_i+b_j, \qquad x_i=e^{-a_i}, \quad y_j=e^{-b_j}.
$$

ODME solves the convex objective:

$$
F=\sum_i s_i^{out}a_i+\sum_j s_j^{in}b_j
-M\sum_{(i,j)\in P}\log(1-e^{-r_{ij}}).
$$

Gauge constraint:

$$
\sum_i a_i - \sum_j b_j = 0.
$$

## Strengths + cost

For costs `d_ij`, missing costs are treated as zero:

$$
r_{ij}=a_i+b_j+\gamma d_{ij}.
$$

ODME solves:

$$
F=\sum_i s_i^{out}a_i+\sum_j s_j^{in}b_j+\gamma C
-M\sum_{(i,j)\in P}\log(1-e^{-r_{ij}}).
$$

The fitted result is a `StrengthCostFit` with `family`, optional `layers`, and
nested conic diagnostics.

## Degrees + total events

Let:

$$
E=\sum_i k_i^{out}=\sum_j k_j^{in}, \qquad T \ge E.
$$

ODME solves one scalar `q` from the positive-weight conditional mean:

$$
\frac{M q}{(1-q)(1-(1-q)^M)}=\frac{T}{E}.
$$

It then fits occupation probabilities with the Bernoulli fixed-degree IPF.

## Strengths + total edges

This zero-inflated model uses `v_ij=lambda` and `r_ij=a_i+b_j`:

$$
\sum_j \mu_{ij}=s_i^{out}, \qquad
\sum_i \mu_{ij}=s_j^{in}, \qquad
\sum_{(i,j)\in P}\pi_{ij}=E.
$$

The current implementation is experimental. It uses an outer bracketed solve for
`lambda`; for fixed `lambda`, each row/column coordinate is solved by scalar
bisection because its constraint equation is monotone in that coordinate.

This is not the naïve multiplicative W IPF update and is not used for the
fixed-strength-only model. It has passed Pareto `N=10` workflow tests and an
out-of-test `N=100` stress check, but it still needs convergence/failure-mode
coverage or a Clarabel conic fallback before it is considered final.

## Diagnostics

W fit wrappers return the same constraint-oriented result types as ME/B fits.
Common fields include `family`, optional `layers`, `converged`, `iterations`,
and `diagnostics`. W conic/root diagnostics are nested under
`diagnostics.conic` and include `max_q`, margin, lifted variables, cones, linear
constraints, and sparse nonzeros.

## Tolerance and iteration defaults

Default `tolerance` and `max_iterations` differ by solver family because
the solvers have different convergence characteristics:

| Family | Default tolerance | Default max_iterations | Rationale |
|--------|------------------:|----------------------:|---|
| Poisson (ME) IPF | 1e-8 to 1e-10 | 10000–50000 | IPF converges linearly; needs many cheap iterations |
| Binomial (B) IPF | 1e-8 to 1e-10 | 10000–50000 | Same structure as ME |
| Geometric (W) conic | 1e-8 | 200–1000 | Clarabel interior-point; few expensive iterations |
| Negative binomial (W) conic | 1e-8 | 200–1000 | Same as geometric |
| W monotone coordinate | 1e-7 to 1e-8 | 500–1000 | Each iteration is O(N²) bisection sweeps |
| Degree-events (all) | 1e-8 to 1e-10 | 10000–50000 | Scalar root + Bernoulli IPF |

The user-facing meaning of `tolerance` is consistent across families:
"maximum absolute change in multipliers between iterations" for IPF/coordinate
solvers, or "solver feasibility/gap tolerance" for conic solvers. In both cases
a smaller tolerance means tighter constraint recovery.

When comparing residuals across families, use `diagnostics.max_strength_residual`
rather than `tolerance` directly, because the relationship between multiplier
tolerance and constraint residual depends on the model's nonlinearity.
