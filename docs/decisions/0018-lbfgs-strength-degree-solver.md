# ADR-0018: L-BFGS Solver for Strength-Degree Fitting

## TL;DR

Replace IPFP/damped-balancing with direct NLL minimization via L-BFGS for ME and B strength-degree (zero-inflated) models. O(N) memory, rayon-parallelized, 100% convergence on heterogeneous inputs up to N=1000.

## Status

Accepted. Implemented for ME (Poisson) and B (Binomial). W (Geometric/NegBin) pending.

## Context

The previous ME strength-degree solver used damped log-domain coordinate balancing (IPFP-style). This approach:

- Diverged on highly heterogeneous networks (10× multiplier variation)
- Required hand-tuned damping constants (`DAMPING=0.2`, `MAX_LOG_STEP=1.5`)
- Had no convergence guarantees
- Was numerically unstable near boundary nodes (k≈s or k≈N_max)

The B solver used alternating bisection on 4 blocks of variables, which was even slower.

## Decision

Minimize the **convex dual NLL** of the grand-canonical ensemble directly using L-BFGS.

### Mathematical formulation

For the zero-inflated model with 4N parameters θ = [α_x, α_y, α_v, α_w]:

```
q_ij = exp(α_x[i] + α_y[j])     (weight-generating parameter)
v_ij = exp(α_v[i] + α_w[j])     (zero-inflation multiplier)
Z_ij = 1 + v_ij · G_F(q_ij)     (pair partition function)
```

where G_F is the family-specific positive-support partition factor:
- ME: G_ME(q) = exp(q) - 1
- B(M): G_B(q) = (1+q)^M - 1
- W(M): G_W(q) = (1-q)^(-M) - 1

The NLL objective:

```
f(θ) = Σ_{(i,j) free} ln(Z_ij) - α_x · s_out - α_y · s_in - α_v · k_out - α_w · k_in
```

Gradients are simply predicted minus target:

```
∂f/∂α_x[i] = Σ_j E[t_ij] - s_out[i]
∂f/∂α_y[j] = Σ_i E[t_ij] - s_in[j]
∂f/∂α_v[i] = Σ_j E[Θ(t_ij>0)] - k_out[i]
∂f/∂α_w[j] = Σ_i E[Θ(t_ij>0)] - k_in[j]
```

This objective is convex (log-partition function of exponential family), so L-BFGS converges to the global minimum.

### Architecture

```
┌─────────────────────────────────────────┐
│  1. Regularization (preprocessing)      │
│     - Relative ε boundary detection     │
│     - Mass-preserving redistribution    │
├─────────────────────────────────────────┤
│  2. Evaluator (O(N) memory, parallel)   │
│     - Rayon fold over rows              │
│     - Thread-local column accumulators  │
│     - No N×N dense matrices             │
├─────────────────────────────────────────┤
│  3. L-BFGS optimizer                    │
│     - 10-vector memory                  │
│     - Backtracking Armijo line search   │
│     - Recentering after each step       │
│     - Steepest-descent fallback         │
└─────────────────────────────────────────┘
```

### Key design choices

| Choice | Rationale |
|--------|-----------|
| Log-domain parameters | Unconstrained optimization; no positivity projection needed |
| Relative ε regularization | Avoids distorting small-scale problems (ε=1e-3 × capacity) |
| Recentering after each step | Prevents drift in the gauge freedom (x↔y, v↔w symmetry) |
| Armijo backtracking (c₁=1e-4) | Sufficient decrease without Wolfe conditions; simpler |
| Step clamping (max 5.0 per component) | Prevents overshooting into overflow regions |
| L-BFGS memory = 10 | Standard choice; no measurable benefit from more |
| Family-specific evaluator | Each family has its own `*_pair_statistics` function |
| Shared optimizer skeleton | `lbfgs_direction`, `recenter_theta`, line search are identical |

## Benchmark

Heterogeneous 10× multiplier variation, no self-loops, tol=1e-6, 5 runs:

| Model | N | Mean | Iterations | Converged |
|-------|------|--------|-----------|-----------|
| ME | 100 | 114ms | 537 | 5/5 |
| ME | 500 | 1.76s | 629 | 5/5 |
| ME | 1000 | 7.57s | 650 | 5/5 |
| B(M=3) | 100 | 53ms | 255 | 5/5 |
| B(M=3) | 500 | 0.97s | 358 | 5/5 |
| B(M=3) | 1000 | 3.88s | 372 | 5/5 |

Scaling: O(N² · iters). Iterations are roughly constant across N.

## Consequences

- Old balancing code deleted (no backward compatibility needed per AGENTS.md)
- `PairMask` integration preserved for partial fitting
- Saturated nodes (k≈N_max) handled via post-hoc clamping of z/w multipliers
- The same architecture applies to W once implemented
- Python API unchanged; `fit_strength_degree_poisson` and `fit_strength_degree_binomial` transparently use L-BFGS

## Files

- `crates/menobis-core/src/fitting/me_lbfgs.rs` — ME solver
- `crates/menobis-core/src/fitting/b_lbfgs.rs` — B solver
- `crates/menobis-core/benches/me_strength_degree.rs` — benchmark
