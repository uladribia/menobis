# ADR-0018: L-BFGS Solver for Zero-Inflated Fitting

## TL;DR

Replace IPFP/damped-balancing/bisection with direct NLL minimization via L-BFGS for ME and B zero-inflated models (strength-degree and strength-edges constraints). O(N) memory, rayon-parallelized, 100% convergence on heterogeneous inputs up to N=1000.

## Status

Accepted. Implemented for all three families (ME, B, W) and both zero-inflated constraint types (strength-degree, strength-edges).

## Context

The previous solvers had different algorithms per constraint type:

- **Strength-degree (ME):** Damped log-domain coordinate balancing. Diverged on heterogeneous networks, required hand-tuned constants, no convergence guarantees.
- **Strength-degree (B):** Alternating bisection on 4 blocks. Even slower than ME.
- **Strength-edges (ME):** Nested bisection over λ with inner IPF per trial. O(N² · inner_iters · bisection_iters).
- **Strength-edges (B):** Same nested bisection, extremely slow for B (31s at N=1000).

All four cases share the same mathematical structure: minimize a convex NLL dual. A single L-BFGS approach replaces all four.

## Decision

Minimize the **convex dual NLL** of the grand-canonical ensemble directly using L-BFGS.

### Mathematical formulation

#### Strength-degree (4N parameters)

θ = [α_x, α_y, α_v, α_w] where q_ij = exp(α_x[i] + α_y[j]) and v_ij = exp(α_v[i] + α_w[j]):

```
f(θ) = Σ_{(i,j)} ln(Z_ij) - α_x·s_out - α_y·s_in - α_v·k_out - α_w·k_in
```

Gradients: ∂f/∂α_x[i] = Σ_j E[t_ij] − s_out[i], etc.

#### Strength-edges (2N+1 parameters)

θ = [α_x, α_y, ln_λ] where q_ij = exp(α_x[i] + α_y[j]) and λ = exp(ln_λ) is a **scalar** (same for all pairs):

```
f(θ) = Σ_{(i,j)} ln(Z_ij) - α_x·s_out - α_y·s_in - ln_λ·E_target
```

Gradients: ∂f/∂ln_λ = Σ_{i,j} E[Θ(t_ij>0)] − E_target.

#### Family-specific partition factors

Both cases use Z_ij = 1 + v_ij · G_F(q_ij) with:

| Family | G_F(q) | E[t_ij \| t>0] |
|--------|--------|----------------|
| ME (Poisson) | exp(q) − 1 | q·exp(q) / (exp(q)−1) |
| B (Binomial, M) | (1+q)^M − 1 | M·q·(1+q)^(M−1) / ((1+q)^M−1) |
| W (NegBin, M) | (1−q)^(−M) − 1 | M·q·(1−q)^(−M−1) / ((1−q)^(−M)−1) |

The NLL is convex (log-partition of exponential family), so L-BFGS converges to the global minimum.

### Architecture

```
┌─────────────────────────────────────────┐
│  1. Regularization (strength-degree)    │
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

## Benchmarks

### Strength-degree (heterogeneous 10× variation, tol=1e-6, 5 runs)

| Model | N | Mean | Iterations | Converged |
|-------|------|--------|-----------|-----------|
| ME | 100 | 114ms | 537 | 5/5 |
| ME | 500 | 1.76s | 629 | 5/5 |
| ME | 1000 | 7.57s | 650 | 5/5 |
| B(M=3) | 100 | 53ms | 255 | 5/5 |
| B(M=3) | 500 | 0.97s | 358 | 5/5 |
| B(M=3) | 1000 | 3.88s | 372 | 5/5 |

### Strength-edges: L-BFGS vs old bisection (release, tol=1e-6)

| Model | N | Bisection | L-BFGS | Speedup |
|-------|------|-----------|--------|---------|
| ME | 100 | 0.070s | 0.007s | **10×** |
| ME | 500 | 1.13s | 0.15s | **8×** |
| ME | 1000 | 4.31s | 0.80s | **5×** |
| B(M=3) | 100 | 0.27s | 0.005s | **56×** |
| B(M=3) | 500 | 7.63s | 0.07s | **102×** |
| B(M=3) | 1000 | 31.9s | 0.48s | **66×** |

Both methods converge at all tested sizes. L-BFGS is the clear winner.

## Consequences

- Old balancing/bisection code deleted for all families (no backward compatibility per AGENTS.md)
- `PairMask` integration preserved for partial fitting
- Saturated nodes (k≈N_max) handled via post-hoc clamping of z/w multipliers
- Python API unchanged; all public fitting functions transparently use the new solvers
- W uses a hybrid approach: Newton for (a,b) strength + bisection for (z,w) degree and scalar λ, because the W feasibility constraint (r>0) prevents joint unconstrained L-BFGS

## Files

| File | Contents |
|------|----------|
| `crates/menobis-core/src/fitting/me_lbfgs.rs` | ME strength-degree + strength-edges L-BFGS |
| `crates/menobis-core/src/fitting/b_lbfgs.rs` | B strength-degree + strength-edges L-BFGS |
| `crates/menobis-core/src/fitting/w_lbfgs.rs` | W strength-edges (bisect λ + Newton inner), W strength-degree (Newton + bisection) |
| `crates/menobis-core/src/fitting/me.rs` | Thin wrappers delegating to L-BFGS |
| `crates/menobis-core/src/fitting/b.rs` | Thin wrappers delegating to L-BFGS |
| `crates/menobis-core/src/fitting/w.rs` | Thin wrappers delegating to Newton/L-BFGS |
| `crates/menobis-core/benches/me_strength_degree.rs` | Benchmark harness |
