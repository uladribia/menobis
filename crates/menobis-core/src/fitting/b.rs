//! Binary and binomial fitting routines.

use super::mask::PairMask;
use super::support::{max_pair_delta, peel_b_strength_saturation, peel_degree_saturation};
use super::{FitResult, StrengthDegreeFitResult, StrengthEdgesFitResult};

/// IPF balancing for masked Bernoulli fixed-degree constraints.
///
/// Uses `PairMask` for O(N+K) memory. Inner loops still iterate all N
/// candidates per node (nonlinear Bernoulli sums).
#[must_use]
pub fn balance_sparse_masked_degree_bernoulli(
    degree_out: &[f64],
    degree_in: &[f64],
    mask: &PairMask,
    tolerance: f64,
    max_iterations: usize,
) -> FitResult {
    let n = degree_out.len();
    let k_avg = degree_out.iter().sum::<f64>() / n.max(1) as f64;
    let n_free = mask.n_free() as f64 / n.max(1) as f64;
    let c = if k_avg < n_free {
        (k_avg / (n_free - k_avg).max(0.01)).sqrt()
    } else {
        1.0
    };
    let mut x: Vec<f64> = degree_out
        .iter()
        .map(|&k| {
            if k > 0.0 && k_avg > 0.0 {
                k / k_avg * c
            } else {
                0.0
            }
        })
        .collect();
    let mut y: Vec<f64> = degree_in
        .iter()
        .map(|&k| {
            if k > 0.0 && k_avg > 0.0 {
                k / k_avg * c
            } else {
                0.0
            }
        })
        .collect();

    for iter in 0..max_iterations {
        let old_x = x.clone();
        let old_y = y.clone();

        for j in 0..n {
            if degree_in[j] <= 0.0 {
                y[j] = 0.0;
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&i| !mask.is_masked(i, j))
                .map(|i| {
                    let aux = 1.0 + x[i] * y[j];
                    x[i] / aux
                })
                .sum();
            y[j] = if denom > 0.0 {
                degree_in[j] / denom
            } else {
                0.0
            };
        }
        for i in 0..n {
            if degree_out[i] <= 0.0 {
                x[i] = 0.0;
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&j| !mask.is_masked(i, j))
                .map(|j| {
                    let aux = 1.0 + x[i] * y[j];
                    y[j] / aux
                })
                .sum();
            x[i] = if denom > 0.0 {
                degree_out[i] / denom
            } else {
                0.0
            };
        }

        let delta = max_pair_delta(&x, &old_x, &y, &old_y);
        let mut max_err = 0.0_f64;
        for i in 0..n {
            let pred: f64 = (0..n)
                .filter(|&j| !mask.is_masked(i, j))
                .map(|j| binary_probability(x[i], y[j]))
                .sum();
            max_err = max_err.max((pred - degree_out[i]).abs());
        }
        for j in 0..n {
            let pred: f64 = (0..n)
                .filter(|&i| !mask.is_masked(i, j))
                .map(|i| binary_probability(x[i], y[j]))
                .sum();
            max_err = max_err.max((pred - degree_in[j]).abs());
        }
        if delta < tolerance || max_err < tolerance {
            return FitResult {
                x,
                y,
                converged: true,
                iterations: iter + 1,
            };
        }
    }

    FitResult {
        x,
        y,
        converged: false,
        iterations: max_iterations,
    }
}

/// IPF balancing for directed Bernoulli fixed-degree constraints.
///
/// Automatically peels degree-saturated nodes (k_i = capacity) before solving
/// the residual sub-problem, guaranteeing convergence at the boundary.
///
/// Delegates to [`balance_sparse_masked_degree_bernoulli`] with appropriate mask.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn balance_degree_bernoulli(
    degree_out: &[f64],
    degree_in: &[f64],
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> FitResult {
    let peeling = peel_degree_saturation(degree_out, degree_in, self_loops, 1.0);
    if peeling.has_saturation {
        // Solve the reduced problem on excess degrees with the saturated mask
        let mask = PairMask::from_dense(degree_out.len(), &peeling.mask);
        let mut result = balance_sparse_masked_degree_bernoulli(
            &peeling.excess_out,
            &peeling.excess_in,
            &mask,
            tolerance,
            max_iterations,
        );
        // Saturated nodes get multiplier large enough that p_ij ≈ 1
        let n = degree_out.len();
        let capacity = if self_loops {
            n as f64
        } else {
            (n.saturating_sub(1)) as f64
        };
        for i in 0..n {
            if degree_out[i] >= capacity - 1e-9 {
                result.x[i] = 1e6;
            }
        }
        for j in 0..n {
            if degree_in[j] >= capacity - 1e-9 {
                result.y[j] = 1e6;
            }
        }
        return result;
    }
    let mask = PairMask::from_self_loops(degree_out.len(), self_loops);
    balance_sparse_masked_degree_bernoulli(degree_out, degree_in, &mask, tolerance, max_iterations)
}

pub(crate) fn binary_probability(x: f64, y: f64) -> f64 {
    let z = x * y;
    z / (1.0 + z)
}

/// Iterative proportional fitting for binomial(M) fixed-strength constraints.
///
/// Automatically peels strength-saturated nodes (s_i = M*capacity) before
/// solving the residual sub-problem, guaranteeing convergence at the boundary.
///
/// Delegates to [`balance_sparse_masked_strength_binomial`] with appropriate mask.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn balance_strength_binomial(
    strength_out: &[f64],
    strength_in: &[f64],
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> FitResult {
    let peeling = peel_b_strength_saturation(strength_out, strength_in, layers, self_loops);
    if peeling.has_saturation {
        let mask = PairMask::from_dense(strength_out.len(), &peeling.mask);
        let mut result = balance_sparse_masked_strength_binomial(
            &peeling.excess_out,
            &peeling.excess_in,
            &mask,
            layers,
            tolerance,
            max_iterations,
        );
        // Saturated nodes get very large multipliers so p_ij -> 1, t_ij -> M
        let n = strength_out.len();
        let m = f64::from(layers);
        let capacity = if self_loops {
            n as f64
        } else {
            (n.saturating_sub(1)) as f64
        };
        let max_s = m * capacity;
        for i in 0..n {
            if strength_out[i] >= max_s - 1e-9 {
                result.x[i] = 1e6;
            }
        }
        for j in 0..n {
            if strength_in[j] >= max_s - 1e-9 {
                result.y[j] = 1e6;
            }
        }
        return result;
    }
    let mask = PairMask::from_self_loops(strength_out.len(), self_loops);
    balance_sparse_masked_strength_binomial(
        strength_out,
        strength_in,
        &mask,
        layers,
        tolerance,
        max_iterations,
    )
}

/// Masked binomial(M) IPF for partial-constraint fitting.
///
/// Uses `PairMask` for O(N+K) memory. Uses log-space geometric damping to
/// prevent multiplier explosion in ill-conditioned cases.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn balance_sparse_masked_strength_binomial(
    strength_out: &[f64],
    strength_in: &[f64],
    mask: &PairMask,
    layers: u32,
    tolerance: f64,
    max_iterations: usize,
) -> FitResult {
    let n = strength_out.len();
    let m = f64::from(layers);
    let total: f64 = strength_out.iter().sum();
    let sqrt_t = total.sqrt().max(1.0);
    let mut x: Vec<f64> = strength_out
        .iter()
        .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
        .collect();
    let mut y: Vec<f64> = strength_in
        .iter()
        .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
        .collect();
    let k_in: Vec<f64> = strength_in.iter().map(|&s| s / m).collect();
    let k_out: Vec<f64> = strength_out.iter().map(|&s| s / m).collect();
    let mut prev_err = f64::INFINITY;
    let mut damping = 1.0_f64;
    let residual_check_interval = 50;

    for iter in 0..max_iterations {
        let mut max_delta = 0.0_f64;
        for j in 0..n {
            if k_in[j] <= 0.0 {
                continue;
            }
            let mut denom = 0.0;
            for i in 0..n {
                if mask.is_masked(i, j) {
                    continue;
                }
                denom += x[i] / (1.0 + x[i] * y[j]);
            }
            let new_y = if denom > 0.0 { k_in[j] / denom } else { 0.0 };
            if damping < 1.0 && y[j] > 0.0 && new_y > 0.0 {
                let damped = (y[j].ln() * (1.0 - damping) + new_y.ln() * damping).exp();
                max_delta = max_delta.max((damped - y[j]).abs());
                y[j] = damped;
            } else {
                max_delta = max_delta.max((new_y - y[j]).abs());
                y[j] = new_y;
            }
        }
        for i in 0..n {
            if k_out[i] <= 0.0 {
                continue;
            }
            let mut denom = 0.0;
            for j in 0..n {
                if mask.is_masked(i, j) {
                    continue;
                }
                denom += y[j] / (1.0 + x[i] * y[j]);
            }
            let new_x = if denom > 0.0 { k_out[i] / denom } else { 0.0 };
            if damping < 1.0 && x[i] > 0.0 && new_x > 0.0 {
                let damped = (x[i].ln() * (1.0 - damping) + new_x.ln() * damping).exp();
                max_delta = max_delta.max((damped - x[i]).abs());
                x[i] = damped;
            } else {
                max_delta = max_delta.max((new_x - x[i]).abs());
                x[i] = new_x;
            }
        }

        if max_delta < tolerance * 1e-6 {
            return FitResult {
                x,
                y,
                converged: true,
                iterations: iter + 1,
            };
        }

        if (iter + 1) % residual_check_interval == 0 || max_delta < tolerance * 0.01 {
            let mut max_err = 0.0_f64;
            for i in 0..n {
                let mut pred = 0.0;
                for j in 0..n {
                    if mask.is_masked(i, j) {
                        continue;
                    }
                    pred += m * x[i] * y[j] / (1.0 + x[i] * y[j]);
                }
                max_err = max_err.max((pred - strength_out[i]).abs());
            }
            for j in 0..n {
                let mut pred = 0.0;
                for i in 0..n {
                    if mask.is_masked(i, j) {
                        continue;
                    }
                    pred += m * x[i] * y[j] / (1.0 + x[i] * y[j]);
                }
                max_err = max_err.max((pred - strength_in[j]).abs());
            }
            if max_err < tolerance {
                return FitResult {
                    x,
                    y,
                    converged: true,
                    iterations: iter + 1,
                };
            }
            if max_err >= prev_err * 0.99 && damping >= 1.0 {
                damping = 0.5;
            } else if max_err >= prev_err * 0.999 && damping < 1.0 {
                damping = (damping * 0.8).max(0.1);
            } else if max_err < prev_err * 0.95 && damping < 1.0 {
                damping = (damping * 1.2).min(1.0);
            }
            prev_err = max_err;
        }
    }

    FitResult {
        x,
        y,
        converged: false,
        iterations: max_iterations,
    }
}

/// Fit B (Binomial) strength-edges using L-BFGS optimization.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_strength_edges_binomial(
    strength_out: &[f64],
    strength_in: &[f64],
    target_edges: f64,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> StrengthEdgesFitResult {
    let n = strength_out.len();
    let mask = PairMask::from_self_loops(n, self_loops);
    super::b_lbfgs::fit_strength_edges_binomial_lbfgs(
        strength_out,
        strength_in,
        target_edges,
        layers,
        &mask,
        tolerance,
        max_iterations,
    )
}

#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_strength_degree_binomial(
    strength_out: &[f64],
    strength_in: &[f64],
    degree_out: &[f64],
    degree_in: &[f64],
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> StrengthDegreeFitResult {
    let n = strength_out.len();
    let mask = PairMask::from_self_loops(n, self_loops);
    super::b_lbfgs::fit_strength_degree_binomial_lbfgs(
        strength_out,
        strength_in,
        degree_out,
        degree_in,
        layers,
        &mask,
        tolerance,
        max_iterations,
    )
}
