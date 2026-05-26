//! Binary and binomial fitting routines.

use super::mask::PairMask;
use super::support::{
    max_pair_delta, peel_b_strength_saturation, peel_degree_saturation, self_loop_mask,
};
use super::{FitResult, StrengthDegreeFitResult, StrengthEdgesFitResult};

pub fn balance_masked_degree_bernoulli(
    degree_out: &[f64],
    degree_in: &[f64],
    mask: &[bool],
    tolerance: f64,
    max_iterations: usize,
) -> FitResult {
    let n = degree_out.len();
    let k_avg = degree_out.iter().sum::<f64>() / n.max(1) as f64;
    let n_free = (0..n * n).filter(|&idx| !mask[idx]).count() as f64 / n.max(1) as f64;
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
                .filter(|&i| !mask[i * n + j])
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
                .filter(|&j| !mask[i * n + j])
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
                .filter(|&j| !mask[i * n + j])
                .map(|j| binary_probability(x[i], y[j]))
                .sum();
            max_err = max_err.max((pred - degree_out[i]).abs());
        }
        for j in 0..n {
            let pred: f64 = (0..n)
                .filter(|&i| !mask[i * n + j])
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

/// Sparse-mask version of [`balance_masked_degree_bernoulli`].
///
/// Uses `PairMask` to avoid O(N²) dense allocation. Inner loops still iterate
/// all N candidates per node (nonlinear Bernoulli sums cannot use precomputed
/// corrections), but memory usage is O(N+K) instead of O(N²).
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
        let mut result = balance_masked_degree_bernoulli(
            &peeling.excess_out,
            &peeling.excess_in,
            &peeling.mask,
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
    let mask = self_loop_mask(degree_out.len(), self_loops);
    balance_masked_degree_bernoulli(degree_out, degree_in, &mask, tolerance, max_iterations)
}

pub(crate) fn binary_probability(x: f64, y: f64) -> f64 {
    let z = x * y;
    z / (1.0 + z)
}

/// Iterative proportional fitting for binomial(M) fixed-strength constraints.
///
/// Automatically peels strength-saturated nodes (s_i = M*capacity) before
/// solving the residual sub-problem, guaranteeing convergence at the boundary.
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
        let mut result = balance_masked_strength_binomial(
            &peeling.excess_out,
            &peeling.excess_in,
            &peeling.mask,
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
    let mask = self_loop_mask(strength_out.len(), self_loops);
    balance_masked_strength_binomial(
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
/// Uses log-space geometric damping to prevent multiplier explosion
/// in ill-conditioned cases (high saturation, heterogeneous strengths).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn balance_masked_strength_binomial(
    strength_out: &[f64],
    strength_in: &[f64],
    mask: &[bool],
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
    let mut damping = 1.0_f64; // geometric damping factor (1.0 = no damping)

    // Check interval for expensive residual verification (every N iterations)
    let residual_check_interval = 50;

    for iter in 0..max_iterations {
        let mut max_delta = 0.0_f64;
        for j in 0..n {
            if k_in[j] <= 0.0 {
                continue;
            }
            let mut denom = 0.0;
            for i in 0..n {
                if mask[i * n + j] {
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
                if mask[i * n + j] {
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

        // Primary convergence: O(N) multiplier-delta check
        if max_delta < tolerance * 1e-6 {
            return FitResult {
                x,
                y,
                converged: true,
                iterations: iter + 1,
            };
        }

        // Periodic O(N²) residual check for tolerance on constraints
        if (iter + 1) % residual_check_interval == 0 || max_delta < tolerance * 0.01 {
            let mut max_err = 0.0_f64;
            for i in 0..n {
                let mut pred = 0.0;
                for j in 0..n {
                    if mask[i * n + j] {
                        continue;
                    }
                    pred += m * x[i] * y[j] / (1.0 + x[i] * y[j]);
                }
                max_err = max_err.max((pred - strength_out[i]).abs());
            }
            for j in 0..n {
                let mut pred = 0.0;
                for i in 0..n {
                    if mask[i * n + j] {
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
            // Detect stalling and enable damping
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

// ---------------------------------------------------------------------------
// Zero-inflated binomial(M) strength-edges and strength-degree fitting
// ---------------------------------------------------------------------------

#[inline]
fn b_g(q: f64, layers: u32) -> f64 {
    (1.0 + q.max(0.0)).powi(layers as i32) - 1.0
}

#[inline]
fn b_zip_mean(lam: f64, q: f64, layers: u32) -> f64 {
    if lam <= 0.0 || q <= 0.0 || layers == 0 {
        return 0.0;
    }
    // Thesis zero-inflated B formula:
    // E[t_ij] = l_ij q_ij G'_B(q_ij) / (1 + l_ij G_B(q_ij)),
    // G_B(q) = (1 + q)^M - 1, q G'_B(q) = M q (1 + q)^(M-1).
    let one_plus = 1.0 + q;
    let numerator = lam * f64::from(layers) * q * one_plus.powi(layers as i32 - 1);
    let denominator = 1.0 + lam * (one_plus.powi(layers as i32) - 1.0);
    (numerator / denominator).clamp(0.0, f64::from(layers))
}

#[inline]
fn b_zip_occupation(lam: f64, q: f64, layers: u32) -> f64 {
    if lam <= 0.0 || q <= 0.0 || layers == 0 {
        return 0.0;
    }
    let lg = lam * b_g(q, layers);
    (lg / (1.0 + lg)).clamp(0.0, 1.0)
}

fn solve_b_strength_edges_factor(target: f64, other: &[f64], lam: f64, layers: u32) -> f64 {
    if target <= 0.0 || other.iter().all(|&v| v <= 0.0) {
        return 0.0;
    }
    let mut low = 0.0_f64;
    let mut high = 1.0_f64;
    for _ in 0..80 {
        let value: f64 = other
            .iter()
            .map(|&v| b_zip_mean(lam, high * v, layers))
            .sum();
        if value >= target || high >= 1e18 {
            break;
        }
        high *= 2.0;
    }
    for _ in 0..80 {
        let mid = 0.5 * (low + high);
        let value: f64 = other
            .iter()
            .map(|&v| b_zip_mean(lam, mid * v, layers))
            .sum();
        if value < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

#[allow(clippy::too_many_arguments)]
fn balance_strength_edges_binomial_for_lambda(
    strength_out: &[f64],
    strength_in: &[f64],
    lam: f64,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
    x_init: &[f64],
    y_init: &[f64],
) -> (Vec<f64>, Vec<f64>, bool, usize) {
    let n = strength_out.len();
    let mut x = x_init.to_vec();
    let mut y = y_init.to_vec();
    let mut other = vec![0.0; n];
    for iter in 0..max_iterations {
        for j in 0..n {
            for i in 0..n {
                other[i] = if self_loops || i != j { x[i] } else { 0.0 };
            }
            y[j] = solve_b_strength_edges_factor(strength_in[j], &other, lam, layers);
        }
        for i in 0..n {
            for j in 0..n {
                other[j] = if self_loops || i != j { y[j] } else { 0.0 };
            }
            x[i] = solve_b_strength_edges_factor(strength_out[i], &other, lam, layers);
        }
        let err =
            b_strength_edges_max_error(&x, &y, lam, layers, strength_out, strength_in, self_loops)
                .0;
        if err <= tolerance.max(1e-10) {
            return (x, y, true, iter + 1);
        }
    }
    (x, y, false, max_iterations)
}

fn b_strength_edges_max_error(
    x: &[f64],
    y: &[f64],
    lam: f64,
    layers: u32,
    strength_out: &[f64],
    strength_in: &[f64],
    self_loops: bool,
) -> (f64, f64) {
    let n = x.len();
    let mut pred_out = vec![0.0; n];
    let mut pred_in = vec![0.0; n];
    let mut edges = 0.0;
    for i in 0..n {
        for j in 0..n {
            if self_loops || i != j {
                let q = x[i] * y[j];
                let mean = b_zip_mean(lam, q, layers);
                pred_out[i] += mean;
                pred_in[j] += mean;
                edges += b_zip_occupation(lam, q, layers);
            }
        }
    }
    let mut max_err = 0.0_f64;
    for i in 0..n {
        max_err = max_err.max((pred_out[i] - strength_out[i]).abs());
        max_err = max_err.max((pred_in[i] - strength_in[i]).abs());
    }
    (max_err, edges)
}

#[must_use]
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
    let pairs = if self_loops {
        n * n
    } else {
        n * n.saturating_sub(1)
    } as f64;
    if n == 0
        || n != strength_in.len()
        || layers == 0
        || target_edges <= 0.0
        || target_edges > pairs
    {
        return StrengthEdgesFitResult {
            x: vec![0.0; n],
            y: vec![0.0; n],
            lam: 0.0,
            converged: false,
            iterations: 0,
        };
    }
    let total = strength_out.iter().sum::<f64>().max(1.0);
    let scale = total.sqrt().max(1.0);
    let mut cur_x: Vec<f64> = strength_out
        .iter()
        .map(|&s| (s / scale).max(1e-12))
        .collect();
    let mut cur_y: Vec<f64> = strength_in
        .iter()
        .map(|&s| (s / scale).max(1e-12))
        .collect();
    let mut low = 1e-14_f64;
    let mut high = 1.0_f64;
    let mut best = (
        cur_x.clone(),
        cur_y.clone(),
        false,
        0_usize,
        high,
        0.0_f64,
        f64::INFINITY,
    );
    for _ in 0..80 {
        let (x, y, conv, it) = balance_strength_edges_binomial_for_lambda(
            strength_out,
            strength_in,
            high,
            layers,
            self_loops,
            tolerance,
            max_iterations.min(5000),
            &cur_x,
            &cur_y,
        );
        let (err, edges) =
            b_strength_edges_max_error(&x, &y, high, layers, strength_out, strength_in, self_loops);
        cur_x = x.clone();
        cur_y = y.clone();
        best = (x, y, conv, it, high, edges, err);
        if edges >= target_edges || high >= 1e14 {
            break;
        }
        low = high;
        high *= 2.0;
    }
    let mut total_it = 0;
    for _ in 0..70 {
        let mid = 0.5 * (low + high);
        let (x, y, conv, it) = balance_strength_edges_binomial_for_lambda(
            strength_out,
            strength_in,
            mid,
            layers,
            self_loops,
            tolerance,
            max_iterations.min(5000),
            &cur_x,
            &cur_y,
        );
        total_it += it;
        let (err, edges) =
            b_strength_edges_max_error(&x, &y, mid, layers, strength_out, strength_in, self_loops);
        cur_x = x.clone();
        cur_y = y.clone();
        best = (x, y, conv, total_it, mid, edges, err);
        if (edges - target_edges).abs() <= tolerance.max(1e-8) * target_edges.max(1.0)
            && err <= tolerance.max(1e-8) * total.max(1.0)
        {
            break;
        }
        if edges < target_edges {
            low = mid;
        } else {
            high = mid;
        }
    }
    StrengthEdgesFitResult {
        x: best.0,
        y: best.1,
        lam: best.4,
        converged: best.2
            || (best.5 - target_edges).abs() <= tolerance.max(1e-7) * target_edges.max(1.0),
        iterations: best.3,
    }
}

fn solve_b_strength_degree_factor_s(
    target: f64,
    other_q: &[f64],
    other_v: &[f64],
    lam_x: f64,
    layers: u32,
) -> f64 {
    if target <= 0.0 {
        return 0.0;
    }
    let mut low = 0.0_f64;
    let mut high = 1.0_f64;
    for _ in 0..80 {
        let value: f64 = other_q
            .iter()
            .zip(other_v)
            .map(|(&oq, &ov)| b_zip_mean(ov, lam_x * high * oq, layers))
            .sum();
        if value >= target || high >= 1e18 {
            break;
        }
        high *= 2.0;
    }
    for _ in 0..80 {
        let mid = 0.5 * (low + high);
        let value: f64 = other_q
            .iter()
            .zip(other_v)
            .map(|(&oq, &ov)| b_zip_mean(ov, lam_x * mid * oq, layers))
            .sum();
        if value < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

fn solve_b_strength_degree_factor_k(
    target: f64,
    other_q: &[f64],
    other_v: &[f64],
    lam_z: f64,
    layers: u32,
) -> f64 {
    if target <= 0.0 {
        return 0.0;
    }
    let mut low = 0.0_f64;
    let mut high = 1.0_f64;
    for _ in 0..80 {
        let value: f64 = other_q
            .iter()
            .zip(other_v)
            .map(|(&oq, &ov)| b_zip_occupation(lam_z * high * ov, oq, layers))
            .sum();
        if value >= target || high >= 1e18 {
            break;
        }
        high *= 2.0;
    }
    for _ in 0..80 {
        let mid = 0.5 * (low + high);
        let value: f64 = other_q
            .iter()
            .zip(other_v)
            .map(|(&oq, &ov)| b_zip_occupation(lam_z * mid * ov, oq, layers))
            .sum();
        if value < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
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
    let total = strength_out.iter().sum::<f64>().max(1.0);
    let scale = total.sqrt().max(1.0);
    let mut x: Vec<f64> = strength_out
        .iter()
        .map(|&s| (s / scale).max(1e-12))
        .collect();
    let mut y: Vec<f64> = strength_in
        .iter()
        .map(|&s| (s / scale).max(1e-12))
        .collect();
    let k_total = degree_out.iter().sum::<f64>().max(1.0);
    let mut z: Vec<f64> = degree_out
        .iter()
        .map(|&k| (k / k_total).max(1e-12))
        .collect();
    let mut w: Vec<f64> = degree_in
        .iter()
        .map(|&k| (k / k_total).max(1e-12))
        .collect();
    let mut oq = vec![0.0; n];
    let mut ov = vec![0.0; n];
    let mut converged = false;
    let mut iterations = 0;
    for iter in 0..max_iterations {
        for j in 0..n {
            for i in 0..n {
                if self_loops || i != j {
                    oq[i] = x[i];
                    ov[i] = z[i] * w[j];
                } else {
                    oq[i] = 0.0;
                    ov[i] = 0.0;
                }
            }
            y[j] = solve_b_strength_degree_factor_s(strength_in[j], &oq, &ov, 1.0, layers);
        }
        for i in 0..n {
            for j in 0..n {
                if self_loops || i != j {
                    oq[j] = y[j];
                    ov[j] = z[i] * w[j];
                } else {
                    oq[j] = 0.0;
                    ov[j] = 0.0;
                }
            }
            x[i] = solve_b_strength_degree_factor_s(strength_out[i], &oq, &ov, 1.0, layers);
        }
        for j in 0..n {
            for i in 0..n {
                if self_loops || i != j {
                    oq[i] = x[i] * y[j];
                    ov[i] = z[i];
                } else {
                    oq[i] = 0.0;
                    ov[i] = 0.0;
                }
            }
            w[j] = solve_b_strength_degree_factor_k(degree_in[j], &oq, &ov, 1.0, layers);
        }
        for i in 0..n {
            for j in 0..n {
                if self_loops || i != j {
                    oq[j] = x[i] * y[j];
                    ov[j] = w[j];
                } else {
                    oq[j] = 0.0;
                    ov[j] = 0.0;
                }
            }
            z[i] = solve_b_strength_degree_factor_k(degree_out[i], &oq, &ov, 1.0, layers);
        }
        let (s_err, k_err) = b_strength_degree_errors(
            &x,
            &y,
            &z,
            &w,
            layers,
            strength_out,
            strength_in,
            degree_out,
            degree_in,
            self_loops,
        );
        iterations = iter + 1;
        if s_err <= tolerance.max(1e-6) * total && k_err <= tolerance.max(1e-6) * k_total {
            converged = true;
            break;
        }
    }
    StrengthDegreeFitResult {
        x,
        y,
        z,
        w,
        converged,
        iterations,
    }
}

#[allow(clippy::too_many_arguments)]
fn b_strength_degree_errors(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    w: &[f64],
    layers: u32,
    strength_out: &[f64],
    strength_in: &[f64],
    degree_out: &[f64],
    degree_in: &[f64],
    self_loops: bool,
) -> (f64, f64) {
    let n = x.len();
    let mut so = vec![0.0; n];
    let mut si = vec![0.0; n];
    let mut ko = vec![0.0; n];
    let mut ki = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            if self_loops || i != j {
                let q = x[i] * y[j];
                let v = z[i] * w[j];
                let mean = b_zip_mean(v, q, layers);
                let occ = b_zip_occupation(v, q, layers);
                so[i] += mean;
                si[j] += mean;
                ko[i] += occ;
                ki[j] += occ;
            }
        }
    }
    let mut se = 0.0_f64;
    let mut ke = 0.0_f64;
    for i in 0..n {
        se = se
            .max((so[i] - strength_out[i]).abs())
            .max((si[i] - strength_in[i]).abs());
        ke = ke
            .max((ko[i] - degree_out[i]).abs())
            .max((ki[i] - degree_in[i]).abs());
    }
    (se, ke)
}
