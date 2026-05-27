//! Multi-edge (ME/Poisson-family) fitting routines.

use super::b::binary_probability;
use super::mask::PairMask;
use super::support::{coord_distance, max_abs_delta, max_pair_delta};
use super::{FitResult, StrengthCostFitResult, StrengthDegreeFitResult, StrengthEdgesFitResult};

/// Fit exact grand-canonical ME fixed-strength-and-edge-count zero-inflated constraints.
///
/// Uses monotone coordinate bisection (same approach as W strength-edges)
/// with Poisson pair mean: E[t_ij] = lam * u / (exp(-u) + lam*(1-exp(-u))).
#[must_use]
pub fn balance_strength_edges_poisson(
    strength_out: &[f64],
    strength_in: &[f64],
    target_edges: f64,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> StrengthEdgesFitResult {
    let n = strength_out.len();
    let n2 = if self_loops {
        (n * n) as f64
    } else {
        (n * (n - 1).max(1)) as f64
    };
    let total: f64 = strength_out.iter().sum();
    if n == 0 || target_edges <= 0.0 || target_edges >= n2 || target_edges > total {
        return StrengthEdgesFitResult {
            x: vec![0.0; n],
            y: vec![0.0; n],
            lam: 0.0,
            converged: false,
            iterations: 0,
        };
    }

    let scale = total.sqrt().max(1.0);
    let mut cur_x: Vec<f64> = strength_out
        .iter()
        .map(|&s| (s / scale).max(1e-12))
        .collect();
    let mut cur_y: Vec<f64> = strength_in
        .iter()
        .map(|&s| (s / scale).max(1e-12))
        .collect();

    let lam_init = target_edges / (n2 - target_edges).max(0.01);
    let mut low = 1e-12_f64;
    let mut high = lam_init.max(1.0);

    for _ in 0..40 {
        let (x, y, _, _) = balance_me_edges_for_lambda(
            strength_out,
            strength_in,
            high,
            self_loops,
            tolerance,
            max_iterations,
            &cur_x,
            &cur_y,
        );
        cur_x = x;
        cur_y = y;
        let edges = expected_edges_me(&cur_x, &cur_y, high, self_loops);
        if edges >= target_edges || high > 1e30 {
            break;
        }
        low = high;
        high *= 2.0;
    }

    let mut total_iterations = 0;
    let mut best_lam = high;
    for _ in 0..60 {
        let mid = 0.5 * (low + high);
        let (x, y, _, iters) = balance_me_edges_for_lambda(
            strength_out,
            strength_in,
            mid,
            self_loops,
            tolerance,
            max_iterations,
            &cur_x,
            &cur_y,
        );
        cur_x = x;
        cur_y = y;
        total_iterations += iters;
        let edges = expected_edges_me(&cur_x, &cur_y, mid, self_loops);
        if (edges - target_edges).abs() < tolerance.max(1e-10) {
            best_lam = mid;
            break;
        }
        if edges < target_edges {
            low = mid;
        } else {
            high = mid;
        }
        best_lam = mid;
    }

    StrengthEdgesFitResult {
        x: cur_x,
        y: cur_y,
        lam: best_lam,
        converged: true,
        iterations: total_iterations,
    }
}

fn me_edges_pair_mean(xi: f64, yj: f64, lam: f64) -> f64 {
    let u = xi * yj;
    if u <= 0.0 {
        return 0.0;
    }
    let e_neg_u = (-u).exp();
    let den = e_neg_u + lam * (1.0 - e_neg_u);
    if den <= 0.0 {
        return 0.0;
    }
    lam * u / den
}

fn expected_edges_me(x: &[f64], y: &[f64], lam: f64, self_loops: bool) -> f64 {
    let mut total = 0.0;
    for (i, &xi) in x.iter().enumerate() {
        for (j, &yj) in y.iter().enumerate() {
            if !self_loops && i == j {
                continue;
            }
            let u = xi * yj;
            if u <= 0.0 {
                continue;
            }
            let e_neg_u = (-u).exp();
            let den = e_neg_u + lam * (1.0 - e_neg_u);
            if den > 0.0 {
                total += lam * (1.0 - e_neg_u) / den;
            }
        }
    }
    total
}

fn solve_me_edges_factor(target: f64, other: &[f64], lam: f64) -> f64 {
    if target <= 0.0 {
        return 0.0;
    }
    let mut low = 0.0_f64;
    let mut high = 1e6_f64;
    for _ in 0..60 {
        let mid = 0.5 * (low + high);
        let value: f64 = other.iter().map(|&v| me_edges_pair_mean(mid, v, lam)).sum();
        if value < target {
            low = mid;
        } else {
            high = mid;
        }
        if high - low < 1e-14 * high.max(1.0) {
            break;
        }
    }
    0.5 * (low + high)
}

#[allow(clippy::too_many_arguments)]
fn balance_me_edges_for_lambda(
    strength_out: &[f64],
    strength_in: &[f64],
    lam: f64,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
    x_init: &[f64],
    y_init: &[f64],
) -> (Vec<f64>, Vec<f64>, bool, usize) {
    let n = strength_out.len();
    let mut x = x_init.to_vec();
    let mut y = y_init.to_vec();
    let mut others = vec![0.0_f64; n];

    for iter in 0..max_iterations {
        let old_x = x.clone();
        let old_y = y.clone();
        for j in 0..n {
            for i in 0..n {
                others[i] = if self_loops || i != j { x[i] } else { 0.0 };
            }
            y[j] = solve_me_edges_factor(strength_in[j], &others, lam);
        }
        for i in 0..n {
            for j in 0..n {
                others[j] = if self_loops || i != j { y[j] } else { 0.0 };
            }
            x[i] = solve_me_edges_factor(strength_out[i], &others, lam);
        }
        let delta = x
            .iter()
            .zip(old_x.iter())
            .chain(y.iter().zip(old_y.iter()))
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        if delta < tolerance {
            return (x, y, true, iter + 1);
        }
    }
    (x, y, false, max_iterations)
}

/// Fit exact grand-canonical ME fixed-strength-degree zero-inflated constraints.
///
/// Uses L-BFGS optimization of the NLL dual objective for the Zero-Inflated
/// Poisson grand-canonical ensemble. This approach directly minimizes the
/// convex dual, providing better numerical stability and scalability than
/// the previous damped log-domain balancing for heterogeneous networks.
///
/// Pair statistics:
/// E[t_ij] = v_ij * q_ij * exp(q_ij) / (1 + v_ij*(exp(q_ij)-1))
/// pi_ij = v_ij * (exp(q_ij)-1) / (1 + v_ij*(exp(q_ij)-1))
/// where q_ij = x_i*y_j and v_ij = z_i*w_j.
#[must_use]
pub fn balance_strength_degree_poisson(
    strength_out: &[f64],
    strength_in: &[f64],
    degree_out: &[f64],
    degree_in: &[f64],
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> StrengthDegreeFitResult {
    let n = strength_out.len();
    let capacity = if self_loops {
        n as f64
    } else {
        n.saturating_sub(1) as f64
    };
    let saturated_out: Vec<bool> = degree_out
        .iter()
        .map(|&k| k >= capacity - tolerance.max(1e-9))
        .collect();
    let saturated_in: Vec<bool> = degree_in
        .iter()
        .map(|&k| k >= capacity - tolerance.max(1e-9))
        .collect();
    let has_saturated = saturated_out.iter().chain(saturated_in.iter()).any(|&v| v);

    let mask = PairMask::from_self_loops(n, self_loops);
    let mut result = super::me_lbfgs::fit_strength_degree_poisson_lbfgs(
        strength_out,
        strength_in,
        degree_out,
        degree_in,
        &mask,
        tolerance,
        max_iterations,
    );
    if has_saturated {
        for (i, &is_saturated) in saturated_out.iter().enumerate() {
            if is_saturated {
                result.z[i] = 1e30;
            }
        }
        for (j, &is_saturated) in saturated_in.iter().enumerate() {
            if is_saturated {
                result.w[j] = 1e30;
            }
        }
        if result
            .x
            .iter()
            .chain(result.y.iter())
            .chain(result.z.iter())
            .chain(result.w.iter())
            .all(|v| v.is_finite())
        {
            result.converged = true;
        }
    }
    result
}

/// ME strength-degree occupation: v*(exp(q)-1) / (1 + v*(exp(q)-1)).
#[cfg(test)]
fn me_sd_occupation(q: f64, v: f64) -> f64 {
    me_sd_pair_statistics_from_values(q, v).0
}

/// Sparse-mask L-BFGS solver for ME strength-degree.
///
/// Uses L-BFGS optimization with `PairMask` to handle frozen pairs.
/// Memory is O(N) per rayon thread.
#[must_use]
pub fn balance_sparse_masked_strength_degree_poisson(
    strength_out: &[f64],
    strength_in: &[f64],
    degree_out: &[f64],
    degree_in: &[f64],
    mask: &PairMask,
    tolerance: f64,
    max_iterations: usize,
) -> StrengthDegreeFitResult {
    super::me_lbfgs::fit_strength_degree_poisson_lbfgs(
        strength_out,
        strength_in,
        degree_out,
        degree_in,
        mask,
        tolerance,
        max_iterations,
    )
}

#[cfg(test)]
fn me_sd_pair_statistics_from_values(q: f64, v: f64) -> (f64, f64) {
    if q <= 0.0 || v <= 0.0 {
        return (0.0, 0.0);
    }
    let exp_q = q.exp();
    let exp_q_m1 = q.exp_m1();
    if exp_q_m1 <= 0.0 {
        return (0.0, 0.0);
    }
    let v_g = v * exp_q_m1;
    let z = 1.0 + v_g;
    let occupation = v_g / z;
    let expected_weight = v * q * exp_q / z;
    (occupation, expected_weight)
}

/// ME strength-degree expected weight:
/// E[t_ij] = v * q * exp(q) / (1 + v * (exp(q) - 1))
/// where q = x_i * y_j, v = z_i * w_j
#[cfg(test)]
#[inline]
fn me_sd_expected_weight(q: f64, v: f64) -> f64 {
    me_sd_pair_statistics_from_values(q, v).1
}

/// Alternating coordinate fitting for directed binary fixed-degree models.
///
/// Solves: k_out_i = sum_j p_ij and k_in_j = sum_i p_ij with
/// p_ij = x_i * y_j / (1 + x_i * y_j).
pub fn balance_weighted_factors(
    excess_out: &[f64],
    excess_in: &[f64],
    degree_x: &[f64],
    degree_y: &[f64],
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> FitResult {
    let n = excess_out.len();
    let total: f64 = excess_out.iter().sum();
    let sqrt_t = total.sqrt().max(1.0);
    let mut a: Vec<f64> = excess_out
        .iter()
        .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
        .collect();
    let mut b: Vec<f64> = excess_in
        .iter()
        .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
        .collect();

    for iter in 0..max_iterations {
        let mut b_new = vec![0.0; n];
        for j in 0..n {
            if excess_in[j] == 0.0 {
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&i| self_loops || i != j)
                .map(|i| binary_probability(degree_x[i], degree_y[j]) * a[i])
                .sum();
            b_new[j] = if denom > 0.0 {
                excess_in[j] / denom
            } else {
                0.0
            };
        }

        let mut a_new = vec![0.0; n];
        for i in 0..n {
            if excess_out[i] == 0.0 {
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&j| self_loops || i != j)
                .map(|j| binary_probability(degree_x[i], degree_y[j]) * b_new[j])
                .sum();
            a_new[i] = if denom > 0.0 {
                excess_out[i] / denom
            } else {
                0.0
            };
        }

        let da = max_abs_delta(&a_new, &a);
        let db = max_abs_delta(&b_new, &b);

        a = a_new;
        b = b_new;

        if da < tolerance && db < tolerance {
            return FitResult {
                x: a,
                y: b,
                converged: true,
                iterations: iter + 1,
            };
        }
    }

    FitResult {
        x: a,
        y: b,
        converged: false,
        iterations: max_iterations,
    }
}

/// IPF balancing for fixed-strength ME with a sparse pair mask.
///
/// Equivalent to [`balance_masked_strength_poisson`] but uses O(N+K) memory
/// instead of O(N²) for the mask, and O(K_i) per-row/column instead of O(N)
/// for sum corrections.
#[must_use]
pub fn balance_sparse_masked_strength_poisson(
    strength_out: &[f64],
    strength_in: &[f64],
    mask: &PairMask,
    tolerance: f64,
    max_iterations: usize,
) -> FitResult {
    let n = strength_out.len();
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

    for iter in 0..max_iterations {
        let old_x = x.clone();
        let old_y = y.clone();

        let sum_x: f64 = x.iter().sum();
        for j in 0..n {
            if strength_in[j] <= 0.0 {
                continue;
            }
            let denom = mask.free_col_sum(j, &x, sum_x);
            y[j] = if denom > 0.0 {
                strength_in[j] / denom
            } else {
                0.0
            };
        }

        let sum_y: f64 = y.iter().sum();
        for i in 0..n {
            if strength_out[i] <= 0.0 {
                continue;
            }
            let denom = mask.free_row_sum(i, &y, sum_y);
            x[i] = if denom > 0.0 {
                strength_out[i] / denom
            } else {
                0.0
            };
        }

        let delta = max_pair_delta(&x, &old_x, &y, &old_y);
        if delta < tolerance {
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

/// Iterative proportional fitting for ME fixed-strength without self-loops.
///
/// Delegates to [`balance_sparse_masked_strength_poisson`] with a self-loop-only mask.
#[must_use]
pub fn balance_strength_poisson(
    s_out: &[f64],
    s_in: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> FitResult {
    let mask = PairMask::from_self_loops(s_out.len(), false);
    balance_sparse_masked_strength_poisson(s_out, s_in, &mask, tolerance, max_iterations)
}

/// Fit fixed-strength Poisson ME model (both self-loops and no-self-loops).
///
/// With self-loops the solution is analytic: `x = s_out / sqrt(T)`,
/// `y = s_in / sqrt(T)`. Without self-loops uses iterative balancing.
#[must_use]
pub fn fit_strength_poisson(
    s_out: &[f64],
    s_in: &[f64],
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> FitResult {
    let total: f64 = s_out.iter().sum();
    if total <= 0.0 {
        return FitResult {
            x: vec![0.0; s_out.len()],
            y: vec![0.0; s_in.len()],
            converged: true,
            iterations: 0,
        };
    }
    if self_loops {
        let sqrt_t = total.sqrt();
        let x: Vec<f64> = s_out.iter().map(|&s| s / sqrt_t).collect();
        let y: Vec<f64> = s_in.iter().map(|&s| s / sqrt_t).collect();
        FitResult {
            x,
            y,
            converged: true,
            iterations: 0,
        }
    } else {
        balance_strength_poisson(s_out, s_in, tolerance, max_iterations)
    }
}

// ---------------------------------------------------------------------------
// ME strength-cost fitting
// ---------------------------------------------------------------------------

/// Strength-cost solver options.
pub struct CostFitOptions {
    pub self_loops: bool,
    pub tolerance: f64,
    pub max_iterations: usize,
}

#[allow(clippy::too_many_arguments)]
fn balance_xy_cost_coordinates(
    strength_out: &[f64],
    strength_in: &[f64],
    coord_x: &[f64],
    coord_y: &[f64],
    gamma: f64,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
    x_init: Option<&[f64]>,
    y_init: Option<&[f64]>,
) -> FitResult {
    let n = strength_out.len();
    let total: f64 = strength_out.iter().sum();
    let sqrt_t = total.sqrt().max(1.0);
    let mut x: Vec<f64> = x_init.map_or_else(
        || {
            strength_out
                .iter()
                .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
                .collect()
        },
        <[f64]>::to_vec,
    );
    let mut y: Vec<f64> = y_init.map_or_else(
        || {
            strength_in
                .iter()
                .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
                .collect()
        },
        <[f64]>::to_vec,
    );

    // Tiered strategy: cache f_ij = exp(-gamma*d_ij) when memory fits within
    // ~256 MB (N < ~5700 for f64). Otherwise recompute per pair.
    let use_cache = n * n * 8 <= 256 * 1024 * 1024;
    let f_cache: Vec<f64> = if use_cache {
        let mut cache = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                cache[i * n + j] = (-gamma * coord_distance(coord_x, coord_y, i, j)).exp();
            }
        }
        if !self_loops {
            for i in 0..n {
                cache[i * n + i] = 0.0;
            }
        }
        cache
    } else {
        Vec::new()
    };

    for iter in 0..max_iterations {
        let old_x = x.clone();
        let old_y = y.clone();
        for j in 0..n {
            if strength_in[j] <= 0.0 {
                y[j] = 0.0;
                continue;
            }
            let denom: f64 = if use_cache {
                (0..n).map(|i| x[i] * f_cache[i * n + j]).sum()
            } else {
                (0..n)
                    .filter(|&i| self_loops || i != j)
                    .map(|i| x[i] * (-gamma * coord_distance(coord_x, coord_y, i, j)).exp())
                    .sum()
            };
            y[j] = if denom > 0.0 {
                strength_in[j] / denom
            } else {
                0.0
            };
        }
        for i in 0..n {
            if strength_out[i] <= 0.0 {
                x[i] = 0.0;
                continue;
            }
            let denom: f64 = if use_cache {
                (0..n).map(|j| y[j] * f_cache[i * n + j]).sum()
            } else {
                (0..n)
                    .filter(|&j| self_loops || i != j)
                    .map(|j| y[j] * (-gamma * coord_distance(coord_x, coord_y, i, j)).exp())
                    .sum()
            };
            x[i] = if denom > 0.0 {
                strength_out[i] / denom
            } else {
                0.0
            };
        }
        if max_pair_delta(&x, &old_x, &y, &old_y) < tolerance {
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

#[allow(clippy::needless_range_loop)]
fn expected_cost_coordinates(
    x: &[f64],
    y: &[f64],
    coord_x: &[f64],
    coord_y: &[f64],
    gamma: f64,
    self_loops: bool,
) -> f64 {
    let n = x.len();
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            if !self_loops && i == j {
                continue;
            }
            let d = coord_distance(coord_x, coord_y, i, j);
            total += x[i] * y[j] * d * (-gamma * d).exp();
        }
    }
    total
}

/// Fit strength-cost using on-the-fly Euclidean distances from projected XY coordinates.
#[must_use]
pub fn fit_strength_cost_poisson_coordinates(
    strength_out: &[f64],
    strength_in: &[f64],
    coord_x: &[f64],
    coord_y: &[f64],
    target_cost: f64,
    opts: &CostFitOptions,
) -> StrengthCostFitResult {
    let solve_at = |gamma: f64, x_init: Option<&[f64]>, y_init: Option<&[f64]>| {
        let fit = balance_xy_cost_coordinates(
            strength_out,
            strength_in,
            coord_x,
            coord_y,
            gamma,
            opts.self_loops,
            opts.tolerance,
            opts.max_iterations,
            x_init,
            y_init,
        );
        let delta =
            expected_cost_coordinates(&fit.x, &fit.y, coord_x, coord_y, gamma, opts.self_loops)
                - target_cost;
        (fit, delta)
    };

    let (fit_zero, delta_zero) = solve_at(0.0, None, None);
    if delta_zero.abs() <= opts.tolerance {
        return StrengthCostFitResult {
            x: fit_zero.x,
            y: fit_zero.y,
            gamma: 0.0,
            converged: true,
            iterations: 1,
        };
    }
    let mut low = 0.0_f64;
    let mut high = 0.0_f64;
    let low_fit = fit_zero.clone();
    let mut high_fit = fit_zero.clone();
    let low_delta = delta_zero;
    let mut high_delta = delta_zero;
    let mut step = 1.0_f64;
    if delta_zero > 0.0 {
        for _ in 0..64 {
            high = step;
            let (fit, delta) = solve_at(high, Some(&high_fit.x), Some(&high_fit.y));
            high_fit = fit;
            high_delta = delta;
            if high_delta <= 0.0 {
                break;
            }
            step *= 2.0;
        }
    } else {
        return StrengthCostFitResult {
            x: fit_zero.x,
            y: fit_zero.y,
            gamma: 0.0,
            converged: delta_zero.abs() <= opts.tolerance,
            iterations: 1,
        };
    }
    if !(low_delta >= 0.0 && high_delta <= 0.0) {
        let (bf, bg) = if low_delta.abs() < high_delta.abs() {
            (low_fit, low)
        } else {
            (high_fit, high)
        };
        return StrengthCostFitResult {
            x: bf.x,
            y: bf.y,
            gamma: bg,
            converged: false,
            iterations: opts.max_iterations,
        };
    }
    let mut best_fit = if low_delta.abs() < high_delta.abs() {
        low_fit
    } else {
        high_fit
    };
    let mut best_gamma = if low_delta.abs() < high_delta.abs() {
        low
    } else {
        high
    };
    let mut best_delta = low_delta.abs().min(high_delta.abs());
    for iter in 0..opts.max_iterations {
        let mid = 0.5 * (low + high);
        let (fit, delta) = solve_at(mid, Some(&best_fit.x), Some(&best_fit.y));
        if delta.abs() < best_delta {
            best_delta = delta.abs();
            best_gamma = mid;
            best_fit = fit.clone();
        }
        if delta.abs() <= opts.tolerance {
            return StrengthCostFitResult {
                x: fit.x,
                y: fit.y,
                gamma: mid,
                converged: true,
                iterations: iter + 1,
            };
        }
        if delta > 0.0 {
            low = mid;
        } else {
            high = mid;
        }
        if (high - low) < 1e-15 {
            break;
        }
    }
    StrengthCostFitResult {
        x: best_fit.x,
        y: best_fit.y,
        gamma: best_gamma,
        converged: best_delta <= opts.tolerance,
        iterations: opts.max_iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        balance_sparse_masked_strength_degree_poisson, balance_sparse_masked_strength_poisson,
        me_sd_expected_weight, me_sd_occupation, me_sd_pair_statistics_from_values,
    };
    use crate::fitting::mask::PairMask;

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn me_strength_degree_pair_statistics_match_hand_computed_values() {
        // Thesis ME zero-inflated formula:
        // G(q)=exp(q)-1, p=vG(q)/(1+vG(q)), E[t]=v q exp(q)/(1+vG(q)).
        let q = 0.4_f64;
        let v = 0.7_f64;
        let exp_q = q.exp();
        let expected_occupation = v * (exp_q - 1.0) / (1.0 + v * (exp_q - 1.0));
        let expected_weight = v * q * exp_q / (1.0 + v * (exp_q - 1.0));

        let (occupation, weight) = me_sd_pair_statistics_from_values(q, v);

        assert!((occupation - expected_occupation).abs() < 1e-14);
        assert!((weight - expected_weight).abs() < 1e-14);
        assert!((me_sd_occupation(q, v) - expected_occupation).abs() < 1e-14);
        assert!((me_sd_expected_weight(q, v) - expected_weight).abs() < 1e-14);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn sparse_masked_strength_degree_recovers_regression_constraints() {
        let x = [0.25_f64, 0.125, 0.125];
        let y = [0.125_f64, 0.125, 0.125];
        let z = [0.125_f64, 0.125, 0.125];
        let w = [0.125_f64, 0.25, 0.25];
        let n = 3;
        let mask = PairMask::from_self_loops(n, true);
        let mut s_out = vec![0.0; n];
        let mut s_in = vec![0.0; n];
        let mut k_out = vec![0.0; n];
        let mut k_in = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                let q = x[i] * y[j];
                let v = z[i] * w[j];
                let occupation = me_sd_occupation(q, v);
                let weight = me_sd_expected_weight(q, v);
                k_out[i] += occupation;
                k_in[j] += occupation;
                s_out[i] += weight;
                s_in[j] += weight;
            }
        }

        let fit = balance_sparse_masked_strength_degree_poisson(
            &s_out, &s_in, &k_out, &k_in, &mask, 1e-8, 50000,
        );

        assert!(fit.converged);
        for i in 0..n {
            let mut row_s = 0.0;
            let mut row_k = 0.0;
            for j in 0..n {
                row_s += me_sd_expected_weight(fit.x[i] * fit.y[j], fit.z[i] * fit.w[j]);
                row_k += me_sd_occupation(fit.x[i] * fit.y[j], fit.z[i] * fit.w[j]);
            }
            assert!((row_s - s_out[i]).abs() < 1e-6);
            assert!((row_k - k_out[i]).abs() < 1e-6);
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn sparse_pair_mask_is_respected_by_strength_degree_solver() {
        let x = [0.3_f64, 0.2, 0.4];
        let y = [0.25_f64, 0.35, 0.2];
        let z = [0.4_f64, 0.3, 0.5];
        let w = [0.45_f64, 0.25, 0.35];
        let n = 3;
        let known_src = [0_u64];
        let known_tgt = [1_u64];
        let mask = PairMask::new(n, true, &known_src, &known_tgt);
        let mut s_out = vec![0.0; n];
        let mut s_in = vec![0.0; n];
        let mut k_out = vec![0.0; n];
        let mut k_in = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                if mask.is_masked(i, j) {
                    continue;
                }
                let q = x[i] * y[j];
                let v = z[i] * w[j];
                let occupation = me_sd_occupation(q, v);
                let weight = me_sd_expected_weight(q, v);
                k_out[i] += occupation;
                k_in[j] += occupation;
                s_out[i] += weight;
                s_in[j] += weight;
            }
        }

        let fit = balance_sparse_masked_strength_degree_poisson(
            &s_out, &s_in, &k_out, &k_in, &mask, 1e-8, 50000,
        );

        assert!(fit.converged);
        let masked_expected = me_sd_expected_weight(fit.x[0] * fit.y[1], fit.z[0] * fit.w[1]);
        assert!(masked_expected.is_finite());
        for i in 0..n {
            let row_s: f64 = (0..n)
                .filter(|&j| !mask.is_masked(i, j))
                .map(|j| me_sd_expected_weight(fit.x[i] * fit.y[j], fit.z[i] * fit.w[j]))
                .sum();
            let row_k: f64 = (0..n)
                .filter(|&j| !mask.is_masked(i, j))
                .map(|j| me_sd_occupation(fit.x[i] * fit.y[j], fit.z[i] * fit.w[j]))
                .sum();
            assert!((row_s - s_out[i]).abs() < 1e-6);
            assert!((row_k - k_out[i]).abs() < 1e-6);
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn sparse_masked_strength_recovers_constraints() {
        let s_out = vec![10.0, 20.0, 30.0, 15.0, 25.0];
        let s_in = vec![18.0, 12.0, 22.0, 28.0, 20.0];
        let known_src = vec![0u64, 2, 3];
        let known_tgt = vec![1u64, 4, 0];
        let n = 5;

        let mask = PairMask::new(n, false, &known_src, &known_tgt);
        let result = balance_sparse_masked_strength_poisson(&s_out, &s_in, &mask, 1e-12, 50000);

        assert!(result.converged);
        // Verify the solver converges and produces positive multipliers
        for i in 0..n {
            assert!(result.x[i] >= 0.0);
            assert!(result.y[i] >= 0.0);
        }
        // Verify constraint recovery (row sums over free pairs)
        for i in 0..n {
            let row_sum: f64 = (0..n)
                .filter(|&j| !mask.is_masked(i, j))
                .map(|j| result.x[i] * result.y[j])
                .sum();
            assert!(
                (row_sum - s_out[i]).abs() < 1e-6,
                "s_out[{i}]: expected {}, got {row_sum}",
                s_out[i]
            );
        }
        for j in 0..n {
            let col_sum: f64 = (0..n)
                .filter(|&i| !mask.is_masked(i, j))
                .map(|i| result.x[i] * result.y[j])
                .sum();
            assert!(
                (col_sum - s_in[j]).abs() < 1e-6,
                "s_in[{j}]: expected {}, got {col_sum}",
                s_in[j]
            );
        }
    }
}

#[cfg(test)]
mod unification_tests {
    use super::*;

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn balance_strength_poisson_recovers_constraints() {
        // Verify that balance_strength_poisson (now delegating to sparse)
        // correctly recovers the input strength sequences.
        let s_out = vec![10.0, 20.0, 30.0, 15.0, 25.0];
        let s_in = vec![18.0, 12.0, 22.0, 28.0, 20.0];
        let n = 5;

        let result = balance_strength_poisson(&s_out, &s_in, 1e-12, 50000);
        assert!(result.converged);

        // Row sums (no self-loops): sum_{j!=i} x[i]*y[j] == s_out[i]
        for i in 0..n {
            let row_sum: f64 = (0..n)
                .filter(|&j| j != i)
                .map(|j| result.x[i] * result.y[j])
                .sum();
            assert!(
                (row_sum - s_out[i]).abs() < 1e-6,
                "s_out[{i}]: expected {}, got {row_sum}",
                s_out[i]
            );
        }
        for j in 0..n {
            let col_sum: f64 = (0..n)
                .filter(|&i| i != j)
                .map(|i| result.x[i] * result.y[j])
                .sum();
            assert!(
                (col_sum - s_in[j]).abs() < 1e-6,
                "s_in[{j}]: expected {}, got {col_sum}",
                s_in[j]
            );
        }
    }
}
