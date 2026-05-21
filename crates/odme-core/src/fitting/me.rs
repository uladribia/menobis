//! Multi-edge (ME/Poisson-family) fitting routines.

use super::b::binary_probability;
use super::support::{max_abs_delta, max_pair_delta};
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
        if edges >= target_edges || high > 1e12 {
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
/// Uses monotone coordinate bisection with ME pair statistics:
/// E[t_ij] = v_ij * u / (exp(-u) + v_ij*(1-exp(-u)))
/// pi_ij = v_ij * (1-exp(-u)) / (exp(-u) + v_ij*(1-exp(-u)))
/// where u = x_i*y_j and v_ij = z_i*w_j.
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
    let total = strength_out.iter().sum::<f64>().max(1.0);
    let scale = total.sqrt();
    let mut x: Vec<f64> = strength_out
        .iter()
        .map(|&s| (s / scale).max(1e-12))
        .collect();
    let mut y: Vec<f64> = strength_in
        .iter()
        .map(|&s| (s / scale).max(1e-12))
        .collect();
    let k_total = degree_out.iter().sum::<f64>().max(1.0);
    let k_scale = (k_total / n.max(1) as f64).sqrt().max(0.1);
    let mut z: Vec<f64> = degree_out
        .iter()
        .map(|&k| (k / k_total * k_scale).max(1e-12))
        .collect();
    let mut w: Vec<f64> = degree_in
        .iter()
        .map(|&k| (k / k_total * k_scale).max(1e-12))
        .collect();
    let mut others_q = vec![0.0_f64; n];
    let mut others_v = vec![0.0_f64; n];

    for iter in 0..max_iterations {
        let old_x = x.clone();
        let old_y = y.clone();
        let old_z = z.clone();
        let old_w = w.clone();

        // Update y (strength_in)
        for j in 0..n {
            for i in 0..n {
                if self_loops || i != j {
                    others_q[i] = x[i];
                    others_v[i] = z[i] * w[j];
                } else {
                    others_q[i] = 0.0;
                    others_v[i] = 0.0;
                }
            }
            y[j] = solve_me_sd_factor_s(strength_in[j], &others_q, &others_v);
        }
        // Update x (strength_out)
        for i in 0..n {
            for j in 0..n {
                if self_loops || i != j {
                    others_q[j] = y[j];
                    others_v[j] = z[i] * w[j];
                } else {
                    others_q[j] = 0.0;
                    others_v[j] = 0.0;
                }
            }
            x[i] = solve_me_sd_factor_s(strength_out[i], &others_q, &others_v);
        }
        // Update w (degree_in)
        for j in 0..n {
            for i in 0..n {
                if self_loops || i != j {
                    others_q[i] = x[i] * y[j];
                    others_v[i] = z[i];
                } else {
                    others_q[i] = 0.0;
                    others_v[i] = 0.0;
                }
            }
            w[j] = solve_me_sd_factor_k(degree_in[j], &others_q, &others_v);
        }
        // Update z (degree_out)
        for i in 0..n {
            for j in 0..n {
                if self_loops || i != j {
                    others_q[j] = x[i] * y[j];
                    others_v[j] = w[j];
                } else {
                    others_q[j] = 0.0;
                    others_v[j] = 0.0;
                }
            }
            z[i] = solve_me_sd_factor_k(degree_out[i], &others_q, &others_v);
        }

        let delta = x
            .iter()
            .zip(old_x.iter())
            .chain(y.iter().zip(old_y.iter()))
            .chain(z.iter().zip(old_z.iter()))
            .chain(w.iter().zip(old_w.iter()))
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        if delta < tolerance {
            return StrengthDegreeFitResult {
                x,
                y,
                z,
                w,
                converged: true,
                iterations: iter + 1,
            };
        }
    }
    StrengthDegreeFitResult {
        x,
        y,
        z,
        w,
        converged: false,
        iterations: max_iterations,
    }
}

/// ME strength-degree pair mean: v*u / (exp(-u) + v*(1-exp(-u)))
fn me_sd_pair_mean(xi: f64, oq: f64, ov: f64) -> f64 {
    let u = xi * oq;
    if u <= 0.0 {
        return 0.0;
    }
    let v = ov;
    let e_neg_u = (-u).exp();
    let den = e_neg_u + v * (1.0 - e_neg_u);
    if den <= 0.0 {
        return 0.0;
    }
    v * u / den
}

/// ME strength-degree occupation: v*(1-exp(-u)) / (exp(-u) + v*(1-exp(-u)))
fn me_sd_occupation(u: f64, v: f64) -> f64 {
    if u <= 0.0 || v <= 0.0 {
        return 0.0;
    }
    let e_neg_u = (-u).exp();
    let den = e_neg_u + v * (1.0 - e_neg_u);
    if den <= 0.0 {
        return 0.0;
    }
    v * (1.0 - e_neg_u) / den
}

fn solve_me_sd_factor_s(target: f64, others_q: &[f64], others_v: &[f64]) -> f64 {
    if target <= 0.0 {
        return 0.0;
    }
    let mut low = 0.0_f64;
    let mut high = 1e6_f64;
    for _ in 0..60 {
        let mid = 0.5 * (low + high);
        let value: f64 = others_q
            .iter()
            .zip(others_v.iter())
            .map(|(&oq, &ov)| me_sd_pair_mean(mid, oq, ov))
            .sum();
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

fn solve_me_sd_factor_k(target: f64, others_q: &[f64], others_v: &[f64]) -> f64 {
    if target <= 0.0 {
        return 0.0;
    }
    let mut low = 0.0_f64;
    let mut high = 1e6_f64;
    for _ in 0..60 {
        let mid = 0.5 * (low + high);
        let value: f64 = others_q
            .iter()
            .zip(others_v.iter())
            .map(|(&oq, &ov)| me_sd_occupation(oq, mid * ov))
            .sum();
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

/// Masked monotone coordinate solver for ME strength-degree.
///
/// Same approach as the non-masked version but skips pairs where mask[i*n+j] is true.
#[must_use]
pub fn balance_masked_strength_degree_poisson(
    strength_out: &[f64],
    strength_in: &[f64],
    degree_out: &[f64],
    degree_in: &[f64],
    mask: &[bool],
    tolerance: f64,
    max_iterations: usize,
) -> StrengthDegreeFitResult {
    let n = strength_out.len();
    let total = strength_out.iter().sum::<f64>().max(1.0);
    let scale = total.sqrt();
    let mut x: Vec<f64> = strength_out
        .iter()
        .map(|&s| (s / scale).max(1e-12))
        .collect();
    let mut y: Vec<f64> = strength_in
        .iter()
        .map(|&s| (s / scale).max(1e-12))
        .collect();
    let k_total = degree_out.iter().sum::<f64>().max(1.0);
    let k_scale = (k_total / n.max(1) as f64).sqrt().max(0.1);
    let mut z: Vec<f64> = degree_out
        .iter()
        .map(|&k| (k / k_total * k_scale).max(1e-12))
        .collect();
    let mut w: Vec<f64> = degree_in
        .iter()
        .map(|&k| (k / k_total * k_scale).max(1e-12))
        .collect();
    let mut others_q = vec![0.0_f64; n];
    let mut others_v = vec![0.0_f64; n];

    for iter in 0..max_iterations {
        let old_x = x.clone();
        let old_y = y.clone();
        let old_z = z.clone();
        let old_w = w.clone();

        for j in 0..n {
            for i in 0..n {
                if !mask[i * n + j] {
                    others_q[i] = x[i];
                    others_v[i] = z[i] * w[j];
                } else {
                    others_q[i] = 0.0;
                    others_v[i] = 0.0;
                }
            }
            y[j] = solve_me_sd_factor_s(strength_in[j], &others_q, &others_v);
        }
        for i in 0..n {
            for j in 0..n {
                if !mask[i * n + j] {
                    others_q[j] = y[j];
                    others_v[j] = z[i] * w[j];
                } else {
                    others_q[j] = 0.0;
                    others_v[j] = 0.0;
                }
            }
            x[i] = solve_me_sd_factor_s(strength_out[i], &others_q, &others_v);
        }
        for j in 0..n {
            for i in 0..n {
                if !mask[i * n + j] {
                    others_q[i] = x[i] * y[j];
                    others_v[i] = z[i];
                } else {
                    others_q[i] = 0.0;
                    others_v[i] = 0.0;
                }
            }
            w[j] = solve_me_sd_factor_k(degree_in[j], &others_q, &others_v);
        }
        for i in 0..n {
            for j in 0..n {
                if !mask[i * n + j] {
                    others_q[j] = x[i] * y[j];
                    others_v[j] = w[j];
                } else {
                    others_q[j] = 0.0;
                    others_v[j] = 0.0;
                }
            }
            z[i] = solve_me_sd_factor_k(degree_out[i], &others_q, &others_v);
        }

        let delta = x
            .iter()
            .zip(old_x.iter())
            .chain(y.iter().zip(old_y.iter()))
            .chain(z.iter().zip(old_z.iter()))
            .chain(w.iter().zip(old_w.iter()))
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        if delta < tolerance {
            return StrengthDegreeFitResult {
                x,
                y,
                z,
                w,
                converged: true,
                iterations: iter + 1,
            };
        }
    }
    StrengthDegreeFitResult {
        x,
        y,
        z,
        w,
        converged: false,
        iterations: max_iterations,
    }
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

/// IPF balancing for fixed-strength ME with a pair mask.
///
/// Pairs where `mask[i * n + j]` is true are skipped in summations.
/// This supports partial-constraint fitting where some p_ij are known.
#[must_use]
pub fn balance_masked_strength_poisson(
    strength_out: &[f64],
    strength_in: &[f64],
    mask: &[bool],
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

        for j in 0..n {
            if strength_in[j] <= 0.0 {
                continue;
            }
            let denom: f64 = (0..n).filter(|&i| !mask[i * n + j]).map(|i| x[i]).sum();
            y[j] = if denom > 0.0 {
                strength_in[j] / denom
            } else {
                0.0
            };
        }
        for i in 0..n {
            if strength_out[i] <= 0.0 {
                continue;
            }
            let denom: f64 = (0..n).filter(|&j| !mask[i * n + j]).map(|j| y[j]).sum();
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
/// Solves: s_out_i = sum_{j != i} x_i * y_j  and  s_in_j = sum_{i != j} x_i * y_j
#[must_use]
pub fn balance_strength_poisson(
    s_out: &[f64],
    s_in: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> FitResult {
    let n = s_out.len();
    let total: f64 = s_out.iter().sum();
    let sqrt_t = total.sqrt();

    let mut x: Vec<f64> = s_out
        .iter()
        .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
        .collect();
    let mut y: Vec<f64> = s_in
        .iter()
        .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
        .collect();

    for iter in 0..max_iterations {
        let sum_x: f64 = x.iter().sum();
        let mut y_new = vec![0.0; n];
        for j in 0..n {
            if s_in[j] == 0.0 {
                continue;
            }
            let denom = sum_x - x[j];
            y_new[j] = if denom > 0.0 { s_in[j] / denom } else { 0.0 };
        }

        let sum_y: f64 = y_new.iter().sum();
        let mut x_new = vec![0.0; n];
        for i in 0..n {
            if s_out[i] == 0.0 {
                continue;
            }
            let denom = sum_y - y_new[i];
            x_new[i] = if denom > 0.0 { s_out[i] / denom } else { 0.0 };
        }

        let dx = max_abs_delta(&x_new, &x);
        let dy = max_abs_delta(&y_new, &y);

        x = x_new;
        y = y_new;

        if dx < tolerance && dy < tolerance {
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

/// Sparse cost matrix for strength-cost models.
pub struct CostMatrix<'a> {
    pub sources: &'a [usize],
    pub targets: &'a [usize],
    pub values: &'a [f64],
}

/// Balance x,y for a fixed gamma via IPF, with optional warm start.
/// IPF balancing for ME strength-cost with sparse cost entries.
///
/// Computes f_ij = exp(-gamma * d_ij) on the fly from cost entries.
/// No dense N*N matrix is allocated. Uses sum_x/sum_y fast path
/// with sparse corrections for cost entries that differ from f=1.
#[allow(clippy::too_many_arguments)]
fn balance_xy_cost_sparse(
    strength_out: &[f64],
    strength_in: &[f64],
    costs: &CostMatrix<'_>,
    n: usize,
    gamma: f64,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
    x_init: Option<&[f64]>,
    y_init: Option<&[f64]>,
) -> FitResult {
    let total: f64 = strength_out.iter().sum();
    let sqrt_t = total.sqrt().max(1.0);
    let mut x: Vec<f64> = match x_init {
        Some(xi) => xi.to_vec(),
        None => strength_out
            .iter()
            .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
            .collect(),
    };
    let mut y: Vec<f64> = match y_init {
        Some(yi) => yi.to_vec(),
        None => strength_in
            .iter()
            .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
            .collect(),
    };

    // Build per-column and per-row f_ij lookup (O(K) memory, not O(N^2))
    let mut col_src: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut row_tgt: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (idx, (&src, &tgt)) in costs.sources.iter().zip(costs.targets.iter()).enumerate() {
        if !self_loops && src == tgt {
            continue;
        }
        if src < n && tgt < n {
            let f_ij = (-gamma * costs.values[idx]).exp();
            col_src[tgt].push((src, f_ij));
            row_tgt[src].push((tgt, f_ij));
        }
    }
    // Track which pairs have explicit costs for the sparse correction path
    let k = costs.sources.len();
    let n_pairs = if self_loops { n * n } else { n * (n - 1) };
    let complete = k >= n_pairs;

    for iter in 0..max_iterations {
        let old_x = x.clone();
        let old_y = y.clone();

        if complete {
            // All pairs covered: sum over explicit entries only
            for j in 0..n {
                if strength_in[j] <= 0.0 {
                    y[j] = 0.0;
                    continue;
                }
                let denom: f64 = col_src[j].iter().map(|&(i, f)| x[i] * f).sum();
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
                let denom: f64 = row_tgt[i].iter().map(|&(j, f)| y[j] * f).sum();
                x[i] = if denom > 0.0 {
                    strength_out[i] / denom
                } else {
                    0.0
                };
            }
        } else {
            // Sparse: base = sum (f=1 for missing pairs) + corrections from entries
            let sum_x: f64 = x.iter().sum();
            for j in 0..n {
                if strength_in[j] <= 0.0 {
                    y[j] = 0.0;
                    continue;
                }
                let mut denom = if self_loops { sum_x } else { sum_x - x[j] };
                for &(src, f_ij) in &col_src[j] {
                    denom += x[src] * (f_ij - 1.0);
                }
                y[j] = if denom > 0.0 {
                    strength_in[j] / denom
                } else {
                    0.0
                };
            }
            let sum_y: f64 = y.iter().sum();
            for i in 0..n {
                if strength_out[i] <= 0.0 {
                    x[i] = 0.0;
                    continue;
                }
                let mut denom = if self_loops { sum_y } else { sum_y - y[i] };
                for &(tgt, f_ij) in &row_tgt[i] {
                    denom += y[tgt] * (f_ij - 1.0);
                }
                x[i] = if denom > 0.0 {
                    strength_out[i] / denom
                } else {
                    0.0
                };
            }
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

/// Compute expected total cost given x, y, gamma and cost entries.
fn expected_cost(
    x: &[f64],
    y: &[f64],
    costs: &CostMatrix<'_>,
    gamma: f64,
    n: usize,
    self_loops: bool,
) -> f64 {
    let mut total = 0.0;
    for (idx, (&src, &tgt)) in costs.sources.iter().zip(costs.targets.iter()).enumerate() {
        if !self_loops && src == tgt {
            continue;
        }
        if src < n && tgt < n {
            let d = costs.values[idx];
            total += x[src] * y[tgt] * d * (-gamma * d).exp();
        }
    }
    total
}

/// Strength-cost solver options.
pub struct CostFitOptions {
    pub self_loops: bool,
    pub tolerance: f64,
    pub max_iterations: usize,
}

use super::support::coord_distance;

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

    for iter in 0..max_iterations {
        let old_x = x.clone();
        let old_y = y.clone();
        for j in 0..n {
            if strength_in[j] <= 0.0 {
                y[j] = 0.0;
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&i| self_loops || i != j)
                .map(|i| x[i] * (-gamma * coord_distance(coord_x, coord_y, i, j)).exp())
                .sum();
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
            let denom: f64 = (0..n)
                .filter(|&j| self_loops || i != j)
                .map(|j| y[j] * (-gamma * coord_distance(coord_x, coord_y, i, j)).exp())
                .sum();
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

/// Fit the strength-cost model using adaptive search on gamma.
///
/// Inner loop: IPF balancing of x, y for fixed gamma.
/// Outer loop: adaptive step on gamma to match target cost,
/// following the thesis algorithm with warm-started x, y.
#[must_use]
pub fn fit_strength_cost_poisson(
    strength_out: &[f64],
    strength_in: &[f64],
    cost_sources: &[usize],
    cost_targets: &[usize],
    cost_values: &[f64],
    target_cost: f64,
    opts: &CostFitOptions,
) -> StrengthCostFitResult {
    let costs = CostMatrix {
        sources: cost_sources,
        targets: cost_targets,
        values: cost_values,
    };
    let n = strength_out.len();
    let self_loops = opts.self_loops;
    let tolerance = opts.tolerance;
    let max_iterations = opts.max_iterations;

    let solve_at = |gamma: f64, x_init: Option<&[f64]>, y_init: Option<&[f64]>| {
        let fit = balance_xy_cost_sparse(
            strength_out,
            strength_in,
            &costs,
            n,
            gamma,
            self_loops,
            tolerance,
            max_iterations,
            x_init,
            y_init,
        );
        let delta = expected_cost(&fit.x, &fit.y, &costs, gamma, n, self_loops) - target_cost;
        (fit, delta)
    };

    // At gamma=0, cost is maximized for ME.
    let (fit_zero, delta_zero) = solve_at(0.0, None, None);
    if delta_zero.abs() <= tolerance {
        return StrengthCostFitResult {
            x: fit_zero.x,
            y: fit_zero.y,
            gamma: 0.0,
            converged: true,
            iterations: 1,
        };
    }
    // Bracket: increasing gamma decreases cost.
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
            converged: delta_zero.abs() <= tolerance,
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
            iterations: max_iterations,
        };
    }
    // Bisection
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
    for iter in 0..max_iterations {
        let mid = 0.5 * (low + high);
        let (fit, delta) = solve_at(mid, Some(&best_fit.x), Some(&best_fit.y));
        if delta.abs() < best_delta {
            best_delta = delta.abs();
            best_gamma = mid;
            best_fit = fit.clone();
        }
        if delta.abs() <= tolerance {
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
        converged: best_delta <= tolerance,
        iterations: max_iterations,
    }
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
    use super::{fit_strength_cost_poisson, CostFitOptions};

    #[test]
    fn recovers_strengths_for_uniform_cost() {
        let s_out = vec![10.0, 20.0, 30.0];
        let s_in = vec![15.0, 25.0, 20.0];
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        let mut cost_vals = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                sources.push(i);
                targets.push(j);
                cost_vals.push(1.0);
            }
        }
        let total_cost: f64 = 60.0 * 0.8;
        let result = fit_strength_cost_poisson(
            &s_out,
            &s_in,
            &sources,
            &targets,
            &cost_vals,
            total_cost,
            &CostFitOptions {
                self_loops: true,
                tolerance: 1e-6,
                max_iterations: 5000,
            },
        );
        let n = 3;
        for (i, &s_out_i) in s_out.iter().enumerate() {
            let row_sum: f64 = (0..n)
                .map(|j| result.x[i] * result.y[j] * (-result.gamma * cost_vals[i * n + j]).exp())
                .sum();
            assert!(
                (row_sum - s_out_i).abs() < 0.1,
                "s_out[{i}]: expected {s_out_i}, got {row_sum}, gamma={}",
                result.gamma,
            );
        }
    }
}
