//! Multi-edge (ME/Poisson-family) fitting routines.

use super::b::binary_probability;
use super::support::{max_abs_delta, max_pair_delta};
use super::{FitResult, StrengthCostFitResult, StrengthDegreeFitResult, StrengthEdgesFitResult};

fn balance_strength_edges_for_lambda(
    strength_out: &[f64],
    strength_in: &[f64],
    lam: f64,
    self_loops: bool,
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
            let denom: f64 = (0..n)
                .filter(|&i| self_loops || i != j)
                .map(|i| {
                    let u = x[i] * y[j];
                    let exp_u = u.exp();
                    let den = 1.0 + lam * (exp_u - 1.0);
                    if den > 0.0 {
                        lam * x[i] * exp_u / den
                    } else {
                        0.0
                    }
                })
                .sum();
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
            let denom: f64 = (0..n)
                .filter(|&j| self_loops || i != j)
                .map(|j| {
                    let u = x[i] * y[j];
                    let exp_u = u.exp();
                    let den = 1.0 + lam * (exp_u - 1.0);
                    if den > 0.0 {
                        lam * y[j] * exp_u / den
                    } else {
                        0.0
                    }
                })
                .sum();
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

fn expected_edges_strength_edges(x: &[f64], y: &[f64], lam: f64, self_loops: bool) -> f64 {
    let mut total = 0.0;
    for (i, &xi) in x.iter().enumerate() {
        for (j, &yj) in y.iter().enumerate() {
            if !self_loops && i == j {
                continue;
            }
            let exp_u = (xi * yj).exp();
            total += lam * (exp_u - 1.0) / (1.0 + lam * (exp_u - 1.0));
        }
    }
    total
}

/// Fit exact grand-canonical ME fixed-strength-and-edge-count zero-inflated constraints.
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
    // Precondition: lambda ≈ E / (N² - E)
    let lam_init = if target_edges < n2 {
        target_edges / (n2 - target_edges).max(0.01)
    } else {
        1.0
    };
    let mut low = 1e-12;
    let mut high = lam_init.max(1.0);
    let mut best = balance_strength_edges_for_lambda(
        strength_out,
        strength_in,
        high,
        self_loops,
        tolerance,
        max_iterations,
    );
    while expected_edges_strength_edges(&best.x, &best.y, high, self_loops) < target_edges
        && high < 1e12
    {
        high *= 2.0;
        best = balance_strength_edges_for_lambda(
            strength_out,
            strength_in,
            high,
            self_loops,
            tolerance,
            max_iterations,
        );
    }
    let mut iterations = 0;
    for iter in 0..80 {
        let mid = 0.5 * (low + high);
        let fit = balance_strength_edges_for_lambda(
            strength_out,
            strength_in,
            mid,
            self_loops,
            tolerance,
            max_iterations,
        );
        let edges = expected_edges_strength_edges(&fit.x, &fit.y, mid, self_loops);
        best = fit;
        if (edges - target_edges).abs() < tolerance {
            iterations = iter + 1;
            high = mid;
            break;
        }
        if edges < target_edges {
            low = mid;
        } else {
            high = mid;
        }
        iterations = iter + 1;
    }
    StrengthEdgesFitResult {
        x: best.x,
        y: best.y,
        lam: high,
        converged: true,
        iterations,
    }
}

/// Fit exact grand-canonical ME fixed-strength-and-degree zero-inflated constraints.
///
/// The model is the thesis case 4 ME equation:
/// E[t_ij] = z_i w_j x_i y_j exp(x_i y_j) / (1 + z_i w_j(exp(x_i y_j)-1)).
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
    let total: f64 = strength_out.iter().sum();
    let sqrt_t = total.sqrt().max(1.0);
    let k_avg = degree_out.iter().sum::<f64>() / n.max(1) as f64;
    let n_eff = if self_loops {
        n as f64
    } else {
        (n - 1).max(1) as f64
    };
    let mut x: Vec<f64> = strength_out
        .iter()
        .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
        .collect();
    let mut y: Vec<f64> = strength_in
        .iter()
        .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
        .collect();
    let c_k = if k_avg < n_eff {
        (k_avg / (n_eff - k_avg).max(0.01)).sqrt()
    } else {
        0.9
    };
    let mut z: Vec<f64> = degree_out
        .iter()
        .map(|&k| {
            if k > 0.0 && k_avg > 0.0 {
                k / k_avg * c_k
            } else {
                0.0
            }
        })
        .collect();
    let mut w: Vec<f64> = degree_in
        .iter()
        .map(|&k| {
            if k > 0.0 && k_avg > 0.0 {
                k / k_avg * c_k
            } else {
                0.0
            }
        })
        .collect();

    for i in 0..n {
        if degree_out[i] <= 0.0 || strength_out[i] <= 0.0 {
            x[i] = 0.0;
            z[i] = 0.0;
        }
        if degree_in[i] <= 0.0 || strength_in[i] <= 0.0 {
            y[i] = 0.0;
            w[i] = 0.0;
        }
    }

    for iter in 0..max_iterations {
        let old_x = x.clone();
        let old_y = y.clone();
        let old_z = z.clone();
        let old_w = w.clone();

        for i in 0..n {
            if strength_out[i] <= 0.0 {
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&j| self_loops || i != j)
                .map(|j| {
                    let u = x[i] * y[j];
                    let v = z[i] * w[j];
                    let exp_u = u.exp();
                    let den = 1.0 + v * (exp_u - 1.0);
                    if den > 0.0 {
                        z[i] * y[j] * w[j] * exp_u / den
                    } else {
                        0.0
                    }
                })
                .sum();
            x[i] = if denom > 0.0 {
                strength_out[i] / denom
            } else {
                0.0
            };
        }
        for j in 0..n {
            if strength_in[j] <= 0.0 {
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&i| self_loops || i != j)
                .map(|i| {
                    let u = x[i] * y[j];
                    let v = z[i] * w[j];
                    let exp_u = u.exp();
                    let den = 1.0 + v * (exp_u - 1.0);
                    if den > 0.0 {
                        w[j] * x[i] * z[i] * exp_u / den
                    } else {
                        0.0
                    }
                })
                .sum();
            y[j] = if denom > 0.0 {
                strength_in[j] / denom
            } else {
                0.0
            };
        }
        for j in 0..n {
            if degree_in[j] <= 0.0 {
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&i| self_loops || i != j)
                .map(|i| {
                    let u = x[i] * y[j];
                    let exp_u = u.exp();
                    let den = 1.0 + z[i] * w[j] * (exp_u - 1.0);
                    if den > 0.0 {
                        z[i] * (exp_u - 1.0) / den
                    } else {
                        0.0
                    }
                })
                .sum();
            w[j] = if denom > 0.0 {
                degree_in[j] / denom
            } else {
                0.0
            };
        }
        for i in 0..n {
            if degree_out[i] <= 0.0 {
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&j| self_loops || i != j)
                .map(|j| {
                    let u = x[i] * y[j];
                    let exp_u = u.exp();
                    let den = 1.0 + z[i] * w[j] * (exp_u - 1.0);
                    if den > 0.0 {
                        w[j] * (exp_u - 1.0) / den
                    } else {
                        0.0
                    }
                })
                .sum();
            z[i] = if denom > 0.0 {
                degree_out[i] / denom
            } else {
                0.0
            };
        }

        let delta =
            max_pair_delta(&x, &old_x, &y, &old_y).max(max_pair_delta(&z, &old_z, &w, &old_w));
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

/// Masked IPF balancing for directed binary fixed-degree.
///
/// Pairs where ``mask[i * n + j]`` is true are skipped in summations.
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
    let total: f64 = strength_out.iter().sum();
    let sqrt_t = total.sqrt().max(1.0);
    let k_avg = degree_out.iter().sum::<f64>() / n.max(1) as f64;
    let n_free = (0..n * n).filter(|&idx| !mask[idx]).count() as f64 / n.max(1) as f64;
    let mut x: Vec<f64> = strength_out
        .iter()
        .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
        .collect();
    let mut y: Vec<f64> = strength_in
        .iter()
        .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
        .collect();
    let c_k = if k_avg < n_free {
        (k_avg / (n_free - k_avg).max(0.01)).sqrt()
    } else {
        0.9
    };
    let mut z: Vec<f64> = degree_out
        .iter()
        .map(|&k| {
            if k > 0.0 && k_avg > 0.0 {
                k / k_avg * c_k
            } else {
                0.0
            }
        })
        .collect();
    let mut w: Vec<f64> = degree_in
        .iter()
        .map(|&k| {
            if k > 0.0 && k_avg > 0.0 {
                k / k_avg * c_k
            } else {
                0.0
            }
        })
        .collect();
    for i in 0..n {
        if degree_out[i] <= 0.0 || strength_out[i] <= 0.0 {
            x[i] = 0.0;
            z[i] = 0.0;
        }
        if degree_in[i] <= 0.0 || strength_in[i] <= 0.0 {
            y[i] = 0.0;
            w[i] = 0.0;
        }
    }
    for iter in 0..max_iterations {
        let old_x = x.clone();
        let old_y = y.clone();
        let old_z = z.clone();
        let old_w = w.clone();
        for i in 0..n {
            if strength_out[i] <= 0.0 {
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&j| !mask[i * n + j])
                .map(|j| {
                    let u = x[i] * y[j];
                    let exp_u = u.exp();
                    let den = 1.0 + z[i] * w[j] * (exp_u - 1.0);
                    if den > 0.0 {
                        z[i] * y[j] * w[j] * exp_u / den
                    } else {
                        0.0
                    }
                })
                .sum();
            x[i] = if denom > 0.0 {
                strength_out[i] / denom
            } else {
                0.0
            };
        }
        for j in 0..n {
            if strength_in[j] <= 0.0 {
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&i| !mask[i * n + j])
                .map(|i| {
                    let u = x[i] * y[j];
                    let exp_u = u.exp();
                    let den = 1.0 + z[i] * w[j] * (exp_u - 1.0);
                    if den > 0.0 {
                        w[j] * x[i] * z[i] * exp_u / den
                    } else {
                        0.0
                    }
                })
                .sum();
            y[j] = if denom > 0.0 {
                strength_in[j] / denom
            } else {
                0.0
            };
        }
        for j in 0..n {
            if degree_in[j] <= 0.0 {
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&i| !mask[i * n + j])
                .map(|i| {
                    let u = x[i] * y[j];
                    let exp_u = u.exp();
                    let den = 1.0 + z[i] * w[j] * (exp_u - 1.0);
                    if den > 0.0 {
                        z[i] * (exp_u - 1.0) / den
                    } else {
                        0.0
                    }
                })
                .sum();
            w[j] = if denom > 0.0 {
                degree_in[j] / denom
            } else {
                0.0
            };
        }
        for i in 0..n {
            if degree_out[i] <= 0.0 {
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&j| !mask[i * n + j])
                .map(|j| {
                    let u = x[i] * y[j];
                    let exp_u = u.exp();
                    let den = 1.0 + z[i] * w[j] * (exp_u - 1.0);
                    if den > 0.0 {
                        w[j] * (exp_u - 1.0) / den
                    } else {
                        0.0
                    }
                })
                .sum();
            z[i] = if denom > 0.0 {
                degree_out[i] / denom
            } else {
                0.0
            };
        }
        let delta =
            max_pair_delta(&x, &old_x, &y, &old_y).max(max_pair_delta(&z, &old_z, &w, &old_w));
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
#[allow(clippy::too_many_arguments)]
fn balance_xy_cost(
    strength_out: &[f64],
    strength_in: &[f64],
    f_mat: &[f64],
    n: usize,
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

    for iter in 0..max_iterations {
        let old_x = x.clone();
        let old_y = y.clone();

        for j in 0..n {
            if strength_in[j] <= 0.0 {
                y[j] = 0.0;
                continue;
            }
            let denom: f64 = (0..n).map(|i| x[i] * f_mat[i * n + j]).sum();
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
            let denom: f64 = (0..n).map(|j| y[j] * f_mat[i * n + j]).sum();
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

/// Build dense f_mat = exp(-gamma * d_ij) from sparse cost entries.
fn build_f_mat(n: usize, costs: &CostMatrix<'_>, gamma: f64, self_loops: bool) -> Vec<f64> {
    let mut f_mat = vec![1.0; n * n];
    if !self_loops {
        for i in 0..n {
            f_mat[i * n + i] = 0.0;
        }
    }
    for (idx, (&src, &tgt)) in costs.sources.iter().zip(costs.targets.iter()).enumerate() {
        if !self_loops && src == tgt {
            continue;
        }
        if src < n && tgt < n {
            f_mat[src * n + tgt] = (-gamma * costs.values[idx]).exp();
        }
    }
    f_mat
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
    let total: f64 = strength_out.iter().sum();

    // Initial gamma guess.
    let gamma_init = if target_cost > 0.0 {
        total / target_cost
    } else {
        1.0
    };
    let mut gamma = gamma_init;
    let mut step = gamma_init * 0.5;
    let mut direction = 1.0_f64;
    let factor = 3.0;

    // Initial fit.
    let f_mat = build_f_mat(n, &costs, gamma, self_loops);
    let mut best_fit = balance_xy_cost(
        strength_out,
        strength_in,
        &f_mat,
        n,
        tolerance,
        max_iterations,
        None,
        None,
    );
    let mut best_delta_c =
        expected_cost(&best_fit.x, &best_fit.y, &costs, gamma, n, self_loops) - target_cost;

    for iter in 0..max_iterations {
        if best_delta_c.abs() < tolerance {
            return StrengthCostFitResult {
                x: best_fit.x,
                y: best_fit.y,
                gamma,
                converged: true,
                iterations: iter + 1,
            };
        }

        let new_gamma = gamma + step * direction;
        if new_gamma <= 0.0 {
            step /= factor;
            direction = -direction;
            continue;
        }

        let f_mat_new = build_f_mat(n, &costs, new_gamma, self_loops);
        let fit = balance_xy_cost(
            strength_out,
            strength_in,
            &f_mat_new,
            n,
            tolerance,
            max_iterations,
            Some(&best_fit.x),
            Some(&best_fit.y),
        );
        let new_delta_c =
            expected_cost(&fit.x, &fit.y, &costs, new_gamma, n, self_loops) - target_cost;

        if new_delta_c.abs() < best_delta_c.abs() {
            if new_delta_c.signum() != best_delta_c.signum() {
                direction = -direction;
            }
            gamma = new_gamma;
            best_delta_c = new_delta_c;
            best_fit = fit;
            step *= factor;
        } else {
            step /= factor;
            direction = -direction;
            if step < 1e-15 {
                break;
            }
        }
    }

    StrengthCostFitResult {
        x: best_fit.x,
        y: best_fit.y,
        gamma,
        converged: best_delta_c.abs() < tolerance,
        iterations: max_iterations,
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
