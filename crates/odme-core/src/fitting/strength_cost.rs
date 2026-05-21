//! Binomial(M) and W strength-cost coordinate fitting.
//!
//! B strength-cost: E[t_ij] = M * x_i * y_j * f_ij / (1 + x_i * y_j * f_ij)
//!   where f_ij = exp(-γ * d_ij).
//!
//! W strength-cost: E[t_ij] = w_mean(a_i + b_j + γ * d_ij, M)
//!   = M * exp(-r_ij) / (1 - exp(-r_ij)) where r_ij = a_i + b_j + γ * d_ij.
//!
//! Both use IPF + gamma bisection, same structure as ME strength-cost
//! but with family-specific E[t_ij] and IPF update equations.

use super::{CostFitOptions, FitResult, StrengthCostFitResult};

// ---------------------------------------------------------------------------
// Shared coordinate distance helper
// ---------------------------------------------------------------------------

use super::support::coord_distance;

// ===========================================================================
// B (Binomial M) strength-cost coordinate fitting
// ===========================================================================

/// B expected weight: E[t_ij] = M * x_i * y_j * f_ij / (1 + x_i * y_j * f_ij)
#[inline]
fn b_expected(x_i: f64, y_j: f64, f_ij: f64, layers: u32) -> f64 {
    let z = x_i * y_j * f_ij;
    f64::from(layers) * z / (1.0 + z)
}

/// IPF balancing for B strength-cost at fixed gamma with coordinate costs.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn balance_b_strength_cost_coordinates(
    strength_out: &[f64],
    strength_in: &[f64],
    coord_x: &[f64],
    coord_y: &[f64],
    gamma: f64,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
    x_init: Option<&[f64]>,
    y_init: Option<&[f64]>,
) -> FitResult {
    let n = strength_out.len();
    let m = f64::from(layers);
    let total: f64 = strength_out.iter().sum();
    // B saturation: E[t_ij] = M*z/(1+z). For small z, E≈M*z. Init z ~ s/(M*n).
    let k_avg = total / m / n.max(1) as f64;
    let c = if k_avg < n as f64 {
        (k_avg / (n as f64 - k_avg).max(0.01)).sqrt()
    } else {
        1.0
    };
    let mut x: Vec<f64> = x_init.map_or_else(
        || {
            strength_out
                .iter()
                .map(|&s| {
                    if s > 0.0 {
                        s / total * k_avg.sqrt().max(0.1) / c.max(0.01)
                    } else {
                        0.0
                    }
                })
                .collect()
        },
        <[f64]>::to_vec,
    );
    let mut y: Vec<f64> = y_init.map_or_else(
        || {
            strength_in
                .iter()
                .map(|&s| {
                    if s > 0.0 {
                        s / total * k_avg.sqrt().max(0.1) / c.max(0.01)
                    } else {
                        0.0
                    }
                })
                .collect()
        },
        <[f64]>::to_vec,
    );

    // B strength constraint: s_out_i = Σ_j M * x_i * y_j * f_ij / (1 + x_i * y_j * f_ij)
    // Divide by M: k_out_i = s_out_i / M = Σ_j x_i * y_j * f_ij / (1 + x_i * y_j * f_ij)
    // IPF update for y_j: k_in_j / Σ_i x_i * f_ij / (1 + x_i * y_j * f_ij)
    let k_out: Vec<f64> = strength_out.iter().map(|&s| s / m).collect();
    let k_in: Vec<f64> = strength_in.iter().map(|&s| s / m).collect();

    for iter in 0..max_iterations {
        for j in 0..n {
            if k_in[j] <= 0.0 {
                y[j] = 0.0;
                continue;
            }
            let mut denom = 0.0;
            for i in 0..n {
                if !self_loops && i == j {
                    continue;
                }
                let f_ij = (-gamma * coord_distance(coord_x, coord_y, i, j)).exp();
                denom += x[i] * f_ij / (1.0 + x[i] * y[j] * f_ij);
            }
            y[j] = if denom > 0.0 { k_in[j] / denom } else { 0.0 };
        }
        for i in 0..n {
            if k_out[i] <= 0.0 {
                x[i] = 0.0;
                continue;
            }
            let mut denom = 0.0;
            for j in 0..n {
                if !self_loops && i == j {
                    continue;
                }
                let f_ij = (-gamma * coord_distance(coord_x, coord_y, i, j)).exp();
                denom += y[j] * f_ij / (1.0 + x[i] * y[j] * f_ij);
            }
            x[i] = if denom > 0.0 { k_out[i] / denom } else { 0.0 };
        }

        // Convergence: check predicted strengths
        let mut max_err = 0.0_f64;
        for i in 0..n {
            let mut pred = 0.0;
            for j in 0..n {
                if !self_loops && i == j {
                    continue;
                }
                let f_ij = (-gamma * coord_distance(coord_x, coord_y, i, j)).exp();
                pred += b_expected(x[i], y[j], f_ij, layers);
            }
            max_err = max_err.max((pred - strength_out[i]).abs());
        }
        if max_err < tolerance {
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

/// Expected total cost for B model with coordinate distances.
#[allow(clippy::needless_range_loop)]
fn b_expected_cost_coordinates(
    x: &[f64],
    y: &[f64],
    coord_x: &[f64],
    coord_y: &[f64],
    gamma: f64,
    layers: u32,
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
            let f_ij = (-gamma * d).exp();
            total += d * b_expected(x[i], y[j], f_ij, layers);
        }
    }
    total
}

/// Fit B(M) strength-cost model with coordinate costs using IPF + gamma bisection.
///
/// Thesis equation: E[t_ij] = M * x_i * y_j * exp(-γ d_ij) / (1 + x_i * y_j * exp(-γ d_ij))
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_strength_cost_binomial_coordinates(
    strength_out: &[f64],
    strength_in: &[f64],
    coord_x: &[f64],
    coord_y: &[f64],
    target_cost: f64,
    layers: u32,
    opts: &CostFitOptions,
) -> StrengthCostFitResult {
    let solve_at = |gamma: f64, x_init: Option<&[f64]>, y_init: Option<&[f64]>| {
        let fit = balance_b_strength_cost_coordinates(
            strength_out,
            strength_in,
            coord_x,
            coord_y,
            gamma,
            layers,
            opts.self_loops,
            opts.tolerance,
            opts.max_iterations,
            x_init,
            y_init,
        );
        let delta = b_expected_cost_coordinates(
            &fit.x,
            &fit.y,
            coord_x,
            coord_y,
            gamma,
            layers,
            opts.self_loops,
        ) - target_cost;
        (fit, delta)
    };

    // At gamma=0, cost is maximized. Find initial bracket.
    let (fit_zero, delta_zero) = solve_at(0.0, None, None);
    if delta_zero.abs() <= opts.tolerance {
        return StrengthCostFitResult {
            x: fit_zero.x,
            y: fit_zero.y,
            gamma: 0.0,
            converged: true,
            iterations: fit_zero.iterations,
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

    for _ in 0..64 {
        if delta_zero > 0.0 {
            // Cost at gamma=0 > target: need larger gamma
            high = step;
            let (fit, delta) = solve_at(high, Some(&high_fit.x), Some(&high_fit.y));
            high_fit = fit;
            high_delta = delta;
            if high_delta <= 0.0 {
                break;
            }
        } else {
            // Cost at gamma=0 < target: infeasible (gamma >= 0 only reduces cost)
            // Return best at gamma=0
            return StrengthCostFitResult {
                x: fit_zero.x,
                y: fit_zero.y,
                gamma: 0.0,
                converged: delta_zero.abs() <= opts.tolerance,
                iterations: fit_zero.iterations,
            };
        }
        step *= 2.0;
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

// ===========================================================================
// B (Binomial M) strength-cost sparse fitting
// ===========================================================================

/// Build dense f_mat = exp(-gamma * d_ij) from sparse cost entries for B model.
fn build_b_f_mat(
    n: usize,
    cost_sources: &[usize],
    cost_targets: &[usize],
    cost_values: &[f64],
    gamma: f64,
    self_loops: bool,
) -> Vec<f64> {
    let mut f_mat = vec![1.0; n * n];
    if !self_loops {
        for i in 0..n {
            f_mat[i * n + i] = 0.0;
        }
    }
    for (idx, (&src, &tgt)) in cost_sources.iter().zip(cost_targets.iter()).enumerate() {
        if !self_loops && src == tgt {
            continue;
        }
        if src < n && tgt < n {
            f_mat[src * n + tgt] = (-gamma * cost_values[idx]).exp();
        }
    }
    f_mat
}

/// IPF balancing for B(M) strength-cost with precomputed f_mat.
#[allow(clippy::too_many_arguments)]
fn balance_b_strength_cost_fmat(
    strength_out: &[f64],
    strength_in: &[f64],
    f_mat: &[f64],
    n: usize,
    layers: u32,
    tolerance: f64,
    max_iterations: usize,
    x_init: Option<&[f64]>,
    y_init: Option<&[f64]>,
) -> FitResult {
    let m = f64::from(layers);
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
    let k_out: Vec<f64> = strength_out.iter().map(|&s| s / m).collect();
    let k_in: Vec<f64> = strength_in.iter().map(|&s| s / m).collect();

    for iter in 0..max_iterations {
        for j in 0..n {
            if k_in[j] <= 0.0 {
                y[j] = 0.0;
                continue;
            }
            let mut denom = 0.0;
            for i in 0..n {
                let f_ij = f_mat[i * n + j];
                if f_ij == 0.0 {
                    continue;
                }
                denom += x[i] * f_ij / (1.0 + x[i] * y[j] * f_ij);
            }
            y[j] = if denom > 0.0 { k_in[j] / denom } else { 0.0 };
        }
        for i in 0..n {
            if k_out[i] <= 0.0 {
                x[i] = 0.0;
                continue;
            }
            let mut denom = 0.0;
            for j in 0..n {
                let f_ij = f_mat[i * n + j];
                if f_ij == 0.0 {
                    continue;
                }
                denom += y[j] * f_ij / (1.0 + x[i] * y[j] * f_ij);
            }
            x[i] = if denom > 0.0 { k_out[i] / denom } else { 0.0 };
        }
        let mut max_err = 0.0_f64;
        for i in 0..n {
            let mut pred = 0.0;
            for j in 0..n {
                let f_ij = f_mat[i * n + j];
                if f_ij == 0.0 {
                    continue;
                }
                pred += b_expected(x[i], y[j], f_ij, layers);
            }
            max_err = max_err.max((pred - strength_out[i]).abs());
        }
        if max_err < tolerance {
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

/// Expected total cost for B model with sparse cost entries.
#[allow(clippy::too_many_arguments)]
fn b_expected_cost_sparse(
    x: &[f64],
    y: &[f64],
    cost_sources: &[usize],
    cost_targets: &[usize],
    cost_values: &[f64],
    gamma: f64,
    n: usize,
    layers: u32,
    self_loops: bool,
) -> f64 {
    let mut total = 0.0;
    for (idx, (&src, &tgt)) in cost_sources.iter().zip(cost_targets.iter()).enumerate() {
        if !self_loops && src == tgt {
            continue;
        }
        if src < n && tgt < n {
            let d = cost_values[idx];
            let f_ij = (-gamma * d).exp();
            total += d * b_expected(x[src], y[tgt], f_ij, layers);
        }
    }
    total
}

/// Fit B(M) strength-cost model with sparse cost entries using IPF + gamma bisection.
///
/// Thesis equation: E[t_ij] = M * x_i * y_j * exp(-gamma * d_ij) / (1 + x_i * y_j * exp(-gamma * d_ij))
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_strength_cost_binomial(
    strength_out: &[f64],
    strength_in: &[f64],
    cost_sources: &[usize],
    cost_targets: &[usize],
    cost_values: &[f64],
    target_cost: f64,
    layers: u32,
    opts: &CostFitOptions,
) -> StrengthCostFitResult {
    let n = strength_out.len();
    let solve_at = |gamma: f64, x_init: Option<&[f64]>, y_init: Option<&[f64]>| {
        let f_mat = build_b_f_mat(
            n,
            cost_sources,
            cost_targets,
            cost_values,
            gamma,
            opts.self_loops,
        );
        let fit = balance_b_strength_cost_fmat(
            strength_out,
            strength_in,
            &f_mat,
            n,
            layers,
            opts.tolerance,
            opts.max_iterations,
            x_init,
            y_init,
        );
        let delta = b_expected_cost_sparse(
            &fit.x,
            &fit.y,
            cost_sources,
            cost_targets,
            cost_values,
            gamma,
            n,
            layers,
            opts.self_loops,
        ) - target_cost;
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

// ===========================================================================
// W (Geometric / Negative Binomial M) strength-cost coordinate fitting
// ===========================================================================

/// Fit W(M) strength-cost model with projected Euclidean XY coordinates.
///
/// Computes distances inline during CSC matrix assembly. No dense 3*N^2
/// cost-triple array is allocated. Memory is O(N^2) for the conic variables
/// (unavoidable in the exponential-cone formulation).
///
/// Thesis equation: E[t_ij] = M * exp(-r_ij) / (1 - exp(-r_ij))
///   where r_ij = a_i + b_j + gamma * d_ij and d_ij = ||coord_i - coord_j||.
#[cfg(feature = "w-conic")]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_strength_cost_w_coordinates(
    strength_out: &[f64],
    strength_in: &[f64],
    coord_x: &[f64],
    coord_y: &[f64],
    target_cost: f64,
    layers: u32,
    opts: &CostFitOptions,
) -> StrengthCostFitResult {
    use clarabel::algebra::CscMatrix;
    use clarabel::solver::{
        DefaultSettings, DefaultSolver, ExponentialConeT, IPSolver, NonnegativeConeT, SolverStatus,
        SupportedConeT, ZeroConeT,
    };

    let n = strength_out.len();
    if n == 0 || n != strength_in.len() || layers == 0 || !target_cost.is_finite() {
        return StrengthCostFitResult {
            x: vec![0.0; n],
            y: vec![0.0; n],
            gamma: 0.0,
            converged: false,
            iterations: 0,
        };
    }

    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .filter(|&(i, j)| opts.self_loops || i != j)
        .collect();
    let pc = pairs.len();
    let gamma_idx = 2 * n;
    let t_offset = gamma_idx + 1;
    let p_offset = t_offset + pc;
    let q_offset = p_offset + pc;
    let variable_count = 2 * n + 1 + 3 * pc;
    let row_count = 1 + 6 * pc + pc;

    let exp_base = 1_usize;
    let nonneg_base = 1 + 6 * pc;

    let mut rhs = vec![0.0_f64; row_count];
    for k in 0..pc {
        rhs[exp_base + 6 * k + 1] = 1.0;
        rhs[exp_base + 6 * k + 4] = 1.0;
    }
    for k in 0..pc {
        rhs[nonneg_base + k] = 1.0;
    }

    // CSC assembly — distances computed inline via coord_distance
    let mut colptr = vec![0_usize; variable_count + 1];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();

    for i in 0..n {
        let start = rowval.len();
        rowval.push(0);
        nzval.push(1.0);
        for (k, &(pi, _)) in pairs.iter().enumerate() {
            if pi == i {
                rowval.push(exp_base + 6 * k + 3);
                nzval.push(1.0);
            }
        }
        colptr[i + 1] = rowval.len() - start;
    }
    for j in 0..n {
        let start = rowval.len();
        rowval.push(0);
        nzval.push(-1.0);
        for (k, &(_, pj)) in pairs.iter().enumerate() {
            if pj == j {
                rowval.push(exp_base + 6 * k + 3);
                nzval.push(1.0);
            }
        }
        colptr[n + j + 1] = rowval.len() - start;
    }
    // Gamma column: inline distance computation
    {
        let start = rowval.len();
        for (k, &(i, j)) in pairs.iter().enumerate() {
            let cost = coord_distance(coord_x, coord_y, i, j);
            if cost != 0.0 {
                rowval.push(exp_base + 6 * k + 3);
                nzval.push(cost);
            }
        }
        colptr[gamma_idx + 1] = rowval.len() - start;
    }
    for k in 0..pc {
        let start = rowval.len();
        rowval.push(exp_base + 6 * k);
        nzval.push(1.0);
        colptr[t_offset + k + 1] = rowval.len() - start;
    }
    for k in 0..pc {
        let start = rowval.len();
        rowval.push(exp_base + 6 * k + 2);
        nzval.push(-1.0);
        rowval.push(nonneg_base + k);
        nzval.push(1.0);
        colptr[p_offset + k + 1] = rowval.len() - start;
    }
    for k in 0..pc {
        let start = rowval.len();
        rowval.push(exp_base + 6 * k + 5);
        nzval.push(-1.0);
        rowval.push(nonneg_base + k);
        nzval.push(1.0);
        colptr[q_offset + k + 1] = rowval.len() - start;
    }
    for i in 0..variable_count {
        colptr[i + 1] += colptr[i];
    }

    let mut objective = vec![0.0_f64; variable_count];
    objective[..n].copy_from_slice(strength_out);
    objective[n..2 * n].copy_from_slice(strength_in);
    objective[gamma_idx] = target_cost;
    for k in 0..pc {
        objective[t_offset + k] = f64::from(layers);
    }

    let p_matrix = CscMatrix::<f64>::zeros((variable_count, variable_count));
    let a_matrix = CscMatrix::new(row_count, variable_count, colptr, rowval, nzval);
    let mut cones: Vec<SupportedConeT<f64>> = Vec::with_capacity(2 * pc + 2);
    cones.push(ZeroConeT(1));
    for _ in 0..(2 * pc) {
        cones.push(ExponentialConeT());
    }
    cones.push(NonnegativeConeT(pc));
    let settings = DefaultSettings {
        verbose: false,
        max_iter: opts.max_iterations.try_into().unwrap_or(u32::MAX),
        tol_gap_abs: opts.tolerance,
        tol_gap_rel: opts.tolerance,
        tol_feas: opts.tolerance,
        ..DefaultSettings::default()
    };

    let Ok(mut solver) =
        DefaultSolver::new(&p_matrix, &objective, &a_matrix, &rhs, &cones, settings)
    else {
        return StrengthCostFitResult {
            x: vec![0.0; n],
            y: vec![0.0; n],
            gamma: 0.0,
            converged: false,
            iterations: 0,
        };
    };
    solver.solve();
    let converged = matches!(
        solver.solution.status,
        SolverStatus::Solved
            | SolverStatus::AlmostSolved
            | SolverStatus::MaxIterations
            | SolverStatus::MaxTime
    );
    let solution = &solver.solution.x;
    let a = &solution[..n];
    let b = &solution[n..2 * n];
    let gamma = solution[gamma_idx];

    StrengthCostFitResult {
        x: a.iter().map(|&ai| (-ai).exp()).collect(),
        y: b.iter().map(|&bj| (-bj).exp()).collect(),
        gamma,
        converged,
        iterations: solver.solution.iterations as usize,
    }
}

/// Fallback when w-conic feature is not enabled.
#[cfg(not(feature = "w-conic"))]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_strength_cost_w_coordinates(
    strength_out: &[f64],
    _strength_in: &[f64],
    _coord_x: &[f64],
    _coord_y: &[f64],
    _target_cost: f64,
    _layers: u32,
    _opts: &CostFitOptions,
) -> StrengthCostFitResult {
    let n = strength_out.len();
    StrengthCostFitResult {
        x: vec![0.0; n],
        y: vec![0.0; n],
        gamma: 0.0,
        converged: false,
        iterations: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::super::w::w_mean;
    use super::*;

    /// Verify B strength-cost produces different results from ME on same input.
    /// Uses strengths feasible for B(M=3): max row sum < M*n = 9.
    #[test]
    fn b_strength_cost_differs_from_me() {
        use super::super::me::fit_strength_cost_poisson_coordinates;
        let s_out = vec![2.0, 3.0, 4.0];
        let s_in = vec![2.5, 3.5, 3.0];
        let cx = vec![0.0, 3.0, 0.0];
        let cy = vec![0.0, 0.0, 4.0];
        let target_cost = 15.0;
        let opts = CostFitOptions {
            self_loops: true,
            tolerance: 1e-4,
            max_iterations: 50000,
        };
        let me = fit_strength_cost_poisson_coordinates(&s_out, &s_in, &cx, &cy, target_cost, &opts);
        let b =
            fit_strength_cost_binomial_coordinates(&s_out, &s_in, &cx, &cy, target_cost, 3, &opts);
        assert!(me.converged, "ME did not converge");
        assert!(b.converged, "B did not converge, gamma={}", b.gamma);
        // B has saturation that ME does not; gamma must differ
        let diff_gamma = (me.gamma - b.gamma).abs();
        assert!(
            diff_gamma > 1e-6,
            "ME and B should produce different gamma: me={}, b={}",
            me.gamma,
            b.gamma
        );
    }

    /// Verify B IPF converges at fixed gamma.
    #[test]
    fn b_ipf_fixed_gamma_converges() {
        let s_out = vec![2.0, 3.0, 4.0];
        let s_in = vec![2.5, 3.5, 3.0];
        let cx = vec![0.0, 3.0, 0.0];
        let cy = vec![0.0, 0.0, 4.0];
        let layers = 3_u32;
        let fit = balance_b_strength_cost_coordinates(
            &s_out, &s_in, &cx, &cy, 0.0, layers, true, 1e-4, 5000, None, None,
        );
        eprintln!(
            "B IPF gamma=0: converged={} iters={} x={:?} y={:?}",
            fit.converged, fit.iterations, &fit.x, &fit.y
        );
        assert!(fit.converged, "B IPF at gamma=0 did not converge");
    }

    /// Verify B strength-cost recovers strengths within tolerance.
    #[test]
    fn b_strength_cost_recovers_strengths() {
        let s_out = vec![2.0, 3.0, 4.0];
        let s_in = vec![2.5, 3.5, 3.0];
        let cx = vec![0.0, 3.0, 0.0];
        let cy = vec![0.0, 0.0, 4.0];
        let target_cost = 12.0;
        let layers = 3_u32;
        let opts = CostFitOptions {
            self_loops: true,
            tolerance: 1e-4,
            max_iterations: 50000,
        };
        let fit = fit_strength_cost_binomial_coordinates(
            &s_out,
            &s_in,
            &cx,
            &cy,
            target_cost,
            layers,
            &opts,
        );
        assert!(fit.converged, "B(3) strength-cost did not converge");
        let n = s_out.len();
        for (i, &expected) in s_out.iter().enumerate() {
            let mut pred = 0.0;
            for j in 0..n {
                let d = coord_distance(&cx, &cy, i, j);
                let f_ij = (-fit.gamma * d).exp();
                pred += b_expected(fit.x[i], fit.y[j], f_ij, layers);
            }
            assert!(
                (pred - expected).abs() < 0.1,
                "s_out[{i}]: expected {expected}, got {pred}"
            );
        }
    }

    /// Verify W strength-cost produces different results from ME on same input.
    #[test]
    fn w_strength_cost_differs_from_me() {
        use super::super::me::fit_strength_cost_poisson_coordinates;
        let s_out = vec![10.0, 20.0, 30.0];
        let s_in = vec![15.0, 25.0, 20.0];
        let cx = vec![0.0, 3.0, 0.0];
        let cy = vec![0.0, 0.0, 4.0];
        let target_cost = 100.0;
        let opts = CostFitOptions {
            self_loops: true,
            tolerance: 1e-4,
            max_iterations: 50000,
        };
        let me = fit_strength_cost_poisson_coordinates(&s_out, &s_in, &cx, &cy, target_cost, &opts);
        let w = fit_strength_cost_w_coordinates(&s_out, &s_in, &cx, &cy, target_cost, 1, &opts);
        assert!(me.converged, "ME did not converge");
        assert!(w.converged, "W did not converge, gamma={}", w.gamma);
        let diff_gamma = (me.gamma - w.gamma).abs();
        assert!(
            diff_gamma > 1e-6,
            "ME and W should produce different gamma: me={}, w={}",
            me.gamma,
            w.gamma
        );
    }

    /// Verify W strength-cost recovers strengths within tolerance.
    #[test]
    fn w_strength_cost_recovers_strengths() {
        let s_out = vec![10.0, 20.0, 30.0];
        let s_in = vec![15.0, 25.0, 20.0];
        let cx = vec![0.0, 3.0, 0.0];
        let cy = vec![0.0, 0.0, 4.0];
        let target_cost = 80.0;
        let layers = 3_u32;
        let opts = CostFitOptions {
            self_loops: true,
            tolerance: 1e-4,
            max_iterations: 50000,
        };
        let fit =
            fit_strength_cost_w_coordinates(&s_out, &s_in, &cx, &cy, target_cost, layers, &opts);
        assert!(fit.converged, "W(3) strength-cost did not converge");
        let n = s_out.len();
        for (i, &expected) in s_out.iter().enumerate() {
            let mut pred = 0.0;
            for j in 0..n {
                let d = coord_distance(&cx, &cy, i, j);
                let a_i = -(fit.x[i].ln());
                let b_j = -(fit.y[j].ln());
                let r = (a_i + b_j + fit.gamma * d).max(1e-10);
                pred += w_mean(r, layers);
            }
            assert!(
                (pred - expected).abs() < 1.0,
                "s_out[{i}]: expected {expected}, got {pred}"
            );
        }
    }
    /// Verify sparse B strength-cost matches coordinate B on same input.
    #[test]
    fn b_sparse_matches_coordinate() {
        let s_out = vec![2.0, 3.0, 4.0];
        let s_in = vec![2.5, 3.5, 3.0];
        let cx = vec![0.0, 3.0, 0.0];
        let cy = vec![0.0, 0.0, 4.0];
        let n = 3;
        let target_cost = 12.0;
        let layers = 3_u32;
        let opts = CostFitOptions {
            self_loops: true,
            tolerance: 1e-4,
            max_iterations: 50000,
        };
        // Build sparse costs from coordinates
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            for j in 0..n {
                sources.push(i);
                targets.push(j);
                values.push(coord_distance(&cx, &cy, i, j));
            }
        }
        let sparse = fit_strength_cost_binomial(
            &s_out,
            &s_in,
            &sources,
            &targets,
            &values,
            target_cost,
            layers,
            &opts,
        );
        let coord = fit_strength_cost_binomial_coordinates(
            &s_out,
            &s_in,
            &cx,
            &cy,
            target_cost,
            layers,
            &opts,
        );
        assert!(sparse.converged, "sparse B did not converge");
        assert!(coord.converged, "coord B did not converge");
        assert!(
            (sparse.gamma - coord.gamma).abs() < 1e-3,
            "sparse gamma={} coord gamma={}",
            sparse.gamma,
            coord.gamma,
        );
    }
}
