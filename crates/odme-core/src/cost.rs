//! Strength-cost model fitting: fixed strength + fixed total cost.
//!
//! E[t_ij] = x_i y_j exp(-gamma d_ij).

use crate::fitting::FitResult;

/// Result of strength-cost model fitting.
#[derive(Clone, Debug)]
pub struct StrengthCostFitResult {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub gamma: f64,
    pub converged: bool,
    pub iterations: usize,
}

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

        let delta = x
            .iter()
            .zip(old_x.iter())
            .chain(y.iter().zip(old_y.iter()))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
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
pub fn fit_strength_cost(
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
    use super::*;

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
        let result = fit_strength_cost(
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
