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

/// Balance x,y for a fixed gamma and sparse cost matrix.
fn balance_xy_cost(
    strength_out: &[f64],
    strength_in: &[f64],
    costs: &CostMatrix<'_>,
    gamma: f64,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> FitResult {
    let n = strength_out.len();
    let cost_sources = costs.sources;
    let cost_targets = costs.targets;
    let cost_values = costs.values;
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

    // Precompute f_ij = exp(-gamma * d_ij) for all cost entries.
    let f: Vec<f64> = cost_values.iter().map(|&d| (-gamma * d).exp()).collect();

    // Build dense f_ij matrix for IPF (sparse would be better for large N
    // but matches original thesis code structure).
    let mut f_mat = vec![0.0; n * n];
    // Default: f_ij = exp(-gamma * 0) = 1 for pairs without explicit cost.
    for i in 0..n {
        for j in 0..n {
            if !self_loops && i == j {
                continue;
            }
            f_mat[i * n + j] = 1.0;
        }
    }
    for (idx, (&src, &tgt)) in cost_sources.iter().zip(cost_targets.iter()).enumerate() {
        if !self_loops && src == tgt {
            continue;
        }
        if src < n && tgt < n {
            f_mat[src * n + tgt] = f[idx];
        }
    }

    for iter in 0..max_iterations {
        let old_x = x.clone();
        let old_y = y.clone();

        // Update y
        for j in 0..n {
            if strength_in[j] <= 0.0 {
                continue;
            }
            let denom: f64 = (0..n).map(|i| x[i] * f_mat[i * n + j]).sum();
            y[j] = if denom > 0.0 {
                strength_in[j] / denom
            } else {
                0.0
            };
        }
        // Update x
        for i in 0..n {
            if strength_out[i] <= 0.0 {
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

/// Compute expected total cost for given x, y, gamma and cost entries.
fn expected_cost(
    x: &[f64],
    y: &[f64],
    costs: &CostMatrix<'_>,
    gamma: f64,
    n: usize,
    self_loops: bool,
) -> f64 {
    let cost_sources = costs.sources;
    let cost_targets = costs.targets;
    let cost_values = costs.values;
    // Sum x_i y_j d_ij exp(-gamma d_ij) over all pairs.
    // For pairs without explicit cost, d_ij = 0 so contribution is 0.
    let mut total = 0.0;
    // Also need x_i y_j * 0 for missing pairs = 0, so only iterate cost entries.
    for (idx, (&src, &tgt)) in cost_sources.iter().zip(cost_targets.iter()).enumerate() {
        if !self_loops && src == tgt {
            continue;
        }
        if src < n && tgt < n {
            let d = cost_values[idx];
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

/// Fit the strength-cost model: fixed strengths + fixed total cost.
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

    // Initial gamma guess: T / C
    let gamma_init = if target_cost > 0.0 {
        total / target_cost
    } else {
        1.0
    };

    let mut gamma = gamma_init;
    let mut delta = gamma_init * 0.5;
    let mut direction = 1.0_f64;

    let mut best_fit = balance_xy_cost(
        strength_out,
        strength_in,
        &costs,
        gamma,
        self_loops,
        tolerance,
        max_iterations,
    );
    let mut best_delta_c =
        expected_cost(&best_fit.x, &best_fit.y, &costs, gamma, n, self_loops) - target_cost;

    let factor = 3.0;

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

        let new_gamma = gamma + delta * direction;
        if new_gamma <= 0.0 {
            delta /= factor;
            direction = -direction;
            continue;
        }

        let fit = balance_xy_cost(
            strength_out,
            strength_in,
            &costs,
            new_gamma,
            self_loops,
            tolerance,
            max_iterations,
        );
        let new_delta_c =
            expected_cost(&fit.x, &fit.y, &costs, new_gamma, n, self_loops) - target_cost;

        if new_delta_c.abs() < best_delta_c.abs() {
            // Accepted
            if new_delta_c.signum() != best_delta_c.signum() {
                direction = -direction;
            }
            gamma = new_gamma;
            best_delta_c = new_delta_c;
            best_fit = fit;
            delta *= factor;
        } else {
            // Rejected
            delta /= factor;
            direction = -direction;
            if delta < 1e-15 {
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
        // With uniform d_ij = 1 and gamma ~0, should recover fixed-strength solution.
        let s_out = vec![10.0, 20.0, 30.0];
        let s_in = vec![15.0, 25.0, 20.0];
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        let mut costs = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                sources.push(i);
                targets.push(j);
                costs.push(1.0);
            }
        }
        let total_cost: f64 = {
            // For E[t_ij] = x_i y_j exp(-gamma), total cost = sum x_i y_j * 1 * exp(-gamma)
            // With gamma=0, total cost = T * 1 = 60
            60.0 * 0.8 // target something reasonable
        };
        let result = fit_strength_cost(
            &s_out,
            &s_in,
            &sources,
            &targets,
            &costs,
            total_cost,
            &CostFitOptions {
                self_loops: true,
                tolerance: 1e-6,
                max_iterations: 5000,
            },
        );
        // Check strengths are recovered
        let n = 3;
        for (i, &s_out_i) in s_out.iter().enumerate() {
            let row_sum: f64 = (0..n)
                .map(|j| result.x[i] * result.y[j] * (-result.gamma * costs[i * n + j]).exp())
                .sum();
            assert!(
                (row_sum - s_out_i).abs() < 0.1,
                "s_out[{i}]: expected {s_out_i}, got {row_sum}"
            );
        }
    }
}
