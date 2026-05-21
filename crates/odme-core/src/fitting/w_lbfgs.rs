//! Newton coordinate-descent W strength-cost solver.
//!
//! Alternating Newton updates on (a, b, γ) with damping.
//! Memory: O(N). Time per iteration: O(N²). No dense cost matrix.

use super::support::coord_distance;
use super::w::w_mean;
use super::{CostFitOptions, StrengthCostFitResult};

/// How pair costs are provided to the W Newton solver.
enum CostMode<'a> {
    /// No cost constraint (gamma fixed at 0).
    NoCost,
    /// Sparse cost entries.
    Sparse {
        col_costs: Vec<Vec<(usize, f64)>>,
        #[allow(dead_code)]
        row_costs: Vec<Vec<(usize, f64)>>,
    },
    /// Projected Euclidean XY coordinates.
    Coordinates { x: &'a [f64], y: &'a [f64] },
}

impl<'a> CostMode<'a> {
    fn build_sparse(
        n: usize,
        sources: &'a [usize],
        targets: &'a [usize],
        values: &'a [f64],
        self_loops: bool,
    ) -> Self {
        let mut col_costs: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut row_costs: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for (idx, (&src, &tgt)) in sources.iter().zip(targets.iter()).enumerate() {
            if !self_loops && src == tgt {
                continue;
            }
            if src < n && tgt < n {
                col_costs[tgt].push((src, values[idx]));
                row_costs[src].push((tgt, values[idx]));
            }
        }
        CostMode::Sparse {
            col_costs,
            row_costs,
        }
    }
}

/// Derivative of w_mean(r, M) w.r.t. r.
#[inline]
fn w_mean_deriv(r: f64, layers: u32) -> f64 {
    if r <= 1e-10 {
        return 0.0;
    }
    let q = (-r).exp();
    let omq = 1.0 - q;
    -f64::from(layers) * q / (omq * omq)
}

/// Fit W(M) fixed-strength (no cost constraint) using Newton coordinate-descent.
#[must_use]
pub fn fit_strength_w_newton(
    strength_out: &[f64],
    strength_in: &[f64],
    layers: u32,
    opts: &CostFitOptions,
) -> StrengthCostFitResult {
    let n = strength_out.len();
    // dummy coords (unused since CostMode::NoCost)
    fit_w_newton_inner(
        strength_out,
        strength_in,
        0.0,
        layers,
        opts,
        &CostMode::NoCost,
        n,
    )
}

/// Fit W(M) strength-cost with sparse cost entries using Newton coordinate-descent.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_strength_cost_w_sparse_newton(
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
    let mode = CostMode::build_sparse(n, cost_sources, cost_targets, cost_values, opts.self_loops);
    fit_w_newton_inner(
        strength_out,
        strength_in,
        target_cost,
        layers,
        opts,
        &mode,
        n,
    )
}

/// Fit W(M) strength-cost with projected coordinates using Newton coordinate-descent.
///
/// No N² memory allocated. O(N²) per iteration.
#[must_use]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn fit_strength_cost_w_lbfgs(
    strength_out: &[f64],
    strength_in: &[f64],
    coord_x: &[f64],
    coord_y: &[f64],
    target_cost: f64,
    layers: u32,
    opts: &CostFitOptions,
) -> StrengthCostFitResult {
    let n = strength_out.len();
    let mode = CostMode::Coordinates {
        x: coord_x,
        y: coord_y,
    };
    fit_w_newton_inner(
        strength_out,
        strength_in,
        target_cost,
        layers,
        opts,
        &mode,
        n,
    )
}

#[inline]
fn pair_dist(mode: &CostMode<'_>, i: usize, j: usize) -> f64 {
    match mode {
        CostMode::NoCost => 0.0,
        CostMode::Coordinates { x, y } => coord_distance(x, y, i, j),
        CostMode::Sparse { col_costs, .. } => {
            // Find distance for pair (i,j) from col_costs[j]
            for &(src, d) in &col_costs[j] {
                if src == i {
                    return d;
                }
            }
            0.0
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn fit_w_newton_inner(
    strength_out: &[f64],
    strength_in: &[f64],
    target_cost: f64,
    layers: u32,
    opts: &CostFitOptions,
    cost_mode: &CostMode<'_>,
    n: usize,
) -> StrengthCostFitResult {
    let total: f64 = strength_out.iter().sum();

    // Initialize: r_avg = M * N_pairs / T, split between a and b.
    let n_pairs = if opts.self_loops { n * n } else { n * (n - 1) };
    let avg_weight = total / n_pairs.max(1) as f64;
    let r_avg = (f64::from(layers) / avg_weight.max(1e-10)).clamp(0.1, 10.0);
    let mut a: Vec<f64> = strength_out
        .iter()
        .map(|&s| {
            let share = (s / total.max(1.0)).max(1e-6);
            (r_avg * 0.5 * (1.0 / (share * n as f64)).sqrt()).clamp(0.01, 20.0)
        })
        .collect();
    let mut b: Vec<f64> = strength_in
        .iter()
        .map(|&s| {
            let share = (s / total.max(1.0)).max(1e-6);
            (r_avg * 0.5 * (1.0 / (share * n as f64)).sqrt()).clamp(0.01, 20.0)
        })
        .collect();
    let mut gamma = 0.0_f64;
    let damping = 0.8_f64;
    let r_min = 0.01_f64; // feasibility margin: all r_ij >= r_min

    for iter in 0..opts.max_iterations {
        // Update b_j: solve pred_in_j = s_in_j via Newton, project to feasibility
        for j in 0..n {
            if strength_in[j] <= 0.0 {
                continue;
            }
            let mut pred = 0.0;
            let mut dpred = 0.0;
            for i in 0..n {
                if !opts.self_loops && i == j {
                    continue;
                }
                let d = pair_dist(cost_mode, i, j);
                let r = (a[i] + b[j] + gamma * d).max(r_min);
                pred += w_mean(r, layers);
                dpred += w_mean_deriv(r, layers);
            }
            if dpred.abs() > 1e-15 {
                let step = -(pred - strength_in[j]) / dpred;
                let new_b = b[j] + damping * step;
                // Project: b_j >= r_min - min_i(a_i + gamma*d_ij)
                let min_complement: f64 = (0..n)
                    .filter(|&i| opts.self_loops || i != j)
                    .map(|i| a[i] + gamma * pair_dist(cost_mode, i, j))
                    .fold(f64::INFINITY, f64::min);
                b[j] = new_b.max(r_min - min_complement);
            }
        }

        // Update a_i: solve pred_out_i = s_out_i via Newton, project to feasibility
        for i in 0..n {
            if strength_out[i] <= 0.0 {
                continue;
            }
            let mut pred = 0.0;
            let mut dpred = 0.0;
            for j in 0..n {
                if !opts.self_loops && i == j {
                    continue;
                }
                let d = pair_dist(cost_mode, i, j);
                let r = (a[i] + b[j] + gamma * d).max(r_min);
                pred += w_mean(r, layers);
                dpred += w_mean_deriv(r, layers);
            }
            if dpred.abs() > 1e-15 {
                let step = -(pred - strength_out[i]) / dpred;
                let new_a = a[i] + damping * step;
                let min_complement: f64 = (0..n)
                    .filter(|&j| opts.self_loops || i != j)
                    .map(|j| b[j] + gamma * pair_dist(cost_mode, i, j))
                    .fold(f64::INFINITY, f64::min);
                a[i] = new_a.max(r_min - min_complement);
            }
        }

        // Update gamma (skip if no cost constraint)
        if matches!(cost_mode, CostMode::NoCost) {
            // No cost constraint; gamma stays at 0
        } else {
            let mut pred_cost = 0.0;
            let mut dpred_cost = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if !opts.self_loops && i == j {
                        continue;
                    }
                    let d = pair_dist(cost_mode, i, j);
                    let r = (a[i] + b[j] + gamma * d).max(r_min);
                    pred_cost += d * w_mean(r, layers);
                    dpred_cost += d * d * w_mean_deriv(r, layers);
                }
            }
            if dpred_cost.abs() > 1e-15 {
                let step = -(pred_cost - target_cost) / dpred_cost;
                let new_gamma = gamma + damping * step;
                // gamma lower bound: for all (i,j), a_i + b_j + gamma*d_ij >= r_min
                // gamma >= (r_min - a_i - b_j) / d_ij for all pairs with d_ij > 0
                let mut gamma_lb = f64::NEG_INFINITY;
                for i in 0..n {
                    for j in 0..n {
                        if !opts.self_loops && i == j {
                            continue;
                        }
                        let d = pair_dist(cost_mode, i, j);
                        if d > 1e-15 {
                            gamma_lb = gamma_lb.max((r_min - a[i] - b[j]) / d);
                        }
                    }
                }
                gamma = new_gamma.max(gamma_lb);
            }
        }

        // Convergence check
        let mut max_err = 0.0_f64;
        for i in 0..n {
            let mut p = 0.0;
            for j in 0..n {
                if !opts.self_loops && i == j {
                    continue;
                }
                let d = pair_dist(cost_mode, i, j);
                let r = (a[i] + b[j] + gamma * d).max(r_min);
                p += w_mean(r, layers);
            }
            max_err = max_err.max((p - strength_out[i]).abs());
        }
        for j in 0..n {
            let mut p = 0.0;
            for i in 0..n {
                if !opts.self_loops && i == j {
                    continue;
                }
                let d = pair_dist(cost_mode, i, j);
                let r = (a[i] + b[j] + gamma * d).max(r_min);
                p += w_mean(r, layers);
            }
            max_err = max_err.max((p - strength_in[j]).abs());
        }

        if max_err < opts.tolerance {
            let x: Vec<f64> = a.iter().map(|&ai| (-ai).exp()).collect();
            let y: Vec<f64> = b.iter().map(|&bj| (-bj).exp()).collect();
            return StrengthCostFitResult {
                x,
                y,
                gamma,
                converged: true,
                iterations: iter + 1,
            };
        }
    }

    let x: Vec<f64> = a.iter().map(|&ai| (-ai).exp()).collect();
    let y: Vec<f64> = b.iter().map(|&bj| (-bj).exp()).collect();
    StrengthCostFitResult {
        x,
        y,
        gamma,
        converged: false,
        iterations: opts.max_iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::super::me::fit_strength_cost_poisson_coordinates;
    use super::*;

    #[test]
    fn w_newton_recovers_strengths_small() {
        let s_out = vec![10.0, 20.0, 30.0];
        let s_in = vec![15.0, 25.0, 20.0];
        let cx = vec![0.0, 3.0, 0.0];
        let cy = vec![0.0, 0.0, 4.0];
        let target_cost = 80.0;
        let opts = CostFitOptions {
            self_loops: true,
            tolerance: 1e-2,
            max_iterations: 5000,
        };
        let fit = fit_strength_cost_w_lbfgs(&s_out, &s_in, &cx, &cy, target_cost, 1, &opts);
        eprintln!(
            "W Newton: converged={} iters={} gamma={}",
            fit.converged, fit.iterations, fit.gamma
        );
        eprintln!("  max_err after {} iters", fit.iterations);
        assert!(
            fit.converged,
            "W Newton did not converge, gamma={}",
            fit.gamma
        );
    }

    #[test]
    fn w_newton_differs_from_me() {
        let s_out = vec![10.0, 20.0, 30.0];
        let s_in = vec![15.0, 25.0, 20.0];
        let cx = vec![0.0, 3.0, 0.0];
        let cy = vec![0.0, 0.0, 4.0];
        let target_cost = 100.0;
        let opts = CostFitOptions {
            self_loops: true,
            tolerance: 1e-2,
            max_iterations: 5000,
        };
        let me = fit_strength_cost_poisson_coordinates(&s_out, &s_in, &cx, &cy, target_cost, &opts);
        let w = fit_strength_cost_w_lbfgs(&s_out, &s_in, &cx, &cy, target_cost, 1, &opts);
        assert!(me.converged, "ME did not converge");
        assert!(w.converged, "W Newton did not converge");
        let diff = (me.gamma - w.gamma).abs();
        assert!(
            diff > 1e-4,
            "ME and W should differ: me={} w={}",
            me.gamma,
            w.gamma
        );
    }
    #[test]
    fn w_newton_fixed_strength_no_cost() {
        let s_out = vec![10.0, 20.0, 30.0];
        let s_in = vec![15.0, 25.0, 20.0];
        let opts = CostFitOptions {
            self_loops: true,
            tolerance: 1e-2,
            max_iterations: 5000,
        };
        let fit = fit_strength_w_newton(&s_out, &s_in, 1, &opts);
        assert!(fit.converged, "W fixed-strength Newton did not converge");
        assert!((fit.gamma).abs() < 1e-10, "gamma should be 0 for no-cost");
        // Verify strengths
        let n = 3;
        for (i, &expected) in s_out.iter().enumerate() {
            let mut pred = 0.0;
            for j in 0..n {
                let r = (-fit.x[i].ln()) + (-fit.y[j].ln());
                pred += w_mean(r.max(1e-10), 1);
            }
            assert!(
                (pred - expected).abs() < 0.5,
                "s_out[{i}]: expected {expected}, got {pred}"
            );
        }
    }

    #[test]
    fn w_newton_sparse_cost_matches_coordinate() {
        let s_out = vec![10.0, 20.0, 30.0];
        let s_in = vec![15.0, 25.0, 20.0];
        let cx = vec![0.0, 3.0, 0.0];
        let cy = vec![0.0, 0.0, 4.0];
        let n = 3;
        let target_cost = 80.0;
        let opts = CostFitOptions {
            self_loops: true,
            tolerance: 1e-2,
            max_iterations: 5000,
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
        let sparse = fit_strength_cost_w_sparse_newton(
            &s_out,
            &s_in,
            &sources,
            &targets,
            &values,
            target_cost,
            1,
            &opts,
        );
        let coord = fit_strength_cost_w_lbfgs(&s_out, &s_in, &cx, &cy, target_cost, 1, &opts);
        assert!(sparse.converged, "sparse W Newton did not converge");
        assert!(coord.converged, "coord W Newton did not converge");
        assert!(
            (sparse.gamma - coord.gamma).abs() < 0.1,
            "sparse={} coord={}",
            sparse.gamma,
            coord.gamma
        );
    }
}
