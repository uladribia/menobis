//! Newton coordinate-descent W strength-cost solver.
//!
//! Alternating Newton updates on (a, b, γ) with damping.
//! Memory: O(N). Time per iteration: O(N²). No dense cost matrix.

use super::me::fit_strength_cost_poisson_coordinates;
use super::support::coord_distance;
use super::w::w_mean;
use super::{CostFitOptions, StrengthCostFitResult};

/// How pair costs are provided to the W Newton solver.
#[allow(dead_code)]
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
    /// Cached dense pair distances for moderate N coordinate-cost fits.
    DenseDistances { distances: Vec<f64>, n: usize },
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
    fit_w_newton_inner(
        strength_out,
        strength_in,
        0.0,
        layers,
        opts,
        &CostMode::NoCost,
        n,
        0.0,
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
    // No ME bootstrap for sparse (would need sparse ME solver)
    fit_w_newton_inner(
        strength_out,
        strength_in,
        target_cost,
        layers,
        opts,
        &mode,
        n,
        0.0,
    )
}

/// Fit W(M) strength-cost with projected coordinates using Newton coordinate-descent.
/// Fit W(M) strength-cost with projected Euclidean XY coordinates.
///
/// Strategy: bisect over gamma (cost as objective); at each gamma,
/// solve (a,b) via Newton coordinate-descent on strength constraints.
/// This separates the gamma search from the strength balancing.
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
    let r_min = 1e-4_f64;
    let m = f64::from(layers);
    let n_active = if opts.self_loops { n } else { n - 1 };
    let coord_mode = if n * n * 8 <= 256 * 1024 * 1024 {
        let mut distances = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                distances[i * n + j] = coord_distance(coord_x, coord_y, i, j);
            }
        }
        CostMode::DenseDistances { distances, n }
    } else {
        CostMode::Coordinates {
            x: coord_x,
            y: coord_y,
        }
    };

    // Initialize (a, b) per-node from individual strength targets
    let a_init: Vec<f64> = strength_out
        .iter()
        .map(|&s| {
            if s <= 0.0 {
                return 10.0;
            }
            let avg = s / n_active.max(1) as f64;
            let q = avg / (avg + m);
            (-(q.clamp(1e-15, 1.0 - 1e-15)).ln() * 0.5).clamp(1e-4, 20.0)
        })
        .collect();
    let b_init: Vec<f64> = strength_in
        .iter()
        .map(|&s| {
            if s <= 0.0 {
                return 10.0;
            }
            let avg = s / n_active.max(1) as f64;
            let q = avg / (avg + m);
            (-(q.clamp(1e-15, 1.0 - 1e-15)).ln() * 0.5).clamp(1e-4, 20.0)
        })
        .collect();

    // Inner solver: L-BFGS in log-space with Newton coordinate-descent fallback.
    let solve_at_gamma =
        |gamma: f64, ai: &[f64], bi: &[f64]| -> (Vec<f64>, Vec<f64>, bool, usize) {
            let lbfgs = solve_ab_lbfgs_fixed_gamma(
                strength_out,
                strength_in,
                layers,
                opts,
                &coord_mode,
                gamma,
                ai,
                bi,
                r_min,
            );
            if lbfgs.2 {
                return lbfgs;
            }

            let mut a = ai.to_vec();
            let mut b = bi.to_vec();
            let mut damping = 0.5_f64;
            let mut prev_err = f64::INFINITY;
            let mut stall = 0_usize;
            let inner_max = opts.max_iterations.min(2000);

            for iter in 0..inner_max {
                let a_bak = a.clone();
                let b_bak = b.clone();

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
                        let d = pair_dist(&coord_mode, i, j);
                        let r = (a[i] + b[j] + gamma * d).max(r_min);
                        pred += w_mean(r, layers);
                        dpred += w_mean_deriv(r, layers);
                    }
                    if dpred.abs() > 1e-15 {
                        let step = -(pred - strength_in[j]) / dpred;
                        let new_b = b[j] + damping * step;
                        let min_c: f64 = (0..n)
                            .filter(|&i| opts.self_loops || i != j)
                            .map(|i| a[i] + gamma * pair_dist(&coord_mode, i, j))
                            .fold(f64::INFINITY, f64::min);
                        b[j] = new_b.max(r_min - min_c);
                    }
                }

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
                        let d = pair_dist(&coord_mode, i, j);
                        let r = (a[i] + b[j] + gamma * d).max(r_min);
                        pred += w_mean(r, layers);
                        dpred += w_mean_deriv(r, layers);
                    }
                    if dpred.abs() > 1e-15 {
                        let step = -(pred - strength_out[i]) / dpred;
                        let new_a = a[i] + damping * step;
                        let min_c: f64 = (0..n)
                            .filter(|&j| opts.self_loops || i != j)
                            .map(|j| b[j] + gamma * pair_dist(&coord_mode, i, j))
                            .fold(f64::INFINITY, f64::min);
                        a[i] = new_a.max(r_min - min_c);
                    }
                }

                let mut max_err = 0.0_f64;
                for i in 0..n {
                    let mut p = 0.0;
                    for j in 0..n {
                        if !opts.self_loops && i == j {
                            continue;
                        }
                        let d = pair_dist(&coord_mode, i, j);
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
                        let d = pair_dist(&coord_mode, i, j);
                        let r = (a[i] + b[j] + gamma * d).max(r_min);
                        p += w_mean(r, layers);
                    }
                    max_err = max_err.max((p - strength_in[j]).abs());
                }

                if max_err < opts.tolerance {
                    return (a, b, true, iter + 1);
                }
                if max_err < prev_err * 0.999 {
                    stall = 0;
                    if damping < 0.9 {
                        damping = (damping * 1.05).min(0.9);
                    }
                } else {
                    stall += 1;
                    if stall >= 3 {
                        a = a_bak;
                        b = b_bak;
                        damping *= 0.5;
                        stall = 0;
                        if damping < 1e-6 {
                            break;
                        }
                    }
                }
                prev_err = max_err;
            }
            (a, b, false, inner_max)
        };

    // Expected cost given (a, b, gamma)
    let expected_cost_fn = |a: &[f64], b: &[f64], gamma: f64| -> f64 {
        let mut cost = 0.0;
        for i in 0..n {
            for j in 0..n {
                if !opts.self_loops && i == j {
                    continue;
                }
                let d = pair_dist(&coord_mode, i, j);
                let r = (a[i] + b[j] + gamma * d).max(r_min);
                cost += d * w_mean(r, layers);
            }
        }
        cost
    };

    // Get ME gamma as initial bracket estimate
    let me_opts = CostFitOptions {
        self_loops: opts.self_loops,
        tolerance: opts.tolerance.max(1.0),
        max_iterations: 200,
    };
    let me_fit = fit_strength_cost_poisson_coordinates(
        strength_out,
        strength_in,
        coord_x,
        coord_y,
        target_cost,
        &me_opts,
    );
    let me_gamma = if me_fit.converged {
        me_fit.gamma.max(0.01)
    } else {
        0.1
    };

    // Solve at gamma=0 for baseline
    let (a0, b0, _, iters0) = solve_at_gamma(0.0, &a_init, &b_init);
    let cost_zero = expected_cost_fn(&a0, &b0, 0.0);
    let mut total_iters = iters0;

    if (cost_zero - target_cost).abs() <= opts.tolerance {
        let x: Vec<f64> = a0.iter().map(|&ai| (-ai).exp()).collect();
        let y: Vec<f64> = b0.iter().map(|&bj| (-bj).exp()).collect();
        return StrengthCostFitResult {
            x,
            y,
            gamma: 0.0,
            converged: true,
            iterations: total_iters,
        };
    }

    // Bracket: find high such that cost(high) < target
    let mut low = 0.0_f64;
    let mut high = me_gamma * 3.0;
    let mut cur_a = a0;
    let mut cur_b = b0;

    for _ in 0..20 {
        let (a, b, _, iters) = solve_at_gamma(high, &cur_a, &cur_b);
        total_iters += iters;
        let ch = expected_cost_fn(&a, &b, high);
        cur_a = a;
        cur_b = b;
        if ch <= target_cost {
            break;
        }
        low = high;
        high *= 2.0;
        if high > 100.0 {
            break;
        }
    }

    // Bisection over gamma
    let mut best_gamma = 0.0;
    let mut best_a = cur_a.clone();
    let mut best_b = cur_b.clone();
    let mut best_cost_err = f64::INFINITY;

    for _ in 0..60 {
        let mid = 0.5 * (low + high);
        let (a, b, _, iters) = solve_at_gamma(mid, &cur_a, &cur_b);
        total_iters += iters;
        let cost_mid = expected_cost_fn(&a, &b, mid);
        let cost_err = (cost_mid - target_cost).abs();
        cur_a = a.clone();
        cur_b = b.clone();

        if cost_err < best_cost_err {
            best_cost_err = cost_err;
            best_gamma = mid;
            best_a = a;
            best_b = b;
        }

        if cost_err <= opts.tolerance {
            break;
        }
        if cost_mid > target_cost {
            low = mid;
        } else {
            high = mid;
        }
        if high - low < 1e-10 * high.max(1.0) {
            break;
        }
    }

    // Final convergence check
    let mut max_err = 0.0_f64;
    for i in 0..n {
        let mut p = 0.0;
        for j in 0..n {
            if !opts.self_loops && i == j {
                continue;
            }
            let d = pair_dist(&coord_mode, i, j);
            let r = (best_a[i] + best_b[j] + best_gamma * d).max(r_min);
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
            let d = pair_dist(&coord_mode, i, j);
            let r = (best_a[i] + best_b[j] + best_gamma * d).max(r_min);
            p += w_mean(r, layers);
        }
        max_err = max_err.max((p - strength_in[j]).abs());
    }
    let converged = max_err < opts.tolerance && best_cost_err < opts.tolerance.max(1.0);

    let x: Vec<f64> = best_a.iter().map(|&ai| (-ai).exp()).collect();
    let y: Vec<f64> = best_b.iter().map(|&bj| (-bj).exp()).collect();
    StrengthCostFitResult {
        x,
        y,
        gamma: best_gamma,
        converged,
        iterations: total_iters,
    }
}

#[inline]
fn pair_dist(mode: &CostMode<'_>, i: usize, j: usize) -> f64 {
    match mode {
        CostMode::NoCost => 0.0,
        CostMode::Coordinates { x, y } => coord_distance(x, y, i, j),
        CostMode::DenseDistances { distances, n } => distances[i * *n + j],
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
fn solve_ab_lbfgs_fixed_gamma(
    strength_out: &[f64],
    strength_in: &[f64],
    layers: u32,
    opts: &CostFitOptions,
    cost_mode: &CostMode<'_>,
    gamma: f64,
    a_init: &[f64],
    b_init: &[f64],
    r_min: f64,
) -> (Vec<f64>, Vec<f64>, bool, usize) {
    if matches!(cost_mode, CostMode::Sparse { .. }) {
        return (a_init.to_vec(), b_init.to_vec(), false, 0);
    }
    let n = strength_out.len();
    let dim = 2 * n;
    let memory = 7_usize;
    let max_iter = opts.max_iterations.min(150);
    let mut z = Vec::with_capacity(dim);
    z.extend_from_slice(a_init);
    z.extend_from_slice(b_init);
    let mut grad = vec![0.0; dim];
    let mut obj = match w_dual_objective_gradient(
        &z,
        &mut grad,
        strength_out,
        strength_in,
        layers,
        opts,
        cost_mode,
        gamma,
        r_min,
    ) {
        Some(value) => value,
        None => return (a_init.to_vec(), b_init.to_vec(), false, 0),
    };
    let mut grad_norm = grad.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if grad_norm < opts.tolerance {
        return (z[..n].to_vec(), z[n..].to_vec(), true, 0);
    }

    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();

    for iter in 0..max_iter {
        let direction = lbfgs_direction(&grad, &s_hist, &y_hist, &rho_hist);
        let mut step = 1.0_f64;
        let directional_deriv: f64 = grad.iter().zip(direction.iter()).map(|(g, p)| g * p).sum();
        if directional_deriv >= 0.0 || !directional_deriv.is_finite() {
            break;
        }

        let mut accepted: Option<(Vec<f64>, Vec<f64>, f64)> = None;
        for _ in 0..30 {
            let candidate: Vec<f64> = z
                .iter()
                .zip(direction.iter())
                .map(|(zi, pi)| zi + step * pi)
                .collect();
            let mut candidate_grad = vec![0.0; dim];
            if let Some(candidate_obj) = w_dual_objective_gradient(
                &candidate,
                &mut candidate_grad,
                strength_out,
                strength_in,
                layers,
                opts,
                cost_mode,
                gamma,
                r_min,
            ) {
                if candidate_obj <= obj + 1e-4 * step * directional_deriv {
                    accepted = Some((candidate, candidate_grad, candidate_obj));
                    break;
                }
            }
            step *= 0.5;
            if step < 1e-12 {
                break;
            }
        }
        let Some((mut next_z, next_grad, next_obj)) = accepted else {
            break;
        };
        recenter_ab(&mut next_z, n);

        let s: Vec<f64> = next_z.iter().zip(z.iter()).map(|(a, b)| a - b).collect();
        let y_vec: Vec<f64> = next_grad
            .iter()
            .zip(grad.iter())
            .map(|(a, b)| a - b)
            .collect();
        let sy: f64 = s.iter().zip(y_vec.iter()).map(|(a, b)| a * b).sum();
        if sy > 1e-12 && sy.is_finite() {
            if s_hist.len() == memory {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y_vec);
            rho_hist.push(1.0 / sy);
        }
        z = next_z;
        grad = next_grad;
        obj = next_obj;
        grad_norm = grad.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if grad_norm < opts.tolerance {
            return (z[..n].to_vec(), z[n..].to_vec(), true, iter + 1);
        }
    }
    (z[..n].to_vec(), z[n..].to_vec(), false, max_iter)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn w_dual_objective_gradient(
    z: &[f64],
    grad: &mut [f64],
    strength_out: &[f64],
    strength_in: &[f64],
    layers: u32,
    opts: &CostFitOptions,
    cost_mode: &CostMode<'_>,
    gamma: f64,
    r_min: f64,
) -> Option<f64> {
    let n = strength_out.len();
    grad.fill(0.0);
    grad[..n].copy_from_slice(strength_out);
    grad[n..(n + n)].copy_from_slice(strength_in);
    let mut obj = 0.0_f64;
    let m = f64::from(layers);
    for i in 0..n {
        obj += strength_out[i] * z[i] + strength_in[i] * z[n + i];
    }
    for i in 0..n {
        for j in 0..n {
            if !opts.self_loops && i == j {
                continue;
            }
            let r = z[i] + z[n + j] + gamma * pair_dist(cost_mode, i, j);
            if !r.is_finite() || r <= r_min {
                return None;
            }
            let q = (-r).exp();
            let one_minus_q = 1.0 - q;
            if one_minus_q <= 0.0 || !one_minus_q.is_finite() {
                return None;
            }
            obj += -m * one_minus_q.ln();
            let mean = m * q / one_minus_q;
            grad[i] -= mean;
            grad[n + j] -= mean;
        }
    }
    if obj.is_finite() {
        Some(obj)
    } else {
        None
    }
}

fn lbfgs_direction(
    grad: &[f64],
    s_hist: &[Vec<f64>],
    y_hist: &[Vec<f64>],
    rho_hist: &[f64],
) -> Vec<f64> {
    let mut q = grad.to_vec();
    let mut alpha = vec![0.0; s_hist.len()];
    for idx in (0..s_hist.len()).rev() {
        alpha[idx] = rho_hist[idx]
            * s_hist[idx]
                .iter()
                .zip(q.iter())
                .map(|(s, qi)| s * qi)
                .sum::<f64>();
        for (qi, yi) in q.iter_mut().zip(y_hist[idx].iter()) {
            *qi -= alpha[idx] * yi;
        }
    }
    if let (Some(s_last), Some(y_last)) = (s_hist.last(), y_hist.last()) {
        let sy: f64 = s_last.iter().zip(y_last.iter()).map(|(s, y)| s * y).sum();
        let yy: f64 = y_last.iter().map(|y| y * y).sum();
        if yy > 0.0 {
            let scale = sy / yy;
            for qi in &mut q {
                *qi *= scale;
            }
        }
    }
    for idx in 0..s_hist.len() {
        let beta = rho_hist[idx]
            * y_hist[idx]
                .iter()
                .zip(q.iter())
                .map(|(y, qi)| y * qi)
                .sum::<f64>();
        for (qi, si) in q.iter_mut().zip(s_hist[idx].iter()) {
            *qi += si * (alpha[idx] - beta);
        }
    }
    q.into_iter().map(|v| -v).collect()
}

fn recenter_ab(z: &mut [f64], n: usize) {
    let mean_a = z[..n].iter().sum::<f64>() / n as f64;
    let mean_b = z[n..].iter().sum::<f64>() / n as f64;
    let shift = 0.5 * (mean_b - mean_a);
    for zi in &mut z[..n] {
        *zi += shift;
    }
    for zi in &mut z[n..] {
        *zi -= shift;
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
    init_gamma: f64,
) -> StrengthCostFitResult {
    let _total: f64 = strength_out.iter().sum();
    let n_active = if opts.self_loops { n } else { n - 1 };

    // Initialize: per-node r based on individual strength.
    // For node i with strength s_i over n_active pairs:
    //   avg w_mean per pair = s_i / n_active
    //   w_mean(r, M) = M*exp(-r)/(1-exp(-r)) ≈ M/r for small r
    //   => r_i ≈ M * n_active / s_i
    let m = f64::from(layers);
    let mut a: Vec<f64> = strength_out
        .iter()
        .map(|&s| {
            if s <= 0.0 {
                return 10.0;
            }
            let avg_per_pair = s / n_active.max(1) as f64;
            // Invert w_mean: w_mean(r,M)=M*q/(1-q) where q=exp(-r)
            // => q = avg/(avg+M), r = -ln(avg/(avg+M))
            let q = avg_per_pair / (avg_per_pair + m);
            let r = -(q.clamp(1e-15, 1.0 - 1e-15)).ln();
            // a_i is half of r (other half is b_j)
            (r * 0.5).clamp(1e-4, 20.0)
        })
        .collect();
    let mut b: Vec<f64> = strength_in
        .iter()
        .map(|&s| {
            if s <= 0.0 {
                return 10.0;
            }
            let avg_per_pair = s / n_active.max(1) as f64;
            let q = avg_per_pair / (avg_per_pair + m);
            let r = -(q.clamp(1e-15, 1.0 - 1e-15)).ln();
            (r * 0.5).clamp(1e-4, 20.0)
        })
        .collect();
    let mut gamma = init_gamma;
    let r_min = 1e-4_f64; // feasibility margin (much smaller than before)

    // Fast path for fixed-strength W/WNB: log-space L-BFGS with strict
    // feasibility line search. If it does not converge, fall back to the
    // robust coordinate Newton path below.
    if matches!(cost_mode, CostMode::NoCost) {
        let (lbfgs_a, lbfgs_b, ok, iters) = solve_ab_lbfgs_fixed_gamma(
            strength_out,
            strength_in,
            layers,
            opts,
            cost_mode,
            gamma,
            &a,
            &b,
            r_min,
        );
        if ok {
            let x: Vec<f64> = lbfgs_a.iter().map(|&ai| (-ai).exp()).collect();
            let y: Vec<f64> = lbfgs_b.iter().map(|&bj| (-bj).exp()).collect();
            return StrengthCostFitResult {
                x,
                y,
                gamma,
                converged: true,
                iterations: iters,
            };
        }
        a = lbfgs_a;
        b = lbfgs_b;
    }

    let mut damping = 0.5_f64; // start conservative
    let mut prev_max_err = f64::INFINITY;
    let mut stall_count = 0_usize;

    for iter in 0..opts.max_iterations {
        let a_before = a.clone();
        let b_before = b.clone();
        let gamma_before = gamma;

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
        if !matches!(cost_mode, CostMode::NoCost) {
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
                // gamma lower bound: a_i + b_j + gamma*d_ij >= r_min
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

        // Convergence: primary O(N) multiplier-delta check
        let max_delta = a
            .iter()
            .zip(a_before.iter())
            .chain(b.iter().zip(b_before.iter()))
            .map(|(&cur, &prev)| (cur - prev).abs())
            .fold(0.0_f64, f64::max)
            .max((gamma - gamma_before).abs());
        if max_delta < opts.tolerance * 1e-8 {
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

        // Periodic O(N²) residual check
        let check_residual = (iter + 1) % 10 == 0 || max_delta < opts.tolerance * 0.01;
        if !check_residual {
            // Skip expensive residual check, just update damping heuristic
            if max_delta < prev_max_err {
                stall_count = 0;
                if damping < 0.9 {
                    damping = (damping * 1.05).min(0.9);
                }
            } else {
                stall_count += 1;
                if stall_count >= 5 {
                    damping = (damping * 0.5).max(1e-4);
                    stall_count = 0;
                }
            }
            prev_max_err = max_delta;
            continue;
        }

        // Full convergence check
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

        // Adaptive damping with backtracking
        if max_err < prev_max_err * 0.999 {
            // Making progress: allow slightly more aggressive steps
            stall_count = 0;
            if damping < 0.9 {
                damping = (damping * 1.05).min(0.9);
            }
        } else {
            // Not making progress: revert and reduce damping
            stall_count += 1;
            if stall_count >= 3 {
                a = a_before;
                b = b_before;
                gamma = gamma_before;
                damping *= 0.5;
                stall_count = 0;
                if damping < 1e-6 {
                    // Cannot make progress at all
                    break;
                }
            }
        }
        prev_max_err = max_err;
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
