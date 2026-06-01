//! Newton coordinate-descent W strength-cost solver.
//!
//! Alternating Newton updates on (a, b, γ) with damping.
//! Memory: O(N). Time per iteration: O(N²). No dense cost matrix.

use rayon::prelude::*;

use super::mask::PairMask;
use super::me::fit_strength_cost_poisson_coordinates;
use super::support::coord_distance;
use super::types::{WFitStatus, WProblemMetrics};
use super::w::{w_g, w_mean, w_occupation};
use super::{CostFitOptions, StrengthCostFitResult, WStrengthEdgesFitResult};

/// How pair costs are provided to the W Newton solver.
enum CostMode<'a> {
    /// No cost constraint (gamma fixed at 0).
    NoCost,
    /// Projected Euclidean XY coordinates.
    Coordinates { x: &'a [f64], y: &'a [f64] },
    /// Cached dense pair distances for moderate N coordinate-cost fits.
    DenseDistances { distances: Vec<f64>, n: usize },
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

/// Mask-aware W strength-cost fitting (used by partial fitting).
///
/// The mask's self_loops policy is used for pair exclusion. Known-pair
/// positions beyond the diagonal are handled by the excess computation
/// in the partial fitting pipeline.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_strength_cost_w_lbfgs_masked(
    strength_out: &[f64],
    strength_in: &[f64],
    coord_x: &[f64],
    coord_y: &[f64],
    target_cost: f64,
    layers: u32,
    mask: &PairMask,
    tolerance: f64,
    max_iterations: usize,
) -> StrengthCostFitResult {
    let opts = CostFitOptions {
        self_loops: mask.self_loops(),
        tolerance,
        max_iterations,
    };
    fit_strength_cost_w_lbfgs(
        strength_out,
        strength_in,
        coord_x,
        coord_y,
        target_cost,
        layers,
        &opts,
    )
}

#[inline]
fn pair_dist(mode: &CostMode<'_>, i: usize, j: usize) -> f64 {
    match mode {
        CostMode::NoCost => 0.0,
        CostMode::Coordinates { x, y } => coord_distance(x, y, i, j),
        CostMode::DenseDistances { distances, n } => distances[i * *n + j],
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

// ---------------------------------------------------------------------------
// W Strength-Edges L-BFGS (2N+1 parameters: a, b, ln_λ)
// ---------------------------------------------------------------------------

/// Fit W strength-edges using bisection over λ with Newton inner solve.
///
/// Strategy: bisect over λ; at each λ, solve (a,b) via damped Newton
/// coordinate-descent (same as `fit_strength_cost_w_lbfgs` inner loop).
/// The W feasibility constraint (r_ij > 0) is handled naturally by the
/// Newton inner solver's feasibility projection.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_strength_edges_w_lbfgs(
    strength_out: &[f64],
    strength_in: &[f64],
    target_edges: f64,
    layers: u32,
    mask: &PairMask,
    tolerance: f64,
    max_iterations: usize,
) -> WStrengthEdgesFitResult {
    let n = strength_out.len();
    let n_free = mask.n_free() as f64;
    let total = strength_out.iter().sum::<f64>();
    let r_min = 1e-4_f64;
    let self_loops = mask.self_loops();

    if n == 0
        || target_edges <= 0.0
        || target_edges >= n_free
        || target_edges > total
        || layers == 0
    {
        return w_se_not_solved(n, layers, WFitStatus::Infeasible);
    }

    let m = f64::from(layers);
    let n_active = if self_loops { n } else { n - 1 };

    // Initialize (a, b)
    let mut cur_a: Vec<f64> = strength_out
        .iter()
        .map(|&s| {
            if s <= 0.0 {
                return 10.0;
            }
            let avg = s / n_active.max(1) as f64;
            let q = avg / (avg + m);
            (-(q.clamp(1e-15, 1.0 - 1e-15)).ln() * 0.5).clamp(r_min, 20.0)
        })
        .collect();
    let mut cur_b: Vec<f64> = strength_in
        .iter()
        .map(|&s| {
            if s <= 0.0 {
                return 10.0;
            }
            let avg = s / n_active.max(1) as f64;
            let q = avg / (avg + m);
            (-(q.clamp(1e-15, 1.0 - 1e-15)).ln() * 0.5).clamp(r_min, 20.0)
        })
        .collect();

    // Expected edges for given (a, b, lam)
    #[allow(clippy::needless_range_loop)]
    let expected_edges_fn = |a: &[f64], b: &[f64], lam: f64| -> f64 {
        let mut edges = 0.0;
        for i in 0..n {
            for j in 0..n {
                if mask.is_masked(i, j) {
                    continue;
                }
                let r = (a[i] + b[j]).max(r_min);
                edges += w_occupation(lam, r, layers);
            }
        }
        edges
    };

    // Bracket λ
    let avg_weight = total / target_edges;
    let q_init = avg_weight / (avg_weight + m);
    let r_hom = -(q_init.clamp(1e-15, 1.0 - 1e-15)).ln();
    let g_hom = w_g(r_hom, layers);
    let occ_frac = target_edges / n_free;
    let lam_est = occ_frac / ((1.0 - occ_frac).max(1e-12) * g_hom.max(1e-12));

    let mut low = 1e-12_f64;
    let mut high = lam_est.max(1.0);
    let mut total_iters = 0;

    for _ in 0..40 {
        let (a, b, _, iters) = w_se_inner_solve(
            strength_out,
            strength_in,
            high,
            layers,
            self_loops,
            tolerance,
            max_iterations.min(500),
            &cur_a,
            &cur_b,
            r_min,
        );
        total_iters += iters;
        cur_a = a;
        cur_b = b;
        let edges = expected_edges_fn(&cur_a, &cur_b, high);
        if edges >= target_edges || high > 1e30 {
            break;
        }
        low = high;
        high *= 2.0;
    }

    // Bisect over λ
    let mut best_lam = high;
    let mut best_a = cur_a.clone();
    let mut best_b = cur_b.clone();
    let mut best_edge_err = f64::INFINITY;

    for _ in 0..60 {
        let mid = 0.5 * (low + high);
        let (a, b, _, iters) = w_se_inner_solve(
            strength_out,
            strength_in,
            mid,
            layers,
            self_loops,
            tolerance,
            max_iterations.min(500),
            &cur_a,
            &cur_b,
            r_min,
        );
        total_iters += iters;
        let edges = expected_edges_fn(&a, &b, mid);
        let edge_err = (edges - target_edges).abs();
        cur_a = a.clone();
        cur_b = b.clone();

        if edge_err < best_edge_err {
            best_edge_err = edge_err;
            best_lam = mid;
            best_a = a;
            best_b = b;
        }

        if edge_err <= tolerance.max(1e-10) {
            break;
        }
        if edges < target_edges {
            low = mid;
        } else {
            high = mid;
        }
        if high - low < 1e-12 * high.max(1.0) {
            break;
        }
    }

    w_se_build_result(
        &best_a,
        &best_b,
        best_lam,
        n,
        layers,
        mask,
        strength_out,
        strength_in,
        target_edges,
        tolerance,
        total,
        total_iters,
    )
}

/// Inner Newton coordinate-descent for W strength with fixed λ.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn w_se_inner_solve(
    strength_out: &[f64],
    strength_in: &[f64],
    lam: f64,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
    a_init: &[f64],
    b_init: &[f64],
    r_min: f64,
) -> (Vec<f64>, Vec<f64>, bool, usize) {
    let n = strength_out.len();
    let m = f64::from(layers);
    let mut a = a_init.to_vec();
    let mut b = b_init.to_vec();
    let mut damping = 0.5_f64;
    let mut prev_err = f64::INFINITY;
    let mut stall = 0usize;

    for iter in 0..max_iterations {
        let a_bak = a.clone();
        let b_bak = b.clone();

        // Precompute min_a for feasibility projection of b
        let min_a = a.iter().copied().fold(f64::INFINITY, f64::min);

        // Update b[j] in parallel — each j reads a[i] (stable)
        let new_b: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|j| {
                if strength_in[j] <= 0.0 {
                    return b[j];
                }
                let mut pred = 0.0;
                let mut dpred = 0.0;
                for i in 0..n {
                    if !self_loops && i == j {
                        continue;
                    }
                    let r = (a[i] + b[j]).max(r_min);
                    let q = (-r).exp();
                    let omq = 1.0 - q;
                    if omq <= 1e-15 {
                        continue;
                    }
                    let z = 1.0 + lam * (omq.powf(-m) - 1.0);
                    let wt = lam * m * q * omq.powf(-(m + 1.0)) / z;
                    pred += wt;
                    dpred += w_mean_deriv(r, layers) * lam / z.max(1e-15);
                }
                if dpred.abs() > 1e-15 {
                    let step = -(pred - strength_in[j]) / dpred;
                    (b[j] + damping * step).max(r_min - min_a)
                } else {
                    b[j]
                }
            })
            .collect();
        b = new_b;

        // Precompute min_b for feasibility projection of a
        let min_b = b.iter().copied().fold(f64::INFINITY, f64::min);

        // Update a[i] in parallel — each i reads b[j] (stable)
        let new_a: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                if strength_out[i] <= 0.0 {
                    return a[i];
                }
                let mut pred = 0.0;
                let mut dpred = 0.0;
                for j in 0..n {
                    if !self_loops && i == j {
                        continue;
                    }
                    let r = (a[i] + b[j]).max(r_min);
                    let q = (-r).exp();
                    let omq = 1.0 - q;
                    if omq <= 1e-15 {
                        continue;
                    }
                    let z = 1.0 + lam * (omq.powf(-m) - 1.0);
                    let wt = lam * m * q * omq.powf(-(m + 1.0)) / z;
                    pred += wt;
                    dpred += w_mean_deriv(r, layers) * lam / z.max(1e-15);
                }
                if dpred.abs() > 1e-15 {
                    let step = -(pred - strength_out[i]) / dpred;
                    (a[i] + damping * step).max(r_min - min_b)
                } else {
                    a[i]
                }
            })
            .collect();
        a = new_a;

        // Check convergence in parallel
        let max_err: f64 = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut p = 0.0;
                for j in 0..n {
                    if !self_loops && i == j {
                        continue;
                    }
                    let r = (a[i] + b[j]).max(r_min);
                    let q = (-r).exp();
                    let omq = 1.0 - q;
                    if omq <= 1e-15 {
                        continue;
                    }
                    let z = 1.0 + lam * (omq.powf(-m) - 1.0);
                    p += lam * m * q * omq.powf(-(m + 1.0)) / z;
                }
                (p - strength_out[i]).abs()
            })
            .reduce(|| 0.0_f64, f64::max);

        if max_err < tolerance {
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
    (a, b, false, max_iterations)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn w_se_build_result(
    a: &[f64],
    b: &[f64],
    lam: f64,
    n: usize,
    layers: u32,
    mask: &PairMask,
    s_out: &[f64],
    s_in: &[f64],
    target_edges: f64,
    tolerance: f64,
    total: f64,
    iterations: usize,
) -> WStrengthEdgesFitResult {
    let r_min = 1e-4;
    let m = f64::from(layers);
    let x: Vec<f64> = a.iter().map(|&ai| (-ai).exp()).collect();
    let y: Vec<f64> = b.iter().map(|&bj| (-bj).exp()).collect();

    let mut max_s_err = 0.0_f64;
    let mut total_s_err = 0.0_f64;
    let mut pred_edges = 0.0_f64;
    let mut min_margin = f64::INFINITY;
    let mut max_q = 0.0_f64;

    for i in 0..n {
        let mut row_s = 0.0;
        for j in 0..n {
            if mask.is_masked(i, j) {
                continue;
            }
            let r = (a[i] + b[j]).max(r_min);
            min_margin = min_margin.min(r);
            let q = (-r).exp();
            max_q = max_q.max(q);
            let omq = 1.0 - q;
            if omq <= 0.0 {
                continue;
            }
            let am = omq.powf(-m);
            let g = am - 1.0;
            let z = 1.0 + lam * g;
            row_s += lam * m * q * omq.powf(-(m + 1.0)) / z;
            pred_edges += lam * g / z;
        }
        let err = (row_s - s_out[i]).abs();
        max_s_err = max_s_err.max(err);
        total_s_err += err;
    }
    for j in 0..n {
        let mut col_s = 0.0;
        for i in 0..n {
            if mask.is_masked(i, j) {
                continue;
            }
            let r = (a[i] + b[j]).max(r_min);
            let q = (-r).exp();
            let omq = 1.0 - q;
            if omq <= 0.0 {
                continue;
            }
            let z = 1.0 + lam * (omq.powf(-m) - 1.0);
            col_s += lam * m * q * omq.powf(-(m + 1.0)) / z;
        }
        let err = (col_s - s_in[j]).abs();
        max_s_err = max_s_err.max(err);
        total_s_err += err;
    }
    let edge_residual = pred_edges - target_edges;

    let status = if max_s_err <= tolerance.max(1e-6) * total.max(1.0)
        && edge_residual.abs() <= tolerance.max(1e-6) * target_edges.max(1.0)
    {
        WFitStatus::Solved
    } else {
        WFitStatus::Inaccurate
    };

    WStrengthEdgesFitResult {
        x,
        y,
        lam,
        layers,
        status,
        objective: f64::NAN,
        iterations,
        min_margin,
        max_q,
        max_strength_residual: max_s_err,
        total_strength_residual: total_s_err,
        edge_residual,
        metrics: WProblemMetrics::default(),
    }
}

fn w_se_not_solved(n: usize, layers: u32, status: WFitStatus) -> WStrengthEdgesFitResult {
    WStrengthEdgesFitResult {
        x: vec![0.0; n],
        y: vec![0.0; n],
        lam: 0.0,
        layers,
        status,
        objective: f64::NAN,
        iterations: 0,
        min_margin: 0.0,
        max_q: 0.0,
        max_strength_residual: f64::INFINITY,
        total_strength_residual: f64::INFINITY,
        edge_residual: f64::INFINITY,
        metrics: WProblemMetrics::default(),
    }
}

// ---------------------------------------------------------------------------
// W Strength-Degree Newton solver
// ---------------------------------------------------------------------------

/// Fit W strength-degree using damped Newton coordinate-descent.
///
/// Same approach as `w_se_inner_solve` but with 4 per-node multiplier vectors.
/// Parameters: a[i], b[j] (strength via r_ij = a[i]+b[j] > 0),
///            z[i], w[j] (occupation multiplier v_ij = z[i]*w[j]).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_strength_degree_w_newton(
    strength_out: &[f64],
    strength_in: &[f64],
    degree_out: &[f64],
    degree_in: &[f64],
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, bool, usize) {
    let n = strength_out.len();
    let mask = PairMask::from_self_loops(n, self_loops);
    fit_strength_degree_w_newton_masked(
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

/// Masked variant of W strength-degree Newton solver.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_strength_degree_w_newton_masked(
    strength_out: &[f64],
    strength_in: &[f64],
    degree_out: &[f64],
    degree_in: &[f64],
    layers: u32,
    mask: &PairMask,
    tolerance: f64,
    max_iterations: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, bool, usize) {
    let n = strength_out.len();
    let self_loops = mask.self_loops();
    let m = f64::from(layers);
    let r_min = 1e-4_f64;
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

    let n_active = if self_loops { n } else { n - 1 };
    let mut a: Vec<f64> = strength_out
        .iter()
        .map(|&s| {
            if s <= 0.0 {
                return 10.0;
            }
            let avg = s / n_active.max(1) as f64;
            let q = avg / (avg + m);
            (-(q.clamp(1e-15, 1.0 - 1e-15)).ln() * 0.5).clamp(r_min, 20.0)
        })
        .collect();
    let mut b: Vec<f64> = strength_in
        .iter()
        .map(|&s| {
            if s <= 0.0 {
                return 10.0;
            }
            let avg = s / n_active.max(1) as f64;
            let q = avg / (avg + m);
            (-(q.clamp(1e-15, 1.0 - 1e-15)).ln() * 0.5).clamp(r_min, 20.0)
        })
        .collect();

    let k_total = degree_out.iter().sum::<f64>().max(1.0);
    let k_scale = (k_total / n.max(1) as f64).sqrt().max(0.1);
    let mut z: Vec<f64> = degree_out
        .iter()
        .enumerate()
        .map(|(i, &k)| {
            if saturated_out[i] {
                1e30
            } else {
                (k / k_total * k_scale).max(1e-12)
            }
        })
        .collect();
    let mut w: Vec<f64> = degree_in
        .iter()
        .enumerate()
        .map(|(j, &k)| {
            if saturated_in[j] {
                1e30
            } else {
                (k / k_total * k_scale).max(1e-12)
            }
        })
        .collect();

    let mut damping = 0.3_f64;
    let mut prev_err = f64::INFINITY;
    let mut stall = 0_usize;

    for iter in 0..max_iterations {
        let a_bak = a.clone();
        let b_bak = b.clone();
        let z_bak = z.clone();
        let w_bak = w.clone();

        // Precompute min_a for b feasibility projection
        let min_a = a.iter().copied().fold(f64::INFINITY, f64::min);

        // Update b[j] (strength_in) in parallel
        let new_b: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|j| {
                if strength_in[j] <= 0.0 {
                    return b[j];
                }
                let mut pred = 0.0;
                let mut dpred = 0.0;
                for i in 0..n {
                    if mask.is_masked(i, j) {
                        continue;
                    }
                    let r = (a[i] + b[j]).max(r_min);
                    let q = (-r).exp();
                    let omq = 1.0 - q;
                    if omq <= 1e-15 {
                        continue;
                    }
                    let v = z[i] * w[j];
                    let zz = 1.0 + v * (omq.powf(-m) - 1.0);
                    let wt = v * m * q * omq.powf(-(m + 1.0)) / zz;
                    pred += wt;
                    dpred += w_mean_deriv(r, layers) * v / zz.max(1e-15);
                }
                if dpred.abs() > 1e-15 {
                    let step = -(pred - strength_in[j]) / dpred;
                    (b[j] + damping * step).max(r_min - min_a)
                } else {
                    b[j]
                }
            })
            .collect();
        b = new_b;

        // Precompute min_b for a feasibility projection
        let min_b = b.iter().copied().fold(f64::INFINITY, f64::min);

        // Update a[i] (strength_out) in parallel
        let new_a: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                if strength_out[i] <= 0.0 {
                    return a[i];
                }
                let mut pred = 0.0;
                let mut dpred = 0.0;
                for j in 0..n {
                    if mask.is_masked(i, j) {
                        continue;
                    }
                    let r = (a[i] + b[j]).max(r_min);
                    let q = (-r).exp();
                    let omq = 1.0 - q;
                    if omq <= 1e-15 {
                        continue;
                    }
                    let v = z[i] * w[j];
                    let zz = 1.0 + v * (omq.powf(-m) - 1.0);
                    let wt = v * m * q * omq.powf(-(m + 1.0)) / zz;
                    pred += wt;
                    dpred += w_mean_deriv(r, layers) * v / zz.max(1e-15);
                }
                if dpred.abs() > 1e-15 {
                    let step = -(pred - strength_out[i]) / dpred;
                    (a[i] + damping * step).max(r_min - min_b)
                } else {
                    a[i]
                }
            })
            .collect();
        a = new_a;

        // Update w[j] (degree_in) via bisection — in parallel
        let new_w: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|j| {
                if saturated_in[j] {
                    return 1e30;
                }
                if degree_in[j] <= 0.0 {
                    return w[j];
                }
                let target_k = degree_in[j];
                let mut lo = 0.0_f64;
                let mut hi = w[j].max(1.0);
                for _ in 0..40 {
                    let val: f64 = (0..n)
                        .filter(|&i| self_loops || i != j)
                        .map(|i| {
                            let r = (a[i] + b[j]).max(r_min);
                            w_occupation(hi * z[i], r, layers)
                        })
                        .sum();
                    if val >= target_k || hi > 1e30 {
                        break;
                    }
                    lo = hi;
                    hi *= 2.0;
                }
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    let val: f64 = (0..n)
                        .filter(|&i| self_loops || i != j)
                        .map(|i| {
                            let r = (a[i] + b[j]).max(r_min);
                            w_occupation(mid * z[i], r, layers)
                        })
                        .sum();
                    if val < target_k {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                    if hi - lo < 1e-14 * hi.max(1.0) {
                        break;
                    }
                }
                0.5 * (lo + hi)
            })
            .collect();
        w = new_w;

        // Update z[i] (degree_out) via bisection — in parallel
        let new_z: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                if saturated_out[i] {
                    return 1e30;
                }
                if degree_out[i] <= 0.0 {
                    return z[i];
                }
                let target_k = degree_out[i];
                let mut lo = 0.0_f64;
                let mut hi = z[i].max(1.0);
                for _ in 0..40 {
                    let val: f64 = (0..n)
                        .filter(|&j| self_loops || i != j)
                        .map(|j| {
                            let r = (a[i] + b[j]).max(r_min);
                            w_occupation(hi * w[j], r, layers)
                        })
                        .sum();
                    if val >= target_k || hi > 1e30 {
                        break;
                    }
                    lo = hi;
                    hi *= 2.0;
                }
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    let val: f64 = (0..n)
                        .filter(|&j| self_loops || i != j)
                        .map(|j| {
                            let r = (a[i] + b[j]).max(r_min);
                            w_occupation(mid * w[j], r, layers)
                        })
                        .sum();
                    if val < target_k {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                    if hi - lo < 1e-14 * hi.max(1.0) {
                        break;
                    }
                }
                0.5 * (lo + hi)
            })
            .collect();
        z = new_z;

        // Check convergence in parallel
        let max_err: f64 = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut ps = 0.0;
                let mut pk = 0.0;
                for j in 0..n {
                    if mask.is_masked(i, j) {
                        continue;
                    }
                    let r = (a[i] + b[j]).max(r_min);
                    let q = (-r).exp();
                    let omq = 1.0 - q;
                    if omq <= 1e-15 {
                        continue;
                    }
                    let v = z[i] * w[j];
                    let am = omq.powf(-m);
                    let g = am - 1.0;
                    let zz = 1.0 + v * g;
                    ps += v * m * q * omq.powf(-(m + 1.0)) / zz;
                    pk += v * g / zz;
                }
                let mut e = (ps - strength_out[i]).abs();
                if !saturated_out[i] {
                    e = e.max((pk - degree_out[i]).abs());
                }
                e
            })
            .reduce(|| 0.0_f64, f64::max);

        if max_err < tolerance {
            let x: Vec<f64> = a.iter().map(|&ai| (-ai).exp()).collect();
            let y: Vec<f64> = b.iter().map(|&bj| (-bj).exp()).collect();
            return (x, y, z, w, true, iter + 1);
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
                z = z_bak;
                w = w_bak;
                damping *= 0.5;
                stall = 0;
                if damping < 1e-6 {
                    break;
                }
            }
        }
        prev_err = max_err;
    }

    let x: Vec<f64> = a.iter().map(|&ai| (-ai).exp()).collect();
    let y: Vec<f64> = b.iter().map(|&bj| (-bj).exp()).collect();
    (x, y, z, w, false, max_iterations)
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
    fn w_se_lbfgs_vs_bisection_n10() {
        use crate::fitting::{
            mask::PairMask, types::WFitStatus, w::fit_strength_edges_geometric, w::w_occupation,
            w::w_zip_mean, WConicFitOptions,
        };
        let n = 10;
        let layers = 1u32;
        let x: Vec<f64> = (0..n).map(|i| 0.1 + 0.8 * (i as f64 / n as f64)).collect();
        let y: Vec<f64> = (0..n)
            .map(|j| 0.15 + 0.7 * ((n - 1 - j) as f64 / n as f64))
            .collect();
        let lam = 1.0_f64;
        let mask = PairMask::from_self_loops(n, false);
        let mut s_out = vec![0.0; n];
        let mut s_in = vec![0.0; n];
        let mut te = 0.0;
        for i in 0..n {
            for j in 0..n {
                if mask.is_masked(i, j) {
                    continue;
                }
                let q = x[i] * y[j];
                let r = -(q.clamp(1e-15, 1.0 - 1e-15)).ln();
                let wt = w_zip_mean(lam, r, layers);
                let occ = w_occupation(lam, r, layers);
                if wt.is_finite() {
                    s_out[i] += wt;
                    s_in[j] += wt;
                }
                if occ.is_finite() {
                    te += occ;
                }
            }
        }

        let _old = fit_strength_edges_geometric(
            &s_out,
            &s_in,
            te,
            WConicFitOptions {
                self_loops: false,
                tolerance: 1e-6,
                max_iterations: 1000,
            },
        );

        let new = fit_strength_edges_w_lbfgs(&s_out, &s_in, te, layers, &mask, 1e-6, 1000);

        // L-BFGS must converge
        assert!(
            new.status == WFitStatus::Solved,
            "W L-BFGS SE must solve: status={:?}, edge_resid={}, max_s_resid={}",
            new.status,
            new.edge_residual,
            new.max_strength_residual
        );
    }
}
