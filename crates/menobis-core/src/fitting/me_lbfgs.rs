//! L-BFGS solver for ME (Poisson) strength-degree (MEECM) fitting.
//!
//! Replaces the damped log-domain balancing approach with a direct minimization
//! of the Negative Log-Likelihood (NLL) of the Grand Canonical ensemble using
//! L-BFGS optimization.
//!
//! # Mathematical formulation
//!
//! For the Zero-Inflated Poisson (MEECM) model:
//! - Parameters: α_x[i], α_y[j] (log strength multipliers), α_v[i], α_w[j] (log degree multipliers)
//! - q_ij = exp(α_x[i] + α_y[j])  (Poisson rate)
//! - v_ij = exp(α_v[i] + α_w[j])  (zero-inflation multiplier)
//! - Z_ij = 1 + v_ij * (exp(q_ij) - 1)  (pair partition function)
//!
//! NLL objective (dual of max-entropy):
//!   f(θ) = Σ_{(i,j) free} ln(Z_ij) - Σ_i α_x[i]*s_out[i] - Σ_j α_y[j]*s_in[j]
//!          - Σ_i α_v[i]*k_out[i] - Σ_j α_w[j]*k_in[j]
//!
//! Gradients:
//!   ∂f/∂α_x[i] = Σ_j E[t_ij] - s_out[i]
//!   ∂f/∂α_y[j] = Σ_i E[t_ij] - s_in[j]
//!   ∂f/∂α_v[i] = Σ_j E[Θ(t_ij>0)] - k_out[i]
//!   ∂f/∂α_w[j] = Σ_i E[Θ(t_ij>0)] - k_in[j]
//!
//! # Memory
//!
//! O(N) per thread. No N×N dense matrices. Rayon-parallelized row accumulation.

use rayon::prelude::*;

use super::mask::PairMask;
use super::{StrengthDegreeFitResult, StrengthEdgesFitResult};

// ---------------------------------------------------------------------------
// Mass-Preserving Heuristic Regularization
// ---------------------------------------------------------------------------

/// Preprocessing: smooth boundary nodes to prevent parameter singularities.
///
/// For a Zero-Inflated Poisson model, boundary conditions force parameters
/// to ±∞:
/// - k_i ≈ s_i implies q→0 (no conditional weight surplus)
/// - k_i ≈ N_max implies v→∞ (full saturation)
///
/// This function slightly pulls boundary nodes inward and redistributes
/// the "mass" to safe interior nodes, preserving total sums.
///
/// Uses **relative** epsilon: only nodes within a fraction of their capacity
/// or strength are considered boundary nodes.
fn regularize_degree_targets(
    strength_out: &[f64],
    strength_in: &[f64],
    degree_out: &[f64],
    degree_in: &[f64],
    mask: &PairMask,
) -> (Vec<f64>, Vec<f64>) {
    let n = strength_out.len();

    let mut k_out = degree_out.to_vec();
    let mut k_in = degree_in.to_vec();

    // Regularize outgoing degrees
    regularize_one_side(&mut k_out, strength_out, n, mask, true);
    // Regularize incoming degrees
    regularize_one_side(&mut k_in, strength_in, n, mask, false);

    // Ensure sum(k_out) == sum(k_in) (total edges must balance)
    let sum_out: f64 = k_out.iter().sum();
    let sum_in: f64 = k_in.iter().sum();
    if sum_out > 0.0 && sum_in > 0.0 && (sum_out - sum_in).abs() > 1e-12 {
        if sum_out > sum_in {
            let scale = sum_in / sum_out;
            for k in &mut k_out {
                *k *= scale;
            }
        } else {
            let scale = sum_out / sum_in;
            for k in &mut k_in {
                *k *= scale;
            }
        }
    }

    (k_out, k_in)
}

/// Regularize one side (out or in) of degree targets.
///
/// Only regularizes nodes that are truly at a boundary:
/// - Capacity boundary: k_i >= capacity - relative_eps
/// - Strength boundary: k_i >= s_i - relative_eps (conditional mean → 1)
///
/// The relative epsilon is `RELATIVE_EPS * capacity` for the upper boundary
/// and `RELATIVE_EPS * s_i` for the strength boundary.
fn regularize_one_side(
    degrees: &mut [f64],
    strengths: &[f64],
    n: usize,
    mask: &PairMask,
    is_out: bool,
) {
    /// Fraction of capacity/strength used as the boundary detection threshold.
    const RELATIVE_EPS: f64 = 1e-3;
    /// Absolute minimum epsilon to avoid issues at very small scales.
    const ABS_MIN_EPS: f64 = 1e-9;

    let mut mass_pool = 0.0_f64;
    let mut is_safe = vec![false; n];

    for i in 0..n {
        let capacity = if is_out {
            (n - mask.masked_cols_in_row(i).len()) as f64
        } else {
            (n - mask.masked_rows_in_col(i).len()) as f64
        };
        let s = strengths[i];
        let k = degrees[i];

        if k <= 0.0 || capacity <= 0.0 {
            degrees[i] = 0.0;
            continue;
        }

        // Clamp degree to valid range
        degrees[i] = k.min(capacity).min(s).max(0.0);

        let cap_eps = (RELATIVE_EPS * capacity).max(ABS_MIN_EPS);
        let str_eps = (RELATIVE_EPS * s).max(ABS_MIN_EPS);

        let at_capacity_boundary = capacity > 0.0 && degrees[i] >= capacity - cap_eps;
        let at_strength_boundary = s > 0.0 && degrees[i] >= s - str_eps;

        if at_capacity_boundary {
            // Pull down from capacity boundary
            let pull = cap_eps.min(degrees[i] * 0.1);
            degrees[i] -= pull;
            mass_pool += pull;
        } else if at_strength_boundary {
            // Pull down from strength boundary
            let pull = str_eps.min(degrees[i] * 0.1);
            degrees[i] -= pull;
            mass_pool += pull;
        } else if degrees[i] > 0.0 && degrees[i] < s * 0.9 && degrees[i] < capacity * 0.9 {
            is_safe[i] = true;
        }
    }

    // Redistribute mass to safe nodes
    let safe_count = is_safe.iter().filter(|&&s| s).count();
    if safe_count > 0 && mass_pool > 0.0 {
        let per_node = mass_pool / safe_count as f64;
        for (i, degree) in degrees.iter_mut().enumerate() {
            if is_safe[i] {
                *degree += per_node;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// O(N) Parallel NLL + Gradient Evaluator
// ---------------------------------------------------------------------------

/// Thread-local accumulators for the NLL evaluation.
#[derive(Clone)]
struct RowAccumulator {
    /// Accumulated NLL contribution (sum of ln(Z_ij))
    nll: f64,
    /// Gradient accumulator for α_y[j]: Σ_i E[t_ij] (per column j)
    grad_s_in: Vec<f64>,
    /// Gradient accumulator for α_w[j]: Σ_i E[Θ_ij] (per column j)
    grad_k_in: Vec<f64>,
}

impl RowAccumulator {
    fn new(n: usize) -> Self {
        Self {
            nll: 0.0,
            grad_s_in: vec![0.0; n],
            grad_k_in: vec![0.0; n],
        }
    }

    fn merge(mut self, other: Self) -> Self {
        self.nll += other.nll;
        for (a, b) in self.grad_s_in.iter_mut().zip(other.grad_s_in.iter()) {
            *a += *b;
        }
        for (a, b) in self.grad_k_in.iter_mut().zip(other.grad_k_in.iter()) {
            *a += *b;
        }
        self
    }
}

/// Pair statistics for the Zero-Inflated Poisson model.
///
/// Given q = exp(log_q) and v = exp(log_v):
/// - Z = 1 + v * (exp(q) - 1)
/// - occupation = v * (exp(q) - 1) / Z
/// - expected_weight = v * q * exp(q) / Z
///
/// Returns (ln_z, occupation, expected_weight).
#[inline]
fn zip_pair_statistics(log_q: f64, log_v: f64) -> (f64, f64, f64) {
    // Handle extreme cases
    if log_q <= -690.0 || log_v <= -690.0 {
        return (0.0, 0.0, 0.0);
    }

    let q = log_q.exp();
    let v = log_v.exp();

    if q > 50.0 {
        // Large q: exp(q) - 1 ≈ exp(q), so Z ≈ v*exp(q)
        // ln(Z) ≈ log_v + q
        // occupation ≈ 1
        // E[t] ≈ q (since v*q*exp(q) / (v*exp(q)) = q)
        let ln_z = log_v + q;
        (ln_z, 1.0, q)
    } else {
        let exp_q = q.exp();
        let exp_q_m1 = q.exp_m1(); // exp(q) - 1, numerically stable for small q

        if exp_q_m1 <= 0.0 {
            return (0.0, 0.0, 0.0);
        }

        let v_g = v * exp_q_m1; // v * G(q)
        let z = 1.0 + v_g;
        let ln_z = z.ln();

        let occupation = v_g / z;
        let expected_weight = v * q * exp_q / z;

        (ln_z, occupation, expected_weight)
    }
}

/// Compute the NLL and its gradient for the MEECM model.
///
/// Memory footprint: O(N) per thread (row-by-row accumulation).
/// Time: O(N²) total, parallelized across rows.
///
/// `theta` is a flat slice of size 4*N:
/// [alpha_x_0..alpha_x_{N-1}, alpha_y_0..alpha_y_{N-1},
///  alpha_v_0..alpha_v_{N-1}, alpha_w_0..alpha_w_{N-1}]
///
/// Returns (nll_value, gradient_4N).
fn meecm_nll_and_gradient(
    theta: &[f64],
    s_out_target: &[f64],
    s_in_target: &[f64],
    k_out_target: &[f64],
    k_in_target: &[f64],
    mask: &PairMask,
) -> (f64, Vec<f64>) {
    let n = s_out_target.len();
    debug_assert_eq!(theta.len(), 4 * n);

    let alpha_x = &theta[0..n];
    let alpha_y = &theta[n..2 * n];
    let alpha_v = &theta[2 * n..3 * n];
    let alpha_w = &theta[3 * n..4 * n];

    // Parallel row accumulation: each thread processes a chunk of rows
    // and accumulates O(N) column-wise sums.
    let (total_nll, s_out_pred, k_out_pred, col_acc) = (0..n)
        .into_par_iter()
        .fold(
            || {
                (
                    0.0_f64,
                    vec![0.0_f64; n],
                    vec![0.0_f64; n],
                    RowAccumulator::new(n),
                )
            },
            |(mut nll, mut s_out, mut k_out, mut acc), i| {
                let a_xi = alpha_x[i];
                let a_vi = alpha_v[i];

                for j in 0..n {
                    if mask.is_masked(i, j) {
                        continue;
                    }

                    let log_q = a_xi + alpha_y[j];
                    let log_v = a_vi + alpha_w[j];

                    let (ln_z, occupation, expected_weight) = zip_pair_statistics(log_q, log_v);

                    nll += ln_z;
                    s_out[i] += expected_weight;
                    k_out[i] += occupation;
                    acc.nll += 0.0; // unused, we track nll separately
                    acc.grad_s_in[j] += expected_weight;
                    acc.grad_k_in[j] += occupation;
                }

                (nll, s_out, k_out, acc)
            },
        )
        .reduce(
            || {
                (
                    0.0_f64,
                    vec![0.0_f64; n],
                    vec![0.0_f64; n],
                    RowAccumulator::new(n),
                )
            },
            |(nll_a, mut s_out_a, mut k_out_a, acc_a), (nll_b, s_out_b, k_out_b, acc_b)| {
                for (a, b) in s_out_a.iter_mut().zip(s_out_b.iter()) {
                    *a += *b;
                }
                for (a, b) in k_out_a.iter_mut().zip(k_out_b.iter()) {
                    *a += *b;
                }
                (nll_a + nll_b, s_out_a, k_out_a, acc_a.merge(acc_b))
            },
        );

    // Assemble the full NLL: Σ ln(Z_ij) - Σ α_x[i]*s_out[i] - ...
    let mut nll = total_nll;
    for i in 0..n {
        nll -= alpha_x[i] * s_out_target[i];
        nll -= alpha_y[i] * s_in_target[i];
        nll -= alpha_v[i] * k_out_target[i];
        nll -= alpha_w[i] * k_in_target[i];
    }

    // Assemble gradient: predicted - target
    let mut grad = vec![0.0_f64; 4 * n];
    for i in 0..n {
        grad[i] = s_out_pred[i] - s_out_target[i]; // ∂f/∂α_x[i]
        grad[n + i] = col_acc.grad_s_in[i] - s_in_target[i]; // ∂f/∂α_y[i]
        grad[2 * n + i] = k_out_pred[i] - k_out_target[i]; // ∂f/∂α_v[i]
        grad[3 * n + i] = col_acc.grad_k_in[i] - k_in_target[i]; // ∂f/∂α_w[i]
    }

    (nll, grad)
}

// ---------------------------------------------------------------------------
// L-BFGS Optimizer
// ---------------------------------------------------------------------------

/// L-BFGS two-loop recursion to compute the search direction.
fn lbfgs_direction(
    grad: &[f64],
    s_hist: &[Vec<f64>],
    y_hist: &[Vec<f64>],
    rho_hist: &[f64],
) -> Vec<f64> {
    let mut q = grad.to_vec();
    let m = s_hist.len();
    let mut alpha = vec![0.0; m];

    // First loop (backward)
    for idx in (0..m).rev() {
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

    // Scaling: H0 = (s^T y / y^T y) * I
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

    // Second loop (forward)
    for idx in 0..m {
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

    // Negate for descent direction
    q.iter_mut().for_each(|v| *v = -*v);
    q
}

/// Recenter theta to avoid drift: shift alpha_x/alpha_y and alpha_v/alpha_w
/// so that their means are balanced.
fn recenter_theta(theta: &mut [f64], n: usize) {
    // Balance alpha_x and alpha_y
    let mean_x = theta[..n].iter().sum::<f64>() / n as f64;
    let mean_y = theta[n..2 * n].iter().sum::<f64>() / n as f64;
    let shift_xy = 0.5 * (mean_y - mean_x);
    for v in &mut theta[..n] {
        *v += shift_xy;
    }
    for v in &mut theta[n..2 * n] {
        *v -= shift_xy;
    }

    // Balance alpha_v and alpha_w
    let mean_v = theta[2 * n..3 * n].iter().sum::<f64>() / n as f64;
    let mean_w = theta[3 * n..4 * n].iter().sum::<f64>() / n as f64;
    let shift_vw = 0.5 * (mean_w - mean_v);
    for v in &mut theta[2 * n..3 * n] {
        *v += shift_vw;
    }
    for v in &mut theta[3 * n..4 * n] {
        *v -= shift_vw;
    }
}

/// Fit ME strength-degree (MEECM) using L-BFGS optimization.
///
/// This is the primary solver for the zero-inflated Poisson model with
/// strength and degree constraints. It directly minimizes the NLL of the
/// grand-canonical ensemble.
///
/// # Arguments
///
/// * `strength_out` - Target outgoing strength sequence
/// * `strength_in` - Target incoming strength sequence
/// * `degree_out` - Target outgoing degree sequence
/// * `degree_in` - Target incoming degree sequence
/// * `mask` - Pair mask (self-loops, frozen pairs)
/// * `tolerance` - Convergence tolerance on max |gradient|
/// * `max_iterations` - Maximum L-BFGS iterations
///
/// # Returns
///
/// `StrengthDegreeFitResult` with x, y (strength multipliers) and z, w (degree multipliers).
#[must_use]
pub fn fit_strength_degree_poisson_lbfgs(
    strength_out: &[f64],
    strength_in: &[f64],
    degree_out: &[f64],
    degree_in: &[f64],
    mask: &PairMask,
    tolerance: f64,
    max_iterations: usize,
) -> StrengthDegreeFitResult {
    let n = strength_out.len();
    if n == 0 {
        return StrengthDegreeFitResult {
            x: vec![],
            y: vec![],
            z: vec![],
            w: vec![],
            converged: true,
            iterations: 0,
        };
    }

    // Step 1: Regularize degree targets to avoid boundary singularities
    let (k_out_target, k_in_target) =
        regularize_degree_targets(strength_out, strength_in, degree_out, degree_in, mask);

    // Step 2: Initialize theta in log-domain
    let mut theta = initialize_theta(strength_out, strength_in, &k_out_target, &k_in_target, mask);

    // Step 3: L-BFGS optimization
    let memory = 10_usize; // L-BFGS memory (number of correction pairs)

    let (mut obj, mut grad) = meecm_nll_and_gradient(
        &theta,
        strength_out,
        strength_in,
        &k_out_target,
        &k_in_target,
        mask,
    );

    let grad_norm = grad.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    // Tolerance scaled by the magnitude of the targets
    let target_scale = strength_out
        .iter()
        .chain(strength_in.iter())
        .chain(k_out_target.iter())
        .chain(k_in_target.iter())
        .copied()
        .fold(1.0_f64, f64::max);
    let grad_tol = tolerance * target_scale;

    if grad_norm < grad_tol {
        return theta_to_result(&theta, n, true, 0);
    }

    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();

    for iter in 0..max_iterations {
        // Compute search direction
        let direction = lbfgs_direction(&grad, &s_hist, &y_hist, &rho_hist);

        // Check descent direction
        let directional_deriv: f64 = grad.iter().zip(direction.iter()).map(|(g, d)| g * d).sum();
        if directional_deriv >= 0.0 || !directional_deriv.is_finite() {
            // Not a descent direction, reset L-BFGS history
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            // Use steepest descent
            let sd_direction: Vec<f64> = grad.iter().map(|g| -g).collect();
            let sd_deriv: f64 = grad.iter().map(|g| -g * g).sum();
            if sd_deriv >= 0.0 {
                break;
            }
            // Try a steepest descent step
            if let Some((next_theta, next_grad, next_obj)) = backtracking_line_search(
                &theta,
                &sd_direction,
                sd_deriv,
                obj,
                strength_out,
                strength_in,
                &k_out_target,
                &k_in_target,
                mask,
                n,
            ) {
                let s: Vec<f64> = next_theta
                    .iter()
                    .zip(theta.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                let y_vec: Vec<f64> = next_grad
                    .iter()
                    .zip(grad.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                let sy: f64 = s.iter().zip(y_vec.iter()).map(|(a, b)| a * b).sum();
                if sy > 1e-12 && sy.is_finite() {
                    s_hist.push(s);
                    y_hist.push(y_vec);
                    rho_hist.push(1.0 / sy);
                }
                theta = next_theta;
                grad = next_grad;
                obj = next_obj;
            } else {
                break;
            }
            continue;
        }

        // Backtracking line search
        let Some((mut next_theta, _next_grad, _next_obj)) = backtracking_line_search(
            &theta,
            &direction,
            directional_deriv,
            obj,
            strength_out,
            strength_in,
            &k_out_target,
            &k_in_target,
            mask,
            n,
        ) else {
            // Line search failed, try reset
            if !s_hist.is_empty() {
                s_hist.clear();
                y_hist.clear();
                rho_hist.clear();
                continue;
            }
            break;
        };

        // Recenter to avoid drift
        recenter_theta(&mut next_theta, n);

        // Re-evaluate after recentering (cheap compared to N² evaluation if drift was small)
        let (recentered_obj, recentered_grad) = meecm_nll_and_gradient(
            &next_theta,
            strength_out,
            strength_in,
            &k_out_target,
            &k_in_target,
            mask,
        );

        // Update L-BFGS history
        let s: Vec<f64> = next_theta
            .iter()
            .zip(theta.iter())
            .map(|(a, b)| a - b)
            .collect();
        let y_vec: Vec<f64> = recentered_grad
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

        theta = next_theta;
        grad = recentered_grad;
        obj = recentered_obj;

        // Check convergence
        let new_grad_norm = grad.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if new_grad_norm < grad_tol {
            return theta_to_result(&theta, n, true, iter + 1);
        }
    }

    // Did not converge within max_iterations
    theta_to_result(&theta, n, false, max_iterations)
}

/// Backtracking line search with Armijo condition.
///
/// Returns `Some((new_theta, new_grad, new_obj))` if a step is accepted.
#[allow(clippy::too_many_arguments)]
fn backtracking_line_search(
    theta: &[f64],
    direction: &[f64],
    directional_deriv: f64,
    current_obj: f64,
    s_out_target: &[f64],
    s_in_target: &[f64],
    k_out_target: &[f64],
    k_in_target: &[f64],
    mask: &PairMask,
    _n: usize,
) -> Option<(Vec<f64>, Vec<f64>, f64)> {
    let c1 = 1e-4; // Armijo constant
    let mut step = 1.0_f64;
    let max_step_clamp = 5.0_f64; // prevent huge jumps

    // Clamp initial step to avoid overshooting
    let max_dir = direction.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
    if max_dir * step > max_step_clamp {
        step = max_step_clamp / max_dir;
    }

    for _ in 0..40 {
        let candidate: Vec<f64> = theta
            .iter()
            .zip(direction.iter())
            .map(|(t, d)| t + step * d)
            .collect();

        let (candidate_obj, candidate_grad) = meecm_nll_and_gradient(
            &candidate,
            s_out_target,
            s_in_target,
            k_out_target,
            k_in_target,
            mask,
        );

        if candidate_obj.is_finite() && candidate_obj <= current_obj + c1 * step * directional_deriv
        {
            return Some((candidate, candidate_grad, candidate_obj));
        }

        step *= 0.5;
        if step < 1e-15 {
            break;
        }
    }

    None
}

/// Initialize theta (log-domain parameters) from target sequences.
fn initialize_theta(
    strength_out: &[f64],
    strength_in: &[f64],
    k_out_target: &[f64],
    k_in_target: &[f64],
    _mask: &PairMask,
) -> Vec<f64> {
    let n = strength_out.len();
    let total_s = strength_out.iter().sum::<f64>().max(1.0);
    let total_k = k_out_target.iter().sum::<f64>().max(1.0);
    let scale_s = total_s.sqrt();
    let scale_k = (total_k / n.max(1) as f64).sqrt().max(0.1);

    let mut theta = vec![0.0_f64; 4 * n];

    // Initialize alpha_x (log strength-out multipliers)
    for i in 0..n {
        theta[i] = if strength_out[i] > 0.0 {
            (strength_out[i] / scale_s).max(1e-12).ln()
        } else {
            -690.0
        };
    }
    // Initialize alpha_y (log strength-in multipliers)
    for j in 0..n {
        theta[n + j] = if strength_in[j] > 0.0 {
            (strength_in[j] / scale_s).max(1e-12).ln()
        } else {
            -690.0
        };
    }
    // Initialize alpha_v (log degree-out multipliers)
    for i in 0..n {
        theta[2 * n + i] = if k_out_target[i] > 0.0 {
            (k_out_target[i] / total_k * scale_k).max(1e-12).ln()
        } else {
            -690.0
        };
    }
    // Initialize alpha_w (log degree-in multipliers)
    for j in 0..n {
        theta[3 * n + j] = if k_in_target[j] > 0.0 {
            (k_in_target[j] / total_k * scale_k).max(1e-12).ln()
        } else {
            -690.0
        };
    }

    // Recenter for numerical balance
    recenter_theta(&mut theta, n);
    theta
}

/// Convert theta back to multipliers (x, y, z, w) in the original domain.
fn theta_to_result(
    theta: &[f64],
    n: usize,
    converged: bool,
    iterations: usize,
) -> StrengthDegreeFitResult {
    let x: Vec<f64> = theta[..n].iter().map(|&v| v.exp()).collect();
    let y: Vec<f64> = theta[n..2 * n].iter().map(|&v| v.exp()).collect();
    let z: Vec<f64> = theta[2 * n..3 * n].iter().map(|&v| v.exp()).collect();
    let w: Vec<f64> = theta[3 * n..4 * n].iter().map(|&v| v.exp()).collect();

    StrengthDegreeFitResult {
        x,
        y,
        z,
        w,
        converged,
        iterations,
    }
}

// ---------------------------------------------------------------------------
// Strength-Edges L-BFGS (2N+1 parameters: α_x, α_y, ln_λ)
// ---------------------------------------------------------------------------

/// NLL and gradient for ME strength-edges (scalar λ, not per-node).
///
/// θ = [α_x(N), α_y(N), ln_λ(1)]  →  2N+1 parameters.
/// q_ij = exp(α_x[i] + α_y[j]),  λ = exp(ln_λ).
/// Z_ij = 1 + λ·(exp(q_ij) - 1).
fn me_strength_edges_nll_and_gradient(
    theta: &[f64],
    s_out_target: &[f64],
    s_in_target: &[f64],
    target_edges: f64,
    mask: &PairMask,
) -> (f64, Vec<f64>) {
    let n = s_out_target.len();
    debug_assert_eq!(theta.len(), 2 * n + 1);

    let alpha_x = &theta[0..n];
    let alpha_y = &theta[n..2 * n];
    let log_lam = theta[2 * n];

    // Reuse zip_pair_statistics with log_v = log_lam for all pairs
    let (total_nll, s_out_pred, s_in_pred, total_occ) = (0..n)
        .into_par_iter()
        .fold(
            || (0.0_f64, vec![0.0_f64; n], vec![0.0_f64; n], 0.0_f64),
            |(mut nll, mut s_out, mut s_in, mut occ), i| {
                for j in 0..n {
                    if mask.is_masked(i, j) {
                        continue;
                    }
                    let log_q = alpha_x[i] + alpha_y[j];
                    let (ln_z, occupation, expected_weight) = zip_pair_statistics(log_q, log_lam);
                    nll += ln_z;
                    s_out[i] += expected_weight;
                    s_in[j] += expected_weight;
                    occ += occupation;
                }
                (nll, s_out, s_in, occ)
            },
        )
        .reduce(
            || (0.0_f64, vec![0.0_f64; n], vec![0.0_f64; n], 0.0_f64),
            |(nll_a, mut so_a, mut si_a, occ_a), (nll_b, so_b, si_b, occ_b)| {
                for (a, b) in so_a.iter_mut().zip(so_b.iter()) {
                    *a += *b;
                }
                for (a, b) in si_a.iter_mut().zip(si_b.iter()) {
                    *a += *b;
                }
                (nll_a + nll_b, so_a, si_a, occ_a + occ_b)
            },
        );

    let mut nll = total_nll;
    for i in 0..n {
        nll -= alpha_x[i] * s_out_target[i];
        nll -= alpha_y[i] * s_in_target[i];
    }
    nll -= log_lam * target_edges;

    let mut grad = vec![0.0_f64; 2 * n + 1];
    for i in 0..n {
        grad[i] = s_out_pred[i] - s_out_target[i];
        grad[n + i] = s_in_pred[i] - s_in_target[i];
    }
    grad[2 * n] = total_occ - target_edges;

    (nll, grad)
}

/// Fit ME strength-edges using L-BFGS optimization.
///
/// Minimizes the NLL with 2N+1 parameters: log strength multipliers + scalar log λ.
#[must_use]
pub fn fit_strength_edges_poisson_lbfgs(
    strength_out: &[f64],
    strength_in: &[f64],
    target_edges: f64,
    mask: &PairMask,
    tolerance: f64,
    max_iterations: usize,
) -> StrengthEdgesFitResult {
    let n = strength_out.len();
    let n_free = mask.n_free() as f64;
    let total: f64 = strength_out.iter().sum();

    if n == 0 || target_edges <= 0.0 || target_edges >= n_free || target_edges > total {
        return StrengthEdgesFitResult {
            x: vec![0.0; n],
            y: vec![0.0; n],
            lam: 0.0,
            converged: false,
            iterations: 0,
        };
    }

    let dim = 2 * n + 1;
    let scale_s = total.sqrt().max(1.0);
    let mut theta = vec![0.0_f64; dim];
    for i in 0..n {
        theta[i] = if strength_out[i] > 0.0 {
            (strength_out[i] / scale_s).max(1e-12).ln()
        } else {
            -690.0
        };
    }
    for j in 0..n {
        theta[n + j] = if strength_in[j] > 0.0 {
            (strength_in[j] / scale_s).max(1e-12).ln()
        } else {
            -690.0
        };
    }
    // Initialize ln_λ from edge density
    let lam_init = target_edges / (n_free - target_edges).max(0.01);
    theta[2 * n] = lam_init.max(1e-12).ln();

    // Recenter x/y
    let mean_x = theta[..n].iter().sum::<f64>() / n as f64;
    let mean_y = theta[n..2 * n].iter().sum::<f64>() / n as f64;
    let shift = 0.5 * (mean_y - mean_x);
    for v in &mut theta[..n] {
        *v += shift;
    }
    for v in &mut theta[n..2 * n] {
        *v -= shift;
    }

    let memory = 10_usize;
    let (mut obj, mut grad) =
        me_strength_edges_nll_and_gradient(&theta, strength_out, strength_in, target_edges, mask);

    let target_scale = strength_out
        .iter()
        .chain(strength_in.iter())
        .copied()
        .fold(target_edges, f64::max);
    let grad_tol = tolerance * target_scale;

    if grad.iter().map(|v| v.abs()).fold(0.0_f64, f64::max) < grad_tol {
        return se_theta_to_result(&theta, n, true, 0);
    }

    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();

    for iter in 0..max_iterations {
        let direction = lbfgs_direction(&grad, &s_hist, &y_hist, &rho_hist);
        let directional_deriv: f64 = grad.iter().zip(direction.iter()).map(|(g, d)| g * d).sum();

        if directional_deriv >= 0.0 || !directional_deriv.is_finite() {
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            let sd: Vec<f64> = grad.iter().map(|g| -g).collect();
            let sd_deriv: f64 = grad.iter().map(|g| -g * g).sum();
            if sd_deriv >= 0.0 {
                break;
            }
            if let Some((next, ng, no)) = se_line_search(
                &theta,
                &sd,
                sd_deriv,
                obj,
                strength_out,
                strength_in,
                target_edges,
                mask,
                n,
            ) {
                update_lbfgs_history(
                    &next,
                    &theta,
                    &ng,
                    &grad,
                    &mut s_hist,
                    &mut y_hist,
                    &mut rho_hist,
                    memory,
                );
                theta = next;
                grad = ng;
                obj = no;
            } else {
                break;
            }
            continue;
        }

        let Some((mut next, _ng, _no)) = se_line_search(
            &theta,
            &direction,
            directional_deriv,
            obj,
            strength_out,
            strength_in,
            target_edges,
            mask,
            n,
        ) else {
            if !s_hist.is_empty() {
                s_hist.clear();
                y_hist.clear();
                rho_hist.clear();
                continue;
            }
            break;
        };

        // Recenter x/y only (not λ)
        let mx = next[..n].iter().sum::<f64>() / n as f64;
        let my = next[n..2 * n].iter().sum::<f64>() / n as f64;
        let sh = 0.5 * (my - mx);
        for v in &mut next[..n] {
            *v += sh;
        }
        for v in &mut next[n..2 * n] {
            *v -= sh;
        }

        let (ro, rg) = me_strength_edges_nll_and_gradient(
            &next,
            strength_out,
            strength_in,
            target_edges,
            mask,
        );

        update_lbfgs_history(
            &next,
            &theta,
            &rg,
            &grad,
            &mut s_hist,
            &mut y_hist,
            &mut rho_hist,
            memory,
        );
        theta = next;
        grad = rg;
        obj = ro;

        if grad.iter().map(|v| v.abs()).fold(0.0_f64, f64::max) < grad_tol {
            return se_theta_to_result(&theta, n, true, iter + 1);
        }
    }

    se_theta_to_result(&theta, n, false, max_iterations)
}

#[allow(clippy::too_many_arguments)]
fn se_line_search(
    theta: &[f64],
    direction: &[f64],
    directional_deriv: f64,
    current_obj: f64,
    s_out: &[f64],
    s_in: &[f64],
    target_edges: f64,
    mask: &PairMask,
    _n: usize,
) -> Option<(Vec<f64>, Vec<f64>, f64)> {
    let c1 = 1e-4;
    let mut step = 1.0_f64;
    let max_dir = direction.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
    if max_dir * step > 5.0 {
        step = 5.0 / max_dir;
    }
    for _ in 0..40 {
        let cand: Vec<f64> = theta
            .iter()
            .zip(direction.iter())
            .map(|(t, d)| t + step * d)
            .collect();
        let (co, cg) = me_strength_edges_nll_and_gradient(&cand, s_out, s_in, target_edges, mask);
        if co.is_finite() && co <= current_obj + c1 * step * directional_deriv {
            return Some((cand, cg, co));
        }
        step *= 0.5;
        if step < 1e-15 {
            break;
        }
    }
    None
}

fn se_theta_to_result(
    theta: &[f64],
    n: usize,
    converged: bool,
    iterations: usize,
) -> StrengthEdgesFitResult {
    StrengthEdgesFitResult {
        x: theta[..n].iter().map(|&v| v.exp()).collect(),
        y: theta[n..2 * n].iter().map(|&v| v.exp()).collect(),
        lam: theta[2 * n].exp(),
        converged,
        iterations,
    }
}

#[allow(clippy::too_many_arguments)]
fn update_lbfgs_history(
    next: &[f64],
    prev: &[f64],
    next_grad: &[f64],
    prev_grad: &[f64],
    s_hist: &mut Vec<Vec<f64>>,
    y_hist: &mut Vec<Vec<f64>>,
    rho_hist: &mut Vec<f64>,
    memory: usize,
) {
    let s: Vec<f64> = next.iter().zip(prev.iter()).map(|(a, b)| a - b).collect();
    let yv: Vec<f64> = next_grad
        .iter()
        .zip(prev_grad.iter())
        .map(|(a, b)| a - b)
        .collect();
    let sy: f64 = s.iter().zip(yv.iter()).map(|(a, b)| a * b).sum();
    if sy > 1e-12 && sy.is_finite() {
        if s_hist.len() == memory {
            s_hist.remove(0);
            y_hist.remove(0);
            rho_hist.remove(0);
        }
        s_hist.push(s);
        y_hist.push(yv);
        rho_hist.push(1.0 / sy);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Compute E[t_ij] from raw multipliers for verification.
    fn me_sd_expected_weight(q: f64, v: f64) -> f64 {
        if q <= 0.0 || v <= 0.0 {
            return 0.0;
        }
        let exp_q = q.exp();
        let exp_q_m1 = q.exp_m1();
        if exp_q_m1 <= 0.0 {
            return 0.0;
        }
        v * q * exp_q / (1.0 + v * exp_q_m1)
    }

    /// Compute E[Θ(t_ij>0)] from raw multipliers for verification.
    fn me_sd_occupation(q: f64, v: f64) -> f64 {
        if q <= 0.0 || v <= 0.0 {
            return 0.0;
        }
        let exp_q_m1 = q.exp_m1();
        if exp_q_m1 <= 0.0 {
            return 0.0;
        }
        v * exp_q_m1 / (1.0 + v * exp_q_m1)
    }

    #[test]
    fn zip_pair_statistics_matches_explicit_formulas() {
        let q = 0.4_f64;
        let v = 0.7_f64;
        let log_q = q.ln();
        let log_v = v.ln();

        let (ln_z, occ, weight) = zip_pair_statistics(log_q, log_v);

        let exp_q = q.exp();
        let expected_z = 1.0 + v * (exp_q - 1.0);
        let expected_occ = v * (exp_q - 1.0) / expected_z;
        let expected_weight = v * q * exp_q / expected_z;

        assert!((ln_z - expected_z.ln()).abs() < 1e-14);
        assert!((occ - expected_occ).abs() < 1e-14);
        assert!((weight - expected_weight).abs() < 1e-14);
    }

    #[test]
    fn zip_pair_statistics_large_q_overflow_safe() {
        // q = 100, v = 1e5 → should not overflow
        let log_q = 100.0_f64.ln();
        let log_v = 1e5_f64.ln();

        let (ln_z, occ, weight) = zip_pair_statistics(log_q, log_v);

        assert!(ln_z.is_finite());
        assert!(occ.is_finite());
        assert!((occ - 1.0).abs() < 1e-10); // near saturation
        assert!((weight - 100.0).abs() < 1e-10); // E[t] ≈ q when saturated
    }

    #[test]
    fn gradient_finite_difference_check() {
        // Verify gradient by finite differences on a small problem
        let n = 3;
        let s_out = vec![5.0, 10.0, 8.0];
        let s_in = vec![7.0, 9.0, 7.0];
        let k_out = vec![1.5, 2.0, 1.8];
        let k_in = vec![1.7, 1.8, 1.8];
        let mask = PairMask::from_self_loops(n, true);

        let theta = vec![
            -0.5, 0.3, 0.1, // alpha_x
            0.0, -0.2, 0.2, // alpha_y
            -1.0, -0.5, -0.8, // alpha_v
            -0.7, -0.9, -0.6, // alpha_w
        ];

        let (obj, grad) = meecm_nll_and_gradient(&theta, &s_out, &s_in, &k_out, &k_in, &mask);
        assert!(obj.is_finite());

        let eps = 1e-7;
        for idx in 0..theta.len() {
            let mut theta_plus = theta.clone();
            theta_plus[idx] += eps;
            let (obj_plus, _) =
                meecm_nll_and_gradient(&theta_plus, &s_out, &s_in, &k_out, &k_in, &mask);

            let mut theta_minus = theta.clone();
            theta_minus[idx] -= eps;
            let (obj_minus, _) =
                meecm_nll_and_gradient(&theta_minus, &s_out, &s_in, &k_out, &k_in, &mask);

            let fd_grad = (obj_plus - obj_minus) / (2.0 * eps);
            assert!(
                (grad[idx] - fd_grad).abs() < 1e-5,
                "gradient[{idx}]: analytic={:.8}, fd={:.8}",
                grad[idx],
                fd_grad
            );
        }
    }

    #[test]
    fn lbfgs_recovers_small_n3_constraints() {
        // Generate targets from known multipliers, then verify recovery
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
                k_out[i] += me_sd_occupation(q, v);
                k_in[j] += me_sd_occupation(q, v);
                s_out[i] += me_sd_expected_weight(q, v);
                s_in[j] += me_sd_expected_weight(q, v);
            }
        }

        let result =
            fit_strength_degree_poisson_lbfgs(&s_out, &s_in, &k_out, &k_in, &mask, 1e-8, 5000);

        assert!(
            result.converged,
            "L-BFGS must converge on N=3 regression test"
        );

        // Verify constraint recovery
        for i in 0..n {
            let mut row_s = 0.0;
            let mut row_k = 0.0;
            for j in 0..n {
                row_s +=
                    me_sd_expected_weight(result.x[i] * result.y[j], result.z[i] * result.w[j]);
                row_k += me_sd_occupation(result.x[i] * result.y[j], result.z[i] * result.w[j]);
            }
            assert!(
                (row_s - s_out[i]).abs() < 1e-5,
                "s_out[{i}]: expected {}, got {row_s}",
                s_out[i]
            );
            assert!(
                (row_k - k_out[i]).abs() < 1e-5,
                "k_out[{i}]: expected {}, got {row_k}",
                k_out[i]
            );
        }
    }

    #[test]
    fn lbfgs_respects_pair_mask() {
        // Same test but with a masked pair
        let x = [0.3_f64, 0.2, 0.4];
        let y = [0.25_f64, 0.35, 0.2];
        let z = [0.4_f64, 0.3, 0.5];
        let w = [0.45_f64, 0.25, 0.35];
        let n = 3;
        let known_src = [0u64];
        let known_tgt = [1u64];
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
                k_out[i] += me_sd_occupation(q, v);
                k_in[j] += me_sd_occupation(q, v);
                s_out[i] += me_sd_expected_weight(q, v);
                s_in[j] += me_sd_expected_weight(q, v);
            }
        }

        let result =
            fit_strength_degree_poisson_lbfgs(&s_out, &s_in, &k_out, &k_in, &mask, 1e-8, 5000);

        assert!(result.converged, "L-BFGS must converge with masked pairs");

        for i in 0..n {
            let row_s: f64 = (0..n)
                .filter(|&j| !mask.is_masked(i, j))
                .map(|j| {
                    me_sd_expected_weight(result.x[i] * result.y[j], result.z[i] * result.w[j])
                })
                .sum();
            let row_k: f64 = (0..n)
                .filter(|&j| !mask.is_masked(i, j))
                .map(|j| me_sd_occupation(result.x[i] * result.y[j], result.z[i] * result.w[j]))
                .sum();
            assert!(
                (row_s - s_out[i]).abs() < 1e-5,
                "s_out[{i}]: expected {}, got {row_s}",
                s_out[i]
            );
            assert!(
                (row_k - k_out[i]).abs() < 1e-5,
                "k_out[{i}]: expected {}, got {row_k}",
                k_out[i]
            );
        }
    }

    #[test]
    fn lbfgs_no_self_loops_n5() {
        // Larger test with no self-loops
        let n = 5;
        let x: Vec<f64> = vec![0.3, 0.5, 0.2, 0.4, 0.35];
        let y: Vec<f64> = vec![0.25, 0.3, 0.45, 0.2, 0.35];
        let z: Vec<f64> = vec![0.4, 0.6, 0.3, 0.5, 0.45];
        let w: Vec<f64> = vec![0.35, 0.4, 0.5, 0.3, 0.45];
        let mask = PairMask::from_self_loops(n, false);

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
                k_out[i] += me_sd_occupation(q, v);
                k_in[j] += me_sd_occupation(q, v);
                s_out[i] += me_sd_expected_weight(q, v);
                s_in[j] += me_sd_expected_weight(q, v);
            }
        }

        let result =
            fit_strength_degree_poisson_lbfgs(&s_out, &s_in, &k_out, &k_in, &mask, 1e-8, 5000);

        assert!(
            result.converged,
            "L-BFGS must converge on N=5 no-self-loops"
        );

        let tol = 1e-4;
        for i in 0..n {
            let row_s: f64 = (0..n)
                .filter(|&j| !mask.is_masked(i, j))
                .map(|j| {
                    me_sd_expected_weight(result.x[i] * result.y[j], result.z[i] * result.w[j])
                })
                .sum();
            let row_k: f64 = (0..n)
                .filter(|&j| !mask.is_masked(i, j))
                .map(|j| me_sd_occupation(result.x[i] * result.y[j], result.z[i] * result.w[j]))
                .sum();
            assert!(
                (row_s - s_out[i]).abs() < tol,
                "s_out[{i}]: expected {}, got {row_s}",
                s_out[i]
            );
            assert!(
                (row_k - k_out[i]).abs() < tol,
                "k_out[{i}]: expected {}, got {row_k}",
                k_out[i]
            );
        }
        for j in 0..n {
            let col_s: f64 = (0..n)
                .filter(|&i| !mask.is_masked(i, j))
                .map(|i| {
                    me_sd_expected_weight(result.x[i] * result.y[j], result.z[i] * result.w[j])
                })
                .sum();
            let col_k: f64 = (0..n)
                .filter(|&i| !mask.is_masked(i, j))
                .map(|i| me_sd_occupation(result.x[i] * result.y[j], result.z[i] * result.w[j]))
                .sum();
            assert!(
                (col_s - s_in[j]).abs() < tol,
                "s_in[{j}]: expected {}, got {col_s}",
                s_in[j]
            );
            assert!(
                (col_k - k_in[j]).abs() < tol,
                "k_in[{j}]: expected {}, got {col_k}",
                k_in[j]
            );
        }
    }

    #[test]
    fn regularization_pulls_boundary_nodes_inward() {
        let n = 4;
        let mask = PairMask::from_self_loops(n, false); // capacity = 3
                                                        // Node 0 has degree = capacity (saturated)
                                                        // Node 1 has degree ≈ strength (at lower boundary)
        let s_out = vec![10.0, 2.0, 8.0, 5.0];
        let s_in = vec![6.0, 7.0, 5.0, 7.0];
        let k_out = vec![3.0, 2.0, 2.0, 1.5]; // node 0 saturated (capacity=3)
        let k_in = vec![2.0, 2.5, 1.5, 2.5];

        let (k_out_reg, k_in_reg) = regularize_degree_targets(&s_out, &s_in, &k_out, &k_in, &mask);

        // Saturated node 0 should have been pulled down
        assert!(k_out_reg[0] < 3.0, "saturated node should be pulled down");
        // All values should be positive
        for (i, &k) in k_out_reg.iter().enumerate() {
            assert!(k >= 0.0, "k_out[{i}] = {k} should be non-negative");
        }
        for (j, &k) in k_in_reg.iter().enumerate() {
            assert!(k >= 0.0, "k_in[{j}] = {k} should be non-negative");
        }
        // Total should be preserved (approximately)
        let orig_total = k_out.iter().sum::<f64>().min(k_in.iter().sum::<f64>());
        let reg_total_out: f64 = k_out_reg.iter().sum();
        let reg_total_in: f64 = k_in_reg.iter().sum();
        assert!(
            (reg_total_out - reg_total_in).abs() < 0.01,
            "out/in totals must balance"
        );
        // Total should be close to original (mass is redistributed, not lost)
        assert!(
            (reg_total_out - orig_total).abs() < 1.0,
            "total should be approximately preserved"
        );
    }

    #[test]
    fn lbfgs_moderate_n10_heterogeneous() {
        // Test with N=10 and heterogeneous multipliers while keeping unit tests fast.
        let n = 10;
        let mask = PairMask::from_self_loops(n, false);

        // Generate heterogeneous multipliers (varying by 10x)
        let x: Vec<f64> = (0..n).map(|i| 0.1 + 0.9 * (i as f64 / n as f64)).collect();
        let y: Vec<f64> = (0..n)
            .map(|j| 0.2 + 0.8 * ((n - 1 - j) as f64 / n as f64))
            .collect();
        let z: Vec<f64> = (0..n)
            .map(|i| 0.3 + 0.5 * ((i * 7 % n) as f64 / n as f64))
            .collect();
        let w: Vec<f64> = (0..n)
            .map(|j| 0.4 + 0.4 * ((j * 3 % n) as f64 / n as f64))
            .collect();

        // Generate targets from multipliers
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
                k_out[i] += me_sd_occupation(q, v);
                k_in[j] += me_sd_occupation(q, v);
                s_out[i] += me_sd_expected_weight(q, v);
                s_in[j] += me_sd_expected_weight(q, v);
            }
        }

        let result =
            fit_strength_degree_poisson_lbfgs(&s_out, &s_in, &k_out, &k_in, &mask, 1e-6, 2000);

        assert!(
            result.converged,
            "L-BFGS must converge on N=10 heterogeneous inputs"
        );

        // Verify constraint recovery with reasonable tolerance
        let tol = 1e-3;
        for i in 0..n {
            let row_s: f64 = (0..n)
                .filter(|&j| !mask.is_masked(i, j))
                .map(|j| {
                    me_sd_expected_weight(result.x[i] * result.y[j], result.z[i] * result.w[j])
                })
                .sum();
            let row_k: f64 = (0..n)
                .filter(|&j| !mask.is_masked(i, j))
                .map(|j| me_sd_occupation(result.x[i] * result.y[j], result.z[i] * result.w[j]))
                .sum();
            assert!(
                (row_s - s_out[i]).abs() < tol,
                "s_out[{i}]: expected {}, got {row_s}",
                s_out[i]
            );
            assert!(
                (row_k - k_out[i]).abs() < tol,
                "k_out[{i}]: expected {}, got {row_k}",
                k_out[i]
            );
        }
    }

    // --- Strength-Edges tests ---

    #[test]
    fn se_gradient_finite_difference() {
        let n = 3;
        let s_out = vec![5.0, 10.0, 8.0];
        let s_in = vec![7.0, 9.0, 7.0];
        let target_edges = 4.0;
        let mask = PairMask::from_self_loops(n, true);
        let theta = vec![-0.5, 0.3, 0.1, 0.0, -0.2, 0.2, -1.0];

        let (_obj, grad) =
            me_strength_edges_nll_and_gradient(&theta, &s_out, &s_in, target_edges, &mask);
        let eps = 1e-7;
        for idx in 0..theta.len() {
            let mut tp = theta.clone();
            tp[idx] += eps;
            let (op, _) =
                me_strength_edges_nll_and_gradient(&tp, &s_out, &s_in, target_edges, &mask);
            let mut tm = theta.clone();
            tm[idx] -= eps;
            let (om, _) =
                me_strength_edges_nll_and_gradient(&tm, &s_out, &s_in, target_edges, &mask);
            let fd = (op - om) / (2.0 * eps);
            assert!(
                (grad[idx] - fd).abs() < 1e-5,
                "grad[{idx}]: analytic={:.8}, fd={:.8}",
                grad[idx],
                fd
            );
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn se_lbfgs_recovers_constraints_n5() {
        let n = 5;
        let mask = PairMask::from_self_loops(n, false);
        let x = [0.3_f64, 0.5, 0.2, 0.4, 0.35];
        let y = [0.25_f64, 0.3, 0.45, 0.2, 0.35];
        let lam = 0.8_f64;
        let mut s_out = vec![0.0; n];
        let mut s_in = vec![0.0; n];
        let mut total_edges = 0.0;
        for i in 0..n {
            for j in 0..n {
                if mask.is_masked(i, j) {
                    continue;
                }
                let q = x[i] * y[j];
                let em1 = q.exp_m1();
                if em1 <= 0.0 {
                    continue;
                }
                let vg = lam * em1;
                let z = 1.0 + vg;
                s_out[i] += lam * q * q.exp() / z;
                s_in[j] += lam * q * q.exp() / z;
                total_edges += vg / z;
            }
        }

        let result =
            fit_strength_edges_poisson_lbfgs(&s_out, &s_in, total_edges, &mask, 1e-6, 5000);
        assert!(result.converged, "L-BFGS SE must converge");

        let tol = 1e-3;
        for i in 0..n {
            let mut row_s = 0.0;
            for j in 0..n {
                if mask.is_masked(i, j) {
                    continue;
                }
                let q = result.x[i] * result.y[j];
                let em1 = q.exp_m1();
                if em1 <= 0.0 {
                    continue;
                }
                row_s += result.lam * q * q.exp() / (1.0 + result.lam * em1);
            }
            assert!(
                (row_s - s_out[i]).abs() < tol,
                "s_out[{i}]: expected {}, got {row_s}",
                s_out[i]
            );
        }
    }
}
