//! L-BFGS solver for B (Binomial) strength-degree fitting.
//!
//! Same approach as ME L-BFGS but with Binomial pair statistics:
//! - G_B(q) = (1+q)^M - 1
//! - E[Θ(t_ij>0)] = v_ij * G_B(q_ij) / (1 + v_ij * G_B(q_ij))
//! - E[t_ij] = v_ij * M * q_ij * (1+q_ij)^(M-1) / (1 + v_ij * G_B(q_ij))
//!
//! NLL objective (dual of max-entropy):
//!   f(θ) = Σ_{(i,j) free} ln(Z_ij) - Σ_i α_x[i]*s_out[i] - Σ_j α_y[j]*s_in[j]
//!          - Σ_i α_v[i]*k_out[i] - Σ_j α_w[j]*k_in[j]
//!
//! where Z_ij = 1 + v_ij * ((1+q_ij)^M - 1), q_ij = exp(α_x[i]+α_y[j]),
//! v_ij = exp(α_v[i]+α_w[j]).
//!
//! Memory: O(N) per thread. No N×N dense matrices.

use rayon::prelude::*;

use super::mask::PairMask;
use super::{StrengthDegreeFitResult, StrengthEdgesFitResult};

// ---------------------------------------------------------------------------
// Regularization (same structure as ME, adapted for B bounds: s <= M*N_max)
// ---------------------------------------------------------------------------

fn regularize_degree_targets(
    strength_out: &[f64],
    strength_in: &[f64],
    degree_out: &[f64],
    degree_in: &[f64],
    mask: &PairMask,
    layers: u32,
) -> (Vec<f64>, Vec<f64>) {
    let n = strength_out.len();
    let m = f64::from(layers);

    let mut k_out = degree_out.to_vec();
    let mut k_in = degree_in.to_vec();

    regularize_one_side(&mut k_out, strength_out, n, mask, m, true);
    regularize_one_side(&mut k_in, strength_in, n, mask, m, false);

    // Balance totals
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

fn regularize_one_side(
    degrees: &mut [f64],
    strengths: &[f64],
    n: usize,
    mask: &PairMask,
    m: f64,
    is_out: bool,
) {
    const RELATIVE_EPS: f64 = 1e-3;
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
        // For B, max strength per node = M * capacity
        let _max_strength = m * capacity;

        if k <= 0.0 || capacity <= 0.0 {
            degrees[i] = 0.0;
            continue;
        }

        // Clamp: k <= capacity, k <= s (for B: k can equal s when M=1 Bernoulli)
        degrees[i] = k.min(capacity).min(s).max(0.0);

        let cap_eps = (RELATIVE_EPS * capacity).max(ABS_MIN_EPS);
        let str_eps = (RELATIVE_EPS * s).max(ABS_MIN_EPS);

        let at_capacity_boundary = capacity > 0.0 && degrees[i] >= capacity - cap_eps;
        let at_strength_boundary = s > 0.0 && degrees[i] >= s - str_eps;

        if at_capacity_boundary {
            let pull = cap_eps.min(degrees[i] * 0.1);
            degrees[i] -= pull;
            mass_pool += pull;
        } else if at_strength_boundary {
            let pull = str_eps.min(degrees[i] * 0.1);
            degrees[i] -= pull;
            mass_pool += pull;
        } else if degrees[i] > 0.0 && degrees[i] < s * 0.9 && degrees[i] < capacity * 0.9 {
            is_safe[i] = true;
        }
    }

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
// B Pair Statistics
// ---------------------------------------------------------------------------

/// Pair statistics for the Zero-Inflated Binomial(M) model.
///
/// Given q = exp(log_q) and v = exp(log_v):
/// - G_B(q) = (1+q)^M - 1
/// - Z = 1 + v * G_B(q)
/// - occupation = v * G_B(q) / Z
/// - expected_weight = v * M * q * (1+q)^(M-1) / Z
///
/// Returns (ln_z, occupation, expected_weight).
#[inline]
fn b_zip_pair_statistics(log_q: f64, log_v: f64, m: u32) -> (f64, f64, f64) {
    if log_q <= -690.0 || log_v <= -690.0 || m == 0 {
        return (0.0, 0.0, 0.0);
    }

    let q = log_q.exp();
    let v = log_v.exp();
    let m_f = f64::from(m);
    let one_plus_q = 1.0 + q;

    // G_B(q) = (1+q)^M - 1
    let g = one_plus_q.powi(m as i32) - 1.0;

    if g <= 0.0 {
        return (0.0, 0.0, 0.0);
    }

    let v_g = v * g;

    // Overflow protection: when v*G is very large
    if v_g > 1e15 {
        // Z ≈ v*G, ln(Z) ≈ log_v + ln(G)
        let ln_z = log_v + g.ln();
        let occupation = 1.0;
        // E[t] ≈ M*q*(1+q)^(M-1) / G = M*q/(1+q) * (1+q)^M / ((1+q)^M-1)
        let expected_weight = m_f * q * one_plus_q.powi(m as i32 - 1) / g;
        return (ln_z, occupation, expected_weight.min(m_f));
    }

    let z = 1.0 + v_g;
    let ln_z = z.ln();
    let occupation = v_g / z;
    // q * G'_B(q) = M * q * (1+q)^(M-1)
    let expected_weight = (v * m_f * q * one_plus_q.powi(m as i32 - 1) / z).min(m_f);

    (ln_z, occupation, expected_weight)
}

// ---------------------------------------------------------------------------
// O(N) Parallel NLL + Gradient Evaluator
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct RowAccumulator {
    grad_s_in: Vec<f64>,
    grad_k_in: Vec<f64>,
}

impl RowAccumulator {
    fn new(n: usize) -> Self {
        Self {
            grad_s_in: vec![0.0; n],
            grad_k_in: vec![0.0; n],
        }
    }

    fn merge(mut self, other: Self) -> Self {
        for (a, b) in self.grad_s_in.iter_mut().zip(other.grad_s_in.iter()) {
            *a += *b;
        }
        for (a, b) in self.grad_k_in.iter_mut().zip(other.grad_k_in.iter()) {
            *a += *b;
        }
        self
    }
}

fn b_nll_and_gradient(
    theta: &[f64],
    s_out_target: &[f64],
    s_in_target: &[f64],
    k_out_target: &[f64],
    k_in_target: &[f64],
    mask: &PairMask,
    layers: u32,
) -> (f64, Vec<f64>) {
    let n = s_out_target.len();
    debug_assert_eq!(theta.len(), 4 * n);

    let alpha_x = &theta[0..n];
    let alpha_y = &theta[n..2 * n];
    let alpha_v = &theta[2 * n..3 * n];
    let alpha_w = &theta[3 * n..4 * n];

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

                    let (ln_z, occupation, expected_weight) =
                        b_zip_pair_statistics(log_q, log_v, layers);

                    nll += ln_z;
                    s_out[i] += expected_weight;
                    k_out[i] += occupation;
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

    let mut nll = total_nll;
    for i in 0..n {
        nll -= alpha_x[i] * s_out_target[i];
        nll -= alpha_y[i] * s_in_target[i];
        nll -= alpha_v[i] * k_out_target[i];
        nll -= alpha_w[i] * k_in_target[i];
    }

    let mut grad = vec![0.0_f64; 4 * n];
    for i in 0..n {
        grad[i] = s_out_pred[i] - s_out_target[i];
        grad[n + i] = col_acc.grad_s_in[i] - s_in_target[i];
        grad[2 * n + i] = k_out_pred[i] - k_out_target[i];
        grad[3 * n + i] = col_acc.grad_k_in[i] - k_in_target[i];
    }

    (nll, grad)
}

// ---------------------------------------------------------------------------
// L-BFGS Optimizer (identical structure to ME, parameterized on layers)
// ---------------------------------------------------------------------------

fn lbfgs_direction(
    grad: &[f64],
    s_hist: &[Vec<f64>],
    y_hist: &[Vec<f64>],
    rho_hist: &[f64],
) -> Vec<f64> {
    let mut q = grad.to_vec();
    let m = s_hist.len();
    let mut alpha = vec![0.0; m];

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

    q.iter_mut().for_each(|v| *v = -*v);
    q
}

fn recenter_theta(theta: &mut [f64], n: usize) {
    let mean_x = theta[..n].iter().sum::<f64>() / n as f64;
    let mean_y = theta[n..2 * n].iter().sum::<f64>() / n as f64;
    let shift_xy = 0.5 * (mean_y - mean_x);
    for v in &mut theta[..n] {
        *v += shift_xy;
    }
    for v in &mut theta[n..2 * n] {
        *v -= shift_xy;
    }

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
    layers: u32,
) -> Option<(Vec<f64>, Vec<f64>, f64)> {
    let c1 = 1e-4;
    let mut step = 1.0_f64;
    let max_step_clamp = 5.0_f64;

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

        let (candidate_obj, candidate_grad) = b_nll_and_gradient(
            &candidate,
            s_out_target,
            s_in_target,
            k_out_target,
            k_in_target,
            mask,
            layers,
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

fn initialize_theta(
    strength_out: &[f64],
    strength_in: &[f64],
    k_out_target: &[f64],
    k_in_target: &[f64],
) -> Vec<f64> {
    let n = strength_out.len();
    let total_s = strength_out.iter().sum::<f64>().max(1.0);
    let total_k = k_out_target.iter().sum::<f64>().max(1.0);
    let scale_s = total_s.sqrt();
    let scale_k = (total_k / n.max(1) as f64).sqrt().max(0.1);

    let mut theta = vec![0.0_f64; 4 * n];

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
    for i in 0..n {
        theta[2 * n + i] = if k_out_target[i] > 0.0 {
            (k_out_target[i] / total_k * scale_k).max(1e-12).ln()
        } else {
            -690.0
        };
    }
    for j in 0..n {
        theta[3 * n + j] = if k_in_target[j] > 0.0 {
            (k_in_target[j] / total_k * scale_k).max(1e-12).ln()
        } else {
            -690.0
        };
    }

    recenter_theta(&mut theta, n);
    theta
}

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

/// Fit B (Binomial) strength-degree using L-BFGS optimization.
///
/// Minimizes the NLL of the grand-canonical Zero-Inflated Binomial(M) ensemble
/// with strength and degree constraints.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_strength_degree_binomial_lbfgs(
    strength_out: &[f64],
    strength_in: &[f64],
    degree_out: &[f64],
    degree_in: &[f64],
    layers: u32,
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

    let (k_out_target, k_in_target) = regularize_degree_targets(
        strength_out,
        strength_in,
        degree_out,
        degree_in,
        mask,
        layers,
    );

    let mut theta = initialize_theta(strength_out, strength_in, &k_out_target, &k_in_target);

    let memory = 10_usize;

    let (mut obj, mut grad) = b_nll_and_gradient(
        &theta,
        strength_out,
        strength_in,
        &k_out_target,
        &k_in_target,
        mask,
        layers,
    );

    let target_scale = strength_out
        .iter()
        .chain(strength_in.iter())
        .chain(k_out_target.iter())
        .chain(k_in_target.iter())
        .copied()
        .fold(1.0_f64, f64::max);
    let grad_tol = tolerance * target_scale;

    let grad_norm = grad.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if grad_norm < grad_tol {
        return theta_to_result(&theta, n, true, 0);
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
            let sd_direction: Vec<f64> = grad.iter().map(|g| -g).collect();
            let sd_deriv: f64 = grad.iter().map(|g| -g * g).sum();
            if sd_deriv >= 0.0 {
                break;
            }
            if let Some((next_theta, next_grad, _next_obj)) = backtracking_line_search(
                &theta,
                &sd_direction,
                sd_deriv,
                obj,
                strength_out,
                strength_in,
                &k_out_target,
                &k_in_target,
                mask,
                layers,
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
                obj = _next_obj;
            } else {
                break;
            }
            continue;
        }

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
            layers,
        ) else {
            if !s_hist.is_empty() {
                s_hist.clear();
                y_hist.clear();
                rho_hist.clear();
                continue;
            }
            break;
        };

        recenter_theta(&mut next_theta, n);

        let (recentered_obj, recentered_grad) = b_nll_and_gradient(
            &next_theta,
            strength_out,
            strength_in,
            &k_out_target,
            &k_in_target,
            mask,
            layers,
        );

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

        let new_grad_norm = grad.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if new_grad_norm < grad_tol {
            return theta_to_result(&theta, n, true, iter + 1);
        }
    }

    theta_to_result(&theta, n, false, max_iterations)
}

// ---------------------------------------------------------------------------
// Strength-Edges L-BFGS (2N+1 parameters: α_x, α_y, ln_λ)
// ---------------------------------------------------------------------------

fn b_strength_edges_nll_and_gradient(
    theta: &[f64],
    s_out_target: &[f64],
    s_in_target: &[f64],
    target_edges: f64,
    mask: &PairMask,
    layers: u32,
) -> (f64, Vec<f64>) {
    let n = s_out_target.len();
    debug_assert_eq!(theta.len(), 2 * n + 1);

    let alpha_x = &theta[0..n];
    let alpha_y = &theta[n..2 * n];
    let log_lam = theta[2 * n];

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
                    let (ln_z, occupation, expected_weight) =
                        b_zip_pair_statistics(log_q, log_lam, layers);
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

/// Fit B (Binomial) strength-edges using L-BFGS optimization.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_strength_edges_binomial_lbfgs(
    strength_out: &[f64],
    strength_in: &[f64],
    target_edges: f64,
    layers: u32,
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
    let (mut obj, mut grad) = b_strength_edges_nll_and_gradient(
        &theta,
        strength_out,
        strength_in,
        target_edges,
        mask,
        layers,
    );

    let target_scale = strength_out
        .iter()
        .chain(strength_in.iter())
        .copied()
        .fold(target_edges, f64::max);
    let grad_tol = tolerance * target_scale;

    if grad.iter().map(|v| v.abs()).fold(0.0_f64, f64::max) < grad_tol {
        return b_se_theta_to_result(&theta, n, true, 0);
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
            if let Some((next, ng, no)) = b_se_line_search(
                &theta,
                &sd,
                sd_deriv,
                obj,
                strength_out,
                strength_in,
                target_edges,
                mask,
                layers,
            ) {
                b_se_update_history(
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

        let Some((mut next, _ng, _no)) = b_se_line_search(
            &theta,
            &direction,
            directional_deriv,
            obj,
            strength_out,
            strength_in,
            target_edges,
            mask,
            layers,
        ) else {
            if !s_hist.is_empty() {
                s_hist.clear();
                y_hist.clear();
                rho_hist.clear();
                continue;
            }
            break;
        };

        // Recenter x/y
        let mx = next[..n].iter().sum::<f64>() / n as f64;
        let my = next[n..2 * n].iter().sum::<f64>() / n as f64;
        let sh = 0.5 * (my - mx);
        for v in &mut next[..n] {
            *v += sh;
        }
        for v in &mut next[n..2 * n] {
            *v -= sh;
        }

        let (ro, rg) = b_strength_edges_nll_and_gradient(
            &next,
            strength_out,
            strength_in,
            target_edges,
            mask,
            layers,
        );

        b_se_update_history(
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
            return b_se_theta_to_result(&theta, n, true, iter + 1);
        }
    }

    b_se_theta_to_result(&theta, n, false, max_iterations)
}

#[allow(clippy::too_many_arguments)]
fn b_se_line_search(
    theta: &[f64],
    direction: &[f64],
    directional_deriv: f64,
    current_obj: f64,
    s_out: &[f64],
    s_in: &[f64],
    target_edges: f64,
    mask: &PairMask,
    layers: u32,
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
        let (co, cg) =
            b_strength_edges_nll_and_gradient(&cand, s_out, s_in, target_edges, mask, layers);
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

fn b_se_theta_to_result(
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
fn b_se_update_history(
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

    fn b_zip_mean(v: f64, q: f64, m: u32) -> f64 {
        if v <= 0.0 || q <= 0.0 || m == 0 {
            return 0.0;
        }
        let m_f = f64::from(m);
        let one_plus_q = 1.0 + q;
        let g = one_plus_q.powi(m as i32) - 1.0;
        v * m_f * q * one_plus_q.powi(m as i32 - 1) / (1.0 + v * g)
    }

    fn b_zip_occupation(v: f64, q: f64, m: u32) -> f64 {
        if v <= 0.0 || q <= 0.0 || m == 0 {
            return 0.0;
        }
        let g = (1.0 + q).powi(m as i32) - 1.0;
        let vg = v * g;
        vg / (1.0 + vg)
    }

    #[test]
    fn b_pair_statistics_matches_explicit() {
        let q = 0.5_f64;
        let v = 0.8_f64;
        let m = 3u32;
        let log_q = q.ln();
        let log_v = v.ln();

        let (ln_z, occ, weight) = b_zip_pair_statistics(log_q, log_v, m);

        let g = (1.0 + q).powi(3) - 1.0;
        let expected_z = 1.0 + v * g;
        let expected_occ = v * g / expected_z;
        let expected_weight = v * 3.0 * q * (1.0 + q).powi(2) / expected_z;

        assert!((ln_z - expected_z.ln()).abs() < 1e-13);
        assert!((occ - expected_occ).abs() < 1e-13);
        assert!((weight - expected_weight).abs() < 1e-13);
    }

    #[test]
    fn gradient_finite_difference_check_b() {
        let n = 3;
        let m = 2u32;
        let s_out = vec![3.0, 5.0, 4.0];
        let s_in = vec![4.0, 4.5, 3.5];
        let k_out = vec![1.5, 2.0, 1.8];
        let k_in = vec![1.7, 1.8, 1.8];
        let mask = PairMask::from_self_loops(n, true);

        let theta = vec![
            -0.5, 0.3, 0.1, // alpha_x
            0.0, -0.2, 0.2, // alpha_y
            -1.0, -0.5, -0.8, // alpha_v
            -0.7, -0.9, -0.6, // alpha_w
        ];

        let (_obj, grad) = b_nll_and_gradient(&theta, &s_out, &s_in, &k_out, &k_in, &mask, m);

        let eps = 1e-7;
        for idx in 0..theta.len() {
            let mut theta_plus = theta.clone();
            theta_plus[idx] += eps;
            let (obj_plus, _) =
                b_nll_and_gradient(&theta_plus, &s_out, &s_in, &k_out, &k_in, &mask, m);

            let mut theta_minus = theta.clone();
            theta_minus[idx] -= eps;
            let (obj_minus, _) =
                b_nll_and_gradient(&theta_minus, &s_out, &s_in, &k_out, &k_in, &mask, m);

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
    fn lbfgs_b_recovers_n3_m3() {
        let n = 3;
        let m = 3u32;
        let x = [0.25_f64, 0.15, 0.2];
        let y = [0.2_f64, 0.15, 0.25];
        let z = [0.3_f64, 0.4, 0.35];
        let w = [0.35_f64, 0.3, 0.4];
        let mask = PairMask::from_self_loops(n, true);

        let mut s_out = vec![0.0; n];
        let mut s_in = vec![0.0; n];
        let mut k_out = vec![0.0; n];
        let mut k_in = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                let q = x[i] * y[j];
                let v = z[i] * w[j];
                k_out[i] += b_zip_occupation(v, q, m);
                k_in[j] += b_zip_occupation(v, q, m);
                s_out[i] += b_zip_mean(v, q, m);
                s_in[j] += b_zip_mean(v, q, m);
            }
        }

        let result =
            fit_strength_degree_binomial_lbfgs(&s_out, &s_in, &k_out, &k_in, m, &mask, 1e-8, 5000);

        assert!(result.converged, "B L-BFGS must converge on N=3, M=3");

        for i in 0..n {
            let row_s: f64 = (0..n)
                .map(|j| b_zip_mean(result.z[i] * result.w[j], result.x[i] * result.y[j], m))
                .sum();
            let row_k: f64 = (0..n)
                .map(|j| b_zip_occupation(result.z[i] * result.w[j], result.x[i] * result.y[j], m))
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
    fn lbfgs_b_no_self_loops_n5_m2() {
        let n = 5;
        let m = 2u32;
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
                k_out[i] += b_zip_occupation(v, q, m);
                k_in[j] += b_zip_occupation(v, q, m);
                s_out[i] += b_zip_mean(v, q, m);
                s_in[j] += b_zip_mean(v, q, m);
            }
        }

        let result =
            fit_strength_degree_binomial_lbfgs(&s_out, &s_in, &k_out, &k_in, m, &mask, 1e-6, 5000);

        assert!(result.converged, "B L-BFGS must converge on N=5, M=2");

        let tol = 1e-4;
        for i in 0..n {
            let row_s: f64 = (0..n)
                .filter(|&j| !mask.is_masked(i, j))
                .map(|j| b_zip_mean(result.z[i] * result.w[j], result.x[i] * result.y[j], m))
                .sum();
            let row_k: f64 = (0..n)
                .filter(|&j| !mask.is_masked(i, j))
                .map(|j| b_zip_occupation(result.z[i] * result.w[j], result.x[i] * result.y[j], m))
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

    #[test]
    fn lbfgs_b_differs_from_me() {
        // Use the same multipliers to generate targets for both B and ME,
        // then verify they produce different fitted parameters.
        let n = 3;
        let m = 3u32;
        let mask = PairMask::from_self_loops(n, true);
        let x = [0.25_f64, 0.15, 0.2];
        let y = [0.2_f64, 0.15, 0.25];
        let z = [0.3_f64, 0.4, 0.35];
        let w = [0.35_f64, 0.3, 0.4];

        // Generate B targets
        let mut s_out = vec![0.0; n];
        let mut s_in = vec![0.0; n];
        let mut k_out = vec![0.0; n];
        let mut k_in = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                let q = x[i] * y[j];
                let v = z[i] * w[j];
                k_out[i] += b_zip_occupation(v, q, m);
                k_in[j] += b_zip_occupation(v, q, m);
                s_out[i] += b_zip_mean(v, q, m);
                s_in[j] += b_zip_mean(v, q, m);
            }
        }

        let me_result = crate::fitting::me_lbfgs::fit_strength_degree_poisson_lbfgs(
            &s_out, &s_in, &k_out, &k_in, &mask, 1e-6, 5000,
        );
        let b_result =
            fit_strength_degree_binomial_lbfgs(&s_out, &s_in, &k_out, &k_in, m, &mask, 1e-6, 5000);

        assert!(me_result.converged, "ME must converge");
        assert!(b_result.converged, "B must converge");

        // Multipliers must differ (different model families)
        let diff: f64 = me_result
            .x
            .iter()
            .zip(b_result.x.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "ME and B(M=3) must produce different x: diff={diff}"
        );
    }

    // --- Strength-Edges tests ---

    #[test]
    fn b_se_gradient_finite_difference() {
        let n = 3;
        let m = 2u32;
        let s_out = vec![3.0, 5.0, 4.0];
        let s_in = vec![4.0, 4.5, 3.5];
        let target_edges = 4.0;
        let mask = PairMask::from_self_loops(n, true);
        let theta = vec![-0.5, 0.3, 0.1, 0.0, -0.2, 0.2, -0.5];

        let (_obj, grad) =
            b_strength_edges_nll_and_gradient(&theta, &s_out, &s_in, target_edges, &mask, m);
        let eps = 1e-7;
        for idx in 0..theta.len() {
            let mut tp = theta.clone();
            tp[idx] += eps;
            let (op, _) =
                b_strength_edges_nll_and_gradient(&tp, &s_out, &s_in, target_edges, &mask, m);
            let mut tm = theta.clone();
            tm[idx] -= eps;
            let (om, _) =
                b_strength_edges_nll_and_gradient(&tm, &s_out, &s_in, target_edges, &mask, m);
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
    fn b_se_lbfgs_recovers_constraints_n5() {
        let n = 5;
        let m = 3u32;
        let mask = PairMask::from_self_loops(n, false);
        let x = [0.3_f64, 0.5, 0.2, 0.4, 0.35];
        let y = [0.25_f64, 0.3, 0.45, 0.2, 0.35];
        let lam = 0.8_f64;
        let m_f = f64::from(m);
        let mut s_out = vec![0.0; n];
        let mut s_in = vec![0.0; n];
        let mut total_edges = 0.0;
        for i in 0..n {
            for j in 0..n {
                if mask.is_masked(i, j) {
                    continue;
                }
                let q = x[i] * y[j];
                let opq = 1.0 + q;
                let g = opq.powi(m as i32) - 1.0;
                if g <= 0.0 {
                    continue;
                }
                let vg = lam * g;
                let z = 1.0 + vg;
                let occ = vg / z;
                let wt = lam * m_f * q * opq.powi(m as i32 - 1) / z;
                s_out[i] += wt;
                s_in[j] += wt;
                total_edges += occ;
            }
        }

        let result =
            fit_strength_edges_binomial_lbfgs(&s_out, &s_in, total_edges, m, &mask, 1e-6, 5000);
        assert!(result.converged, "B L-BFGS SE must converge");

        let tol = 1e-3;
        for i in 0..n {
            let mut row_s = 0.0;
            for j in 0..n {
                if mask.is_masked(i, j) {
                    continue;
                }
                let q = result.x[i] * result.y[j];
                let opq = 1.0 + q;
                let g = opq.powi(m as i32) - 1.0;
                if g <= 0.0 {
                    continue;
                }
                row_s += result.lam * m_f * q * opq.powi(m as i32 - 1) / (1.0 + result.lam * g);
            }
            assert!(
                (row_s - s_out[i]).abs() < tol,
                "s_out[{i}]: expected {}, got {row_s}",
                s_out[i]
            );
        }
    }
}
