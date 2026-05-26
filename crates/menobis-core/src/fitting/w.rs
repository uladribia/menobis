//! W-ensemble fitting kernels.
//!
//! This module contains scalar helpers shared by geometric (`M = 1`) and
//! negative-binomial (`M > 1`) W fitting. Solver-specific code should build on
//! these numerically stable kernels.

/// Compute `-ln(1 - exp(-r))` for `r > 0`.
///
/// This is the W independent-pair barrier term. It uses `expm1` near zero and
/// `ln_1p` away from zero to avoid cancellation.
#[must_use]
pub fn neg_ln_1m_exp_neg(r: f64) -> f64 {
    if !r.is_finite() || r <= 0.0 {
        return f64::NAN;
    }
    if r <= std::f64::consts::LN_2 {
        -((-r).exp_m1().abs()).ln()
    } else {
        -(-(-r).exp()).ln_1p()
    }
}

/// Compute `A_M(r) = (1 - exp(-r))^(-M)` for `r > 0` and `M >= 1`.
#[must_use]
pub fn w_a(r: f64, layers: u32) -> f64 {
    if layers == 0 {
        return f64::NAN;
    }
    (f64::from(layers) * neg_ln_1m_exp_neg(r)).exp()
}

/// Compute `G_M(r) = A_M(r) - 1` for `r > 0` and `M >= 1`.
#[must_use]
pub fn w_g(r: f64, layers: u32) -> f64 {
    if layers == 0 {
        return f64::NAN;
    }
    (f64::from(layers) * neg_ln_1m_exp_neg(r)).exp_m1()
}

/// Compute `ln(G_M(r))` for `r > 0` and `M >= 1`.
#[must_use]
pub fn w_log_g(r: f64, layers: u32) -> f64 {
    let log_a = f64::from(layers) * neg_ln_1m_exp_neg(r);
    log_a.exp_m1().ln()
}

/// Compute independent W expected weight `M exp(-r) / (1 - exp(-r))`.
#[must_use]
pub fn w_mean(r: f64, layers: u32) -> f64 {
    if !r.is_finite() || r <= 0.0 || layers == 0 {
        return f64::NAN;
    }
    f64::from(layers) * (-r).exp() / (-(-r).exp_m1())
}

/// Compute zero-inflated W occupation probability.
///
/// `v` is the occupation multiplier and `r` is the positive-weight inverse/log
/// parameter in `q = exp(-r)`.
#[must_use]
pub fn w_occupation(v: f64, r: f64, layers: u32) -> f64 {
    if !v.is_finite() || v < 0.0 {
        return f64::NAN;
    }
    let vg = v * w_g(r, layers);
    vg / (1.0 + vg)
}

/// Compute zero-inflated W expected weight.
#[must_use]
pub fn w_zip_mean(v: f64, r: f64, layers: u32) -> f64 {
    if !v.is_finite() || v < 0.0 || !r.is_finite() || r <= 0.0 || layers == 0 {
        return f64::NAN;
    }
    let q = (-r).exp();
    let m = f64::from(layers);
    let log_numerator_without_v = m.ln() + q.ln() - (m + 1.0) * (1.0 - q).ln();
    let numerator = v * log_numerator_without_v.exp();
    let denominator = 1.0 + v * w_g(r, layers);
    numerator / denominator
}

/// Compute positive-weight conditional mean for W degree-events models.
///
/// Formula: `M q / ((1 - q) * (1 - (1 - q)^M))`.
#[must_use]
pub fn w_positive_mean(q: f64, layers: u32) -> f64 {
    if !q.is_finite() || q <= 0.0 || q >= 1.0 || layers == 0 {
        return f64::NAN;
    }
    let m = f64::from(layers);
    let one_minus_q = 1.0 - q;
    m * q / (one_minus_q * (1.0 - one_minus_q.powf(m)))
}

use super::{
    WConicFitOptions, WFitStatus, WProblemMetrics, WStrengthCostFitResult,
    WStrengthDegreeFitResult, WStrengthEdgesFitResult, WStrengthFitResult, WStrengthResiduals,
};

/// Compute independent W fixed-strength residuals from inverse/log multipliers.
///
/// `a` and `b` are inverse/log variables with `r_ij = a_i + b_j` and
/// `q_ij = exp(-r_ij)`. The pair expectation is `M q_ij / (1 - q_ij)`.
#[must_use]
pub fn independent_strength_residuals(
    a: &[f64],
    b: &[f64],
    layers: u32,
    strength_out: &[f64],
    strength_in: &[f64],
    self_loops: bool,
) -> WStrengthResiduals {
    let n = strength_out.len();
    let mut predicted_out = vec![0.0; n];
    let mut predicted_in = vec![0.0; n];
    let mut min_margin = f64::INFINITY;
    let mut max_q = 0.0_f64;

    for (i, &ai) in a.iter().enumerate().take(n) {
        for (j, &bj) in b.iter().enumerate().take(n) {
            if !self_loops && i == j {
                continue;
            }
            let r = ai + bj;
            min_margin = min_margin.min(r);
            max_q = max_q.max((-r).exp());
            let expected = w_mean(r, layers);
            predicted_out[i] += expected;
            predicted_in[j] += expected;
        }
    }

    let mut max_abs = 0.0_f64;
    let mut total_abs = 0.0_f64;
    for i in 0..n {
        let out_abs = (predicted_out[i] - strength_out[i]).abs();
        let in_abs = (predicted_in[i] - strength_in[i]).abs();
        max_abs = max_abs.max(out_abs).max(in_abs);
        total_abs += out_abs + in_abs;
    }

    WStrengthResiduals {
        max_abs,
        total_abs,
        min_margin,
        max_q,
    }
}

#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn strength_edges_residuals(
    a: &[f64],
    b: &[f64],
    lam: f64,
    layers: u32,
    strength_out: &[f64],
    strength_in: &[f64],
    target_edges: f64,
    self_loops: bool,
) -> (WStrengthResiduals, f64) {
    let n = strength_out.len();
    let mut predicted_out = vec![0.0; n];
    let mut predicted_in = vec![0.0; n];
    let mut predicted_edges = 0.0;
    let mut min_margin = f64::INFINITY;
    let mut max_q = 0.0_f64;

    for (i, &ai) in a.iter().enumerate().take(n) {
        for (j, &bj) in b.iter().enumerate().take(n) {
            if !self_loops && i == j {
                continue;
            }
            let r = ai + bj;
            min_margin = min_margin.min(r);
            max_q = max_q.max((-r).exp());
            let mean = w_zip_mean(lam, r, layers);
            let occupation = w_occupation(lam, r, layers);
            predicted_out[i] += mean;
            predicted_in[j] += mean;
            predicted_edges += occupation;
        }
    }

    let mut max_abs = 0.0_f64;
    let mut total_abs = 0.0_f64;
    for i in 0..n {
        let out_abs = (predicted_out[i] - strength_out[i]).abs();
        let in_abs = (predicted_in[i] - strength_in[i]).abs();
        max_abs = max_abs.max(out_abs).max(in_abs);
        total_abs += out_abs + in_abs;
    }

    (
        WStrengthResiduals {
            max_abs,
            total_abs,
            min_margin,
            max_q,
        },
        predicted_edges - target_edges,
    )
}

#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn strength_cost_residuals(
    a: &[f64],
    b: &[f64],
    gamma: f64,
    costs: &std::collections::HashMap<(usize, usize), f64>,
    layers: u32,
    strength_out: &[f64],
    strength_in: &[f64],
    target_cost: f64,
    self_loops: bool,
) -> (WStrengthResiduals, f64) {
    let n = strength_out.len();
    let mut predicted_out = vec![0.0; n];
    let mut predicted_in = vec![0.0; n];
    let mut predicted_cost = 0.0;
    let mut min_margin = f64::INFINITY;
    let mut max_q = 0.0_f64;

    for (i, &ai) in a.iter().enumerate().take(n) {
        for (j, &bj) in b.iter().enumerate().take(n) {
            if !self_loops && i == j {
                continue;
            }
            let cost = costs.get(&(i, j)).copied().unwrap_or(0.0);
            let r = ai + bj + gamma * cost;
            min_margin = min_margin.min(r);
            max_q = max_q.max((-r).exp());
            let expected = w_mean(r, layers);
            predicted_out[i] += expected;
            predicted_in[j] += expected;
            predicted_cost += cost * expected;
        }
    }

    let mut max_abs = 0.0_f64;
    let mut total_abs = 0.0_f64;
    for i in 0..n {
        let out_abs = (predicted_out[i] - strength_out[i]).abs();
        let in_abs = (predicted_in[i] - strength_in[i]).abs();
        max_abs = max_abs.max(out_abs).max(in_abs);
        total_abs += out_abs + in_abs;
    }

    (
        WStrengthResiduals {
            max_abs,
            total_abs,
            min_margin,
            max_q,
        },
        predicted_cost - target_cost,
    )
}

fn initial_independent_strength_guess(
    strength_out: &[f64],
    strength_in: &[f64],
    layers: u32,
) -> (Vec<f64>, Vec<f64>) {
    let total = strength_out.iter().sum::<f64>().max(f64::EPSILON);
    let m = f64::from(layers);
    let x: Vec<f64> = strength_out
        .iter()
        .map(|&s| {
            (s / (s + m * total.sqrt()))
                .clamp(1e-12, 1.0 - 1e-12)
                .sqrt()
        })
        .collect();
    let y: Vec<f64> = strength_in
        .iter()
        .map(|&s| {
            (s / (s + m * total.sqrt()))
                .clamp(1e-12, 1.0 - 1e-12)
                .sqrt()
        })
        .collect();
    let a = x.iter().map(|&xi| -xi.ln()).collect();
    let b = y.iter().map(|&yj| -yj.ln()).collect();
    (a, b)
}

fn pair_count(n: usize, self_loops: bool) -> usize {
    if self_loops {
        n * n
    } else {
        n * n.saturating_sub(1)
    }
}

fn w_strength_edges_metrics(n: usize, self_loops: bool) -> WProblemMetrics {
    let pairs = pair_count(n, self_loops);
    WProblemMetrics {
        variables: 2 * n + 1,
        auxiliary_variables: pairs,
        exponential_cones: 2 * pairs,
        power_cones: 0,
        linear_constraints: 1,
        sparse_nonzeros: 5 * pairs + 2 * n,
    }
}

/// Start fixed-strength W geometric fitting.
///
/// This public kernel establishes validation, diagnostics, and the conic-solver
/// boundary. The actual conic solve is enabled behind the `w-conic` feature and
/// will replace the current not-yet-solved status as the Clarabel model is
/// assembled.
#[must_use]
pub fn fit_strength_geometric(
    strength_out: &[f64],
    strength_in: &[f64],
    opts: WConicFitOptions,
) -> WStrengthFitResult {
    let cost_opts = super::CostFitOptions {
        self_loops: opts.self_loops,
        tolerance: opts.tolerance,
        max_iterations: opts.max_iterations,
    };
    let result = super::w_lbfgs::fit_strength_w_newton(strength_out, strength_in, 1, &cost_opts);
    newton_to_w_strength_result(result, 1, strength_out, strength_in, opts)
}

/// Start fixed-strength W negative-binomial fitting.
#[must_use]
pub fn fit_strength_negative_binomial(
    strength_out: &[f64],
    strength_in: &[f64],
    layers: u32,
    opts: WConicFitOptions,
) -> WStrengthFitResult {
    let cost_opts = super::CostFitOptions {
        self_loops: opts.self_loops,
        tolerance: opts.tolerance,
        max_iterations: opts.max_iterations,
    };
    let result =
        super::w_lbfgs::fit_strength_w_newton(strength_out, strength_in, layers, &cost_opts);
    newton_to_w_strength_result(result, layers, strength_out, strength_in, opts)
}

// ---------------------------------------------------------------------------
// W strength-edges fitting
// ---------------------------------------------------------------------------

fn fit_strength_edges_w_not_solved(
    strength_out: &[f64],
    strength_in: &[f64],
    target_edges: f64,
    layers: u32,
    opts: WConicFitOptions,
    status: WFitStatus,
) -> WStrengthEdgesFitResult {
    let (a, b) = initial_independent_strength_guess(strength_out, strength_in, layers);
    let (residuals, edge_residual) = strength_edges_residuals(
        &a,
        &b,
        1.0,
        layers,
        strength_out,
        strength_in,
        target_edges,
        opts.self_loops,
    );
    WStrengthEdgesFitResult {
        x: a.iter().map(|&ai| (-ai).exp()).collect(),
        y: b.iter().map(|&bj| (-bj).exp()).collect(),
        lam: 1.0,
        layers,
        status,
        objective: f64::NAN,
        iterations: 0,
        min_margin: residuals.min_margin,
        max_q: residuals.max_q,
        max_strength_residual: residuals.max_abs,
        total_strength_residual: residuals.total_abs,
        edge_residual,
        metrics: w_strength_edges_metrics(strength_out.len(), opts.self_loops),
    }
}

#[must_use]
pub fn fit_strength_edges_geometric(
    strength_out: &[f64],
    strength_in: &[f64],
    target_edges: f64,
    opts: WConicFitOptions,
) -> WStrengthEdgesFitResult {
    fit_strength_edges_w(strength_out, strength_in, target_edges, 1, opts)
}

#[must_use]
pub fn fit_strength_edges_negative_binomial(
    strength_out: &[f64],
    strength_in: &[f64],
    target_edges: f64,
    layers: u32,
    opts: WConicFitOptions,
) -> WStrengthEdgesFitResult {
    fit_strength_edges_w(strength_out, strength_in, target_edges, layers, opts)
}

fn solve_w_positive_mean(avg_weight: f64, layers: u32) -> f64 {
    if layers == 1 {
        return solve_ztg_q(avg_weight);
    }
    solve_ztnb_q(avg_weight, layers)
}

fn strength_edges_pair_mean(x: f64, y: f64, lam: f64, layers: u32) -> f64 {
    let q = (x * y).clamp(0.0, 1.0 - 1e-14);
    if q <= 0.0 {
        return 0.0;
    }
    w_zip_mean(lam, -q.ln(), layers)
}

fn strength_edges_pair_occupation(x: f64, y: f64, lam: f64, layers: u32) -> f64 {
    let q = (x * y).clamp(0.0, 1.0 - 1e-14);
    if q <= 0.0 {
        return 0.0;
    }
    w_occupation(lam, -q.ln(), layers)
}

fn solve_strength_edges_factor(target: f64, other: &[f64], lam: f64, layers: u32) -> f64 {
    if target <= 0.0 {
        return 0.0;
    }
    let max_other = other.iter().copied().fold(0.0_f64, f64::max);
    if max_other <= 0.0 {
        return 0.0;
    }
    let mut low = 0.0_f64;
    let mut high = ((1.0 - 1e-12) / max_other).min(1e30);
    for _ in 0..60 {
        let mid = 0.5 * (low + high);
        let value: f64 = other
            .iter()
            .map(|&v| strength_edges_pair_mean(mid, v, lam, layers))
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

#[allow(clippy::too_many_arguments)]
fn balance_strength_edges_for_lambda(
    strength_out: &[f64],
    strength_in: &[f64],
    lam: f64,
    layers: u32,
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
            y[j] = solve_strength_edges_factor(strength_in[j], &others, lam, layers);
        }
        for i in 0..n {
            for j in 0..n {
                others[j] = if self_loops || i != j { y[j] } else { 0.0 };
            }
            x[i] = solve_strength_edges_factor(strength_out[i], &others, lam, layers);
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

fn strength_edges_expected_edges(
    x: &[f64],
    y: &[f64],
    lam: f64,
    layers: u32,
    self_loops: bool,
) -> f64 {
    let n = x.len();
    let mut total = 0.0;
    for (i, &xi) in x.iter().enumerate().take(n) {
        for (j, &yj) in y.iter().enumerate().take(n) {
            if self_loops || i != j {
                total += strength_edges_pair_occupation(xi, yj, lam, layers);
            }
        }
    }
    total
}

fn fit_strength_edges_w(
    strength_out: &[f64],
    strength_in: &[f64],
    target_edges: f64,
    layers: u32,
    opts: WConicFitOptions,
) -> WStrengthEdgesFitResult {
    let n = strength_out.len();
    let pairs = pair_count(n, opts.self_loops);
    let total_strength = strength_out.iter().sum::<f64>();
    if n == 0
        || n != strength_in.len()
        || layers == 0
        || target_edges <= 0.0
        || target_edges >= pairs as f64
        || target_edges > total_strength
    {
        return fit_strength_edges_w_not_solved(
            strength_out,
            strength_in,
            target_edges,
            layers,
            opts,
            WFitStatus::Infeasible,
        );
    }

    let avg_weight = total_strength / target_edges;
    let q = solve_w_positive_mean(avg_weight, layers).clamp(1e-15, 1.0 - 1e-15);
    let occupation = target_edges / pairs as f64;
    let g = w_g(-q.ln(), layers);
    let homogeneous_lam = occupation / ((1.0 - occupation) * g);

    let scale = total_strength.sqrt().max(1.0);
    let mut cur_x: Vec<f64> = strength_out
        .iter()
        .map(|&s| (s / (s + scale)).clamp(1e-12, 0.5))
        .collect();
    let mut cur_y: Vec<f64> = strength_in
        .iter()
        .map(|&s| (s / (s + scale)).clamp(1e-12, 0.5))
        .collect();

    let mut low = 1e-12_f64;
    let mut high = homogeneous_lam.max(1.0);
    let mut best = None;

    for _ in 0..80 {
        let (x, y, converged, iterations) = balance_strength_edges_for_lambda(
            strength_out,
            strength_in,
            high,
            layers,
            opts.self_loops,
            opts.tolerance,
            opts.max_iterations,
            &cur_x,
            &cur_y,
        );
        cur_x = x.clone();
        cur_y = y.clone();
        let edges = strength_edges_expected_edges(&x, &y, high, layers, opts.self_loops);
        best = Some((x, y, converged, iterations, high, edges));
        if edges >= target_edges || high > 1e30 {
            break;
        }
        low = high;
        high *= 2.0;
    }

    let mut total_iterations = 0;
    for _ in 0..60 {
        let mid = 0.5 * (low + high);
        let (x, y, converged, iterations) = balance_strength_edges_for_lambda(
            strength_out,
            strength_in,
            mid,
            layers,
            opts.self_loops,
            opts.tolerance,
            opts.max_iterations,
            &cur_x,
            &cur_y,
        );
        cur_x = x.clone();
        cur_y = y.clone();
        total_iterations += iterations;
        let edges = strength_edges_expected_edges(&x, &y, mid, layers, opts.self_loops);
        best = Some((x, y, converged, total_iterations, mid, edges));
        if (edges - target_edges).abs() <= opts.tolerance.max(1e-10) {
            break;
        }
        if edges < target_edges {
            low = mid;
        } else {
            high = mid;
        }
    }

    let Some((x, y, converged, iterations, lam, _)) = best else {
        return fit_strength_edges_w_not_solved(
            strength_out,
            strength_in,
            target_edges,
            layers,
            opts,
            WFitStatus::Failed,
        );
    };
    let a: Vec<f64> = x.iter().map(|&xi| -xi.ln()).collect();
    let b: Vec<f64> = y.iter().map(|&yi| -yi.ln()).collect();
    let (residuals, edge_residual) = strength_edges_residuals(
        &a,
        &b,
        lam,
        layers,
        strength_out,
        strength_in,
        target_edges,
        opts.self_loops,
    );
    let status = if converged
        && residuals.max_abs <= opts.tolerance.max(1e-6) * total_strength.max(1.0)
        && edge_residual.abs() <= opts.tolerance.max(1e-6) * target_edges.max(1.0)
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
        min_margin: residuals.min_margin,
        max_q: residuals.max_q,
        max_strength_residual: residuals.max_abs,
        total_strength_residual: residuals.total_abs,
        edge_residual,
        metrics: w_strength_edges_metrics(n, opts.self_loops),
    }
}

// ---------------------------------------------------------------------------
// W strength-degree fitting
// ---------------------------------------------------------------------------

fn w_strength_degree_metrics(n: usize, self_loops: bool) -> WProblemMetrics {
    let pairs = pair_count(n, self_loops);
    WProblemMetrics {
        variables: 4 * n,
        auxiliary_variables: pairs,
        exponential_cones: 0,
        power_cones: 0,
        linear_constraints: 2,
        sparse_nonzeros: 4 * n * pairs,
    }
}

#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn strength_degree_residuals(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    w: &[f64],
    layers: u32,
    strength_out: &[f64],
    strength_in: &[f64],
    degree_out: &[f64],
    degree_in: &[f64],
    self_loops: bool,
) -> (WStrengthResiduals, f64) {
    let n = strength_out.len();
    let mut pred_s_out = vec![0.0; n];
    let mut pred_s_in = vec![0.0; n];
    let mut pred_k_out = vec![0.0; n];
    let mut pred_k_in = vec![0.0; n];
    let mut min_margin = f64::INFINITY;
    let mut max_q = 0.0_f64;

    for i in 0..n {
        for j in 0..n {
            if !self_loops && i == j {
                continue;
            }
            let q = x[i] * y[j];
            max_q = max_q.max(q);
            let r = -(q.max(1e-300)).ln();
            min_margin = min_margin.min(r);
            let v = z[i] * w[j];
            let mean = w_zip_mean(v, r, layers);
            let occ = w_occupation(v, r, layers);
            pred_s_out[i] += mean;
            pred_s_in[j] += mean;
            pred_k_out[i] += occ;
            pred_k_in[j] += occ;
        }
    }

    let mut max_s = 0.0_f64;
    let mut total_s = 0.0_f64;
    let mut max_k = 0.0_f64;
    for i in 0..n {
        let so = (pred_s_out[i] - strength_out[i]).abs();
        let si = (pred_s_in[i] - strength_in[i]).abs();
        max_s = max_s.max(so).max(si);
        total_s += so + si;
        let ko = (pred_k_out[i] - degree_out[i]).abs();
        let ki = (pred_k_in[i] - degree_in[i]).abs();
        max_k = max_k.max(ko).max(ki);
    }

    (
        WStrengthResiduals {
            max_abs: max_s,
            total_abs: total_s,
            min_margin,
            max_q,
        },
        max_k,
    )
}

fn solve_strength_degree_factor_s(
    target: f64,
    other_q: &[f64],
    other_v: &[f64],
    lam_z: f64,
    layers: u32,
) -> f64 {
    if target <= 0.0 {
        return 0.0;
    }
    let max_oq = other_q.iter().copied().fold(0.0_f64, f64::max);
    if max_oq <= 0.0 {
        return 0.0;
    }
    let mut low = 0.0_f64;
    let mut high = ((1.0 - 1e-12) / max_oq).min(1e30);
    for _ in 0..60 {
        let mid = 0.5 * (low + high);
        let value: f64 = other_q
            .iter()
            .zip(other_v.iter())
            .map(|(&oq, &ov)| {
                let q = (mid * oq).clamp(0.0, 1.0 - 1e-14);
                if q <= 0.0 {
                    return 0.0;
                }
                w_zip_mean(lam_z * ov, -(q.ln()), layers)
            })
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

fn solve_strength_degree_factor_k(
    target: f64,
    other_q: &[f64],
    other_v: &[f64],
    lam_x: f64,
    layers: u32,
) -> f64 {
    if target <= 0.0 {
        return 0.0;
    }
    let mut low = 0.0_f64;
    let mut high = 1.0_f64;
    for _ in 0..80 {
        let value: f64 = other_q
            .iter()
            .zip(other_v.iter())
            .map(|(&oq, &ov)| {
                let q = (lam_x * oq).clamp(0.0, 1.0 - 1e-14);
                if q <= 0.0 {
                    return 0.0;
                }
                w_occupation(high * ov, -(q.ln()), layers)
            })
            .sum();
        if value >= target || high >= 1e30 {
            break;
        }
        high *= 2.0;
    }
    for _ in 0..80 {
        let mid = 0.5 * (low + high);
        let value: f64 = other_q
            .iter()
            .zip(other_v.iter())
            .map(|(&oq, &ov)| {
                let q = (lam_x * oq).clamp(0.0, 1.0 - 1e-14);
                if q <= 0.0 {
                    return 0.0;
                }
                w_occupation(mid * ov, -(q.ln()), layers)
            })
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

#[allow(clippy::too_many_arguments)]
fn balance_strength_degree_w(
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
    let total = strength_out.iter().sum::<f64>().max(1.0);
    let scale = total.sqrt();
    let mut x: Vec<f64> = strength_out
        .iter()
        .map(|&s| (s / (s + scale)).clamp(1e-12, 0.5))
        .collect();
    let mut y: Vec<f64> = strength_in
        .iter()
        .map(|&s| (s / (s + scale)).clamp(1e-12, 0.5))
        .collect();
    let k_total = degree_out.iter().sum::<f64>().max(1.0);
    let k_scale = (k_total / n.max(1) as f64).sqrt().max(0.1);
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
    let mut other_q = vec![0.0_f64; n];
    let mut other_v = vec![0.0_f64; n];

    for iter in 0..max_iterations {
        let old_x = x.clone();
        let old_y = y.clone();
        let old_z = z.clone();
        let old_w = w.clone();

        // Update y (strength_in)
        for j in 0..n {
            for i in 0..n {
                if self_loops || i != j {
                    other_q[i] = x[i];
                    other_v[i] = z[i] * w[j];
                } else {
                    other_q[i] = 0.0;
                    other_v[i] = 0.0;
                }
            }
            y[j] = solve_strength_degree_factor_s(strength_in[j], &other_q, &other_v, 1.0, layers);
        }
        // Update x (strength_out)
        for i in 0..n {
            for j in 0..n {
                if self_loops || i != j {
                    other_q[j] = y[j];
                    other_v[j] = z[i] * w[j];
                } else {
                    other_q[j] = 0.0;
                    other_v[j] = 0.0;
                }
            }
            x[i] = solve_strength_degree_factor_s(strength_out[i], &other_q, &other_v, 1.0, layers);
        }
        // Update w (degree_in). Degree-saturated nodes keep a large multiplier,
        // forcing occupation pi_ij -> 1 while x/y continue fitting positive weights.
        for j in 0..n {
            if saturated_in[j] {
                w[j] = 1e30;
                continue;
            }
            for i in 0..n {
                if self_loops || i != j {
                    other_q[i] = x[i] * y[j];
                    other_v[i] = z[i];
                } else {
                    other_q[i] = 0.0;
                    other_v[i] = 0.0;
                }
            }
            w[j] = solve_strength_degree_factor_k(degree_in[j], &other_q, &other_v, 1.0, layers);
        }
        // Update z (degree_out). Degree-saturated nodes keep a large multiplier.
        for i in 0..n {
            if saturated_out[i] {
                z[i] = 1e30;
                continue;
            }
            for j in 0..n {
                if self_loops || i != j {
                    other_q[j] = x[i] * y[j];
                    other_v[j] = w[j];
                } else {
                    other_q[j] = 0.0;
                    other_v[j] = 0.0;
                }
            }
            z[i] = solve_strength_degree_factor_k(degree_out[i], &other_q, &other_v, 1.0, layers);
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
            return (x, y, z, w, true, iter + 1);
        }
    }
    (x, y, z, w, false, max_iterations)
}

#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_strength_degree_geometric(
    strength_out: &[f64],
    strength_in: &[f64],
    degree_out: &[f64],
    degree_in: &[f64],
    opts: WConicFitOptions,
) -> WStrengthDegreeFitResult {
    fit_strength_degree_w(strength_out, strength_in, degree_out, degree_in, 1, opts)
}

#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_strength_degree_negative_binomial(
    strength_out: &[f64],
    strength_in: &[f64],
    degree_out: &[f64],
    degree_in: &[f64],
    layers: u32,
    opts: WConicFitOptions,
) -> WStrengthDegreeFitResult {
    fit_strength_degree_w(
        strength_out,
        strength_in,
        degree_out,
        degree_in,
        layers,
        opts,
    )
}

#[allow(clippy::too_many_arguments)]
fn fit_strength_degree_w(
    strength_out: &[f64],
    strength_in: &[f64],
    degree_out: &[f64],
    degree_in: &[f64],
    layers: u32,
    opts: WConicFitOptions,
) -> WStrengthDegreeFitResult {
    let n = strength_out.len();
    if n == 0
        || n != strength_in.len()
        || n != degree_out.len()
        || n != degree_in.len()
        || layers == 0
    {
        return WStrengthDegreeFitResult {
            x: vec![0.0; n],
            y: vec![0.0; n],
            z: vec![0.0; n],
            w: vec![0.0; n],
            layers,
            status: WFitStatus::Infeasible,
            objective: f64::NAN,
            iterations: 0,
            min_margin: 0.0,
            max_q: 0.0,
            max_strength_residual: 0.0,
            total_strength_residual: 0.0,
            max_degree_residual: 0.0,
            metrics: w_strength_degree_metrics(n, opts.self_loops),
        };
    }

    let (x, y, z, w, converged, iterations) = balance_strength_degree_w(
        strength_out,
        strength_in,
        degree_out,
        degree_in,
        layers,
        opts.self_loops,
        opts.tolerance,
        opts.max_iterations,
    );

    let total_strength = strength_out.iter().sum::<f64>().max(1.0);
    let total_degree = degree_out.iter().sum::<f64>().max(1.0);
    let (residuals, max_degree_residual) = strength_degree_residuals(
        &x,
        &y,
        &z,
        &w,
        layers,
        strength_out,
        strength_in,
        degree_out,
        degree_in,
        opts.self_loops,
    );
    let status = if converged
        && residuals.max_abs <= opts.tolerance.max(1e-6) * total_strength
        && max_degree_residual <= opts.tolerance.max(2e-6) * total_degree
    {
        WFitStatus::Solved
    } else {
        WFitStatus::Inaccurate
    };
    WStrengthDegreeFitResult {
        x,
        y,
        z,
        w,
        layers,
        status,
        objective: f64::NAN,
        iterations,
        min_margin: residuals.min_margin,
        max_q: residuals.max_q,
        max_strength_residual: residuals.max_abs,
        total_strength_residual: residuals.total_abs,
        max_degree_residual,
        metrics: w_strength_degree_metrics(n, opts.self_loops),
    }
}

// ---------------------------------------------------------------------------
// W strength-cost fitting
// ---------------------------------------------------------------------------

fn cost_map(
    cost_sources: &[u64],
    cost_targets: &[u64],
    cost_values: &[f64],
) -> std::collections::HashMap<(usize, usize), f64> {
    cost_sources
        .iter()
        .zip(cost_targets.iter())
        .zip(cost_values.iter())
        .map(|((&i, &j), &d)| ((i as usize, j as usize), d))
        .collect()
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn fit_strength_cost_geometric(
    strength_out: &[f64],
    strength_in: &[f64],
    cost_sources: &[u64],
    cost_targets: &[u64],
    cost_values: &[f64],
    target_cost: f64,
    opts: WConicFitOptions,
) -> WStrengthCostFitResult {
    let sources_usize: Vec<usize> = cost_sources.iter().map(|&s| s as usize).collect();
    let targets_usize: Vec<usize> = cost_targets.iter().map(|&t| t as usize).collect();
    let cost_opts = super::CostFitOptions {
        self_loops: opts.self_loops,
        tolerance: opts.tolerance,
        max_iterations: opts.max_iterations,
    };
    let result = super::w_lbfgs::fit_strength_cost_w_sparse_newton(
        strength_out,
        strength_in,
        &sources_usize,
        &targets_usize,
        cost_values,
        target_cost,
        1,
        &cost_opts,
    );
    newton_to_w_strength_cost_result(
        result,
        1,
        strength_out,
        strength_in,
        cost_sources,
        cost_targets,
        cost_values,
        target_cost,
        opts,
    )
}

#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_strength_cost_negative_binomial(
    strength_out: &[f64],
    strength_in: &[f64],
    cost_sources: &[u64],
    cost_targets: &[u64],
    cost_values: &[f64],
    target_cost: f64,
    layers: u32,
    opts: WConicFitOptions,
) -> WStrengthCostFitResult {
    let sources_usize: Vec<usize> = cost_sources.iter().map(|&s| s as usize).collect();
    let targets_usize: Vec<usize> = cost_targets.iter().map(|&t| t as usize).collect();
    let cost_opts = super::CostFitOptions {
        self_loops: opts.self_loops,
        tolerance: opts.tolerance,
        max_iterations: opts.max_iterations,
    };
    let result = super::w_lbfgs::fit_strength_cost_w_sparse_newton(
        strength_out,
        strength_in,
        &sources_usize,
        &targets_usize,
        cost_values,
        target_cost,
        layers,
        &cost_opts,
    );
    newton_to_w_strength_cost_result(
        result,
        layers,
        strength_out,
        strength_in,
        cost_sources,
        cost_targets,
        cost_values,
        target_cost,
        opts,
    )
}

// ---------------------------------------------------------------------------
// Newton result converters
// ---------------------------------------------------------------------------

fn newton_to_w_strength_result(
    result: super::StrengthCostFitResult,
    layers: u32,
    strength_out: &[f64],
    strength_in: &[f64],
    opts: WConicFitOptions,
) -> WStrengthFitResult {
    let n = strength_out.len();
    let status = if result.converged {
        WFitStatus::Solved
    } else {
        WFitStatus::Inaccurate
    };
    // Convert probability-space x,y to log-space a,b for residual computation
    let a: Vec<f64> = result
        .x
        .iter()
        .map(|&xi| if xi > 0.0 { -xi.ln() } else { 50.0 })
        .collect();
    let b: Vec<f64> = result
        .y
        .iter()
        .map(|&yj| if yj > 0.0 { -yj.ln() } else { 50.0 })
        .collect();
    let residuals =
        independent_strength_residuals(&a, &b, layers, strength_out, strength_in, opts.self_loops);
    WStrengthFitResult {
        x: result.x,
        y: result.y,
        layers,
        status,
        objective: 0.0,
        iterations: result.iterations,
        min_margin: residuals.min_margin,
        max_q: residuals.max_q,
        max_strength_residual: residuals.max_abs,
        total_strength_residual: residuals.total_abs,
        metrics: WProblemMetrics {
            variables: 2 * n,
            auxiliary_variables: 0,
            exponential_cones: 0,
            power_cones: 0,
            linear_constraints: 2 * n,
            sparse_nonzeros: 0,
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn newton_to_w_strength_cost_result(
    result: super::StrengthCostFitResult,
    layers: u32,
    strength_out: &[f64],
    strength_in: &[f64],
    cost_sources: &[u64],
    cost_targets: &[u64],
    cost_values: &[f64],
    target_cost: f64,
    opts: WConicFitOptions,
) -> WStrengthCostFitResult {
    let n = strength_out.len();
    let status = if result.converged {
        WFitStatus::Solved
    } else {
        WFitStatus::Inaccurate
    };
    let a: Vec<f64> = result
        .x
        .iter()
        .map(|&xi| if xi > 0.0 { -xi.ln() } else { 50.0 })
        .collect();
    let b: Vec<f64> = result
        .y
        .iter()
        .map(|&yj| if yj > 0.0 { -yj.ln() } else { 50.0 })
        .collect();
    let costs = cost_map(cost_sources, cost_targets, cost_values);
    let (residuals, cost_residual) = strength_cost_residuals(
        &a,
        &b,
        result.gamma,
        &costs,
        layers,
        strength_out,
        strength_in,
        target_cost,
        opts.self_loops,
    );
    WStrengthCostFitResult {
        x: result.x,
        y: result.y,
        gamma: result.gamma,
        layers,
        status,
        objective: 0.0,
        iterations: result.iterations,
        min_margin: residuals.min_margin,
        max_q: residuals.max_q,
        max_strength_residual: residuals.max_abs,
        total_strength_residual: residuals.total_abs,
        cost_residual,
        metrics: WProblemMetrics {
            variables: 2 * n + 1,
            auxiliary_variables: 0,
            exponential_cones: 0,
            power_cones: 0,
            linear_constraints: 2 * n + 1,
            sparse_nonzeros: 0,
        },
    }
}

// ---------------------------------------------------------------------------
// W degree-events fitting
// ---------------------------------------------------------------------------

use super::b::balance_degree_bernoulli;

pub struct DegreeEventsFitResult {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub q: f64,
    pub positive_mean: f64,
    pub converged: bool,
    pub iterations: usize,
}

/// Solve q from positive geometric mean = 1/(1-q) = avg_weight.
fn solve_ztg_q(avg_weight: f64) -> f64 {
    // positive geometric mean = 1/(1-q), so q = 1 - 1/avg_weight
    (1.0 - 1.0 / avg_weight).clamp(0.0, 1.0 - 1e-15)
}

/// Solve q from positive negative binomial(M) mean = Mq/((1-q)(1-(1-q)^M)) = avg_weight via bisection.
fn solve_ztnb_q(avg_weight: f64, layers: u32) -> f64 {
    let m = f64::from(layers);
    let mut low = 1e-15_f64;
    let mut high = 1.0 - 1e-15_f64;
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        let p0 = (1.0 - mid).powi(layers as i32);
        let val = m * mid / ((1.0 - mid) * (1.0 - p0));
        if val < avg_weight {
            low = mid;
        } else {
            high = mid;
        }
        if high - low < 1e-15 {
            break;
        }
    }
    0.5 * (low + high)
}

/// Fit the W degree-events geometric model.
///
/// Decomposes into:
/// 1. Solve q from positive geometric mean = T/E (analytic).
/// 2. Fit occupation via standard Bernoulli degree IPF.
#[must_use]
pub fn fit_degree_events_geometric(
    degree_out: &[f64],
    degree_in: &[f64],
    total_events: u64,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> DegreeEventsFitResult {
    let e: f64 = degree_out.iter().sum();
    let t = total_events as f64;
    let avg_weight = if e > 0.0 { t / e } else { 1.0 };
    let q = solve_ztg_q(avg_weight);
    let fit =
        balance_degree_bernoulli(degree_out, degree_in, self_loops, tolerance, max_iterations);
    DegreeEventsFitResult {
        x: fit.x,
        y: fit.y,
        q,
        positive_mean: avg_weight,
        converged: fit.converged,
        iterations: fit.iterations,
    }
}

/// Fit the W degree-events negative binomial(M) model.
///
/// Decomposes into:
/// 1. Solve q from positive negative binomial(M) mean = T/E via bisection.
/// 2. Fit occupation via standard Bernoulli degree IPF.
#[must_use]
pub fn fit_degree_events_negative_binomial(
    degree_out: &[f64],
    degree_in: &[f64],
    total_events: u64,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> DegreeEventsFitResult {
    let e: f64 = degree_out.iter().sum();
    let t = total_events as f64;
    let avg_weight = if e > 0.0 { t / e } else { 1.0 };
    let q = solve_ztnb_q(avg_weight, layers);
    let fit =
        balance_degree_bernoulli(degree_out, degree_in, self_loops, tolerance, max_iterations);
    DegreeEventsFitResult {
        x: fit.x,
        y: fit.y,
        q,
        positive_mean: avg_weight,
        converged: fit.converged,
        iterations: fit.iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        independent_strength_residuals, neg_ln_1m_exp_neg, w_a, w_g, w_log_g, w_mean, w_occupation,
        w_positive_mean, w_zip_mean,
    };
    use crate::fitting::{
        fit_strength_geometric, WConicFitOptions, WFitStatus, WProblemMetrics, WStrengthFitResult,
    };

    const TOL: f64 = 1e-12;

    #[test]
    fn w_fit_result_carries_solver_diagnostics() {
        let result = WStrengthFitResult {
            x: vec![0.5],
            y: vec![0.25],
            layers: 1,
            status: WFitStatus::Solved,
            objective: 1.25,
            iterations: 7,
            min_margin: 0.1,
            max_q: 0.9,
            max_strength_residual: 1e-9,
            total_strength_residual: 2e-9,
            metrics: WProblemMetrics {
                variables: 2,
                auxiliary_variables: 3,
                exponential_cones: 4,
                power_cones: 0,
                linear_constraints: 1,
                sparse_nonzeros: 8,
            },
        };

        assert_eq!(result.status, WFitStatus::Solved);
        assert_eq!(result.layers, 1);
        assert_eq!(result.metrics.exponential_cones, 4);
        assert!(result.max_q < 1.0);
    }

    #[test]
    fn independent_strength_residuals_recover_homogeneous_self_loop_fit() {
        let strength_out = [2.0, 2.0];
        let strength_in = [2.0, 2.0];
        let q = 0.5_f64;
        let a = -q.sqrt().ln();
        let b = -q.sqrt().ln();
        let residuals =
            independent_strength_residuals(&[a, a], &[b, b], 1, &strength_out, &strength_in, true);

        assert!(residuals.max_abs <= TOL);
        assert!(residuals.total_abs <= TOL);
        assert!((residuals.min_margin + q.ln()).abs() <= TOL);
        assert!((residuals.max_q - q).abs() <= TOL);
    }

    #[test]
    fn fixed_strength_geometric_reports_problem_metrics() {
        let result = fit_strength_geometric(
            &[2.0, 2.0],
            &[2.0, 2.0],
            WConicFitOptions {
                self_loops: true,
                tolerance: 1e-6,
                max_iterations: 5000,
            },
        );

        assert_eq!(result.layers, 1);
        assert_eq!(result.metrics.variables, 4);
        assert_eq!(result.metrics.auxiliary_variables, 0);
        assert!(matches!(
            result.status,
            WFitStatus::Solved | WFitStatus::Failed | WFitStatus::Inaccurate
        ));
        if matches!(result.status, WFitStatus::Solved) {
            assert!(result.max_q < 1.0);
        }
    }

    #[test]
    fn independent_geometric_kernels_match_closed_forms() {
        let r = 2.0_f64;
        let q = (-r).exp();

        assert!((neg_ln_1m_exp_neg(r) + (1.0 - q).ln()).abs() < TOL);
        assert!((w_a(r, 1) - 1.0 / (1.0 - q)).abs() < TOL);
        assert!((w_g(r, 1) - q / (1.0 - q)).abs() < TOL);
        assert!((w_log_g(r, 1) - (q / (1.0 - q)).ln()).abs() < TOL);
        assert!((w_mean(r, 1) - q / (1.0 - q)).abs() < TOL);
    }

    #[test]
    fn independent_negative_binomial_kernels_match_closed_forms() {
        let r = 1.5_f64;
        let layers = 3_u32;
        let m = f64::from(layers);
        let q = (-r).exp();
        let a = (1.0 - q).powf(-m);

        assert!((w_a(r, layers) - a).abs() < TOL);
        assert!((w_g(r, layers) - (a - 1.0)).abs() < TOL);
        assert!((w_log_g(r, layers) - (a - 1.0).ln()).abs() < TOL);
        assert!((w_mean(r, layers) - m * q / (1.0 - q)).abs() < TOL);
    }

    #[test]
    fn zero_inflated_kernels_match_partition_equations() {
        let r = 1.25_f64;
        let layers = 4_u32;
        let v = 0.7_f64;
        let q = (-r).exp();
        let m = f64::from(layers);
        let g = (1.0 - q).powf(-m) - 1.0;
        let z = 1.0 + v * g;
        let expected_pi = v * g / z;
        let expected_mean = v * m * q * (1.0 - q).powf(-m - 1.0) / z;

        assert!((w_occupation(v, r, layers) - expected_pi).abs() < TOL);
        assert!((w_zip_mean(v, r, layers) - expected_mean).abs() < TOL);
    }

    #[test]
    fn positive_mean_conditions_on_positive_weight() {
        let q = 0.25_f64;
        let layers = 3_u32;
        let m = f64::from(layers);
        let expected = m * q / ((1.0 - q) * (1.0 - (1.0 - q).powf(m)));

        assert!((w_positive_mean(q, layers) - expected).abs() < TOL);
        assert!((w_positive_mean(q, 1) - 1.0 / (1.0 - q)).abs() < TOL);
    }

    #[test]
    fn kernels_are_stable_for_large_r() {
        let r = 80.0_f64;

        assert!(neg_ln_1m_exp_neg(r).is_finite());
        assert!(w_g(r, 2).is_finite());
        assert!(w_log_g(r, 2).is_finite());
        assert!(w_mean(r, 2).is_finite());
        assert!(w_g(r, 2) > 0.0);
        assert!(w_log_g(r, 2) < 0.0);
    }
}
