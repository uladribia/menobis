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

use super::mask::PairMask;
use super::{
    WConicFitOptions, WFitStatus, WProblemMetrics, WStrengthDegreeFitResult,
    WStrengthEdgesFitResult, WStrengthFitResult, WStrengthResiduals,
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

/// Fixed-strength W geometric fitting.
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

/// Fixed-strength W negative-binomial fitting.
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

/// Fixed strength-edges W geometric fitting.
#[must_use]
pub fn fit_strength_edges_geometric(
    strength_out: &[f64],
    strength_in: &[f64],
    target_edges: f64,
    opts: WConicFitOptions,
) -> WStrengthEdgesFitResult {
    let n = strength_out.len();
    let mask = PairMask::from_self_loops(n, opts.self_loops);
    super::w_lbfgs::fit_strength_edges_w_lbfgs(
        strength_out,
        strength_in,
        target_edges,
        1,
        &mask,
        opts.tolerance,
        opts.max_iterations,
    )
}

/// Fixed strength-edges W negative-binomial fitting.
#[must_use]
pub fn fit_strength_edges_negative_binomial(
    strength_out: &[f64],
    strength_in: &[f64],
    target_edges: f64,
    layers: u32,
    opts: WConicFitOptions,
) -> WStrengthEdgesFitResult {
    let n = strength_out.len();
    let mask = PairMask::from_self_loops(n, opts.self_loops);
    super::w_lbfgs::fit_strength_edges_w_lbfgs(
        strength_out,
        strength_in,
        target_edges,
        layers,
        &mask,
        opts.tolerance,
        opts.max_iterations,
    )
}

// ---------------------------------------------------------------------------
// W strength-degree fitting
// ---------------------------------------------------------------------------

fn pair_count(n: usize, self_loops: bool) -> usize {
    if self_loops {
        n * n
    } else {
        n * n.saturating_sub(1)
    }
}

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

    let (x, y, z, w, converged, iterations) = super::w_lbfgs::fit_strength_degree_w_newton(
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
