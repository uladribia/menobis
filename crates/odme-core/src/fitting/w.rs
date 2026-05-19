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
        -(-r).exp_m1().ln()
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
    WConicFitOptions, WFitStatus, WProblemMetrics, WStrengthFitResult, WStrengthResiduals,
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

fn w_strength_metrics(n: usize, self_loops: bool) -> WProblemMetrics {
    let pairs = if self_loops {
        n * n
    } else {
        n * n.saturating_sub(1)
    };
    WProblemMetrics {
        variables: 2 * n,
        auxiliary_variables: pairs,
        exponential_cones: 2 * pairs,
        power_cones: 0,
        linear_constraints: 1,
        sparse_nonzeros: 4 * pairs + 2 * n,
    }
}

fn fit_strength_w_not_solved(
    strength_out: &[f64],
    strength_in: &[f64],
    layers: u32,
    opts: WConicFitOptions,
    status: WFitStatus,
) -> WStrengthFitResult {
    let (a, b) = initial_independent_strength_guess(strength_out, strength_in, layers);
    let residuals =
        independent_strength_residuals(&a, &b, layers, strength_out, strength_in, opts.self_loops);
    WStrengthFitResult {
        x: a.iter().map(|&ai| (-ai).exp()).collect(),
        y: b.iter().map(|&bj| (-bj).exp()).collect(),
        layers,
        status,
        objective: f64::NAN,
        iterations: 0,
        min_margin: residuals.min_margin,
        max_q: residuals.max_q,
        max_strength_residual: residuals.max_abs,
        total_strength_residual: residuals.total_abs,
        metrics: w_strength_metrics(strength_out.len(), opts.self_loops),
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
    fit_strength_w(strength_out, strength_in, 1, opts)
}

/// Start fixed-strength W negative-binomial fitting.
#[must_use]
pub fn fit_strength_negative_binomial(
    strength_out: &[f64],
    strength_in: &[f64],
    layers: u32,
    opts: WConicFitOptions,
) -> WStrengthFitResult {
    fit_strength_w(strength_out, strength_in, layers, opts)
}

#[cfg(not(feature = "w-conic"))]
fn fit_strength_w(
    strength_out: &[f64],
    strength_in: &[f64],
    layers: u32,
    opts: WConicFitOptions,
) -> WStrengthFitResult {
    fit_strength_w_not_solved(strength_out, strength_in, layers, opts, WFitStatus::Failed)
}

#[cfg(feature = "w-conic")]
fn fit_strength_w(
    strength_out: &[f64],
    strength_in: &[f64],
    layers: u32,
    opts: WConicFitOptions,
) -> WStrengthFitResult {
    use clarabel::algebra::CscMatrix;
    use clarabel::solver::{
        DefaultSettings, DefaultSolver, ExponentialConeT, IPSolver, NonnegativeConeT, SolverStatus,
        SupportedConeT, ZeroConeT,
    };

    let n = strength_out.len();
    if n == 0 || n != strength_in.len() || layers == 0 {
        return fit_strength_w_not_solved(
            strength_out,
            strength_in,
            layers,
            opts,
            WFitStatus::Infeasible,
        );
    }

    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .filter(|&(i, j)| opts.self_loops || i != j)
        .collect();
    let pair_count = pairs.len();
    let variable_count = 2 * n + 3 * pair_count;
    let row_count = 1 + 6 * pair_count + pair_count;
    let t_offset = 2 * n;
    let p_offset = t_offset + pair_count;
    let q_offset = p_offset + pair_count;
    let mut dense_a = vec![vec![0.0_f64; variable_count]; row_count];
    let mut rhs = vec![0.0_f64; row_count];
    let mut row = 0;

    // Gauge: sum(a) - sum(b) = 0.
    for i in 0..n {
        dense_a[row][i] = 1.0;
        dense_a[row][n + i] = -1.0;
    }
    row += 1;

    for (pair_idx, &(i, j)) in pairs.iter().enumerate() {
        let t_idx = t_offset + pair_idx;
        let p_idx = p_offset + pair_idx;
        let q_idx = q_offset + pair_idx;

        // p_ij >= exp(-t_ij): slack (-t, 1, p) in K_exp.
        dense_a[row][t_idx] = 1.0;
        row += 1;
        rhs[row] = 1.0;
        row += 1;
        dense_a[row][p_idx] = -1.0;
        row += 1;

        // q_ij >= exp(-(a_i + b_j)): slack (-(a+b), 1, q) in K_exp.
        dense_a[row][i] = 1.0;
        dense_a[row][n + j] = 1.0;
        row += 1;
        rhs[row] = 1.0;
        row += 1;
        dense_a[row][q_idx] = -1.0;
        row += 1;
    }

    // exp(-t_ij) + exp(-r_ij) <= 1.
    for pair_idx in 0..pair_count {
        dense_a[row][p_offset + pair_idx] = 1.0;
        dense_a[row][q_offset + pair_idx] = 1.0;
        rhs[row] = 1.0;
        row += 1;
    }

    let mut objective = vec![0.0_f64; variable_count];
    objective[..n].copy_from_slice(strength_out);
    objective[n..2 * n].copy_from_slice(strength_in);
    for pair_idx in 0..pair_count {
        objective[t_offset + pair_idx] = f64::from(layers);
    }

    let p_matrix = CscMatrix::<f64>::zeros((variable_count, variable_count));
    let a_matrix = dense_to_csc(&dense_a, row_count, variable_count);
    let mut cones: Vec<SupportedConeT<f64>> = Vec::with_capacity(2 * pair_count + 2);
    cones.push(ZeroConeT(1));
    for _ in 0..(2 * pair_count) {
        cones.push(ExponentialConeT());
    }
    cones.push(NonnegativeConeT(pair_count));
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
        return fit_strength_w_not_solved(
            strength_out,
            strength_in,
            layers,
            opts,
            WFitStatus::Failed,
        );
    };
    solver.solve();
    let status = match solver.solution.status {
        SolverStatus::Solved => WFitStatus::Solved,
        SolverStatus::AlmostSolved | SolverStatus::MaxIterations | SolverStatus::MaxTime => {
            WFitStatus::Inaccurate
        }
        SolverStatus::PrimalInfeasible
        | SolverStatus::DualInfeasible
        | SolverStatus::AlmostPrimalInfeasible
        | SolverStatus::AlmostDualInfeasible => WFitStatus::Infeasible,
        _ => WFitStatus::Failed,
    };
    if !matches!(status, WFitStatus::Solved | WFitStatus::Inaccurate) {
        return fit_strength_w_not_solved(strength_out, strength_in, layers, opts, status);
    }

    let solution = &solver.solution.x;
    let a = solution[..n].to_vec();
    let b = solution[n..2 * n].to_vec();
    let residuals =
        independent_strength_residuals(&a, &b, layers, strength_out, strength_in, opts.self_loops);
    WStrengthFitResult {
        x: a.iter().map(|&ai| (-ai).exp()).collect(),
        y: b.iter().map(|&bj| (-bj).exp()).collect(),
        layers,
        status,
        objective: solver.solution.obj_val,
        iterations: solver.solution.iterations as usize,
        min_margin: residuals.min_margin,
        max_q: residuals.max_q,
        max_strength_residual: residuals.max_abs,
        total_strength_residual: residuals.total_abs,
        metrics: w_strength_metrics(n, opts.self_loops),
    }
}

#[cfg(feature = "w-conic")]
fn dense_to_csc(dense: &[Vec<f64>], rows: usize, cols: usize) -> clarabel::algebra::CscMatrix<f64> {
    let mut colptr = Vec::with_capacity(cols + 1);
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    colptr.push(0);
    for col in 0..cols {
        for (row, values) in dense.iter().enumerate().take(rows) {
            let value = values[col];
            if value != 0.0 {
                rowval.push(row);
                nzval.push(value);
            }
        }
        colptr.push(rowval.len());
    }
    clarabel::algebra::CscMatrix::new(rows, cols, colptr, rowval, nzval)
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
                tolerance: 1e-9,
                max_iterations: 100,
            },
        );

        assert_eq!(result.layers, 1);
        assert_eq!(result.metrics.variables, 4);
        assert_eq!(result.metrics.auxiliary_variables, 4);
        assert!(matches!(
            result.status,
            WFitStatus::Solved | WFitStatus::Failed | WFitStatus::Inaccurate
        ));
        if result.status == WFitStatus::Solved {
            assert!(result.max_strength_residual < 1e-4);
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
