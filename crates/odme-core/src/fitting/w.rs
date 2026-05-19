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
        neg_ln_1m_exp_neg, w_a, w_g, w_log_g, w_mean, w_occupation, w_positive_mean, w_zip_mean,
    };

    const TOL: f64 = 1e-12;

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
