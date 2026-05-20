//! Binary and binomial fitting routines.

use super::support::{max_pair_delta, self_loop_mask};
use super::FitResult;

pub fn balance_masked_degree_bernoulli(
    degree_out: &[f64],
    degree_in: &[f64],
    mask: &[bool],
    tolerance: f64,
    max_iterations: usize,
) -> FitResult {
    let n = degree_out.len();
    let k_avg = degree_out.iter().sum::<f64>() / n.max(1) as f64;
    let n_free = (0..n * n).filter(|&idx| !mask[idx]).count() as f64 / n.max(1) as f64;
    let c = if k_avg < n_free {
        (k_avg / (n_free - k_avg).max(0.01)).sqrt()
    } else {
        1.0
    };
    let mut x: Vec<f64> = degree_out
        .iter()
        .map(|&k| {
            if k > 0.0 && k_avg > 0.0 {
                k / k_avg * c
            } else {
                0.0
            }
        })
        .collect();
    let mut y: Vec<f64> = degree_in
        .iter()
        .map(|&k| {
            if k > 0.0 && k_avg > 0.0 {
                k / k_avg * c
            } else {
                0.0
            }
        })
        .collect();

    for iter in 0..max_iterations {
        let old_x = x.clone();
        let old_y = y.clone();

        for j in 0..n {
            if degree_in[j] <= 0.0 {
                y[j] = 0.0;
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&i| !mask[i * n + j])
                .map(|i| {
                    let aux = 1.0 + x[i] * y[j];
                    x[i] / aux
                })
                .sum();
            y[j] = if denom > 0.0 {
                degree_in[j] / denom
            } else {
                0.0
            };
        }
        for i in 0..n {
            if degree_out[i] <= 0.0 {
                x[i] = 0.0;
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&j| !mask[i * n + j])
                .map(|j| {
                    let aux = 1.0 + x[i] * y[j];
                    y[j] / aux
                })
                .sum();
            x[i] = if denom > 0.0 {
                degree_out[i] / denom
            } else {
                0.0
            };
        }

        let delta = max_pair_delta(&x, &old_x, &y, &old_y);
        let mut max_err = 0.0_f64;
        for i in 0..n {
            let pred: f64 = (0..n)
                .filter(|&j| !mask[i * n + j])
                .map(|j| binary_probability(x[i], y[j]))
                .sum();
            max_err = max_err.max((pred - degree_out[i]).abs());
        }
        for j in 0..n {
            let pred: f64 = (0..n)
                .filter(|&i| !mask[i * n + j])
                .map(|i| binary_probability(x[i], y[j]))
                .sum();
            max_err = max_err.max((pred - degree_in[j]).abs());
        }
        if delta < tolerance || max_err < tolerance {
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

/// IPF balancing for directed Bernoulli fixed-degree constraints.
#[must_use]
pub fn balance_degree_bernoulli(
    degree_out: &[f64],
    degree_in: &[f64],
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> FitResult {
    let mask = self_loop_mask(degree_out.len(), self_loops);
    balance_masked_degree_bernoulli(degree_out, degree_in, &mask, tolerance, max_iterations)
}

pub(crate) fn binary_probability(x: f64, y: f64) -> f64 {
    let z = x * y;
    z / (1.0 + z)
}

/// Iterative proportional fitting for binomial(M) fixed-strength constraints.
#[must_use]
pub fn balance_strength_binomial(
    strength_out: &[f64],
    strength_in: &[f64],
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> FitResult {
    let mask = self_loop_mask(strength_out.len(), self_loops);
    balance_masked_strength_binomial(
        strength_out,
        strength_in,
        &mask,
        layers,
        tolerance,
        max_iterations,
    )
}

/// Masked binomial(M) IPF for partial-constraint fitting.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn balance_masked_strength_binomial(
    strength_out: &[f64],
    strength_in: &[f64],
    mask: &[bool],
    layers: u32,
    tolerance: f64,
    max_iterations: usize,
) -> FitResult {
    let n = strength_out.len();
    let m = f64::from(layers);
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
    let k_in: Vec<f64> = strength_in.iter().map(|&s| s / m).collect();
    let k_out: Vec<f64> = strength_out.iter().map(|&s| s / m).collect();

    for iter in 0..max_iterations {
        for j in 0..n {
            if k_in[j] <= 0.0 {
                continue;
            }
            let mut denom = 0.0;
            for i in 0..n {
                if mask[i * n + j] {
                    continue;
                }
                denom += x[i] / (1.0 + x[i] * y[j]);
            }
            y[j] = if denom > 0.0 { k_in[j] / denom } else { 0.0 };
        }
        for i in 0..n {
            if k_out[i] <= 0.0 {
                continue;
            }
            let mut denom = 0.0;
            for j in 0..n {
                if mask[i * n + j] {
                    continue;
                }
                denom += y[j] / (1.0 + x[i] * y[j]);
            }
            x[i] = if denom > 0.0 { k_out[i] / denom } else { 0.0 };
        }

        let mut max_err = 0.0_f64;
        for i in 0..n {
            let mut pred = 0.0;
            for j in 0..n {
                if mask[i * n + j] {
                    continue;
                }
                pred += m * x[i] * y[j] / (1.0 + x[i] * y[j]);
            }
            max_err = max_err.max((pred - strength_out[i]).abs());
        }
        for j in 0..n {
            let mut pred = 0.0;
            for i in 0..n {
                if mask[i * n + j] {
                    continue;
                }
                pred += m * x[i] * y[j] / (1.0 + x[i] * y[j]);
            }
            max_err = max_err.max((pred - strength_in[j]).abs());
        }
        if max_err < tolerance {
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
