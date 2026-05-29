//! Full-pipeline partial-constraint fitting.
//!
//! Each function takes raw inputs (sequences, known pairs, options) and returns
//! a sparse rate table. All mask building, excess computation, balancing, IPF,
//! and result assembly happens in Rust.

use super::mask::PairMask;
use super::support::max_pair_delta;
use super::{
    balance_sparse_masked_degree_bernoulli, balance_sparse_masked_strength_degree_poisson,
    balance_sparse_masked_strength_poisson, fit_strength_cost_binomial_coordinates,
    fit_strength_cost_w_lbfgs, CostFitOptions, FitResult, PartialFitResult, StrengthCostFitResult,
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn compute_excess(
    out_seq: &[f64],
    in_seq: &[f64],
    known_src: &[u64],
    known_tgt: &[u64],
    known_contrib: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut excess_out: Vec<f64> = out_seq.to_vec();
    let mut excess_in: Vec<f64> = in_seq.to_vec();
    for ((&s, &t), &c) in known_src
        .iter()
        .zip(known_tgt.iter())
        .zip(known_contrib.iter())
    {
        let si = s as usize;
        let ti = t as usize;
        if si < excess_out.len() {
            excess_out[si] -= c;
        }
        if ti < excess_in.len() {
            excess_in[ti] -= c;
        }
    }
    // Check feasibility
    if excess_out.iter().any(|&v| v < -1e-6) || excess_in.iter().any(|&v| v < -1e-6) {
        return None;
    }
    // Clamp to zero
    for v in excess_out.iter_mut() {
        *v = v.max(0.0);
    }
    for v in excess_in.iter_mut() {
        *v = v.max(0.0);
    }
    Some((excess_out, excess_in))
}

fn balance_excess(excess_out: &mut [f64], excess_in: &mut [f64]) {
    let diff: f64 = excess_out.iter().sum::<f64>() - excess_in.iter().sum::<f64>();
    if diff.abs() > 1e-6 {
        if diff > 0.0 {
            if let Some(idx) = excess_in
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
            {
                excess_in[idx] += diff;
            }
        } else if let Some(idx) = excess_out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
        {
            excess_out[idx] -= diff;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn assemble_result_sparse(
    n: usize,
    known_src: &[u64],
    known_tgt: &[u64],
    known_rate: &[f64],
    mask: &PairMask,
    free_rate_fn: impl Fn(usize, usize) -> f64,
    converged: bool,
    iterations: usize,
) -> PartialFitResult {
    let mut sources = Vec::new();
    let mut targets = Vec::new();
    let mut rates = Vec::new();
    // Known pairs
    for ((&s, &t), &r) in known_src
        .iter()
        .zip(known_tgt.iter())
        .zip(known_rate.iter())
    {
        if r > 0.0 {
            sources.push(s);
            targets.push(t);
            rates.push(r);
        }
    }
    // Free pairs: iterate all (i,j) and skip masked
    for i in 0..n {
        for j in 0..n {
            if mask.is_masked(i, j) {
                continue;
            }
            let rate = free_rate_fn(i, j);
            if rate > 0.0 {
                sources.push(i as u64);
                targets.push(j as u64);
                rates.push(rate);
            }
        }
    }
    PartialFitResult {
        sources,
        targets,
        rates,
        converged,
        iterations,
    }
}

/// Full partial strength-Poisson fit: excess → sparse masked IPF → rate table.
///
/// Uses `PairMask` for O(N+K) memory instead of O(N²).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_partial_strength(
    strength_out: &[f64],
    strength_in: &[f64],
    known_src: &[u64],
    known_tgt: &[u64],
    known_rate: &[f64],
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PartialFitResult {
    let n = infer_n(strength_out.len(), known_src, known_tgt);
    let s_out = pad_to_n(strength_out, n);
    let s_in = pad_to_n(strength_in, n);
    let mask = PairMask::new(n, self_loops, known_src, known_tgt);

    let (mut excess_out, mut excess_in) =
        match compute_excess(&s_out, &s_in, known_src, known_tgt, known_rate) {
            Some(v) => v,
            None => {
                return assemble_result_sparse(
                    n,
                    known_src,
                    known_tgt,
                    known_rate,
                    &mask,
                    |_, _| 0.0,
                    false,
                    0,
                )
            }
        };

    if excess_out.iter().sum::<f64>() <= 0.0 {
        return assemble_result_sparse(
            n,
            known_src,
            known_tgt,
            known_rate,
            &mask,
            |_, _| 0.0,
            true,
            0,
        );
    }

    balance_excess(&mut excess_out, &mut excess_in);
    let fit = balance_sparse_masked_strength_poisson(
        &excess_out,
        &excess_in,
        &mask,
        tolerance,
        max_iterations,
    );
    let x = fit.x;
    let y = fit.y;
    assemble_result_sparse(
        n,
        known_src,
        known_tgt,
        known_rate,
        &mask,
        |i, j| x[i] * y[j],
        fit.converged,
        fit.iterations,
    )
}

/// Full partial degree-Bernoulli fit: excess → masked IPF → rate table.
#[must_use]
pub fn fit_partial_degree(
    degree_out: &[f64],
    degree_in: &[f64],
    known_src: &[u64],
    known_tgt: &[u64],
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PartialFitResult {
    let n = infer_n(degree_out.len(), known_src, known_tgt);
    let k_out = pad_to_n(degree_out, n);
    let k_in = pad_to_n(degree_in, n);
    let known_binary: Vec<f64> = vec![1.0; known_src.len()];
    let mask = PairMask::new(n, self_loops, known_src, known_tgt);

    let (mut excess_out, mut excess_in) =
        match compute_excess(&k_out, &k_in, known_src, known_tgt, &known_binary) {
            Some(v) => v,
            None => {
                return assemble_result_sparse(
                    n,
                    known_src,
                    known_tgt,
                    &known_binary,
                    &mask,
                    |_, _| 0.0,
                    false,
                    0,
                )
            }
        };

    balance_excess(&mut excess_out, &mut excess_in);
    let fit = balance_sparse_masked_degree_bernoulli(
        &excess_out,
        &excess_in,
        &mask,
        tolerance,
        max_iterations,
    );
    let x = fit.x;
    let y = fit.y;
    let known_rate_ones: Vec<f64> = vec![1.0; known_src.len()];
    assemble_result_sparse(
        n,
        known_src,
        known_tgt,
        &known_rate_ones,
        &mask,
        |i, j| {
            let z = x[i] * y[j];
            z / (1.0 + z)
        },
        fit.converged,
        fit.iterations,
    )
}

/// Full partial strength-degree fit: excess → masked 4-var IPF → rate table.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn fit_partial_strength_degree(
    strength_out: &[f64],
    strength_in: &[f64],
    degree_out: &[f64],
    degree_in: &[f64],
    known_src: &[u64],
    known_tgt: &[u64],
    known_rate: &[f64],
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PartialFitResult {
    let n = infer_n(strength_out.len(), known_src, known_tgt);
    let s_out = pad_to_n(strength_out, n);
    let s_in = pad_to_n(strength_in, n);
    let k_out = pad_to_n(degree_out, n);
    let k_in = pad_to_n(degree_in, n);
    let mask = PairMask::new(n, self_loops, known_src, known_tgt);

    let (mut excess_s_out, mut excess_s_in) =
        match compute_excess(&s_out, &s_in, known_src, known_tgt, known_rate) {
            Some(v) => v,
            None => {
                return assemble_result_sparse(
                    n,
                    known_src,
                    known_tgt,
                    known_rate,
                    &mask,
                    |_, _| 0.0,
                    false,
                    0,
                )
            }
        };

    let known_binary: Vec<f64> = vec![1.0; known_src.len()];
    let (mut excess_k_out, mut excess_k_in) =
        match compute_excess(&k_out, &k_in, known_src, known_tgt, &known_binary) {
            Some(v) => v,
            None => {
                return assemble_result_sparse(
                    n,
                    known_src,
                    known_tgt,
                    known_rate,
                    &mask,
                    |_, _| 0.0,
                    false,
                    0,
                )
            }
        };

    if excess_s_out.iter().sum::<f64>() <= 0.0 {
        return assemble_result_sparse(
            n,
            known_src,
            known_tgt,
            known_rate,
            &mask,
            |_, _| 0.0,
            true,
            0,
        );
    }

    balance_excess(&mut excess_s_out, &mut excess_s_in);
    balance_excess(&mut excess_k_out, &mut excess_k_in);

    let fit = balance_sparse_masked_strength_degree_poisson(
        &excess_s_out,
        &excess_s_in,
        &excess_k_out,
        &excess_k_in,
        &mask,
        tolerance,
        max_iterations,
    );
    let x = fit.x;
    let y = fit.y;
    let z = fit.z;
    let w = fit.w;
    assemble_result_sparse(
        n,
        known_src,
        known_tgt,
        known_rate,
        &mask,
        |i, j| {
            let u = x[i] * y[j];
            let v = z[i] * w[j];
            let e_neg_u = (-u).exp();
            let den = e_neg_u + v * (1.0 - e_neg_u);
            if den > 0.0 {
                v * u / den
            } else {
                0.0
            }
        },
        fit.converged,
        fit.iterations,
    )
}

/// Full partial strength-edges fit: excess → full fit on excess → rate table.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn fit_partial_strength_edges(
    strength_out: &[f64],
    strength_in: &[f64],
    known_src: &[u64],
    known_tgt: &[u64],
    known_rate: &[f64],
    target_edges: f64,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PartialFitResult {
    let n = infer_n(strength_out.len(), known_src, known_tgt);
    let s_out = pad_to_n(strength_out, n);
    let s_in = pad_to_n(strength_in, n);
    let mask = PairMask::new(n, self_loops, known_src, known_tgt);
    let excess_edges = (target_edges - known_src.len() as f64).max(0.0);

    let (mut excess_out, mut excess_in) =
        match compute_excess(&s_out, &s_in, known_src, known_tgt, known_rate) {
            Some(v) => v,
            None => {
                return assemble_result_sparse(
                    n,
                    known_src,
                    known_tgt,
                    known_rate,
                    &mask,
                    |_, _| 0.0,
                    false,
                    0,
                )
            }
        };

    if excess_out.iter().sum::<f64>() <= 0.0 || excess_edges <= 0.0 {
        return assemble_result_sparse(
            n,
            known_src,
            known_tgt,
            known_rate,
            &mask,
            |_, _| 0.0,
            true,
            0,
        );
    }

    balance_excess(&mut excess_out, &mut excess_in);
    let fit = super::me_lbfgs::fit_strength_edges_poisson_lbfgs(
        &excess_out,
        &excess_in,
        excess_edges,
        &mask,
        tolerance,
        max_iterations,
    );
    let x = fit.x;
    let y = fit.y;
    let lam = fit.lam;
    assemble_result_sparse(
        n,
        known_src,
        known_tgt,
        known_rate,
        &mask,
        |i, j| {
            let u = x[i] * y[j];
            let e_neg_u = (-u).exp();
            let den = e_neg_u + lam * (1.0 - e_neg_u);
            if den > 0.0 {
                lam * u / den
            } else {
                0.0
            }
        },
        fit.converged,
        fit.iterations,
    )
}

use super::support::coord_distance;

#[allow(clippy::too_many_arguments)]
fn balance_masked_coordinate_strength_cost_fixed_gamma(
    strength_out: &[f64],
    strength_in: &[f64],
    coord_x: &[f64],
    coord_y: &[f64],
    mask: &PairMask,
    gamma: f64,
    tolerance: f64,
    max_iterations: usize,
    x_init: Option<&[f64]>,
    y_init: Option<&[f64]>,
) -> FitResult {
    let n = strength_out.len();
    let total: f64 = strength_out.iter().sum();
    let sqrt_t = total.sqrt().max(1.0);
    let mut x: Vec<f64> = x_init.map_or_else(
        || {
            strength_out
                .iter()
                .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
                .collect()
        },
        <[f64]>::to_vec,
    );
    let mut y: Vec<f64> = y_init.map_or_else(
        || {
            strength_in
                .iter()
                .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
                .collect()
        },
        <[f64]>::to_vec,
    );

    for iter in 0..max_iterations {
        let old_x = x.clone();
        let old_y = y.clone();
        for j in 0..n {
            if strength_in[j] <= 0.0 {
                y[j] = 0.0;
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&i| !mask.is_masked(i, j))
                .map(|i| {
                    let d = coord_distance(coord_x, coord_y, i, j);
                    x[i] * (-gamma * d).clamp(-700.0, 700.0).exp()
                })
                .sum();
            y[j] = if denom > 0.0 {
                strength_in[j] / denom
            } else {
                0.0
            };
        }
        for i in 0..n {
            if strength_out[i] <= 0.0 {
                x[i] = 0.0;
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&j| !mask.is_masked(i, j))
                .map(|j| {
                    let d = coord_distance(coord_x, coord_y, i, j);
                    y[j] * (-gamma * d).clamp(-700.0, 700.0).exp()
                })
                .sum();
            x[i] = if denom > 0.0 {
                strength_out[i] / denom
            } else {
                0.0
            };
        }
        if max_pair_delta(&x, &old_x, &y, &old_y) < tolerance {
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

fn expected_masked_coordinate_cost(
    x: &[f64],
    y: &[f64],
    coord_x: &[f64],
    coord_y: &[f64],
    mask: &PairMask,
    gamma: f64,
) -> f64 {
    let n = x.len();
    let mut total = 0.0;
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        for j in 0..n {
            if mask.is_masked(i, j) {
                continue;
            }
            let d = coord_distance(coord_x, coord_y, i, j);
            total += x[i] * y[j] * d * (-gamma * d).clamp(-700.0, 700.0).exp();
        }
    }
    total
}

#[allow(clippy::too_many_arguments)]
fn fit_masked_strength_cost_poisson_coordinates(
    strength_out: &[f64],
    strength_in: &[f64],
    coord_x: &[f64],
    coord_y: &[f64],
    target_cost: f64,
    mask: &PairMask,
    tolerance: f64,
    max_iterations: usize,
) -> StrengthCostFitResult {
    let solve_at = |gamma: f64, x_init: Option<&[f64]>, y_init: Option<&[f64]>| {
        let fit = balance_masked_coordinate_strength_cost_fixed_gamma(
            strength_out,
            strength_in,
            coord_x,
            coord_y,
            mask,
            gamma,
            tolerance,
            max_iterations,
            x_init,
            y_init,
        );
        let delta = expected_masked_coordinate_cost(&fit.x, &fit.y, coord_x, coord_y, mask, gamma)
            - target_cost;
        (fit, delta)
    };

    let (fit_zero, delta_zero) = solve_at(0.0, None, None);
    if delta_zero.abs() <= tolerance {
        return StrengthCostFitResult {
            x: fit_zero.x,
            y: fit_zero.y,
            gamma: 0.0,
            converged: fit_zero.converged,
            iterations: fit_zero.iterations,
        };
    }
    let mut low = 0.0;
    let mut high = 0.0;
    let mut low_fit = fit_zero.clone();
    let mut high_fit = fit_zero.clone();
    let mut low_delta = delta_zero;
    let mut high_delta = delta_zero;
    let mut step = 1.0_f64;
    for _ in 0..64 {
        if delta_zero > 0.0 {
            high = step;
            let (fit, delta) = solve_at(high, Some(&high_fit.x), Some(&high_fit.y));
            high_fit = fit;
            high_delta = delta;
            if high_delta <= 0.0 {
                break;
            }
        } else {
            low = -step;
            let (fit, delta) = solve_at(low, Some(&low_fit.x), Some(&low_fit.y));
            low_fit = fit;
            low_delta = delta;
            if low_delta >= 0.0 {
                break;
            }
        }
        step *= 2.0;
    }
    if !(low_delta >= 0.0 && high_delta <= 0.0) {
        let (best_fit, gamma) = if low_delta.abs() < high_delta.abs() {
            (low_fit, low)
        } else {
            (high_fit, high)
        };
        return StrengthCostFitResult {
            x: best_fit.x,
            y: best_fit.y,
            gamma,
            converged: false,
            iterations: max_iterations,
        };
    }
    let mut best_gamma = if low_delta.abs() < high_delta.abs() {
        low
    } else {
        high
    };
    let mut best_fit = if low_delta.abs() < high_delta.abs() {
        low_fit
    } else {
        high_fit
    };
    let mut best_delta = low_delta.abs().min(high_delta.abs());
    for iter in 0..max_iterations {
        let mid = 0.5 * (low + high);
        let (fit, delta) = solve_at(mid, Some(&best_fit.x), Some(&best_fit.y));
        if delta.abs() < best_delta {
            best_delta = delta.abs();
            best_gamma = mid;
            best_fit = fit.clone();
        }
        if delta.abs() <= tolerance {
            return StrengthCostFitResult {
                x: fit.x,
                y: fit.y,
                gamma: mid,
                converged: fit.converged,
                iterations: iter + 1,
            };
        }
        if delta > 0.0 {
            low = mid;
        } else {
            high = mid;
        }
    }
    StrengthCostFitResult {
        x: best_fit.x,
        y: best_fit.y,
        gamma: best_gamma,
        converged: best_delta <= tolerance,
        iterations: max_iterations,
    }
}

#[allow(clippy::too_many_arguments)]
fn fit_partial_strength_cost_coordinates_with(
    strength_out: &[f64],
    strength_in: &[f64],
    known_src: &[u64],
    known_tgt: &[u64],
    known_rate: &[f64],
    coord_x: &[f64],
    coord_y: &[f64],
    target_cost: f64,
    self_loops: bool,
    fit_free: impl FnOnce(&[f64], &[f64], f64) -> StrengthCostFitResult,
    rate: impl Fn(usize, usize, &StrengthCostFitResult) -> f64,
) -> PartialFitResult {
    let n = infer_n(strength_out.len(), known_src, known_tgt);
    let s_out = pad_to_n(strength_out, n);
    let s_in = pad_to_n(strength_in, n);
    let mask = PairMask::new(n, self_loops, known_src, known_tgt);
    let known_cost: f64 = known_src
        .iter()
        .zip(known_tgt.iter())
        .zip(known_rate.iter())
        .map(|((&s, &t), &r)| r * coord_distance(coord_x, coord_y, s as usize, t as usize))
        .sum();
    let excess_cost = (target_cost - known_cost).max(0.0);
    let (mut excess_out, mut excess_in) =
        match compute_excess(&s_out, &s_in, known_src, known_tgt, known_rate) {
            Some(v) => v,
            None => {
                return assemble_result_sparse(
                    n,
                    known_src,
                    known_tgt,
                    known_rate,
                    &mask,
                    |_, _| 0.0,
                    false,
                    0,
                )
            }
        };
    if excess_out.iter().sum::<f64>() <= 0.0 {
        return assemble_result_sparse(
            n,
            known_src,
            known_tgt,
            known_rate,
            &mask,
            |_, _| 0.0,
            true,
            0,
        );
    }
    balance_excess(&mut excess_out, &mut excess_in);
    let fit = fit_free(&excess_out, &excess_in, excess_cost);
    assemble_result_sparse(
        n,
        known_src,
        known_tgt,
        known_rate,
        &mask,
        |i, j| rate(i, j, &fit),
        fit.converged,
        fit.iterations,
    )
}

/// Full partial strength-cost fit with projected Euclidean coordinate costs.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn fit_partial_strength_cost_coordinates(
    strength_out: &[f64],
    strength_in: &[f64],
    known_src: &[u64],
    known_tgt: &[u64],
    known_rate: &[f64],
    coord_x: &[f64],
    coord_y: &[f64],
    target_cost: f64,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PartialFitResult {
    let n = infer_n(strength_out.len(), known_src, known_tgt);
    let mask = PairMask::new(n, self_loops, known_src, known_tgt);
    fit_partial_strength_cost_coordinates_with(
        strength_out,
        strength_in,
        known_src,
        known_tgt,
        known_rate,
        coord_x,
        coord_y,
        target_cost,
        self_loops,
        |excess_out, excess_in, excess_cost| {
            fit_masked_strength_cost_poisson_coordinates(
                excess_out,
                excess_in,
                coord_x,
                coord_y,
                excess_cost,
                &mask,
                tolerance,
                max_iterations,
            )
        },
        |i, j, fit| {
            let d = coord_distance(coord_x, coord_y, i, j);
            fit.x[i] * fit.y[j] * (-fit.gamma * d).clamp(-700.0, 700.0).exp()
        },
    )
}

/// Full partial B(M) strength-cost fit with projected Euclidean coordinate costs.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn fit_partial_strength_cost_binomial_coordinates(
    strength_out: &[f64],
    strength_in: &[f64],
    known_src: &[u64],
    known_tgt: &[u64],
    known_rate: &[f64],
    coord_x: &[f64],
    coord_y: &[f64],
    target_cost: f64,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PartialFitResult {
    let opts = CostFitOptions {
        self_loops,
        tolerance,
        max_iterations,
    };
    fit_partial_strength_cost_coordinates_with(
        strength_out,
        strength_in,
        known_src,
        known_tgt,
        known_rate,
        coord_x,
        coord_y,
        target_cost,
        self_loops,
        |excess_out, excess_in, excess_cost| {
            fit_strength_cost_binomial_coordinates(
                excess_out,
                excess_in,
                coord_x,
                coord_y,
                excess_cost,
                layers,
                &opts,
            )
        },
        |i, j, fit| {
            let d = coord_distance(coord_x, coord_y, i, j);
            let q = fit.x[i] * fit.y[j] * (-fit.gamma * d).clamp(-700.0, 700.0).exp();
            f64::from(layers) * q / (1.0 + q)
        },
    )
}

/// Full partial W strength-cost fit with projected Euclidean coordinate costs.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn fit_partial_strength_cost_w_coordinates(
    strength_out: &[f64],
    strength_in: &[f64],
    known_src: &[u64],
    known_tgt: &[u64],
    known_rate: &[f64],
    coord_x: &[f64],
    coord_y: &[f64],
    target_cost: f64,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PartialFitResult {
    let opts = CostFitOptions {
        self_loops,
        tolerance,
        max_iterations,
    };
    fit_partial_strength_cost_coordinates_with(
        strength_out,
        strength_in,
        known_src,
        known_tgt,
        known_rate,
        coord_x,
        coord_y,
        target_cost,
        self_loops,
        |excess_out, excess_in, excess_cost| {
            fit_strength_cost_w_lbfgs(
                excess_out,
                excess_in,
                coord_x,
                coord_y,
                excess_cost,
                layers,
                &opts,
            )
        },
        |i, j, fit| {
            let d = coord_distance(coord_x, coord_y, i, j);
            let q = fit.x[i] * fit.y[j] * (-fit.gamma * d).clamp(-700.0, 700.0).exp();
            if q <= 0.0 || q >= 1.0 {
                0.0
            } else {
                f64::from(layers) * q / (1.0 - q)
            }
        },
    )
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn infer_n(seq_len: usize, known_src: &[u64], known_tgt: &[u64]) -> usize {
    let mut n = seq_len;
    if let Some(&max_s) = known_src.iter().max() {
        n = n.max(max_s as usize + 1);
    }
    if let Some(&max_t) = known_tgt.iter().max() {
        n = n.max(max_t as usize + 1);
    }
    n
}

fn pad_to_n(arr: &[f64], n: usize) -> Vec<f64> {
    let mut v = arr.to_vec();
    v.resize(n, 0.0);
    v
}
