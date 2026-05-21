//! Full-pipeline partial-constraint fitting.
//!
//! Each function takes raw inputs (sequences, known pairs, options) and returns
//! a sparse rate table. All mask building, excess computation, balancing, IPF,
//! and result assembly happens in Rust.

use super::support::{max_pair_delta, self_loop_mask};
use super::{
    balance_masked_degree_bernoulli, balance_masked_strength_degree_poisson,
    balance_masked_strength_poisson, balance_strength_edges_poisson, FitResult, PartialFitResult,
    StrengthCostFitResult,
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn build_mask(n: usize, known_src: &[u64], known_tgt: &[u64], self_loops: bool) -> Vec<bool> {
    let mut mask = self_loop_mask(n, self_loops);
    for (&s, &t) in known_src.iter().zip(known_tgt.iter()) {
        let idx = (s as usize) * n + (t as usize);
        if idx < mask.len() {
            mask[idx] = true;
        }
    }
    mask
}

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

/// IPF for masked ME strength-cost at fixed gamma. No dense N^2 factor matrix.
/// Uses sparse per-column/row f_ij lookup (O(K) memory).
#[allow(clippy::too_many_arguments)]
fn balance_masked_strength_cost_poisson_fixed_gamma(
    strength_out: &[f64],
    strength_in: &[f64],
    cost_sources: &[usize],
    cost_targets: &[usize],
    cost_values: &[f64],
    mask: &[bool],
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

    // Build per-column/row sparse f_ij (only free pairs with cost entries)
    let mut col_src: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut row_tgt: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (idx, (&src, &tgt)) in cost_sources.iter().zip(cost_targets.iter()).enumerate() {
        if src < n && tgt < n && !mask[src * n + tgt] {
            let f_ij = (-gamma * cost_values[idx]).clamp(-700.0, 700.0).exp();
            col_src[tgt].push((src, f_ij));
            row_tgt[src].push((tgt, f_ij));
        }
    }
    // Also add free pairs with no cost entry (f_ij = 1.0) using correction trick
    let k = cost_sources.len();
    let n_pairs: usize = (0..n * n).filter(|&idx| !mask[idx]).count();
    let complete = k >= n_pairs;

    for iter in 0..max_iterations {
        let old_x = x.clone();
        let old_y = y.clone();

        if complete {
            for j in 0..n {
                if strength_in[j] <= 0.0 {
                    y[j] = 0.0;
                    continue;
                }
                let denom: f64 = col_src[j].iter().map(|&(i, f)| x[i] * f).sum();
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
                let denom: f64 = row_tgt[i].iter().map(|&(j, f)| y[j] * f).sum();
                x[i] = if denom > 0.0 {
                    strength_out[i] / denom
                } else {
                    0.0
                };
            }
        } else {
            // Sparse: base sum over free pairs (f=1), correct with entries
            let sum_x: f64 = x.iter().sum();
            for j in 0..n {
                if strength_in[j] <= 0.0 {
                    y[j] = 0.0;
                    continue;
                }
                // base: sum of x[i] for free pairs in column j
                let masked_out: f64 = (0..n).filter(|&i| mask[i * n + j]).map(|i| x[i]).sum();
                let mut denom = sum_x - masked_out;
                for &(src, f_ij) in &col_src[j] {
                    denom += x[src] * (f_ij - 1.0);
                }
                y[j] = if denom > 0.0 {
                    strength_in[j] / denom
                } else {
                    0.0
                };
            }
            let sum_y: f64 = y.iter().sum();
            for i in 0..n {
                if strength_out[i] <= 0.0 {
                    x[i] = 0.0;
                    continue;
                }
                let masked_out: f64 = (0..n).filter(|&j| mask[i * n + j]).map(|j| y[j]).sum();
                let mut denom = sum_y - masked_out;
                for &(tgt, f_ij) in &row_tgt[i] {
                    denom += y[tgt] * (f_ij - 1.0);
                }
                x[i] = if denom > 0.0 {
                    strength_out[i] / denom
                } else {
                    0.0
                };
            }
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

fn expected_masked_cost(
    x: &[f64],
    y: &[f64],
    cost_sources: &[usize],
    cost_targets: &[usize],
    cost_values: &[f64],
    mask: &[bool],
    gamma: f64,
) -> f64 {
    let n = x.len();
    let mut total = 0.0;
    for (idx, (&source, &target)) in cost_sources.iter().zip(cost_targets.iter()).enumerate() {
        if source < n && target < n && !mask[source * n + target] {
            let cost = cost_values[idx];
            let exponent = (-gamma * cost).clamp(-700.0, 700.0);
            total += x[source] * y[target] * cost * exponent.exp();
        }
    }
    total
}

#[allow(clippy::too_many_arguments)]
fn fit_masked_strength_cost_poisson(
    strength_out: &[f64],
    strength_in: &[f64],
    cost_sources: &[usize],
    cost_targets: &[usize],
    cost_values: &[f64],
    target_cost: f64,
    mask: &[bool],
    tolerance: f64,
    max_iterations: usize,
) -> StrengthCostFitResult {
    let solve_at = |gamma: f64, x_init: Option<&[f64]>, y_init: Option<&[f64]>| {
        let fit = balance_masked_strength_cost_poisson_fixed_gamma(
            strength_out,
            strength_in,
            cost_sources,
            cost_targets,
            cost_values,
            mask,
            gamma,
            tolerance,
            max_iterations,
            x_init,
            y_init,
        );
        let delta = expected_masked_cost(
            &fit.x,
            &fit.y,
            cost_sources,
            cost_targets,
            cost_values,
            mask,
            gamma,
        ) - target_cost;
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
        let best_fit = if low_delta.abs() < high_delta.abs() {
            low_fit
        } else {
            high_fit
        };
        let gamma = if low_delta.abs() < high_delta.abs() {
            low
        } else {
            high
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
fn assemble_result(
    n: usize,
    known_src: &[u64],
    known_tgt: &[u64],
    known_rate: &[f64],
    mask: &[bool],
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
    // Free pairs
    for i in 0..n {
        for j in 0..n {
            if mask[i * n + j] {
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

// ---------------------------------------------------------------------------
// Public full-pipeline functions
// ---------------------------------------------------------------------------

/// Full partial strength-Poisson fit: excess → masked IPF → rate table.
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
    let mask = build_mask(n, known_src, known_tgt, self_loops);

    let (mut excess_out, mut excess_in) =
        match compute_excess(&s_out, &s_in, known_src, known_tgt, known_rate) {
            Some(v) => v,
            None => {
                return assemble_result(
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
        return assemble_result(
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
    let fit =
        balance_masked_strength_poisson(&excess_out, &excess_in, &mask, tolerance, max_iterations);
    let x = fit.x;
    let y = fit.y;
    assemble_result(
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
    let mask = build_mask(n, known_src, known_tgt, self_loops);

    let (mut excess_out, mut excess_in) =
        match compute_excess(&k_out, &k_in, known_src, known_tgt, &known_binary) {
            Some(v) => v,
            None => {
                return assemble_result(
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
    let fit =
        balance_masked_degree_bernoulli(&excess_out, &excess_in, &mask, tolerance, max_iterations);
    let x = fit.x;
    let y = fit.y;
    let known_rate_ones: Vec<f64> = vec![1.0; known_src.len()];
    assemble_result(
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
    let mask = build_mask(n, known_src, known_tgt, self_loops);

    let (mut excess_s_out, mut excess_s_in) =
        match compute_excess(&s_out, &s_in, known_src, known_tgt, known_rate) {
            Some(v) => v,
            None => {
                return assemble_result(
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
                return assemble_result(
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
        return assemble_result(
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

    let fit = balance_masked_strength_degree_poisson(
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
    assemble_result(
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
    let mask = build_mask(n, known_src, known_tgt, self_loops);
    let excess_edges = (target_edges - known_src.len() as f64).max(0.0);

    let (mut excess_out, mut excess_in) =
        match compute_excess(&s_out, &s_in, known_src, known_tgt, known_rate) {
            Some(v) => v,
            None => {
                return assemble_result(
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
        return assemble_result(
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
    let fit = balance_strength_edges_poisson(
        &excess_out,
        &excess_in,
        excess_edges,
        self_loops,
        tolerance,
        max_iterations,
    );
    let x = fit.x;
    let y = fit.y;
    let lam = fit.lam;
    assemble_result(
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
    mask: &[bool],
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
                .filter(|&i| !mask[i * n + j])
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
                .filter(|&j| !mask[i * n + j])
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
    mask: &[bool],
    gamma: f64,
) -> f64 {
    let n = x.len();
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            if mask[i * n + j] {
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
    mask: &[bool],
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

/// Full partial strength-cost fit: excess → fit on excess with free costs → rate table.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn fit_partial_strength_cost(
    strength_out: &[f64],
    strength_in: &[f64],
    known_src: &[u64],
    known_tgt: &[u64],
    known_rate: &[f64],
    cost_sources: &[usize],
    cost_targets: &[usize],
    cost_values: &[f64],
    target_cost: f64,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PartialFitResult {
    let n = infer_n(strength_out.len(), known_src, known_tgt);
    let s_out = pad_to_n(strength_out, n);
    let s_in = pad_to_n(strength_in, n);
    let mask = build_mask(n, known_src, known_tgt, self_loops);

    // Build cost map
    let mut cost_map = std::collections::HashMap::new();
    for ((&cs, &ct), &cv) in cost_sources
        .iter()
        .zip(cost_targets.iter())
        .zip(cost_values.iter())
    {
        cost_map.insert((cs, ct), cv);
    }

    // Compute known cost contribution
    let known_cost: f64 = known_src
        .iter()
        .zip(known_tgt.iter())
        .zip(known_rate.iter())
        .map(|((&s, &t), &r)| {
            r * cost_map
                .get(&(s as usize, t as usize))
                .copied()
                .unwrap_or(0.0)
        })
        .sum();
    let excess_cost = (target_cost - known_cost).max(0.0);

    let (mut excess_out, mut excess_in) =
        match compute_excess(&s_out, &s_in, known_src, known_tgt, known_rate) {
            Some(v) => v,
            None => {
                return assemble_result(
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
        return assemble_result(
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

    // Filter costs to free pairs only
    let mut free_src = Vec::new();
    let mut free_tgt = Vec::new();
    let mut free_val = Vec::new();
    for ((&cs, &ct), &cv) in cost_sources
        .iter()
        .zip(cost_targets.iter())
        .zip(cost_values.iter())
    {
        if cs < n && ct < n && !mask[cs * n + ct] {
            free_src.push(cs);
            free_tgt.push(ct);
            free_val.push(cv);
        }
    }

    let fit = fit_masked_strength_cost_poisson(
        &excess_out,
        &excess_in,
        &free_src,
        &free_tgt,
        &free_val,
        excess_cost,
        &mask,
        tolerance,
        max_iterations,
    );
    let x = fit.x;
    let y = fit.y;
    let gamma = fit.gamma;
    assemble_result(
        n,
        known_src,
        known_tgt,
        known_rate,
        &mask,
        |i, j| {
            let d = cost_map.get(&(i, j)).copied().unwrap_or(0.0);
            x[i] * y[j] * (-gamma * d).exp()
        },
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
    let s_out = pad_to_n(strength_out, n);
    let s_in = pad_to_n(strength_in, n);
    let mask = build_mask(n, known_src, known_tgt, self_loops);
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
                return assemble_result(
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
        return assemble_result(
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
    let fit = fit_masked_strength_cost_poisson_coordinates(
        &excess_out,
        &excess_in,
        coord_x,
        coord_y,
        excess_cost,
        &mask,
        tolerance,
        max_iterations,
    );
    let x = fit.x;
    let y = fit.y;
    let gamma = fit.gamma;
    assemble_result(
        n,
        known_src,
        known_tgt,
        known_rate,
        &mask,
        |i, j| {
            let d = coord_distance(coord_x, coord_y, i, j);
            x[i] * y[j] * (-gamma * d).clamp(-700.0, 700.0).exp()
        },
        fit.converged,
        fit.iterations,
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
