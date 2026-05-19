//! Full-pipeline partial-constraint fitting.
//!
//! Each function takes raw inputs (sequences, known pairs, options) and returns
//! a sparse rate table. All mask building, excess computation, balancing, IPF,
//! and result assembly happens in Rust.

use crate::cost::fit_strength_cost;
use crate::fitting::{
    balance_masked_degree_bernoulli, balance_masked_strength_degree_poisson,
    balance_masked_strength_poisson, balance_strength_edges_poisson,
};

/// Result of a partial-constraint fit: sparse rate table.
#[derive(Clone, Debug)]
pub struct PartialFitResult {
    pub sources: Vec<u64>,
    pub targets: Vec<u64>,
    pub rates: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn build_mask(n: usize, known_src: &[u64], known_tgt: &[u64], self_loops: bool) -> Vec<bool> {
    let mut mask = vec![false; n * n];
    for (&s, &t) in known_src.iter().zip(known_tgt.iter()) {
        let idx = (s as usize) * n + (t as usize);
        if idx < mask.len() {
            mask[idx] = true;
        }
    }
    if !self_loops {
        for i in 0..n {
            mask[i * n + i] = true;
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
        } else {
            if let Some(idx) = excess_out
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
            {
                excess_out[idx] -= diff;
            }
        }
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
            let exp_u = u.exp();
            let den = 1.0 + v * (exp_u - 1.0);
            if den > 0.0 {
                v * u * exp_u / den
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
            let exp_u = u.exp();
            let den = 1.0 + lam * (exp_u - 1.0);
            if den > 0.0 {
                lam * u * exp_u / den
            } else {
                0.0
            }
        },
        fit.converged,
        fit.iterations,
    )
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

    let fit = fit_strength_cost(
        &excess_out,
        &excess_in,
        &free_src,
        &free_tgt,
        &free_val,
        excess_cost,
        &crate::cost::CostFitOptions {
            self_loops,
            tolerance,
            max_iterations,
        },
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
