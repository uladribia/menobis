//! Python bindings for ODME.

use odme_core::clustering::{
    clustering_coefficients as core_clustering,
    weighted_clustering_coefficients as core_weighted_clustering,
};
use odme_core::cost::{fit_strength_cost as core_fit_strength_cost, CostFitOptions};
use odme_core::fitting::{
    balance_binary_degrees, balance_masked_binary_degrees, balance_masked_strength,
    balance_masked_strength_degree_me, balance_no_self_loops, balance_strength_degree_me,
    balance_strength_edges_me, balance_weighted_factors,
};
use odme_core::generation::{
    sample_custom_pij_events_multinomial as core_sample_custom_pij_events_multinomial,
    sample_custom_pij_events_poisson as core_sample_custom_pij_events_poisson,
    sample_fixed_degree_events_me as core_sample_fixed_degree_events_me,
    sample_microcanonical as core_sample_microcanonical,
    sample_multinomial as core_sample_multinomial, sample_poisson as core_sample_poisson,
    sample_poisson_multinomial as core_sample_poisson_multinomial,
    sample_strength_degree_me as core_sample_strength_degree_me,
    sample_strength_edges_me as core_sample_strength_edges_me,
};
use odme_core::graph::{
    directed_degrees as core_directed_degrees, directed_strengths as core_directed_strengths,
    WeightedEdge,
};
use odme_core::stats::{
    compute_all_node_stats as core_compute_all_stats,
    weight_distribution as core_weight_distribution,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type FitPair = (Vec<f64>, Vec<f64>, bool, usize);
type FitStrengthEdges = (Vec<f64>, Vec<f64>, f64, bool, usize);
type FitStrengthDegree = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, bool, usize);

type FitStrengthCost = (Vec<f64>, Vec<f64>, f64, bool, usize);

/// Return the version of the Rust core exposed through Python.
#[pyfunction]
fn rust_core_version() -> &'static str {
    odme_core::VERSION
}

fn build_edges(
    node_count: usize,
    sources: Vec<usize>,
    targets: Vec<usize>,
    weights: Vec<u64>,
) -> PyResult<Vec<WeightedEdge>> {
    if sources.len() != targets.len() || sources.len() != weights.len() {
        return Err(PyValueError::new_err(
            "sources, targets, and weights must have the same length",
        ));
    }
    let mut edges = Vec::with_capacity(sources.len());
    for ((source, target), weight) in sources.into_iter().zip(targets).zip(weights) {
        if source >= node_count || target >= node_count {
            return Err(PyValueError::new_err(
                "edge endpoint is outside the declared node count",
            ));
        }
        if weight == 0 {
            continue;
        }
        edges.push(WeightedEdge::new(source, target, weight));
    }
    Ok(edges)
}

#[pyfunction]
fn directed_strengths(
    node_count: usize,
    sources: Vec<usize>,
    targets: Vec<usize>,
    weights: Vec<u64>,
) -> PyResult<(Vec<u64>, Vec<u64>)> {
    let edges = build_edges(node_count, sources, targets, weights)?;
    let s = core_directed_strengths(node_count, &edges);
    Ok((s.out, s.incoming))
}

#[pyfunction]
fn directed_degrees(
    node_count: usize,
    sources: Vec<usize>,
    targets: Vec<usize>,
    weights: Vec<u64>,
) -> PyResult<(Vec<u64>, Vec<u64>)> {
    let edges = build_edges(node_count, sources, targets, weights)?;
    let d = core_directed_degrees(node_count, &edges);
    Ok((d.out, d.incoming))
}

/// Compute all per-node statistics in a single pass.
/// Returns a tuple of vectors:
/// (s_out, s_in, k_out, k_in, y2_out, y2_in, s_nn_out, s_nn_in, k_nn_out, k_nn_in)
#[pyfunction]
#[allow(clippy::type_complexity)]
fn compute_all_node_stats(
    node_count: usize,
    sources: Vec<usize>,
    targets: Vec<usize>,
    weights: Vec<u64>,
) -> PyResult<(
    Vec<u64>,
    Vec<u64>,
    Vec<u64>,
    Vec<u64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
)> {
    let edges = build_edges(node_count, sources, targets, weights)?;
    let s = core_compute_all_stats(node_count, &edges);
    Ok((
        s.strength_out,
        s.strength_in,
        s.degree_out,
        s.degree_in,
        s.y2_out,
        s.y2_in,
        s.s_nn_out,
        s.s_nn_in,
        s.k_nn_out,
        s.k_nn_in,
    ))
}

/// Compute the weight distribution P(w).
/// Returns (weights, counts).
#[pyfunction]
fn weight_distribution(
    sources: Vec<usize>,
    targets: Vec<usize>,
    weights: Vec<u64>,
) -> PyResult<(Vec<u64>, Vec<u64>)> {
    // node_count doesn't matter for weight distribution, just need valid edges
    let max_node = sources
        .iter()
        .chain(targets.iter())
        .copied()
        .max()
        .unwrap_or(0)
        + 1;
    let edges = build_edges(max_node, sources, targets, weights)?;
    let dist = core_weight_distribution(&edges);
    Ok((dist.weights, dist.counts))
}

#[pyfunction]
fn fit_masked_strength(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    mask: Vec<bool>,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<FitPair> {
    if strength_out.len() != strength_in.len() {
        return Err(PyValueError::new_err(
            "strength arrays must have same length",
        ));
    }
    let n = strength_out.len();
    if mask.len() != n * n {
        return Err(PyValueError::new_err("mask must have length n*n"));
    }
    let result = balance_masked_strength(
        &strength_out,
        &strength_in,
        &mask,
        tolerance,
        max_iterations,
    );
    Ok((result.x, result.y, result.converged, result.iterations))
}

#[pyfunction]
fn fit_masked_binary_degrees(
    degree_out: Vec<f64>,
    degree_in: Vec<f64>,
    mask: Vec<bool>,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<FitPair> {
    let n = degree_out.len();
    if n != degree_in.len() || mask.len() != n * n {
        return Err(PyValueError::new_err("array length mismatch"));
    }
    let r =
        balance_masked_binary_degrees(&degree_out, &degree_in, &mask, tolerance, max_iterations);
    Ok((r.x, r.y, r.converged, r.iterations))
}

#[pyfunction]
fn fit_masked_strength_degree_me(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    degree_out: Vec<f64>,
    degree_in: Vec<f64>,
    mask: Vec<bool>,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<FitStrengthDegree> {
    let n = strength_out.len();
    if n != strength_in.len()
        || n != degree_out.len()
        || n != degree_in.len()
        || mask.len() != n * n
    {
        return Err(PyValueError::new_err("array length mismatch"));
    }
    let r = balance_masked_strength_degree_me(
        &strength_out,
        &strength_in,
        &degree_out,
        &degree_in,
        &mask,
        tolerance,
        max_iterations,
    );
    Ok((r.x, r.y, r.z, r.w, r.converged, r.iterations))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn fit_strength_cost(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    cost_sources: Vec<usize>,
    cost_targets: Vec<usize>,
    cost_values: Vec<f64>,
    target_cost: f64,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<FitStrengthCost> {
    if strength_out.len() != strength_in.len() {
        return Err(PyValueError::new_err(
            "strength arrays must have same length",
        ));
    }
    if cost_sources.len() != cost_targets.len() || cost_sources.len() != cost_values.len() {
        return Err(PyValueError::new_err("cost arrays must have same length"));
    }
    let result = core_fit_strength_cost(
        &strength_out,
        &strength_in,
        &cost_sources,
        &cost_targets,
        &cost_values,
        target_cost,
        &CostFitOptions {
            self_loops,
            tolerance,
            max_iterations,
        },
    );
    Ok((
        result.x,
        result.y,
        result.gamma,
        result.converged,
        result.iterations,
    ))
}

#[pyfunction]
fn fit_binary_degrees(
    degree_out: Vec<f64>,
    degree_in: Vec<f64>,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<FitPair> {
    if degree_out.len() != degree_in.len() {
        return Err(PyValueError::new_err(
            "degree_out and degree_in must have same length",
        ));
    }
    let result = balance_binary_degrees(
        &degree_out,
        &degree_in,
        self_loops,
        tolerance,
        max_iterations,
    );
    Ok((result.x, result.y, result.converged, result.iterations))
}

#[pyfunction]
fn fit_strength_edges_me(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    target_edges: f64,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<FitStrengthEdges> {
    if strength_out.len() != strength_in.len() {
        return Err(PyValueError::new_err(
            "strength arrays must have same length",
        ));
    }
    let result = balance_strength_edges_me(
        &strength_out,
        &strength_in,
        target_edges,
        self_loops,
        tolerance,
        max_iterations,
    );
    Ok((
        result.x,
        result.y,
        result.lam,
        result.converged,
        result.iterations,
    ))
}

#[pyfunction]
fn fit_strength_degree_me(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    degree_out: Vec<f64>,
    degree_in: Vec<f64>,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<FitStrengthDegree> {
    if strength_out.len() != strength_in.len()
        || strength_out.len() != degree_out.len()
        || strength_out.len() != degree_in.len()
    {
        return Err(PyValueError::new_err(
            "strength and degree arrays must have same length",
        ));
    }
    let result = balance_strength_degree_me(
        &strength_out,
        &strength_in,
        &degree_out,
        &degree_in,
        self_loops,
        tolerance,
        max_iterations,
    );
    Ok((
        result.x,
        result.y,
        result.z,
        result.w,
        result.converged,
        result.iterations,
    ))
}

#[pyfunction]
fn fit_weighted_factors(
    excess_out: Vec<f64>,
    excess_in: Vec<f64>,
    degree_x: Vec<f64>,
    degree_y: Vec<f64>,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<FitPair> {
    if excess_out.len() != excess_in.len()
        || excess_out.len() != degree_x.len()
        || excess_out.len() != degree_y.len()
    {
        return Err(PyValueError::new_err(
            "excess and degree multiplier arrays must have same length",
        ));
    }
    let result = balance_weighted_factors(
        &excess_out,
        &excess_in,
        &degree_x,
        &degree_y,
        self_loops,
        tolerance,
        max_iterations,
    );
    Ok((result.x, result.y, result.converged, result.iterations))
}

#[pyfunction]
fn fit_balance_no_self_loops(
    s_out: Vec<f64>,
    s_in: Vec<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<FitPair> {
    if s_out.len() != s_in.len() {
        return Err(PyValueError::new_err(
            "s_out and s_in must have the same length",
        ));
    }
    let result = balance_no_self_loops(&s_out, &s_in, tolerance, max_iterations);
    Ok((result.x, result.y, result.converged, result.iterations))
}

#[pyfunction]
fn sample_microcanonical(
    strength_out: Vec<u64>,
    strength_in: Vec<u64>,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if strength_out.len() != strength_in.len() {
        return Err(PyValueError::new_err(
            "strength_out and strength_in must have same length",
        ));
    }
    let total_out: u64 = strength_out.iter().sum();
    let total_in: u64 = strength_in.iter().sum();
    if total_out != total_in {
        return Err(PyValueError::new_err(
            "microcanonical requires balanced strengths",
        ));
    }
    let sample = core_sample_microcanonical(&strength_out, &strength_in, seed);
    Ok((sample.sources, sample.targets, sample.weights))
}

#[pyfunction]
fn sample_custom_pij_events_poisson(
    sources: Vec<u64>,
    targets: Vec<u64>,
    probabilities: Vec<f64>,
    total_events: u64,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if sources.len() != targets.len() || sources.len() != probabilities.len() {
        return Err(PyValueError::new_err(
            "custom p_ij arrays must have same length",
        ));
    }
    let sample = core_sample_custom_pij_events_poisson(
        &sources,
        &targets,
        &probabilities,
        total_events,
        seed,
    );
    Ok((sample.sources, sample.targets, sample.weights))
}

#[pyfunction]
fn sample_custom_pij_events_multinomial(
    sources: Vec<u64>,
    targets: Vec<u64>,
    probabilities: Vec<f64>,
    total_events: u64,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if sources.len() != targets.len() || sources.len() != probabilities.len() {
        return Err(PyValueError::new_err(
            "custom p_ij arrays must have same length",
        ));
    }
    let sample = core_sample_custom_pij_events_multinomial(
        &sources,
        &targets,
        &probabilities,
        total_events,
        seed,
    );
    Ok((sample.sources, sample.targets, sample.weights))
}

#[pyfunction]
fn sample_poisson_multinomial(
    x: Vec<f64>,
    y: Vec<f64>,
    self_loops: bool,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    let sample = core_sample_poisson_multinomial(&x, &y, self_loops, seed);
    Ok((sample.sources, sample.targets, sample.weights))
}

#[pyfunction]
fn sample_strength_edges_me(
    x: Vec<f64>,
    y: Vec<f64>,
    lam: f64,
    self_loops: bool,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    let sample = core_sample_strength_edges_me(&x, &y, lam, self_loops, seed);
    Ok((sample.sources, sample.targets, sample.weights))
}

#[pyfunction]
fn sample_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_poisson(&x, &y, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
fn sample_fixed_degree_events_me(
    x: Vec<f64>,
    y: Vec<f64>,
    total_events: u64,
    self_loops: bool,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    let sample = core_sample_fixed_degree_events_me(&x, &y, total_events, self_loops, seed);
    Ok((sample.sources, sample.targets, sample.weights))
}

#[pyfunction]
fn sample_strength_degree_me(
    degree_x: Vec<f64>,
    degree_y: Vec<f64>,
    excess_x: Vec<f64>,
    excess_y: Vec<f64>,
    self_loops: bool,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if degree_x.len() != degree_y.len()
        || degree_x.len() != excess_x.len()
        || degree_x.len() != excess_y.len()
    {
        return Err(PyValueError::new_err(
            "all multiplier arrays must have same length",
        ));
    }
    let sample = core_sample_strength_degree_me(
        &degree_x, &degree_y, &excess_x, &excess_y, self_loops, seed,
    );
    Ok((sample.sources, sample.targets, sample.weights))
}

#[pyfunction]
fn sample_multinomial(
    x: Vec<f64>,
    y: Vec<f64>,
    total_events: u64,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_multinomial(&x, &y, total_events, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
fn clustering_coefficients(
    node_count: usize,
    sources: Vec<usize>,
    targets: Vec<usize>,
    weights: Vec<u64>,
) -> PyResult<Vec<f64>> {
    let edges = build_edges(node_count, sources, targets, weights)?;
    Ok(core_clustering(node_count, &edges))
}

#[pyfunction]
fn weighted_clustering_coefficients(
    node_count: usize,
    sources: Vec<usize>,
    targets: Vec<usize>,
    weights: Vec<u64>,
) -> PyResult<Vec<f64>> {
    let edges = build_edges(node_count, sources, targets, weights)?;
    Ok(core_weighted_clustering(node_count, &edges))
}

#[pymodule]
fn _odme(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(rust_core_version, module)?)?;
    module.add_function(wrap_pyfunction!(directed_strengths, module)?)?;
    module.add_function(wrap_pyfunction!(directed_degrees, module)?)?;
    module.add_function(wrap_pyfunction!(compute_all_node_stats, module)?)?;
    module.add_function(wrap_pyfunction!(weight_distribution, module)?)?;
    module.add_function(wrap_pyfunction!(fit_masked_binary_degrees, module)?)?;
    module.add_function(wrap_pyfunction!(fit_masked_strength_degree_me, module)?)?;
    module.add_function(wrap_pyfunction!(fit_masked_strength, module)?)?;
    module.add_function(wrap_pyfunction!(fit_strength_cost, module)?)?;
    module.add_function(wrap_pyfunction!(fit_binary_degrees, module)?)?;
    module.add_function(wrap_pyfunction!(fit_strength_edges_me, module)?)?;
    module.add_function(wrap_pyfunction!(fit_strength_degree_me, module)?)?;
    module.add_function(wrap_pyfunction!(fit_weighted_factors, module)?)?;
    module.add_function(wrap_pyfunction!(fit_balance_no_self_loops, module)?)?;
    module.add_function(wrap_pyfunction!(sample_microcanonical, module)?)?;
    module.add_function(wrap_pyfunction!(sample_custom_pij_events_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(
        sample_custom_pij_events_multinomial,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sample_poisson_multinomial, module)?)?;
    module.add_function(wrap_pyfunction!(sample_strength_edges_me, module)?)?;
    module.add_function(wrap_pyfunction!(sample_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(sample_fixed_degree_events_me, module)?)?;
    module.add_function(wrap_pyfunction!(sample_strength_degree_me, module)?)?;
    module.add_function(wrap_pyfunction!(sample_multinomial, module)?)?;
    module.add_function(wrap_pyfunction!(clustering_coefficients, module)?)?;
    module.add_function(wrap_pyfunction!(weighted_clustering_coefficients, module)?)?;
    Ok(())
}
