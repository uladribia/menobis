//! Python bindings for ODME.

use odme_core::clustering::{
    clustering_coefficients as core_clustering,
    weighted_clustering_coefficients as core_weighted_clustering,
};
use odme_core::fitting::balance_no_self_loops;
use odme_core::generation::{
    sample_multinomial as core_sample_multinomial, sample_poisson as core_sample_poisson,
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
fn fit_balance_no_self_loops(
    s_out: Vec<f64>,
    s_in: Vec<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, bool, usize)> {
    if s_out.len() != s_in.len() {
        return Err(PyValueError::new_err(
            "s_out and s_in must have the same length",
        ));
    }
    let result = balance_no_self_loops(&s_out, &s_in, tolerance, max_iterations);
    Ok((result.x, result.y, result.converged, result.iterations))
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
    module.add_function(wrap_pyfunction!(fit_balance_no_self_loops, module)?)?;
    module.add_function(wrap_pyfunction!(sample_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(sample_multinomial, module)?)?;
    module.add_function(wrap_pyfunction!(clustering_coefficients, module)?)?;
    module.add_function(wrap_pyfunction!(weighted_clustering_coefficients, module)?)?;
    Ok(())
}
