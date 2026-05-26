//! Python bindings for MENoBiS.

use menobis_core::clustering::{
    clustering_coefficients as core_clustering,
    weighted_clustering_coefficients as core_weighted_clustering,
};
use menobis_core::filter::{
    absent_custom_poisson as core_absent_custom_poisson,
    absent_degree_events_binomial as core_absent_degree_events_binomial,
    absent_degree_events_geometric as core_absent_degree_events_geometric,
    absent_degree_events_negative_binomial as core_absent_degree_events_negative_binomial,
    absent_degree_events_poisson as core_absent_degree_events_poisson,
    absent_strength_binomial as core_absent_strength_binomial,
    absent_strength_cost_binomial as core_absent_strength_cost_binomial,
    absent_strength_cost_geometric as core_absent_strength_cost_geometric,
    absent_strength_cost_negative_binomial as core_absent_strength_cost_negative_binomial,
    absent_strength_cost_poisson as core_absent_strength_cost_poisson,
    absent_strength_degree_binomial as core_absent_strength_degree_binomial,
    absent_strength_degree_geometric as core_absent_strength_degree_geometric,
    absent_strength_degree_negative_binomial as core_absent_strength_degree_negative_binomial,
    absent_strength_degree_poisson as core_absent_strength_degree_poisson,
    absent_strength_edges_binomial as core_absent_strength_edges_binomial,
    absent_strength_edges_geometric as core_absent_strength_edges_geometric,
    absent_strength_edges_negative_binomial as core_absent_strength_edges_negative_binomial,
    absent_strength_edges_poisson as core_absent_strength_edges_poisson,
    absent_strength_geometric as core_absent_strength_geometric,
    absent_strength_negative_binomial as core_absent_strength_negative_binomial,
    absent_strength_poisson as core_absent_strength_poisson,
    benjamini_hochberg as core_benjamini_hochberg,
    filter_custom_poisson as core_filter_custom_poisson,
    filter_degree_events_binomial as core_filter_degree_events_binomial,
    filter_degree_events_geometric as core_filter_degree_events_geometric,
    filter_degree_events_negative_binomial as core_filter_degree_events_negative_binomial,
    filter_degree_events_poisson as core_filter_degree_events_poisson,
    filter_strength_binomial as core_filter_strength_binomial,
    filter_strength_cost_binomial as core_filter_strength_cost_binomial,
    filter_strength_cost_geometric as core_filter_strength_cost_geometric,
    filter_strength_cost_negative_binomial as core_filter_strength_cost_negative_binomial,
    filter_strength_cost_poisson as core_filter_strength_cost_poisson,
    filter_strength_degree_binomial as core_filter_strength_degree_binomial,
    filter_strength_degree_geometric as core_filter_strength_degree_geometric,
    filter_strength_degree_negative_binomial as core_filter_strength_degree_negative_binomial,
    filter_strength_degree_poisson as core_filter_strength_degree_poisson,
    filter_strength_edges_binomial as core_filter_strength_edges_binomial,
    filter_strength_edges_geometric as core_filter_strength_edges_geometric,
    filter_strength_edges_negative_binomial as core_filter_strength_edges_negative_binomial,
    filter_strength_edges_poisson as core_filter_strength_edges_poisson,
    filter_strength_geometric as core_filter_strength_geometric,
    filter_strength_negative_binomial as core_filter_strength_negative_binomial,
    filter_strength_poisson as core_filter_strength_poisson,
};
use menobis_core::fitting::mask::PairMask;
use menobis_core::fitting::{
    balance_degree_bernoulli, balance_sparse_masked_degree_bernoulli,
    balance_sparse_masked_strength_binomial, balance_sparse_masked_strength_degree_poisson,
    balance_sparse_masked_strength_poisson, balance_strength_binomial,
    balance_strength_degree_poisson, balance_strength_edges_poisson, balance_strength_poisson,
    balance_weighted_factors, fit_degree_events_geometric as core_fit_degree_events_geometric,
    fit_degree_events_negative_binomial as core_fit_degree_events_negative_binomial,
    fit_strength_cost_geometric as core_fit_strength_cost_geometric,
    fit_strength_cost_negative_binomial as core_fit_strength_cost_negative_binomial,
    fit_strength_degree_binomial as core_fit_strength_degree_binomial,
    fit_strength_degree_geometric as core_fit_strength_degree_geometric,
    fit_strength_degree_negative_binomial as core_fit_strength_degree_negative_binomial,
    fit_strength_edges_binomial as core_fit_strength_edges_binomial,
    fit_strength_edges_geometric as core_fit_strength_edges_geometric,
    fit_strength_edges_negative_binomial as core_fit_strength_edges_negative_binomial,
    fit_strength_geometric as core_fit_strength_geometric,
    fit_strength_negative_binomial as core_fit_strength_negative_binomial,
    fit_strength_poisson as core_fit_strength_poisson, WConicFitOptions,
};
use menobis_core::fitting::{
    fit_partial_degree as core_fit_partial_degree,
    fit_partial_strength as core_fit_partial_strength,
    fit_partial_strength_cost as core_fit_partial_strength_cost,
    fit_partial_strength_cost_binomial_coordinates as core_fit_partial_strength_cost_binomial_coordinates,
    fit_partial_strength_cost_coordinates as core_fit_partial_strength_cost_coordinates,
    fit_partial_strength_cost_w_coordinates as core_fit_partial_strength_cost_w_coordinates,
    fit_partial_strength_degree as core_fit_partial_strength_degree,
    fit_partial_strength_edges as core_fit_partial_strength_edges,
};
use menobis_core::fitting::{
    fit_strength_cost_binomial as core_fit_strength_cost_binomial,
    fit_strength_cost_binomial_coordinates as core_fit_strength_cost_binomial_coordinates,
    fit_strength_cost_poisson as core_fit_strength_cost,
    fit_strength_cost_poisson_coordinates as core_fit_strength_cost_coordinates,
    fit_strength_cost_w_lbfgs as core_fit_strength_cost_w_coordinates,
    fit_strength_cost_w_lbfgs as core_fit_strength_cost_w_lbfgs, CostFitOptions,
};
use menobis_core::generation::{
    sample_custom_multinomial as core_sample_custom_multinomial,
    sample_custom_poisson as core_sample_custom_poisson,
    sample_degree_events_binomial as core_sample_degree_events_binomial,
    sample_degree_events_geometric as core_sample_degree_events_geometric,
    sample_degree_events_negative_binomial as core_sample_degree_events_negative_binomial,
    sample_degree_events_poisson as core_sample_degree_events_poisson,
    sample_strength_binomial as core_sample_strength_binomial,
    sample_strength_cost_binomial as core_sample_strength_cost_binomial,
    sample_strength_cost_geometric as core_sample_strength_cost_geometric,
    sample_strength_cost_negative_binomial as core_sample_strength_cost_negative_binomial,
    sample_strength_cost_poisson as core_sample_strength_cost_poisson,
    sample_strength_degree_binomial as core_sample_strength_degree_binomial,
    sample_strength_degree_geometric as core_sample_strength_degree_geometric,
    sample_strength_degree_negative_binomial as core_sample_strength_degree_negative_binomial,
    sample_strength_degree_poisson as core_sample_strength_degree_poisson,
    sample_strength_edges_binomial as core_sample_strength_edges_binomial,
    sample_strength_edges_geometric as core_sample_strength_edges_geometric,
    sample_strength_edges_negative_binomial as core_sample_strength_edges_negative_binomial,
    sample_strength_edges_poisson as core_sample_strength_edges_poisson,
    sample_strength_geometric as core_sample_strength_geometric,
    sample_strength_multinomial as core_sample_strength_multinomial,
    sample_strength_negative_binomial as core_sample_strength_negative_binomial,
    sample_strength_poisson as core_sample_strength_poisson,
    sample_strength_poisson_multinomial as core_sample_strength_poisson_multinomial,
    sample_strength_stub_matching as core_sample_strength_stub_matching, SparseCostEntries,
};
use menobis_core::graph::{
    directed_degrees as core_directed_degrees, directed_strengths as core_directed_strengths,
    WeightedEdge,
};
use menobis_core::stats::{
    compute_all_node_stats as core_compute_all_stats,
    weight_distribution as core_weight_distribution,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type FitPair = (Vec<f64>, Vec<f64>, bool, usize);
type FitStrengthEdges = (Vec<f64>, Vec<f64>, f64, bool, usize);
type FitStrengthDegree = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, bool, usize);

type FitStrengthCost = (Vec<f64>, Vec<f64>, f64, bool, usize);
type WStrengthFit = (
    Vec<f64>,
    Vec<f64>,
    u32,
    String,
    f64,
    usize,
    f64,
    f64,
    f64,
    f64,
    (usize, usize, usize, usize, usize, usize),
);
type WStrengthCostFit = (
    Vec<f64>,
    Vec<f64>,
    f64,
    u32,
    String,
    f64,
    usize,
    (f64, f64, f64, f64, f64),
    (usize, usize, usize, usize, usize, usize),
);
type WStrengthEdgesFit = (
    Vec<f64>,
    Vec<f64>,
    f64,
    u32,
    String,
    f64,
    usize,
    (f64, f64, f64, f64, f64),
    (usize, usize, usize, usize, usize, usize),
);
type WStrengthDegreeFit = (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    u32,
    String,
    f64,
    usize,
    (f64, f64, f64, f64, f64),
    (usize, usize, usize, usize, usize, usize),
);
type ObservedFilter = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
type AbsentFilter = (Vec<u64>, Vec<u64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// Return the version of the Rust core exposed through Python.
#[pyfunction]
fn rust_core_version() -> &'static str {
    menobis_core::VERSION
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
fn fit_masked_strength_poisson(
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
    let pair_mask = PairMask::from_dense(n, &mask);
    let result = balance_sparse_masked_strength_poisson(
        &strength_out,
        &strength_in,
        &pair_mask,
        tolerance,
        max_iterations,
    );
    Ok((result.x, result.y, result.converged, result.iterations))
}

#[pyfunction]
fn fit_masked_degree_bernoulli(
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
    let pair_mask = PairMask::from_dense(n, &mask);
    let r = balance_sparse_masked_degree_bernoulli(
        &degree_out,
        &degree_in,
        &pair_mask,
        tolerance,
        max_iterations,
    );
    Ok((r.x, r.y, r.converged, r.iterations))
}

#[pyfunction]
fn fit_masked_strength_degree_poisson(
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
    let pair_mask = PairMask::from_dense(n, &mask);
    let r = balance_sparse_masked_strength_degree_poisson(
        &strength_out,
        &strength_in,
        &degree_out,
        &degree_in,
        &pair_mask,
        tolerance,
        max_iterations,
    );
    Ok((r.x, r.y, r.z, r.w, r.converged, r.iterations))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn fit_strength_cost_poisson(
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

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn fit_strength_cost_poisson_coordinates(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
    target_cost: f64,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<FitStrengthCost> {
    if strength_out.len() != strength_in.len()
        || strength_out.len() != coord_x.len()
        || strength_out.len() != coord_y.len()
    {
        return Err(PyValueError::new_err(
            "strength and coordinate arrays must have same length",
        ));
    }
    let result = core_fit_strength_cost_coordinates(
        &strength_out,
        &strength_in,
        &coord_x,
        &coord_y,
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

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn fit_strength_cost_binomial(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    cost_sources: Vec<usize>,
    cost_targets: Vec<usize>,
    cost_values: Vec<f64>,
    target_cost: f64,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<FitStrengthCost> {
    if strength_out.len() != strength_in.len() {
        return Err(PyValueError::new_err(
            "strength_out and strength_in must have same length",
        ));
    }
    let result = core_fit_strength_cost_binomial(
        &strength_out,
        &strength_in,
        &cost_sources,
        &cost_targets,
        &cost_values,
        target_cost,
        layers,
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

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn fit_strength_cost_binomial_coordinates(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
    target_cost: f64,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<FitStrengthCost> {
    if strength_out.len() != strength_in.len()
        || strength_out.len() != coord_x.len()
        || strength_out.len() != coord_y.len()
    {
        return Err(PyValueError::new_err(
            "strength and coordinate arrays must have same length",
        ));
    }
    let result = core_fit_strength_cost_binomial_coordinates(
        &strength_out,
        &strength_in,
        &coord_x,
        &coord_y,
        target_cost,
        layers,
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

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn fit_strength_cost_w_coordinates(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
    target_cost: f64,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<FitStrengthCost> {
    if strength_out.len() != strength_in.len()
        || strength_out.len() != coord_x.len()
        || strength_out.len() != coord_y.len()
    {
        return Err(PyValueError::new_err(
            "strength and coordinate arrays must have same length",
        ));
    }
    let result = core_fit_strength_cost_w_coordinates(
        &strength_out,
        &strength_in,
        &coord_x,
        &coord_y,
        target_cost,
        layers,
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

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn fit_strength_cost_w_lbfgs(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
    target_cost: f64,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<FitStrengthCost> {
    if strength_out.len() != strength_in.len()
        || strength_out.len() != coord_x.len()
        || strength_out.len() != coord_y.len()
    {
        return Err(PyValueError::new_err(
            "strength and coordinate arrays must have same length",
        ));
    }
    let result = core_fit_strength_cost_w_lbfgs(
        &strength_out,
        &strength_in,
        &coord_x,
        &coord_y,
        target_cost,
        layers,
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
fn fit_degree_bernoulli(
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
    let result = balance_degree_bernoulli(
        &degree_out,
        &degree_in,
        self_loops,
        tolerance,
        max_iterations,
    );
    Ok((result.x, result.y, result.converged, result.iterations))
}

#[pyfunction]
fn fit_strength_edges_poisson(
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
    let result = balance_strength_edges_poisson(
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
fn fit_strength_edges_binomial(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    target_edges: f64,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<FitStrengthEdges> {
    if strength_out.len() != strength_in.len() {
        return Err(PyValueError::new_err(
            "strength arrays must have same length",
        ));
    }
    let result = core_fit_strength_edges_binomial(
        &strength_out,
        &strength_in,
        target_edges,
        layers,
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
fn fit_strength_degree_poisson(
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
    let result = balance_strength_degree_poisson(
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
#[allow(clippy::too_many_arguments)]
fn fit_strength_degree_binomial(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    degree_out: Vec<f64>,
    degree_in: Vec<f64>,
    layers: u32,
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
    let result = core_fit_strength_degree_binomial(
        &strength_out,
        &strength_in,
        &degree_out,
        &degree_in,
        layers,
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
fn fit_strength_poisson(
    s_out: Vec<f64>,
    s_in: Vec<f64>,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<FitPair> {
    if s_out.len() != s_in.len() {
        return Err(PyValueError::new_err(
            "s_out and s_in must have the same length",
        ));
    }
    let result = core_fit_strength_poisson(&s_out, &s_in, self_loops, tolerance, max_iterations);
    Ok((result.x, result.y, result.converged, result.iterations))
}

fn w_strength_fit_tuple(result: menobis_core::fitting::WStrengthFitResult) -> WStrengthFit {
    let status = format!("{:?}", result.status).to_lowercase();
    let metrics = result.metrics;
    (
        result.x,
        result.y,
        result.layers,
        status,
        result.objective,
        result.iterations,
        result.min_margin,
        result.max_q,
        result.max_strength_residual,
        result.total_strength_residual,
        (
            metrics.variables,
            metrics.auxiliary_variables,
            metrics.exponential_cones,
            metrics.power_cones,
            metrics.linear_constraints,
            metrics.sparse_nonzeros,
        ),
    )
}

fn w_strength_cost_fit_tuple(
    result: menobis_core::fitting::WStrengthCostFitResult,
) -> WStrengthCostFit {
    let status = format!("{:?}", result.status).to_lowercase();
    let metrics = result.metrics;
    (
        result.x,
        result.y,
        result.gamma,
        result.layers,
        status,
        result.objective,
        result.iterations,
        (
            result.min_margin,
            result.max_q,
            result.max_strength_residual,
            result.total_strength_residual,
            result.cost_residual,
        ),
        (
            metrics.variables,
            metrics.auxiliary_variables,
            metrics.exponential_cones,
            metrics.power_cones,
            metrics.linear_constraints,
            metrics.sparse_nonzeros,
        ),
    )
}

fn w_strength_edges_fit_tuple(
    result: menobis_core::fitting::WStrengthEdgesFitResult,
) -> WStrengthEdgesFit {
    let status = format!("{:?}", result.status).to_lowercase();
    let metrics = result.metrics;
    (
        result.x,
        result.y,
        result.lam,
        result.layers,
        status,
        result.objective,
        result.iterations,
        (
            result.min_margin,
            result.max_q,
            result.max_strength_residual,
            result.total_strength_residual,
            result.edge_residual,
        ),
        (
            metrics.variables,
            metrics.auxiliary_variables,
            metrics.exponential_cones,
            metrics.power_cones,
            metrics.linear_constraints,
            metrics.sparse_nonzeros,
        ),
    )
}

fn w_strength_degree_fit_tuple(
    result: menobis_core::fitting::WStrengthDegreeFitResult,
) -> WStrengthDegreeFit {
    let status = format!("{:?}", result.status).to_lowercase();
    let metrics = result.metrics;
    (
        result.x,
        result.y,
        result.z,
        result.w,
        result.layers,
        status,
        result.objective,
        result.iterations,
        (
            result.min_margin,
            result.max_q,
            result.max_strength_residual,
            result.total_strength_residual,
            result.max_degree_residual,
        ),
        (
            metrics.variables,
            metrics.auxiliary_variables,
            metrics.exponential_cones,
            metrics.power_cones,
            metrics.linear_constraints,
            metrics.sparse_nonzeros,
        ),
    )
}

#[pyfunction]
fn fit_strength_geometric(
    s_out: Vec<f64>,
    s_in: Vec<f64>,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<WStrengthFit> {
    if s_out.len() != s_in.len() {
        return Err(PyValueError::new_err(
            "s_out and s_in must have the same length",
        ));
    }
    let result = core_fit_strength_geometric(
        &s_out,
        &s_in,
        WConicFitOptions {
            self_loops,
            tolerance,
            max_iterations,
        },
    );
    Ok(w_strength_fit_tuple(result))
}

#[pyfunction]
fn fit_strength_negative_binomial(
    s_out: Vec<f64>,
    s_in: Vec<f64>,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<WStrengthFit> {
    if s_out.len() != s_in.len() {
        return Err(PyValueError::new_err(
            "s_out and s_in must have the same length",
        ));
    }
    if layers <= 1 {
        return Err(PyValueError::new_err(
            "negative binomial W fitting requires layers > 1; use geometric for M = 1",
        ));
    }
    let result = core_fit_strength_negative_binomial(
        &s_out,
        &s_in,
        layers,
        WConicFitOptions {
            self_loops,
            tolerance,
            max_iterations,
        },
    );
    Ok(w_strength_fit_tuple(result))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_strength_cost_geometric(
    s_out: Vec<f64>,
    s_in: Vec<f64>,
    cost_sources: Vec<u64>,
    cost_targets: Vec<u64>,
    cost_values: Vec<f64>,
    target_cost: f64,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<WStrengthCostFit> {
    if s_out.len() != s_in.len() {
        return Err(PyValueError::new_err(
            "s_out and s_in must have the same length",
        ));
    }
    if cost_sources.len() != cost_targets.len() || cost_sources.len() != cost_values.len() {
        return Err(PyValueError::new_err(
            "cost_sources, cost_targets, and cost_values must have the same length",
        ));
    }
    let result = core_fit_strength_cost_geometric(
        &s_out,
        &s_in,
        &cost_sources,
        &cost_targets,
        &cost_values,
        target_cost,
        WConicFitOptions {
            self_loops,
            tolerance,
            max_iterations,
        },
    );
    Ok(w_strength_cost_fit_tuple(result))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_strength_cost_negative_binomial(
    s_out: Vec<f64>,
    s_in: Vec<f64>,
    cost_sources: Vec<u64>,
    cost_targets: Vec<u64>,
    cost_values: Vec<f64>,
    target_cost: f64,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<WStrengthCostFit> {
    if s_out.len() != s_in.len() {
        return Err(PyValueError::new_err(
            "s_out and s_in must have the same length",
        ));
    }
    if cost_sources.len() != cost_targets.len() || cost_sources.len() != cost_values.len() {
        return Err(PyValueError::new_err(
            "cost_sources, cost_targets, and cost_values must have the same length",
        ));
    }
    if layers <= 1 {
        return Err(PyValueError::new_err(
            "negative binomial W fitting requires layers > 1; use geometric for M = 1",
        ));
    }
    let result = core_fit_strength_cost_negative_binomial(
        &s_out,
        &s_in,
        &cost_sources,
        &cost_targets,
        &cost_values,
        target_cost,
        layers,
        WConicFitOptions {
            self_loops,
            tolerance,
            max_iterations,
        },
    );
    Ok(w_strength_cost_fit_tuple(result))
}

#[pyfunction]
fn fit_strength_edges_geometric(
    s_out: Vec<f64>,
    s_in: Vec<f64>,
    target_edges: f64,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<WStrengthEdgesFit> {
    if s_out.len() != s_in.len() {
        return Err(PyValueError::new_err(
            "s_out and s_in must have the same length",
        ));
    }
    let result = core_fit_strength_edges_geometric(
        &s_out,
        &s_in,
        target_edges,
        WConicFitOptions {
            self_loops,
            tolerance,
            max_iterations,
        },
    );
    Ok(w_strength_edges_fit_tuple(result))
}

#[pyfunction]
fn fit_strength_edges_negative_binomial(
    s_out: Vec<f64>,
    s_in: Vec<f64>,
    target_edges: f64,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<WStrengthEdgesFit> {
    if s_out.len() != s_in.len() {
        return Err(PyValueError::new_err(
            "s_out and s_in must have the same length",
        ));
    }
    if layers <= 1 {
        return Err(PyValueError::new_err(
            "negative binomial W fitting requires layers > 1; use geometric for M = 1",
        ));
    }
    let result = core_fit_strength_edges_negative_binomial(
        &s_out,
        &s_in,
        target_edges,
        layers,
        WConicFitOptions {
            self_loops,
            tolerance,
            max_iterations,
        },
    );
    Ok(w_strength_edges_fit_tuple(result))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_strength_degree_geometric(
    s_out: Vec<f64>,
    s_in: Vec<f64>,
    k_out: Vec<f64>,
    k_in: Vec<f64>,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<WStrengthDegreeFit> {
    if s_out.len() != s_in.len() || s_out.len() != k_out.len() || s_out.len() != k_in.len() {
        return Err(PyValueError::new_err(
            "all input arrays must have the same length",
        ));
    }
    let result = core_fit_strength_degree_geometric(
        &s_out,
        &s_in,
        &k_out,
        &k_in,
        WConicFitOptions {
            self_loops,
            tolerance,
            max_iterations,
        },
    );
    Ok(w_strength_degree_fit_tuple(result))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_strength_degree_negative_binomial(
    s_out: Vec<f64>,
    s_in: Vec<f64>,
    k_out: Vec<f64>,
    k_in: Vec<f64>,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<WStrengthDegreeFit> {
    if s_out.len() != s_in.len() || s_out.len() != k_out.len() || s_out.len() != k_in.len() {
        return Err(PyValueError::new_err(
            "all input arrays must have the same length",
        ));
    }
    if layers <= 1 {
        return Err(PyValueError::new_err(
            "negative binomial W fitting requires layers > 1; use geometric for M = 1",
        ));
    }
    let result = core_fit_strength_degree_negative_binomial(
        &s_out,
        &s_in,
        &k_out,
        &k_in,
        layers,
        WConicFitOptions {
            self_loops,
            tolerance,
            max_iterations,
        },
    );
    Ok(w_strength_degree_fit_tuple(result))
}

#[pyfunction]
fn fit_strength_poisson_no_self_loops(
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
    let result = balance_strength_poisson(&s_out, &s_in, tolerance, max_iterations);
    Ok((result.x, result.y, result.converged, result.iterations))
}

#[pyfunction]
fn sample_strength_stub_matching(
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
            "stub_matching requires balanced strengths",
        ));
    }
    let sample = core_sample_strength_stub_matching(&strength_out, &strength_in, seed);
    Ok((sample.sources, sample.targets, sample.weights))
}

#[pyfunction]
fn sample_custom_poisson(
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
    let sample = core_sample_custom_poisson(&sources, &targets, &probabilities, total_events, seed);
    Ok((sample.sources, sample.targets, sample.weights))
}

#[pyfunction]
fn sample_custom_multinomial(
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
    let sample =
        core_sample_custom_multinomial(&sources, &targets, &probabilities, total_events, seed);
    Ok((sample.sources, sample.targets, sample.weights))
}

#[pyfunction]
fn sample_strength_poisson_multinomial(
    x: Vec<f64>,
    y: Vec<f64>,
    self_loops: bool,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    let sample = core_sample_strength_poisson_multinomial(&x, &y, self_loops, seed);
    Ok((sample.sources, sample.targets, sample.weights))
}

#[pyfunction]
fn sample_strength_edges_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    lam: f64,
    self_loops: bool,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    let sample = core_sample_strength_edges_poisson(&x, &y, lam, self_loops, seed);
    Ok((sample.sources, sample.targets, sample.weights))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sample_strength_cost_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    cost_sources: Vec<usize>,
    cost_targets: Vec<usize>,
    cost_values: Vec<f64>,
    self_loops: bool,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    if cost_sources.len() != cost_targets.len() || cost_sources.len() != cost_values.len() {
        return Err(PyValueError::new_err("cost arrays must have same length"));
    }
    let costs = SparseCostEntries {
        sources: &cost_sources,
        targets: &cost_targets,
        values: &cost_values,
    };
    let sample = core_sample_strength_cost_poisson(&x, &y, gamma, &costs, self_loops, seed);
    Ok((sample.sources, sample.targets, sample.weights))
}

#[pyfunction]
fn sample_strength_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_strength_poisson(&x, &y, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
fn sample_strength_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_strength_geometric(&x, &y, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
fn sample_strength_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_strength_binomial(&x, &y, layers, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sample_strength_cost_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    cost_sources: Vec<usize>,
    cost_targets: Vec<usize>,
    cost_values: Vec<f64>,
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let costs = SparseCostEntries {
        sources: &cost_sources,
        targets: &cost_targets,
        values: &cost_values,
    };
    let edges = core_sample_strength_cost_binomial(&x, &y, gamma, &costs, layers, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sample_strength_cost_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    cost_sources: Vec<usize>,
    cost_targets: Vec<usize>,
    cost_values: Vec<f64>,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let costs = SparseCostEntries {
        sources: &cost_sources,
        targets: &cost_targets,
        values: &cost_values,
    };
    let edges = core_sample_strength_cost_geometric(&x, &y, gamma, &costs, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sample_strength_cost_negative_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    cost_sources: Vec<usize>,
    cost_targets: Vec<usize>,
    cost_values: Vec<f64>,
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let costs = SparseCostEntries {
        sources: &cost_sources,
        targets: &cost_targets,
        values: &cost_values,
    };
    let edges = core_sample_strength_cost_negative_binomial(
        &x, &y, gamma, &costs, layers, self_loops, seed,
    );
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
fn sample_strength_edges_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    lam: f64,
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_strength_edges_binomial(&x, &y, lam, layers, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
fn sample_strength_degree_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    w: Vec<f64>,
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_strength_degree_binomial(&x, &y, &z, &w, layers, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
fn sample_degree_events_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    positive_weight_rate: f64,
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges =
        core_sample_degree_events_binomial(&x, &y, positive_weight_rate, layers, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
fn sample_strength_edges_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    lam: f64,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_strength_edges_geometric(&x, &y, lam, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
fn sample_strength_edges_negative_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    lam: f64,
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_strength_edges_negative_binomial(&x, &y, lam, layers, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
fn sample_strength_degree_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    w: Vec<f64>,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_strength_degree_geometric(&x, &y, &z, &w, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
fn sample_strength_degree_negative_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    w: Vec<f64>,
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges =
        core_sample_strength_degree_negative_binomial(&x, &y, &z, &w, layers, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
fn sample_degree_events_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    positive_weight_rate: f64,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_degree_events_geometric(&x, &y, positive_weight_rate, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
fn sample_degree_events_negative_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    positive_weight_rate: f64,
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_degree_events_negative_binomial(
        &x,
        &y,
        positive_weight_rate,
        layers,
        self_loops,
        seed,
    );
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
fn sample_strength_negative_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_strength_negative_binomial(&x, &y, layers, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
fn fit_strength_binomial(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> (Vec<f64>, Vec<f64>, bool, usize) {
    let result = balance_strength_binomial(
        &strength_out,
        &strength_in,
        layers,
        self_loops,
        tolerance,
        max_iterations,
    );
    (result.x, result.y, result.converged, result.iterations)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_degree_events_geometric(
    degree_out: Vec<f64>,
    degree_in: Vec<f64>,
    total_events: u64,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> (Vec<f64>, Vec<f64>, f64, f64, bool, usize) {
    let result = core_fit_degree_events_geometric(
        &degree_out,
        &degree_in,
        total_events,
        self_loops,
        tolerance,
        max_iterations,
    );
    (
        result.x,
        result.y,
        result.q,
        result.positive_mean,
        result.converged,
        result.iterations,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_degree_events_negative_binomial(
    degree_out: Vec<f64>,
    degree_in: Vec<f64>,
    total_events: u64,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> (Vec<f64>, Vec<f64>, f64, f64, bool, usize) {
    let result = core_fit_degree_events_negative_binomial(
        &degree_out,
        &degree_in,
        total_events,
        layers,
        self_loops,
        tolerance,
        max_iterations,
    );
    (
        result.x,
        result.y,
        result.q,
        result.positive_mean,
        result.converged,
        result.iterations,
    )
}

#[pyfunction]
fn fit_masked_strength_binomial(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    mask: Vec<bool>,
    layers: u32,
    tolerance: f64,
    max_iterations: usize,
) -> (Vec<f64>, Vec<f64>, bool, usize) {
    let n = strength_out.len();
    let pair_mask = PairMask::from_dense(n, &mask);
    let result = balance_sparse_masked_strength_binomial(
        &strength_out,
        &strength_in,
        &pair_mask,
        layers,
        tolerance,
        max_iterations,
    );
    (result.x, result.y, result.converged, result.iterations)
}

type PartialResult = (Vec<u64>, Vec<u64>, Vec<f64>, bool, usize);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_partial_strength_poisson_full(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    known_src: Vec<u64>,
    known_tgt: Vec<u64>,
    known_rate: Vec<f64>,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PartialResult {
    let r = core_fit_partial_strength(
        &strength_out,
        &strength_in,
        &known_src,
        &known_tgt,
        &known_rate,
        self_loops,
        tolerance,
        max_iterations,
    );
    (r.sources, r.targets, r.rates, r.converged, r.iterations)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_partial_degree_poisson_full(
    degree_out: Vec<f64>,
    degree_in: Vec<f64>,
    known_src: Vec<u64>,
    known_tgt: Vec<u64>,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PartialResult {
    let r = core_fit_partial_degree(
        &degree_out,
        &degree_in,
        &known_src,
        &known_tgt,
        self_loops,
        tolerance,
        max_iterations,
    );
    (r.sources, r.targets, r.rates, r.converged, r.iterations)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_partial_strength_degree_poisson_full(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    degree_out: Vec<f64>,
    degree_in: Vec<f64>,
    known_src: Vec<u64>,
    known_tgt: Vec<u64>,
    known_rate: Vec<f64>,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PartialResult {
    let r = core_fit_partial_strength_degree(
        &strength_out,
        &strength_in,
        &degree_out,
        &degree_in,
        &known_src,
        &known_tgt,
        &known_rate,
        self_loops,
        tolerance,
        max_iterations,
    );
    (r.sources, r.targets, r.rates, r.converged, r.iterations)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_partial_strength_edges_poisson_full(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    known_src: Vec<u64>,
    known_tgt: Vec<u64>,
    known_rate: Vec<f64>,
    target_edges: f64,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PartialResult {
    let r = core_fit_partial_strength_edges(
        &strength_out,
        &strength_in,
        &known_src,
        &known_tgt,
        &known_rate,
        target_edges,
        self_loops,
        tolerance,
        max_iterations,
    );
    (r.sources, r.targets, r.rates, r.converged, r.iterations)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_partial_strength_cost_poisson_full(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    known_src: Vec<u64>,
    known_tgt: Vec<u64>,
    known_rate: Vec<f64>,
    cost_sources: Vec<usize>,
    cost_targets: Vec<usize>,
    cost_values: Vec<f64>,
    target_cost: f64,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PartialResult {
    let r = core_fit_partial_strength_cost(
        &strength_out,
        &strength_in,
        &known_src,
        &known_tgt,
        &known_rate,
        &cost_sources,
        &cost_targets,
        &cost_values,
        target_cost,
        self_loops,
        tolerance,
        max_iterations,
    );
    (r.sources, r.targets, r.rates, r.converged, r.iterations)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_partial_strength_cost_poisson_coordinates_full(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    known_src: Vec<u64>,
    known_tgt: Vec<u64>,
    known_rate: Vec<f64>,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
    target_cost: f64,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<PartialResult> {
    if strength_out.len() != strength_in.len()
        || strength_out.len() != coord_x.len()
        || strength_out.len() != coord_y.len()
    {
        return Err(PyValueError::new_err(
            "strength and coordinate arrays must have same length",
        ));
    }
    let r = core_fit_partial_strength_cost_coordinates(
        &strength_out,
        &strength_in,
        &known_src,
        &known_tgt,
        &known_rate,
        &coord_x,
        &coord_y,
        target_cost,
        self_loops,
        tolerance,
        max_iterations,
    );
    Ok((r.sources, r.targets, r.rates, r.converged, r.iterations))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_partial_strength_cost_binomial_coordinates_full(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    known_src: Vec<u64>,
    known_tgt: Vec<u64>,
    known_rate: Vec<f64>,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
    target_cost: f64,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<PartialResult> {
    if strength_out.len() != strength_in.len()
        || strength_out.len() != coord_x.len()
        || strength_out.len() != coord_y.len()
    {
        return Err(PyValueError::new_err(
            "strength and coordinate arrays must have same length",
        ));
    }
    let r = core_fit_partial_strength_cost_binomial_coordinates(
        &strength_out,
        &strength_in,
        &known_src,
        &known_tgt,
        &known_rate,
        &coord_x,
        &coord_y,
        target_cost,
        layers,
        self_loops,
        tolerance,
        max_iterations,
    );
    Ok((r.sources, r.targets, r.rates, r.converged, r.iterations))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_partial_strength_cost_w_coordinates_full(
    strength_out: Vec<f64>,
    strength_in: Vec<f64>,
    known_src: Vec<u64>,
    known_tgt: Vec<u64>,
    known_rate: Vec<f64>,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
    target_cost: f64,
    layers: u32,
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<PartialResult> {
    if strength_out.len() != strength_in.len()
        || strength_out.len() != coord_x.len()
        || strength_out.len() != coord_y.len()
    {
        return Err(PyValueError::new_err(
            "strength and coordinate arrays must have same length",
        ));
    }
    let r = core_fit_partial_strength_cost_w_coordinates(
        &strength_out,
        &strength_in,
        &known_src,
        &known_tgt,
        &known_rate,
        &coord_x,
        &coord_y,
        target_cost,
        layers,
        self_loops,
        tolerance,
        max_iterations,
    );
    Ok((r.sources, r.targets, r.rates, r.converged, r.iterations))
}

#[pyfunction]
fn sample_degree_events_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    positive_weight_rate: f64,
    self_loops: bool,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    let sample = core_sample_degree_events_poisson(&x, &y, positive_weight_rate, self_loops, seed);
    Ok((sample.sources, sample.targets, sample.weights))
}

#[pyfunction]
fn sample_strength_degree_poisson(
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
    let sample = core_sample_strength_degree_poisson(
        &degree_x, &degree_y, &excess_x, &excess_y, self_loops, seed,
    );
    Ok((sample.sources, sample.targets, sample.weights))
}

#[pyfunction]
fn sample_strength_multinomial(
    x: Vec<f64>,
    y: Vec<f64>,
    total_events: u64,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_strength_multinomial(&x, &y, total_events, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
fn filter_strength_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    if sources.len() != targets.len() || sources.len() != weights.len() {
        return Err(PyValueError::new_err(
            "sources, targets, and weights must have same length",
        ));
    }
    let result = core_filter_strength_poisson(&x, &y, &sources, &targets, &weights);
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_strength_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    let result = core_absent_strength_poisson(
        &x,
        &y,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
fn filter_custom_poisson(
    rate_sources: Vec<u64>,
    rate_targets: Vec<u64>,
    rates: Vec<f64>,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    if rate_sources.len() != rate_targets.len() || rate_sources.len() != rates.len() {
        return Err(PyValueError::new_err("rate arrays must have same length"));
    }
    let result = core_filter_custom_poisson(
        &rate_sources,
        &rate_targets,
        &rates,
        &sources,
        &targets,
        &weights,
    );
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (rate_sources, rate_targets, rates, sources, targets, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_custom_poisson(
    rate_sources: Vec<u64>,
    rate_targets: Vec<u64>,
    rates: Vec<f64>,
    sources: Vec<u64>,
    targets: Vec<u64>,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    if rate_sources.len() != rate_targets.len() || rate_sources.len() != rates.len() {
        return Err(PyValueError::new_err("rate arrays must have same length"));
    }
    let result = core_absent_custom_poisson(
        &rate_sources,
        &rate_targets,
        &rates,
        &sources,
        &targets,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
fn filter_strength_edges_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    lam: f64,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    let result = core_filter_strength_edges_poisson(&x, &y, lam, &sources, &targets, &weights);
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, lam, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_strength_edges_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    lam: f64,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    let result = core_absent_strength_edges_poisson(
        &x,
        &y,
        lam,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn filter_strength_cost_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    cost_sources: Vec<usize>,
    cost_targets: Vec<usize>,
    cost_values: Vec<f64>,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    let result = core_filter_strength_cost_poisson(
        &x,
        &y,
        gamma,
        &cost_sources,
        &cost_targets,
        &cost_values,
        &sources,
        &targets,
        &weights,
    );
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, gamma, cost_sources, cost_targets, cost_values, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_strength_cost_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    cost_sources: Vec<usize>,
    cost_targets: Vec<usize>,
    cost_values: Vec<f64>,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    let result = core_absent_strength_cost_poisson(
        &x,
        &y,
        gamma,
        &cost_sources,
        &cost_targets,
        &cost_values,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
fn filter_strength_degree_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    w: Vec<f64>,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    if x.len() != y.len() || x.len() != z.len() || x.len() != w.len() {
        return Err(PyValueError::new_err("x, y, z, w must have same length"));
    }
    let result = core_filter_strength_degree_poisson(&x, &y, &z, &w, &sources, &targets, &weights);
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, z, w, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_strength_degree_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    w: Vec<f64>,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    if x.len() != y.len() || x.len() != z.len() || x.len() != w.len() {
        return Err(PyValueError::new_err("x, y, z, w must have same length"));
    }
    let result = core_absent_strength_degree_poisson(
        &x,
        &y,
        &z,
        &w,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
fn filter_degree_events_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    positive_weight_rate: f64,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    let result = core_filter_degree_events_poisson(
        &x,
        &y,
        positive_weight_rate,
        &sources,
        &targets,
        &weights,
    );
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, positive_weight_rate, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_degree_events_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    positive_weight_rate: f64,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    let result = core_absent_degree_events_poisson(
        &x,
        &y,
        positive_weight_rate,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
fn filter_strength_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    let result = core_filter_strength_geometric(&x, &y, &sources, &targets, &weights);
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_strength_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    let result = core_absent_strength_geometric(
        &x,
        &y,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
fn filter_strength_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    let result = core_filter_strength_binomial(&x, &y, layers, &sources, &targets, &weights);
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, layers, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_strength_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    let result = core_absent_strength_binomial(
        &x,
        &y,
        layers,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn filter_strength_cost_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    cost_sources: Vec<usize>,
    cost_targets: Vec<usize>,
    cost_values: Vec<f64>,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    let result = core_filter_strength_cost_binomial(
        &x,
        &y,
        gamma,
        &cost_sources,
        &cost_targets,
        &cost_values,
        layers,
        &sources,
        &targets,
        &weights,
    );
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, gamma, cost_sources, cost_targets, cost_values, layers, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_strength_cost_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    cost_sources: Vec<usize>,
    cost_targets: Vec<usize>,
    cost_values: Vec<f64>,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    let result = core_absent_strength_cost_binomial(
        &x,
        &y,
        gamma,
        &cost_sources,
        &cost_targets,
        &cost_values,
        layers,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
fn filter_strength_edges_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    lam: f64,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    let result =
        core_filter_strength_edges_binomial(&x, &y, lam, layers, &sources, &targets, &weights);
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, lam, layers, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_strength_edges_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    lam: f64,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    let result = core_absent_strength_edges_binomial(
        &x,
        &y,
        lam,
        layers,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn filter_strength_degree_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    w: Vec<f64>,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    let result =
        core_filter_strength_degree_binomial(&x, &y, &z, &w, layers, &sources, &targets, &weights);
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, z, w, layers, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_strength_degree_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    w: Vec<f64>,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    let result = core_absent_strength_degree_binomial(
        &x,
        &y,
        &z,
        &w,
        layers,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
fn filter_degree_events_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    positive_weight_rate: f64,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    let result = core_filter_degree_events_binomial(
        &x,
        &y,
        positive_weight_rate,
        layers,
        &sources,
        &targets,
        &weights,
    );
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, positive_weight_rate, layers, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_degree_events_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    positive_weight_rate: f64,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    let result = core_absent_degree_events_binomial(
        &x,
        &y,
        positive_weight_rate,
        layers,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn filter_strength_cost_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    cost_sources: Vec<usize>,
    cost_targets: Vec<usize>,
    cost_values: Vec<f64>,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    let result = core_filter_strength_cost_geometric(
        &x,
        &y,
        gamma,
        &cost_sources,
        &cost_targets,
        &cost_values,
        &sources,
        &targets,
        &weights,
    );
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, gamma, cost_sources, cost_targets, cost_values, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_strength_cost_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    cost_sources: Vec<usize>,
    cost_targets: Vec<usize>,
    cost_values: Vec<f64>,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    let result = core_absent_strength_cost_geometric(
        &x,
        &y,
        gamma,
        &cost_sources,
        &cost_targets,
        &cost_values,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn filter_strength_cost_negative_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    cost_sources: Vec<usize>,
    cost_targets: Vec<usize>,
    cost_values: Vec<f64>,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    let result = core_filter_strength_cost_negative_binomial(
        &x,
        &y,
        gamma,
        &cost_sources,
        &cost_targets,
        &cost_values,
        layers,
        &sources,
        &targets,
        &weights,
    );
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, gamma, cost_sources, cost_targets, cost_values, layers, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_strength_cost_negative_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    cost_sources: Vec<usize>,
    cost_targets: Vec<usize>,
    cost_values: Vec<f64>,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    let result = core_absent_strength_cost_negative_binomial(
        &x,
        &y,
        gamma,
        &cost_sources,
        &cost_targets,
        &cost_values,
        layers,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
fn filter_strength_edges_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    lam: f64,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    let result = core_filter_strength_edges_geometric(&x, &y, lam, &sources, &targets, &weights);
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
fn filter_strength_edges_negative_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    lam: f64,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    let result = core_filter_strength_edges_negative_binomial(
        &x, &y, lam, layers, &sources, &targets, &weights,
    );
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn filter_strength_degree_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    w: Vec<f64>,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    let result =
        core_filter_strength_degree_geometric(&x, &y, &z, &w, &sources, &targets, &weights);
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn filter_strength_degree_negative_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    w: Vec<f64>,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    let result = core_filter_strength_degree_negative_binomial(
        &x, &y, &z, &w, layers, &sources, &targets, &weights,
    );
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
fn filter_degree_events_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    positive_weight_rate: f64,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    let result = core_filter_degree_events_geometric(
        &x,
        &y,
        positive_weight_rate,
        &sources,
        &targets,
        &weights,
    );
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
fn filter_degree_events_negative_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    positive_weight_rate: f64,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    let result = core_filter_degree_events_negative_binomial(
        &x,
        &y,
        positive_weight_rate,
        layers,
        &sources,
        &targets,
        &weights,
    );
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, lam, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_strength_edges_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    lam: f64,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    let result = core_absent_strength_edges_geometric(
        &x,
        &y,
        lam,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, lam, layers, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_strength_edges_negative_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    lam: f64,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    let result = core_absent_strength_edges_negative_binomial(
        &x,
        &y,
        lam,
        layers,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, z, w, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_strength_degree_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    w: Vec<f64>,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    let result = core_absent_strength_degree_geometric(
        &x,
        &y,
        &z,
        &w,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, z, w, layers, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_strength_degree_negative_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    w: Vec<f64>,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    let result = core_absent_strength_degree_negative_binomial(
        &x,
        &y,
        &z,
        &w,
        layers,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, positive_weight_rate, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_degree_events_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    positive_weight_rate: f64,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    let result = core_absent_degree_events_geometric(
        &x,
        &y,
        positive_weight_rate,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, positive_weight_rate, layers, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_degree_events_negative_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    positive_weight_rate: f64,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    let result = core_absent_degree_events_negative_binomial(
        &x,
        &y,
        positive_weight_rate,
        layers,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
fn filter_strength_negative_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    let result =
        core_filter_strength_negative_binomial(&x, &y, layers, &sources, &targets, &weights);
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, layers, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
fn absent_strength_negative_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> PyResult<AbsentFilter> {
    let result = core_absent_strength_negative_binomial(
        &x,
        &y,
        layers,
        &sources,
        &targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
    );
    Ok((
        result.sources,
        result.targets,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
fn benjamini_hochberg(pvalues: Vec<f64>, alpha: f64) -> Vec<bool> {
    core_benjamini_hochberg(&pvalues, alpha)
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
fn _menobis(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(rust_core_version, module)?)?;
    module.add_function(wrap_pyfunction!(directed_strengths, module)?)?;
    module.add_function(wrap_pyfunction!(directed_degrees, module)?)?;
    module.add_function(wrap_pyfunction!(compute_all_node_stats, module)?)?;
    module.add_function(wrap_pyfunction!(weight_distribution, module)?)?;
    module.add_function(wrap_pyfunction!(fit_masked_degree_bernoulli, module)?)?;
    module.add_function(wrap_pyfunction!(
        fit_masked_strength_degree_poisson,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(fit_masked_strength_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(fit_strength_cost_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(
        fit_strength_cost_poisson_coordinates,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        fit_strength_cost_binomial_coordinates,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(fit_strength_cost_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(fit_strength_cost_w_coordinates, module)?)?;
    module.add_function(wrap_pyfunction!(fit_strength_cost_w_lbfgs, module)?)?;
    module.add_function(wrap_pyfunction!(fit_degree_bernoulli, module)?)?;
    module.add_function(wrap_pyfunction!(fit_strength_edges_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(fit_strength_degree_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(fit_strength_edges_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(fit_strength_degree_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(fit_weighted_factors, module)?)?;
    module.add_function(wrap_pyfunction!(fit_strength_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(fit_strength_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(fit_strength_negative_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(fit_strength_cost_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(
        fit_strength_cost_negative_binomial,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(fit_strength_edges_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(
        fit_strength_edges_negative_binomial,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(fit_strength_degree_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(
        fit_strength_degree_negative_binomial,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        fit_strength_poisson_no_self_loops,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sample_strength_stub_matching, module)?)?;
    module.add_function(wrap_pyfunction!(sample_custom_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(sample_custom_multinomial, module)?)?;
    module.add_function(wrap_pyfunction!(
        sample_strength_poisson_multinomial,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sample_strength_edges_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(sample_strength_cost_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(sample_strength_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(sample_degree_events_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(sample_strength_degree_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(sample_strength_multinomial, module)?)?;
    module.add_function(wrap_pyfunction!(sample_strength_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(sample_strength_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(sample_strength_cost_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(sample_strength_cost_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(
        sample_strength_cost_negative_binomial,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sample_strength_edges_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(sample_strength_edges_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(
        sample_strength_edges_negative_binomial,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sample_strength_degree_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(sample_strength_degree_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(
        sample_strength_degree_negative_binomial,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sample_degree_events_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(sample_degree_events_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(
        sample_degree_events_negative_binomial,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sample_strength_negative_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(fit_strength_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(fit_degree_events_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(
        fit_degree_events_negative_binomial,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(fit_masked_strength_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(fit_partial_strength_poisson_full, module)?)?;
    module.add_function(wrap_pyfunction!(fit_partial_degree_poisson_full, module)?)?;
    module.add_function(wrap_pyfunction!(
        fit_partial_strength_degree_poisson_full,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        fit_partial_strength_edges_poisson_full,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        fit_partial_strength_cost_poisson_full,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        fit_partial_strength_cost_poisson_coordinates_full,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        fit_partial_strength_cost_binomial_coordinates_full,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        fit_partial_strength_cost_w_coordinates_full,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(filter_strength_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(absent_strength_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(filter_custom_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(absent_custom_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(filter_strength_edges_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(absent_strength_edges_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(filter_strength_cost_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(absent_strength_cost_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(filter_strength_degree_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(absent_strength_degree_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(filter_degree_events_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(absent_degree_events_poisson, module)?)?;
    module.add_function(wrap_pyfunction!(filter_strength_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(absent_strength_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(filter_strength_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(absent_strength_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(filter_strength_cost_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(absent_strength_cost_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(filter_strength_edges_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(absent_strength_edges_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(filter_strength_degree_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(absent_strength_degree_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(filter_degree_events_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(absent_degree_events_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(filter_strength_cost_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(absent_strength_cost_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(
        filter_strength_cost_negative_binomial,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        absent_strength_cost_negative_binomial,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(filter_strength_edges_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(absent_strength_edges_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(
        filter_strength_edges_negative_binomial,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        absent_strength_edges_negative_binomial,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(filter_strength_degree_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(absent_strength_degree_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(
        filter_strength_degree_negative_binomial,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        absent_strength_degree_negative_binomial,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(filter_degree_events_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(absent_degree_events_geometric, module)?)?;
    module.add_function(wrap_pyfunction!(
        filter_degree_events_negative_binomial,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        absent_degree_events_negative_binomial,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(filter_strength_negative_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(absent_strength_negative_binomial, module)?)?;
    module.add_function(wrap_pyfunction!(benjamini_hochberg, module)?)?;
    module.add_function(wrap_pyfunction!(clustering_coefficients, module)?)?;
    module.add_function(wrap_pyfunction!(weighted_clustering_coefficients, module)?)?;
    Ok(())
}
