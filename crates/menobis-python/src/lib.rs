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
    absent_strength_cost_binomial_coordinates as core_absent_strength_cost_binomial,
    absent_strength_cost_geometric_coordinates as core_absent_strength_cost_geometric,
    absent_strength_cost_negative_binomial_coordinates as core_absent_strength_cost_negative_binomial,
    absent_strength_cost_poisson_coordinates as core_absent_strength_cost_poisson,
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
    filter_strength_cost_binomial_coordinates as core_filter_strength_cost_binomial,
    filter_strength_cost_geometric_coordinates as core_filter_strength_cost_geometric,
    filter_strength_cost_negative_binomial_coordinates as core_filter_strength_cost_negative_binomial,
    filter_strength_cost_poisson_coordinates as core_filter_strength_cost_poisson,
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
    fit_partial_strength_binomial as core_fit_partial_strength_binomial,
    fit_partial_strength_cost_binomial_coordinates as core_fit_partial_strength_cost_binomial_coordinates,
    fit_partial_strength_cost_coordinates as core_fit_partial_strength_cost_coordinates,
    fit_partial_strength_cost_w_coordinates as core_fit_partial_strength_cost_w_coordinates,
    fit_partial_strength_degree as core_fit_partial_strength_degree,
    fit_partial_strength_degree_binomial as core_fit_partial_strength_degree_binomial,
    fit_partial_strength_edges as core_fit_partial_strength_edges,
    fit_partial_strength_edges_binomial as core_fit_partial_strength_edges_binomial,
    fit_partial_strength_edges_w as core_fit_partial_strength_edges_w,
};
use menobis_core::fitting::{
    fit_strength_cost_binomial_coordinates as core_fit_strength_cost_binomial_coordinates,
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
    sample_strength_cost_binomial_coordinates as core_sample_strength_cost_binomial_coordinates,
    sample_strength_cost_geometric_coordinates as core_sample_strength_cost_geometric_coordinates,
    sample_strength_cost_negative_binomial_coordinates as core_sample_strength_cost_negative_binomial_coordinates,
    sample_strength_cost_poisson_coordinates as core_sample_strength_cost_poisson_coordinates,
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
    sample_strength_stub_matching as core_sample_strength_stub_matching,
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

mod filter;
mod fitting;
mod generation;
mod stats;

macro_rules! add_pyfunction {
    ($module:expr, $function:path) => {
        $module.add_function(wrap_pyfunction!($function, $module)?)
    };
}

#[pymodule]
fn _menobis(module: &Bound<'_, PyModule>) -> PyResult<()> {
    add_pyfunction!(module, rust_core_version)?;
    add_pyfunction!(module, stats::directed_strengths)?;
    add_pyfunction!(module, stats::directed_degrees)?;
    add_pyfunction!(module, stats::compute_all_node_stats)?;
    add_pyfunction!(module, stats::weight_distribution)?;
    add_pyfunction!(module, fitting::fit_masked_degree_bernoulli)?;
    add_pyfunction!(module, fitting::fit_masked_strength_degree_poisson)?;
    add_pyfunction!(module, fitting::fit_masked_strength_poisson)?;
    add_pyfunction!(module, fitting::fit_strength_cost_poisson_coordinates)?;
    add_pyfunction!(module, fitting::fit_strength_cost_binomial_coordinates)?;
    add_pyfunction!(module, fitting::fit_strength_cost_w_coordinates)?;
    add_pyfunction!(module, fitting::fit_strength_cost_w_lbfgs)?;
    add_pyfunction!(module, fitting::fit_degree_bernoulli)?;
    add_pyfunction!(module, fitting::fit_strength_edges_poisson)?;
    add_pyfunction!(module, fitting::fit_strength_degree_poisson)?;
    add_pyfunction!(module, fitting::fit_strength_edges_binomial)?;
    add_pyfunction!(module, fitting::fit_strength_degree_binomial)?;
    add_pyfunction!(module, fitting::fit_weighted_factors)?;
    add_pyfunction!(module, fitting::fit_strength_poisson)?;
    add_pyfunction!(module, fitting::fit_strength_geometric)?;
    add_pyfunction!(module, fitting::fit_strength_negative_binomial)?;
    add_pyfunction!(module, fitting::fit_strength_edges_geometric)?;
    add_pyfunction!(module, fitting::fit_strength_edges_negative_binomial)?;
    add_pyfunction!(module, fitting::fit_strength_degree_geometric)?;
    add_pyfunction!(module, fitting::fit_strength_degree_negative_binomial)?;
    add_pyfunction!(module, fitting::fit_strength_poisson_no_self_loops)?;
    add_pyfunction!(module, generation::sample_strength_stub_matching)?;
    add_pyfunction!(module, generation::sample_custom_poisson)?;
    add_pyfunction!(module, generation::sample_custom_multinomial)?;
    add_pyfunction!(module, generation::sample_strength_edges_poisson)?;
    add_pyfunction!(module, generation::sample_strength_cost_poisson_coordinates)?;
    add_pyfunction!(module, generation::sample_strength_poisson)?;
    add_pyfunction!(module, generation::sample_degree_events_poisson)?;
    add_pyfunction!(module, generation::sample_strength_degree_poisson)?;
    add_pyfunction!(module, generation::sample_strength_multinomial)?;
    add_pyfunction!(module, generation::sample_strength_geometric)?;
    add_pyfunction!(module, generation::sample_strength_binomial)?;
    add_pyfunction!(
        module,
        generation::sample_strength_cost_binomial_coordinates
    )?;
    add_pyfunction!(
        module,
        generation::sample_strength_cost_geometric_coordinates
    )?;
    add_pyfunction!(
        module,
        generation::sample_strength_cost_negative_binomial_coordinates
    )?;
    add_pyfunction!(module, generation::sample_strength_edges_binomial)?;
    add_pyfunction!(module, generation::sample_strength_edges_geometric)?;
    add_pyfunction!(module, generation::sample_strength_edges_negative_binomial)?;
    add_pyfunction!(module, generation::sample_strength_degree_binomial)?;
    add_pyfunction!(module, generation::sample_strength_degree_geometric)?;
    add_pyfunction!(module, generation::sample_strength_degree_negative_binomial)?;
    add_pyfunction!(module, generation::sample_degree_events_binomial)?;
    add_pyfunction!(module, generation::sample_degree_events_geometric)?;
    add_pyfunction!(module, generation::sample_degree_events_negative_binomial)?;
    add_pyfunction!(module, generation::sample_strength_negative_binomial)?;
    add_pyfunction!(module, fitting::fit_strength_binomial)?;
    add_pyfunction!(module, fitting::fit_degree_events_geometric)?;
    add_pyfunction!(module, fitting::fit_degree_events_negative_binomial)?;
    add_pyfunction!(module, fitting::fit_masked_strength_binomial)?;
    add_pyfunction!(module, fitting::fit_partial_strength_poisson_full)?;
    add_pyfunction!(module, fitting::fit_partial_degree_poisson_full)?;
    add_pyfunction!(module, fitting::fit_partial_strength_degree_poisson_full)?;
    add_pyfunction!(module, fitting::fit_partial_strength_edges_poisson_full)?;
    add_pyfunction!(
        module,
        fitting::fit_partial_strength_cost_poisson_coordinates_full
    )?;
    add_pyfunction!(
        module,
        fitting::fit_partial_strength_cost_binomial_coordinates_full
    )?;
    add_pyfunction!(
        module,
        fitting::fit_partial_strength_cost_w_coordinates_full
    )?;
    add_pyfunction!(module, fitting::fit_partial_strength_binomial_full)?;
    add_pyfunction!(module, fitting::fit_partial_strength_edges_binomial_full)?;
    add_pyfunction!(module, fitting::fit_partial_strength_degree_binomial_full)?;
    add_pyfunction!(module, fitting::fit_partial_strength_edges_w_full)?;
    add_pyfunction!(module, filter::filter_strength_poisson)?;
    add_pyfunction!(module, filter::absent_strength_poisson)?;
    add_pyfunction!(module, filter::filter_custom_poisson)?;
    add_pyfunction!(module, filter::absent_custom_poisson)?;
    add_pyfunction!(module, filter::filter_strength_edges_poisson)?;
    add_pyfunction!(module, filter::absent_strength_edges_poisson)?;
    add_pyfunction!(module, filter::filter_strength_cost_poisson)?;
    add_pyfunction!(module, filter::absent_strength_cost_poisson)?;
    add_pyfunction!(module, filter::filter_strength_degree_poisson)?;
    add_pyfunction!(module, filter::absent_strength_degree_poisson)?;
    add_pyfunction!(module, filter::filter_degree_events_poisson)?;
    add_pyfunction!(module, filter::absent_degree_events_poisson)?;
    add_pyfunction!(module, filter::filter_strength_geometric)?;
    add_pyfunction!(module, filter::absent_strength_geometric)?;
    add_pyfunction!(module, filter::filter_strength_binomial)?;
    add_pyfunction!(module, filter::absent_strength_binomial)?;
    add_pyfunction!(module, filter::filter_strength_cost_binomial)?;
    add_pyfunction!(module, filter::absent_strength_cost_binomial)?;
    add_pyfunction!(module, filter::filter_strength_edges_binomial)?;
    add_pyfunction!(module, filter::absent_strength_edges_binomial)?;
    add_pyfunction!(module, filter::filter_strength_degree_binomial)?;
    add_pyfunction!(module, filter::absent_strength_degree_binomial)?;
    add_pyfunction!(module, filter::filter_degree_events_binomial)?;
    add_pyfunction!(module, filter::absent_degree_events_binomial)?;
    add_pyfunction!(module, filter::filter_strength_cost_geometric)?;
    add_pyfunction!(module, filter::absent_strength_cost_geometric)?;
    add_pyfunction!(module, filter::filter_strength_cost_negative_binomial)?;
    add_pyfunction!(module, filter::absent_strength_cost_negative_binomial)?;
    add_pyfunction!(module, filter::filter_strength_edges_geometric)?;
    add_pyfunction!(module, filter::absent_strength_edges_geometric)?;
    add_pyfunction!(module, filter::filter_strength_edges_negative_binomial)?;
    add_pyfunction!(module, filter::absent_strength_edges_negative_binomial)?;
    add_pyfunction!(module, filter::filter_strength_degree_geometric)?;
    add_pyfunction!(module, filter::absent_strength_degree_geometric)?;
    add_pyfunction!(module, filter::filter_strength_degree_negative_binomial)?;
    add_pyfunction!(module, filter::absent_strength_degree_negative_binomial)?;
    add_pyfunction!(module, filter::filter_degree_events_geometric)?;
    add_pyfunction!(module, filter::absent_degree_events_geometric)?;
    add_pyfunction!(module, filter::filter_degree_events_negative_binomial)?;
    add_pyfunction!(module, filter::absent_degree_events_negative_binomial)?;
    add_pyfunction!(module, filter::filter_strength_negative_binomial)?;
    add_pyfunction!(module, filter::absent_strength_negative_binomial)?;
    add_pyfunction!(module, stats::benjamini_hochberg)?;
    add_pyfunction!(module, stats::clustering_coefficients)?;
    add_pyfunction!(module, stats::weighted_clustering_coefficients)?;
    Ok(())
}
