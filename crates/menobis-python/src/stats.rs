use super::*;

#[pyfunction]
pub(crate) fn directed_strengths(
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
pub(crate) fn directed_degrees(
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
pub(crate) fn compute_all_node_stats(
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
pub(crate) fn weight_distribution(
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
pub(crate) fn benjamini_hochberg(pvalues: Vec<f64>, alpha: f64) -> Vec<bool> {
    core_benjamini_hochberg(&pvalues, alpha)
}

#[pyfunction]
pub(crate) fn clustering_coefficients(
    node_count: usize,
    sources: Vec<usize>,
    targets: Vec<usize>,
    weights: Vec<u64>,
) -> PyResult<Vec<f64>> {
    let edges = build_edges(node_count, sources, targets, weights)?;
    Ok(core_clustering(node_count, &edges))
}

#[pyfunction]
pub(crate) fn weighted_clustering_coefficients(
    node_count: usize,
    sources: Vec<usize>,
    targets: Vec<usize>,
    weights: Vec<u64>,
) -> PyResult<Vec<f64>> {
    let edges = build_edges(node_count, sources, targets, weights)?;
    Ok(core_weighted_clustering(node_count, &edges))
}
