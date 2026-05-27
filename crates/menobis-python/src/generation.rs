use super::*;

#[pyfunction]
pub(crate) fn sample_strength_stub_matching(
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
pub(crate) fn sample_custom_poisson(
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
pub(crate) fn sample_custom_multinomial(
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
pub(crate) fn sample_strength_edges_poisson(
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
pub(crate) fn sample_strength_cost_poisson_coordinates(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
    self_loops: bool,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    if coord_x.len() != x.len() || coord_y.len() != x.len() {
        return Err(PyValueError::new_err(
            "coord_x and coord_y must match x/y length",
        ));
    }
    let sample = core_sample_strength_cost_poisson_coordinates(
        &x, &y, gamma, &coord_x, &coord_y, self_loops, seed,
    );
    Ok((sample.sources, sample.targets, sample.weights))
}

#[pyfunction]
pub(crate) fn sample_strength_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_strength_poisson(&x, &y, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
pub(crate) fn sample_strength_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_strength_geometric(&x, &y, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}

#[pyfunction]
pub(crate) fn sample_strength_binomial(
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
pub(crate) fn sample_strength_cost_binomial_coordinates(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if x.len() != y.len() || coord_x.len() != x.len() || coord_y.len() != x.len() {
        return Err(PyValueError::new_err(
            "x, y, coord_x, and coord_y must have the same length",
        ));
    }
    let edges = core_sample_strength_cost_binomial_coordinates(
        &x, &y, gamma, &coord_x, &coord_y, layers, self_loops, seed,
    );
    Ok((edges.sources, edges.targets, edges.weights))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_strength_cost_geometric_coordinates(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
    self_loops: bool,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if x.len() != y.len() || coord_x.len() != x.len() || coord_y.len() != x.len() {
        return Err(PyValueError::new_err(
            "x, y, coord_x, and coord_y must have the same length",
        ));
    }
    let edges = core_sample_strength_cost_geometric_coordinates(
        &x, &y, gamma, &coord_x, &coord_y, self_loops, seed,
    );
    Ok((edges.sources, edges.targets, edges.weights))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_strength_cost_negative_binomial_coordinates(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if x.len() != y.len() || coord_x.len() != x.len() || coord_y.len() != x.len() {
        return Err(PyValueError::new_err(
            "x, y, coord_x, and coord_y must have the same length",
        ));
    }
    let edges = core_sample_strength_cost_negative_binomial_coordinates(
        &x, &y, gamma, &coord_x, &coord_y, layers, self_loops, seed,
    );
    Ok((edges.sources, edges.targets, edges.weights))
}

#[pyfunction]
pub(crate) fn sample_strength_edges_binomial(
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
pub(crate) fn sample_strength_degree_binomial(
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
pub(crate) fn sample_degree_events_binomial(
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
pub(crate) fn sample_strength_edges_geometric(
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
pub(crate) fn sample_strength_edges_negative_binomial(
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
pub(crate) fn sample_strength_degree_geometric(
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
pub(crate) fn sample_strength_degree_negative_binomial(
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
pub(crate) fn sample_degree_events_geometric(
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
pub(crate) fn sample_degree_events_negative_binomial(
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
pub(crate) fn sample_strength_negative_binomial(
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
pub(crate) fn sample_degree_events_poisson(
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
pub(crate) fn sample_strength_degree_poisson(
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
pub(crate) fn sample_strength_multinomial(
    x: Vec<f64>,
    y: Vec<f64>,
    total_events: u64,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_strength_multinomial(&x, &y, total_events, self_loops, seed);
    (edges.sources, edges.targets, edges.weights)
}
