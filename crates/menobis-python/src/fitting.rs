use super::*;

#[pyfunction]
pub(crate) fn fit_masked_strength_poisson(
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
pub(crate) fn fit_masked_degree_bernoulli(
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
pub(crate) fn fit_masked_strength_degree_poisson(
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
pub(crate) fn fit_strength_cost_poisson_coordinates(
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
pub(crate) fn fit_strength_cost_binomial_coordinates(
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
pub(crate) fn fit_strength_cost_w_coordinates(
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
pub(crate) fn fit_strength_cost_w_lbfgs(
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
pub(crate) fn fit_degree_bernoulli(
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
pub(crate) fn fit_strength_edges_poisson(
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
pub(crate) fn fit_strength_edges_binomial(
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
pub(crate) fn fit_strength_degree_poisson(
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
pub(crate) fn fit_weighted_factors(
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
pub(crate) fn fit_strength_degree_binomial(
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
pub(crate) fn fit_strength_poisson(
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

pub(crate) fn w_strength_fit_tuple(
    result: menobis_core::fitting::WStrengthFitResult,
) -> WStrengthFit {
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

pub(crate) fn w_strength_edges_fit_tuple(
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

pub(crate) fn w_strength_degree_fit_tuple(
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
pub(crate) fn fit_strength_geometric(
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
pub(crate) fn fit_strength_negative_binomial(
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
pub(crate) fn fit_strength_edges_geometric(
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
pub(crate) fn fit_strength_edges_negative_binomial(
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
pub(crate) fn fit_strength_degree_geometric(
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
pub(crate) fn fit_strength_degree_negative_binomial(
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
pub(crate) fn fit_strength_poisson_no_self_loops(
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
pub(crate) fn fit_strength_binomial(
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
pub(crate) fn fit_degree_events_geometric(
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
pub(crate) fn fit_degree_events_negative_binomial(
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
pub(crate) fn fit_masked_strength_binomial(
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
pub(crate) fn fit_partial_strength_poisson_full(
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
pub(crate) fn fit_partial_degree_poisson_full(
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
pub(crate) fn fit_partial_strength_degree_poisson_full(
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
pub(crate) fn fit_partial_strength_edges_poisson_full(
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
pub(crate) fn fit_partial_strength_cost_poisson_coordinates_full(
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
pub(crate) fn fit_partial_strength_cost_binomial_coordinates_full(
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
pub(crate) fn fit_partial_strength_cost_w_coordinates_full(
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
