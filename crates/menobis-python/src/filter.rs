use super::*;

#[pyfunction]
pub(crate) fn filter_strength_poisson(
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
pub(crate) fn absent_strength_poisson(
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
pub(crate) fn filter_custom_poisson(
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
pub(crate) fn absent_custom_poisson(
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
pub(crate) fn filter_strength_edges_poisson(
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
pub(crate) fn absent_strength_edges_poisson(
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
pub(crate) fn filter_strength_cost_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    let result = core_filter_strength_cost_poisson(
        &x, &y, gamma, &coord_x, &coord_y, &sources, &targets, &weights,
    );
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, gamma, coord_x, coord_y, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn absent_strength_cost_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
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
        &coord_x,
        &coord_y,
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
pub(crate) fn filter_strength_degree_poisson(
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
pub(crate) fn absent_strength_degree_poisson(
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
pub(crate) fn filter_degree_events_poisson(
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
pub(crate) fn absent_degree_events_poisson(
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
pub(crate) fn filter_strength_geometric(
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
pub(crate) fn absent_strength_geometric(
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
pub(crate) fn filter_strength_binomial(
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
pub(crate) fn absent_strength_binomial(
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
pub(crate) fn filter_strength_cost_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    let result = core_filter_strength_cost_binomial(
        &x, &y, gamma, &coord_x, &coord_y, layers, &sources, &targets, &weights,
    );
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, gamma, coord_x, coord_y, layers, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn absent_strength_cost_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
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
        &coord_x,
        &coord_y,
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
pub(crate) fn filter_strength_edges_binomial(
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
pub(crate) fn absent_strength_edges_binomial(
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
pub(crate) fn filter_strength_degree_binomial(
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
pub(crate) fn absent_strength_degree_binomial(
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
pub(crate) fn filter_degree_events_binomial(
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
pub(crate) fn absent_degree_events_binomial(
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
pub(crate) fn filter_strength_cost_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    let result = core_filter_strength_cost_geometric(
        &x, &y, gamma, &coord_x, &coord_y, &sources, &targets, &weights,
    );
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, gamma, coord_x, coord_y, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn absent_strength_cost_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
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
        &coord_x,
        &coord_y,
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
pub(crate) fn filter_strength_cost_negative_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
    layers: u32,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<u64>,
) -> PyResult<ObservedFilter> {
    let result = core_filter_strength_cost_negative_binomial(
        &x, &y, gamma, &coord_x, &coord_y, layers, &sources, &targets, &weights,
    );
    Ok((
        result.upper_pvalues,
        result.lower_pvalues,
        result.expected,
        result.occupation,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, gamma, coord_x, coord_y, layers, sources, targets, self_loops, alpha_lower, min_occupation, min_expected, max_absent=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn absent_strength_cost_negative_binomial(
    x: Vec<f64>,
    y: Vec<f64>,
    gamma: f64,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
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
        &coord_x,
        &coord_y,
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
pub(crate) fn filter_strength_edges_geometric(
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
pub(crate) fn filter_strength_edges_negative_binomial(
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
pub(crate) fn filter_strength_degree_geometric(
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
pub(crate) fn filter_strength_degree_negative_binomial(
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
pub(crate) fn filter_degree_events_geometric(
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
pub(crate) fn filter_degree_events_negative_binomial(
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
pub(crate) fn absent_strength_edges_geometric(
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
pub(crate) fn absent_strength_edges_negative_binomial(
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
pub(crate) fn absent_strength_degree_geometric(
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
pub(crate) fn absent_strength_degree_negative_binomial(
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
pub(crate) fn absent_degree_events_geometric(
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
pub(crate) fn absent_degree_events_negative_binomial(
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
pub(crate) fn filter_strength_negative_binomial(
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
pub(crate) fn absent_strength_negative_binomial(
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
