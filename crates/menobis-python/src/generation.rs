use rand::rngs::StdRng;
use rand::SeedableRng;

use super::*;

/// Return type of `sample_fixed_strength_with_cost`: sampled network plus
/// gamma-fit diagnostics.
type StrengthCostSample = (
    Vec<u64>, // sources
    Vec<u64>, // targets
    Vec<u64>, // occ_nums
    f64,      // gamma
    f64,      // expected cost estimate
    f64,      // standard error
    bool,     // converged
    f64,      // observed cost
    f64,      // residual
    u64,      // proposals
    u64,      // accepted
    usize,    // iterations
);

/// Exact ME microcanonical sampler with fixed (E,T).
///
/// Draws an exact sample from the ME microcanonical distribution with
/// fixed occupied-pair count `E` and fixed total occupation `T`.
/// Pair indices are computed on the fly from `node_count` + `self_loops`;
/// no pair list is materialised.
#[pyfunction]
pub(crate) fn sample_me_fixed_et(
    node_count: usize,
    self_loops: bool,
    residual_edges: usize,
    residual_total: u64,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    match core_sample_me_fixed_et(node_count, self_loops, residual_edges, residual_total, seed) {
        Ok(sample) => Ok((sample.sources, sample.targets, sample.occ_nums)),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

/// Exact ME microcanonical sampler with fixed (E,T) on an explicit
/// admissible-pair set (after masks/fixed-pair subtraction).
#[pyfunction]
pub(crate) fn sample_me_fixed_et_explicit(
    admissible_sources: Vec<u64>,
    admissible_targets: Vec<u64>,
    residual_edges: usize,
    residual_total: u64,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if admissible_sources.len() != admissible_targets.len() {
        return Err(PyValueError::new_err(
            "admissible_sources and admissible_targets must have same length",
        ));
    }
    match core_sample_me_fixed_et_explicit(
        &admissible_sources,
        &admissible_targets,
        residual_edges,
        residual_total,
        seed,
    ) {
        Ok(sample) => Ok((sample.sources, sample.targets, sample.occ_nums)),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

/// Exact B microcanonical sampler with fixed (E,T) and M layers.
#[pyfunction]
pub(crate) fn sample_b_fixed_et(
    node_count: usize,
    self_loops: bool,
    layers: u32,
    residual_edges: usize,
    residual_total: u64,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    match core_sample_b_fixed_et(
        node_count,
        self_loops,
        layers as u64,
        residual_edges,
        residual_total,
        seed,
    ) {
        Ok(sample) => Ok((sample.sources, sample.targets, sample.occ_nums)),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

/// Exact B microcanonical sampler with explicit pair arrays.
#[pyfunction]
pub(crate) fn sample_b_fixed_et_explicit(
    admissible_sources: Vec<u64>,
    admissible_targets: Vec<u64>,
    layers: u32,
    residual_edges: usize,
    residual_total: u64,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if admissible_sources.len() != admissible_targets.len() {
        return Err(PyValueError::new_err(
            "admissible_sources and admissible_targets must have same length",
        ));
    }
    match core_sample_b_fixed_et_explicit(
        &admissible_sources,
        &admissible_targets,
        layers as u64,
        residual_edges,
        residual_total,
        seed,
    ) {
        Ok(sample) => Ok((sample.sources, sample.targets, sample.occ_nums)),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

/// Exact W microcanonical sampler with fixed (E,T) and M layers.
#[pyfunction]
pub(crate) fn sample_w_fixed_et(
    node_count: usize,
    self_loops: bool,
    layers: u32,
    residual_edges: usize,
    residual_total: u64,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    match core_sample_w_fixed_et(
        node_count,
        self_loops,
        layers as u64,
        residual_edges,
        residual_total,
        seed,
    ) {
        Ok(sample) => Ok((sample.sources, sample.targets, sample.occ_nums)),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

/// Exact W microcanonical sampler with explicit pair arrays.
#[pyfunction]
pub(crate) fn sample_w_fixed_et_explicit(
    admissible_sources: Vec<u64>,
    admissible_targets: Vec<u64>,
    layers: u32,
    residual_edges: usize,
    residual_total: u64,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if admissible_sources.len() != admissible_targets.len() {
        return Err(PyValueError::new_err(
            "admissible_sources and admissible_targets must have same length",
        ));
    }
    match core_sample_w_fixed_et_explicit(
        &admissible_sources,
        &admissible_targets,
        layers as u64,
        residual_edges,
        residual_total,
        seed,
    ) {
        Ok(sample) => Ok((sample.sources, sample.targets, sample.occ_nums)),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

/// Sample the symmetric EDGES_EVENTS model.
#[pyfunction]
pub(crate) fn sample_edges_events(
    node_count: usize,
    q: f64,
    occupation: f64,
    family: &str,
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    let occ_family = match family {
        "poisson" => menobis_core::model::family::OccupationFamily::ME,
        "binomial" => menobis_core::model::family::OccupationFamily::B { layers },
        "geometric" => menobis_core::model::family::OccupationFamily::W { layers: 1 },
        "negative_binomial" => menobis_core::model::family::OccupationFamily::W { layers },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown occupation family: {other}"
            )))
        }
    };
    let sample = core_sample_edges_events(node_count, q, occupation, occ_family, self_loops, seed);
    Ok((sample.sources, sample.targets, sample.occ_nums))
}

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
    match core_sample_strength_stub_matching(&strength_out, &strength_in, seed) {
        Ok(sample) => Ok((sample.sources, sample.targets, sample.occ_nums)),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
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
    Ok((sample.sources, sample.targets, sample.occ_nums))
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
    Ok((sample.sources, sample.targets, sample.occ_nums))
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
    Ok((sample.sources, sample.targets, sample.occ_nums))
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
    Ok((sample.sources, sample.targets, sample.occ_nums))
}

#[pyfunction]
pub(crate) fn sample_strength_poisson(
    x: Vec<f64>,
    y: Vec<f64>,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_strength_poisson(&x, &y, self_loops, seed);
    (edges.sources, edges.targets, edges.occ_nums)
}

#[pyfunction]
pub(crate) fn sample_strength_geometric(
    x: Vec<f64>,
    y: Vec<f64>,
    self_loops: bool,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let edges = core_sample_strength_geometric(&x, &y, self_loops, seed);
    (edges.sources, edges.targets, edges.occ_nums)
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
    (edges.sources, edges.targets, edges.occ_nums)
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
    Ok((edges.sources, edges.targets, edges.occ_nums))
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
    Ok((edges.sources, edges.targets, edges.occ_nums))
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
    Ok((edges.sources, edges.targets, edges.occ_nums))
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
    (edges.sources, edges.targets, edges.occ_nums)
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
    (edges.sources, edges.targets, edges.occ_nums)
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
    (edges.sources, edges.targets, edges.occ_nums)
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
    (edges.sources, edges.targets, edges.occ_nums)
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
    (edges.sources, edges.targets, edges.occ_nums)
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
    (edges.sources, edges.targets, edges.occ_nums)
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
    (edges.sources, edges.targets, edges.occ_nums)
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
    (edges.sources, edges.targets, edges.occ_nums)
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
    (edges.sources, edges.targets, edges.occ_nums)
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
    (edges.sources, edges.targets, edges.occ_nums)
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
    Ok((sample.sources, sample.targets, sample.occ_nums))
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
    Ok((sample.sources, sample.targets, sample.occ_nums))
}

/// Exact microcanonical sampler with fixed out-degree, in-degree, and total events.
///
/// Supports ME, B, and W families.  Uses a directed double-edge-switch MCMC
/// for the support and reuses the existing fixed-(E,T) occupation allocators.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_degree_events_fixed_kt(
    family: &str,
    degree_out: Vec<u32>,
    degree_in: Vec<u32>,
    total_events: u64,
    layers: u32,
    burn_in_sweeps: usize,
    sweeps_per_sample: usize,
    seed: u64,
    self_loops: bool,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    use menobis_core::model::family::OccupationFamily;

    let config = CoreFixedKTConfig {
        mcmc: CoreFixedDegreeMcmcConfig {
            burn_in_sweeps,
            sweeps_per_sample,
            proposals_per_sweep: None,
            seed,
        },
        self_loops,
        admissible_pairs: None,
    };
    let result = match family {
        "ME" => core_sample_fixed_kt(
            OccupationFamily::ME,
            &degree_out,
            &degree_in,
            total_events,
            &config,
        ),
        "B" => core_sample_fixed_kt(
            OccupationFamily::B { layers },
            &degree_out,
            &degree_in,
            total_events,
            &config,
        ),
        "W" => core_sample_fixed_kt(
            OccupationFamily::W { layers },
            &degree_out,
            &degree_in,
            total_events,
            &config,
        ),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown family: {other}. Use ME, B, or W"
            )))
        }
    };
    match result {
        Ok(sample) => Ok((sample.sources, sample.targets, sample.occ_nums)),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
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
    (edges.sources, edges.targets, edges.occ_nums)
}

/// Sample from the microcanonical fixed-strength ensemble.
///
/// Routes to the ME direct stub-matching backend or the generic 4-cycle
/// MCMC backend depending on the problem configuration.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_fixed_strength(
    family: &str,
    strength_out: Vec<u64>,
    strength_in: Vec<u64>,
    self_loops: bool,
    fixed_sources: Vec<u64>,
    fixed_targets: Vec<u64>,
    fixed_occnums: Vec<u64>,
    layers: u32,
    burn_in_sweeps: usize,
    sweeps_per_sample: usize,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    use menobis_core::generation::microcanonical::mcmc::McmcConfig;
    use menobis_core::generation::microcanonical::occupation_mcmc::domain::PairDomain;
    use menobis_core::generation::microcanonical::occupation_mcmc::problem::FixedStrengthProblem;
    use menobis_core::model::family::OccupationFamily;

    let family_enum = match family {
        "ME" => OccupationFamily::ME,
        "B" => OccupationFamily::B { layers },
        "W" => OccupationFamily::W { layers },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown family: {other}. Use ME, B, or W"
            )))
        }
    };

    let n = strength_out.len();
    let domain = PairDomain::Complete {
        node_count: n,
        self_loops,
    };

    let fixed_pairs: Vec<_> = fixed_sources
        .iter()
        .zip(fixed_targets.iter())
        .zip(fixed_occnums.iter())
        .map(|((&s, &t), &o)| (s, t, o))
        .collect();
    let has_fixed = !fixed_pairs.is_empty();

    let problem = match FixedStrengthProblem::new(
        family_enum,
        strength_out,
        strength_in,
        domain,
        fixed_pairs,
    ) {
        Ok(p) => p,
        Err(e) => return Err(PyValueError::new_err(e.to_string())),
    };

    let residual = match problem.into_residual() {
        Ok(r) => r,
        Err(e) => return Err(PyValueError::new_err(e.to_string())),
    };

    let config = McmcConfig::new(burn_in_sweeps, sweeps_per_sample, seed);

    match core_sample_fixed_strength(residual, config, has_fixed) {
        Ok((mut network, _backend)) => {
            // Merge fixed pairs back into the sampled residual network.
            if has_fixed {
                let mut all: Vec<(u64, u64, u64)> = fixed_sources
                    .iter()
                    .zip(fixed_targets.iter())
                    .zip(fixed_occnums.iter())
                    .map(|((&s, &t), &o)| (s, t, o))
                    .collect();
                all.extend(
                    network
                        .sources
                        .iter()
                        .zip(network.targets.iter())
                        .zip(network.occ_nums.iter())
                        .map(|((&s, &t), &o)| (s, t, o)),
                );
                all.sort_unstable();
                network.sources = all.iter().map(|&(s, _, _)| s).collect();
                network.targets = all.iter().map(|&(_, t, _)| t).collect();
                network.occ_nums = all.iter().map(|&(_, _, o)| o).collect();
            }
            Ok((network.sources, network.targets, network.occ_nums))
        }
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

/// Sample from the microcanonical fixed-strength ensemble with expected cost.
///
/// Fits gamma via stochastic bisection, then draws one sample at the fitted gamma.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_fixed_strength_with_cost(
    family: &str,
    strength_out: Vec<u64>,
    strength_in: Vec<u64>,
    coord_x: Vec<f64>,
    coord_y: Vec<f64>,
    observed_total_cost: f64,
    self_loops: bool,
    fixed_sources: Vec<u64>,
    fixed_targets: Vec<u64>,
    fixed_occnums: Vec<u64>,
    layers: u32,
    warm_start_sweeps: usize,
    adaptation_sweeps: usize,
    estimation_sweeps: usize,
    samples_per_iteration: usize,
    max_iterations: usize,
    absolute_cost_tolerance: f64,
    relative_cost_tolerance: f64,
    confidence_multiplier: f64,
    batch_count: usize,
    burn_in_sweeps: usize,
    sweeps_per_sample: usize,
    seed: u64,
) -> PyResult<StrengthCostSample> {
    use menobis_core::generation::microcanonical::mcmc::McmcConfig;
    use menobis_core::generation::microcanonical::occupation_mcmc::cost_fit::{
        fit_gamma, FixedStrengthCostFitConfig,
    };
    use menobis_core::generation::microcanonical::occupation_mcmc::domain::PairDomain;
    use menobis_core::generation::microcanonical::occupation_mcmc::problem::FixedStrengthProblem;
    use menobis_core::model::family::OccupationFamily;
    use menobis_core::pairs::EuclideanCostProvider;

    let family_enum = match family {
        "ME" => OccupationFamily::ME,
        "B" => OccupationFamily::B { layers },
        "W" => OccupationFamily::W { layers },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown family: {other}. Use ME, B, or W"
            )))
        }
    };

    let n = strength_out.len();
    let domain = PairDomain::Complete {
        node_count: n,
        self_loops,
    };

    let fixed_pairs: Vec<_> = fixed_sources
        .iter()
        .zip(fixed_targets.iter())
        .zip(fixed_occnums.iter())
        .map(|((&s, &t), &o)| (s, t, o))
        .collect();
    let has_fixed = !fixed_pairs.is_empty();

    let problem = match FixedStrengthProblem::new(
        family_enum,
        strength_out,
        strength_in,
        domain,
        fixed_pairs,
    ) {
        Ok(p) => p,
        Err(e) => return Err(PyValueError::new_err(e.to_string())),
    };

    let residual = match problem.into_residual() {
        Ok(r) => r,
        Err(e) => return Err(PyValueError::new_err(e.to_string())),
    };

    // Build cost provider from coordinates.
    let costs = if coord_x.len() != n || coord_y.len() != n {
        return Err(PyValueError::new_err(
            "coord_x and coord_y must match node count",
        ));
    } else {
        EuclideanCostProvider {
            x: &coord_x,
            y: &coord_y,
        }
    };

    let mcmc_config = McmcConfig::new(burn_in_sweeps, sweeps_per_sample, seed);

    // Build chain with cost provider.
    let core_result = menobis_core::generation::microcanonical::occupation_mcmc::chain::
        sample_fixed_strength_with_cost(residual, &costs, mcmc_config, has_fixed);
    let (mut chain, _backend) = match core_result {
        Ok(r) => r,
        Err(e) => return Err(PyValueError::new_err(e.to_string())),
    };

    // Compute fixed-pair cost.
    let fixed_cost = if has_fixed {
        match menobis_core::generation::microcanonical::occupation_mcmc::cost::fixed_pairs_cost(
            &fixed_sources
                .iter()
                .zip(fixed_targets.iter())
                .zip(fixed_occnums.iter())
                .map(|((&s, &t), &o)| (s, t, o))
                .collect::<Vec<_>>(),
            &costs,
        ) {
            Ok(c) => c,
            Err(e) => return Err(PyValueError::new_err(e.to_string())),
        }
    } else {
        0.0
    };

    // Fit config.
    let fit_config = FixedStrengthCostFitConfig {
        warm_start_sweeps,
        adaptation_sweeps,
        estimation_sweeps,
        samples_per_iteration,
        max_iterations,
        absolute_cost_tolerance,
        relative_cost_tolerance,
        confidence_multiplier,
        batch_count,
        ..FixedStrengthCostFitConfig::default()
    };

    let mut rng = StdRng::seed_from_u64(seed);

    // Fit gamma.
    let fit_result = match fit_gamma(
        &mut chain,
        &mut rng,
        &costs,
        observed_total_cost,
        fixed_cost,
        &fit_config,
    ) {
        Ok(r) => r,
        Err(e) => {
            // Even if not converged, we may have a best result.
            // Return what we have with converged=false.
            return Err(PyValueError::new_err(e.to_string()));
        }
    };

    // Set the fitted gamma and burn in again before final sample.
    {
        let mut target = menobis_core::generation::microcanonical::occupation_mcmc::
            target::StrengthTarget::with_costs(family_enum, &costs);
        target.set_gamma(fit_result.gamma);
        chain.set_target(target);
    }
    chain.burn_in(&mut rng);

    // Merge fixed pairs back into the sampled residual network.
    let mut network = chain.sample(&mut rng);
    if has_fixed {
        let fixed_src: Vec<u64> = fixed_sources;
        let fixed_tgt: Vec<u64> = fixed_targets;
        let fixed_occ: Vec<u64> = fixed_occnums;
        let mut all: Vec<(u64, u64, u64)> = fixed_src
            .iter()
            .zip(fixed_tgt.iter())
            .zip(fixed_occ.iter())
            .map(|((&s, &t), &o)| (s, t, o))
            .collect();
        all.extend(
            network
                .sources
                .iter()
                .zip(network.targets.iter())
                .zip(network.occ_nums.iter())
                .map(|((&s, &t), &o)| (s, t, o)),
        );
        all.sort_unstable();
        network.sources = all.iter().map(|&(s, _, _)| s).collect();
        network.targets = all.iter().map(|&(_, t, _)| t).collect();
        network.occ_nums = all.iter().map(|&(_, _, o)| o).collect();
    }

    Ok((
        network.sources,
        network.targets,
        network.occ_nums,
        fit_result.gamma,
        fit_result.expected_cost_estimate,
        fit_result.expected_cost_standard_error,
        fit_result.converged,
        fit_result.observed_cost,
        fit_result.residual,
        fit_result.mcmc_proposals,
        fit_result.mcmc_accepted,
        fit_result.iterations,
    ))
}

/// Unified model sampling across ensembles, families, and constraints.
///
/// One entry point mirrors the Python `sample_model` surface.  Rust
/// validates the constraint-required parameters and dispatches:
///
/// - `microcanonical`: EDGES_EVENTS → fixed-(E,T), DEGREE_EVENTS →
///   fixed-(k,T), STRENGTH → fixed-strength MCMC.
/// - `grand_canonical`: all six constraints via the grand-canonical
///   router (fitted multipliers x, y, ...).
/// - `canonical`: ME STRENGTH multinomial with fixed total events.
///
/// Fixed-pair residualization and microcanonical STRENGTH_COST gamma
/// fitting remain in the Python layer.
#[pyfunction]
#[pyo3(signature = (
    ensemble, family, constraint,
    node_count=None, self_loops=true, layers=1, seed=0,
    burn_in_sweeps=50, sweeps_per_sample=10,
    x=None, y=None, z=None, w=None, lam=None, q=None, occupation=None,
    gamma=None, coord_x=None, coord_y=None,
    total_events=None,
    strength_out=None, strength_in=None, degree_out=None, degree_in=None,
    residual_edges=None, residual_total=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_model(
    ensemble: &str,
    family: &str,
    constraint: &str,
    node_count: Option<usize>,
    self_loops: bool,
    layers: u32,
    seed: u64,
    burn_in_sweeps: usize,
    sweeps_per_sample: usize,
    x: Option<Vec<f64>>,
    y: Option<Vec<f64>>,
    z: Option<Vec<f64>>,
    w: Option<Vec<f64>>,
    lam: Option<f64>,
    q: Option<f64>,
    occupation: Option<f64>,
    gamma: Option<f64>,
    coord_x: Option<Vec<f64>>,
    coord_y: Option<Vec<f64>>,
    total_events: Option<u64>,
    strength_out: Option<Vec<u64>>,
    strength_in: Option<Vec<u64>>,
    degree_out: Option<Vec<u32>>,
    degree_in: Option<Vec<u32>>,
    residual_edges: Option<usize>,
    residual_total: Option<u64>,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    match ensemble {
        "microcanonical" => sample_model_microcanonical(
            family,
            constraint,
            node_count,
            self_loops,
            layers,
            seed,
            burn_in_sweeps,
            sweeps_per_sample,
            strength_out,
            strength_in,
            degree_out,
            degree_in,
            residual_edges,
            residual_total,
        ),
        "grandcanonical" => sample_model_grandcanonical(
            family, constraint, self_loops, layers, seed, x, y, z, w, lam, q, occupation, gamma,
            coord_x, coord_y, node_count,
        ),
        "canonical" => {
            sample_model_canonical(family, constraint, self_loops, seed, x, y, total_events)
        }
        other => Err(PyValueError::new_err(format!("invalid ensemble: {other}"))),
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_model_microcanonical(
    family: &str,
    constraint: &str,
    node_count: Option<usize>,
    self_loops: bool,
    layers: u32,
    seed: u64,
    burn_in_sweeps: usize,
    sweeps_per_sample: usize,
    strength_out: Option<Vec<u64>>,
    strength_in: Option<Vec<u64>>,
    degree_out: Option<Vec<u32>>,
    degree_in: Option<Vec<u32>>,
    residual_edges: Option<usize>,
    residual_total: Option<u64>,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    use menobis_core::generation::microcanonical::route::{
        sample_microcanonical as route_sample, MicrocanonicalConfig,
    };
    use menobis_core::model::family::OccupationFamily;
    use menobis_core::model::problem::PreparedProblem;

    let fam = match family {
        "ME" => OccupationFamily::ME,
        "B" => OccupationFamily::B { layers },
        "W" => OccupationFamily::W { layers },
        other => {
            return Err(PyValueError::new_err(format!("invalid family: {other}")));
        }
    };
    let missing = |name: &str| -> PyErr {
        PyValueError::new_err(format!("microcanonical {constraint} requires {name}"))
    };
    let problem = match constraint {
        "edges_events" => {
            let n = node_count.ok_or_else(|| missing("node_count"))?;
            let e = residual_edges.ok_or_else(|| missing("residual_edges"))?;
            let t = residual_total.ok_or_else(|| missing("residual_total"))?;
            PreparedProblem::new(
                fam,
                n,
                self_loops,
                admissible_pairs(n, self_loops),
                Some(e),
                Some(t),
                None,
                None,
                None,
                None,
            )
        }
        "degree_events" => {
            let d_out = degree_out.ok_or_else(|| missing("degree_out"))?;
            let d_in = degree_in.ok_or_else(|| missing("degree_in"))?;
            let t = residual_total.ok_or_else(|| missing("residual_total"))?;
            let n = d_out.len();
            PreparedProblem::new(
                fam,
                n,
                self_loops,
                admissible_pairs(n, self_loops),
                None,
                Some(t),
                Some(d_out),
                Some(d_in),
                None,
                None,
            )
        }
        "strength" => {
            let s_out = strength_out.ok_or_else(|| missing("strength_out"))?;
            let s_in = strength_in.ok_or_else(|| missing("strength_in"))?;
            let n = s_out.len();
            PreparedProblem::new(
                fam,
                n,
                self_loops,
                admissible_pairs(n, self_loops),
                None,
                None,
                None,
                None,
                Some(s_out),
                Some(s_in),
            )
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "microcanonical does not support constraint={other}"
            )));
        }
    };
    let config = MicrocanonicalConfig {
        seed,
        burn_in_sweeps,
        sweeps_per_sample,
        self_loops,
    };
    match route_sample(&problem, &config) {
        Ok(net) => Ok((net.sources, net.targets, net.occ_nums)),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_model_grandcanonical(
    family: &str,
    constraint: &str,
    self_loops: bool,
    layers: u32,
    seed: u64,
    x: Option<Vec<f64>>,
    y: Option<Vec<f64>>,
    z: Option<Vec<f64>>,
    w: Option<Vec<f64>>,
    lam: Option<f64>,
    q: Option<f64>,
    occupation: Option<f64>,
    gamma: Option<f64>,
    coord_x: Option<Vec<f64>>,
    coord_y: Option<Vec<f64>>,
    node_count: Option<usize>,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    use menobis_core::generation::grandcanonical::{sample_grandcanonical, GrandCanonicalCase};
    use menobis_core::model::family::OccupationFamily;

    let fam = match family {
        "ME" => OccupationFamily::ME,
        "B" => OccupationFamily::B { layers },
        "W" if layers == 1 => OccupationFamily::W { layers: 1 },
        "W" => OccupationFamily::W { layers },
        other => {
            return Err(PyValueError::new_err(format!("invalid family: {other}")));
        }
    };
    let case = match constraint {
        "strength" => GrandCanonicalCase::Strength,
        "strength_edges" => GrandCanonicalCase::StrengthEdges,
        "strength_degree" => GrandCanonicalCase::StrengthDegree,
        "strength_cost" => GrandCanonicalCase::StrengthCost,
        "degree_events" => GrandCanonicalCase::DegreeEvents,
        "edges_events" => GrandCanonicalCase::EdgesEvents,
        other => {
            return Err(PyValueError::new_err(format!(
                "grand_canonical does not support constraint={other}"
            )));
        }
    };
    let missing = |name: &str| -> PyErr {
        PyValueError::new_err(format!("grand_canonical {constraint} requires {name}"))
    };
    let (x, y) = if constraint == "edges_events" {
        (Vec::new(), Vec::new())
    } else {
        let x = x.ok_or_else(|| missing("x"))?;
        let y = y.ok_or_else(|| missing("y"))?;
        (x, y)
    };
    match sample_grandcanonical(
        case,
        fam,
        &x,
        &y,
        lam,
        q,
        z.as_deref(),
        w.as_deref(),
        gamma,
        coord_x.as_deref(),
        coord_y.as_deref(),
        node_count,
        q,
        occupation,
        self_loops,
        seed,
    ) {
        Ok(net) => Ok((net.sources, net.targets, net.occ_nums)),
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

fn sample_model_canonical(
    family: &str,
    constraint: &str,
    self_loops: bool,
    seed: u64,
    x: Option<Vec<f64>>,
    y: Option<Vec<f64>>,
    total_events: Option<u64>,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    if family != "ME" || constraint != "strength" {
        return Err(PyValueError::new_err(
            "canonical supports only family=ME, constraint=STRENGTH",
        ));
    }
    let x = x.ok_or_else(|| PyValueError::new_err("canonical requires x"))?;
    let y = y.ok_or_else(|| PyValueError::new_err("canonical requires y"))?;
    let t = total_events.ok_or_else(|| PyValueError::new_err("canonical requires total_events"))?;
    let net = menobis_core::generation::canonical::sample_strength_multinomial(
        &x, &y, t, self_loops, seed,
    );
    Ok((net.sources, net.targets, net.occ_nums))
}

/// Number of admissible ordered pairs for a complete domain.
fn admissible_pairs(node_count: usize, self_loops: bool) -> usize {
    if self_loops {
        node_count.saturating_mul(node_count)
    } else {
        node_count.saturating_mul(node_count.saturating_sub(1))
    }
}
