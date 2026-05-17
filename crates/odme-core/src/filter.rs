//! Statistical filtering kernels for fitted ODME null models.

use crate::distribution::{PairDistribution, WeightFamily};
use crate::pairs::{
    row_ranges, CandidateSupport, DegreeEventsProvider, FixedStrengthProvider,
    PairDistributionProvider, SparsePoissonRateMapProvider, SparsePoissonRateProvider,
    StrengthCostProvider, StrengthDegreeProvider, StrengthEdgesProvider, PARALLEL_PAIR_THRESHOLD,
    SPARSE_CHUNK_SIZE,
};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// P-values and expectations for observed positive edges.
#[derive(Clone, Debug, Default)]
pub struct ObservedFilterResult {
    pub upper_pvalues: Vec<f64>,
    pub lower_pvalues: Vec<f64>,
    pub expected: Vec<f64>,
    pub occupation: Vec<f64>,
}

/// Absent pairs significant under the lower tail.
#[derive(Clone, Debug, Default)]
pub struct AbsentFilterResult {
    pub sources: Vec<u64>,
    pub targets: Vec<u64>,
    pub lower_pvalues: Vec<f64>,
    pub expected: Vec<f64>,
    pub occupation: Vec<f64>,
}

/// Options for provider-backed absent-edge detection.
#[derive(Clone, Copy, Debug)]
pub struct AbsentFilterOptions {
    pub alpha_lower: f64,
    pub min_occupation: f64,
    pub min_expected: f64,
    pub max_absent: Option<usize>,
}

#[must_use]
pub fn filter_observed_provider<P>(
    sources: &[u64],
    targets: &[u64],
    weights: &[u64],
    provider: &P,
) -> ObservedFilterResult
where
    P: PairDistributionProvider,
{
    filter_observed_with(sources, targets, weights, |i, j| {
        provider
            .distribution(i, j)
            .unwrap_or(PairDistribution::Poisson { rate: 0.0 })
    })
}

#[must_use]
pub fn filter_observed_with<F>(
    sources: &[u64],
    targets: &[u64],
    weights: &[u64],
    provider: F,
) -> ObservedFilterResult
where
    F: Fn(usize, usize) -> PairDistribution + Sync,
{
    let rows: Vec<(f64, f64, f64, f64)> = sources
        .par_iter()
        .zip(targets.par_iter())
        .zip(weights.par_iter())
        .map(|((&source, &target), &weight)| {
            let dist = provider(source as usize, target as usize);
            (
                dist.upper_pvalue(weight),
                dist.lower_pvalue(weight),
                dist.expected(),
                dist.occupation_probability(),
            )
        })
        .collect();

    let mut result = ObservedFilterResult {
        upper_pvalues: Vec::with_capacity(rows.len()),
        lower_pvalues: Vec::with_capacity(rows.len()),
        expected: Vec::with_capacity(rows.len()),
        occupation: Vec::with_capacity(rows.len()),
    };
    for (upper, lower, expected, occupation) in rows {
        result.upper_pvalues.push(upper);
        result.lower_pvalues.push(lower);
        result.expected.push(expected);
        result.occupation.push(occupation);
    }
    result
}

#[must_use]
pub fn detect_absent_provider<P>(
    provider: &P,
    observed_sources: &[u64],
    observed_targets: &[u64],
    options: AbsentFilterOptions,
) -> AbsentFilterResult
where
    P: PairDistributionProvider,
{
    match provider.support() {
        CandidateSupport::AllPairs {
            node_count,
            self_loops,
        } => detect_absent_all_pairs_provider(
            provider,
            node_count,
            self_loops,
            observed_sources,
            observed_targets,
            options,
        ),
        CandidateSupport::SparsePairs { sources, targets } => detect_absent_sparse_provider(
            provider,
            sources,
            targets,
            observed_sources,
            observed_targets,
            options,
        ),
    }
}

fn detect_absent_all_pairs_provider<P>(
    provider: &P,
    node_count: usize,
    self_loops: bool,
    observed_sources: &[u64],
    observed_targets: &[u64],
    options: AbsentFilterOptions,
) -> AbsentFilterResult
where
    P: PairDistributionProvider,
{
    let observed: HashSet<(usize, usize)> = observed_sources
        .iter()
        .zip(observed_targets.iter())
        .map(|(&source, &target)| (source as usize, target as usize))
        .collect();
    let candidate_pairs = if self_loops {
        node_count.saturating_mul(node_count)
    } else {
        node_count.saturating_mul(node_count.saturating_sub(1))
    };

    let scan_range = |start: usize, end: usize| {
        let mut local = AbsentFilterResult::default();
        for i in start..end {
            for j in 0..node_count {
                if (!self_loops && i == j) || observed.contains(&(i, j)) {
                    continue;
                }
                if let Some(dist) = provider.distribution(i, j) {
                    push_absent_if_significant(&mut local, i as u64, j as u64, dist, options);
                }
            }
        }
        local
    };

    let mut result = if candidate_pairs < PARALLEL_PAIR_THRESHOLD {
        scan_range(0, node_count)
    } else {
        let chunks: Vec<AbsentFilterResult> = row_ranges(node_count)
            .into_par_iter()
            .map(|(start, end)| scan_range(start, end))
            .collect();
        merge_absent(chunks)
    };

    if let Some(limit) = options.max_absent {
        truncate_absent(&mut result, limit);
    }
    result
}

fn detect_absent_sparse_provider<P>(
    provider: &P,
    candidate_sources: &[u64],
    candidate_targets: &[u64],
    observed_sources: &[u64],
    observed_targets: &[u64],
    options: AbsentFilterOptions,
) -> AbsentFilterResult
where
    P: PairDistributionProvider,
{
    let observed: HashSet<(u64, u64)> = observed_sources
        .iter()
        .zip(observed_targets.iter())
        .map(|(&source, &target)| (source, target))
        .collect();

    let scan_range = |start: usize, end: usize| {
        let mut local = AbsentFilterResult::default();
        for index in start..end {
            let source = candidate_sources[index];
            let target = candidate_targets[index];
            if observed.contains(&(source, target)) {
                continue;
            }
            if let Some(dist) = provider.distribution_at(index, source as usize, target as usize) {
                push_absent_if_significant(&mut local, source, target, dist, options);
            }
        }
        local
    };

    let mut result = if candidate_sources.len() < SPARSE_CHUNK_SIZE {
        scan_range(0, candidate_sources.len())
    } else {
        let chunks: Vec<AbsentFilterResult> = (0..candidate_sources.len())
            .step_by(SPARSE_CHUNK_SIZE)
            .map(|start| {
                (
                    start,
                    (start + SPARSE_CHUNK_SIZE).min(candidate_sources.len()),
                )
            })
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(start, end)| scan_range(start, end))
            .collect();
        merge_absent(chunks)
    };

    if let Some(limit) = options.max_absent {
        truncate_absent(&mut result, limit);
    }
    result
}

fn push_absent_if_significant(
    result: &mut AbsentFilterResult,
    source: u64,
    target: u64,
    dist: PairDistribution,
    options: AbsentFilterOptions,
) {
    let occupation = dist.occupation_probability();
    let expected = dist.expected();
    if occupation < options.min_occupation || expected < options.min_expected {
        return;
    }
    let lower = dist.lower_pvalue(0);
    if lower < options.alpha_lower {
        result.sources.push(source);
        result.targets.push(target);
        result.lower_pvalues.push(lower);
        result.expected.push(expected);
        result.occupation.push(occupation);
    }
}

#[must_use]
pub fn filter_strength_poisson(
    x: &[f64],
    y: &[f64],
    sources: &[u64],
    targets: &[u64],
    weights: &[u64],
) -> ObservedFilterResult {
    filter_observed_provider(
        sources,
        targets,
        weights,
        &FixedStrengthProvider {
            family: WeightFamily::Poisson,
            x,
            y,
            self_loops: true,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn absent_strength_poisson(
    x: &[f64],
    y: &[f64],
    sources: &[u64],
    targets: &[u64],
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> AbsentFilterResult {
    detect_absent_provider(
        &FixedStrengthProvider {
            family: WeightFamily::Poisson,
            x,
            y,
            self_loops,
        },
        sources,
        targets,
        AbsentFilterOptions {
            alpha_lower,
            min_occupation,
            min_expected,
            max_absent,
        },
    )
}

#[must_use]
pub fn filter_custom_poisson(
    rate_sources: &[u64],
    rate_targets: &[u64],
    rates: &[f64],
    sources: &[u64],
    targets: &[u64],
    weights: &[u64],
) -> ObservedFilterResult {
    let rate_map: HashMap<(usize, usize), f64> = rate_sources
        .iter()
        .zip(rate_targets.iter())
        .zip(rates.iter())
        .map(|((&source, &target), &rate)| ((source as usize, target as usize), rate))
        .collect();
    filter_observed_provider(
        sources,
        targets,
        weights,
        &SparsePoissonRateMapProvider {
            sources: rate_sources,
            targets: rate_targets,
            map: &rate_map,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn absent_custom_poisson(
    rate_sources: &[u64],
    rate_targets: &[u64],
    rates: &[f64],
    observed_sources: &[u64],
    observed_targets: &[u64],
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> AbsentFilterResult {
    detect_absent_provider(
        &SparsePoissonRateProvider {
            sources: rate_sources,
            targets: rate_targets,
            rates,
        },
        observed_sources,
        observed_targets,
        AbsentFilterOptions {
            alpha_lower,
            min_occupation,
            min_expected,
            max_absent,
        },
    )
}

#[must_use]
pub fn filter_strength_edges_poisson(
    x: &[f64],
    y: &[f64],
    lam: f64,
    sources: &[u64],
    targets: &[u64],
    weights: &[u64],
) -> ObservedFilterResult {
    filter_observed_provider(
        sources,
        targets,
        weights,
        &StrengthEdgesProvider {
            family: WeightFamily::Poisson,
            x,
            y,
            lambda: lam,
            self_loops: true,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn absent_strength_edges_poisson(
    x: &[f64],
    y: &[f64],
    lam: f64,
    sources: &[u64],
    targets: &[u64],
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> AbsentFilterResult {
    detect_absent_provider(
        &StrengthEdgesProvider {
            family: WeightFamily::Poisson,
            x,
            y,
            lambda: lam,
            self_loops,
        },
        sources,
        targets,
        AbsentFilterOptions {
            alpha_lower,
            min_occupation,
            min_expected,
            max_absent,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn filter_strength_cost_poisson(
    x: &[f64],
    y: &[f64],
    gamma: f64,
    cost_sources: &[usize],
    cost_targets: &[usize],
    cost_values: &[f64],
    sources: &[u64],
    targets: &[u64],
    weights: &[u64],
) -> ObservedFilterResult {
    let costs = build_cost_map(cost_sources, cost_targets, cost_values);
    filter_observed_provider(
        sources,
        targets,
        weights,
        &StrengthCostProvider {
            family: WeightFamily::Poisson,
            x,
            y,
            gamma,
            costs: &costs,
            self_loops: true,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn absent_strength_cost_poisson(
    x: &[f64],
    y: &[f64],
    gamma: f64,
    cost_sources: &[usize],
    cost_targets: &[usize],
    cost_values: &[f64],
    observed_sources: &[u64],
    observed_targets: &[u64],
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> AbsentFilterResult {
    let costs = build_cost_map(cost_sources, cost_targets, cost_values);
    detect_absent_provider(
        &StrengthCostProvider {
            family: WeightFamily::Poisson,
            x,
            y,
            gamma,
            costs: &costs,
            self_loops,
        },
        observed_sources,
        observed_targets,
        AbsentFilterOptions {
            alpha_lower,
            min_occupation,
            min_expected,
            max_absent,
        },
    )
}

#[must_use]
pub fn filter_strength_degree_poisson(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    w: &[f64],
    sources: &[u64],
    targets: &[u64],
    weights: &[u64],
) -> ObservedFilterResult {
    filter_observed_provider(
        sources,
        targets,
        weights,
        &StrengthDegreeProvider {
            family: WeightFamily::Poisson,
            x,
            y,
            z,
            w,
            self_loops: true,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn absent_strength_degree_poisson(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    w: &[f64],
    sources: &[u64],
    targets: &[u64],
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> AbsentFilterResult {
    detect_absent_provider(
        &StrengthDegreeProvider {
            family: WeightFamily::Poisson,
            x,
            y,
            z,
            w,
            self_loops,
        },
        sources,
        targets,
        AbsentFilterOptions {
            alpha_lower,
            min_occupation,
            min_expected,
            max_absent,
        },
    )
}

#[must_use]
pub fn filter_degree_events_poisson(
    x: &[f64],
    y: &[f64],
    positive_weight_rate: f64,
    sources: &[u64],
    targets: &[u64],
    weights: &[u64],
) -> ObservedFilterResult {
    filter_observed_provider(
        sources,
        targets,
        weights,
        &DegreeEventsProvider {
            family: WeightFamily::Poisson,
            x,
            y,
            positive_weight_rate,
            self_loops: true,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn absent_degree_events_poisson(
    x: &[f64],
    y: &[f64],
    positive_weight_rate: f64,
    sources: &[u64],
    targets: &[u64],
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> AbsentFilterResult {
    detect_absent_provider(
        &DegreeEventsProvider {
            family: WeightFamily::Poisson,
            x,
            y,
            positive_weight_rate,
            self_loops,
        },
        sources,
        targets,
        AbsentFilterOptions {
            alpha_lower,
            min_occupation,
            min_expected,
            max_absent,
        },
    )
}

// --- Geometric family ---

#[must_use]
pub fn filter_strength_geometric(
    x: &[f64],
    y: &[f64],
    sources: &[u64],
    targets: &[u64],
    weights: &[u64],
) -> ObservedFilterResult {
    filter_observed_provider(
        sources,
        targets,
        weights,
        &FixedStrengthProvider {
            family: WeightFamily::Geometric,
            x,
            y,
            self_loops: true,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn absent_strength_geometric(
    x: &[f64],
    y: &[f64],
    sources: &[u64],
    targets: &[u64],
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> AbsentFilterResult {
    detect_absent_provider(
        &FixedStrengthProvider {
            family: WeightFamily::Geometric,
            x,
            y,
            self_loops,
        },
        sources,
        targets,
        AbsentFilterOptions {
            alpha_lower,
            min_occupation,
            min_expected,
            max_absent,
        },
    )
}

// --- Binomial family ---

#[must_use]
pub fn filter_strength_binomial(
    x: &[f64],
    y: &[f64],
    layers: u32,
    sources: &[u64],
    targets: &[u64],
    weights: &[u64],
) -> ObservedFilterResult {
    filter_observed_provider(
        sources,
        targets,
        weights,
        &FixedStrengthProvider {
            family: WeightFamily::Binomial(layers),
            x,
            y,
            self_loops: true,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn absent_strength_binomial(
    x: &[f64],
    y: &[f64],
    layers: u32,
    sources: &[u64],
    targets: &[u64],
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> AbsentFilterResult {
    detect_absent_provider(
        &FixedStrengthProvider {
            family: WeightFamily::Binomial(layers),
            x,
            y,
            self_loops,
        },
        sources,
        targets,
        AbsentFilterOptions {
            alpha_lower,
            min_occupation,
            min_expected,
            max_absent,
        },
    )
}

// --- Negative binomial family ---

#[must_use]
pub fn filter_strength_neg_binomial(
    x: &[f64],
    y: &[f64],
    layers: u32,
    sources: &[u64],
    targets: &[u64],
    weights: &[u64],
) -> ObservedFilterResult {
    filter_observed_provider(
        sources,
        targets,
        weights,
        &FixedStrengthProvider {
            family: WeightFamily::NegBinomial(layers),
            x,
            y,
            self_loops: true,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn absent_strength_neg_binomial(
    x: &[f64],
    y: &[f64],
    layers: u32,
    sources: &[u64],
    targets: &[u64],
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> AbsentFilterResult {
    detect_absent_provider(
        &FixedStrengthProvider {
            family: WeightFamily::NegBinomial(layers),
            x,
            y,
            self_loops,
        },
        sources,
        targets,
        AbsentFilterOptions {
            alpha_lower,
            min_occupation,
            min_expected,
            max_absent,
        },
    )
}

// --- Binomial constraint variants ---

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn filter_strength_cost_binomial(
    x: &[f64],
    y: &[f64],
    gamma: f64,
    cost_sources: &[usize],
    cost_targets: &[usize],
    cost_values: &[f64],
    layers: u32,
    sources: &[u64],
    targets: &[u64],
    weights: &[u64],
) -> ObservedFilterResult {
    let costs = build_cost_map(cost_sources, cost_targets, cost_values);
    filter_observed_provider(
        sources,
        targets,
        weights,
        &StrengthCostProvider {
            family: WeightFamily::Binomial(layers),
            x,
            y,
            gamma,
            costs: &costs,
            self_loops: true,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn absent_strength_cost_binomial(
    x: &[f64],
    y: &[f64],
    gamma: f64,
    cost_sources: &[usize],
    cost_targets: &[usize],
    cost_values: &[f64],
    layers: u32,
    observed_sources: &[u64],
    observed_targets: &[u64],
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> AbsentFilterResult {
    let costs = build_cost_map(cost_sources, cost_targets, cost_values);
    detect_absent_provider(
        &StrengthCostProvider {
            family: WeightFamily::Binomial(layers),
            x,
            y,
            gamma,
            costs: &costs,
            self_loops,
        },
        observed_sources,
        observed_targets,
        AbsentFilterOptions {
            alpha_lower,
            min_occupation,
            min_expected,
            max_absent,
        },
    )
}

#[must_use]
pub fn filter_strength_edges_binomial(
    x: &[f64],
    y: &[f64],
    lam: f64,
    layers: u32,
    sources: &[u64],
    targets: &[u64],
    weights: &[u64],
) -> ObservedFilterResult {
    filter_observed_provider(
        sources,
        targets,
        weights,
        &StrengthEdgesProvider {
            family: WeightFamily::Binomial(layers),
            x,
            y,
            lambda: lam,
            self_loops: true,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn absent_strength_edges_binomial(
    x: &[f64],
    y: &[f64],
    lam: f64,
    layers: u32,
    sources: &[u64],
    targets: &[u64],
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> AbsentFilterResult {
    detect_absent_provider(
        &StrengthEdgesProvider {
            family: WeightFamily::Binomial(layers),
            x,
            y,
            lambda: lam,
            self_loops,
        },
        sources,
        targets,
        AbsentFilterOptions {
            alpha_lower,
            min_occupation,
            min_expected,
            max_absent,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn filter_strength_degree_binomial(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    w: &[f64],
    layers: u32,
    sources: &[u64],
    targets: &[u64],
    weights: &[u64],
) -> ObservedFilterResult {
    filter_observed_provider(
        sources,
        targets,
        weights,
        &StrengthDegreeProvider {
            family: WeightFamily::Binomial(layers),
            x,
            y,
            z,
            w,
            self_loops: true,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn absent_strength_degree_binomial(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    w: &[f64],
    layers: u32,
    sources: &[u64],
    targets: &[u64],
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> AbsentFilterResult {
    detect_absent_provider(
        &StrengthDegreeProvider {
            family: WeightFamily::Binomial(layers),
            x,
            y,
            z,
            w,
            self_loops,
        },
        sources,
        targets,
        AbsentFilterOptions {
            alpha_lower,
            min_occupation,
            min_expected,
            max_absent,
        },
    )
}

#[must_use]
pub fn filter_degree_events_binomial(
    x: &[f64],
    y: &[f64],
    positive_weight_rate: f64,
    layers: u32,
    sources: &[u64],
    targets: &[u64],
    weights: &[u64],
) -> ObservedFilterResult {
    filter_observed_provider(
        sources,
        targets,
        weights,
        &DegreeEventsProvider {
            family: WeightFamily::Binomial(layers),
            x,
            y,
            positive_weight_rate,
            self_loops: true,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn absent_degree_events_binomial(
    x: &[f64],
    y: &[f64],
    positive_weight_rate: f64,
    layers: u32,
    sources: &[u64],
    targets: &[u64],
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
) -> AbsentFilterResult {
    detect_absent_provider(
        &DegreeEventsProvider {
            family: WeightFamily::Binomial(layers),
            x,
            y,
            positive_weight_rate,
            self_loops,
        },
        sources,
        targets,
        AbsentFilterOptions {
            alpha_lower,
            min_occupation,
            min_expected,
            max_absent,
        },
    )
}

fn build_cost_map(
    cost_sources: &[usize],
    cost_targets: &[usize],
    cost_values: &[f64],
) -> HashMap<(usize, usize), f64> {
    cost_sources
        .iter()
        .zip(cost_targets.iter())
        .zip(cost_values.iter())
        .map(|((&s, &t), &v)| ((s, t), v))
        .collect()
}

fn merge_absent(chunks: Vec<AbsentFilterResult>) -> AbsentFilterResult {
    let total = chunks.iter().map(|chunk| chunk.sources.len()).sum();
    let mut result = AbsentFilterResult {
        sources: Vec::with_capacity(total),
        targets: Vec::with_capacity(total),
        lower_pvalues: Vec::with_capacity(total),
        expected: Vec::with_capacity(total),
        occupation: Vec::with_capacity(total),
    };
    for mut chunk in chunks {
        result.sources.append(&mut chunk.sources);
        result.targets.append(&mut chunk.targets);
        result.lower_pvalues.append(&mut chunk.lower_pvalues);
        result.expected.append(&mut chunk.expected);
        result.occupation.append(&mut chunk.occupation);
    }
    result
}

fn truncate_absent(result: &mut AbsentFilterResult, limit: usize) {
    result.sources.truncate(limit);
    result.targets.truncate(limit);
    result.lower_pvalues.truncate(limit);
    result.expected.truncate(limit);
    result.occupation.truncate(limit);
}

#[must_use]
pub fn benjamini_hochberg(pvalues: &[f64], alpha: f64) -> Vec<bool> {
    let m = pvalues.len();
    if m == 0 {
        return Vec::new();
    }
    let mut indexed: Vec<(usize, f64)> = pvalues.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut cutoff_rank = None;
    for (rank_zero, &(_, pvalue)) in indexed.iter().enumerate() {
        let rank = rank_zero + 1;
        if pvalue <= alpha * rank as f64 / m as f64 {
            cutoff_rank = Some(rank_zero);
        }
    }
    let mut rejected = vec![false; m];
    if let Some(cutoff) = cutoff_rank {
        for &(idx, _) in indexed.iter().take(cutoff + 1) {
            rejected[idx] = true;
        }
    }
    rejected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_pvalues_match_small_hand_values() {
        let dist = PairDistribution::Poisson { rate: 2.0 };
        assert!((dist.lower_pvalue(0) - (-2.0_f64).exp()).abs() < 1e-12);
        assert!((dist.upper_pvalue(0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zip_zero_probability_is_one_minus_occupation() {
        let dist = PairDistribution::ZipPoisson {
            occupation: 0.7,
            rate: 2.0,
        };
        assert!((dist.lower_pvalue(0) - 0.3).abs() < 1e-12);
        assert!((dist.upper_pvalue(0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn absent_filter_uses_occupation_threshold() {
        let x = vec![1.0, 1.0];
        let y = vec![1.0, 1.0];
        let absent = absent_strength_poisson(&x, &y, &[0], &[1], true, 0.9, 0.5, 0.0, None);
        assert!(absent
            .sources
            .iter()
            .zip(absent.targets.iter())
            .all(|pair| pair != (&0, &1)));
    }

    #[test]
    fn all_pairs_absent_provider_respects_self_loop_mask() {
        let x = vec![10.0, 10.0];
        let y = vec![10.0, 10.0];
        let absent = absent_strength_poisson(&x, &y, &[], &[], false, 0.1, 0.0, 0.0, None);
        assert!(absent
            .sources
            .iter()
            .zip(absent.targets.iter())
            .all(|(&source, &target)| source != target));
    }

    #[test]
    fn sparse_absent_provider_preserves_support_order_and_limit() {
        let absent = absent_custom_poisson(
            &[2, 0, 1],
            &[0, 1, 2],
            &[10.0, 10.0, 10.0],
            &[0],
            &[1],
            0.1,
            0.0,
            0.0,
            Some(2),
        );
        assert_eq!(absent.sources, vec![2, 1]);
        assert_eq!(absent.targets, vec![0, 2]);
    }

    #[test]
    fn benjamini_hochberg_rejects_prefix() {
        let rejected = benjamini_hochberg(&[0.001, 0.02, 0.5], 0.05);
        assert_eq!(rejected, vec![true, true, false]);
    }
}
