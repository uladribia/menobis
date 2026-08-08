//! Grand-canonical generation: independent pair sampling from fitted
//! multipliers. Pair occupations are drawn independently from the
//! family pair distribution.

use crate::distribution::PairDistribution;
use crate::model::family::OccupationFamily;
use crate::pairs::{
    chunk_seed, row_ranges, CandidateSupport, DegreeEventsProvider, EdgesEventsProvider,
    EuclideanCostProvider, FixedStrengthProvider, NormalizedSparsePoissonProvider,
    PairDistributionProvider, StrengthCostProvider, StrengthDegreeProvider, StrengthEdgesProvider,
    PARALLEL_PAIR_THRESHOLD, SPARSE_CHUNK_SIZE,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use super::output::{merge_samples, SampledNetwork};

#[derive(Clone, Copy, Debug)]
struct PairDraw {
    source: u64,
    target: u64,
    distribution: PairDistribution,
}

/// Internal ontology for provider-backed independent-pair samplers.
enum SamplingModel<'a> {
    FixedStrength {
        x: &'a [f64],
        y: &'a [f64],
        family: OccupationFamily,
        self_loops: bool,
    },
    DegreeEvents {
        x: &'a [f64],
        y: &'a [f64],
        positive_intensity: f64,
        family: OccupationFamily,
        self_loops: bool,
    },
    StrengthEdges {
        x: &'a [f64],
        y: &'a [f64],
        lambda: f64,
        family: OccupationFamily,
        self_loops: bool,
    },
    StrengthDegree {
        x: &'a [f64],
        y: &'a [f64],
        z: &'a [f64],
        w: &'a [f64],
        family: OccupationFamily,
        self_loops: bool,
    },
}

fn push_sampled_pair(result: &mut SampledNetwork, pair: PairDraw, rng: &mut StdRng) {
    let occ_num = pair.distribution.sample(rng);
    if occ_num > 0 {
        result.sources.push(pair.source);
        result.targets.push(pair.target);
        result.occ_nums.push(occ_num);
    }
}

fn sample_independent_pairs<I>(pairs: I, rng: &mut StdRng) -> SampledNetwork
where
    I: IntoIterator<Item = PairDraw>,
{
    let mut result = SampledNetwork::default();
    for pair in pairs {
        push_sampled_pair(&mut result, pair, rng);
    }
    result
}

fn sample_model(model: SamplingModel<'_>, seed: u64) -> SampledNetwork {
    match model {
        SamplingModel::FixedStrength {
            x,
            y,
            family,
            self_loops,
        } => sample_provider(
            &FixedStrengthProvider {
                x,
                y,
                family: family.into(),
                self_loops,
            },
            seed,
        ),
        SamplingModel::DegreeEvents {
            x,
            y,
            positive_intensity,
            family,
            self_loops,
        } => sample_provider(
            &DegreeEventsProvider {
                x,
                y,
                positive_intensity,
                family: family.into(),
                self_loops,
            },
            seed,
        ),
        SamplingModel::StrengthEdges {
            x,
            y,
            lambda,
            family,
            self_loops,
        } => sample_provider(
            &StrengthEdgesProvider {
                x,
                y,
                lambda,
                family: family.into(),
                self_loops,
            },
            seed,
        ),
        SamplingModel::StrengthDegree {
            x,
            y,
            z,
            w,
            family,
            self_loops,
        } => sample_provider(
            &StrengthDegreeProvider {
                x,
                y,
                z,
                w,
                family: family.into(),
                self_loops,
            },
            seed,
        ),
    }
}

fn sample_provider<P>(provider: &P, seed: u64) -> SampledNetwork
where
    P: PairDistributionProvider,
{
    match provider.support() {
        CandidateSupport::AllPairs {
            node_count,
            self_loops,
        } => sample_all_pairs_by_rows(node_count, self_loops, seed, |i, j| {
            provider.distribution(i, j)
        }),
        CandidateSupport::SparsePairs { sources, targets } => {
            sample_sparse_provider(provider, sources, targets, seed)
        }
    }
}

fn sample_sparse_provider<P>(
    provider: &P,
    sources: &[u64],
    targets: &[u64],
    seed: u64,
) -> SampledNetwork
where
    P: PairDistributionProvider,
{
    if sources.len() < SPARSE_CHUNK_SIZE {
        let pairs = sources.iter().zip(targets.iter()).enumerate().filter_map(
            |(index, (&source, &target))| {
                provider
                    .distribution_at(index, source as usize, target as usize)
                    .map(|distribution| PairDraw {
                        source,
                        target,
                        distribution,
                    })
            },
        );
        return sample_independent_pairs(pairs, &mut StdRng::seed_from_u64(seed));
    }

    let chunks: Vec<SampledNetwork> = (0..sources.len())
        .step_by(SPARSE_CHUNK_SIZE)
        .map(|start| (start, (start + SPARSE_CHUNK_SIZE).min(sources.len())))
        .collect::<Vec<_>>()
        .into_par_iter()
        .enumerate()
        .map(|(chunk_index, (start, end))| {
            let mut rng = StdRng::seed_from_u64(chunk_seed(seed, chunk_index));
            let mut result = SampledNetwork::default();
            for index in start..end {
                if let Some(distribution) = provider.distribution_at(
                    index,
                    sources[index] as usize,
                    targets[index] as usize,
                ) {
                    push_sampled_pair(
                        &mut result,
                        PairDraw {
                            source: sources[index],
                            target: targets[index],
                            distribution,
                        },
                        &mut rng,
                    );
                }
            }
            result
        })
        .collect();
    merge_samples(chunks)
}

fn sample_all_pairs_by_rows<F>(n: usize, self_loops: bool, seed: u64, pair_fn: F) -> SampledNetwork
where
    F: Fn(usize, usize) -> Option<PairDistribution> + Sync,
{
    let candidate_pairs = if self_loops {
        n.saturating_mul(n)
    } else {
        n.saturating_mul(n.saturating_sub(1))
    };
    if candidate_pairs < PARALLEL_PAIR_THRESHOLD {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut result = SampledNetwork::default();
        for i in 0..n {
            for j in 0..n {
                if !self_loops && i == j {
                    continue;
                }
                if let Some(distribution) = pair_fn(i, j) {
                    push_sampled_pair(
                        &mut result,
                        PairDraw {
                            source: i as u64,
                            target: j as u64,
                            distribution,
                        },
                        &mut rng,
                    );
                }
            }
        }
        return result;
    }

    let chunks: Vec<SampledNetwork> = row_ranges(n)
        .into_par_iter()
        .enumerate()
        .map(|(chunk_index, (start, end))| {
            let mut rng = StdRng::seed_from_u64(chunk_seed(seed, chunk_index));
            let mut result = SampledNetwork::default();
            for i in start..end {
                for j in 0..n {
                    if !self_loops && i == j {
                        continue;
                    }
                    if let Some(distribution) = pair_fn(i, j) {
                        push_sampled_pair(
                            &mut result,
                            PairDraw {
                                source: i as u64,
                                target: j as u64,
                                distribution,
                            },
                            &mut rng,
                        );
                    }
                }
            }
            result
        })
        .collect();
    merge_samples(chunks)
}

/// Sample custom p_ij grand-canonical Poisson graph with E[t_ij] = T p_ij.
pub fn sample_custom_poisson(
    sources: &[u64],
    targets: &[u64],
    probabilities: &[f64],
    total_events: u64,
    seed: u64,
) -> SampledNetwork {
    let p_sum: f64 = probabilities.iter().sum();
    if p_sum <= 0.0 {
        return SampledNetwork::default();
    }
    sample_provider(
        &NormalizedSparsePoissonProvider {
            sources,
            targets,
            probabilities,
            total_events,
            probability_sum: p_sum,
        },
        seed,
    )
}

pub fn sample_strength_poisson(
    x: &[f64],
    y: &[f64],
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_model(
        SamplingModel::FixedStrength {
            x,
            y,
            family: OccupationFamily::ME,
            self_loops,
        },
        seed,
    )
}

/// Sample from independent Geometric(1 - x_i*y_j) for all (i, j).
#[must_use]
pub fn sample_strength_geometric(
    x: &[f64],
    y: &[f64],
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_model(
        SamplingModel::FixedStrength {
            x,
            y,
            family: OccupationFamily::W { layers: 1 },
            self_loops,
        },
        seed,
    )
}

/// Sample from independent Binomial(M, x_i*y_j/(1+x_i*y_j)) for all (i, j).
#[must_use]
pub fn sample_strength_binomial(
    x: &[f64],
    y: &[f64],
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_model(
        SamplingModel::FixedStrength {
            x,
            y,
            family: OccupationFamily::B { layers },
            self_loops,
        },
        seed,
    )
}

/// Sample from independent NegativeBinomial(M, 1-x_i*y_j) for all (i, j).
#[must_use]
pub fn sample_strength_negative_binomial(
    x: &[f64],
    y: &[f64],
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_model(
        SamplingModel::FixedStrength {
            x,
            y,
            family: OccupationFamily::W { layers },
            self_loops,
        },
        seed,
    )
}

/// Sample from independent Poisson(x_i * y_j * exp(-gamma d_ij)).
///
/// Costs are generated on demand as Euclidean distances from projected XY
/// coordinates. MENoBiS does not accept user-supplied dense cost matrices.
#[must_use]
pub fn sample_strength_cost_poisson_coordinates(
    x: &[f64],
    y: &[f64],
    gamma: f64,
    coord_x: &[f64],
    coord_y: &[f64],
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_strength_cost_coordinates(
        x,
        y,
        gamma,
        coord_x,
        coord_y,
        OccupationFamily::ME,
        self_loops,
        seed,
    )
}

#[allow(clippy::too_many_arguments)]
fn sample_strength_cost_coordinates(
    x: &[f64],
    y: &[f64],
    gamma: f64,
    coord_x: &[f64],
    coord_y: &[f64],
    family: OccupationFamily,
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    let costs = EuclideanCostProvider {
        x: coord_x,
        y: coord_y,
    };
    sample_provider(
        &StrengthCostProvider {
            x,
            y,
            gamma,
            costs: &costs,
            family: family.into(),
            self_loops,
        },
        seed,
    )
}
/// Sample the symmetric EDGES_EVENTS grand-canonical model.
///
/// Every candidate pair shares the zero-inflated distribution with global
/// occupation probability and positive-support parameter `q`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn sample_edges_events(
    node_count: usize,
    q: f64,
    occupation: f64,
    family: OccupationFamily,
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_provider(
        &EdgesEventsProvider {
            node_count,
            q,
            occupation,
            family: family.into(),
            self_loops,
        },
        seed,
    )
}

/// Sample degree-events ME: Bernoulli occupation + positive Poisson(q).
#[must_use]
pub fn sample_degree_events_poisson(
    x: &[f64],
    y: &[f64],
    positive_intensity: f64,
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_model(
        SamplingModel::DegreeEvents {
            x,
            y,
            positive_intensity,
            family: OccupationFamily::ME,
            self_loops,
        },
        seed,
    )
}

/// Sample from the exact ME fixed-strength-degree zero-inflated model.
#[must_use]
pub fn sample_strength_degree_poisson(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    w: &[f64],
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_model(
        SamplingModel::StrengthDegree {
            x,
            y,
            z,
            w,
            family: OccupationFamily::ME,
            self_loops,
        },
        seed,
    )
}

/// Sample exact ME fixed-strength-and-edge-count zero-inflated model.
#[must_use]
pub fn sample_strength_edges_poisson(
    x: &[f64],
    y: &[f64],
    lam: f64,
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_model(
        SamplingModel::StrengthEdges {
            x,
            y,
            lambda: lam,
            family: OccupationFamily::ME,
            self_loops,
        },
        seed,
    )
}

/// Sample strength-cost binomial from Euclidean coordinate costs.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn sample_strength_cost_binomial_coordinates(
    x: &[f64],
    y: &[f64],
    gamma: f64,
    coord_x: &[f64],
    coord_y: &[f64],
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_strength_cost_coordinates(
        x,
        y,
        gamma,
        coord_x,
        coord_y,
        OccupationFamily::B { layers },
        self_loops,
        seed,
    )
}
/// Sample strength-cost geometric from Euclidean coordinate costs.
#[must_use]
pub fn sample_strength_cost_geometric_coordinates(
    x: &[f64],
    y: &[f64],
    gamma: f64,
    coord_x: &[f64],
    coord_y: &[f64],
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_strength_cost_coordinates(
        x,
        y,
        gamma,
        coord_x,
        coord_y,
        OccupationFamily::W { layers: 1 },
        self_loops,
        seed,
    )
}
/// Sample strength-cost negative binomial from Euclidean coordinate costs.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn sample_strength_cost_negative_binomial_coordinates(
    x: &[f64],
    y: &[f64],
    gamma: f64,
    coord_x: &[f64],
    coord_y: &[f64],
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_strength_cost_coordinates(
        x,
        y,
        gamma,
        coord_x,
        coord_y,
        OccupationFamily::W { layers },
        self_loops,
        seed,
    )
}
/// Sample strength-edges binomial zero-inflated: Bernoulli occupation + positive binomial(M, p).
#[must_use]
pub fn sample_strength_edges_binomial(
    x: &[f64],
    y: &[f64],
    lam: f64,
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_model(
        SamplingModel::StrengthEdges {
            x,
            y,
            lambda: lam,
            family: OccupationFamily::B { layers },
            self_loops,
        },
        seed,
    )
}

/// Sample strength-degree binomial zero-inflated: Bernoulli occupation + positive binomial(M, p).
#[must_use]
pub fn sample_strength_degree_binomial(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    w: &[f64],
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_model(
        SamplingModel::StrengthDegree {
            x,
            y,
            z,
            w,
            family: OccupationFamily::B { layers },
            self_loops,
        },
        seed,
    )
}

/// Sample degree-events binomial zero-inflated: Bernoulli occupation + positive binomial(M, mu).
#[must_use]
pub fn sample_degree_events_binomial(
    x: &[f64],
    y: &[f64],
    positive_intensity: f64,
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_model(
        SamplingModel::DegreeEvents {
            x,
            y,
            positive_intensity,
            family: OccupationFamily::B { layers },
            self_loops,
        },
        seed,
    )
}

/// Sample strength-edges geometric zero-inflated: Bernoulli occupation + positive geometric.
#[must_use]
pub fn sample_strength_edges_geometric(
    x: &[f64],
    y: &[f64],
    lam: f64,
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_model(
        SamplingModel::StrengthEdges {
            x,
            y,
            lambda: lam,
            family: OccupationFamily::W { layers: 1 },
            self_loops,
        },
        seed,
    )
}

/// Sample strength-edges negative binomial zero-inflated: Bernoulli occupation + positive negative binomial(M).
#[must_use]
pub fn sample_strength_edges_negative_binomial(
    x: &[f64],
    y: &[f64],
    lam: f64,
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_model(
        SamplingModel::StrengthEdges {
            x,
            y,
            lambda: lam,
            family: OccupationFamily::W { layers },
            self_loops,
        },
        seed,
    )
}

/// Sample strength-degree geometric zero-inflated: Bernoulli occupation + positive geometric.
#[must_use]
pub fn sample_strength_degree_geometric(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    w: &[f64],
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_model(
        SamplingModel::StrengthDegree {
            x,
            y,
            z,
            w,
            family: OccupationFamily::W { layers: 1 },
            self_loops,
        },
        seed,
    )
}

/// Sample strength-degree negative binomial zero-inflated: Bernoulli occupation + positive negative binomial(M).
#[must_use]
pub fn sample_strength_degree_negative_binomial(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    w: &[f64],
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_model(
        SamplingModel::StrengthDegree {
            x,
            y,
            z,
            w,
            family: OccupationFamily::W { layers },
            self_loops,
        },
        seed,
    )
}

/// Sample degree-events geometric zero-inflated: Bernoulli occupation + positive geometric.
#[must_use]
pub fn sample_degree_events_geometric(
    x: &[f64],
    y: &[f64],
    positive_intensity: f64,
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_model(
        SamplingModel::DegreeEvents {
            x,
            y,
            positive_intensity,
            family: OccupationFamily::W { layers: 1 },
            self_loops,
        },
        seed,
    )
}

/// Sample degree-events negative binomial zero-inflated: Bernoulli occupation + positive negative binomial(M).
#[must_use]
pub fn sample_degree_events_negative_binomial(
    x: &[f64],
    y: &[f64],
    positive_intensity: f64,
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    sample_model(
        SamplingModel::DegreeEvents {
            x,
            y,
            positive_intensity,
            family: OccupationFamily::W { layers },
            self_loops,
        },
        seed,
    )
}

/// Grand-canonical constraint case for the unified sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrandCanonicalCase {
    /// Fixed strength sequence (multipliers x, y).
    Strength,
    /// Fixed strength + expected edge count (x, y, lam).
    StrengthEdges,
    /// Fixed strength + expected degree (x, y, z, w).
    StrengthDegree,
    /// Fixed strength + expected cost (x, y, gamma, coords).
    StrengthCost,
    /// Fixed degree + total events (x, y, positive intensity).
    DegreeEvents,
    /// Global edges + events (node_count, q, occupation).
    EdgesEvents,
}

/// Unified grand-canonical sampling from fitted parameters.
///
/// Centralizes the per-constraint dispatch previously exposed as 24
/// separate family functions.  `x`/`y` are always required; the other
/// parameters are required per constraint:
///
/// - `Strength`: x, y
/// - `StrengthEdges`: x, y, lam
/// - `StrengthDegree`: x, y, z, w
/// - `StrengthCost`: x, y, gamma, coord_x, coord_y
/// - `DegreeEvents`: x, y, positive_intensity
/// - `EdgesEvents`: node_count, q, occupation
///
/// Returns a [`SampledNetwork`] or a structured error message when a
/// constraint-required parameter is missing.
#[allow(clippy::too_many_arguments)]
pub fn sample_grandcanonical(
    case: GrandCanonicalCase,
    family: OccupationFamily,
    x: &[f64],
    y: &[f64],
    lam: Option<f64>,
    positive_intensity: Option<f64>,
    z: Option<&[f64]>,
    w: Option<&[f64]>,
    gamma: Option<f64>,
    coord_x: Option<&[f64]>,
    coord_y: Option<&[f64]>,
    node_count: Option<usize>,
    q: Option<f64>,
    occupation: Option<f64>,
    self_loops: bool,
    seed: u64,
) -> Result<SampledNetwork, String> {
    match case {
        GrandCanonicalCase::Strength => Ok(sample_model(
            SamplingModel::FixedStrength {
                x,
                y,
                family,
                self_loops,
            },
            seed,
        )),
        GrandCanonicalCase::StrengthEdges => {
            let lam = lam.ok_or_else(|| "STRENGTH_EDGES requires lam".to_string())?;
            Ok(sample_model(
                SamplingModel::StrengthEdges {
                    x,
                    y,
                    lambda: lam,
                    family,
                    self_loops,
                },
                seed,
            ))
        }
        GrandCanonicalCase::StrengthDegree => {
            let z = z.ok_or_else(|| "STRENGTH_DEGREE requires z".to_string())?;
            let w = w.ok_or_else(|| "STRENGTH_DEGREE requires w".to_string())?;
            Ok(sample_model(
                SamplingModel::StrengthDegree {
                    x,
                    y,
                    z,
                    w,
                    family,
                    self_loops,
                },
                seed,
            ))
        }
        GrandCanonicalCase::StrengthCost => {
            let gamma = gamma.ok_or_else(|| "STRENGTH_COST requires gamma".to_string())?;
            let cx = coord_x.ok_or_else(|| "STRENGTH_COST requires coord_x".to_string())?;
            let cy = coord_y.ok_or_else(|| "STRENGTH_COST requires coord_y".to_string())?;
            Ok(sample_strength_cost_coordinates(
                x, y, gamma, cx, cy, family, self_loops, seed,
            ))
        }
        GrandCanonicalCase::DegreeEvents => {
            let pi = positive_intensity
                .ok_or_else(|| "DEGREE_EVENTS requires positive_intensity".to_string())?;
            Ok(sample_model(
                SamplingModel::DegreeEvents {
                    x,
                    y,
                    positive_intensity: pi,
                    family,
                    self_loops,
                },
                seed,
            ))
        }
        GrandCanonicalCase::EdgesEvents => {
            let n = node_count.ok_or_else(|| "EDGES_EVENTS requires node_count".to_string())?;
            let q = q.ok_or_else(|| "EDGES_EVENTS requires q".to_string())?;
            let occ = occupation.ok_or_else(|| "EDGES_EVENTS requires occupation".to_string())?;
            Ok(sample_edges_events(n, q, occ, family, self_loops, seed))
        }
    }
}
