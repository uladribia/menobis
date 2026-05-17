//! Seeded network generation for node-factorized models.

use crate::distribution::{zero_truncated_poisson_mean, PairDistribution};
use crate::pairs::{
    chunk_seed, row_ranges, CandidateSupport, DegreeEventsZipProvider,
    FixedStrengthBinomialProvider, FixedStrengthGeometricProvider,
    FixedStrengthNegBinomialProvider, FixedStrengthPoissonProvider,
    NormalizedSparsePoissonProvider, PairDistributionProvider, StrengthCostPoissonProvider,
    StrengthDegreeZipProvider, StrengthEdgesZipProvider, PARALLEL_PAIR_THRESHOLD,
    SPARSE_CHUNK_SIZE,
};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Binomial, Distribution, Poisson};
use rayon::prelude::*;
use std::collections::HashMap;

const ZTP_MEAN_EPSILON: f64 = 1e-10;

/// Sparse edge output from a generation run.
#[derive(Clone, Debug, Default)]
pub struct SampledEdges {
    pub sources: Vec<u64>,
    pub targets: Vec<u64>,
    pub weights: Vec<u64>,
}

#[derive(Clone, Copy, Debug)]
struct PairDraw {
    source: u64,
    target: u64,
    distribution: PairDistribution,
}

/// Sparse cost entries for strength-cost generation.
pub struct SparseCostEntries<'a> {
    pub sources: &'a [usize],
    pub targets: &'a [usize],
    pub values: &'a [f64],
}

fn merge_samples(chunks: Vec<SampledEdges>) -> SampledEdges {
    let total_edges = chunks.iter().map(|chunk| chunk.sources.len()).sum();
    let mut result = SampledEdges {
        sources: Vec::with_capacity(total_edges),
        targets: Vec::with_capacity(total_edges),
        weights: Vec::with_capacity(total_edges),
    };
    for mut chunk in chunks {
        result.sources.append(&mut chunk.sources);
        result.targets.append(&mut chunk.targets);
        result.weights.append(&mut chunk.weights);
    }
    result
}

fn push_sampled_pair(result: &mut SampledEdges, pair: PairDraw, rng: &mut StdRng) {
    let weight = pair.distribution.sample(rng);
    if weight > 0 {
        result.sources.push(pair.source);
        result.targets.push(pair.target);
        result.weights.push(weight);
    }
}

fn sample_independent_pairs<I>(pairs: I, rng: &mut StdRng) -> SampledEdges
where
    I: IntoIterator<Item = PairDraw>,
{
    let mut result = SampledEdges::default();
    for pair in pairs {
        push_sampled_pair(&mut result, pair, rng);
    }
    result
}

fn sample_provider<P>(provider: &P, seed: u64) -> SampledEdges
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
) -> SampledEdges
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

    let chunks: Vec<SampledEdges> = (0..sources.len())
        .step_by(SPARSE_CHUNK_SIZE)
        .map(|start| (start, (start + SPARSE_CHUNK_SIZE).min(sources.len())))
        .collect::<Vec<_>>()
        .into_par_iter()
        .enumerate()
        .map(|(chunk_index, (start, end))| {
            let mut rng = StdRng::seed_from_u64(chunk_seed(seed, chunk_index));
            let mut result = SampledEdges::default();
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

fn sample_all_pairs_by_rows<F>(n: usize, self_loops: bool, seed: u64, pair_fn: F) -> SampledEdges
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
        let mut result = SampledEdges::default();
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

    let chunks: Vec<SampledEdges> = row_ranges(n)
        .into_par_iter()
        .enumerate()
        .map(|(chunk_index, (start, end))| {
            let mut rng = StdRng::seed_from_u64(chunk_seed(seed, chunk_index));
            let mut result = SampledEdges::default();
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
#[must_use]
pub fn sample_custom_pij_events_poisson(
    sources: &[u64],
    targets: &[u64],
    probabilities: &[f64],
    total_events: u64,
    seed: u64,
) -> SampledEdges {
    let p_sum: f64 = probabilities.iter().sum();
    if p_sum <= 0.0 {
        return SampledEdges::default();
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

/// Sample custom p_ij canonical multinomial graph with fixed T.
#[must_use]
pub fn sample_custom_pij_events_multinomial(
    sources: &[u64],
    targets: &[u64],
    probabilities: &[f64],
    total_events: u64,
    seed: u64,
) -> SampledEdges {
    let mut rng = StdRng::seed_from_u64(seed);
    sparse_multinomial_sample(sources, targets, probabilities, total_events, &mut rng)
}

/// Microcanonical stub-matching sampler for fixed-strength ME with self-loops.
///
/// Creates `s_out[i]` outgoing stubs for each node `i` and `s_in[j]` incoming
/// stubs for each node `j`, then pairs them by random shuffle. This produces
/// an unbiased uniform sample from the space of all integer-weight directed
/// graphs with the exact given strength sequence and self-loops allowed.
///
/// **Important**: this uniform sampling property only holds when self-loops are
/// allowed. Without self-loops the rejection/constraint introduces bias that
/// requires more sophisticated algorithms (e.g., MCMC) to correct.
#[must_use]
pub fn sample_microcanonical(strength_out: &[u64], strength_in: &[u64], seed: u64) -> SampledEdges {
    let n = strength_out.len();
    let total_out: u64 = strength_out.iter().sum();
    let total_in: u64 = strength_in.iter().sum();
    assert_eq!(
        total_out, total_in,
        "microcanonical requires balanced strengths"
    );
    let t = total_out as usize;

    // Build outgoing stubs: node i appears s_out[i] times.
    let mut out_stubs: Vec<u64> = Vec::with_capacity(t);
    for (i, &s) in strength_out.iter().enumerate() {
        for _ in 0..s {
            out_stubs.push(i as u64);
        }
    }

    // Build incoming stubs: node j appears s_in[j] times.
    let mut in_stubs: Vec<u64> = Vec::with_capacity(t);
    for (j, &s) in strength_in.iter().enumerate() {
        for _ in 0..s {
            in_stubs.push(j as u64);
        }
    }

    // Shuffle incoming stubs.
    let mut rng = StdRng::seed_from_u64(seed);
    use rand::seq::SliceRandom;
    in_stubs.shuffle(&mut rng);

    // Count edge weights from stub pairings.
    let mut weight_map = std::collections::HashMap::new();
    for (&src, &tgt) in out_stubs.iter().zip(in_stubs.iter()) {
        *weight_map.entry((src, tgt)).or_insert(0u64) += 1;
    }

    let mut result = SampledEdges::default();
    let mut pairs: Vec<_> = weight_map.into_iter().collect();
    pairs.sort_unstable();
    for ((src, tgt), w) in pairs {
        result.sources.push(src);
        result.targets.push(tgt);
        result.weights.push(w);
    }
    let _ = n; // used only in assert context
    result
}

/// Sample from independent Poisson(x_i * y_j) for all (i, j).
#[must_use]
pub fn sample_poisson(x: &[f64], y: &[f64], self_loops: bool, seed: u64) -> SampledEdges {
    sample_provider(&FixedStrengthPoissonProvider { x, y, self_loops }, seed)
}

/// Sample from independent Geometric(1 - x_i*y_j) for all (i, j).
#[must_use]
pub fn sample_geometric(x: &[f64], y: &[f64], self_loops: bool, seed: u64) -> SampledEdges {
    sample_provider(&FixedStrengthGeometricProvider { x, y, self_loops }, seed)
}

/// Sample from independent Binomial(M, x_i*y_j/(1+x_i*y_j)) for all (i, j).
#[must_use]
pub fn sample_binomial(
    x: &[f64],
    y: &[f64],
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> SampledEdges {
    sample_provider(
        &FixedStrengthBinomialProvider {
            x,
            y,
            layers,
            self_loops,
        },
        seed,
    )
}

/// Sample from independent NegBinomial(M, 1-x_i*y_j) for all (i, j).
#[must_use]
pub fn sample_neg_binomial(
    x: &[f64],
    y: &[f64],
    layers: u32,
    self_loops: bool,
    seed: u64,
) -> SampledEdges {
    sample_provider(
        &FixedStrengthNegBinomialProvider {
            x,
            y,
            layers,
            self_loops,
        },
        seed,
    )
}

/// Sample from independent Poisson(x_i * y_j * exp(-gamma d_ij)).
///
/// Cost entries are read sparsely. Missing pairs have d_ij = 0, matching the
/// fitting API's current semantics, and rates are generated on the fly.
#[must_use]
pub fn sample_strength_cost_me(
    x: &[f64],
    y: &[f64],
    gamma: f64,
    costs: &SparseCostEntries<'_>,
    self_loops: bool,
    seed: u64,
) -> SampledEdges {
    let cost_map: HashMap<(usize, usize), f64> = costs
        .sources
        .iter()
        .zip(costs.targets.iter())
        .zip(costs.values.iter())
        .map(|((&source, &target), &cost)| ((source, target), cost))
        .collect();
    sample_provider(
        &StrengthCostPoissonProvider {
            x,
            y,
            gamma,
            costs: &cost_map,
            self_loops,
        },
        seed,
    )
}

fn solve_zero_truncated_poisson_rate(mean: f64) -> f64 {
    if mean < 1.0 - ZTP_MEAN_EPSILON {
        return f64::NAN;
    }
    if mean <= 1.0 + ZTP_MEAN_EPSILON {
        return 0.0;
    }
    let mut low = 0.0;
    let mut high = mean.max(1.0);
    while zero_truncated_poisson_mean(high) < mean {
        high *= 2.0;
    }
    for _ in 0..100 {
        let mid = 0.5 * (low + high);
        if zero_truncated_poisson_mean(mid) < mean {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

/// Sample original fixed-degree ME weighted model.
#[must_use]
pub fn sample_fixed_degree_events_me(
    x: &[f64],
    y: &[f64],
    total_events: u64,
    self_loops: bool,
    seed: u64,
) -> SampledEdges {
    let mut expected_edges = 0.0;
    for (i, &xi) in x.iter().enumerate() {
        for (j, &yj) in y.iter().enumerate() {
            if !self_loops && i == j {
                continue;
            }
            let z = xi * yj;
            expected_edges += z / (1.0 + z);
        }
    }
    let mean_existing_weight = if expected_edges > 0.0 {
        total_events as f64 / expected_edges
    } else {
        1.0
    };
    let rate = solve_zero_truncated_poisson_rate(mean_existing_weight.max(1.0));
    sample_provider(
        &DegreeEventsZipProvider {
            x,
            y,
            positive_weight_rate: rate,
            self_loops,
        },
        seed,
    )
}

/// Sample from the exact ME fixed-strength-degree ZIP model.
#[must_use]
pub fn sample_strength_degree_me(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    w: &[f64],
    self_loops: bool,
    seed: u64,
) -> SampledEdges {
    sample_provider(
        &StrengthDegreeZipProvider {
            x,
            y,
            z,
            w,
            self_loops,
        },
        seed,
    )
}

/// Poisson-total multinomial sampling with node-factorized probabilities.
#[must_use]
pub fn sample_poisson_multinomial(
    x: &[f64],
    y: &[f64],
    self_loops: bool,
    seed: u64,
) -> SampledEdges {
    let mut total_rate = 0.0;
    for (i, &xi) in x.iter().enumerate() {
        for (j, &yj) in y.iter().enumerate() {
            if !self_loops && i == j {
                continue;
            }
            total_rate += xi * yj;
        }
    }
    if total_rate <= 0.0 {
        return SampledEdges::default();
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let total_events = match Poisson::new(total_rate) {
        Ok(dist) => dist.sample(&mut rng) as u64,
        Err(_) => 0,
    };
    sample_multinomial(x, y, total_events, self_loops, seed.wrapping_add(1))
}

/// Sample exact ME fixed-strength-and-edge-count ZIP model.
#[must_use]
pub fn sample_strength_edges_me(
    x: &[f64],
    y: &[f64],
    lam: f64,
    self_loops: bool,
    seed: u64,
) -> SampledEdges {
    sample_provider(
        &StrengthEdgesZipProvider {
            x,
            y,
            lambda: lam,
            self_loops,
        },
        seed,
    )
}

/// Multinomial sampling with node-factorized probabilities.
#[must_use]
pub fn sample_multinomial(
    x: &[f64],
    y: &[f64],
    total_events: u64,
    self_loops: bool,
    seed: u64,
) -> SampledEdges {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = x.len();
    let y_sum: f64 = y.iter().sum();
    let mut result = SampledEdges::default();

    let row_rates: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            if self_loops {
                xi * y_sum
            } else {
                xi * (y_sum - y[i])
            }
        })
        .collect();
    let total_rate: f64 = row_rates.iter().sum();
    if total_rate == 0.0 {
        return result;
    }

    let row_events = multinomial_sample(&row_rates, total_events, &mut rng);
    let non_empty_rows = row_events.iter().filter(|&&events| events > 0).count();
    if n.saturating_mul(non_empty_rows) < PARALLEL_PAIR_THRESHOLD {
        for (i, &t_i) in row_events.iter().enumerate() {
            append_multinomial_row(&mut result, i, t_i, y, self_loops, &mut rng);
        }
        return result;
    }

    let chunks: Vec<SampledEdges> = row_events
        .par_iter()
        .enumerate()
        .map(|(i, &t_i)| {
            let mut local = SampledEdges::default();
            let mut row_rng = StdRng::seed_from_u64(chunk_seed(seed, i));
            append_multinomial_row(&mut local, i, t_i, y, self_loops, &mut row_rng);
            local
        })
        .collect();
    merge_samples(chunks)
}

fn append_multinomial_row(
    result: &mut SampledEdges,
    i: usize,
    total_events: u64,
    y: &[f64],
    self_loops: bool,
    rng: &mut StdRng,
) {
    if total_events == 0 {
        return;
    }
    let mut col_rates: Vec<f64> = y.to_vec();
    if !self_loops {
        col_rates[i] = 0.0;
    }
    let col_events = multinomial_sample(&col_rates, total_events, rng);
    for (j, &count) in col_events.iter().enumerate() {
        if count > 0 {
            result.sources.push(i as u64);
            result.targets.push(j as u64);
            result.weights.push(count);
        }
    }
}

fn sparse_multinomial_sample(
    sources: &[u64],
    targets: &[u64],
    rates: &[f64],
    total: u64,
    rng: &mut StdRng,
) -> SampledEdges {
    if rates.len() < SPARSE_CHUNK_SIZE || total == 0 {
        return sparse_multinomial_sample_serial(sources, targets, rates, total, rng);
    }
    let ranges: Vec<(usize, usize)> = (0..rates.len())
        .step_by(SPARSE_CHUNK_SIZE)
        .map(|start| (start, (start + SPARSE_CHUNK_SIZE).min(rates.len())))
        .collect();
    let chunk_rates: Vec<f64> = ranges
        .iter()
        .map(|&(start, end)| rates[start..end].iter().sum())
        .collect();
    let chunk_events = multinomial_sample(&chunk_rates, total, rng);
    let base_seed = rng.random::<u64>();
    let chunks: Vec<SampledEdges> = ranges
        .into_par_iter()
        .zip(chunk_events.into_par_iter())
        .enumerate()
        .map(|(chunk_index, ((start, end), events))| {
            let mut local_rng = StdRng::seed_from_u64(chunk_seed(base_seed, chunk_index));
            sparse_multinomial_sample_serial(
                &sources[start..end],
                &targets[start..end],
                &rates[start..end],
                events,
                &mut local_rng,
            )
        })
        .collect();
    merge_samples(chunks)
}

fn sparse_multinomial_sample_serial(
    sources: &[u64],
    targets: &[u64],
    rates: &[f64],
    total: u64,
    rng: &mut StdRng,
) -> SampledEdges {
    let mut result = SampledEdges::default();
    let rate_sum: f64 = rates.iter().sum();
    if rate_sum == 0.0 || total == 0 {
        return result;
    }

    let mut remaining = total;
    let mut remaining_rate = rate_sum;
    let mut last_positive: Option<(u64, u64)> = None;

    for ((&source, &target), &rate) in sources.iter().zip(targets.iter()).zip(rates.iter()) {
        if rate > 0.0 {
            last_positive = Some((source, target));
        }
        if remaining == 0 || remaining_rate <= 0.0 {
            break;
        }
        let p = (rate / remaining_rate).min(1.0);
        if p <= 0.0 {
            remaining_rate -= rate;
            continue;
        }
        let count = draw_binomial_prefix_count(remaining, p, rng);
        if count > 0 {
            result.sources.push(source);
            result.targets.push(target);
            result.weights.push(count);
        }
        remaining -= count;
        remaining_rate -= rate;
    }

    if remaining > 0 {
        if let Some((source, target)) = last_positive {
            result.sources.push(source);
            result.targets.push(target);
            result.weights.push(remaining);
        }
    }
    result
}

fn draw_binomial_prefix_count(remaining: u64, p: f64, rng: &mut StdRng) -> u64 {
    if p >= 1.0 {
        remaining
    } else if remaining == 1 {
        u64::from(rng.random::<f64>() < p)
    } else {
        match Binomial::new(remaining, p) {
            Ok(dist) => dist.sample(rng),
            Err(_) => 0,
        }
    }
}

fn multinomial_sample(rates: &[f64], total: u64, rng: &mut StdRng) -> Vec<u64> {
    let n = rates.len();
    let mut result = vec![0_u64; n];
    let rate_sum: f64 = rates.iter().sum();
    if rate_sum == 0.0 || total == 0 {
        return result;
    }

    let mut remaining = total;
    let mut remaining_rate = rate_sum;

    for i in 0..n {
        if remaining == 0 || remaining_rate <= 0.0 {
            break;
        }
        let p = (rates[i] / remaining_rate).min(1.0);
        if p <= 0.0 {
            remaining_rate -= rates[i];
            continue;
        }
        let count = draw_binomial_prefix_count(remaining, p, rng);
        result[i] = count;
        remaining -= count;
        remaining_rate -= rates[i];
    }
    if remaining > 0 {
        for i in (0..n).rev() {
            if rates[i] > 0.0 {
                result[i] += remaining;
                break;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::{
        sample_microcanonical, sample_multinomial, sample_poisson, sample_strength_cost_me,
        sample_strength_degree_me,
    };

    #[test]
    fn poisson_is_reproducible() {
        let x = vec![3.0, 5.0];
        let y = vec![4.0, 6.0];
        let a = sample_poisson(&x, &y, true, 42);
        let b = sample_poisson(&x, &y, true, 42);
        assert_eq!(a.sources, b.sources);
        assert_eq!(a.targets, b.targets);
        assert_eq!(a.weights, b.weights);
    }

    #[test]
    fn multinomial_preserves_total() {
        let x = vec![3.0, 5.0, 2.0];
        let y = vec![4.0, 6.0, 1.0];
        let total = 1000;
        let edges = sample_multinomial(&x, &y, total, true, 42);
        let sum: u64 = edges.weights.iter().sum();
        assert_eq!(sum, total);
    }

    #[test]
    fn zip_is_reproducible() {
        let dx = vec![1.0, 2.0];
        let dy = vec![1.5, 0.5];
        let ex = vec![10.0, 20.0];
        let ey = vec![30.0, 40.0];
        let a = sample_strength_degree_me(&dx, &dy, &ex, &ey, true, 42);
        let b = sample_strength_degree_me(&dx, &dy, &ex, &ey, true, 42);
        assert_eq!(a.sources, b.sources);
        assert_eq!(a.targets, b.targets);
        assert_eq!(a.weights, b.weights);
        assert!(a.weights.iter().all(|&w| w > 0));
    }

    #[test]
    fn microcanonical_preserves_exact_strengths() {
        let s_out = vec![10, 20, 30];
        let s_in = vec![15, 25, 20];
        let edges = sample_microcanonical(&s_out, &s_in, 42);
        let total: u64 = edges.weights.iter().sum();
        assert_eq!(total, 60);
        let mut actual_out = vec![0u64; 3];
        let mut actual_in = vec![0u64; 3];
        for ((&src, &tgt), &w) in edges
            .sources
            .iter()
            .zip(edges.targets.iter())
            .zip(edges.weights.iter())
        {
            actual_out[src as usize] += w;
            actual_in[tgt as usize] += w;
        }
        assert_eq!(actual_out, s_out);
        assert_eq!(actual_in, s_in);
    }

    #[test]
    fn no_self_loops() {
        let x = vec![10.0, 10.0, 10.0];
        let y = vec![10.0, 10.0, 10.0];
        let edges = sample_poisson(&x, &y, false, 42);
        for (s, t) in edges.sources.iter().zip(edges.targets.iter()) {
            assert_ne!(s, t, "self-loop found: {s} -> {t}");
        }
    }

    #[test]
    fn zero_truncated_poisson_rate_handles_boundary_mean() {
        assert_eq!(super::solve_zero_truncated_poisson_rate(1.0), 0.0);
        assert_eq!(
            super::solve_zero_truncated_poisson_rate(1.0 + super::ZTP_MEAN_EPSILON / 2.0),
            0.0,
        );
        assert!(super::solve_zero_truncated_poisson_rate(1.01).is_finite());
        assert!(super::solve_zero_truncated_poisson_rate(0.99).is_nan());
    }

    #[test]
    fn degree_events_with_unit_positive_weight_mean_does_not_hang() {
        let x = vec![0.5; 100];
        let y = vec![0.5; 100];
        let edges = super::sample_fixed_degree_events_me(&x, &y, 2000, true, 42);
        assert!(edges.weights.iter().all(|&w| w >= 1));
    }

    #[test]
    fn strength_cost_sampler_streams_large_factorized_model() {
        let n = 1000;
        let x = vec![0.0005; n];
        let y = vec![0.0005; n];
        let costs = super::SparseCostEntries {
            sources: &[],
            targets: &[],
            values: &[],
        };
        let edges = sample_strength_cost_me(&x, &y, 0.1, &costs, true, 7);
        assert_eq!(edges.sources.len(), edges.targets.len());
        assert_eq!(edges.sources.len(), edges.weights.len());
    }
}
