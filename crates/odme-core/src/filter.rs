//! Statistical filtering kernels for fitted ODME null models.

use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

const PARALLEL_PAIR_THRESHOLD: usize = 1_000_000;
const ROW_CHUNK_SIZE: usize = 128;

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

#[derive(Clone, Copy, Debug)]
pub enum NullDistribution {
    Poisson { rate: f64 },
    ZipPoisson { occupation: f64, rate: f64 },
}

impl NullDistribution {
    #[must_use]
    pub fn expected(self) -> f64 {
        match self {
            Self::Poisson { rate } => rate.max(0.0),
            Self::ZipPoisson { occupation, rate } => {
                occupation.max(0.0) * zero_truncated_poisson_mean(rate)
            }
        }
    }

    #[must_use]
    pub fn occupation(self) -> f64 {
        match self {
            Self::Poisson { rate } => 1.0 - (-rate.max(0.0)).exp(),
            Self::ZipPoisson { occupation, .. } => occupation.clamp(0.0, 1.0),
        }
    }

    #[must_use]
    pub fn lower_pvalue(self, weight: u64) -> f64 {
        match self {
            Self::Poisson { rate } => poisson_cdf(weight, rate),
            Self::ZipPoisson { occupation, rate } => {
                let p = occupation.clamp(0.0, 1.0);
                if weight == 0 {
                    1.0 - p
                } else {
                    (1.0 - p) + p * zero_truncated_poisson_cdf(weight, rate)
                }
            }
        }
        .clamp(0.0, 1.0)
    }

    #[must_use]
    pub fn upper_pvalue(self, weight: u64) -> f64 {
        match self {
            Self::Poisson { rate } => poisson_sf_inclusive(weight, rate),
            Self::ZipPoisson { occupation, rate } => {
                if weight == 0 {
                    1.0
                } else {
                    occupation.clamp(0.0, 1.0) * zero_truncated_poisson_sf_inclusive(weight, rate)
                }
            }
        }
        .clamp(0.0, 1.0)
    }
}

#[must_use]
pub fn filter_observed_with<F>(
    sources: &[u64],
    targets: &[u64],
    weights: &[u64],
    provider: F,
) -> ObservedFilterResult
where
    F: Fn(usize, usize) -> NullDistribution + Sync,
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
                dist.occupation(),
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

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn detect_absent_with<F>(
    n: usize,
    observed_sources: &[u64],
    observed_targets: &[u64],
    self_loops: bool,
    alpha_lower: f64,
    min_occupation: f64,
    min_expected: f64,
    max_absent: Option<usize>,
    provider: F,
) -> AbsentFilterResult
where
    F: Fn(usize, usize) -> NullDistribution + Sync,
{
    let observed: HashSet<(usize, usize)> = observed_sources
        .iter()
        .zip(observed_targets.iter())
        .map(|(&source, &target)| (source as usize, target as usize))
        .collect();
    let candidate_pairs = if self_loops {
        n.saturating_mul(n)
    } else {
        n.saturating_mul(n.saturating_sub(1))
    };

    let scan_range = |start: usize, end: usize| {
        let mut local = AbsentFilterResult::default();
        for i in start..end {
            for j in 0..n {
                if (!self_loops && i == j) || observed.contains(&(i, j)) {
                    continue;
                }
                let dist = provider(i, j);
                let occupation = dist.occupation();
                let expected = dist.expected();
                if occupation < min_occupation || expected < min_expected {
                    continue;
                }
                let lower = dist.lower_pvalue(0);
                if lower < alpha_lower {
                    local.sources.push(i as u64);
                    local.targets.push(j as u64);
                    local.lower_pvalues.push(lower);
                    local.expected.push(expected);
                    local.occupation.push(occupation);
                }
            }
        }
        local
    };

    let mut result = if candidate_pairs < PARALLEL_PAIR_THRESHOLD {
        scan_range(0, n)
    } else {
        let chunks: Vec<AbsentFilterResult> = (0..n)
            .step_by(ROW_CHUNK_SIZE)
            .map(|start| (start, (start + ROW_CHUNK_SIZE).min(n)))
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(start, end)| scan_range(start, end))
            .collect();
        merge_absent(chunks)
    };

    if let Some(limit) = max_absent {
        truncate_absent(&mut result, limit);
    }
    result
}

#[must_use]
pub fn filter_fixed_strength_poisson(
    x: &[f64],
    y: &[f64],
    sources: &[u64],
    targets: &[u64],
    weights: &[u64],
) -> ObservedFilterResult {
    filter_observed_with(sources, targets, weights, |i, j| {
        NullDistribution::Poisson { rate: x[i] * y[j] }
    })
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn absent_fixed_strength_poisson(
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
    detect_absent_with(
        x.len(),
        sources,
        targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
        |i, j| NullDistribution::Poisson { rate: x[i] * y[j] },
    )
}

#[must_use]
pub fn filter_custom_poisson_rates(
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
    filter_observed_with(sources, targets, weights, |i, j| {
        NullDistribution::Poisson {
            rate: rate_map.get(&(i, j)).copied().unwrap_or(0.0),
        }
    })
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn absent_custom_poisson_rates(
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
    let observed: HashSet<(u64, u64)> = observed_sources
        .iter()
        .zip(observed_targets.iter())
        .map(|(&source, &target)| (source, target))
        .collect();
    let mut result = AbsentFilterResult::default();
    for ((&source, &target), &rate) in rate_sources.iter().zip(rate_targets.iter()).zip(rates) {
        if observed.contains(&(source, target)) {
            continue;
        }
        let dist = NullDistribution::Poisson { rate };
        let occupation = dist.occupation();
        let expected = dist.expected();
        if occupation < min_occupation || expected < min_expected {
            continue;
        }
        let lower = dist.lower_pvalue(0);
        if lower < alpha_lower {
            result.sources.push(source);
            result.targets.push(target);
            result.lower_pvalues.push(lower);
            result.expected.push(expected);
            result.occupation.push(occupation);
            if max_absent.is_some_and(|limit| result.sources.len() >= limit) {
                break;
            }
        }
    }
    result
}

#[must_use]
pub fn filter_strength_edges_zip(
    x: &[f64],
    y: &[f64],
    lam: f64,
    sources: &[u64],
    targets: &[u64],
    weights: &[u64],
) -> ObservedFilterResult {
    filter_observed_with(sources, targets, weights, |i, j| {
        strength_edges_distribution(x[i], y[j], lam)
    })
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn absent_strength_edges_zip(
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
    detect_absent_with(
        x.len(),
        sources,
        targets,
        self_loops,
        alpha_lower,
        min_occupation,
        min_expected,
        max_absent,
        |i, j| strength_edges_distribution(x[i], y[j], lam),
    )
}

#[must_use]
pub fn strength_edges_distribution(xi: f64, yj: f64, lam: f64) -> NullDistribution {
    let rate = xi * yj;
    let expm1 = rate.exp_m1();
    let den = 1.0 + lam * expm1;
    let occupation = if den > 0.0 { lam * expm1 / den } else { 0.0 };
    NullDistribution::ZipPoisson { occupation, rate }
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

fn zero_truncated_poisson_mean(rate: f64) -> f64 {
    if rate <= 0.0 {
        1.0
    } else {
        rate / (1.0 - (-rate).exp())
    }
}

fn zero_truncated_poisson_cdf(weight: u64, rate: f64) -> f64 {
    if weight == 0 {
        return 0.0;
    }
    if rate <= 0.0 {
        return 1.0;
    }
    let numerator = poisson_cdf(weight, rate) - (-rate).exp();
    let denominator = 1.0 - (-rate).exp();
    (numerator / denominator).clamp(0.0, 1.0)
}

fn zero_truncated_poisson_sf_inclusive(weight: u64, rate: f64) -> f64 {
    if weight <= 1 && rate <= 0.0 {
        return 1.0;
    }
    if rate <= 0.0 {
        return 0.0;
    }
    let numerator = poisson_sf_inclusive(weight, rate);
    let denominator = 1.0 - (-rate).exp();
    (numerator / denominator).clamp(0.0, 1.0)
}

fn poisson_cdf(weight: u64, rate: f64) -> f64 {
    if rate <= 0.0 {
        return 1.0;
    }
    let mut term = (-rate).exp();
    let mut sum = term;
    for k in 1..=weight {
        term *= rate / k as f64;
        sum += term;
        if term == 0.0 {
            break;
        }
    }
    sum.clamp(0.0, 1.0)
}

fn poisson_sf_inclusive(weight: u64, rate: f64) -> f64 {
    if weight == 0 {
        1.0
    } else {
        (1.0 - poisson_cdf(weight - 1, rate)).clamp(0.0, 1.0)
    }
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
        let dist = NullDistribution::Poisson { rate: 2.0 };
        assert!((dist.lower_pvalue(0) - (-2.0_f64).exp()).abs() < 1e-12);
        assert!((dist.upper_pvalue(0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zip_zero_probability_is_one_minus_occupation() {
        let dist = NullDistribution::ZipPoisson {
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
        let absent = absent_fixed_strength_poisson(&x, &y, &[0], &[1], true, 0.9, 0.5, 0.0, None);
        assert!(absent
            .sources
            .iter()
            .zip(absent.targets.iter())
            .all(|pair| pair != (&0, &1)));
    }

    #[test]
    fn benjamini_hochberg_rejects_prefix() {
        let rejected = benjamini_hochberg(&[0.001, 0.02, 0.5], 0.05);
        assert_eq!(rejected, vec![true, true, false]);
    }
}
