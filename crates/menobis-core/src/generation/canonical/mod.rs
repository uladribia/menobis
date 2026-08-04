//! Canonical generation: fixed-total multinomial sampling.
//!
//! At least one global integer quantity (total events) is exact while pair
//! occupations remain conditionally multinomial.

use crate::pairs::{chunk_seed, PARALLEL_PAIR_THRESHOLD, SPARSE_CHUNK_SIZE};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Binomial, Distribution};
use rayon::prelude::*;

use super::output::{merge_samples, SampledNetwork};

pub fn sample_custom_multinomial(
    sources: &[u64],
    targets: &[u64],
    probabilities: &[f64],
    total_events: u64,
    seed: u64,
) -> SampledNetwork {
    let mut rng = StdRng::seed_from_u64(seed);
    sparse_multinomial_sample(sources, targets, probabilities, total_events, &mut rng)
}

/// Exact-strength stub-matching sampler for fixed-strength ME with self-loops.
///
/// Creates `s_out[i]` outgoing stubs for each node `i` and `s_in[j]` incoming
/// stubs for each node `j`, then pairs them by random shuffle. This produces
/// an unbiased uniform sample from the space of all integer-occupation directed
/// graphs with the exact given strength sequence and self-loops allowed.
///
/// **Important**: this uniform sampling property only holds when self-loops are
/// allowed. Without self-loops the rejection/constraint introduces bias that
/// requires more sophisticated algorithms (e.g., MCMC) to correct.
pub fn sample_strength_multinomial(
    x: &[f64],
    y: &[f64],
    total_events: u64,
    self_loops: bool,
    seed: u64,
) -> SampledNetwork {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = x.len();
    let y_sum: f64 = y.iter().sum();
    let mut result = SampledNetwork::default();

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

    let chunks: Vec<SampledNetwork> = row_events
        .par_iter()
        .enumerate()
        .map(|(i, &t_i)| {
            let mut local = SampledNetwork::default();
            let mut row_rng = StdRng::seed_from_u64(chunk_seed(seed, i));
            append_multinomial_row(&mut local, i, t_i, y, self_loops, &mut row_rng);
            local
        })
        .collect();
    merge_samples(chunks)
}

fn append_multinomial_row(
    result: &mut SampledNetwork,
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
            result.occ_nums.push(count);
        }
    }
}

fn sparse_multinomial_sample(
    sources: &[u64],
    targets: &[u64],
    rates: &[f64],
    total: u64,
    rng: &mut StdRng,
) -> SampledNetwork {
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
    let chunks: Vec<SampledNetwork> = ranges
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
) -> SampledNetwork {
    let mut result = SampledNetwork::default();
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
            result.occ_nums.push(count);
        }
        remaining -= count;
        remaining_rate -= rate;
    }

    if remaining > 0 {
        if let Some((source, target)) = last_positive {
            result.sources.push(source);
            result.targets.push(target);
            result.occ_nums.push(remaining);
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
