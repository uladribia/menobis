//! Seeded network generation for node-factorized models.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Bernoulli, Binomial, Distribution, Poisson};

/// Sparse edge output from a generation run.
#[derive(Clone, Debug, Default)]
pub struct SampledEdges {
    pub sources: Vec<u64>,
    pub targets: Vec<u64>,
    pub weights: Vec<u64>,
}

/// Sample custom p_ij grand-canonical Poisson graph with E[t_ij] = T p_ij.
#[must_use]
pub fn sample_custom_pij_poisson(
    sources: &[u64],
    targets: &[u64],
    probabilities: &[f64],
    total_events: u64,
    seed: u64,
) -> SampledEdges {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut result = SampledEdges::default();
    let p_sum: f64 = probabilities.iter().sum();
    if p_sum <= 0.0 {
        return result;
    }
    for ((&source, &target), &probability) in
        sources.iter().zip(targets.iter()).zip(probabilities.iter())
    {
        let rate = total_events as f64 * probability / p_sum;
        if rate <= 0.0 {
            continue;
        }
        let w = match Poisson::new(rate) {
            Ok(dist) => dist.sample(&mut rng) as u64,
            Err(_) => 0,
        };
        if w > 0 {
            result.sources.push(source);
            result.targets.push(target);
            result.weights.push(w);
        }
    }
    result
}

/// Sample custom p_ij canonical multinomial graph with fixed T.
#[must_use]
pub fn sample_custom_pij_multinomial(
    sources: &[u64],
    targets: &[u64],
    probabilities: &[f64],
    total_events: u64,
    seed: u64,
) -> SampledEdges {
    let mut rng = StdRng::seed_from_u64(seed);
    let counts = multinomial_sample(probabilities, total_events, &mut rng);
    let mut result = SampledEdges::default();
    for ((&source, &target), &weight) in sources.iter().zip(targets.iter()).zip(counts.iter()) {
        if weight > 0 {
            result.sources.push(source);
            result.targets.push(target);
            result.weights.push(weight);
        }
    }
    result
}

/// Sample from independent Poisson(x_i * y_j) for all (i, j).
#[must_use]
pub fn sample_poisson(x: &[f64], y: &[f64], self_loops: bool, seed: u64) -> SampledEdges {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut result = SampledEdges::default();

    for (i, &xi) in x.iter().enumerate() {
        if xi == 0.0 {
            continue;
        }
        for (j, &yj) in y.iter().enumerate() {
            if !self_loops && i == j {
                continue;
            }
            let rate = xi * yj;
            if rate <= 0.0 {
                continue;
            }
            let dist = match Poisson::new(rate) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let w = dist.sample(&mut rng) as u64;
            if w > 0 {
                result.sources.push(i as u64);
                result.targets.push(j as u64);
                result.weights.push(w);
            }
        }
    }

    result
}

fn zero_truncated_poisson_mean(rate: f64) -> f64 {
    if rate <= 0.0 {
        return 1.0;
    }
    rate / (1.0 - (-rate).exp())
}

fn solve_zero_truncated_poisson_rate(mean: f64) -> f64 {
    if mean <= 1.0 {
        return 1e-12;
    }
    let mut low = 1e-12;
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

fn sample_zero_truncated_poisson(rate: f64, rng: &mut StdRng) -> u64 {
    if rate <= 0.0 {
        return 1;
    }
    let dist = match Poisson::new(rate) {
        Ok(d) => d,
        Err(_) => return 1,
    };
    loop {
        let value = dist.sample(rng) as u64;
        if value > 0 {
            return value;
        }
    }
}

/// Sample original fixed-degree ME weighted model.
#[must_use]
pub fn sample_fixed_degree_zip(
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
    let rate = solve_zero_truncated_poisson_rate(mean_existing_weight);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut result = SampledEdges::default();
    for (i, &xi) in x.iter().enumerate() {
        for (j, &yj) in y.iter().enumerate() {
            if !self_loops && i == j {
                continue;
            }
            let z = xi * yj;
            let p = z / (1.0 + z);
            if p <= 0.0 {
                continue;
            }
            let present = match Bernoulli::new(p.min(1.0)) {
                Ok(dist) => dist.sample(&mut rng),
                Err(_) => false,
            };
            if present {
                result.sources.push(i as u64);
                result.targets.push(j as u64);
                result
                    .weights
                    .push(sample_zero_truncated_poisson(rate, &mut rng));
            }
        }
    }
    result
}

/// Sample from the exact ME fixed-strength-degree ZIP model.
#[must_use]
pub fn sample_strength_degree_zip(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    w: &[f64],
    self_loops: bool,
    seed: u64,
) -> SampledEdges {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut result = SampledEdges::default();

    for (i, &xi) in x.iter().enumerate() {
        for (j, &yj) in y.iter().enumerate() {
            if !self_loops && i == j {
                continue;
            }
            let u = xi * yj;
            let v = z[i] * w[j];
            let exp_u = u.exp();
            let den = 1.0 + v * (exp_u - 1.0);
            let p = if den > 0.0 {
                v * (exp_u - 1.0) / den
            } else {
                0.0
            };
            if p <= 0.0 {
                continue;
            }
            let present = match Bernoulli::new(p.min(1.0)) {
                Ok(dist) => dist.sample(&mut rng),
                Err(_) => false,
            };
            if !present {
                continue;
            }
            result.sources.push(i as u64);
            result.targets.push(j as u64);
            result
                .weights
                .push(sample_zero_truncated_poisson(u, &mut rng));
        }
    }

    result
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

    for (i, &t_i) in row_events.iter().enumerate() {
        if t_i == 0 {
            continue;
        }
        let mut col_rates: Vec<f64> = y.to_vec();
        if !self_loops {
            col_rates[i] = 0.0;
        }
        let col_events = multinomial_sample(&col_rates, t_i, &mut rng);
        for (j, &cj) in col_events.iter().enumerate().take(n) {
            if cj > 0 {
                result.sources.push(i as u64);
                result.targets.push(j as u64);
                result.weights.push(cj);
            }
        }
    }

    result
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
        let count = if p >= 1.0 {
            remaining
        } else if remaining == 1 {
            u64::from(rng.random::<f64>() < p)
        } else {
            match Binomial::new(remaining, p) {
                Ok(dist) => dist.sample(rng),
                Err(_) => 0,
            }
        };
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
    use super::{sample_multinomial, sample_poisson, sample_strength_degree_zip};

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
        let a = sample_strength_degree_zip(&dx, &dy, &ex, &ey, true, 42);
        let b = sample_strength_degree_zip(&dx, &dy, &ex, &ey, true, 42);
        assert_eq!(a.sources, b.sources);
        assert_eq!(a.targets, b.targets);
        assert_eq!(a.weights, b.weights);
        assert!(a.weights.iter().all(|&w| w > 0));
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
}
