//! Seeded network generation for node-factorized models.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Binomial, Distribution, Poisson};

/// Sparse edge output from a generation run.
#[derive(Clone, Debug, Default)]
pub struct SampledEdges {
    pub sources: Vec<u64>,
    pub targets: Vec<u64>,
    pub weights: Vec<u64>,
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
    use super::{sample_multinomial, sample_poisson};

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
    fn no_self_loops() {
        let x = vec![10.0, 10.0, 10.0];
        let y = vec![10.0, 10.0, 10.0];
        let edges = sample_poisson(&x, &y, false, 42);
        for (s, t) in edges.sources.iter().zip(edges.targets.iter()) {
            assert_ne!(s, t, "self-loop found: {s} -> {t}");
        }
    }
}
