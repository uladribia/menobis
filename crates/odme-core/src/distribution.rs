//! Pair-level null distributions shared by generation and filtering.

use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Bernoulli, Distribution, Poisson};

const ZTP_REJECTION_MIN_RATE: f64 = 0.05;

/// Distribution for a single candidate pair `(i, j)`.
#[derive(Clone, Copy, Debug)]
pub enum PairDistribution {
    /// Independent Poisson edge weight.
    Poisson { rate: f64 },
    /// Bernoulli occupation followed by a zero-truncated Poisson positive weight.
    ZipPoisson { occupation: f64, rate: f64 },
}

impl PairDistribution {
    /// Expected edge weight.
    #[must_use]
    pub fn expected(self) -> f64 {
        match self {
            Self::Poisson { rate } => rate.max(0.0),
            Self::ZipPoisson { occupation, rate } => {
                occupation.max(0.0) * zero_truncated_poisson_mean(rate)
            }
        }
    }

    /// Binary occupation probability `P(t_ij > 0)`.
    #[must_use]
    pub fn occupation_probability(self) -> f64 {
        match self {
            Self::Poisson { rate } => 1.0 - (-rate.max(0.0)).exp(),
            Self::ZipPoisson { occupation, .. } => occupation.clamp(0.0, 1.0),
        }
    }

    /// Inclusive lower-tail probability `P(T <= weight)`.
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

    /// Inclusive upper-tail probability `P(T >= weight)`.
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

    /// Draw one sample from this pair distribution.
    pub fn sample(self, rng: &mut StdRng) -> u64 {
        match self {
            Self::Poisson { rate } => sample_poisson(rate, rng),
            Self::ZipPoisson { occupation, rate } => {
                if occupation <= 0.0 {
                    return 0;
                }
                let present = match Bernoulli::new(occupation.min(1.0)) {
                    Ok(dist) => dist.sample(rng),
                    Err(_) => false,
                };
                if present {
                    sample_zero_truncated_poisson(rate, rng)
                } else {
                    0
                }
            }
        }
    }
}

/// Strength-edges ZIP distribution for one pair.
#[must_use]
pub fn strength_edges_distribution(xi: f64, yj: f64, lam: f64) -> PairDistribution {
    let rate = xi * yj;
    let expm1 = rate.exp_m1();
    let den = 1.0 + lam * expm1;
    let occupation = if den > 0.0 { lam * expm1 / den } else { 0.0 };
    PairDistribution::ZipPoisson { occupation, rate }
}

/// Strength-degree ZIP distribution for one pair.
#[must_use]
pub fn strength_degree_distribution(xi: f64, yj: f64, zi: f64, wj: f64) -> PairDistribution {
    let rate = xi * yj;
    let v = zi * wj;
    let expm1 = rate.exp_m1();
    let den = 1.0 + v * expm1;
    let occupation = if den > 0.0 { v * expm1 / den } else { 0.0 };
    PairDistribution::ZipPoisson { occupation, rate }
}

/// Zero-truncated Poisson mean.
#[must_use]
pub fn zero_truncated_poisson_mean(rate: f64) -> f64 {
    if rate <= 0.0 {
        1.0
    } else {
        rate / (1.0 - (-rate).exp())
    }
}

/// Poisson lower-tail CDF.
#[must_use]
pub fn poisson_cdf(weight: u64, rate: f64) -> f64 {
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

/// Inclusive Poisson survival probability.
#[must_use]
pub fn poisson_sf_inclusive(weight: u64, rate: f64) -> f64 {
    if weight == 0 {
        1.0
    } else {
        (1.0 - poisson_cdf(weight - 1, rate)).clamp(0.0, 1.0)
    }
}

/// Zero-truncated Poisson CDF.
#[must_use]
pub fn zero_truncated_poisson_cdf(weight: u64, rate: f64) -> f64 {
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

/// Inclusive zero-truncated Poisson survival probability.
#[must_use]
pub fn zero_truncated_poisson_sf_inclusive(weight: u64, rate: f64) -> f64 {
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

fn sample_poisson(rate: f64, rng: &mut StdRng) -> u64 {
    if rate <= 0.0 {
        return 0;
    }
    match Poisson::new(rate) {
        Ok(dist) => dist.sample(rng) as u64,
        Err(_) => 0,
    }
}

/// Draw from a zero-truncated Poisson distribution.
pub fn sample_zero_truncated_poisson(rate: f64, rng: &mut StdRng) -> u64 {
    if rate <= 0.0 || !rate.is_finite() {
        return 1;
    }
    if rate < ZTP_REJECTION_MIN_RATE {
        let normalizer = -rate.exp_m1();
        if normalizer <= 0.0 {
            return 1;
        }
        let mut cumulative = 0.0;
        let mut probability = (-rate).exp() * rate / normalizer;
        let draw = rng.random::<f64>();
        let mut value = 1_u64;
        loop {
            cumulative += probability;
            if draw <= cumulative || value >= 64 {
                return value;
            }
            value += 1;
            probability *= rate / value as f64;
        }
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
}
