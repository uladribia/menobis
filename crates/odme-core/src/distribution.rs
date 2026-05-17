//! Pair-level null distributions shared by generation and filtering.

use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Bernoulli, Binomial, Distribution, Geometric, Poisson};

const ZTP_REJECTION_MIN_RATE: f64 = 0.05;

/// Distribution for a single candidate pair `(i, j)`.
#[derive(Clone, Copy, Debug)]
pub enum PairDistribution {
    /// Independent Poisson edge weight. E[t] = rate.
    Poisson { rate: f64 },
    /// Bernoulli occupation + zero-truncated Poisson positive weight.
    ZipPoisson { occupation: f64, rate: f64 },
    /// Geometric with p_success = 1 - xy. Support {0,1,...}. E[t] = xy/(1-xy).
    GeometricDist { xy: f64 },
    /// Binomial(M, xy/(1+xy)). Support {0,...,M}. E[t] = M*xy/(1+xy).
    BinomialDist { xy: f64, layers: u32 },
    /// NegBinomial(M, 1-xy). Support {0,1,...}. E[t] = M*xy/(1-xy).
    NegBinomialDist { xy: f64, layers: u32 },
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
            Self::GeometricDist { xy } => {
                let xy = xy.max(0.0);
                if xy >= 1.0 {
                    f64::INFINITY
                } else {
                    xy / (1.0 - xy)
                }
            }
            Self::BinomialDist { xy, layers } => {
                f64::from(layers) * (xy / (1.0 + xy)).clamp(0.0, 1.0)
            }
            Self::NegBinomialDist { xy, layers } => {
                let xy = xy.max(0.0);
                if xy >= 1.0 {
                    f64::INFINITY
                } else {
                    f64::from(layers) * xy / (1.0 - xy)
                }
            }
        }
    }

    /// Binary occupation probability `P(t_ij > 0)`.
    #[must_use]
    pub fn occupation_probability(self) -> f64 {
        match self {
            Self::Poisson { rate } => 1.0 - (-rate.max(0.0)).exp(),
            Self::ZipPoisson { occupation, .. } => occupation.clamp(0.0, 1.0),
            Self::GeometricDist { xy } => xy.clamp(0.0, 1.0),
            Self::BinomialDist { xy, layers } => {
                let p = (xy / (1.0 + xy)).clamp(0.0, 1.0);
                1.0 - (1.0 - p).powi(layers as i32)
            }
            Self::NegBinomialDist { xy, layers } => {
                let p_zero = (1.0 - xy.max(0.0)).max(0.0).powi(layers as i32);
                1.0 - p_zero
            }
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
            Self::GeometricDist { xy } => geometric_cdf(weight, xy),
            Self::BinomialDist { xy, layers } => binomial_cdf(weight, xy / (1.0 + xy), layers),
            Self::NegBinomialDist { xy, layers } => neg_binomial_cdf(weight, xy, layers),
        }
        .clamp(0.0, 1.0)
    }

    /// Inclusive upper-tail probability `P(T >= weight)`.
    #[must_use]
    pub fn upper_pvalue(self, weight: u64) -> f64 {
        if weight == 0 {
            return 1.0;
        }
        match self {
            Self::Poisson { rate } => poisson_sf_inclusive(weight, rate),
            Self::ZipPoisson { occupation, rate } => {
                occupation.clamp(0.0, 1.0) * zero_truncated_poisson_sf_inclusive(weight, rate)
            }
            Self::GeometricDist { xy } => (1.0 - geometric_cdf(weight - 1, xy)).clamp(0.0, 1.0),
            Self::BinomialDist { xy, layers } => {
                (1.0 - binomial_cdf(weight - 1, xy / (1.0 + xy), layers)).clamp(0.0, 1.0)
            }
            Self::NegBinomialDist { xy, layers } => {
                (1.0 - neg_binomial_cdf(weight - 1, xy, layers)).clamp(0.0, 1.0)
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
            Self::GeometricDist { xy } => sample_geometric(xy, rng),
            Self::BinomialDist { xy, layers } => sample_binomial(xy, layers, rng),
            Self::NegBinomialDist { xy, layers } => sample_neg_binomial(xy, layers, rng),
        }
    }
}

// ---------------------------------------------------------------------------
// Poisson helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Geometric helpers: P(T=k) = xy^k * (1-xy) for k=0,1,... (shifted by -1
// from the standard geometric; matches gsl_ran_geometric(p)-1 convention).
// ---------------------------------------------------------------------------

/// Geometric CDF: P(T <= k) = 1 - xy^(k+1).
fn geometric_cdf(weight: u64, xy: f64) -> f64 {
    let xy = xy.clamp(0.0, 1.0);
    if xy <= 0.0 {
        return 1.0;
    }
    (1.0 - xy.powi((weight + 1) as i32)).clamp(0.0, 1.0)
}

fn sample_geometric(xy: f64, rng: &mut StdRng) -> u64 {
    let p = (1.0 - xy).clamp(1e-15, 1.0);
    match Geometric::new(p) {
        Ok(dist) => dist.sample(rng), // rand_distr Geometric is 0-based {0,1,...}
        Err(_) => 0,
    }
}

// ---------------------------------------------------------------------------
// Binomial helpers: Bin(M, p) where p = xy/(1+xy).
// ---------------------------------------------------------------------------

/// Binomial CDF by direct summation.
fn binomial_cdf(weight: u64, p: f64, layers: u32) -> f64 {
    let p = p.clamp(0.0, 1.0);
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return if weight >= u64::from(layers) {
            1.0
        } else {
            0.0
        };
    }
    let n = layers;
    let mut sum = 0.0;
    let mut log_binom = 0.0_f64; // log C(n, k)
    for k in 0..=weight.min(u64::from(n)) {
        let log_pmf =
            log_binom + (k as f64) * p.ln() + ((u64::from(n) - k) as f64) * (1.0 - p).ln();
        sum += log_pmf.exp();
        if k < u64::from(n) {
            log_binom += ((u64::from(n) - k) as f64).ln() - ((k + 1) as f64).ln();
        }
    }
    sum.clamp(0.0, 1.0)
}

fn sample_binomial(xy: f64, layers: u32, rng: &mut StdRng) -> u64 {
    let p = (xy / (1.0 + xy)).clamp(0.0, 1.0);
    match Binomial::new(u64::from(layers), p) {
        Ok(dist) => dist.sample(rng),
        Err(_) => 0,
    }
}

// ---------------------------------------------------------------------------
// Negative binomial helpers: NB(M, 1-xy). E[T] = M*xy/(1-xy).
// GSL convention: gsl_ran_negative_binomial(p, n) has E = n*(1-p)/p.
// With p = 1-xy, E = n*xy/(1-xy). Same as our formula with n = layers.
// ---------------------------------------------------------------------------

/// Negative binomial CDF by direct summation.
fn neg_binomial_cdf(weight: u64, xy: f64, layers: u32) -> f64 {
    let xy = xy.clamp(0.0, 1.0 - 1e-15);
    let p_success = 1.0 - xy; // probability of "success" (stop)
    if p_success >= 1.0 || xy <= 0.0 {
        return 1.0; // all mass at 0
    }
    let r = f64::from(layers);
    let mut sum = 0.0;
    let mut log_coeff = 0.0_f64; // log C(k+r-1, k)
    let log_p = p_success.ln();
    let log_q = xy.ln();
    for k in 0..=weight {
        let log_pmf = log_coeff + r * log_p + (k as f64) * log_q;
        sum += log_pmf.exp();
        if sum >= 1.0 {
            return 1.0;
        }
        // update: C(k+r, k+1) = C(k+r-1, k) * (k+r) / (k+1)
        log_coeff += ((k as f64) + r).ln() - ((k + 1) as f64).ln();
    }
    sum.clamp(0.0, 1.0)
}

fn sample_neg_binomial(xy: f64, layers: u32, rng: &mut StdRng) -> u64 {
    // NB(r, p) where p = 1-xy: sample r successes, count failures.
    // Use gamma-Poisson mixture: Gamma(r, xy/(1-xy)) then Poisson(lambda).
    let xy = xy.clamp(0.0, 1.0 - 1e-15);
    if xy <= 0.0 {
        return 0;
    }
    let r = f64::from(layers);
    let scale = xy / (1.0 - xy);
    // Sample from Gamma(r, scale)
    use rand_distr::Gamma;
    let lambda = match Gamma::new(r, scale) {
        Ok(dist) => dist.sample(rng),
        Err(_) => return 0,
    };
    sample_poisson(lambda, rng)
}

// ---------------------------------------------------------------------------
// Poisson samplers (existing)
// ---------------------------------------------------------------------------

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

    #[test]
    fn geometric_expected_value() {
        let dist = PairDistribution::GeometricDist { xy: 0.5 };
        assert!((dist.expected() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn geometric_cdf_boundary() {
        let dist = PairDistribution::GeometricDist { xy: 0.5 };
        assert!((dist.lower_pvalue(0) - 0.5).abs() < 1e-12); // P(T<=0) = 1-0.5 = 0.5
        assert!((dist.upper_pvalue(0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn binomial_expected_value() {
        // Bin(10, 0.5/(1+0.5)) = Bin(10, 1/3)
        let dist = PairDistribution::BinomialDist {
            xy: 0.5,
            layers: 10,
        };
        assert!((dist.expected() - 10.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn binomial_cdf_full_range() {
        let dist = PairDistribution::BinomialDist { xy: 1.0, layers: 5 };
        // P(T<=5) = 1.0 since max is 5
        assert!((dist.lower_pvalue(5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn neg_binomial_expected_value() {
        // NB(3, 1-0.4) with xy=0.4: E = 3*0.4/0.6 = 2.0
        let dist = PairDistribution::NegBinomialDist { xy: 0.4, layers: 3 };
        assert!((dist.expected() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn neg_binomial_cdf_at_zero() {
        // P(T=0) = (1-xy)^M = 0.6^3 = 0.216
        let dist = PairDistribution::NegBinomialDist { xy: 0.4, layers: 3 };
        assert!((dist.lower_pvalue(0) - 0.216).abs() < 1e-10);
    }
}
