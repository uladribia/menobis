//! Pair-level null distributions shared by generation and filtering.
//!
//! Taxonomy: `WeightFamily` selects the pair-level weight distribution.
//! `PairDistribution` is the concrete distribution for one `(i,j)` pair.
//! Zero-inflated variants combine a Bernoulli occupation draw with a
//! positive-weight conditional distribution.

use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Bernoulli, Binomial, Distribution, Geometric, Poisson};

const POSITIVE_POISSON_REJECTION_MIN_RATE: f64 = 0.05;

// ---------------------------------------------------------------------------
// Weight family enum — selects the distribution type
// ---------------------------------------------------------------------------

/// Weight distribution family for ME null models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightFamily {
    /// Poisson(λ). E[t] = λ.
    Poisson,
    /// Geometric(1−p). E[t] = p/(1−p).
    Geometric,
    /// Binomial(M, p/(1+p)). E[t] = M·p/(1+p).
    Binomial(u32),
    /// NegativeBinomial(M, 1−p). E[t] = M·p/(1−p).
    NegativeBinomial(u32),
}

impl WeightFamily {
    /// Build a `PairDistribution` from multiplier product `xy = x_i * y_j`.
    #[must_use]
    pub fn distribution(self, xy: f64) -> PairDistribution {
        match self {
            Self::Poisson => PairDistribution::Poisson { rate: xy },
            Self::Geometric => PairDistribution::Geometric { xy },
            Self::Binomial(m) => PairDistribution::Binomial { xy, layers: m },
            Self::NegativeBinomial(m) => PairDistribution::NegativeBinomial { xy, layers: m },
        }
    }

    /// Build a zero-inflated `PairDistribution` from occupation probability
    /// and the positive-weight rate parameter.
    #[must_use]
    pub fn zip_distribution(self, occupation: f64, rate: f64) -> PairDistribution {
        match self {
            Self::Poisson => PairDistribution::ZeroInflatedPoisson { occupation, rate },
            Self::Binomial(m) => PairDistribution::ZeroInflatedBinomial {
                occupation,
                xy: rate,
                layers: m,
            },
            Self::Geometric => PairDistribution::ZeroInflatedGeometric {
                occupation,
                xy: rate,
            },
            Self::NegativeBinomial(m) => PairDistribution::ZeroInflatedNegativeBinomial {
                occupation,
                xy: rate,
                layers: m,
            },
        }
    }

    /// Expected weight E[t_ij] given multiplier product xy.
    #[must_use]
    pub fn expected_weight(self, xy: f64) -> f64 {
        self.distribution(xy).expected()
    }
}

// ---------------------------------------------------------------------------
// PairDistribution — concrete distribution for one (i,j) pair
// ---------------------------------------------------------------------------

/// Distribution for a single candidate pair `(i, j)`.
#[derive(Clone, Copy, Debug)]
pub enum PairDistribution {
    /// Independent Poisson. E[t] = rate.
    Poisson { rate: f64 },
    /// Zero-inflated Poisson (Bernoulli occupation + positive Poisson positive weight).
    ZeroInflatedPoisson { occupation: f64, rate: f64 },
    /// Geometric with param xy. E[t] = xy/(1−xy).
    Geometric { xy: f64 },
    /// Binomial(M, xy/(1+xy)). E[t] = M·xy/(1+xy).
    Binomial { xy: f64, layers: u32 },
    /// NegativeBinomial(M, 1−xy). E[t] = M·xy/(1−xy).
    NegativeBinomial { xy: f64, layers: u32 },
    /// Zero-inflated binomial: Bernoulli occupation + positive binomial(M, p) positive weight.
    ZeroInflatedBinomial {
        occupation: f64,
        xy: f64,
        layers: u32,
    },
    /// Zero-inflated geometric: Bernoulli occupation + positive geometric positive weight.
    ZeroInflatedGeometric { occupation: f64, xy: f64 },
    /// Zero-inflated negative binomial: Bernoulli occupation + positive negative binomial(M) positive weight.
    ZeroInflatedNegativeBinomial {
        occupation: f64,
        xy: f64,
        layers: u32,
    },
}

impl PairDistribution {
    /// Expected edge weight.
    #[must_use]
    pub fn expected(self) -> f64 {
        match self {
            Self::Poisson { rate } => rate.max(0.0),
            Self::ZeroInflatedPoisson { occupation, rate } => {
                occupation.max(0.0) * positive_edge_poisson_mean(rate)
            }
            Self::Geometric { xy } => {
                let xy = xy.max(0.0);
                if xy >= 1.0 {
                    f64::INFINITY
                } else {
                    xy / (1.0 - xy)
                }
            }
            Self::Binomial { xy, layers } => f64::from(layers) * (xy / (1.0 + xy)).clamp(0.0, 1.0),
            Self::NegativeBinomial { xy, layers } => {
                let xy = xy.max(0.0);
                if xy >= 1.0 {
                    f64::INFINITY
                } else {
                    f64::from(layers) * xy / (1.0 - xy)
                }
            }
            Self::ZeroInflatedBinomial {
                occupation,
                xy,
                layers,
            } => {
                let p = (xy / (1.0 + xy)).clamp(0.0, 1.0);
                let m = f64::from(layers);
                let positive_binomial_mean = if (1.0 - p).powi(layers as i32) >= 1.0 {
                    1.0
                } else {
                    m * p / (1.0 - (1.0 - p).powi(layers as i32))
                };
                occupation.max(0.0) * positive_binomial_mean
            }
            Self::ZeroInflatedGeometric { occupation, xy } => {
                // positive geometric mean = 1/(1-q) where q = xy.
                let q = xy.clamp(0.0, 1.0 - 1e-15);
                let ztg_mean = 1.0 / (1.0 - q);
                occupation.max(0.0) * ztg_mean
            }
            Self::ZeroInflatedNegativeBinomial {
                occupation,
                xy,
                layers,
            } => {
                // positive negative binomial mean = Mq / ((1-q)(1-(1-q)^M)).
                let q = xy.clamp(0.0, 1.0 - 1e-15);
                let m = f64::from(layers);
                let p0 = (1.0 - q).powi(layers as i32);
                let ztnb_mean = if p0 >= 1.0 {
                    1.0
                } else {
                    m * q / ((1.0 - q) * (1.0 - p0))
                };
                occupation.max(0.0) * ztnb_mean
            }
        }
    }

    /// Binary occupation probability `P(t_ij > 0)`.
    #[must_use]
    pub fn occupation_probability(self) -> f64 {
        match self {
            Self::Poisson { rate } => 1.0 - (-rate.max(0.0)).exp(),
            Self::ZeroInflatedPoisson { occupation, .. } => occupation.clamp(0.0, 1.0),
            Self::Geometric { xy } => xy.clamp(0.0, 1.0),
            Self::Binomial { xy, layers } => {
                let p = (xy / (1.0 + xy)).clamp(0.0, 1.0);
                1.0 - (1.0 - p).powi(layers as i32)
            }
            Self::NegativeBinomial { xy, layers } => {
                1.0 - (1.0 - xy.max(0.0)).max(0.0).powi(layers as i32)
            }
            Self::ZeroInflatedBinomial { occupation, .. } => occupation.clamp(0.0, 1.0),
            Self::ZeroInflatedGeometric { occupation, .. } => occupation.clamp(0.0, 1.0),
            Self::ZeroInflatedNegativeBinomial { occupation, .. } => occupation.clamp(0.0, 1.0),
        }
    }

    /// Inclusive lower-tail probability `P(T <= weight)`.
    #[must_use]
    pub fn lower_pvalue(self, weight: u64) -> f64 {
        match self {
            Self::Poisson { rate } => poisson_cdf(weight, rate),
            Self::ZeroInflatedPoisson { occupation, rate } => {
                let p = occupation.clamp(0.0, 1.0);
                if weight == 0 {
                    1.0 - p
                } else {
                    (1.0 - p) + p * positive_edge_poisson_cdf(weight, rate)
                }
            }
            Self::Geometric { xy } => geometric_cdf(weight, xy),
            Self::Binomial { xy, layers } => binomial_cdf(weight, xy / (1.0 + xy), layers),
            Self::NegativeBinomial { xy, layers } => negative_binomial_cdf(weight, xy, layers),
            Self::ZeroInflatedBinomial {
                occupation,
                xy,
                layers,
            } => {
                let p = occupation.clamp(0.0, 1.0);
                if weight == 0 {
                    1.0 - p
                } else {
                    let bin_p = (xy / (1.0 + xy)).clamp(0.0, 1.0);
                    (1.0 - p) + p * positive_binomial_cdf(weight, bin_p, layers)
                }
            }
            Self::ZeroInflatedGeometric { occupation, xy } => {
                let p = occupation.clamp(0.0, 1.0);
                if weight == 0 {
                    1.0 - p
                } else {
                    // positive geometric CDF: P(K<=k|K>=1) = (Geo_CDF(k) - Geo_PMF(0)) / (1 - Geo_PMF(0))
                    // Geo_CDF(k) = 1 - q^{k+1}, Geo_PMF(0) = 1-q, so
                    // positive geometric_CDF(k) = (1 - q^{k+1} - (1-q)) / q = (q - q^{k+1})/q = 1 - q^k
                    let q = xy.clamp(0.0, 1.0 - 1e-15);
                    let ztg_cdf = 1.0 - q.powi(weight as i32);
                    (1.0 - p) + p * ztg_cdf
                }
            }
            Self::ZeroInflatedNegativeBinomial {
                occupation,
                xy,
                layers,
            } => {
                let p = occupation.clamp(0.0, 1.0);
                if weight == 0 {
                    1.0 - p
                } else {
                    // positive negative binomial CDF: (negative_binomial_cdf(k) - negative_binomial_pmf(0)) / (1 - negative_binomial_pmf(0))
                    let q = xy.clamp(0.0, 1.0 - 1e-15);
                    let p0 = (1.0 - q).powi(layers as i32);
                    let nb_cdf = negative_binomial_cdf(weight, q, layers);
                    let ztnb_cdf = if p0 >= 1.0 {
                        1.0
                    } else {
                        (nb_cdf - p0) / (1.0 - p0)
                    };
                    (1.0 - p) + p * ztnb_cdf.clamp(0.0, 1.0)
                }
            }
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
            Self::ZeroInflatedPoisson { occupation, rate } => {
                occupation.clamp(0.0, 1.0) * positive_edge_poisson_sf_inclusive(weight, rate)
            }
            Self::Geometric { xy } => (1.0 - geometric_cdf(weight - 1, xy)).clamp(0.0, 1.0),
            Self::Binomial { xy, layers } => {
                (1.0 - binomial_cdf(weight - 1, xy / (1.0 + xy), layers)).clamp(0.0, 1.0)
            }
            Self::NegativeBinomial { xy, layers } => {
                (1.0 - negative_binomial_cdf(weight - 1, xy, layers)).clamp(0.0, 1.0)
            }
            Self::ZeroInflatedBinomial {
                occupation,
                xy,
                layers,
            } => {
                let p = occupation.clamp(0.0, 1.0);
                let bin_p = (xy / (1.0 + xy)).clamp(0.0, 1.0);
                (p * positive_binomial_sf_inclusive(weight, bin_p, layers)).clamp(0.0, 1.0)
            }
            Self::ZeroInflatedGeometric { occupation, xy } => {
                // P(K>=k|K>=1) = 1 - positive geometric_CDF(k-1) = q^{k-1}
                let p = occupation.clamp(0.0, 1.0);
                let q = xy.clamp(0.0, 1.0 - 1e-15);
                (p * q.powi((weight - 1) as i32)).clamp(0.0, 1.0)
            }
            Self::ZeroInflatedNegativeBinomial {
                occupation,
                xy,
                layers,
            } => {
                let p = occupation.clamp(0.0, 1.0);
                let q = xy.clamp(0.0, 1.0 - 1e-15);
                let p0 = (1.0 - q).powi(layers as i32);
                if p0 >= 1.0 {
                    return 0.0;
                }
                let nb_cdf_prev = negative_binomial_cdf(weight - 1, q, layers);
                let ztnb_sf = (1.0 - (nb_cdf_prev - p0) / (1.0 - p0)).clamp(0.0, 1.0);
                (p * ztnb_sf).clamp(0.0, 1.0)
            }
        }
        .clamp(0.0, 1.0)
    }

    /// Draw one sample from this pair distribution.
    pub fn sample(self, rng: &mut StdRng) -> u64 {
        match self {
            Self::Poisson { rate } => sample_poisson(rate, rng),
            Self::ZeroInflatedPoisson { occupation, rate } => {
                if occupation <= 0.0 {
                    return 0;
                }
                let present = match Bernoulli::new(occupation.min(1.0)) {
                    Ok(dist) => dist.sample(rng),
                    Err(_) => false,
                };
                if present {
                    sample_positive_edge_poisson(rate, rng)
                } else {
                    0
                }
            }
            Self::Geometric { xy } => sample_geometric(xy, rng),
            Self::Binomial { xy, layers } => sample_binomial(xy, layers, rng),
            Self::NegativeBinomial { xy, layers } => sample_negative_binomial(xy, layers, rng),
            Self::ZeroInflatedBinomial {
                occupation,
                xy,
                layers,
            } => {
                if occupation <= 0.0 {
                    return 0;
                }
                let present = match Bernoulli::new(occupation.min(1.0)) {
                    Ok(dist) => dist.sample(rng),
                    Err(_) => false,
                };
                if present {
                    sample_positive_binomial(xy, layers, rng)
                } else {
                    0
                }
            }
            Self::ZeroInflatedGeometric { occupation, xy } => {
                if occupation <= 0.0 {
                    return 0;
                }
                let present = match Bernoulli::new(occupation.min(1.0)) {
                    Ok(dist) => dist.sample(rng),
                    Err(_) => false,
                };
                if present {
                    sample_positive_geometric(xy, rng)
                } else {
                    0
                }
            }
            Self::ZeroInflatedNegativeBinomial {
                occupation,
                xy,
                layers,
            } => {
                if occupation <= 0.0 {
                    return 0;
                }
                let present = match Bernoulli::new(occupation.min(1.0)) {
                    Ok(dist) => dist.sample(rng),
                    Err(_) => false,
                };
                if present {
                    sample_positive_negative_binomial(xy, layers, rng)
                } else {
                    0
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// zero-inflated distribution constructors (constraint-specific)
// ---------------------------------------------------------------------------

/// Strength-edges zero-inflated: Bernoulli(p) + positive Poisson(rate) where rate = x_i * y_j.
#[must_use]
pub fn strength_edges_distribution(xi: f64, yj: f64, lam: f64) -> PairDistribution {
    let rate = xi * yj;
    let expm1 = rate.exp_m1();
    let den = 1.0 + lam * expm1;
    let occupation = if den > 0.0 { lam * expm1 / den } else { 0.0 };
    PairDistribution::ZeroInflatedPoisson { occupation, rate }
}

/// Strength-degree zero-inflated: Bernoulli(p) + positive Poisson(rate) where rate = x_i * y_j.
#[must_use]
pub fn strength_degree_distribution(xi: f64, yj: f64, zi: f64, wj: f64) -> PairDistribution {
    let rate = xi * yj;
    let v = zi * wj;
    let expm1 = rate.exp_m1();
    let den = 1.0 + v * expm1;
    let occupation = if den > 0.0 { v * expm1 / den } else { 0.0 };
    PairDistribution::ZeroInflatedPoisson { occupation, rate }
}

// ---------------------------------------------------------------------------
// Poisson helpers
// ---------------------------------------------------------------------------

/// Positive Poisson mean conditional on edge existence.
#[must_use]
pub fn positive_edge_poisson_mean(rate: f64) -> f64 {
    if rate <= 0.0 {
        1.0
    } else {
        rate / (1.0 - (-rate).exp())
    }
}

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

#[must_use]
pub fn poisson_sf_inclusive(weight: u64, rate: f64) -> f64 {
    if weight == 0 {
        1.0
    } else {
        (1.0 - poisson_cdf(weight - 1, rate)).clamp(0.0, 1.0)
    }
}

#[must_use]
pub fn positive_edge_poisson_cdf(weight: u64, rate: f64) -> f64 {
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

#[must_use]
pub fn positive_edge_poisson_sf_inclusive(weight: u64, rate: f64) -> f64 {
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

// ---------------------------------------------------------------------------
// Geometric helpers
// ---------------------------------------------------------------------------

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
        Ok(dist) => dist.sample(rng),
        Err(_) => 0,
    }
}

// ---------------------------------------------------------------------------
// Binomial helpers
// ---------------------------------------------------------------------------

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
    let mut log_binom = 0.0_f64;
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
// Negative binomial helpers
// ---------------------------------------------------------------------------

fn negative_binomial_cdf(weight: u64, xy: f64, layers: u32) -> f64 {
    let xy = xy.clamp(0.0, 1.0 - 1e-15);
    let p_success = 1.0 - xy;
    if p_success >= 1.0 || xy <= 0.0 {
        return 1.0;
    }
    let r = f64::from(layers);
    let mut sum = 0.0;
    let mut log_coeff = 0.0_f64;
    let log_p = p_success.ln();
    let log_q = xy.ln();
    for k in 0..=weight {
        let log_pmf = log_coeff + r * log_p + (k as f64) * log_q;
        sum += log_pmf.exp();
        if sum >= 1.0 {
            return 1.0;
        }
        log_coeff += ((k as f64) + r).ln() - ((k + 1) as f64).ln();
    }
    sum.clamp(0.0, 1.0)
}

fn sample_negative_binomial(xy: f64, layers: u32, rng: &mut StdRng) -> u64 {
    let xy = xy.clamp(0.0, 1.0 - 1e-15);
    if xy <= 0.0 {
        return 0;
    }
    let r = f64::from(layers);
    let scale = xy / (1.0 - xy);
    use rand_distr::Gamma;
    let lambda = match Gamma::new(r, scale) {
        Ok(dist) => dist.sample(rng),
        Err(_) => return 0,
    };
    sample_poisson(lambda, rng)
}

// ---------------------------------------------------------------------------
// Positive binomial helpers conditional on edge existence
// ---------------------------------------------------------------------------

/// positive binomial CDF: P(T <= k | T > 0) = (Bin_CDF(k) - Bin_PMF(0)) / (1 - Bin_PMF(0)).
fn positive_binomial_cdf(weight: u64, p: f64, layers: u32) -> f64 {
    if weight == 0 {
        return 0.0;
    }
    let p0 = (1.0 - p).powi(layers as i32);
    if p0 >= 1.0 {
        return 1.0;
    }
    let num = binomial_cdf(weight, p, layers) - p0;
    let den = 1.0 - p0;
    (num / den).clamp(0.0, 1.0)
}

/// positive binomial survival: P(T >= k | T > 0).
fn positive_binomial_sf_inclusive(weight: u64, p: f64, layers: u32) -> f64 {
    if weight <= 1 {
        return 1.0;
    }
    (1.0 - positive_binomial_cdf(weight - 1, p, layers)).clamp(0.0, 1.0)
}

/// Sample from positive-edge Bin(M, p) by rejection.
fn sample_positive_binomial(xy: f64, layers: u32, rng: &mut StdRng) -> u64 {
    let p = (xy / (1.0 + xy)).clamp(0.0, 1.0);
    if p <= 0.0 {
        return 1;
    }
    let dist = match Binomial::new(u64::from(layers), p) {
        Ok(d) => d,
        Err(_) => return 1,
    };
    for _ in 0..10000 {
        let v = dist.sample(rng);
        if v > 0 {
            return v;
        }
    }
    1
}

/// Sample from positive-edge Geometric(1-q) by rejection.
fn sample_positive_geometric(xy: f64, rng: &mut StdRng) -> u64 {
    // Geometric P(k) = (1-q)*q^k for k>=0. Condition on k>=1.
    // Efficient: sample from Geometric and add 1, since P(k>=1) follows
    // the same Geometric shifted. Actually P(K>=1|K>=0) ~ Geo shifted by 1.
    // Simpler: just rejection.
    let q = xy.clamp(0.0, 1.0 - 1e-15);
    if q <= 0.0 {
        return 1;
    }
    let p = 1.0 - q;
    let dist = match Geometric::new(p) {
        Ok(d) => d,
        Err(_) => return 1,
    };
    // For Geometric(p), P(0) = p = 1-q. For q < 1, P(K>=1) = q.
    // Rejection is efficient when q is not tiny.
    for _ in 0..10000 {
        let v = dist.sample(rng);
        if v > 0 {
            return v;
        }
    }
    1
}

/// Sample from positive-edge NegativeBinomial(M, 1-q) by rejection.
fn sample_positive_negative_binomial(xy: f64, layers: u32, rng: &mut StdRng) -> u64 {
    let q = xy.clamp(0.0, 1.0 - 1e-15);
    if q <= 0.0 {
        return 1;
    }
    // P(0) = (1-q)^M. Rejection rate = (1-q)^M which is acceptable for moderate q/M.
    for _ in 0..10000 {
        let v = sample_negative_binomial(q, layers, rng);
        if v > 0 {
            return v;
        }
    }
    1
}

/// Occupation probability for strength-edges binomial zero-inflated.
/// Uses (1+xy)^M - 1 instead of exp(xy) - 1.
#[must_use]
pub fn strength_edges_binomial_occupation(xy: f64, lam: f64, layers: u32) -> f64 {
    let factor = (1.0 + xy).powi(layers as i32) - 1.0;
    let den = 1.0 + lam * factor;
    if den > 0.0 {
        lam * factor / den
    } else {
        0.0
    }
}

/// Occupation probability for strength-degree binomial zero-inflated.
#[must_use]
pub fn strength_degree_binomial_occupation(xy: f64, vij: f64, layers: u32) -> f64 {
    let factor = (1.0 + xy).powi(layers as i32) - 1.0;
    let den = 1.0 + vij * factor;
    if den > 0.0 {
        vij * factor / den
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Poisson samplers
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

/// Draw from a positive-edge Poisson distribution.
pub fn sample_positive_edge_poisson(rate: f64, rng: &mut StdRng) -> u64 {
    if rate <= 0.0 || !rate.is_finite() {
        return 1;
    }
    if rate < POSITIVE_POISSON_REJECTION_MIN_RATE {
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
        let dist = PairDistribution::ZeroInflatedPoisson {
            occupation: 0.7,
            rate: 2.0,
        };
        assert!((dist.lower_pvalue(0) - 0.3).abs() < 1e-12);
        assert!((dist.upper_pvalue(0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn geometric_expected_value() {
        let dist = PairDistribution::Geometric { xy: 0.5 };
        assert!((dist.expected() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn geometric_cdf_boundary() {
        let dist = PairDistribution::Geometric { xy: 0.5 };
        assert!((dist.lower_pvalue(0) - 0.5).abs() < 1e-12);
        assert!((dist.upper_pvalue(0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn binomial_expected_value() {
        let dist = PairDistribution::Binomial {
            xy: 0.5,
            layers: 10,
        };
        assert!((dist.expected() - 10.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn binomial_cdf_full_range() {
        let dist = PairDistribution::Binomial { xy: 1.0, layers: 5 };
        assert!((dist.lower_pvalue(5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn negative_binomial_expected_value() {
        let dist = PairDistribution::NegativeBinomial { xy: 0.4, layers: 3 };
        assert!((dist.expected() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn negative_binomial_cdf_at_zero() {
        let dist = PairDistribution::NegativeBinomial { xy: 0.4, layers: 3 };
        assert!((dist.lower_pvalue(0) - 0.216).abs() < 1e-10);
    }

    #[test]
    fn zip_binomial_expected_value() {
        let p: f64 = 0.5 / 1.5;
        let m: f64 = 5.0;
        let positive_binomial_mean = m * p / (1.0 - (1.0 - p).powi(5));
        let dist = PairDistribution::ZeroInflatedBinomial {
            occupation: 0.8,
            xy: 0.5,
            layers: 5,
        };
        assert!((dist.expected() - 0.8 * positive_binomial_mean).abs() < 1e-8);
    }

    #[test]
    fn zip_binomial_zero_probability() {
        let dist = PairDistribution::ZeroInflatedBinomial {
            occupation: 0.6,
            xy: 0.5,
            layers: 5,
        };
        assert!((dist.lower_pvalue(0) - 0.4).abs() < 1e-12);
        assert!((dist.upper_pvalue(0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zip_binomial_weights_bounded_by_layers() {
        use rand::SeedableRng;
        let dist = PairDistribution::ZeroInflatedBinomial {
            occupation: 0.9,
            xy: 0.8,
            layers: 5,
        };
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..1000 {
            let w = dist.sample(&mut rng);
            assert!(w <= 5, "weight {w} exceeds layers 5");
        }
    }

    #[test]
    fn strength_edges_binomial_occupation_formula() {
        let factor = (1.0 + 0.3_f64).powi(4) - 1.0;
        let expected = 0.5 * factor / (1.0 + 0.5 * factor);
        let actual = super::strength_edges_binomial_occupation(0.3, 0.5, 4);
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn zip_geometric_expected_value() {
        // ZeroInflatedGeometric: occupation * positive geometric mean.
        // positive geometric(q) mean = q / ((1-q) * (1 - (1-q))) = q / ((1-q)*q) = 1/(1-q).
        // Wait: Geometric P(k) = (1-q)*q^k for k>=0. positive geometric conditions on k>=1:
        // P(k|k>=1) = (1-q)*q^k / q = (1-q)*q^{k-1} for k>=1. Mean = 1/(1-q).
        // So ZeroInflatedGeometric expected = occupation * 1/(1-q).
        let occ = 0.7;
        let xy = 0.4; // q = xy
        let ztg_mean = 1.0 / (1.0 - xy);
        let dist = PairDistribution::ZeroInflatedGeometric {
            occupation: occ,
            xy,
        };
        assert!((dist.expected() - occ * ztg_mean).abs() < 1e-12);
    }

    #[test]
    fn zip_geometric_zero_probability() {
        let dist = PairDistribution::ZeroInflatedGeometric {
            occupation: 0.6,
            xy: 0.3,
        };
        assert!((dist.lower_pvalue(0) - 0.4).abs() < 1e-12);
        assert!((dist.upper_pvalue(0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zip_geometric_samples_are_non_negative() {
        use rand::SeedableRng;
        let dist = PairDistribution::ZeroInflatedGeometric {
            occupation: 0.8,
            xy: 0.5,
        };
        let mut rng = StdRng::seed_from_u64(99);
        let mut has_zero = false;
        let mut has_positive = false;
        for _ in 0..1000 {
            let w = dist.sample(&mut rng);
            if w == 0 {
                has_zero = true;
            } else {
                has_positive = true;
            }
        }
        assert!(has_zero, "expected some zeros from ZeroInflatedGeometric");
        assert!(
            has_positive,
            "expected some positive weights from ZeroInflatedGeometric"
        );
    }

    #[test]
    fn zip_negative_binomial_expected_value() {
        // ZeroInflatedNegativeBinomial: occupation * positive negative binomial mean.
        // negative binomial(M, 1-q) has P(k=0) = (1-q)^M. Mean = Mq/(1-q).
        // positive negative binomial mean = Mq/((1-q)*(1-(1-q)^M)).
        // zero-inflated negative binomial expected = occupation * positive negative binomial mean.
        let occ = 0.8;
        let xy = 0.4; // q = xy
        let m = 3_u32;
        let ztnb_mean = (m as f64) * xy / ((1.0 - xy) * (1.0 - (1.0 - xy).powi(m as i32)));
        let dist = PairDistribution::ZeroInflatedNegativeBinomial {
            occupation: occ,
            xy,
            layers: m,
        };
        assert!((dist.expected() - occ * ztnb_mean).abs() < 1e-10);
    }

    #[test]
    fn zip_negative_binomial_zero_probability() {
        let dist = PairDistribution::ZeroInflatedNegativeBinomial {
            occupation: 0.5,
            xy: 0.3,
            layers: 2,
        };
        assert!((dist.lower_pvalue(0) - 0.5).abs() < 1e-12);
        assert!((dist.upper_pvalue(0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zip_negative_binomial_samples_are_non_negative() {
        use rand::SeedableRng;
        let dist = PairDistribution::ZeroInflatedNegativeBinomial {
            occupation: 0.7,
            xy: 0.4,
            layers: 3,
        };
        let mut rng = StdRng::seed_from_u64(123);
        let mut has_zero = false;
        let mut has_positive = false;
        for _ in 0..1000 {
            let w = dist.sample(&mut rng);
            if w == 0 {
                has_zero = true;
            } else {
                has_positive = true;
            }
        }
        assert!(
            has_zero,
            "expected some zeros from ZeroInflatedNegativeBinomial"
        );
        assert!(
            has_positive,
            "expected some positive from ZeroInflatedNegativeBinomial"
        );
    }

    #[test]
    fn weight_family_builds_correct_distribution() {
        let dist = WeightFamily::Poisson.distribution(2.0);
        assert!((dist.expected() - 2.0).abs() < 1e-12);

        let dist = WeightFamily::Geometric.distribution(0.5);
        assert!((dist.expected() - 1.0).abs() < 1e-12);

        let dist = WeightFamily::Binomial(10).distribution(0.5);
        assert!((dist.expected() - 10.0 / 3.0).abs() < 1e-10);

        let dist = WeightFamily::NegativeBinomial(3).distribution(0.4);
        assert!((dist.expected() - 2.0).abs() < 1e-12);
    }
}
