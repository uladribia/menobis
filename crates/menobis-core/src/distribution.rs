//! Pair-level null distributions shared by generation and filtering.
//!
//! Taxonomy: `OccupationFamily` selects the pair-level occ_num distribution.
//! `PairDistribution` is the concrete distribution for one `(i,j)` pair.
//! Zero-inflated variants combine a Bernoulli occupation draw with a
//! positive-occ_num conditional distribution.

use crate::OccNum;
use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Bernoulli, Binomial, Distribution, Geometric, Poisson};

const POSITIVE_POISSON_REJECTION_MIN_RATE: f64 = 0.05;

/// Stable log-gamma `ln Gamma(x)` for `x > 0`.
///
/// `libm::lgamma` returns `(value, sign)`; the sign is always positive for
/// `x > 0`, so the sign is ignored.
fn ln_gamma(x: f64) -> f64 {
    libm::lgamma(x)
}

// ---------------------------------------------------------------------------
// Occupation family enum — selects the distribution type
// ---------------------------------------------------------------------------

/// Thesis model-family kind (thesis §2.1 base measures).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelFamilyKind {
    /// MultiEdge: `d_ME(t) = 1/t!`.
    MultiEdge,
    /// Weighted: `d_W,M(t) = C(M+t-1, t)`.
    Weighted,
    /// BinaryLayers: `d_B,M(t) = C(M, t)` for `0 <= t <= M`.
    BinaryLayers,
}

/// Occupation-number support of a family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccupationSupport {
    /// `t in [0, oo)` (ME and W).
    Unbounded,
    /// `t in [0, M]` (B with M layers).
    Bounded(u32),
}

/// Occupation distribution family for ME null models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccupationFamily {
    /// Poisson(λ). E[t] = λ.
    Poisson,
    /// Geometric(1−p). E[t] = p/(1−p).
    Geometric,
    /// Binomial(M, p/(1+p)). E[t] = M·p/(1+p).
    Binomial(u32),
    /// NegativeBinomial(M, 1−p). E[t] = M·p/(1−p).
    NegativeBinomial(u32),
}

impl OccupationFamily {
    /// Thesis family kind (base-measure ontology).
    #[must_use]
    pub fn model_family(self) -> ModelFamilyKind {
        match self {
            Self::Poisson => ModelFamilyKind::MultiEdge,
            Self::Geometric | Self::NegativeBinomial(_) => ModelFamilyKind::Weighted,
            Self::Binomial(_) => ModelFamilyKind::BinaryLayers,
        }
    }

    /// Layer count `M` for B/W; `None` for ME.
    #[must_use]
    pub fn layers(self) -> Option<u32> {
        match self {
            Self::Poisson | Self::Geometric => None,
            Self::Binomial(m) | Self::NegativeBinomial(m) => Some(m),
        }
    }

    /// Occupation-number support of the family.
    #[must_use]
    pub fn occupation_support(self) -> OccupationSupport {
        match self {
            Self::Poisson | Self::Geometric | Self::NegativeBinomial(_) => {
                OccupationSupport::Unbounded
            }
            Self::Binomial(m) => OccupationSupport::Bounded(m),
        }
    }

    /// Whether `occ_num` is a valid occupation number for this family.
    ///
    /// B rejects occupation numbers above its layer capacity `M`.
    #[must_use]
    pub fn validate_occnum(self, occ_num: OccNum) -> bool {
        match self {
            Self::Binomial(m) => occ_num <= OccNum::from(m),
            _ => true,
        }
    }

    /// Log of the thesis local degeneracy `ln d_F(t)`.
    ///
    /// Base measures (thesis §2.1):
    /// - ME: `d(t) = 1/t!` → `-ln Gamma(t+1)`
    /// - W:  `d(t) = C(M+t-1, t)` → `ln Gamma(M+t) - ln Gamma(t+1) - ln Gamma(M)`
    /// - B:  `d(t) = C(M, t)` → `ln Gamma(M+1) - ln Gamma(t+1) - ln Gamma(M-t+1)`
    ///
    /// Uses stable log-gamma. Occupations outside the family support return
    /// `-inf` (zero degeneracy).
    #[must_use]
    pub fn log_local_degeneracy(self, occ_num: OccNum) -> f64 {
        match self {
            Self::Poisson => -ln_gamma((occ_num + 1) as f64),
            Self::Geometric => {
                // W with M=1: C(1+t-1, t) = C(t, t) = 1, ln d = 0.
                0.0
            }
            Self::NegativeBinomial(m) => {
                let mf = f64::from(m);
                let t = occ_num as f64;
                ln_gamma(mf + t) - ln_gamma(t + 1.0) - ln_gamma(mf)
            }
            Self::Binomial(m) => {
                if occ_num > OccNum::from(m) {
                    return f64::NEG_INFINITY;
                }
                let mf = f64::from(m);
                let t = occ_num as f64;
                ln_gamma(mf + 1.0) - ln_gamma(t + 1.0) - ln_gamma(mf - t + 1.0)
            }
        }
    }

    /// Difference of log local degeneracies: `ln d(new) - ln d(old)`.
    ///
    /// This is the local Metropolis acceptance weight for changing a pair
    /// occupation from `old` to `new` under the family base measure.
    #[must_use]
    pub fn delta_log_local_degeneracy(self, old_occ_num: OccNum, new_occ_num: OccNum) -> f64 {
        self.log_local_degeneracy(new_occ_num) - self.log_local_degeneracy(old_occ_num)
    }

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
    /// and the positive-occ_num rate parameter.
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

    /// Expected occ_num E[t_ij] given multiplier product xy.
    #[must_use]
    pub fn expected_occ_num(self, xy: f64) -> f64 {
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
    /// Zero-inflated Poisson (Bernoulli occupation + positive Poisson positive occ_num).
    ZeroInflatedPoisson { occupation: f64, rate: f64 },
    /// Geometric with param xy. E[t] = xy/(1−xy).
    Geometric { xy: f64 },
    /// Binomial(M, xy/(1+xy)). E[t] = M·xy/(1+xy).
    Binomial { xy: f64, layers: u32 },
    /// NegativeBinomial(M, 1−xy). E[t] = M·xy/(1−xy).
    NegativeBinomial { xy: f64, layers: u32 },
    /// Zero-inflated binomial: Bernoulli occupation + positive binomial(M, p) positive occ_num.
    ZeroInflatedBinomial {
        occupation: f64,
        xy: f64,
        layers: u32,
    },
    /// Zero-inflated geometric: Bernoulli occupation + positive geometric positive occ_num.
    ZeroInflatedGeometric { occupation: f64, xy: f64 },
    /// Zero-inflated negative binomial: Bernoulli occupation + positive negative binomial(M) positive occ_num.
    ZeroInflatedNegativeBinomial {
        occupation: f64,
        xy: f64,
        layers: u32,
    },
}

impl PairDistribution {
    /// Expected edge occ_num.
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

    /// Inclusive lower-tail probability `P(T <= occ_num)`.
    #[must_use]
    pub fn lower_pvalue(self, occ_num: u64) -> f64 {
        match self {
            Self::Poisson { rate } => poisson_cdf(occ_num, rate),
            Self::ZeroInflatedPoisson { occupation, rate } => {
                let p = occupation.clamp(0.0, 1.0);
                if occ_num == 0 {
                    1.0 - p
                } else {
                    (1.0 - p) + p * positive_edge_poisson_cdf(occ_num, rate)
                }
            }
            Self::Geometric { xy } => geometric_cdf(occ_num, xy),
            Self::Binomial { xy, layers } => binomial_cdf(occ_num, xy / (1.0 + xy), layers),
            Self::NegativeBinomial { xy, layers } => negative_binomial_cdf(occ_num, xy, layers),
            Self::ZeroInflatedBinomial {
                occupation,
                xy,
                layers,
            } => {
                let p = occupation.clamp(0.0, 1.0);
                if occ_num == 0 {
                    1.0 - p
                } else {
                    let bin_p = (xy / (1.0 + xy)).clamp(0.0, 1.0);
                    (1.0 - p) + p * positive_binomial_cdf(occ_num, bin_p, layers)
                }
            }
            Self::ZeroInflatedGeometric { occupation, xy } => {
                let p = occupation.clamp(0.0, 1.0);
                if occ_num == 0 {
                    1.0 - p
                } else {
                    // positive geometric CDF: P(K<=k|K>=1) = (Geo_CDF(k) - Geo_PMF(0)) / (1 - Geo_PMF(0))
                    // Geo_CDF(k) = 1 - q^{k+1}, Geo_PMF(0) = 1-q, so
                    // positive geometric_CDF(k) = (1 - q^{k+1} - (1-q)) / q = (q - q^{k+1})/q = 1 - q^k
                    let q = xy.clamp(0.0, 1.0 - 1e-15);
                    let ztg_cdf = 1.0 - q.powi(occ_num as i32);
                    (1.0 - p) + p * ztg_cdf
                }
            }
            Self::ZeroInflatedNegativeBinomial {
                occupation,
                xy,
                layers,
            } => {
                let p = occupation.clamp(0.0, 1.0);
                if occ_num == 0 {
                    1.0 - p
                } else {
                    // positive negative binomial CDF: (negative_binomial_cdf(k) - negative_binomial_pmf(0)) / (1 - negative_binomial_pmf(0))
                    let q = xy.clamp(0.0, 1.0 - 1e-15);
                    let p0 = (1.0 - q).powi(layers as i32);
                    let nb_cdf = negative_binomial_cdf(occ_num, q, layers);
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

    /// Inclusive upper-tail probability `P(T >= occ_num)`.
    #[must_use]
    pub fn upper_pvalue(self, occ_num: u64) -> f64 {
        if occ_num == 0 {
            return 1.0;
        }
        match self {
            Self::Poisson { rate } => poisson_sf_inclusive(occ_num, rate),
            Self::ZeroInflatedPoisson { occupation, rate } => {
                occupation.clamp(0.0, 1.0) * positive_edge_poisson_sf_inclusive(occ_num, rate)
            }
            Self::Geometric { xy } => (1.0 - geometric_cdf(occ_num - 1, xy)).clamp(0.0, 1.0),
            Self::Binomial { xy, layers } => {
                (1.0 - binomial_cdf(occ_num - 1, xy / (1.0 + xy), layers)).clamp(0.0, 1.0)
            }
            Self::NegativeBinomial { xy, layers } => {
                (1.0 - negative_binomial_cdf(occ_num - 1, xy, layers)).clamp(0.0, 1.0)
            }
            Self::ZeroInflatedBinomial {
                occupation,
                xy,
                layers,
            } => {
                let p = occupation.clamp(0.0, 1.0);
                let bin_p = (xy / (1.0 + xy)).clamp(0.0, 1.0);
                (p * positive_binomial_sf_inclusive(occ_num, bin_p, layers)).clamp(0.0, 1.0)
            }
            Self::ZeroInflatedGeometric { occupation, xy } => {
                // P(K>=k|K>=1) = 1 - positive geometric_CDF(k-1) = q^{k-1}
                let p = occupation.clamp(0.0, 1.0);
                let q = xy.clamp(0.0, 1.0 - 1e-15);
                (p * q.powi((occ_num - 1) as i32)).clamp(0.0, 1.0)
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
                let nb_cdf_prev = negative_binomial_cdf(occ_num - 1, q, layers);
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
pub fn poisson_cdf(occ_num: u64, rate: f64) -> f64 {
    if rate <= 0.0 {
        return 1.0;
    }
    let mut term = (-rate).exp();
    let mut sum = term;
    for k in 1..=occ_num {
        term *= rate / k as f64;
        sum += term;
        if term == 0.0 {
            break;
        }
    }
    sum.clamp(0.0, 1.0)
}

#[must_use]
pub fn poisson_sf_inclusive(occ_num: u64, rate: f64) -> f64 {
    if occ_num == 0 {
        1.0
    } else {
        (1.0 - poisson_cdf(occ_num - 1, rate)).clamp(0.0, 1.0)
    }
}

#[must_use]
pub fn positive_edge_poisson_cdf(occ_num: u64, rate: f64) -> f64 {
    if occ_num == 0 {
        return 0.0;
    }
    if rate <= 0.0 {
        return 1.0;
    }
    let numerator = poisson_cdf(occ_num, rate) - (-rate).exp();
    let denominator = 1.0 - (-rate).exp();
    (numerator / denominator).clamp(0.0, 1.0)
}

#[must_use]
pub fn positive_edge_poisson_sf_inclusive(occ_num: u64, rate: f64) -> f64 {
    if occ_num <= 1 && rate <= 0.0 {
        return 1.0;
    }
    if rate <= 0.0 {
        return 0.0;
    }
    let numerator = poisson_sf_inclusive(occ_num, rate);
    let denominator = 1.0 - (-rate).exp();
    (numerator / denominator).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Geometric helpers
// ---------------------------------------------------------------------------

fn geometric_cdf(occ_num: u64, xy: f64) -> f64 {
    let xy = xy.clamp(0.0, 1.0);
    if xy <= 0.0 {
        return 1.0;
    }
    (1.0 - xy.powi((occ_num + 1) as i32)).clamp(0.0, 1.0)
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

fn binomial_cdf(occ_num: u64, p: f64, layers: u32) -> f64 {
    let p = p.clamp(0.0, 1.0);
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return if occ_num >= u64::from(layers) {
            1.0
        } else {
            0.0
        };
    }
    let n = layers;
    let mut sum = 0.0;
    let mut log_binom = 0.0_f64;
    for k in 0..=occ_num.min(u64::from(n)) {
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

fn negative_binomial_cdf(occ_num: u64, xy: f64, layers: u32) -> f64 {
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
    for k in 0..=occ_num {
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
fn positive_binomial_cdf(occ_num: u64, p: f64, layers: u32) -> f64 {
    if occ_num == 0 {
        return 0.0;
    }
    let p0 = (1.0 - p).powi(layers as i32);
    if p0 >= 1.0 {
        return 1.0;
    }
    let num = binomial_cdf(occ_num, p, layers) - p0;
    let den = 1.0 - p0;
    (num / den).clamp(0.0, 1.0)
}

/// positive binomial survival: P(T >= k | T > 0).
fn positive_binomial_sf_inclusive(occ_num: u64, p: f64, layers: u32) -> f64 {
    if occ_num <= 1 {
        return 1.0;
    }
    (1.0 - positive_binomial_cdf(occ_num - 1, p, layers)).clamp(0.0, 1.0)
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

    // -----------------------------------------------------------------------
    // P0.4 shared family measure tests
    // -----------------------------------------------------------------------

    #[test]
    fn family_kind_mapping() {
        assert_eq!(
            OccupationFamily::Poisson.model_family(),
            ModelFamilyKind::MultiEdge
        );
        assert_eq!(
            OccupationFamily::Geometric.model_family(),
            ModelFamilyKind::Weighted
        );
        assert_eq!(
            OccupationFamily::NegativeBinomial(3).model_family(),
            ModelFamilyKind::Weighted
        );
        assert_eq!(
            OccupationFamily::Binomial(5).model_family(),
            ModelFamilyKind::BinaryLayers
        );
    }

    #[test]
    fn layer_extraction() {
        assert_eq!(OccupationFamily::Poisson.layers(), None);
        assert_eq!(OccupationFamily::Geometric.layers(), None);
        assert_eq!(OccupationFamily::Binomial(7).layers(), Some(7));
        assert_eq!(OccupationFamily::NegativeBinomial(4).layers(), Some(4));
    }

    #[test]
    fn bounded_and_unbounded_support() {
        assert_eq!(
            OccupationFamily::Poisson.occupation_support(),
            OccupationSupport::Unbounded
        );
        assert_eq!(
            OccupationFamily::NegativeBinomial(9).occupation_support(),
            OccupationSupport::Unbounded
        );
        assert_eq!(
            OccupationFamily::Binomial(6).occupation_support(),
            OccupationSupport::Bounded(6)
        );
    }

    #[test]
    fn b_capacity_rejection() {
        let b = OccupationFamily::Binomial(3);
        assert!(b.validate_occnum(0));
        assert!(b.validate_occnum(3));
        assert!(!b.validate_occnum(4));
        assert!(OccupationFamily::Poisson.validate_occnum(u64::MAX));
    }

    #[test]
    fn small_exact_me_degeneracy() {
        // d_ME(t) = 1/t!
        let me = OccupationFamily::Poisson;
        assert!((me.log_local_degeneracy(0) - 0.0).abs() < 1e-12);
        assert!((me.log_local_degeneracy(1) + 0.0).abs() < 1e-12); // 1/1! = 1
        assert!((me.log_local_degeneracy(2) - (-2.0_f64.ln())).abs() < 1e-12); // 1/2!
        assert!((me.log_local_degeneracy(3) - (-(6.0_f64).ln())).abs() < 1e-12);
        // 1/3!
    }

    #[test]
    fn small_exact_b_degeneracy() {
        // d_B(t) = C(M, t)
        let b = OccupationFamily::Binomial(4);
        assert!((b.log_local_degeneracy(0) - 0.0).abs() < 1e-12); // C(4,0)=1
        assert!((b.log_local_degeneracy(1) - (4.0_f64).ln()).abs() < 1e-12); // C(4,1)=4
        assert!((b.log_local_degeneracy(2) - (6.0_f64).ln()).abs() < 1e-12); // C(4,2)=6
        assert!((b.log_local_degeneracy(4) - 0.0).abs() < 1e-12); // C(4,4)=1
                                                                  // Above capacity: zero degeneracy
        assert!(b.log_local_degeneracy(5).is_infinite());
        assert!(b.log_local_degeneracy(5) < 0.0);
    }

    #[test]
    fn small_exact_w_degeneracy() {
        // d_W(t) = C(M+t-1, t); M=3: C(2+t, t)
        let w = OccupationFamily::NegativeBinomial(3);
        assert!((w.log_local_degeneracy(0) - 0.0).abs() < 1e-12); // C(2,0)=1
        assert!((w.log_local_degeneracy(1) - (3.0_f64).ln()).abs() < 1e-12); // C(3,1)=3
        assert!((w.log_local_degeneracy(2) - (6.0_f64).ln()).abs() < 1e-12); // C(4,2)=6
        assert!((w.log_local_degeneracy(3) - (10.0_f64).ln()).abs() < 1e-12); // C(5,3)=10
                                                                              // M=1 (geometric): d(t) = C(t, t) = 1
        let g = OccupationFamily::Geometric;
        assert!((g.log_local_degeneracy(0) - 0.0).abs() < 1e-12);
        assert!((g.log_local_degeneracy(17) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn delta_equals_full_difference() {
        for family in [
            OccupationFamily::Poisson,
            OccupationFamily::Geometric,
            OccupationFamily::Binomial(5),
            OccupationFamily::NegativeBinomial(3),
        ] {
            for old in 0..6 {
                for new in 0..6 {
                    let delta = family.delta_log_local_degeneracy(old, new);
                    let full = family.log_local_degeneracy(new) - family.log_local_degeneracy(old);
                    assert!(
                        (delta - full).abs() < 1e-10,
                        "{family:?} old={old} new={new}: {delta} vs {full}"
                    );
                }
            }
        }
    }

    #[test]
    fn large_occupation_stability() {
        // No overflow/NaN for large occupations (stable log-gamma).
        let me = OccupationFamily::Poisson;
        let large = 10_000_u64;
        let v = me.log_local_degeneracy(large);
        assert!(v.is_finite());
        // ME ratio: d(t+1)/d(t) = 1/(t+1) -> ln = -ln(t+1)
        let ratio = me.delta_log_local_degeneracy(large, large + 1);
        assert!((ratio - (-((large + 1) as f64).ln())).abs() < 1e-6);

        let w = OccupationFamily::NegativeBinomial(50);
        assert!(w.log_local_degeneracy(100_000).is_finite());
    }

    #[test]
    fn consistency_with_grand_canonical_pmf() {
        // The unnormalized family weight W(t) = d_F(t) * q^t must match the
        // grand-canonical PMF shape: for Poisson, d(t) q^t = q^t / t!.
        let q = 2.5_f64;
        let me = OccupationFamily::Poisson;
        let w0 = (me.log_local_degeneracy(0) + 0.0 * q.ln()).exp();
        let w1 = (me.log_local_degeneracy(1) + 1.0 * q.ln()).exp();
        let w2 = (me.log_local_degeneracy(2) + 2.0 * q.ln()).exp();
        let w3 = (me.log_local_degeneracy(3) + 3.0 * q.ln()).exp();
        // ratio w1/w0 = q, w2/w1 = q/2, w3/w2 = q/3 (Poisson recurrence)
        assert!((w1 / w0 - q).abs() < 1e-10);
        assert!((w2 / w1 - q / 2.0).abs() < 1e-10);
        assert!((w3 / w2 - q / 3.0).abs() < 1e-10);
    }

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
            assert!(w <= 5, "occ_num {w} exceeds layers 5");
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
            "expected some positive occupations from ZeroInflatedGeometric"
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
        let dist = OccupationFamily::Poisson.distribution(2.0);
        assert!((dist.expected() - 2.0).abs() < 1e-12);

        let dist = OccupationFamily::Geometric.distribution(0.5);
        assert!((dist.expected() - 1.0).abs() < 1e-12);

        let dist = OccupationFamily::Binomial(10).distribution(0.5);
        assert!((dist.expected() - 10.0 / 3.0).abs() < 1e-10);

        let dist = OccupationFamily::NegativeBinomial(3).distribution(0.4);
        assert!((dist.expected() - 2.0).abs() < 1e-12);
    }
}
