//! Base-measure family abstraction for MENoBiS occupation-number models.
//!
//! Defines the three occupation families — ME (MultiEdge), B (Aggregated
//! Binary), and W (Weighted) — through their local degeneracy (base
//! measure) and support.  This is the single source of truth for
//! microcanonical and canonical degeneracy formulas.
//!
//! # Mapping
//!
//! | Family | GC distribution | Degeneracy |
//! |--------|-----------------|------------|
//! | ME     | Poisson         | `1 / t!`   |
//! | B(M)   | Binomial(M)     | `C(M, t)`  |
//! | W(M)   | NegBin(M)       | `C(M+t-1,t)`|

use crate::OccNum;

/// Thesis model-family base measure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccupationFamily {
    /// MultiEdge: d_ME(t) = 1/t!.
    ME,
    /// Aggregated binary: d_B,M(t) = C(M, t), 0 ≤ t ≤ M.
    B { layers: u32 },
    /// Weighted: d_W,M(t) = C(M+t-1, t).
    W { layers: u32 },
}

impl OccupationFamily {
    /// Maximum admissible occupation number per pair (None = unbounded).
    pub fn max_occupation(&self) -> Option<OccNum> {
        match self {
            Self::ME | Self::W { .. } => None,
            Self::B { layers } => Some(*layers as OccNum),
        }
    }

    /// Log of the local base measure (degeneracy) for occupation `t`.
    ///
    /// - ME: `ln d(t) = -ln Γ(t+1)`
    /// - B:  `ln d(t) = ln C(M, t)` (zero degeneracy for t > M)
    /// - W:  `ln d(t) = ln C(M+t-1, t)`
    pub fn log_base_measure(&self, t: OccNum) -> f64 {
        let tf = t as f64;
        match self {
            Self::ME => -libm::lgamma(tf + 1.0),
            Self::B { layers } => {
                if t > *layers as OccNum {
                    return f64::NEG_INFINITY;
                }
                let mf = *layers as f64;
                libm::lgamma(mf + 1.0) - libm::lgamma(tf + 1.0) - libm::lgamma(mf - tf + 1.0)
            }
            Self::W { layers } => {
                let mf = *layers as f64;
                libm::lgamma(mf + tf) - libm::lgamma(tf + 1.0) - libm::lgamma(mf)
            }
        }
    }

    /// Whether `occ_num` is a valid occupation number for this family.
    ///
    /// B rejects occupation numbers above its layer capacity `M`.
    pub fn validate_occnum(&self, occ_num: OccNum) -> bool {
        match self {
            Self::B { layers } => occ_num <= *layers as OccNum,
            _ => true,
        }
    }

    /// Layer count `M` for B and W families; `None` for ME.
    pub fn layers(&self) -> Option<u32> {
        match self {
            Self::ME => None,
            Self::B { layers } | Self::W { layers } => Some(*layers),
        }
    }

    /// Difference of log base measures: `ln d(new) − ln d(old)`.
    ///
    /// Used for local Metropolis acceptance.  Defaults to computing the
    /// full difference; subclasses or future specialisation may replace
    /// this with a cheaper ratio.
    pub fn delta_log_base_measure(&self, old: OccNum, new: OccNum) -> f64 {
        self.log_base_measure(new) - self.log_base_measure(old)
    }
}

// Conversion to the existing distribution-level enum for backward
// compatibility during the refactor.  Will be removed once all callers
// are migrated to the new family type.
impl From<OccupationFamily> for crate::distribution::OccupationFamily {
    fn from(family: OccupationFamily) -> Self {
        match family {
            OccupationFamily::ME => Self::Poisson,
            OccupationFamily::B { layers } => Self::Binomial(layers),
            OccupationFamily::W { layers: 1 } => Self::Geometric,
            OccupationFamily::W { layers } => Self::NegativeBinomial(layers),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn me_log_base(t: u64) -> f64 {
        -libm::lgamma(t as f64 + 1.0)
    }

    fn b_log_base(t: u64, m: u32) -> f64 {
        if t > m as u64 {
            return f64::NEG_INFINITY;
        }
        let mf = m as f64;
        libm::lgamma(mf + 1.0) - libm::lgamma(t as f64 + 1.0) - libm::lgamma(mf - t as f64 + 1.0)
    }

    fn w_log_base(t: u64, m: u32) -> f64 {
        let mf = m as f64;
        libm::lgamma(mf + t as f64) - libm::lgamma(t as f64 + 1.0) - libm::lgamma(mf)
    }

    #[test]
    fn me_values() {
        let f = OccupationFamily::ME;
        assert!((f.log_base_measure(0) - me_log_base(0)).abs() < 1e-12);
        assert!((f.log_base_measure(1) - me_log_base(1)).abs() < 1e-12);
        assert!((f.log_base_measure(5) - me_log_base(5)).abs() < 1e-12);
    }

    #[test]
    fn b_values() {
        let f = OccupationFamily::B { layers: 4 };
        for t in 0..=4 {
            assert!((f.log_base_measure(t) - b_log_base(t, 4)).abs() < 1e-12);
        }
        assert!(f.log_base_measure(5).is_infinite() && f.log_base_measure(5) < 0.0);
    }

    #[test]
    fn w_values() {
        let f = OccupationFamily::W { layers: 3 };
        for t in 0..5u64 {
            assert!((f.log_base_measure(t) - w_log_base(t, 3)).abs() < 1e-12);
        }
    }

    #[test]
    fn delta_equals_full_difference() {
        for family in [
            OccupationFamily::ME,
            OccupationFamily::B { layers: 5 },
            OccupationFamily::W { layers: 3 },
            OccupationFamily::W { layers: 1 },
        ] {
            for old in 0..6u64 {
                for new in 0..6u64 {
                    let delta = family.delta_log_base_measure(old, new);
                    let full = family.log_base_measure(new) - family.log_base_measure(old);
                    assert!(
                        (delta - full).abs() < 1e-10,
                        "{family:?}: {delta} vs {full}"
                    );
                }
            }
        }
    }

    #[test]
    fn max_occupation() {
        assert_eq!(OccupationFamily::ME.max_occupation(), None);
        assert_eq!(OccupationFamily::W { layers: 3 }.max_occupation(), None);
        assert_eq!(OccupationFamily::B { layers: 7 }.max_occupation(), Some(7));
    }

    #[test]
    fn conversion_to_distribution_family() {
        use crate::distribution::OccupationFamily as DistFam;
        assert_eq!(DistFam::from(OccupationFamily::ME), DistFam::Poisson);
        assert_eq!(
            DistFam::from(OccupationFamily::B { layers: 5 }),
            DistFam::Binomial(5)
        );
        assert_eq!(
            DistFam::from(OccupationFamily::W { layers: 1 }),
            DistFam::Geometric
        );
        assert_eq!(
            DistFam::from(OccupationFamily::W { layers: 3 }),
            DistFam::NegativeBinomial(3)
        );
    }
}
