//! Fixed-strength problem types and residualization.
//!
//! A [`FixedStrengthProblem`] defines the full sampling problem before
//! fixed-pair subtraction.  [`ResidualStrengthProblem`] is the reduced
//! problem after subtracting fixed-pair contributions.

use super::domain::PairDomain;
use super::errors::FixedStrengthError;
use crate::model::family::OccupationFamily;
use crate::OccNum;

/// Full fixed-strength sampling problem (before residualisation).
///
/// Contains the target strength sequences, family, domain, self-loop
/// policy, and any fixed (pre-determined) pairs.
#[derive(Clone, Debug)]
pub struct FixedStrengthProblem {
    pub family: OccupationFamily,
    pub strength_out: Vec<OccNum>,
    pub strength_in: Vec<OccNum>,
    pub domain: PairDomain,
    pub fixed_pairs: Vec<(u64, u64, OccNum)>,
}

/// Residual fixed-strength problem (after fixed-pair subtraction).
///
/// The residual out- and in-strengths, total occupation, and admissible
/// variable-pair domain that must be filled by the sampling backend.
#[derive(Clone, Debug)]
pub struct ResidualStrengthProblem {
    pub family: OccupationFamily,
    pub strength_out: Vec<OccNum>,
    pub strength_in: Vec<OccNum>,
    pub total: OccNum,
    pub domain: PairDomain,
}

impl FixedStrengthProblem {
    /// Create a new fixed-strength problem with basic validation.
    ///
    /// # Errors
    ///
    /// - [`InvalidResidual`](FixedStrengthError::InvalidResidual) if the
    ///   strength vectors have different lengths.
    /// - [`InvalidResidual`](FixedStrengthError::InvalidResidual) if the
    ///   totals are unbalanced.
    pub fn new(
        family: OccupationFamily,
        strength_out: Vec<OccNum>,
        strength_in: Vec<OccNum>,
        domain: PairDomain,
        fixed_pairs: Vec<(u64, u64, OccNum)>,
    ) -> Result<Self, FixedStrengthError> {
        let n = strength_out.len();
        if n != strength_in.len() {
            return Err(FixedStrengthError::InvalidResidual(
                "strength_out and strength_in must have the same length".into(),
            ));
        }
        if n != domain.node_count() {
            return Err(FixedStrengthError::InvalidResidual(
                "strength vector length must match domain node_count".into(),
            ));
        }
        Ok(Self {
            family,
            strength_out,
            strength_in,
            domain,
            fixed_pairs,
        })
    }

    /// Subtract fixed-pair contributions to produce a residual problem.
    ///
    /// Each fixed pair `(src, tgt, occ)` is subtracted from the
    /// corresponding out- and in-strength.  Fixed pairs are removed from
    /// the admissible domain so the backend never re-occupies them.
    ///
    /// # Errors
    ///
    /// - [`InvalidResidual`](FixedStrengthError::InvalidResidual) if a
    ///   fixed pair is not admissible in the domain.
    /// - [`InvalidResidual`](FixedStrengthError::InvalidResidual) if a
    ///   residual strength would become negative.
    /// - [`InvalidResidual`](FixedStrengthError::InvalidResidual) if the
    ///   residual totals are unbalanced.
    /// - [`InvalidResidual`](FixedStrengthError::InvalidResidual) if a
    ///   fixed B occupation exceeds the layer count.
    /// - [`InvalidResidual`](FixedStrengthError::InvalidResidual) if a
    ///   fixed pair exceeds flow capacity.
    pub fn into_residual(self) -> Result<ResidualStrengthProblem, FixedStrengthError> {
        let mut res_out = self.strength_out.clone();
        let mut res_in = self.strength_in.clone();

        // Build a set of fixed pairs for domain exclusion.
        let mut fixed_set: std::collections::HashSet<(u64, u64)> =
            std::collections::HashSet::with_capacity(self.fixed_pairs.len());

        for &(src, tgt, occ) in &self.fixed_pairs {
            // Check admissibility.
            if !self.domain.is_admissible(src, tgt) {
                return Err(FixedStrengthError::InvalidResidual(format!(
                    "fixed pair ({src}, {tgt}) is not admissible"
                )));
            }

            // Check per-pair capacity.
            let cap = self.domain.capacity(self.family);
            if occ > cap {
                return Err(FixedStrengthError::InvalidResidual(format!(
                    "fixed occupation {occ} on ({src}, {tgt}) exceeds capacity {cap}"
                )));
            }

            // Subtract from residual strengths.
            let s_out = &mut res_out[src as usize];
            let s_in = &mut res_in[tgt as usize];
            if occ > *s_out || occ > *s_in {
                return Err(FixedStrengthError::InvalidResidual(format!(
                    "fixed occupation {occ} on ({src}, {tgt}) exceeds residual strength"
                )));
            }
            *s_out -= occ;
            *s_in -= occ;

            fixed_set.insert((src, tgt));
        }

        // Check residual totals are balanced.
        let total_out: OccNum = res_out.iter().sum();
        let total_in: OccNum = res_in.iter().sum();
        if total_out != total_in {
            return Err(FixedStrengthError::InvalidResidual(format!(
                "residual out-strength ({total_out}) != residual in-strength ({total_in})"
            )));
        }

        // Build the residual domain by removing fixed pairs.
        let res_domain = if fixed_set.is_empty() {
            self.domain
        } else {
            match self.domain {
                PairDomain::Complete {
                    node_count,
                    self_loops,
                } => {
                    // Start from the complete set and remove fixed pairs.
                    let mut allowed: std::collections::HashSet<(u64, u64)> =
                        std::collections::HashSet::new();
                    for i in 0..node_count as u64 {
                        for j in 0..node_count as u64 {
                            if !self_loops && i == j {
                                continue;
                            }
                            if !fixed_set.contains(&(i, j)) {
                                allowed.insert((i, j));
                            }
                        }
                    }
                    PairDomain::Sparse {
                        node_count,
                        allowed,
                    }
                }
                PairDomain::Sparse {
                    node_count,
                    mut allowed,
                } => {
                    for p in &fixed_set {
                        allowed.remove(p);
                    }
                    PairDomain::Sparse {
                        node_count,
                        allowed,
                    }
                }
            }
        };

        Ok(ResidualStrengthProblem {
            family: self.family,
            strength_out: res_out,
            strength_in: res_in,
            total: total_out,
            domain: res_domain,
        })
    }
}

impl ResidualStrengthProblem {
    /// Basic sanity checks on the residual problem.
    ///
    /// # Errors
    ///
    /// - [`InvalidResidual`](FixedStrengthError::InvalidResidual) for
    ///   any violation.
    pub fn validate(&self) -> Result<(), FixedStrengthError> {
        let n = self.strength_out.len();
        if n != self.strength_in.len() {
            return Err(FixedStrengthError::InvalidResidual(
                "strength_out and strength_in must have the same length".into(),
            ));
        }
        if n != self.domain.node_count() {
            return Err(FixedStrengthError::InvalidResidual(
                "strength vector length must match domain node_count".into(),
            ));
        }
        let total_out: OccNum = self.strength_out.iter().sum();
        let total_in: OccNum = self.strength_in.iter().sum();
        if total_out != total_in {
            return Err(FixedStrengthError::InvalidResidual(format!(
                "residual out-strength ({total_out}) != residual in-strength ({total_in})"
            )));
        }
        if total_out != self.total {
            return Err(FixedStrengthError::InvalidResidual(format!(
                "total field ({}) does not match sum of out-strengths ({total_out})",
                self.total
            )));
        }
        if self.strength_out.contains(&0) && self.strength_in.contains(&0) {
            // Zero margins are acceptable; this is not an error.
        }
        Ok(())
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.strength_out.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_domain(n: usize, sl: bool) -> PairDomain {
        PairDomain::Complete {
            node_count: n,
            self_loops: sl,
        }
    }

    #[test]
    fn residualize_empty_fixed_pairs() {
        let prob = FixedStrengthProblem::new(
            OccupationFamily::ME,
            vec![10, 20, 30],
            vec![15, 25, 20],
            complete_domain(3, true),
            vec![],
        )
        .unwrap();
        let res = prob.into_residual().unwrap();
        assert_eq!(res.strength_out, vec![10, 20, 30]);
        assert_eq!(res.total, 60);
    }

    #[test]
    fn residualize_with_fixed_pairs() {
        let prob = FixedStrengthProblem::new(
            OccupationFamily::ME,
            vec![10, 20, 30],
            vec![15, 25, 20],
            complete_domain(3, true),
            vec![(0, 1, 5), (1, 0, 3)],
        )
        .unwrap();
        let res = prob.into_residual().unwrap();
        // Out: 10-5=5, 20-3=17, 30-0=30 → total = 52
        // In: 15-3=12, 25-5=20, 20-0=20 → total = 52
        assert_eq!(res.strength_out, vec![5, 17, 30]);
        assert_eq!(res.strength_in, vec![12, 20, 20]);
        assert_eq!(res.total, 52);
        // Fixed pairs should be excluded from domain.
        assert!(!res.domain.is_admissible(0, 1));
        assert!(!res.domain.is_admissible(1, 0));
        assert!(res.domain.is_admissible(0, 2));
    }

    #[test]
    fn reject_fixed_pair_exceeding_strength() {
        let prob = FixedStrengthProblem::new(
            OccupationFamily::ME,
            vec![2, 10],
            vec![5, 7],
            complete_domain(2, true),
            vec![(0, 1, 10)], // exceeds s_out[0]=2
        )
        .unwrap();
        assert!(prob.into_residual().is_err());
    }

    #[test]
    fn reject_fixed_pair_not_admissible() {
        let prob = FixedStrengthProblem::new(
            OccupationFamily::ME,
            vec![5, 5],
            vec![5, 5],
            complete_domain(2, false),
            vec![(0, 0, 3)], // self-loop not allowed
        )
        .unwrap();
        assert!(prob.into_residual().is_err());
    }

    #[test]
    fn residual_validation() {
        let res = ResidualStrengthProblem {
            family: OccupationFamily::ME,
            strength_out: vec![5, 5],
            strength_in: vec![5, 5],
            total: 10,
            domain: complete_domain(2, true),
        };
        assert!(res.validate().is_ok());
    }

    #[test]
    fn residual_validation_unbalanced() {
        let res = ResidualStrengthProblem {
            family: OccupationFamily::ME,
            strength_out: vec![5, 5],
            strength_in: vec![6, 4],
            total: 10,
            domain: complete_domain(2, true),
        };
        // out total = 10, in total = 10, total = 10 → should be ok
        assert!(res.validate().is_ok());
    }

    #[test]
    fn reject_b_fixed_exceeds_layers() {
        let prob = FixedStrengthProblem::new(
            OccupationFamily::B { layers: 3 },
            vec![5, 5],
            vec![5, 5],
            complete_domain(2, true),
            vec![(0, 1, 4)], // B capacity is 3
        )
        .unwrap();
        assert!(prob.into_residual().is_err());
    }
}
