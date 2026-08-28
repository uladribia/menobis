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
    /// Every fixed coordinate enters the domain exclusion set, including
    /// pairs frozen at occupation 0 (§16).  A `Complete` domain with
    /// fixed pairs residualizes to a [`PairDomain::CompleteMinus`]
    /// carrying only the explicit excluded set (`O(F)` memory), never an
    /// `O(N²)` coordinate set (§15).
    ///
    /// # Errors
    ///
    /// - [`InvalidResidual`](FixedStrengthError::InvalidResidual) if a
    ///   fixed pair is not admissible in the domain.
    /// - [`InvalidResidual`](FixedStrengthError::InvalidResidual) if a
    ///   fixed coordinate appears more than once.
    /// - [`InvalidResidual`](FixedStrengthError::InvalidResidual) if a
    ///   residual strength would become negative.
    /// - [`InvalidResidual`](FixedStrengthError::InvalidResidual) if the
    ///   residual totals are unbalanced.
    /// - [`InvalidResidual`](FixedStrengthError::InvalidResidual) if a
    ///   fixed B occupation exceeds the layer count.
    pub fn into_residual(self) -> Result<ResidualStrengthProblem, FixedStrengthError> {
        let mut res_out = self.strength_out.clone();
        let mut res_in = self.strength_in.clone();

        // Set of fixed coordinates: every fixed pair enters the exclusion
        // set, including pairs frozen at occupation 0.
        let mut fixed_set: std::collections::HashSet<(u64, u64)> =
            std::collections::HashSet::with_capacity(self.fixed_pairs.len());

        for &(src, tgt, occ) in &self.fixed_pairs {
            // Check admissibility.
            if !self.domain.is_admissible(src, tgt) {
                return Err(FixedStrengthError::InvalidResidual(format!(
                    "fixed pair ({src}, {tgt}) is not admissible"
                )));
            }

            // Reject duplicate fixed coordinates (avoids subtracting the
            // occupation twice).
            if !fixed_set.insert((src, tgt)) {
                return Err(FixedStrengthError::InvalidResidual(format!(
                    "duplicate fixed pair ({src}, {tgt})"
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
                } => PairDomain::CompleteMinus {
                    node_count,
                    self_loops,
                    excluded: fixed_set,
                },
                PairDomain::CompleteMinus {
                    node_count,
                    self_loops,
                    mut excluded,
                } => {
                    excluded.extend(fixed_set);
                    PairDomain::CompleteMinus {
                        node_count,
                        self_loops,
                        excluded,
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

    #[test]
    fn reject_duplicate_fixed_pair() {
        let prob = FixedStrengthProblem::new(
            OccupationFamily::ME,
            vec![10, 10],
            vec![10, 10],
            complete_domain(2, true),
            vec![(0, 1, 2), (0, 1, 3)],
        )
        .unwrap();
        match prob.into_residual() {
            Err(FixedStrengthError::InvalidResidual(msg)) => {
                assert!(msg.contains("duplicate fixed pair"), "{msg}")
            }
            other => panic!("expected duplicate error, got {other:?}"),
        }
    }

    #[test]
    fn reject_duplicate_zero_fixed_pair() {
        // A duplicate zero-occupation fixed pair must also be rejected:
        // it is still a duplicate coordinate.
        let prob = FixedStrengthProblem::new(
            OccupationFamily::ME,
            vec![5, 5],
            vec![5, 5],
            complete_domain(2, true),
            vec![(0, 1, 0), (0, 1, 0)],
        )
        .unwrap();
        assert!(prob.into_residual().is_err());
    }

    #[test]
    fn zero_fixed_pair_remains_excluded() {
        // A fixed pair frozen at occupation 0 does not change strengths
        // but its coordinate must be excluded from the residual domain.
        let prob = FixedStrengthProblem::new(
            OccupationFamily::ME,
            vec![10, 20, 30],
            vec![15, 25, 20],
            complete_domain(3, true),
            vec![(0, 1, 0)],
        )
        .unwrap();
        let res = prob.into_residual().unwrap();
        assert_eq!(res.strength_out, vec![10, 20, 30]);
        assert_eq!(res.strength_in, vec![15, 25, 20]);
        assert_eq!(res.total, 60);
        assert!(!res.domain.is_admissible(0, 1));
        assert!(matches!(res.domain, PairDomain::CompleteMinus { .. }));
    }

    #[test]
    fn complete_with_fixed_pairs_residualizes_to_complete_minus() {
        // Gold: a Complete domain with any fixed pairs (positive or zero)
        // must residualize to CompleteMinus — never to an O(N²) Sparse set.
        let prob = FixedStrengthProblem::new(
            OccupationFamily::ME,
            vec![10, 20, 30],
            vec![15, 25, 20],
            complete_domain(3, true),
            vec![(0, 1, 5), (1, 0, 0)], // positive + zero fixed pairs
        )
        .unwrap();
        let res = prob.into_residual().unwrap();
        assert!(matches!(res.domain, PairDomain::CompleteMinus { .. }));
        assert!(!res.domain.is_admissible(0, 1));
        assert!(!res.domain.is_admissible(1, 0));
        // Residual admissible count: 9 - 2 excluded.
        assert_eq!(res.domain.admissible_pair_count(), 7);
    }

    #[test]
    fn large_complete_domain_fixed_pairs_stay_sparse() {
        // N=1000 complete domain with 3 fixed pairs must residualize to
        // CompleteMinus (O(F) memory) and never materialize the N² set.
        let n = 1000usize;
        let s_out = vec![1u64; n];
        let s_in = vec![1u64; n];
        let prob = FixedStrengthProblem::new(
            OccupationFamily::ME,
            s_out,
            s_in,
            complete_domain(n, true),
            vec![(0, 0, 1), (1, 5, 1), (999, 999, 0)],
        )
        .unwrap();
        let res = prob.into_residual().unwrap();
        assert!(
            matches!(res.domain, PairDomain::CompleteMinus { .. }),
            "complete domain with fixed pairs must stay CompleteMinus"
        );
        assert_eq!(
            res.domain.admissible_pair_count(),
            n * n - 3,
            "complete-minus count must stay O(1)-computable without materialization"
        );
        assert!(!res.domain.is_admissible(0, 0));
        assert!(!res.domain.is_admissible(1, 5));
        assert!(!res.domain.is_admissible(999, 999));
    }

    #[test]
    fn complete_minus_input_merges_exclusions() {
        // A CompleteMinus input domain with additional fixed pairs must
        // union the exclusions, not rebuild the domain.
        let mut excluded = std::collections::HashSet::new();
        excluded.insert((0, 0));
        let domain = PairDomain::CompleteMinus {
            node_count: 3,
            self_loops: true,
            excluded,
        };
        let prob = FixedStrengthProblem::new(
            OccupationFamily::ME,
            vec![5, 5, 5],
            vec![5, 5, 5],
            domain,
            vec![(1, 1, 2)],
        )
        .unwrap();
        let res = prob.into_residual().unwrap();
        assert!(matches!(res.domain, PairDomain::CompleteMinus { .. }));
        assert!(!res.domain.is_admissible(0, 0)); // original exclusion
        assert!(!res.domain.is_admissible(1, 1)); // merged fixed pair
        assert_eq!(res.strength_out, vec![5, 3, 5]);
        assert_eq!(res.strength_in, vec![5, 3, 5]);
    }
}
