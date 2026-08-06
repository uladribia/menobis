//! Pair domain abstraction for fixed-strength sampling.
//!
//! A [`PairDomain`] defines which ordered pairs are admissible and what
//! per-pair capacity the family imposes.  It serves:
//!
//! - Feasibility (max-flow edge construction).
//! - Initializer (greedy table construction).
//! - MCMC cycle validation (admissibility and capacity checks).
//!
//! Only used by the `fixed_strength` module; `fixed_kt` and `fixed_et`
//! keep their own simpler `Option<&[(u64,u64)]>` representation.

use std::collections::HashSet;

use crate::distribution::OccupationFamily;
use crate::OccNum;

/// Domain of admissible ordered pairs for a fixed-strength problem.
///
/// # Variants
///
/// - [`Complete`](PairDomain::Complete): all ordered pairs (optionally
///   excluding the diagonal).
/// - [`Sparse`](PairDomain::Sparse): an explicit set of allowed pairs.
#[derive(Clone, Debug)]
pub enum PairDomain {
    /// All `N × N` ordered pairs, optionally excluding self-loops.
    Complete { node_count: usize, self_loops: bool },
    /// An explicit set of admissible pairs.
    Sparse {
        node_count: usize,
        allowed: HashSet<(u64, u64)>,
    },
}

impl PairDomain {
    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        match self {
            PairDomain::Complete { node_count, .. } => *node_count,
            PairDomain::Sparse { node_count, .. } => *node_count,
        }
    }

    /// Whether self-loops are allowed.
    pub fn self_loops_allowed(&self) -> bool {
        match self {
            PairDomain::Complete { self_loops, .. } => *self_loops,
            PairDomain::Sparse { allowed, .. } => {
                // Self-loops are allowed if any (i,i) is in the set.
                allowed.iter().any(|(s, t)| s == t)
            }
        }
    }

    /// Returns `true` if the ordered pair `(src, tgt)` is admissible.
    #[inline]
    pub fn is_admissible(&self, src: u64, tgt: u64) -> bool {
        match self {
            PairDomain::Complete {
                node_count,
                self_loops,
            } => {
                if !self_loops && src == tgt {
                    return false;
                }
                (src as usize) < *node_count && (tgt as usize) < *node_count
            }
            PairDomain::Sparse { allowed, .. } => allowed.contains(&(src, tgt)),
        }
    }

    /// Maximum occupation per admissible pair for the given family.
    ///
    /// For ME and W this is unbounded (`OccNum::MAX`).
    /// For B this is the layer count `M`.
    #[inline]
    pub fn capacity(&self, family: OccupationFamily) -> OccNum {
        match family {
            OccupationFamily::Binomial(m) => m as OccNum,
            _ => OccNum::MAX,
        }
    }

    /// Maximum flow-network capacity for edge `src → tng` given residual
    /// marginals.  Used by the transportation max-flow initializer.
    #[inline]
    pub fn flow_capacity(
        &self,
        family: OccupationFamily,
        src_out_res: OccNum,
        tgt_in_res: OccNum,
    ) -> OccNum {
        let cap = self.capacity(family);
        cap.min(src_out_res).min(tgt_in_res)
    }

    /// Iterate over all admissible ordered pairs.
    ///
    /// For complete domains this materialises the iterator and may be
    /// expensive for large N.  Use only during initialisation.
    pub fn iter_admissible(&self) -> Box<dyn Iterator<Item = (u64, u64)> + '_> {
        match self {
            PairDomain::Complete {
                node_count,
                self_loops,
            } => {
                let n = *node_count;
                let sl = *self_loops;
                Box::new((0..n as u64).flat_map(move |i| {
                    (0..n as u64).filter_map(
                        move |j| {
                            if !sl && i == j {
                                None
                            } else {
                                Some((i, j))
                            }
                        },
                    )
                }))
            }
            PairDomain::Sparse { allowed, .. } => {
                let pairs: Vec<_> = allowed.iter().copied().collect();
                Box::new(pairs.into_iter())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_admits_all_ordered_pairs() {
        let d = PairDomain::Complete {
            node_count: 3,
            self_loops: true,
        };
        assert!(d.is_admissible(0, 0));
        assert!(d.is_admissible(0, 1));
        assert!(d.is_admissible(2, 1));
        assert!(!d.is_admissible(3, 0)); // out of range
    }

    #[test]
    fn no_self_loops() {
        let d = PairDomain::Complete {
            node_count: 3,
            self_loops: false,
        };
        assert!(!d.is_admissible(0, 0));
        assert!(d.is_admissible(0, 1));
        assert!(d.is_admissible(1, 0));
    }

    #[test]
    fn sparse_domain() {
        let mut allowed = HashSet::new();
        allowed.insert((0, 1));
        allowed.insert((1, 0));
        let d = PairDomain::Sparse {
            node_count: 2,
            allowed,
        };
        assert!(d.is_admissible(0, 1));
        assert!(d.is_admissible(1, 0));
        assert!(!d.is_admissible(0, 0));
        assert!(!d.is_admissible(1, 1));
    }

    #[test]
    fn b_capacity_is_layers() {
        let d = PairDomain::Complete {
            node_count: 5,
            self_loops: true,
        };
        assert_eq!(d.capacity(OccupationFamily::Binomial(3)), 3);
        assert_eq!(d.capacity(OccupationFamily::Binomial(10)), 10);
    }

    #[test]
    fn me_w_capacity_unbounded() {
        let d = PairDomain::Complete {
            node_count: 5,
            self_loops: true,
        };
        assert_eq!(d.capacity(OccupationFamily::Poisson), OccNum::MAX);
        assert_eq!(d.capacity(OccupationFamily::Geometric), OccNum::MAX);
        assert_eq!(
            d.capacity(OccupationFamily::NegativeBinomial(3)),
            OccNum::MAX
        );
    }

    #[test]
    fn flow_capacity_min_of_marginals() {
        let d = PairDomain::Complete {
            node_count: 5,
            self_loops: true,
        };
        // ME: capacity = min(src_out_res, tgt_in_res) (since no family cap)
        assert_eq!(d.flow_capacity(OccupationFamily::Poisson, 5, 3), 3);
        assert_eq!(d.flow_capacity(OccupationFamily::Poisson, 2, 10), 2);
        // B: capacity = min(M, src_out_res, tgt_in_res)
        assert_eq!(d.flow_capacity(OccupationFamily::Binomial(4), 5, 3), 3);
        assert_eq!(d.flow_capacity(OccupationFamily::Binomial(4), 2, 10), 2);
        assert_eq!(d.flow_capacity(OccupationFamily::Binomial(2), 5, 3), 2);
    }

    #[test]
    fn iter_admissible_complete_count() {
        let d = PairDomain::Complete {
            node_count: 4,
            self_loops: true,
        };
        let count = d.iter_admissible().count();
        assert_eq!(count, 16); // N^2 = 16
    }

    #[test]
    fn iter_admissible_loopless_count() {
        let d = PairDomain::Complete {
            node_count: 4,
            self_loops: false,
        };
        let count = d.iter_admissible().count();
        assert_eq!(count, 12); // N^2 - N = 12
    }
}
