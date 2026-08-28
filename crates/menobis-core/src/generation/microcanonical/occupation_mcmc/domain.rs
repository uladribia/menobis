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

use crate::model::family::OccupationFamily;
use crate::OccNum;

/// Domain of admissible ordered pairs for a fixed-strength problem.
///
/// # Variants
///
/// - [`Complete`](PairDomain::Complete): all ordered pairs (optionally
///   excluding the diagonal).
/// - [`CompleteMinus`](PairDomain::CompleteMinus): the complete/self-loop
///   policy minus a small explicit excluded set (fixed coordinates).
///   Memory is `O(F)` where `F` is the number of excluded pairs — the
///   `N × N` coordinate set is never materialized.
/// - [`Sparse`](PairDomain::Sparse): an explicit set of allowed pairs.
#[derive(Clone, Debug)]
pub enum PairDomain {
    /// All `N × N` ordered pairs, optionally excluding self-loops.
    Complete { node_count: usize, self_loops: bool },
    /// All ordered pairs allowed by the complete/self-loop policy minus a
    /// small explicit excluded set (e.g., fixed-pair coordinates).
    CompleteMinus {
        node_count: usize,
        self_loops: bool,
        excluded: HashSet<(u64, u64)>,
    },
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
            PairDomain::Complete { node_count, .. }
            | PairDomain::CompleteMinus { node_count, .. }
            | PairDomain::Sparse { node_count, .. } => *node_count,
        }
    }

    /// Whether self-loops are allowed.
    pub fn self_loops_allowed(&self) -> bool {
        match self {
            PairDomain::Complete { self_loops, .. }
            | PairDomain::CompleteMinus { self_loops, .. } => *self_loops,
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
            PairDomain::CompleteMinus {
                node_count,
                self_loops,
                excluded,
            } => {
                if !self_loops && src == tgt {
                    return false;
                }
                if (src as usize) >= *node_count || (tgt as usize) >= *node_count {
                    return false;
                }
                !excluded.contains(&(src, tgt))
            }
            PairDomain::Sparse { allowed, .. } => allowed.contains(&(src, tgt)),
        }
    }

    /// Number of admissible ordered pairs.
    ///
    /// - [`Complete`](PairDomain::Complete): `O(1)` arithmetic.
    /// - [`CompleteMinus`](PairDomain::CompleteMinus): `O(1)` arithmetic
    ///   plus `excluded.len()` (validated residuals exclude only
    ///   admissible coordinates).
    /// - [`Sparse`](PairDomain::Sparse): `allowed.len()`.
    pub fn admissible_pair_count(&self) -> usize {
        match self {
            PairDomain::Complete {
                node_count,
                self_loops,
            } => complete_policy_count(*node_count, *self_loops),
            PairDomain::CompleteMinus {
                node_count,
                self_loops,
                excluded,
            } => {
                let base = complete_policy_count(*node_count, *self_loops);
                // Subtract only excluded coordinates that would otherwise be
                // admissible (in-range and allowed by the self-loop policy) —
                // consistent with [`PairDomain::admissible_degree_caps`], so
                // the per-node cap sums always equal this count.
                let subtracted = excluded
                    .iter()
                    .filter(|&&(s, t)| {
                        let (s, t) = (s as usize, t as usize);
                        (*self_loops || s != t) && s < *node_count && t < *node_count
                    })
                    .count();
                base.saturating_sub(subtracted)
            }
            PairDomain::Sparse { allowed, .. } => allowed.len(),
        }
    }

    /// Per-node admissible-coordinate caps `(out_caps, in_caps)`:
    /// `out_caps[i]` counts admissible residual coordinates `(i, ·)` and
    /// `in_caps[j]` counts admissible residual coordinates `(·, j)`.
    ///
    /// Used by the fixed-(s,k) degree-target validation (§10.6): the
    /// residual out/in degree of every node must fit in its admissible
    /// slot count.
    ///
    /// Complexity:
    ///
    /// - [`Complete`](PairDomain::Complete): `O(N)`.
    /// - [`CompleteMinus`](PairDomain::CompleteMinus): `O(N + F)` where
    ///   `F` is the excluded-coordinate count — never enumerates the
    ///   `N × N` grid.
    /// - [`Sparse`](PairDomain::Sparse): `O(N + A)` over explicit pairs.
    pub fn admissible_degree_caps(&self) -> (Vec<usize>, Vec<usize>) {
        match self {
            PairDomain::Complete {
                node_count,
                self_loops,
            } => {
                let cap = if *self_loops {
                    *node_count
                } else {
                    node_count.saturating_sub(1)
                };
                (vec![cap; *node_count], vec![cap; *node_count])
            }
            PairDomain::CompleteMinus {
                node_count,
                self_loops,
                excluded,
            } => {
                let mut out_caps = vec![0usize; *node_count];
                let mut in_caps = vec![0usize; *node_count];
                for i in 0..*node_count {
                    let base = if *self_loops {
                        *node_count
                    } else {
                        node_count.saturating_sub(1)
                    };
                    out_caps[i] = base;
                    in_caps[i] = base;
                }
                // An excluded coordinate reduces the row/column cap by 1
                // exactly when it would otherwise have been admissible
                // (in-range and allowed by the self-loop policy).
                for &(s, t) in excluded {
                    let (s, t) = (s as usize, t as usize);
                    let otherwise_admissible =
                        (*self_loops || s != t) && s < *node_count && t < *node_count;
                    if otherwise_admissible {
                        out_caps[s] = out_caps[s].saturating_sub(1);
                        in_caps[t] = in_caps[t].saturating_sub(1);
                    }
                }
                (out_caps, in_caps)
            }
            PairDomain::Sparse {
                node_count,
                allowed,
            } => {
                let mut out_caps = vec![0usize; *node_count];
                let mut in_caps = vec![0usize; *node_count];
                for &(s, t) in allowed {
                    let (s, t) = (s as usize, t as usize);
                    if s < *node_count {
                        out_caps[s] += 1;
                    }
                    if t < *node_count {
                        in_caps[t] += 1;
                    }
                }
                (out_caps, in_caps)
            }
        }
    }

    /// Whether the compressed constructor may leave mass on coordinates
    /// outside this domain and therefore needs the structural
    /// inadmissible-pair repair (spec §15.3, §19).
    ///
    /// - [`Sparse`](PairDomain::Sparse): always — the constructor ignores
    ///   the pair set.
    /// - [`CompleteMinus`](PairDomain::CompleteMinus): only when at least
    ///   one coordinate is excluded (fixed pairs).
    /// - [`Complete`](PairDomain::Complete): never — self-loop mass is
    ///   handled by the Phase D loop repair, and every in-range
    ///   coordinate is admissible.
    pub fn requires_admissibility_repair(&self) -> bool {
        match self {
            PairDomain::Sparse { .. } => true,
            PairDomain::CompleteMinus { excluded, .. } => !excluded.is_empty(),
            PairDomain::Complete { .. } => false,
        }
    }

    /// Maximum occupation per admissible pair for the given family.
    ///
    /// For ME and W this is unbounded (`OccNum::MAX`).
    /// For B this is the layer count `M`.
    #[inline]
    pub fn capacity(&self, family: OccupationFamily) -> OccNum {
        match family {
            OccupationFamily::B { layers: m } => m as OccNum,
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
    /// ⚠️ **Complete domains**: This materialises an N×N iterator,
    /// producing O(N²) pairs.  Only safe for small N or sparse domains.
    /// For large N with a sparse domain, the O(E_allowed) variant is
    /// used instead and is safe.
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
            PairDomain::CompleteMinus {
                node_count,
                self_loops,
                excluded,
            } => {
                let n = *node_count;
                let sl = *self_loops;
                let ex = excluded;
                Box::new((0..n as u64).flat_map(move |i| {
                    (0..n as u64).filter_map(move |j| {
                        if (!sl && i == j) || ex.contains(&(i, j)) {
                            None
                        } else {
                            Some((i, j))
                        }
                    })
                }))
            }
            PairDomain::Sparse { allowed, .. } => {
                let pairs: Vec<_> = allowed.iter().copied().collect();
                Box::new(pairs.into_iter())
            }
        }
    }
}

/// Number of ordered pairs allowed by the complete/self-loop policy.
///
/// `O(1)`: `N × N` minus the diagonal when self-loops are disabled.
#[inline]
fn complete_policy_count(node_count: usize, self_loops: bool) -> usize {
    let all = node_count.saturating_mul(node_count);
    if self_loops {
        all
    } else {
        all.saturating_sub(node_count)
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
    fn complete_minus_excludes_fixed() {
        let mut excluded = HashSet::new();
        excluded.insert((0, 1));
        excluded.insert((2, 2));
        let d = PairDomain::CompleteMinus {
            node_count: 3,
            self_loops: true,
            excluded,
        };
        assert!(!d.is_admissible(0, 1));
        assert!(!d.is_admissible(2, 2));
        assert!(d.is_admissible(0, 0));
        assert!(d.is_admissible(2, 0));
        assert!(!d.is_admissible(3, 0)); // out of range
        assert!(!d.is_admissible(0, 3));
    }

    #[test]
    fn complete_minus_loopless_policy() {
        let mut excluded = HashSet::new();
        excluded.insert((1, 2));
        let d = PairDomain::CompleteMinus {
            node_count: 3,
            self_loops: false,
            excluded,
        };
        assert!(!d.is_admissible(0, 0)); // self-loop blocked by policy
        assert!(!d.is_admissible(1, 2)); // excluded
        assert!(d.is_admissible(1, 0));
        assert!(d.is_admissible(2, 1));
    }

    #[test]
    fn admissible_pair_count_complete() {
        let with_loops = PairDomain::Complete {
            node_count: 4,
            self_loops: true,
        };
        assert_eq!(with_loops.admissible_pair_count(), 16);
        let loopless = PairDomain::Complete {
            node_count: 4,
            self_loops: false,
        };
        assert_eq!(loopless.admissible_pair_count(), 12);
    }

    #[test]
    fn admissible_pair_count_complete_minus() {
        let mut excluded = HashSet::new();
        excluded.insert((0, 1));
        excluded.insert((3, 3));
        excluded.insert((1, 0));
        let d = PairDomain::CompleteMinus {
            node_count: 4,
            self_loops: true,
            excluded,
        };
        assert_eq!(d.admissible_pair_count(), 13); // 16 - 3
                                                   // Loopless policy removes the diagonal in addition to exclusions.
        let mut excluded = HashSet::new();
        excluded.insert((0, 1));
        let d = PairDomain::CompleteMinus {
            node_count: 3,
            self_loops: false,
            excluded,
        };
        assert_eq!(d.admissible_pair_count(), 5); // 9 - 3 (diagonal) - 1
    }

    #[test]
    fn admissible_pair_count_sparse() {
        let mut allowed = HashSet::new();
        allowed.insert((0, 1));
        allowed.insert((1, 0));
        allowed.insert((2, 2));
        let d = PairDomain::Sparse {
            node_count: 3,
            allowed,
        };
        assert_eq!(d.admissible_pair_count(), 3);
    }

    #[test]
    fn iter_admissible_complete_minus_count() {
        let mut excluded = HashSet::new();
        excluded.insert((0, 1));
        excluded.insert((2, 2));
        let d = PairDomain::CompleteMinus {
            node_count: 4,
            self_loops: true,
            excluded,
        };
        assert_eq!(d.iter_admissible().count(), d.admissible_pair_count());
        let mut seen = d.iter_admissible().collect::<Vec<_>>();
        seen.sort_unstable();
        assert!(!seen.contains(&(0, 1)));
        assert!(!seen.contains(&(2, 2)));
        assert_eq!(seen.len(), 14);
    }

    #[test]
    fn requires_admissibility_repair_policy() {
        assert!(!PairDomain::Complete {
            node_count: 3,
            self_loops: true,
        }
        .requires_admissibility_repair());
        assert!(!PairDomain::CompleteMinus {
            node_count: 3,
            self_loops: true,
            excluded: HashSet::new(),
        }
        .requires_admissibility_repair());
        assert!(PairDomain::CompleteMinus {
            node_count: 3,
            self_loops: true,
            excluded: HashSet::from([(0, 1)]),
        }
        .requires_admissibility_repair());
        assert!(PairDomain::Sparse {
            node_count: 3,
            allowed: HashSet::new(),
        }
        .requires_admissibility_repair());
    }

    #[test]
    fn admissible_degree_caps_complete() {
        let with_loops = PairDomain::Complete {
            node_count: 4,
            self_loops: true,
        };
        let (out, inp) = with_loops.admissible_degree_caps();
        assert_eq!(out, vec![4; 4]);
        assert_eq!(inp, vec![4; 4]);

        let loopless = PairDomain::Complete {
            node_count: 4,
            self_loops: false,
        };
        let (out, inp) = loopless.admissible_degree_caps();
        assert_eq!(out, vec![3; 4]);
        assert_eq!(inp, vec![3; 4]);
    }

    #[test]
    fn admissible_degree_caps_complete_minus() {
        let mut excluded = std::collections::HashSet::new();
        excluded.insert((0, 1));
        excluded.insert((2, 2));
        excluded.insert((1, 0));
        let d = PairDomain::CompleteMinus {
            node_count: 3,
            self_loops: true,
            excluded,
        };
        let (out, inp) = d.admissible_degree_caps();
        // Row 0 loses (0,1); row 1 loses (1,0); row 2 loses (2,2).
        assert_eq!(out, vec![2, 2, 2]);
        // Col 1 loses (0,1); col 0 loses (1,0); col 2 loses (2,2).
        assert_eq!(inp, vec![2, 2, 2]);
        assert_eq!(out.iter().sum::<usize>(), d.admissible_pair_count());
        assert_eq!(inp.iter().sum::<usize>(), d.admissible_pair_count());
    }

    #[test]
    fn admissible_degree_caps_complete_minus_loopless_diagonal() {
        // Loopless policy already excludes the diagonal; an explicit
        // diagonal exclusion must not double-subtract.
        let mut excluded = std::collections::HashSet::new();
        excluded.insert((0, 0));
        excluded.insert((0, 1));
        let d = PairDomain::CompleteMinus {
            node_count: 3,
            self_loops: false,
            excluded,
        };
        let (out, inp) = d.admissible_degree_caps();
        // Row 0: policy gives 2, excluded (0,0) is not otherwise admissible,
        // excluded (0,1) subtracts 1 -> 1.
        assert_eq!(out, vec![1, 2, 2]);
        assert_eq!(inp, vec![2, 1, 2]);
        assert_eq!(out.iter().sum::<usize>(), d.admissible_pair_count());
        assert_eq!(inp.iter().sum::<usize>(), d.admissible_pair_count());
    }

    #[test]
    fn admissible_degree_caps_sparse() {
        let mut allowed = std::collections::HashSet::new();
        allowed.insert((0, 1));
        allowed.insert((1, 0));
        allowed.insert((2, 2));
        allowed.insert((1, 1));
        let d = PairDomain::Sparse {
            node_count: 3,
            allowed,
        };
        let (out, inp) = d.admissible_degree_caps();
        assert_eq!(out, vec![1, 2, 1]);
        assert_eq!(inp, vec![1, 2, 1]);
        assert_eq!(out.iter().sum::<usize>(), d.admissible_pair_count());
        assert_eq!(inp.iter().sum::<usize>(), d.admissible_pair_count());
    }

    #[test]
    fn b_capacity_is_layers() {
        let d = PairDomain::Complete {
            node_count: 5,
            self_loops: true,
        };
        assert_eq!(d.capacity(OccupationFamily::B { layers: 3 }), 3);
        assert_eq!(d.capacity(OccupationFamily::B { layers: 10 }), 10);
    }

    #[test]
    fn me_w_capacity_unbounded() {
        let d = PairDomain::Complete {
            node_count: 5,
            self_loops: true,
        };
        assert_eq!(d.capacity(OccupationFamily::ME), OccNum::MAX);
        assert_eq!(d.capacity(OccupationFamily::W { layers: 1 }), OccNum::MAX);
        assert_eq!(d.capacity(OccupationFamily::W { layers: 3 }), OccNum::MAX);
    }

    #[test]
    fn flow_capacity_min_of_marginals() {
        let d = PairDomain::Complete {
            node_count: 5,
            self_loops: true,
        };
        // ME: capacity = min(src_out_res, tgt_in_res) (since no family cap)
        assert_eq!(d.flow_capacity(OccupationFamily::ME, 5, 3), 3);
        assert_eq!(d.flow_capacity(OccupationFamily::ME, 2, 10), 2);
        // B: capacity = min(M, src_out_res, tgt_in_res)
        assert_eq!(d.flow_capacity(OccupationFamily::B { layers: 4 }, 5, 3), 3);
        assert_eq!(d.flow_capacity(OccupationFamily::B { layers: 4 }, 2, 10), 2);
        assert_eq!(d.flow_capacity(OccupationFamily::B { layers: 2 }, 5, 3), 2);
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
