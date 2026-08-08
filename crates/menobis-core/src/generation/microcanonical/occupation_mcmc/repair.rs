//! Targeted loop repair for complete loopless fixed-strength ME/W networks.
//!
//! Provides guaranteed deterministic loop elimination via rectangle
//! repair (spec 14–17).  This module implements the **Phase D** targeted
//! repair approach: no generic framework, no max flow, no annealing.
//!
//! # Algorithm (spec 14–17)
//!
//! 1. [Feasibility check (spec 14)]: O(N) verification that
//!    `s_i^out + s_i^in ≤ T` for all nodes.
//! 2. [Rectangle repair (spec 15)]: For each self-loop, select an
//!    occupied donor cell and apply a 4-cycle that shifts mass from
//!    diagonal (i,i) and donor (c,d) to cross cells (i,d) and (c,i).
//! 3. [Termination (spec 16)]: Each step strictly decreases total
//!    loop mass L(t). Donor existence is guaranteed by feasibility.
//! 4. [Donor selection (spec 17)]: Randomized occupied-pair sampling
//!    with bounded linear-scan fallback — no O(N²) enumeration.

use rand::Rng;

use super::domain::PairDomain;
use super::errors::FixedStrengthError;
use super::state::StrengthState;
use crate::OccNum;

/// Maximum steps for the repair loop (spec 21).
const MAX_REPAIR_STEPS: u64 = 10_000_000;

/// Maximum random donor attempts before falling back to linear scan (spec 17).
const MAX_RANDOM_DONOR_ATTEMPTS: usize = 20;

/// O(N) feasibility check for complete loopless ME/W (spec 14).
///
/// A loopless realization exists iff:
///
/// ```text
/// s_i^out + s_i^in ≤ T   ∀i,
/// ```
///
/// where T = Σ s_i^out = Σ s_i^in.
///
/// Returns `true` if feasible.
pub fn loopless_feasibility_check(strength_out: &[OccNum], strength_in: &[OccNum]) -> bool {
    let total: OccNum = strength_out.iter().sum();
    debug_assert_eq!(
        total,
        strength_in.iter().sum::<OccNum>(),
        "out and in totals must be equal"
    );
    strength_out
        .iter()
        .zip(strength_in.iter())
        .all(|(&s_out, &s_in)| s_out + s_in <= total)
}

/// Find all self-loops in the state.
///
/// Returns the list of `(node, occupation)` pairs where `src == tgt`.
fn find_self_loops(state: &StrengthState) -> Vec<(u64, OccNum)> {
    state
        .occupied_pairs()
        .iter()
        .filter_map(|&(src, tgt)| {
            if src == tgt {
                Some((src, state.get(src, tgt)))
            } else {
                None
            }
        })
        .collect()
}

/// Find a donor cell for self-loop repair (spec 17).
///
/// Attempts `MAX_RANDOM_DONOR_ATTEMPTS` random draws from occupied
/// pairs, rejecting candidates where `src == i || tgt == i`.  If random
/// sampling fails, falls back to a bounded linear scan over all occupied
/// pairs (guaranteed to succeed when feasibility holds, spec 16).
///
/// Returns `None` only if no valid donor exists (should not happen
/// when the feasibility condition holds and `t_ii > 0`).
fn find_donor(state: &StrengthState, i: u64, rng: &mut impl Rng) -> Option<(u64, u64)> {
    // Phase 1: random sampling from occupied pairs.
    for _ in 0..MAX_RANDOM_DONOR_ATTEMPTS {
        if let Some(((src, tgt), _)) = state.choose_random_occupied(rng) {
            if src != i && tgt != i {
                return Some((src, tgt));
            }
        }
    }

    // Phase 2: bounded linear scan fallback.
    for &(src, tgt) in state.occupied_pairs() {
        if src != i && tgt != i {
            return Some((src, tgt));
        }
    }

    None
}

/// Guaranteed loop repair for complete loopless ME/W (spec 15–17).
///
/// Eliminates all self-loops from the state while preserving every
/// source and target strength exactly.  The repair is guaranteed to
/// progress because each step strictly decreases total loop mass
/// L(t) (spec 16), and donor existence is guaranteed by the loopless
/// feasibility condition (spec 14).
///
/// # Algorithm
///
/// For each self-loop `(i, i)` with occupation `t_ii`:
/// 1. Find a donor cell `(c, d)` with `c ≠ i` and `d ≠ i` (spec 17).
/// 2. Compute `δ = min(t_ii, t_cd)`.
/// 3. Apply rectangle (spec 15):
///    ```text
///    t_ii' = t_ii - δ
///    t_cd' = t_cd - δ
///    t_id' = t_id + δ
///    t_ci' = t_ci + δ
///    ```
///
/// Each step preserves strengths exactly (each row/column gets one
/// `-δ` and one `+δ`).  Increment cells `(i,d)` and `(c,i)` are
/// non-diagonal because `d ≠ i` and `c ≠ i`, so no new self-loops
/// are introduced.
///
/// # Errors
///
/// Returns [`FixedStrengthError::RepairDidNotConverge`] if the repair
/// exceeds `MAX_REPAIR_STEPS` steps or self-loops remain afterward.
pub fn repair_self_loops(
    state: &mut StrengthState,
    domain: &PairDomain,
    rng: &mut impl Rng,
) -> Result<(), FixedStrengthError> {
    debug_assert!(
        !domain.self_loops_allowed(),
        "repair_self_loops should only be called when self-loops are forbidden"
    );

    // Discover initial self-loops.
    let mut self_loops: Vec<(u64, OccNum)> = find_self_loops(state);
    let mut steps: u64 = 0;

    while !self_loops.is_empty() && steps < MAX_REPAIR_STEPS {
        // Pop one self-loop from the list (O(1) removal from back).
        let (i, _) = self_loops.swap_remove(self_loops.len() - 1);

        // Refresh t_ii from state — may have changed via cross-cell
        // increments from previous steps involving this node.
        let t_ii = state.get(i, i);

        if t_ii == 0 {
            continue;
        }

        // Find a donor cell (c, d) with c != i and d != i.
        let (c, d) =
            find_donor(state, i, rng).ok_or_else(|| FixedStrengthError::RepairDidNotConverge {
                remaining_loops: self_loops.len() + 1,
                remaining_capacity_violations: 0,
                remaining_forbidden_occupations: 0,
                restart_count: 0,
                steps,
            })?;

        let t_cd = state.get(c, d);
        debug_assert!(
            t_cd > 0,
            "donor cell ({c}, {d}) must be occupied (find_donor guarantees this)"
        );

        // Compute delta = min(t_ii, t_cd).
        let delta = t_ii.min(t_cd);
        debug_assert!(delta > 0, "delta must be positive");

        // Apply rectangle delta set via state.set().
        // Cells: (i,i,-δ), (c,d,-δ), (i,d,+δ), (c,i,+δ)
        let t_ii_new = t_ii - delta;
        let t_cd_new = t_cd - delta;
        let t_id_new = state.get(i, d) + delta;
        let t_ci_new = state.get(c, i) + delta;

        state.set(i, i, t_ii_new);
        state.set(c, d, t_cd_new);
        state.set(i, d, t_id_new);
        state.set(c, i, t_ci_new);

        steps += 1;

        // Re-add self-loop to list if it still has mass after this step.
        if t_ii_new > 0 {
            self_loops.push((i, t_ii_new));
        }
    }

    if !self_loops.is_empty() {
        return Err(FixedStrengthError::RepairDidNotConverge {
            remaining_loops: self_loops.len(),
            remaining_capacity_violations: 0,
            remaining_forbidden_occupations: 0,
            restart_count: 0,
            steps,
        });
    }

    // Verify no self-loops remain (debug-only assertion).
    debug_assert_eq!(
        find_self_loops(state).len(),
        0,
        "repair claimed success but self-loops remain"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn loopless_feasibility_check_passes() {
        // T = 10, each node: s_out + s_in <= 10
        let out = vec![5, 3, 2];
        let inp = vec![4, 4, 2];
        assert!(
            loopless_feasibility_check(&out, &inp),
            "feasible sequences should pass"
        );
    }

    #[test]
    fn loopless_feasibility_check_fails() {
        // T = 10, node 0: 10 + 1 = 11 > 10
        let out = vec![10, 0];
        let inp = vec![1, 9];
        assert!(
            !loopless_feasibility_check(&out, &inp),
            "infeasible sequences should fail"
        );
    }

    #[test]
    fn loopless_feasibility_check_boundary() {
        // T = 10, node 0: 5 + 5 = 10 == T (boundary case, still feasible)
        let out = vec![5, 5];
        let inp = vec![5, 5];
        assert!(
            loopless_feasibility_check(&out, &inp),
            "boundary case s_i^out + s_i^in == T should pass"
        );
    }

    #[test]
    fn repair_eliminates_all_self_loops() {
        let mut rng = StdRng::seed_from_u64(42);
        let domain = PairDomain::Complete {
            node_count: 3,
            self_loops: false,
        };

        // State with one self-loop at (0,0)=3.
        // Strengths: out=[5,3,2], in=[4,4,2], T=10.
        // All nodes satisfy s_out + s_in <= 10.
        let pairs: Vec<((u64, u64), OccNum)> = vec![
            ((0, 0), 3), // self-loop
            ((0, 1), 2),
            ((1, 0), 1),
            ((1, 1), 1),
            ((1, 2), 1),
            ((2, 1), 1),
            ((2, 2), 1),
        ];
        let mut state = StrengthState::new(3, pairs);

        // The state has self-loops on nodes 0, 1, and 2.
        assert!(
            !find_self_loops(&state).is_empty(),
            "state must have self-loops before repair"
        );
        let out_before = state.out_strengths.clone();
        let in_before = state.in_strengths.clone();

        repair_self_loops(&mut state, &domain, &mut rng).unwrap();

        assert_eq!(
            find_self_loops(&state).len(),
            0,
            "all self-loops must be eliminated"
        );
        assert_eq!(
            state.out_strengths, out_before,
            "out-strengths preserved after repair"
        );
        assert_eq!(
            state.in_strengths, in_before,
            "in-strengths preserved after repair"
        );
    }

    #[test]
    fn repair_preserves_strengths_multiple_self_loops() {
        let mut rng = StdRng::seed_from_u64(99);
        let domain = PairDomain::Complete {
            node_count: 3,
            self_loops: false,
        };

        // State with two self-loops: (0,0)=2, (1,1)=2.
        // Strengths: out=[4,4,2], in=[4,4,2], T=10.
        // All nodes satisfy s_out + s_in <= 10.
        let pairs: Vec<((u64, u64), OccNum)> = vec![
            ((0, 0), 2), // self-loop
            ((0, 1), 2),
            ((1, 0), 1),
            ((1, 1), 2), // self-loop
            ((1, 2), 1),
            ((2, 1), 1),
            ((2, 2), 1),
        ];
        let mut state = StrengthState::new(3, pairs);

        // Verify pre-repair self-loops.
        let loops_before = find_self_loops(&state);
        assert!(loops_before.len() >= 2);

        let out_before = state.out_strengths.clone();
        let in_before = state.in_strengths.clone();

        repair_self_loops(&mut state, &domain, &mut rng).unwrap();

        assert_eq!(
            find_self_loops(&state).len(),
            0,
            "all self-loops must be eliminated"
        );
        assert_eq!(state.out_strengths, out_before);
        assert_eq!(state.in_strengths, in_before);
    }

    #[test]
    fn repair_no_op_when_no_self_loops() {
        let mut rng = StdRng::seed_from_u64(42);
        let domain = PairDomain::Complete {
            node_count: 3,
            self_loops: false,
        };

        // State with no self-loops.
        let pairs: Vec<((u64, u64), OccNum)> = vec![
            ((0, 1), 3),
            ((0, 2), 2),
            ((1, 0), 3),
            ((1, 2), 1),
            ((2, 0), 2),
            ((2, 1), 2),
        ];
        let mut state = StrengthState::new(3, pairs);
        let out_before = state.out_strengths.clone();
        let in_before = state.in_strengths.clone();

        assert_eq!(find_self_loops(&state).len(), 0);

        repair_self_loops(&mut state, &domain, &mut rng).unwrap();

        assert_eq!(find_self_loops(&state).len(), 0);
        assert_eq!(state.out_strengths, out_before);
        assert_eq!(state.in_strengths, in_before);
    }

    #[test]
    fn find_donor_none_for_one_node() {
        // N=1: only cell is (0,0) self-loop, no possible donor.
        // This case is unreachable in production (feasibility_check rejects it),
        // but find_donor should still return None gracefully.
        let mut rng = StdRng::seed_from_u64(42);
        let pairs: Vec<((u64, u64), OccNum)> = vec![((0, 0), 5)];
        let state = StrengthState::new(1, pairs);
        assert!(
            find_donor(&state, 0, &mut rng).is_none(),
            "N=1 cannot have a donor"
        );
    }

    #[test]
    fn repair_self_loops_deterministic_with_seed() {
        // Same seed → same result (modulo RNG effects on donor selection).
        let domain = PairDomain::Complete {
            node_count: 3,
            self_loops: false,
        };
        let pairs: Vec<((u64, u64), OccNum)> = vec![
            ((0, 0), 3),
            ((0, 1), 2),
            ((1, 0), 1),
            ((1, 1), 1),
            ((1, 2), 1),
            ((2, 1), 1),
            ((2, 2), 1),
        ];

        let mut state_a = StrengthState::new(3, pairs.clone());
        let mut state_b = StrengthState::new(3, pairs);
        let mut rng_a = StdRng::seed_from_u64(12345);
        let mut rng_b = StdRng::seed_from_u64(12345);

        repair_self_loops(&mut state_a, &domain, &mut rng_a).unwrap();
        repair_self_loops(&mut state_b, &domain, &mut rng_b).unwrap();

        // Check that the resulting states are identical (same pairs and occupations).
        let mut pairs_a: Vec<_> = state_a.iter_occupied().collect();
        let mut pairs_b: Vec<_> = state_b.iter_occupied().collect();
        pairs_a.sort_unstable();
        pairs_b.sort_unstable();
        assert_eq!(pairs_a, pairs_b, "deterministic repair with same seed");
    }
}
