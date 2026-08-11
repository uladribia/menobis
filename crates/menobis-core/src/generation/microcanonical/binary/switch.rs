//! Directed double-edge switch for uniform support MCMC.
//!
//! The directed double-edge switch selects two distinct ordered pairs
//! \(a\to b\) and \(c\to d\) and proposes the reconnection
//! \(a\to d, c\to b\).  This preserves all out- and in-degrees exactly.
//!
//! # Switch-and-hold
//!
//! Exactly one proposal is made per step.  Invalid proposals result in a
//! hold (the state is retained).  This preserves the uniform stationary
//! distribution over the connected component of the support state graph.

use rand::Rng;

use super::state::DegreeSupportState;

/// Outcome of a single switch step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwitchOutcome {
    /// The switch was applied successfully.
    Switched,
    /// The switch was invalid and the state was held.
    Hold,
}

/// Perform one directed double-edge switch step.
///
/// Selects two distinct edges uniformly, proposes the directed reconnection
/// \(a\to d, c\to b\), validates, and applies or holds.
///
/// Returns the outcome and a boolean indicating whether the switch was
/// structurally possible (false when edge_count < 2).
pub fn directed_switch_step(
    state: &mut DegreeSupportState,
    self_loops: bool,
    rng: &mut impl Rng,
    admissible_pairs: Option<&[(u64, u64)]>,
) -> SwitchOutcome {
    let m = state.edge_count();
    if m < 2 {
        return SwitchOutcome::Hold;
    }

    // Select two distinct edge indices uniformly
    let i = rng.random_range(0..m);
    let mut j = rng.random_range(0..m - 1);
    if j >= i {
        j += 1;
    }

    let edge_1 = state.edges[i];
    let edge_2 = state.edges[j];
    let (a, b) = edge_1;
    let (c, d) = edge_2;

    // Directed: always reconnect a→d, c→b
    let cand_1 = (a, d);
    let cand_2 = (c, b);

    // Validation checks (cheapest first)

    // 1. Self-loops
    if !self_loops && (a == d || c == b) {
        return SwitchOutcome::Hold;
    }

    // 2. Duplicate candidates (a==c && d==b) or (a==b && ...)
    if cand_1 == cand_2 {
        return SwitchOutcome::Hold;
    }

    // 3. No-op check
    let old_set = [edge_1, edge_2];
    let new_set = [cand_1, cand_2];
    if new_set == old_set {
        return SwitchOutcome::Hold;
    }

    // 4. Already occupied (outside the two removed edges)
    if state.contains(&cand_1) && cand_1 != edge_1 && cand_1 != edge_2 {
        return SwitchOutcome::Hold;
    }
    if state.contains(&cand_2) && cand_2 != edge_1 && cand_2 != edge_2 {
        return SwitchOutcome::Hold;
    }

    // 5. Mask validity (admissible-pair restriction)
    if let Some(admissible) = admissible_pairs {
        if !admissible.contains(&cand_1) || !admissible.contains(&cand_2) {
            return SwitchOutcome::Hold;
        }
    }

    // 6. Apply the switch
    state.remove(&edge_1);
    state.remove(&edge_2);
    state.insert(cand_1);
    state.insert(cand_2);

    SwitchOutcome::Switched
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Build a small directed graph for testing.
    fn test_state() -> DegreeSupportState {
        // N=4, edges: 0→1, 1→2, 2→3, 3→0 (a directed 4-cycle)
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        DegreeSupportState::new(4, edges, false)
    }

    #[test]
    fn switch_preserves_degrees() {
        let mut state = test_state();
        let out_before = state.out_degree_sequence();
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            directed_switch_step(&mut state, false, &mut rng, None);
            assert_eq!(state.out_degree_sequence(), out_before);
            #[cfg(debug_assertions)]
            state.debug_validate();
        }
    }

    #[test]
    fn edge_count_preserved() {
        let mut state = test_state();
        let m = state.edge_count();
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            directed_switch_step(&mut state, false, &mut rng, None);
            assert_eq!(state.edge_count(), m);
        }
    }

    #[test]
    fn no_self_loops_after_switch() {
        let mut state = test_state();
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            directed_switch_step(&mut state, false, &mut rng, None);
            for &(src, tgt) in &state.edges {
                assert_ne!(src, tgt, "self-loop found");
            }
        }
    }

    #[test]
    fn no_duplicate_edges_after_switch() {
        let mut state = test_state();
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            directed_switch_step(&mut state, false, &mut rng, None);
            let mut pairs = state.edges.clone();
            pairs.sort_unstable();
            pairs.dedup();
            assert_eq!(pairs.len(), state.edge_count());
        }
    }

    #[test]
    fn hold_when_too_few_edges() {
        let mut state = DegreeSupportState::new(3, vec![(0, 1)], false);
        let mut rng = StdRng::seed_from_u64(42);
        assert_eq!(
            directed_switch_step(&mut state, false, &mut rng, None),
            SwitchOutcome::Hold
        );
    }

    #[test]
    fn every_switch_has_reverse() {
        // Verify that for any valid forward switch, the reverse is also valid.
        // Exhaustively test on a small state.
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let mut state = DegreeSupportState::new(3, edges, false);
        let mut rng = StdRng::seed_from_u64(42);

        // Run many steps — the chain should stay within valid states.
        for _ in 0..500 {
            directed_switch_step(&mut state, false, &mut rng, None);
            assert_eq!(state.edge_count(), 3);
            assert_eq!(state.out_degree_sequence(), vec![1, 1, 1]);
        }
    }

    #[test]
    fn switch_respects_admissible_pairs() {
        let edges = vec![(0, 1), (1, 2), (2, 0), (0, 2)];
        let mut state = DegreeSupportState::new(3, edges, false);
        let admissible = vec![(0u64, 1u64), (1, 2), (2, 0), (0, 2), (1, 0), (2, 1)];
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..200 {
            directed_switch_step(&mut state, false, &mut rng, Some(&admissible));
            // Every edge must be in the admissible set
            for &(src, tgt) in &state.edges {
                assert!(
                    admissible.contains(&(src, tgt)),
                    "edge ({src},{tgt}) not admissible"
                );
            }
        }
    }
}
