//! Fixed-strength + fixed-occupied-pair-count (s, E) kernel support.
//!
//! Extends the fixed-strength MCMC with an exact occupied-pair-count
//! constraint `E(target) = E_target`.  Everything here reuses the shared
//! occupied-cell 4-cycle proposal machinery from [`super::move_cycle`];
//! no second proposal law is introduced.
//!
//! # Exact-E local kernel (§7)
//!
//! [`exact_e_local_step`] draws the ordinary fixed-strength proposal and
//! **holds** any proposal whose occupied-pair count would leave the
//! exact-`E` fiber:
//!
//! ```text
//! if occupied_after != edge_target: hold
//! else: ordinary Metropolis–Hastings acceptance (extra weight 0)
//! ```
//!
//! The conditioning on `E` multiplies all allowed states by the same
//! constant, so the local kernel is reversible for the exact conditional
//! target `pi_(s,E)` (proved and verified by the tiny-fiber transition
//! matrix oracles).
//!
//! The local kernel alone is not connected on every fiber (§5) — the
//! bridge path (later phases) restores connectivity without changing the
//! target.

use rand::Rng;

use super::domain::PairDomain;
use super::move_cycle::{draw_cycle4_proposal, log_alpha_with_extra, metropolis_accept};
use super::state::StrengthState;
use super::target::StrengthTarget;

/// Outcome of one exact-E local step, distinguishing the edge-count veto
/// from ordinary holds for mobility diagnostics (§20).
#[allow(dead_code)] // production sweep consumes this in a later phase
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EdgeLocalOutcome {
    /// The in-fiber proposal was applied.
    Accepted,
    /// Proposal invalid at the family/domain/capacity level.
    HeldInvalid,
    /// Valid proposal that would leave the exact-E fiber; vetoed.
    HeldEdgeVeto,
    /// Valid in-fiber proposal rejected by the Metropolis criterion.
    Rejected,
}

/// One exact-E local Metropolis step (allocation-free).
///
/// Draws the ordinary occupied-cell proposal; holds if it would change
/// the occupied-pair count away from `edge_target`; otherwise evaluates
/// the ordinary fixed-strength acceptance (`extra_log_weight = 0`).
/// Fixed-E logic lives here, never inside [`StrengthTarget`] (§12.3).
#[allow(dead_code)] // production sweep consumes this in a later phase
pub(crate) fn exact_e_local_step(
    state: &mut StrengthState,
    target: &StrengthTarget,
    domain: &PairDomain,
    rng: &mut impl Rng,
    edge_target: usize,
) -> EdgeLocalOutcome {
    let Some(proposal) = draw_cycle4_proposal(state, target, domain, rng) else {
        return EdgeLocalOutcome::HeldInvalid;
    };
    if proposal.occupied_after != edge_target {
        return EdgeLocalOutcome::HeldEdgeVeto;
    }
    let Some(log_alpha) = log_alpha_with_extra(&proposal, target, 0.0) else {
        return EdgeLocalOutcome::HeldInvalid;
    };
    if !metropolis_accept(log_alpha, rng) {
        return EdgeLocalOutcome::Rejected;
    }
    proposal.apply(state);
    EdgeLocalOutcome::Accepted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::microcanonical::occupation_mcmc::compressed::compressed_aggregated_matching;
    use crate::model::family::OccupationFamily;
    use crate::OccNum;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_state(
        n: usize,
        so: &[OccNum],
        si: &[OccNum],
        family: OccupationFamily,
        sl: bool,
    ) -> StrengthState {
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: sl,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let table = compressed_aggregated_matching(so, si, family, &domain, &mut rng).unwrap();
        StrengthState::new(n, table)
    }

    #[test]
    fn local_veto_preserves_edge_target_and_strengths() {
        // ME N=2, s_out=s_in=[2,2], E_target=2.  The only valid 4-cycle
        // from either fiber state passes through the all-ones table with
        // E=4 (§5), so every valid proposal is vetoed: E must never move.
        let n = 2;
        let so = vec![2u64; 2];
        let si = vec![2u64; 2];
        let mut state = make_state(n, &so, &si, OccupationFamily::ME, true);
        let target = StrengthTarget::new(OccupationFamily::ME);
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };
        let mut rng = StdRng::seed_from_u64(7);
        let mut vetoes = 0u64;
        let mut accepted = 0u64;
        for _ in 0..1000 {
            match exact_e_local_step(&mut state, &target, &domain, &mut rng, 2) {
                EdgeLocalOutcome::HeldEdgeVeto => vetoes += 1,
                EdgeLocalOutcome::Accepted => accepted += 1,
                _ => {}
            }
            assert_eq!(state.occupied_count(), 2, "E drifted from target");
            assert_eq!(state.out_strengths, so);
            assert_eq!(state.in_strengths, si);
            state.debug_validate();
        }
        assert!(vetoes > 0, "expected edge-count vetoes on the §5 fiber");
        // This fiber is the textbook disconnected case: no in-fiber move.
        assert_eq!(accepted, 0, "no in-fiber proposal exists on the §5 fiber");
    }

    #[test]
    fn local_kernel_moves_within_nontrivial_fiber() {
        // ME N=2, s=[3,3]/[3,3], E_target=4.  Start directly inside the
        // fiber with all four cells occupied: (0,0)=2,(1,1)=2,(0,1)=1,
        // (1,0)=1.  The proposal between (0,0) and (1,1) keeps every cell
        // occupied (E stays 4, delta log weight = 0) so it is always
        // accepted; the proposal between (0,1) and (1,0) drops E to 2 and
        // is vetoed.  The fiber is non-singleton (e.g. the
        // (0,0)=1,(1,1)=1,(0,1)=2,(1,0)=2 state), so the fixed-E kernel
        // must accept in-fiber moves while never leaving E=4.
        let n = 2;
        let so = vec![3u64; 2];
        let si = vec![3u64; 2];
        let mut state =
            StrengthState::new(n, vec![((0, 0), 2), ((1, 1), 2), ((0, 1), 1), ((1, 0), 1)]);
        let target = StrengthTarget::new(OccupationFamily::ME);
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };
        let mut rng = StdRng::seed_from_u64(99);
        let mut accepted = 0u64;
        for _ in 0..3000 {
            if exact_e_local_step(&mut state, &target, &domain, &mut rng, 4)
                == EdgeLocalOutcome::Accepted
            {
                accepted += 1;
            }
            assert_eq!(state.occupied_count(), 4, "E drifted from target");
            assert_eq!(state.out_strengths, so);
            assert_eq!(state.in_strengths, si);
        }
        assert!(accepted > 0, "expected in-fiber movement");
    }

    #[test]
    fn local_step_is_deterministic_by_seed() {
        let n = 3;
        let so = vec![2u64; 3];
        let si = vec![2u64; 3];

        let run = |seed: u64| -> Vec<OccNum> {
            let mut state = make_state(n, &so, &si, OccupationFamily::ME, true);
            let target = StrengthTarget::new(OccupationFamily::ME);
            let domain = PairDomain::Complete {
                node_count: n,
                self_loops: true,
            };
            let mut rng = StdRng::seed_from_u64(seed);
            for _ in 0..500 {
                exact_e_local_step(&mut state, &target, &domain, &mut rng, 3);
            }
            let mut pairs: Vec<_> = state.iter_occupied().collect();
            pairs.sort_unstable();
            pairs
                .into_iter()
                .flat_map(|((s, t), o)| vec![s, t, o])
                .collect()
        };

        assert_eq!(run(42), run(42), "same seed must reproduce the walk");
    }
}
