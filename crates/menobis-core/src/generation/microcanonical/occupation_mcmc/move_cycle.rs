//! Occupied-cell 4-cycle MCMC kernel for fixed-strength sampling.
//!
//! The proposal selects two distinct **occupied** pairs from the current
//! state, decrements both, and increments the two cross cells (spec §22).
//! A Metropolis–Hastings correction (§23) accounts for the state-dependent
//! proposal probability.
//!
//! # Structure (§12)
//!
//! The kernel is factored into three reusable pieces sharing one proposal
//! record (§12.1–12.3):
//!
//! 1. [`draw_cycle4_proposal`] — draws a proposal under exactly the
//!    original selection law (no allocation, fixed-size proposal record).
//! 2. [`log_alpha_with_extra`] — one shared Metropolis log-ratio evaluator
//!    (`Δlog π + log q_reverse − log q_forward + extra_log_weight`).
//! 3. [`metropolis_accept`] — the acceptance decision.
//!
//! [`occupied_cycle4_step`] composes the three with `extra_log_weight = 0`
//! and is behaviorally identical to the pre-refactor kernel (same proposal
//! law, same acceptance, same RNG consumption).  The exact-E local kernel,
//! the auxiliary edge-biased bridge, and the edge initialization repair
//! reuse the same machinery in later modules.

use rand::Rng;

use super::domain::PairDomain;
use super::rectangle::{build_four_cell, validate_four_cell};
use super::state::StrengthState;
use super::target::StrengthTarget;
use crate::generation::microcanonical::mcmc::McmcOutcome;
use crate::OccNum;

/// A single occupied-cell 4-cycle proposal, fully precomputed on the stack.
///
/// Carries the four cell occupations before/after, the occupied-pair
/// counts before/after, and the exact forward/reverse proposal
/// log-probabilities under the `master` selection law (§12.1).  No heap
/// allocation — the record is `Copy`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cycle4Proposal {
    /// First decrement cell `(a, b)`.
    pub a: u64,
    /// First decrement cell `(a, b)`.
    pub b: u64,
    /// Second decrement cell `(c, d)`.
    pub c: u64,
    /// Second decrement cell `(c, d)`.
    pub d: u64,

    /// Old occupation of `(a, b)`.
    pub old_ab: OccNum,
    /// Old occupation of `(c, d)`.
    pub old_cd: OccNum,
    /// Old occupation of `(a, d)`.
    pub old_ad: OccNum,
    /// Old occupation of `(c, b)`.
    pub old_cb: OccNum,

    /// New occupation of `(a, b)`.
    pub new_ab: OccNum,
    /// New occupation of `(c, d)`.
    pub new_cd: OccNum,
    /// New occupation of `(a, d)`.
    pub new_ad: OccNum,
    /// New occupation of `(c, b)`.
    pub new_cb: OccNum,

    /// Occupied-pair count of the current state.
    #[allow(dead_code)] // consumed by the exact-E veto and repair phases
    pub occupied_before: usize,
    /// Occupied-pair count of the proposed state.
    #[allow(dead_code)] // consumed by the exact-E veto and repair phases
    pub occupied_after: usize,

    /// `log q(x → y)` under the exact `master` selection law.
    pub log_q_forward: f64,
    /// `log q(y → x)` under the exact `master` selection law.
    pub log_q_reverse: f64,
}

impl Cycle4Proposal {
    /// Change in the occupied-pair count: `occupied_after − occupied_before`.
    #[inline]
    #[allow(dead_code)] // consumed by the exact-E veto and repair phases
    pub fn delta_edges(&self) -> i64 {
        self.occupied_after as i64 - self.occupied_before as i64
    }

    /// Apply the four cell updates to `state`.
    pub fn apply(&self, state: &mut StrengthState) {
        state.set(self.a, self.b, self.new_ab);
        state.set(self.c, self.d, self.new_cd);
        state.set(self.a, self.d, self.new_ad);
        state.set(self.c, self.b, self.new_cb);
    }

    /// Undo the proposal: restore the four old occupations exactly via
    /// [`StrengthState::set`].
    #[allow(dead_code)] // consumed by the bridge undo path
    pub fn undo(&self, state: &mut StrengthState) {
        state.set(self.a, self.b, self.old_ab);
        state.set(self.c, self.d, self.old_cd);
        state.set(self.a, self.d, self.old_ad);
        state.set(self.c, self.b, self.old_cb);
    }
}

/// Draw a 4-cycle proposal under the exact `master` selection law (§12.2).
///
/// 1. chooses the first occupied cell uniformly;
/// 2. computes the valid-partner count `v_ab`;
/// 3. chooses the second occupied cell with the existing unbounded
///    rejection logic;
/// 4. uses the fixed direction (decrement `(a,b)`, `(c,d)`; increment
///    `(a,d)`, `(c,b)`);
/// 5. validates the rectangle through the shared [`validate_four_cell`];
/// 6. computes proposed occupations, `occupied_after`, and the exact
///    forward/reverse proposal log-probabilities.
///
/// Returns `None` whenever the original step would have returned
/// `Held` before the acceptance phase; callers map `None` to `Held`.
///
/// No random orientation, no bounded partner rejection, no candidate
/// vectors, no allocation.  Here `draw` never consumes RNG beyond the
/// original law, so composing it with the evaluator leaves the RNG
/// stream identical to the pre-refactor kernel.
pub(crate) fn draw_cycle4_proposal(
    state: &StrengthState,
    target: &StrengthTarget,
    domain: &PairDomain,
    rng: &mut impl Rng,
) -> Option<Cycle4Proposal> {
    let m = state.occupied_count();
    if m < 2 {
        return None;
    }

    // ---- 1. Select P1 = (a,b) uniformly from occupied pairs ----
    let a_idx = rng.random_range(0..m);
    let (a, b) = state.occupied_pairs()[a_idx];

    // ---- 2. Compute v_ab (valid P2 candidates for P1=(a,b)) ----
    // Inclusion–exclusion: m − row_occ[a] − col_occ[b] + 1.
    let v_ab =
        m as i64 - state.row_occ_count[a as usize] as i64 - state.col_occ_count[b as usize] as i64
            + 1;
    if v_ab <= 0 {
        return None;
    }
    debug_assert!(v_ab >= 1);

    // ---- 3. Select P2 = (c,d) by unbounded rejection ----
    let (c, d) = loop {
        let idx = rng.random_range(0..m);
        if idx == a_idx {
            continue;
        }
        let (cs, cd) = state.occupied_pairs()[idx];
        if cs != a && cd != b {
            break (cs, cd);
        }
    };

    // ---- 4. Build deltas (fixed direction per §22) ----
    let deltas = build_four_cell(a, c, b, d);

    // ---- 5. Read current occupations of the four cells ----
    let old_ab = state.get(a, b);
    let old_cd = state.get(c, d);
    let old_ad = state.get(a, d);
    let old_cb = state.get(c, b);

    // ---- 6. Validate all four cells ----
    if !validate_four_cell(state, target, domain, &deltas) {
        return None;
    }

    // ---- 7. Proposed occupations ----
    // Bounds are guaranteed by validate_four_cell, so checked arithmetic
    // is safe.
    let new_ab = old_ab.checked_sub(1).unwrap();
    let new_cd = old_cd.checked_sub(1).unwrap();
    let new_ad = old_ad.checked_add(1).unwrap();
    let new_cb = old_cb.checked_add(1).unwrap();

    // ---- 8. Hastings proposal log-ratio (§23) ----
    let v_cd =
        m as i64 - state.row_occ_count[c as usize] as i64 - state.col_occ_count[d as usize] as i64
            + 1;
    if v_cd <= 0 {
        return None;
    }

    // Compute m', row_occ'_a/c, col_occ'_b/d O(1) from transitions.
    let leaves_ab = if old_ab == 1 { 1 } else { 0 };
    let leaves_cd = if old_cd == 1 { 1 } else { 0 };
    let enters_ad = if old_ad == 0 { 1 } else { 0 };
    let enters_cb = if old_cb == 0 { 1 } else { 0 };
    let leaves = leaves_ab + leaves_cd;
    let enters = enters_ad + enters_cb;

    let m_prime = m as i64 + enters - leaves;
    if m_prime <= 0 {
        return None;
    }

    let row_a_prime = state.row_occ_count[a as usize] as i64 + enters_ad - leaves_ab;
    let row_c_prime = state.row_occ_count[c as usize] as i64 + enters_cb - leaves_cd;
    let col_b_prime = state.col_occ_count[b as usize] as i64 + enters_cb - leaves_ab;
    let col_d_prime = state.col_occ_count[d as usize] as i64 + enters_ad - leaves_cd;

    let v_ad_prime = m_prime - row_a_prime - col_d_prime + 1;
    let v_cb_prime = m_prime - row_c_prime - col_b_prime + 1;

    if v_ad_prime <= 0 || v_cb_prime <= 0 {
        // The reverse move would have no valid second choice.
        return None;
    }

    let log_q_forward = (1.0 / v_ab as f64 + 1.0 / v_cd as f64).ln() - (m as f64).ln();
    let log_q_reverse =
        (1.0 / v_ad_prime as f64 + 1.0 / v_cb_prime as f64).ln() - (m_prime as f64).ln();

    Some(Cycle4Proposal {
        a,
        b,
        c,
        d,
        old_ab,
        old_cd,
        old_ad,
        old_cb,
        new_ab,
        new_cd,
        new_ad,
        new_cb,
        occupied_before: m,
        occupied_after: m_prime as usize,
        log_q_forward,
        log_q_reverse,
    })
}

/// Shared Metropolis log-ratio evaluator (§12.3).
///
/// `log_alpha = Δlog π_family + log q_reverse − log q_forward + extra_log_weight`
///
/// - Ordinary fixed strength: `extra_log_weight = 0`.
/// - Auxiliary edge-biased bridge: `extra_log_weight = delta_edge_potential`
///   (Phase 6).
/// - Exact-E local moves must veto proposals that change the occupied-pair
///   count *before* calling this evaluator; fixed-E logic never enters
///   [`StrengthTarget`].
///
/// Returns `None` only if the family target rejects a cell change (`Held`
/// for the caller); `validate_four_cell` inside the draw prevents this for
/// every proposal this function receives.
pub(crate) fn log_alpha_with_extra(
    proposal: &Cycle4Proposal,
    target: &StrengthTarget,
    extra_log_weight: f64,
) -> Option<f64> {
    let delta_log_pi =
        target.delta_log_weight(proposal.a, proposal.b, proposal.old_ab, proposal.new_ab)?
            + target.delta_log_weight(proposal.c, proposal.d, proposal.old_cd, proposal.new_cd)?
            + target.delta_log_weight(proposal.a, proposal.d, proposal.old_ad, proposal.new_ad)?
            + target.delta_log_weight(proposal.c, proposal.b, proposal.old_cb, proposal.new_cb)?;
    Some(delta_log_pi + proposal.log_q_reverse - proposal.log_q_forward + extra_log_weight)
}

/// Metropolis decision for a precomputed log-ratio.
///
/// Accepts with probability `min(1, exp(log_alpha))`.  The uniform draw
/// is consumed only when `log_alpha < 0`, exactly as in the pre-refactor
/// kernel.
#[inline]
pub(crate) fn metropolis_accept(log_alpha: f64, rng: &mut impl Rng) -> bool {
    if log_alpha < 0.0 {
        let log_u = (rng.random::<f64>() + f64::MIN_POSITIVE).ln();
        if log_u >= log_alpha {
            return false;
        }
    }
    true
}

/// Perform one occupied-cell 4-cycle MCMC step (allocation-free).
///
/// Composes [`draw_cycle4_proposal`] → [`log_alpha_with_extra`] →
/// [`metropolis_accept`] with `extra_log_weight = 0`.  Mathematically
/// identical to the pre-refactor kernel (same proposal law, same
/// acceptance, same RNG consumption) — verified by the exact Hastings
/// oracle in `menobis-test-oracles`.
///
/// Returns [`McmcOutcome::Held`] if fewer than 2 occupied pairs exist,
/// no valid second pair exists for the drawn first pair, the rectangle
/// is invalid, or the reverse move would be infeasible.
pub fn occupied_cycle4_step(
    state: &mut StrengthState,
    target: &StrengthTarget,
    domain: &PairDomain,
    rng: &mut impl Rng,
) -> McmcOutcome {
    let Some(proposal) = draw_cycle4_proposal(state, target, domain, rng) else {
        return McmcOutcome::Held;
    };
    let Some(log_alpha) = log_alpha_with_extra(&proposal, target, 0.0) else {
        return McmcOutcome::Held;
    };
    if !metropolis_accept(log_alpha, rng) {
        return McmcOutcome::Rejected;
    }
    proposal.apply(state);
    McmcOutcome::Accepted
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
    fn occupied_cell_preserves_strengths_me() {
        let n = 4;
        let so = vec![5u64, 3, 7, 2];
        let si = vec![4u64, 6, 3, 4];
        let mut state = make_state(n, &so, &si, OccupationFamily::ME, true);
        let target = StrengthTarget::new(OccupationFamily::ME);
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            occupied_cycle4_step(&mut state, &target, &domain, &mut rng);
            assert_eq!(state.out_strengths, so, "out-strengths changed");
            assert_eq!(state.in_strengths, si, "in-strengths changed");
        }
    }

    #[test]
    fn occupied_cell_acceptance_rate_nonzero() {
        let n = 5;
        let so = vec![10u64; 5];
        let si = vec![10u64; 5];
        let mut state = make_state(n, &so, &si, OccupationFamily::ME, true);
        let target = StrengthTarget::new(OccupationFamily::ME);
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };
        let mut rng = StdRng::seed_from_u64(99);
        let mut accepted = 0u64;
        let trials = 500;
        for _ in 0..trials {
            if occupied_cycle4_step(&mut state, &target, &domain, &mut rng) == McmcOutcome::Accepted
            {
                accepted += 1;
            }
        }
        assert!(
            accepted > trials / 20,
            "acceptance rate too low: {accepted}/{trials}"
        );
    }

    #[test]
    fn occupied_cell_no_self_loops() {
        let n = 4;
        let so = vec![5u64; 4];
        let si = vec![5u64; 4];
        let mut state = make_state(n, &so, &si, OccupationFamily::ME, false);
        let target = StrengthTarget::new(OccupationFamily::ME);
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: false,
        };
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..200 {
            occupied_cycle4_step(&mut state, &target, &domain, &mut rng);
            for &(src, tgt) in state.occupied_pairs() {
                assert_ne!(src, tgt, "self-loop appeared");
            }
        }
    }

    #[test]
    fn occupied_cell_b_capacity() {
        let n = 3;
        let layers = 3u32;
        // Build a state that respects B capacity directly.
        // Each cell gets at most M, and strength sums are exact.
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };
        // strength_out = [3,3,3], strength_in = [3,3,3], Σ=9
        // Pack: (0,0,3), (1,1,3), (2,2,3) — all ≤ 3.
        let pairs = vec![((0, 0), 3), ((1, 1), 3), ((2, 2), 3)];
        let mut state = StrengthState::new(n, pairs);
        let target = StrengthTarget::new(OccupationFamily::B { layers });
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..200 {
            occupied_cycle4_step(&mut state, &target, &domain, &mut rng);
            for (_, occ) in state.iter_occupied() {
                assert!(
                    occ <= layers as OccNum,
                    "B occupation {occ} exceeds M={layers}"
                );
            }
        }
    }

    #[test]
    fn occupied_cell_occupied_count_changes() {
        let n = 4;
        let so = vec![3u64; 4];
        let si = vec![3u64; 4];
        let mut state = make_state(n, &so, &si, OccupationFamily::ME, true);
        let target = StrengthTarget::new(OccupationFamily::ME);
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let initial = state.occupied_count();
        for _ in 0..500 {
            occupied_cycle4_step(&mut state, &target, &domain, &mut rng);
        }
        assert!(
            state.occupied_count() != initial,
            "occupied count unchanged after 500 moves"
        );
    }

    #[test]
    fn occupied_cell_reproducible() {
        let n = 4;
        let so = vec![3u64; 4];
        let si = vec![3u64; 4];

        let run = |seed: u64| -> Vec<OccNum> {
            let mut state = make_state(n, &so, &si, OccupationFamily::ME, true);
            let target = StrengthTarget::new(OccupationFamily::ME);
            let domain = PairDomain::Complete {
                node_count: n,
                self_loops: true,
            };
            let mut rng = StdRng::seed_from_u64(seed);
            for _ in 0..100 {
                occupied_cycle4_step(&mut state, &target, &domain, &mut rng);
            }
            let mut pairs = state.iter_occupied().collect::<Vec<_>>();
            pairs.sort_unstable();
            pairs
                .into_iter()
                .flat_map(|((s, t), o)| vec![s, t, o])
                .collect()
        };

        assert_eq!(run(42), run(42));
    }

    #[test]
    fn occupied_cell_held_when_only_one_occupied() {
        // Construct a state with only one occupied pair.
        let mut state = StrengthState::new(2, vec![((0, 1), 5)]);
        let target = StrengthTarget::new(OccupationFamily::ME);
        let domain = PairDomain::Complete {
            node_count: 2,
            self_loops: true,
        };
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..10 {
            assert_eq!(
                occupied_cycle4_step(&mut state, &target, &domain, &mut rng),
                McmcOutcome::Held
            );
        }
    }

    #[test]
    fn occupied_cell_enforces_a_ne_c_and_b_ne_d() {
        // Build a dense state, run many proposals, and verify no proposal
        // violates a==c or b==d.
        let n = 3;
        let so = vec![5u64; 3];
        let si = vec![5u64; 3];
        let mut state = make_state(n, &so, &si, OccupationFamily::ME, true);
        let target = StrengthTarget::new(OccupationFamily::ME);
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };
        let mut rng = StdRng::seed_from_u64(12345);
        for _ in 0..200 {
            // We can't easily inspect the internal proposal selection.
            // Instead, verify that the state remains valid after each step.
            let outcome = occupied_cycle4_step(&mut state, &target, &domain, &mut rng);
            assert!(
                outcome == McmcOutcome::Accepted
                    || outcome == McmcOutcome::Held
                    || outcome == McmcOutcome::Rejected
            );
            state.debug_validate();
        }
    }

    /// Full sparse snapshot of a state for exact round-trip comparison.
    type StateSnapshot = (Vec<((u64, u64), OccNum)>, Vec<OccNum>, Vec<OccNum>);
    fn state_snapshot(state: &StrengthState) -> StateSnapshot {
        let mut pairs: Vec<((u64, u64), OccNum)> = state.iter_occupied().collect();
        pairs.sort_unstable();
        (
            pairs,
            state.out_strengths.clone(),
            state.in_strengths.clone(),
        )
    }

    #[test]
    fn proposal_apply_undo_roundtrip() {
        // Cycle4Proposal::apply then undo must restore the exact state
        // (occupied pairs, marginals) — the bridge undo contract (§19).
        let n = 4;
        let so = vec![5u64, 3, 7, 2];
        let si = vec![4u64, 6, 3, 4];
        let mut state = make_state(n, &so, &si, OccupationFamily::ME, true);
        let target = StrengthTarget::new(OccupationFamily::ME);
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };
        let mut rng = StdRng::seed_from_u64(1234);
        let mut applied = 0usize;
        for _ in 0..500 {
            let before = state_snapshot(&state);
            let Some(proposal) = draw_cycle4_proposal(&state, &target, &domain, &mut rng) else {
                continue;
            };
            proposal.apply(&mut state);
            applied += 1;
            // occupied_after delta_edges must agree with the applied state.
            assert_eq!(state.occupied_count(), proposal.occupied_after);
            assert_eq!(
                state.occupied_count() as i64 - proposal.occupied_before as i64,
                proposal.delta_edges()
            );
            state.debug_validate();
            proposal.undo(&mut state);
            assert_eq!(
                state_snapshot(&state),
                before,
                "undo must restore the exact pre-proposal state"
            );
            state.debug_validate();
        }
        assert!(applied > 0, "no proposals were applied");
    }

    #[test]
    fn composed_path_matches_step_with_same_seed() {
        // The refactored draw → evaluate → accept → apply sequence must be
        // indistinguishable (same RNG consumption, same outcomes) from the
        // public occupied_cycle4_step wrapper.
        let n = 4;
        let so = vec![5u64, 3, 7, 2];
        let si = vec![4u64, 6, 3, 4];
        let target = StrengthTarget::new(OccupationFamily::ME);
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };

        let run_composed = |seed: u64| -> Vec<McmcOutcome> {
            let mut state = make_state(n, &so, &si, OccupationFamily::ME, true);
            let mut rng = StdRng::seed_from_u64(seed);
            (0..100usize)
                .map(|_| {
                    let Some(proposal) = draw_cycle4_proposal(&state, &target, &domain, &mut rng)
                    else {
                        return McmcOutcome::Held;
                    };
                    let log_alpha = log_alpha_with_extra(&proposal, &target, 0.0).unwrap();
                    if !metropolis_accept(log_alpha, &mut rng) {
                        return McmcOutcome::Rejected;
                    }
                    proposal.apply(&mut state);
                    McmcOutcome::Accepted
                })
                .collect()
        };

        let run_step = |seed: u64| -> Vec<McmcOutcome> {
            let mut state = make_state(n, &so, &si, OccupationFamily::ME, true);
            let mut rng = StdRng::seed_from_u64(seed);
            (0..100usize)
                .map(|_| occupied_cycle4_step(&mut state, &target, &domain, &mut rng))
                .collect()
        };

        for seed in [1u64, 42, 99, 12345] {
            assert_eq!(run_composed(seed), run_step(seed), "seed {seed}");
        }
    }
}
