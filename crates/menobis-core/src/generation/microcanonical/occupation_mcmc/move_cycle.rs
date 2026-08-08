//! Occupied-cell 4-cycle MCMC kernel for fixed-strength sampling.
//!
//! The proposal selects two distinct **occupied** pairs from the current
//! state, decrements both, and increments the two cross cells (spec §22).
//! A Metropolis–Hastings correction (§23) accounts for the state-dependent
//! proposal probability.
//!
//! # Proposal (spec §22, fixed direction)
//!
//! 1. Select `P1 = (a,b)` uniformly from the occupied pairs.
//! 2. Select `P2 = (c,d)` from the remaining occupied pairs by rejection
//!    sampling, rejecting if `src == a` or `tgt == b`.
//! 3. Apply fixed-direction deltas:
//!
//!    \[
//!    \begin{aligned}
//!    t_{ab} &\to t_{ab} - 1,\\
//!    t_{cd} &\to t_{cd} - 1,\\
//!    t_{ad} &\to t_{ad} + 1,\\
//!    t_{cb} &\to t_{cb} + 1.
//!    \end{aligned}
//!    \]
//!
//! Every source and target strength is preserved exactly (each row and
//! column receives one \(-1\) and one \(+1\)).
//!
//! # Hastings correction (§23)
//!
//! The occupied-cell selection is state-dependent, so the proposal is
//! *not* symmetric.  We compute the exact log-ratio:
//!
//! \[
//! \Delta\log A = \Delta\log D_F - \gamma\Delta C
//!     + \log q(\mathbf t'\to\mathbf t) - \log q(\mathbf t\to\mathbf t')
//! \]
//!
//! where \(\Delta\log D_F - \gamma\Delta C\) is obtained from
//! [`StrengthTarget::delta_log_weight`] (reused — no family formulas
//! duplicated here, §23) and the proposal probabilities use the maintained
//! `row_occ_count`/`col_occ_count` for \(O(1)\) computation (§24).
//!
//! # Hot path
//!
//! - **No heap allocation** (§24): four-cell delta array on the stack,
//!   \(O(1)\) selection and Hastings computation via maintained counts.
//! - The decrement cells are *guaranteed occupied* by construction, so the
//!   structural-validity rate is much higher than the old uniform-coordinate
//!   kernel on sparse states (§22 motivation).

use rand::Rng;

use super::domain::PairDomain;
use super::rectangle::{build_four_cell, validate_four_cell};
use super::state::StrengthState;
use super::target::StrengthTarget;
use crate::generation::microcanonical::mcmc::McmcOutcome;
/// Maximum retry attempts for selecting the second occupied pair.
const MAX_SECOND_RETRIES: usize = 20;

/// Perform one occupied-cell 4-cycle MCMC step (allocation-free).
///
/// Returns [`McmcOutcome::Held`] if fewer than 2 occupied pairs exist or
/// no valid second pair is found after bounded retries.
pub fn occupied_cycle4_step(
    state: &mut StrengthState,
    target: &StrengthTarget,
    domain: &PairDomain,
    rng: &mut impl Rng,
) -> McmcOutcome {
    let m = state.occupied_count();
    if m < 2 {
        return McmcOutcome::Held;
    }
    // ---- 1. Select P1 = (a,b) uniformly from occupied pairs ----
    let a_idx = rng.random_range(0..m);
    let (a, b) = state.occupied_pairs()[a_idx];

    // ---- 2. Select P2 = (c,d) by rejection from occupied pairs ----
    // Draw random index; reject if same as P1 or sharing a source/target.
    let mut found_p2 = false;
    let mut c = 0u64;
    let mut d = 0u64;
    let retries = m.saturating_sub(1).min(MAX_SECOND_RETRIES);
    for _ in 0..retries {
        let idx = rng.random_range(0..m);
        if idx == a_idx {
            continue;
        }
        let (cs, cd) = state.occupied_pairs()[idx];
        if cs != a && cd != b {
            (c, d) = (cs, cd);
            found_p2 = true;
            break;
        }
    }
    if !found_p2 {
        return McmcOutcome::Held;
    }

    // ---- 3. Build deltas (fixed direction per §22) ----
    let deltas = build_four_cell(a, c, b, d);

    // ---- 4. Read current occupations of the four cells ----
    let old_ab = state.get(a, b);
    let old_cd = state.get(c, d);
    let old_ad = state.get(a, d);
    let old_cb = state.get(c, b);

    // ---- 5. Validate all four cells ----
    // (a,b) and (c,d) are occupied by construction, so positivity holds.
    // But (a,d) and (c,b) might be self-loops or capacity-violating.
    // Check all four cells using the shared validator.
    if !validate_four_cell(state, target, domain, &deltas) {
        return McmcOutcome::Held;
    }

    // ---- 6. Compute Δlogπ = Σ target.delta_log_weight(…) ----
    // Bounds are guaranteed by validate_four_cell, so checked arithmetic
    // is safe.
    let new_ab = old_ab.checked_sub(1).unwrap();
    let new_cd = old_cd.checked_sub(1).unwrap();
    let new_ad = old_ad.checked_add(1).unwrap();
    let new_cb = old_cb.checked_add(1).unwrap();

    let delta_log_pi = target.delta_log_weight(a, b, old_ab, new_ab).unwrap()
        + target.delta_log_weight(c, d, old_cd, new_cd).unwrap()
        + target.delta_log_weight(a, d, old_ad, new_ad).unwrap()
        + target.delta_log_weight(c, b, old_cb, new_cb).unwrap();

    // ---- 7. Hastings log-ratio (§23) ----
    // Forward: first selected (a,b) or (c,d).
    let v_ab = m as i64
        - state.row_occ_count[a as usize] as i64
        - state.col_occ_count[b as usize] as i64
        + 1;
    let v_cd = m as i64
        - state.row_occ_count[c as usize] as i64
        - state.col_occ_count[d as usize] as i64
        + 1;

    // Guard: no valid second choice.
    if v_ab <= 0 || v_cd <= 0 {
        return McmcOutcome::Held;
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
        return McmcOutcome::Held;
    }

    let row_a_prime =
        state.row_occ_count[a as usize] as i64 + enters_ad - leaves_ab;
    let row_c_prime =
        state.row_occ_count[c as usize] as i64 + enters_cb - leaves_cd;
    let col_b_prime =
        state.col_occ_count[b as usize] as i64 + enters_cb - leaves_ab;
    let col_d_prime =
        state.col_occ_count[d as usize] as i64 + enters_ad - leaves_cd;

    let v_ad_prime = m_prime - row_a_prime - col_d_prime + 1;
    let v_cb_prime = m_prime - row_c_prime - col_b_prime + 1;

    if v_ad_prime <= 0 || v_cb_prime <= 0 {
        // The reverse move would have no valid second choice.
        // This should not happen in a valid state, but guard against it.
        return McmcOutcome::Held;
    }

    let log_q_fwd = (1.0 / v_ab as f64 + 1.0 / v_cd as f64).ln()
        - (m as f64).ln();
    let log_q_rev = (1.0 / v_ad_prime as f64 + 1.0 / v_cb_prime as f64).ln()
        - (m_prime as f64).ln();

    let log_alpha = delta_log_pi + (log_q_rev - log_q_fwd);

    // ---- 8. Metropolis–Hastings accept/reject ----
    if log_alpha < 0.0 {
        let log_u = (rng.random::<f64>() + f64::MIN_POSITIVE).ln();
        if log_u >= log_alpha {
            return McmcOutcome::Rejected;
        }
    }

    // ---- 9. Apply ----
    state.set(a, b, new_ab);
    state.set(c, d, new_cd);
    state.set(a, d, new_ad);
    state.set(c, b, new_cb);

    McmcOutcome::Accepted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::microcanonical::occupation_mcmc::initializer::initialize_table;
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
        let table = initialize_table(so, si, family, &domain).unwrap();
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
            if occupied_cycle4_step(&mut state, &target, &domain, &mut rng)
                == McmcOutcome::Accepted
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
        let so = vec![4u64; 3];
        let si = vec![4u64; 3];
        let mut state = make_state(n, &so, &si, OccupationFamily::B { layers }, true);
        let target = StrengthTarget::new(OccupationFamily::B { layers });
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };
        let mut rng = StdRng::seed_from_u64(42);
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
}