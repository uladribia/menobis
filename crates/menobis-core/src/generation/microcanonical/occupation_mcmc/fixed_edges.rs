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

use super::chain::repair_state;
use super::domain::PairDomain;
use super::errors::FixedStrengthError;
use super::initializer::initialize_table;
use super::move_cycle::{
    draw_cycle4_proposal, log_alpha_with_extra, metropolis_accept, Cycle4Proposal,
};
use super::problem::ResidualStrengthProblem;
use super::state::StrengthState;
use super::target::StrengthTarget;
use crate::model::family::OccupationFamily;
use crate::OccNum;

/// Outcome of one exact-E local step, distinguishing the edge-count veto
/// from ordinary holds for mobility diagnostics (§20).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeLocalOutcome {
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
pub fn exact_e_local_step(
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

// ---------------------------------------------------------------------------
// Auxiliary bridge kernel (§8–§10)
// ---------------------------------------------------------------------------

/// Tuning constants for the auxiliary edge-biased bridge (§6, §8–§10).
///
/// These are performance parameters, not correctness parameters — any
/// `lambda > 0` and any finite step cap preserve the exact stationary
/// proof (§11).  Kept internal; not exposed through the public API.
#[derive(Clone, Copy, Debug)]
pub struct BridgeConfig {
    /// Mixture weight `ρ`: probability of attempting a bridge per outer
    /// proposal (§6, §18).
    pub bridge_probability: f64,
    /// Edge-distance potential strength `λ` (§8).
    pub bridge_lambda: f64,
    /// Maximum auxiliary substeps inside one bridge attempt (§10).
    pub bridge_max_steps: usize,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            bridge_probability: 0.05,
            bridge_lambda: 1.0,
            bridge_max_steps: 16,
        }
    }
}

/// Outcome of one auxiliary edge-potential MH substep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuxSubstepOutcome {
    /// Proposal applied.
    Accepted,
    /// Proposal invalid at the family/domain/capacity level.
    HeldInvalid,
    /// Valid proposal rejected by the Metropolis criterion.
    Rejected,
}

/// One auxiliary MH substep targeting `mu_lambda(t) ∝ pi_s(t) exp(−λ
/// |E(t) − E_target|)` (§9).
///
/// Uses exactly the same occupied-cell proposal law and Hastings proposal
/// ratio; the only difference from the ordinary fixed-strength acceptance
/// is the edge-distance potential:
///
/// ```text
/// log_alpha_aux = delta_log_family + log_q_reverse − log_q_forward
///                 − λ (|E_new − E_target| − |E_old − E_target|)
/// ```
///
/// Returns the outcome plus the applied proposal when accepted (the
/// bridge records accepted proposals for deterministic undo, §19).
pub fn auxiliary_substep(
    state: &mut StrengthState,
    target: &StrengthTarget,
    domain: &PairDomain,
    rng: &mut impl Rng,
    edge_target: usize,
    lambda: f64,
) -> (AuxSubstepOutcome, Option<Cycle4Proposal>) {
    let Some(proposal) = draw_cycle4_proposal(state, target, domain, rng) else {
        return (AuxSubstepOutcome::HeldInvalid, None);
    };
    let d_old = proposal.occupied_before.abs_diff(edge_target) as f64;
    let d_new = proposal.occupied_after.abs_diff(edge_target) as f64;
    let delta_edge_potential = -lambda * (d_new - d_old);
    let Some(log_alpha) = log_alpha_with_extra(&proposal, target, delta_edge_potential) else {
        return (AuxSubstepOutcome::HeldInvalid, None);
    };
    if !metropolis_accept(log_alpha, rng) {
        return (AuxSubstepOutcome::Rejected, None);
    }
    proposal.apply(state);
    (AuxSubstepOutcome::Accepted, Some(proposal))
}

/// Result of one bridge attempt, for mobility diagnostics (§20).
#[derive(Clone, Copy, Debug, Default)]
pub struct BridgeOutcome {
    /// Whether the bridge returned to the exact-E fiber with a new state.
    pub success: bool,
    /// Whether the first substep left the exact-E fiber.
    pub departed: bool,
    /// Auxiliary substeps executed.
    pub substeps: usize,
    /// Accepted auxiliary substeps.
    pub accepted_substeps: usize,
}

/// Bridge counters for the production mixed kernel (§20).  Diagnostics
/// only; a timeout is a self-loop in the exact chain, never an error.
#[derive(Clone, Debug, Default)]
pub struct FixedEdgeCounters {
    /// Outer proposals (local steps + bridge attempts).
    pub outer_proposals: u64,
    /// Local kernel: accepted in-fiber moves.
    pub local_accepted: u64,
    /// Local kernel: invalid proposals.
    pub local_held_invalid: u64,
    /// Local kernel: live proposals vetoed because they would change E.
    pub local_held_edge_veto: u64,
    /// Local kernel: Metropolis rejections.
    pub local_mh_rejected: u64,
    /// Bridge attempts.
    pub bridge_attempts: u64,
    /// Bridge attempts whose first substep left the fiber.
    pub bridge_departures: u64,
    /// Bridge attempts that returned to the fiber with a new state.
    pub bridge_successful_returns: u64,
    /// Bridge attempts that timed out and restored the origin.
    pub bridge_timeouts: u64,
    /// Total auxiliary substeps executed inside bridges.
    pub bridge_auxiliary_substeps: u64,
    /// Accepted auxiliary substeps inside bridges.
    pub bridge_auxiliary_accepted: u64,
}

/// One bridge attempt (§10): a censored excursion of the exact reversible
/// auxiliary chain, starting and ending in the exact-E fiber.
///
/// A successful bridge path has exactly the form
///
/// ```text
/// x ∈ A, z1 ∉ A, ..., z_(k−1) ∉ A, y ∈ A        (2 ≤ k ≤ bridge_max_steps)
/// ```
///
/// - the first substep must depart the fiber (held/rejected or an
///   accepted in-fiber move aborts and restores the origin);
/// - the first return to the fiber stops and keeps the returned state;
/// - if no return occurs within the cap, every accepted substep is undone
///   deterministically (no state cloning, §19) and the origin is
///   restored — a self-loop in the exact chain.
///
/// Precondition: `state.occupied_count() == edge_target` (in the fiber).
pub fn bridge_step(
    state: &mut StrengthState,
    target: &StrengthTarget,
    domain: &PairDomain,
    rng: &mut impl Rng,
    edge_target: usize,
    config: &BridgeConfig,
) -> BridgeOutcome {
    let mut accepted_proposals: Vec<Cycle4Proposal> = Vec::with_capacity(config.bridge_max_steps);
    let mut substeps = 0usize;
    let mut aux_accepted = 0usize;

    for step in 0..config.bridge_max_steps {
        substeps += 1;
        let (outcome, applied) = auxiliary_substep(
            state,
            target,
            domain,
            rng,
            edge_target,
            config.bridge_lambda,
        );
        match outcome {
            AuxSubstepOutcome::Accepted => {
                aux_accepted += 1;
                let proposal = applied.expect("accepted substep must return the proposal");
                if state.occupied_count() == edge_target {
                    if step == 0 {
                        // In-fiber move on the first substep: abort and
                        // restore the origin.
                        proposal.undo(state);
                        return BridgeOutcome {
                            success: false,
                            departed: false,
                            substeps,
                            accepted_substeps: aux_accepted,
                        };
                    }
                    // First return to the fiber: keep the state.
                    return BridgeOutcome {
                        success: true,
                        departed: true,
                        substeps,
                        accepted_substeps: aux_accepted,
                    };
                }
                accepted_proposals.push(proposal);
            }
            AuxSubstepOutcome::HeldInvalid | AuxSubstepOutcome::Rejected => {
                if step == 0 {
                    // First substep held or rejected: still in the fiber
                    // (we started there).  Abort as a bridge self-loop.
                    return BridgeOutcome {
                        success: false,
                        departed: false,
                        substeps,
                        accepted_substeps: aux_accepted,
                    };
                }
                // Outside the fiber, no move: keep exploring.
            }
        }
    }

    // Cap reached without a return: undo every accepted substep in
    // reverse order and restore the origin (§19).
    for proposal in accepted_proposals.iter().rev() {
        proposal.undo(state);
    }
    BridgeOutcome {
        success: false,
        departed: true,
        substeps,
        accepted_substeps: aux_accepted,
    }
}

// ---------------------------------------------------------------------------
// Edge-target feasibility (§14)
// ---------------------------------------------------------------------------

/// Internal edge-repair tuning knobs (§13.3).  Safety limits only, not
/// correctness constants; kept out of the public API.
#[derive(Clone, Copy, Debug)]
pub struct EdgeRepairConfig {
    /// Maximum rectangle-repair steps per reconstruction restart.
    pub max_steps_per_restart: u64,
    /// Maximum reconstruction restarts before giving up.
    pub max_restarts: u32,
}

impl Default for EdgeRepairConfig {
    fn default() -> Self {
        Self {
            max_steps_per_restart: 1_000_000,
            max_restarts: 5,
        }
    }
}

/// Result summary of a successful (or best-effort) edge repair.
#[derive(Clone, Copy, Debug)]
pub struct EdgeRepairOutcome {
    /// Total repair steps used across all restarts.
    pub steps: u64,
    /// Reconstruction restarts performed (0 = first attempt).
    pub restarts: u32,
    /// Occupied-pair count of the first constructed state.
    pub initial_edges: usize,
    /// Best occupied-pair count reached.
    pub best_edges: usize,
    /// Absolute distance `|best − target|` of the best state.
    pub best_distance: usize,
}

/// Validate the residual edge target against necessary feasibility bounds
/// (§14).  These are necessary conditions only — a target passing them
/// may still be infeasible for sparse structural-zero domains, in which
/// case the repair exhausts with [`EdgeRepairExhausted`] instead of
/// [`InvalidEdgeTarget`](FixedStrengthError::InvalidEdgeTarget).
///
/// With `T = Σ residual strengths` and `A = admissible pair count`:
///
/// - `T == 0 ⇒ E_target == 0`;
/// - `T > 0 ⇒ E_target ≥ 1`;
/// - `E_target ≤ T` (every occupied pair carries at least one event);
/// - `E_target ≤ A` (cannot occupy more coordinates than exist);
/// - B: `E_target ≥ ceil(T/M)` and `E_target ≥ Σ_rows ceil(s/M)` and
///   the column analogue;
/// - ME/W: `E_target ≥` number of positive-strength rows/columns.
pub fn validate_edge_target(
    residual: &ResidualStrengthProblem,
    target_edges: usize,
) -> Result<(), FixedStrengthError> {
    let t: OccNum = residual.total;
    let a: usize = residual.domain.admissible_pair_count();
    let e = target_edges;

    if t == 0 {
        if e != 0 {
            return Err(FixedStrengthError::InvalidEdgeTarget(format!(
                "total occupation is 0, edge target must be 0, got {e}"
            )));
        }
        return Ok(());
    }

    if e == 0 {
        return Err(FixedStrengthError::InvalidEdgeTarget(format!(
            "edge target {e} must be at least 1 when total occupation {t} > 0"
        )));
    }
    if e > t as usize {
        return Err(FixedStrengthError::InvalidEdgeTarget(format!(
            "edge target {e} exceeds total occupation {t}"
        )));
    }
    if e > a {
        return Err(FixedStrengthError::InvalidEdgeTarget(format!(
            "edge target {e} exceeds admissible pairs {a}"
        )));
    }

    match residual.family {
        OccupationFamily::B { layers } => {
            let m = layers as OccNum;
            // Each occupied pair carries at most M events.
            let min_m = (t as f64 / m as f64).ceil() as usize;
            if e < min_m {
                return Err(FixedStrengthError::InvalidEdgeTarget(format!(
                    "edge target {e} below required minimum {min_m} (B capacity M={layers}, total {t})"
                )));
            }
            // Each row/column occupied coordinate carries at most M events.
            let min_rows: OccNum = residual.strength_out.iter().map(|&s| s.div_ceil(m)).sum();
            if e < min_rows as usize {
                return Err(FixedStrengthError::InvalidEdgeTarget(format!(
                    "edge target {e} below required minimum {min_rows} (out-strength capacity M={layers})"
                )));
            }
            let min_cols: OccNum = residual.strength_in.iter().map(|&s| s.div_ceil(m)).sum();
            if e < min_cols as usize {
                return Err(FixedStrengthError::InvalidEdgeTarget(format!(
                    "edge target {e} below required minimum {min_cols} (in-strength capacity M={layers})"
                )));
            }
        }
        _ => {
            // ME / W: every positive-strength row/column needs at least
            // one occupied pair.
            let min_nodes = residual
                .strength_out
                .iter()
                .filter(|&&s| s > 0)
                .count()
                .max(residual.strength_in.iter().filter(|&&s| s > 0).count());
            if e < min_nodes {
                return Err(FixedStrengthError::InvalidEdgeTarget(format!(
                    "edge target {e} below required minimum {min_nodes} (positive node rows/columns)"
                )));
            }
        }
    }
    Ok(())
}

/// Residual edge target: `E_residual = E_full − E_fixed`, where
/// `E_fixed` is the number of **unique** fixed coordinates with positive
/// occupation (§16).  Zero-occupation fixed pairs contribute 0 but still
/// exclude their coordinate from the residual domain (§3).
///
/// Precondition: duplicate fixed coordinates must already have been
/// rejected (see [`FixedStrengthProblem::into_residual`](
/// super::problem::FixedStrengthProblem::into_residual)).  The unique
/// count is computed defensively regardless.
//
// # Errors
//
// - [`InvalidEdgeTarget`](FixedStrengthError::InvalidEdgeTarget) if
//   `E_full < E_fixed`.
pub fn residual_edge_target(
    full_target: usize,
    fixed_pairs: &[(u64, u64, OccNum)],
) -> Result<usize, FixedStrengthError> {
    let mut seen = std::collections::HashSet::with_capacity(fixed_pairs.len());
    let mut fixed_positive = 0usize;
    for &(src, tgt, occ) in fixed_pairs {
        if occ > 0 && seen.insert((src, tgt)) {
            fixed_positive += 1;
        }
    }
    if full_target < fixed_positive {
        return Err(FixedStrengthError::InvalidEdgeTarget(format!(
            "edge target {full_target} is below the {fixed_positive} positive fixed pairs"
        )));
    }
    Ok(full_target - fixed_positive)
}

// ---------------------------------------------------------------------------
// Edge-count initialization repair (§13)
// ---------------------------------------------------------------------------

/// Fresh randomized fixed-strength state: compressed construction + the
/// same structural repair phases as the fixed-strength pipeline.
///
/// Each call continues from the caller's RNG stream, so reconstruction
/// restarts produce different states (§13.4).
pub fn edge_repair_rebuild(
    problem: &ResidualStrengthProblem,
    rng: &mut impl Rng,
) -> Result<StrengthState, FixedStrengthError> {
    let table = initialize_table(
        &problem.strength_out,
        &problem.strength_in,
        problem.family,
        &problem.domain,
        rng,
    )?;
    let mut state = StrengthState::new(problem.domain.node_count(), table);
    let (_, _, _) = repair_state(&mut state, problem, rng)?;
    Ok(state)
}

/// Bring a state to the exact residual edge count using the **biased**
/// initialization-only acceptance rule (§13.2):
///
/// ```text
/// d_old = |occupied_before − E_target|
/// d_new = |occupied_after  − E_target|
///
/// if d_new < d_old:                     accept
/// else if d_new == d_old:                accept with probability 0.10
/// else:                                  accept with probability exp(−2·(d_new−d_old))
/// ```
///
/// The proposal law is exactly the shared fixed-direction cycle candidate
/// (no separate rectangle selection, §13.1).  This phase only needs to
/// find *one* feasible exact-`E` start; it is never part of the
/// stationary sampler (§4, §41).
///
/// On exhaustion of a restart budget, the state is discarded, a fresh
/// randomized fixed-strength state is reconstructed (structural repair
/// included), and the repair retries (§13.3).  If every restart fails,
/// an [`EdgeRepairExhausted`](FixedStrengthError::EdgeRepairExhausted)
/// error is returned with best/target E and restart/step diagnostics.
pub fn repair_to_edge_target(
    state: &mut StrengthState,
    problem: &ResidualStrengthProblem,
    rng: &mut impl Rng,
    edge_target: usize,
    config: &EdgeRepairConfig,
) -> Result<EdgeRepairOutcome, FixedStrengthError> {
    let target = &StrengthTarget::new(problem.family);
    let domain = &problem.domain;

    let initial_edges = state.occupied_count();
    let mut best_edges = initial_edges;
    let mut best_distance = initial_edges.abs_diff(edge_target);
    let mut total_steps: u64 = 0;

    for restart in 0..config.max_restarts {
        if restart > 0 {
            // Discard and reconstruct from the same RNG stream.
            *state = edge_repair_rebuild(problem, rng)?;
        }

        let mut steps: u64 = 0;
        loop {
            if state.occupied_count() == edge_target {
                if best_distance != 0 {
                    best_edges = edge_target;
                    best_distance = 0;
                }
                return Ok(EdgeRepairOutcome {
                    steps: total_steps,
                    restarts: restart,
                    initial_edges,
                    best_edges,
                    best_distance,
                });
            }
            if steps >= config.max_steps_per_restart {
                break;
            }

            let Some(proposal) = draw_cycle4_proposal(state, target, domain, rng) else {
                // Structurally invalid proposal: no move; keep trying.
                steps += 1;
                total_steps += 1;
                continue;
            };
            steps += 1;
            total_steps += 1;

            // Track the best state reached (for structured diagnostics).
            let d_cur = state.occupied_count().abs_diff(edge_target);
            if d_cur < best_distance {
                best_distance = d_cur;
                best_edges = state.occupied_count();
            }

            let d_old = proposal.occupied_before.abs_diff(edge_target);
            let d_new = proposal.occupied_after.abs_diff(edge_target);

            let accept = if d_new < d_old {
                true
            } else if d_new == d_old {
                rng.random::<f64>() < 0.10
            } else {
                rng.random::<f64>() < (-2.0 * (d_new - d_old) as f64).exp()
            };
            if accept {
                proposal.apply(state);
                let d_after = state.occupied_count().abs_diff(edge_target);
                if d_after < best_distance {
                    best_distance = d_after;
                    best_edges = state.occupied_count();
                }
            }
        }
    }

    Err(FixedStrengthError::EdgeRepairExhausted {
        best_edges,
        target_edges: edge_target,
        best_distance,
        restarts: config.max_restarts,
        total_steps,
    })
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

    // -------------------------------------------------------------------
    // Phase 5: edge-target validation + initialization repair
    // -------------------------------------------------------------------

    fn make_problem(
        family: OccupationFamily,
        so: Vec<OccNum>,
        si: Vec<OccNum>,
        sl: bool,
    ) -> ResidualStrengthProblem {
        let domain = PairDomain::Complete {
            node_count: so.len(),
            self_loops: sl,
        };
        crate::generation::microcanonical::occupation_mcmc::problem::FixedStrengthProblem::new(
            family,
            so,
            si,
            domain,
            vec![],
        )
        .unwrap()
        .into_residual()
        .unwrap()
    }

    fn sorted_pairs(state: &StrengthState) -> Vec<((u64, u64), OccNum)> {
        let mut pairs: Vec<_> = state.iter_occupied().collect();
        pairs.sort_unstable();
        pairs
    }

    #[test]
    fn validate_edge_target_enforces_bounds() {
        // E > T: strengths [4;4], T=16.
        let problem = make_problem(
            OccupationFamily::ME,
            vec![4, 4, 4, 4],
            vec![4, 4, 4, 4],
            true,
        );
        assert!(matches!(
            validate_edge_target(&problem, 17),
            Err(FixedStrengthError::InvalidEdgeTarget(msg)) if msg.contains("exceeds total occupation")
        ));
        assert!(validate_edge_target(&problem, 16).is_ok());

        // E == 0 with T > 0.
        assert!(matches!(
            validate_edge_target(&problem, 0),
            Err(FixedStrengthError::InvalidEdgeTarget(msg)) if msg.contains("at least 1")
        ));

        // T == 0 ⇒ E == 0.
        let zero = make_problem(OccupationFamily::ME, vec![0, 0], vec![0, 0], true);
        assert!(validate_edge_target(&zero, 0).is_ok());
        assert!(matches!(
            validate_edge_target(&zero, 1),
            Err(FixedStrengthError::InvalidEdgeTarget(_))
        ));

        // E > A: ME N=2 s=[3,3], T=6, A=4 → E=5 passes E ≤ T but fails E ≤ A.
        let problem = make_problem(OccupationFamily::ME, vec![3, 3], vec![3, 3], true);
        assert!(matches!(
            validate_edge_target(&problem, 5),
            Err(FixedStrengthError::InvalidEdgeTarget(msg)) if msg.contains("exceeds admissible pairs")
        ));
        assert!(validate_edge_target(&problem, 4).is_ok());

        // B capacity: M=2, strengths [4;4], T=16 → min ceil(16/2)=8.
        let b = make_problem(
            OccupationFamily::B { layers: 2 },
            vec![4, 4, 4, 4],
            vec![4, 4, 4, 4],
            true,
        );
        assert!(matches!(
            validate_edge_target(&b, 7),
            Err(FixedStrengthError::InvalidEdgeTarget(msg)) if msg.contains("required minimum")
        ));
        assert!(validate_edge_target(&b, 8).is_ok());

        // B row bound dominates: M=2, strengths [5,1,1,1] → rows ⌈5/2⌉=3+1+1+1=6.
        let b = make_problem(
            OccupationFamily::B { layers: 2 },
            vec![5, 1, 1, 1],
            vec![1, 1, 1, 5],
            true,
        );
        assert!(matches!(
            validate_edge_target(&b, 5),
            Err(FixedStrengthError::InvalidEdgeTarget(msg)) if msg.contains("required minimum")
        ));
        assert!(validate_edge_target(&b, 6).is_ok());

        // ME positive rows/columns bound: strengths [2,2,2] → min 3.
        let me = make_problem(OccupationFamily::ME, vec![2, 2, 2], vec![2, 2, 2], true);
        assert!(matches!(
            validate_edge_target(&me, 2),
            Err(FixedStrengthError::InvalidEdgeTarget(msg)) if msg.contains("required minimum")
        ));
        assert!(validate_edge_target(&me, 3).is_ok());
    }

    #[test]
    fn residual_edge_target_subtracts_positive_fixed() {
        // Two unique positive fixed pairs and one zero pair: E_res = E_full − 2.
        let fixed = vec![(0, 1, 3), (1, 0, 0), (2, 2, 5)];
        assert_eq!(residual_edge_target(10, &fixed).unwrap(), 8);
        // Duplicates are counted once (defensive; upstream rejects them).
        let dup = vec![(0, 1, 3), (0, 1, 4)];
        assert_eq!(residual_edge_target(5, &dup).unwrap(), 4);
        // E_full below fixed positive count.
        assert!(matches!(
            residual_edge_target(1, &fixed),
            Err(FixedStrengthError::InvalidEdgeTarget(_))
        ));
    }

    #[test]
    fn repair_reaches_exact_e_on_known_feasible_fibers() {
        type RepairCase = (OccupationFamily, Vec<OccNum>, Vec<OccNum>, bool, Vec<usize>);
        // Feasible E values proven by explicit constructions:
        // - ME N=2 sl s=2: E=2 {(0,0)=2,(1,1)=2}; E=4 all-ones.
        // - ME N=2 loopless s=2: the only state is the cross matching, E=2.
        // - ME N=2 sl s=[3,1]/[1,3]: E=2 {(0,1)=3,(1,0)=1}; E=3 {(0,0)=1,(0,1)=2,(1,1)=1}.
        // - W M=1, B M=2: same N=2 topologies as ME.
        let cases: Vec<RepairCase> = vec![
            (
                OccupationFamily::ME,
                vec![2, 2],
                vec![2, 2],
                true,
                vec![2, 4],
            ),
            (OccupationFamily::ME, vec![2, 2], vec![2, 2], false, vec![2]),
            (
                OccupationFamily::ME,
                vec![3, 1],
                vec![1, 3],
                true,
                vec![2, 3],
            ),
            (
                OccupationFamily::W { layers: 1 },
                vec![2, 2],
                vec![2, 2],
                true,
                vec![2, 4],
            ),
            (
                OccupationFamily::B { layers: 2 },
                vec![2, 2],
                vec![2, 2],
                true,
                vec![2, 4],
            ),
        ];
        for (family, so, si, sl, feasible) in cases {
            let problem = make_problem(family, so.clone(), si.clone(), sl);
            for e in feasible {
                let mut rng = StdRng::seed_from_u64(42);
                let mut state = edge_repair_rebuild(&problem, &mut rng).unwrap();
                let outcome = repair_to_edge_target(
                    &mut state,
                    &problem,
                    &mut rng,
                    e,
                    &EdgeRepairConfig::default(),
                )
                .unwrap();
                assert_eq!(state.occupied_count(), e, "{family:?} sl={sl} E={e}");
                assert_eq!(state.out_strengths, so, "out-strengths changed");
                assert_eq!(state.in_strengths, si, "in-strengths changed");
                assert_eq!(outcome.best_distance, 0);
                state.debug_validate();
            }
        }
    }

    #[test]
    fn repair_infeasible_target_exhausts_with_structured_error() {
        // ME N=2 s=[2,2]: feasible E ∈ {2,4}; E=3 passes the necessary
        // bounds (3 ≤ T=4, 3 ≤ A=4, ≥ 2 positive rows) but is infeasible.
        let problem = make_problem(OccupationFamily::ME, vec![2, 2], vec![2, 2], true);
        validate_edge_target(&problem, 3).unwrap();

        let mut rng = StdRng::seed_from_u64(42);
        let mut state = edge_repair_rebuild(&problem, &mut rng).unwrap();
        let config = EdgeRepairConfig {
            max_steps_per_restart: 40,
            max_restarts: 2,
        };
        match repair_to_edge_target(&mut state, &problem, &mut rng, 3, &config) {
            Err(FixedStrengthError::EdgeRepairExhausted {
                best_edges,
                target_edges,
                best_distance,
                restarts,
                total_steps,
            }) => {
                assert_eq!(target_edges, 3);
                assert_eq!(restarts, 2);
                assert!(
                    best_edges == 2 || best_edges == 4,
                    "best_edges {best_edges}"
                );
                assert!(best_distance >= 1);
                assert!(total_steps > 0);
            }
            other => panic!("expected EdgeRepairExhausted, got {other:?}"),
        }
    }

    #[test]
    fn repair_reproducible_by_seed() {
        let run = |seed: u64| -> Vec<((u64, u64), OccNum)> {
            let problem = make_problem(
                OccupationFamily::ME,
                vec![3, 3, 3, 3],
                vec![3, 3, 3, 3],
                true,
            );
            let mut rng = StdRng::seed_from_u64(seed);
            let mut state = edge_repair_rebuild(&problem, &mut rng).unwrap();
            repair_to_edge_target(
                &mut state,
                &problem,
                &mut rng,
                6,
                &EdgeRepairConfig::default(),
            )
            .unwrap();
            assert_eq!(state.occupied_count(), 6);
            sorted_pairs(&state)
        };
        assert_eq!(run(1), run(1), "same seed must reproduce the repair");
    }

    #[test]
    fn repair_rebuild_restarts_differ_on_same_stream() {
        // Repeated reconstruction from one RNG stream must produce
        // different initial states (§13.4: restarts reconstruct
        // differently), while staying reproducible per seed.
        let problem = make_problem(
            OccupationFamily::ME,
            vec![7, 5, 3, 9, 2],
            vec![4, 8, 6, 3, 5],
            true,
        );
        let mut rng = StdRng::seed_from_u64(42);
        let mut distinct = std::collections::HashSet::new();
        for _ in 0..8 {
            let state = edge_repair_rebuild(&problem, &mut rng).unwrap();
            distinct.insert(sorted_pairs(&state));
        }
        assert!(
            distinct.len() > 1,
            "restarts from the same stream should produce distinct states, got {}",
            distinct.len()
        );
    }

    // -------------------------------------------------------------------
    // Phase 6: auxiliary bridge
    // -------------------------------------------------------------------

    #[test]
    fn bridge_connects_counterexample_fiber() {
        // The §5 fiber: ME N=2 s=[2,2] sl=true, E=2, states A={(0,0)=2,
        // (1,1)=2} and B={(0,1)=2,(1,0)=2}.  The local kernel alone is a
        // pure self-loop here; the bridge must connect A and B.
        let n = 2;
        let so = vec![2u64; 2];
        let si = vec![2u64; 2];
        let _state_a = sorted_pairs(&StrengthState::new(n, vec![((0, 0), 2), ((1, 1), 2)]));
        let state_b = sorted_pairs(&StrengthState::new(n, vec![((0, 1), 2), ((1, 0), 2)]));
        let target = StrengthTarget::new(OccupationFamily::ME);
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };
        let config = BridgeConfig::default();

        let mut rng = StdRng::seed_from_u64(7);
        let mut landed_on_b = 0usize;
        let mut successes = 0usize;
        for _ in 0..500 {
            // Reset to A before each bridge attempt.
            let mut state = StrengthState::new(n, vec![((0, 0), 2), ((1, 1), 2)]);
            let outcome = bridge_step(&mut state, &target, &domain, &mut rng, 2, &config);
            assert_eq!(state.occupied_count(), 2, "bridge left the fiber");
            assert_eq!(state.out_strengths, so);
            assert_eq!(state.in_strengths, si);
            if outcome.success {
                successes += 1;
                if sorted_pairs(&state) == state_b {
                    landed_on_b += 1;
                }
            }
        }
        assert!(successes > 0, "expected some successful bridge returns");
        assert!(
            landed_on_b > 0,
            "bridge never reached the other fiber state B ({landed_on_b}/500)"
        );
    }

    #[test]
    fn bridge_max_steps_one_always_undoes() {
        // With bridge_max_steps = 1 no bridge can complete (a success
        // needs at least depart + return).  Every accepted departure must
        // be deterministically undone, restoring the exact origin.
        let n = 2;
        let start = vec![((0, 0), 2), ((1, 1), 2)];
        let origin = sorted_pairs(&StrengthState::new(n, start.clone()));
        let target = StrengthTarget::new(OccupationFamily::ME);
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };
        let config = BridgeConfig {
            bridge_max_steps: 1,
            ..BridgeConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(9);
        let mut departed = 0usize;
        let mut timeouts = 0usize;
        for _ in 0..100 {
            let mut state = StrengthState::new(n, start.clone());
            let outcome = bridge_step(&mut state, &target, &domain, &mut rng, 2, &config);
            assert!(!outcome.success, "max_steps=1 cannot succeed");
            assert_eq!(
                sorted_pairs(&state),
                origin,
                "failed bridge must restore the exact origin"
            );
            if outcome.departed {
                departed += 1;
            }
            assert_eq!(state.occupied_count(), 2);
            if outcome.substeps == 1 && outcome.accepted_substeps == 1 {
                timeouts += 1; // departed then undid
            }
        }
        assert!(departed > 0, "expected some departures with max_steps=1");
        assert!(
            timeouts > 0,
            "expected some accepted-departure-then-undo transitions"
        );
    }

    #[test]
    fn bridge_preserves_strengths_and_fiber_invariants() {
        // Larger fiber: ME N=3 s=[2,2,2] sl=true, E=3, started on the
        // diagonal state.  Bridges must never leave the fiber or drift
        // the strengths, and some must succeed.
        let n = 3;
        let so = vec![2u64; 3];
        let si = vec![2u64; 3];
        let start = vec![((0, 0), 2), ((1, 1), 2), ((2, 2), 2)];
        let target = StrengthTarget::new(OccupationFamily::ME);
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };
        let config = BridgeConfig::default();
        let mut rng = StdRng::seed_from_u64(123);
        let mut successes = 0usize;
        for _ in 0..300 {
            let mut state = StrengthState::new(n, start.clone());
            let outcome = bridge_step(&mut state, &target, &domain, &mut rng, 3, &config);
            assert_eq!(state.occupied_count(), 3, "bridge left the fiber");
            assert_eq!(state.out_strengths, so);
            assert_eq!(state.in_strengths, si);
            state.debug_validate();
            if outcome.success {
                successes += 1;
            }
        }
        assert!(successes > 0, "expected some successful bridges");
    }
}
