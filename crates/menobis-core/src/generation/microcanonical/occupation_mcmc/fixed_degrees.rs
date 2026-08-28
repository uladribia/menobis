//! Fixed-strength + fixed-degree (s, k) kernel support.
//!
//! Extends the finished fixed-(s,E) machinery with an exact directed
//! degree constraint: `k_out[i] = #{(i,·) : t > 0}` and
//! `k_in[j] = #{(·,j) : t > 0}` fixed to target vectors.
//!
//! # Structure (§2–§4 of the fixed-(s,k) plan)
//!
//! The final target is
//!
//! ```text
//! pi_(s,k)(t) ∝ ∏_ij d_F(t_ij)
//! ```
//!
//! over states with exact strengths and exact degree vectors.  The
//! degree vectors imply a unique edge count `E = Σ k_out`.
//! The central identity is
//!
//! ```text
//! pi_(s,k) = pi_(s,E)( · | k = k_target )
//! ```
//!
//! so the finished exact fixed-(s,E) kernel `K_E` from
//! [`super::fixed_edges`] is reused whole as the reversible proposal of
//! a degree-distance-biased auxiliary chain, and the production kernel
//! is the capped first-return trace of that auxiliary chain onto the
//! exact degree fiber `A_k = {x : D(x) = 0}` (§17).
//!
//! This module intentionally contains **only** fixed-degree-specific
//! code (validation, residualization, degree distance, the auxiliary
//! degree step and trace).  Family mathematics never lives here — the
//! family log-weight changes always flow through
//! [`super::target::StrengthTarget`] / `OccupationFamily`.

use super::domain::PairDomain;
use super::errors::FixedStrengthError;
use super::fixed_edges::{
    fixed_edge_step_recorded, validate_edge_target, BridgeConfig, FixedEdgeCounters,
};
use super::move_cycle::Cycle4Proposal;
use super::problem::ResidualStrengthProblem;
use super::state::StrengthState;
use super::target::StrengthTarget;
use crate::OccNum;
use rand::Rng;

/// Residual directed degree target after fixed-pair subtraction (§9).
///
/// `out`/`in_` are the residual degree vectors and `edge_count` their
/// common sum `E_residual = Σ k_out = Σ k_in`.
///
/// Internal type only: created once after fixed-pair residualization,
/// never exposed through the Python/boundary API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResidualDegreeTarget {
    pub out: Vec<u32>,
    pub in_: Vec<u32>,
    pub edge_count: usize,
}

/// Subtract the degree contribution of fixed pairs from full degree
/// vectors (§7.1, §7.2).
///
/// For every fixed pair with `occupation > 0`, subtract `1` from
/// `k_out[src]` and `1` from `k_in[tgt]`; zero-occupation fixed pairs
/// contribute nothing but still exclude their coordinate via the
/// strength residualization.
///
/// # Ordering (§7.2)
///
/// The caller must have already run
/// [`FixedStrengthProblem::into_residual`]
/// (super::problem::FixedStrengthProblem::into_residual) — which is the
/// authoritative duplicate/admissibility validation — and only then call
/// this with the stored fixed-pair list.  Duplicates must never reach
/// this function.
///
/// # Errors
///
/// - [`InvalidDegreeTarget`](FixedStrengthError::InvalidDegreeTarget)
///   if a positive fixed pair exceeds the corresponding degree target
///   (the degree of some node drops below zero), or if the residual
///   degree sums are unbalanced.
#[allow(dead_code)] // consumed by the fixed-(s,k) orchestrator (later phase)
pub(crate) fn residualize_degree_target(
    full_degree_out: &[u32],
    full_degree_in: &[u32],
    fixed_pairs: &[(u64, u64, OccNum)],
) -> Result<ResidualDegreeTarget, FixedStrengthError> {
    if full_degree_out.len() != full_degree_in.len() {
        return Err(FixedStrengthError::InvalidDegreeTarget(
            "degree_out and degree_in must have the same length".into(),
        ));
    }
    let mut out = full_degree_out.to_vec();
    let mut in_ = full_degree_in.to_vec();
    for &(src, tgt, occ) in fixed_pairs {
        if occ > 0 {
            let ko = &mut out[src as usize];
            if *ko == 0 {
                return Err(FixedStrengthError::InvalidDegreeTarget(format!(
                    "positive fixed pair ({src}, {tgt}) exceeds degree_out[{src}] (0)"
                )));
            }
            *ko -= 1;
            let ki = &mut in_[tgt as usize];
            if *ki == 0 {
                return Err(FixedStrengthError::InvalidDegreeTarget(format!(
                    "positive fixed pair ({src}, {tgt}) exceeds degree_in[{tgt}] (0)"
                )));
            }
            *ki -= 1;
        }
    }
    let edge_count_out: usize = out.iter().map(|&k| k as usize).sum();
    let edge_count_in: usize = in_.iter().map(|&k| k as usize).sum();
    if edge_count_out != edge_count_in {
        return Err(FixedStrengthError::InvalidDegreeTarget(format!(
            "residual degree sums are unbalanced: {edge_count_out} (out) != {edge_count_in} (in)"
        )));
    }
    Ok(ResidualDegreeTarget {
        out,
        in_,
        edge_count: edge_count_out,
    })
}

/// Validate a residual degree target against necessary feasibility
/// bounds (§10).  These are necessary conditions only, computed without
/// enumerating `N²` pairs; a target passing them may still be infeasible
/// for sparse structural-zero domains, in which case degree repair
/// exhausts with a structured repair error instead of this validator.
///
/// Checks (§10.1–§10.7):
///
/// 1. vector shapes match the residual node count;
/// 2. the out/in degree sums are equal (they define `E`);
/// 3. zero-compatibility: `s_out[i] == 0 ⇔ k_out[i] == 0` (and the
///    in-column analogue);
/// 4. every occupied pair carries at least one event: `k ≤ s`;
/// 5. B capacity: `s ≤ M·k` for each node (widened arithmetic);
/// 6. domain slot capacity per row/column via
///    [`PairDomain::admissible_degree_caps`];
/// 7. the existing edge-target bounds via [`validate_edge_target`]
///    (global edge/strength/capacity bounds).
#[allow(dead_code)] // consumed by the fixed-(s,k) orchestrator (later phase)
pub(crate) fn validate_degree_target(
    residual: &ResidualStrengthProblem,
    degree: &ResidualDegreeTarget,
) -> Result<(), FixedStrengthError> {
    let n = residual.strength_out.len();
    let err = |msg: String| Err(FixedStrengthError::InvalidDegreeTarget(msg));

    // ---- 1. Shape (§10.1) ----
    if degree.out.len() != n || degree.in_.len() != n {
        return err(format!(
            "degree vectors must have length {n}, got out={} in={}",
            degree.out.len(),
            degree.in_.len()
        ));
    }

    // ---- 2. Sums (§10.2) ----
    // residualize_degree_target already balances the sums; re-verify
    // defensively so direct construction cannot bypass it.
    let sum_out: u64 = degree.out.iter().map(|&k| k as u64).sum();
    let sum_in: u64 = degree.in_.iter().map(|&k| k as u64).sum();
    if sum_out != sum_in {
        return err(format!(
            "degree sums are unbalanced ({sum_out} out vs {sum_in} in)"
        ));
    }
    let e = sum_out as usize;
    if e != degree.edge_count {
        return err(format!(
            "edge_count field ({}) does not match sum of out-degrees ({e})",
            degree.edge_count
        ));
    }

    // ---- 3. Zero compatibility (§10.3) ----
    for i in 0..n {
        if (residual.strength_out[i] == 0) != (degree.out[i] == 0) {
            return err(format!(
                "node {i}: strength_out {} and degree_out {} must both be zero or both positive",
                residual.strength_out[i], degree.out[i]
            ));
        }
        if (residual.strength_in[i] == 0) != (degree.in_[i] == 0) {
            return err(format!(
                "node {i}: strength_in {} and degree_in {} must both be zero or both positive",
                residual.strength_in[i], degree.in_[i]
            ));
        }
    }

    // ---- 4. One event per occupied pair (§10.4) ----
    for i in 0..n {
        if (degree.out[i] as OccNum) > residual.strength_out[i] {
            return err(format!(
                "node {i}: degree_out {} exceeds strength_out {}",
                degree.out[i], residual.strength_out[i]
            ));
        }
        if (degree.in_[i] as OccNum) > residual.strength_in[i] {
            return err(format!(
                "node {i}: degree_in {} exceeds strength_in {}",
                degree.in_[i], residual.strength_in[i]
            ));
        }
    }

    // ---- 5. B capacity (§10.5) ----
    if let crate::model::family::OccupationFamily::B { layers } = residual.family {
        let m = layers as u64;
        for i in 0..n {
            if (residual.strength_out[i] as u128) > m as u128 * degree.out[i] as u128 {
                return err(format!(
                    "node {i}: strength_out {} exceeds B capacity M={layers} * degree_out {}",
                    residual.strength_out[i], degree.out[i]
                ));
            }
            if (residual.strength_in[i] as u128) > m as u128 * degree.in_[i] as u128 {
                return err(format!(
                    "node {i}: strength_in {} exceeds B capacity M={layers} * degree_in {}",
                    residual.strength_in[i], degree.in_[i]
                ));
            }
        }
    }

    // ---- 6. Domain slot capacity (§10.6) ----
    let (out_caps, in_caps) = residual.domain.admissible_degree_caps();
    for i in 0..n {
        if (degree.out[i] as usize) > out_caps[i] {
            return err(format!(
                "node {i}: degree_out {} exceeds admissible row slots {}",
                degree.out[i], out_caps[i]
            ));
        }
        if (degree.in_[i] as usize) > in_caps[i] {
            return err(format!(
                "node {i}: degree_in {} exceeds admissible column slots {}",
                degree.in_[i], in_caps[i]
            ));
        }
    }

    // ---- 7. Reuse global edge-target bounds (§10.7) ----
    validate_edge_target(residual, e)?;

    Ok(())
}

/// Raw degree distance of the current state to the residual degree
/// target (§12):
///
/// ```text
/// D_raw = Σ_i |k_out[i] − target_out[i]| + Σ_j |k_in[j] − target_in[j]|
/// ```
///
/// Every endpoint of the fixed-(s,E) kernel has the same total `E` as
/// the target, so `D_raw` is even; callers that need a half-normalized
/// distance use [`degree_distance`].
///
/// O(N) scan — used only when repair begins and in debug/test
/// verification, never in the auxiliary hot path.
#[allow(dead_code)] // consumed by the degree repair/trace phases
pub(crate) fn degree_distance_raw(
    state_out: &[usize],
    state_in: &[usize],
    target: &ResidualDegreeTarget,
) -> u64 {
    let mut d = 0u64;
    for i in 0..target.out.len() {
        d += (state_out[i] as u64).abs_diff(target.out[i] as u64);
        d += (state_in[i] as u64).abs_diff(target.in_[i] as u64);
    }
    d
}

/// Half-normalized degree distance `D = D_raw / 2` (§12).
#[allow(dead_code)] // consumed by the degree repair/trace phases
pub(crate) fn degree_distance(
    state_out: &[usize],
    state_in: &[usize],
    target: &ResidualDegreeTarget,
) -> u64 {
    degree_distance_raw(state_out, state_in, target) / 2
}

/// O(1) raw L1 degree-distance delta of one 4-cycle proposal (§11,
/// §16.4).///
/// The cycle touches rows `a` and `c` (out-degrees) and columns `b` and
/// `d` (in-degrees); because `a != c` and `b != d` no term is duplicated.
/// The per-cell before/after counts are cached in
/// [`Cycle4Proposal`] (super::move_cycle::Cycle4Proposal), so the
/// result is
///
/// ```text
/// |out_a_after − k*[a]| − |out_a_before − k*[a]|
/// + |out_c_after − k*[c]| − |out_c_before − k*[c]|
/// + |in_b_after  − k*[b]| − |in_b_before  − k*[b]|
/// + |in_d_after  − k*[d]| − |in_d_before  − k*[d]|
/// ```
///
/// For a sequence of proposals these deltas telescope to the full
/// endpoint change; the auxiliary hot path never scans all `N` nodes.
#[allow(dead_code)] // consumed by the degree repair/trace phases
pub(crate) fn proposal_degree_delta_l1(
    proposal: &super::move_cycle::Cycle4Proposal,
    target_out: &[u32],
    target_in: &[u32],
) -> i64 {
    let pop = |before: usize, after: usize, tgt: &[u32], node: usize| {
        let t = tgt[node] as i64;
        (after as i64 - t).abs() - (before as i64 - t).abs()
    };
    pop(
        proposal.out_a_before,
        proposal.out_a_after,
        target_out,
        proposal.a as usize,
    ) + pop(
        proposal.out_c_before,
        proposal.out_c_after,
        target_out,
        proposal.c as usize,
    ) + pop(
        proposal.in_b_before,
        proposal.in_b_after,
        target_in,
        proposal.b as usize,
    ) + pop(
        proposal.in_d_before,
        proposal.in_d_after,
        target_in,
        proposal.d as usize,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::microcanonical::occupation_mcmc::domain::PairDomain;
    use crate::generation::microcanonical::occupation_mcmc::problem::FixedStrengthProblem;
    use crate::model::family::OccupationFamily;
    use std::collections::HashSet;

    fn complete_domain(n: usize, sl: bool) -> PairDomain {
        PairDomain::Complete {
            node_count: n,
            self_loops: sl,
        }
    }

    fn residual(
        family: OccupationFamily,
        so: Vec<OccNum>,
        si: Vec<OccNum>,
        sl: bool,
        fixed: Vec<(u64, u64, OccNum)>,
    ) -> ResidualStrengthProblem {
        FixedStrengthProblem::new(family, so, si.clone(), complete_domain(si.len(), sl), fixed)
            .unwrap()
            .into_residual()
            .unwrap()
    }

    fn degree(out: Vec<u32>, in_: Vec<u32>) -> ResidualDegreeTarget {
        let edge_count: usize = out.iter().map(|&k| k as usize).sum();
        ResidualDegreeTarget {
            out,
            in_,
            edge_count,
        }
    }

    // -------------------------------------------------------------------
    // §7: fixed-pair degree residualization
    // -------------------------------------------------------------------

    #[test]
    fn positive_fixed_pair_subtracts_from_degrees() {
        // Full table: s=[3,3]/[3,3] with (0,0)=2,(0,1)=1,(1,0)=1,(1,1)=2 is
        // feasible (E=4, k_out=[2,2], k_in=[2,2]).  Freeze (0,1) at its
        // observed occupation 1: residual strengths [2,3]/[3,2], residual
        // degrees k_out=[1,2], k_in=[1,2], E_res=3.
        let prob = residual(
            OccupationFamily::ME,
            vec![3, 3],
            vec![3, 3],
            true,
            vec![(0, 1, 1)],
        );
        assert_eq!(prob.strength_out, vec![2, 3]);
        assert_eq!(prob.strength_in, vec![3, 2]);
        assert!(!prob.domain.is_admissible(0, 1));

        let deg = residualize_degree_target(&[2, 2], &[2, 2], &[(0, 1, 1)]).unwrap();
        // Free pairs are {(0,0),(1,0),(1,1)}: k_out = [1,2], k_in = [2,1].
        assert_eq!(deg.out, vec![1, 2]);
        assert_eq!(deg.in_, vec![2, 1]);
        assert_eq!(deg.edge_count, 3);
        validate_degree_target(&prob, &deg).unwrap();
    }

    #[test]
    fn zero_fixed_pair_leaves_degrees_unchanged() {
        // Zero fixed pair (0,1,0) freezes an absent coordinate: strengths
        // and degrees unchanged, coordinate excluded from the domain.
        let prob = residual(
            OccupationFamily::ME,
            vec![2, 2],
            vec![2, 2],
            true,
            vec![(0, 1, 0)],
        );
        assert_eq!(prob.strength_out, vec![2, 2]);
        assert_eq!(prob.strength_in, vec![2, 2]);
        assert!(!prob.domain.is_admissible(0, 1));

        let deg = residualize_degree_target(&[1, 1], &[1, 1], &[(0, 1, 0)]).unwrap();
        assert_eq!(deg.out, vec![1, 1]);
        assert_eq!(deg.in_, vec![1, 1]);
        assert_eq!(deg.edge_count, 2);
        validate_degree_target(&prob, &deg).unwrap();
    }

    #[test]
    fn duplicate_fixed_pair_rejected_by_strength_residualization() {
        // Duplicates must be rejected by into_residual before any degree
        // double-subtraction (§7.2 ordering).
        let prob = FixedStrengthProblem::new(
            OccupationFamily::ME,
            vec![3, 3],
            vec![3, 3],
            complete_domain(2, true),
            vec![(0, 1, 1), (0, 1, 1)],
        )
        .unwrap();
        match prob.into_residual() {
            Err(FixedStrengthError::InvalidResidual(msg)) => {
                assert!(msg.contains("duplicate"), "{msg}")
            }
            other => panic!("expected duplicate error, got {other:?}"),
        }
    }

    #[test]
    fn positive_fixed_pair_exceeding_degree_target() {
        // k_out[0] == 0 but a positive fixed pair (0,1) would make it
        // negative — the degree residualization must reject it.
        match residualize_degree_target(&[0, 2], &[2, 2], &[(0, 1, 1)]) {
            Err(FixedStrengthError::InvalidDegreeTarget(msg)) => {
                assert!(msg.contains("exceeds degree_out"), "{msg}")
            }
            other => panic!("expected InvalidDegreeTarget, got {other:?}"),
        }
    }

    #[test]
    fn unbalanced_residual_degrees_rejected() {
        match residualize_degree_target(&[2, 1], &[1, 1], &[]) {
            Err(FixedStrengthError::InvalidDegreeTarget(msg)) => {
                assert!(msg.contains("unbalanced"), "{msg}")
            }
            other => panic!("expected InvalidDegreeTarget, got {other:?}"),
        }
    }

    // -------------------------------------------------------------------
    // §10: validation rules
    // -------------------------------------------------------------------

    #[test]
    fn validate_degree_shape() {
        let prob = residual(OccupationFamily::ME, vec![3, 3], vec![3, 3], true, vec![]);
        let bad = degree(vec![1, 1, 0], vec![2, 0, 0]); // 3 nodes vs n=2
        assert!(matches!(
            validate_degree_target(&prob, &bad),
            Err(FixedStrengthError::InvalidDegreeTarget(msg)) if msg.contains("length")
        ));
    }

    #[test]
    fn validate_zero_compatibility() {
        let prob = residual(OccupationFamily::ME, vec![2, 0], vec![0, 2], true, vec![]);
        // strength_out[1] == 0 but degree_out[1] == 1: inconsistent.
        let bad = degree(vec![1, 1], vec![0, 2]);
        assert!(matches!(
            validate_degree_target(&prob, &bad),
            Err(FixedStrengthError::InvalidDegreeTarget(msg)) if msg.contains("both be zero")
        ));
        // Consistent zero nodes.
        let good = degree(vec![2, 0], vec![0, 2]);
        validate_degree_target(&prob, &good).unwrap();
    }

    #[test]
    fn validate_degree_greater_than_strength() {
        let prob = residual(OccupationFamily::ME, vec![2, 2], vec![2, 2], true, vec![]);
        // k_out[0] = 3 > s_out[0] = 2.
        let bad = degree(vec![3, 1], vec![2, 2]);
        assert!(matches!(
            validate_degree_target(&prob, &bad),
            Err(FixedStrengthError::InvalidDegreeTarget(msg)) if msg.contains("exceeds strength_out")
        ));
    }

    #[test]
    fn validate_b_capacity() {
        // B M=2: node 0 with s_out=5 and k_out=2 violates 5 <= 2*2=4.
        let prob = residual(
            OccupationFamily::B { layers: 2 },
            vec![5, 5, 5],
            vec![5, 5, 5],
            true,
            vec![],
        );
        let bad = degree(vec![2, 3, 3], vec![3, 2, 3]);
        assert!(matches!(
            validate_degree_target(&prob, &bad),
            Err(FixedStrengthError::InvalidDegreeTarget(msg)) if msg.contains("B capacity")
        ));
        // Feasible: k_out=[3,3,3], k_in=[3,3,3], E=9 = A with 5 <= 2*3.
        let good = degree(vec![3, 3, 3], vec![3, 3, 3]);
        validate_degree_target(&prob, &good).unwrap();
    }

    #[test]
    fn validate_domain_slot_capacity() {
        // CompleteMinus excluding two coordinates of column 1: column 1 has
        // only 1 admissible slot but k_in[1]=2 is requested — validated
        // without N^2 enumeration (§10.6).  The degree vectors are balanced
        // (sum 6 both sides) and pass every earlier check so the column-cap
        // rule is what binds.
        let excluded = HashSet::from([(0, 1), (1, 1)]);
        let domain = PairDomain::CompleteMinus {
            node_count: 3,
            self_loops: true,
            excluded,
        };
        let prob = FixedStrengthProblem::new(
            OccupationFamily::ME,
            vec![3, 3, 3],
            vec![1, 2, 6],
            domain,
            vec![],
        )
        .unwrap()
        .into_residual()
        .unwrap();
        // in_caps = [3, 1, 3]; k_in = [1, 2, 3] with k_in[1] > 1.
        let bad = degree(vec![2, 2, 2], vec![1, 2, 3]);
        assert!(matches!(
            validate_degree_target(&prob, &bad),
            Err(FixedStrengthError::InvalidDegreeTarget(msg)) if msg.contains("admissible column slots")
        ));
    }

    #[test]
    fn validate_degree_target_edge_call_through() {
        // §10.7 reuses validate_edge_target.  Every bound it enforces is
        // implied by the earlier §10.3–§10.6 checks for consistent degree
        // vectors (zero-compat + one-event imply E >= positive rows;
        // per-node B capacity implies E >= ceil(T/M); caps imply E <= A;
        // k <= s per node implies E <= T).  This test proves the call
        // through is not an error by passing the maximal dense E=A case.
        let prob = residual(OccupationFamily::ME, vec![3, 3], vec![3, 3], true, vec![]);
        let dense = degree(vec![2, 2], vec![2, 2]); // E=4 = A
        validate_degree_target(&prob, &dense).unwrap();
        assert_eq!(prob.domain.admissible_pair_count(), 4);
    }

    #[test]
    fn degree_distance_definition() {
        let target = degree(vec![2, 1], vec![1, 2]);
        let out = vec![1usize, 2];
        let inp = vec![2usize, 1];
        // D_raw = |1-2|+|2-1|+|2-1|+|1-2| = 4; D = 2.
        assert_eq!(degree_distance_raw(&out, &inp, &target), 4);
        assert_eq!(degree_distance(&out, &inp, &target), 2);
        // Exact match -> 0.
        assert_eq!(degree_distance_raw(&[2, 1], &[1, 2], &target), 0);
    }

    // -------------------------------------------------------------------
    // §37: cached proposal degree deltas vs independent recomputation
    // -------------------------------------------------------------------

    /// Independent recomputation of `(k_out, k_in)` from the sparse state.
    fn recompute_degrees(
        state: &crate::generation::microcanonical::occupation_mcmc::state::StrengthState,
    ) -> (Vec<usize>, Vec<usize>) {
        let n = state.node_count;
        let mut out = vec![0usize; n];
        let mut inp = vec![0usize; n];
        for &(s, t) in state.occupied_pairs() {
            out[s as usize] += 1;
            inp[t as usize] += 1;
        }
        (out, inp)
    }

    /// Draw a proposal from `state`, apply it, and verify:
    ///  1. the cached before/after counts equal an independent scan;
    ///  2. the cached O(1) delta equals an independent full-scan delta.
    fn assert_proposal_metadata_matches_scan(
        state: &mut crate::generation::microcanonical::occupation_mcmc::state::StrengthState,
    ) {
        use crate::generation::microcanonical::occupation_mcmc::move_cycle::draw_cycle4_proposal;
        use crate::generation::microcanonical::occupation_mcmc::target::StrengthTarget;
        use crate::model::family::OccupationFamily;
        use rand::SeedableRng;
        let target = StrengthTarget::new(OccupationFamily::ME);
        let n = state.node_count;
        // Arbitrary but feasible-looking targets sized to the state.
        let target_out = vec![2u32; n];
        let target_in = vec![2u32; n];
        let degree_target = ResidualDegreeTarget {
            out: target_out.clone(),
            in_: target_in.clone(),
            edge_count: 2 * n,
        };
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };
        let Some(proposal) = draw_cycle4_proposal(
            state,
            &target,
            &domain,
            &mut rand::rngs::StdRng::seed_from_u64(1),
        ) else {
            return;
        };

        let (before_out, before_in) = recompute_degrees(state);
        assert_eq!(
            proposal.out_a_before, before_out[proposal.a as usize],
            "out_a_before mismatch"
        );
        assert_eq!(
            proposal.out_c_before, before_out[proposal.c as usize],
            "out_c_before mismatch"
        );
        assert_eq!(
            proposal.in_b_before, before_in[proposal.b as usize],
            "in_b_before mismatch"
        );
        assert_eq!(
            proposal.in_d_before, before_in[proposal.d as usize],
            "in_d_before mismatch"
        );

        proposal.apply(state);
        let (after_out, after_in) = recompute_degrees(state);
        assert_eq!(
            proposal.out_a_after, after_out[proposal.a as usize],
            "out_a_after mismatch"
        );
        assert_eq!(
            proposal.out_c_after, after_out[proposal.c as usize],
            "out_c_after mismatch"
        );
        assert_eq!(
            proposal.in_b_after, after_in[proposal.b as usize],
            "in_b_after mismatch"
        );
        assert_eq!(
            proposal.in_d_after, after_in[proposal.d as usize],
            "in_d_after mismatch"
        );

        // Independent full-scan raw delta.
        let delta_scan = degree_distance_raw(&after_out, &after_in, &degree_target) as i64
            - degree_distance_raw(&before_out, &before_in, &degree_target) as i64;
        assert_eq!(
            proposal_degree_delta_l1(&proposal, &target_out, &target_in),
            delta_scan,
            "cached delta must equal independent scan"
        );

        // Undo must restore the exact degree vectors (round-trip).
        proposal.undo(state);
        let (restored_out, restored_in) = recompute_degrees(state);
        assert_eq!(restored_out, before_out);
        assert_eq!(restored_in, before_in);
    }

    #[test]
    fn proposal_degree_metadata_matches_scan_across_patterns() {
        use crate::generation::microcanonical::occupation_mcmc::state::StrengthState;

        // Pattern A — no support change: all four cells stay occupied.
        //       (0,0)=2  (1,1)=2  (0,1)=1  (1,0)=1
        let state_a =
            StrengthState::new(2, vec![((0, 0), 2), ((1, 1), 2), ((0, 1), 1), ((1, 0), 1)]);
        // Pattern B — two decrements disappear, two cross cells appear:
        //       (0,0)=1  (1,1)=1  and (0,1),(1,0) absent.
        let state_b = StrengthState::new(2, vec![((0, 0), 1), ((1, 1), 1)]);
        // Pattern C — one decrement disappears: (0,0)=1 leaves the support.
        let state_c =
            StrengthState::new(2, vec![((0, 0), 1), ((1, 1), 2), ((0, 1), 1), ((1, 0), 1)]);
        // Pattern D — one cross cell enters: (0,1) appears while others stay.
        let state_d = StrengthState::new(2, vec![((0, 0), 2), ((1, 1), 2), ((1, 0), 1)]);
        // Mixed dense state.
        let state_e = StrengthState::new(
            3,
            vec![
                ((0, 0), 3),
                ((1, 1), 2),
                ((2, 2), 3),
                ((0, 2), 1),
                ((2, 0), 1),
            ],
        );

        for mut state in [state_a, state_b, state_c, state_d, state_e] {
            for _ in 0..200 {
                assert_proposal_metadata_matches_scan(&mut state);
                state.debug_validate();
            }
        }
    }

    #[test]
    fn proposal_delta_telescopes_over_sequences() {
        use crate::generation::microcanonical::occupation_mcmc::move_cycle::draw_cycle4_proposal;
        use crate::generation::microcanonical::occupation_mcmc::state::StrengthState;
        use crate::generation::microcanonical::occupation_mcmc::target::StrengthTarget;
        use crate::model::family::OccupationFamily;
        use rand::SeedableRng;

        let mut state =
            StrengthState::new(2, vec![((0, 0), 2), ((1, 1), 2), ((0, 1), 1), ((1, 0), 1)]);
        let target = StrengthTarget::new(OccupationFamily::ME);
        let domain = PairDomain::Complete {
            node_count: 2,
            self_loops: true,
        };
        let degree_target = degree(vec![2, 2], vec![2, 2]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);

        let mut applied: Vec<
            crate::generation::microcanonical::occupation_mcmc::move_cycle::Cycle4Proposal,
        > = Vec::new();
        let (start_out, start_in) = recompute_degrees(&state);
        let start_raw = degree_distance_raw(&start_out, &start_in, &degree_target);
        let mut running_raw = start_raw as i64;

        for _ in 0..50 {
            let Some(p) = draw_cycle4_proposal(&state, &target, &domain, &mut rng) else {
                continue;
            };
            running_raw += proposal_degree_delta_l1(&p, &degree_target.out, &degree_target.in_);
            p.apply(&mut state);
            applied.push(p);
        }

        // Summed cached deltas must equal the full-scan endpoint change.
        let (end_out, end_in) = recompute_degrees(&state);
        let end_raw = degree_distance_raw(&end_out, &end_in, &degree_target) as i64;
        assert_eq!(
            running_raw, end_raw,
            "deltas must telescope to the endpoint change"
        );

        // Undo in reverse: the running scalar must return to the origin.
        for p in applied.iter().rev() {
            running_raw -= proposal_degree_delta_l1(p, &degree_target.out, &degree_target.in_);
            p.undo(&mut state);
        }
        let (restored_out, restored_in) = recompute_degrees(&state);
        assert_eq!(
            degree_distance_raw(&restored_out, &restored_in, &degree_target) as i64,
            running_raw
        );
        assert_eq!(running_raw, start_raw as i64);
    }
}

// ---------------------------------------------------------------------------
// Phase 6: degree-biased auxiliary step (§16, §24)
// ---------------------------------------------------------------------------

/// One degree-biased auxiliary substep (§16): propose one **complete**
/// production fixed-(s,E) transition `K_E` (recorded variant) and apply
/// the outer degree-potential MH decision
///
/// ```text
/// alpha(x,y) = min(1, exp(−λ·(D(y)−D(x))))
/// ```
///
/// where `D` is the half-normalized degree distance (§12).  Because `K_E`
/// is reversible for `pi_(s,E)`, every internal factor (family
/// degeneracy, occupied-cell Hastings q, fixed-E bridge path, internal
/// edge bias) cancels at this outer level — never recomputed here.
///
/// The caller maintains:
/// - `record`: the flat undo log (may already hold the excursion
///   prefix); `record[start..]` appended by this call is exactly the
///   deterministic undo log of this one `K_E` transition;
/// - `running_raw`: the current raw `D_raw(x)` (even), updated by the
///   telescoped O(1) per-cycle deltas — never an O(N) scan (§16.4).
///
/// Returns `true` if the endpoint is accepted (state changed, `running_raw`
/// advanced); `false` if the outer MH rejects (the recorded transition is
/// undone in reverse, the log truncated, `running_raw` unchanged).
///
/// Both degree repair (§15) and the production trace (§17) use this exact
/// primitive — one stochastic policy, two stopping rules (§24).
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // consumed by Phase 7 repair / Phase 8 trace
pub(crate) fn degree_auxiliary_step(
    state: &mut StrengthState,
    target: &StrengthTarget,
    domain: &PairDomain,
    rng: &mut impl Rng,
    edge_target: usize,
    bridge_config: &BridgeConfig,
    counters: &mut FixedEdgeCounters,
    degree_target: &ResidualDegreeTarget,
    lambda: f64,
    record: &mut Vec<Cycle4Proposal>,
    running_raw: &mut u64,
) -> bool {
    let start = record.len();
    fixed_edge_step_recorded(
        state,
        target,
        domain,
        rng,
        edge_target,
        bridge_config,
        counters,
        record,
    );
    let delta_t: i64 = record[start..]
        .iter()
        .map(|p| proposal_degree_delta_l1(p, &degree_target.out, &degree_target.in_))
        .sum();
    if delta_t > 0 {
        // alpha = exp(−λ·ΔT/2) < 1; consume one uniform draw only when
        // the acceptance is proper (consistent with `metropolis_accept`).
        let log_alpha = -lambda * delta_t as f64 / 2.0;
        let log_u = (rng.random::<f64>() + f64::MIN_POSITIVE).ln();
        if log_u >= log_alpha {
            for p in record[start..].iter().rev() {
                p.undo(state);
            }
            record.truncate(start);
            return false;
        }
    }
    *running_raw = ((*running_raw) as i64 + delta_t) as u64;
    true
}

#[cfg(test)]
mod phase6_tests {
    use super::*;
    use crate::model::family::OccupationFamily;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn me_target(deg: &ResidualDegreeTarget) -> (StrengthTarget<'static>, PairDomain) {
        let target = StrengthTarget::new(OccupationFamily::ME);
        let domain = PairDomain::Complete {
            node_count: deg.out.len(),
            self_loops: true,
        };
        (target, domain)
    }

    #[allow(clippy::type_complexity)]
    fn full_snapshot(state: &StrengthState) -> (Vec<((u64, u64), OccNum)>, Vec<usize>, Vec<usize>) {
        let mut pairs: Vec<_> = state.iter_occupied().collect();
        pairs.sort_unstable();
        (
            pairs,
            state.row_occ_count.clone(),
            state.col_occ_count.clone(),
        )
    }

    #[test]
    fn auxiliary_step_lambda_zero_accepts_every_endpoint() {
        // λ = 0 (§39): the outer degree potential never rejects; every
        // recorded K_E endpoint is applied and the running raw distance
        // must track the independently recomputed D_raw after every step.
        // Strengths are exact invariants of the state (never a target).
        let n = 3;
        let so = vec![3u64, 3, 0];
        let si = vec![2u64, 2, 2];
        let deg = ResidualDegreeTarget {
            out: vec![2, 1, 1],
            in_: vec![1, 1, 2],
            edge_count: 4,
        };
        let (target, domain) = me_target(&deg);
        let mut rng = StdRng::seed_from_u64(11);
        let mut state =
            StrengthState::new(n, vec![((0, 0), 2), ((1, 1), 2), ((0, 2), 1), ((1, 2), 1)]);
        let mut running = degree_distance_raw(&state.row_occ_count, &state.col_occ_count, &deg);
        let mut counters = FixedEdgeCounters::default();
        let mut record = Vec::new();
        for _ in 0..500 {
            let accepted = degree_auxiliary_step(
                &mut state,
                &target,
                &domain,
                &mut rng,
                deg.edge_count,
                &BridgeConfig::default(),
                &mut counters,
                &deg,
                0.0,
                &mut record,
                &mut running,
            );
            assert!(accepted, "λ=0 must accept every endpoint");
            let scan = degree_distance_raw(&state.row_occ_count, &state.col_occ_count, &deg);
            assert_eq!(running, scan, "running raw must equal independent scan");
            assert_eq!(state.occupied_count(), deg.edge_count);
            assert_eq!(state.out_strengths, so);
            assert_eq!(state.in_strengths, si);
            state.debug_validate();
            record.clear();
        }
    }

    #[test]
    fn auxiliary_step_rejection_restores_exact_state() {
        // When the outer MH rejects, the recorded K_E transition must be
        // undone deterministically: exact occupied pairs, strengths,
        // degree caches, E, and the running distance.
        let n = 3;
        let so = vec![3u64, 4, 1];
        let si = vec![1u64, 3, 4];
        let deg = ResidualDegreeTarget {
            out: vec![1, 2, 2],
            in_: vec![2, 2, 1],
            edge_count: 5,
        };
        let (target, domain) = me_target(&deg);
        let mut rng = StdRng::seed_from_u64(5);
        let mut state = StrengthState::new(
            n,
            vec![
                ((0, 0), 1),
                ((0, 1), 2),
                ((1, 1), 1),
                ((1, 2), 3),
                ((2, 2), 1),
            ],
        );
        let mut running = degree_distance_raw(&state.row_occ_count, &state.col_occ_count, &deg);
        let mut counters = FixedEdgeCounters::default();
        let mut record = Vec::new();
        let mut rejections = 0usize;
        for _ in 0..2000 {
            let origin = full_snapshot(&state);
            let origin_running = running;
            let start = record.len();
            let accepted = degree_auxiliary_step(
                &mut state,
                &target,
                &domain,
                &mut rng,
                deg.edge_count,
                &BridgeConfig::default(),
                &mut counters,
                &deg,
                1.0,
                &mut record,
                &mut running,
            );
            if accepted {
                let scan = degree_distance_raw(&state.row_occ_count, &state.col_occ_count, &deg);
                assert_eq!(running, scan, "accepted: running raw must equal scan");
                if record.len() > start {
                    let _ = true; // genuine K_E endpoint move
                } else {
                    // Accepted K_E self-loop (§16.3): nothing applied, so
                    // the state must be the exact origin.
                    assert_eq!(full_snapshot(&state), origin);
                }
                record.clear();
            } else {
                rejections += 1;
                assert_eq!(
                    full_snapshot(&state),
                    origin,
                    "rejection must restore the exact pre-K_E state"
                );
                assert_eq!(
                    running, origin_running,
                    "rejection must not touch running raw"
                );
                assert_eq!(record.len(), start, "rejection must truncate the record");
            }
            assert_eq!(state.occupied_count(), deg.edge_count);
            assert_eq!(state.out_strengths, so);
            assert_eq!(state.in_strengths, si);
            state.debug_validate();
        }
        assert!(rejections > 0, "expected some degree-MH rejections");
    }

    #[test]
    fn auxiliary_step_deterministic_by_seed() {
        let deg = ResidualDegreeTarget {
            out: vec![1, 1, 1],
            in_: vec![1, 1, 1],
            edge_count: 3,
        };
        let (target, domain) = me_target(&deg);
        let run = |seed: u64| -> Vec<OccNum> {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut state = StrengthState::new(3, vec![((0, 0), 1), ((1, 1), 1), ((2, 2), 1)]);
            let mut running = degree_distance_raw(&state.row_occ_count, &state.col_occ_count, &deg);
            let mut counters = FixedEdgeCounters::default();
            let mut record = Vec::new();
            for _ in 0..300 {
                degree_auxiliary_step(
                    &mut state,
                    &target,
                    &domain,
                    &mut rng,
                    deg.edge_count,
                    &BridgeConfig::default(),
                    &mut counters,
                    &deg,
                    1.0,
                    &mut record,
                    &mut running,
                );
                record.clear();
            }
            let mut pairs: Vec<_> = state.iter_occupied().collect();
            pairs.sort_unstable();
            pairs
                .into_iter()
                .flat_map(|((s, t), o)| vec![s, t, o])
                .collect()
        };
        assert_eq!(run(3), run(3), "same seed must reproduce the walk");
    }

    /// Exact auxiliary row on the ME N=2 s=[3,3]/[3,3] loops fiber
    /// (§39 independent exact math): the E=4 fiber has exactly two states
    /// with degree (2,2)/(2,2),
    ///
    /// ```text
    /// s1 = {(0,0):1,(0,1):2,(1,0):2,(1,1):1}   a=1
    /// s2 = {(0,0):2,(0,1):1,(1,0):1,(1,1):2}   a=2
    /// ```
    ///
    /// Exact law: from s1 the only 4-cycle proposals are (0,0),(1,1)
    /// -> s0 and (0,1),(1,0) -> s2, each with q_fwd = (1/v+1/v)/4 = 0.5
    /// (v_ab = 4−row−col+1 = 1 for every involved P1).  E=4 local kernel
    /// vetoes the s0 move: local[s1->s2] = 0.5.  The bridge from s1
    /// departs only through the aux move to s0 (mass ≈ 0.015) and every
    /// such departure returns to s1 (s0's only move is back), so the
    /// bridge is a pure self-loop.  Hence
    /// K[s1][s2] = 0.95·0.5 and K[s1][s1] = 0.95·0.5 + 0.05.
    fn expected_aux_row_s1() -> (f64, f64) {
        let k12 = 0.95 * 0.5; // 0.475
        let k11 = 0.95 * 0.5 + 0.05; // 0.525
        (k11, k12)
    }

    #[test]
    fn auxiliary_step_matches_exact_q_row() {
        // Production one-step endpoint frequencies from the same origin
        // must match the independent exact Q row (§39 / §22.1).
        let (p_self, p_to_s2) = expected_aux_row_s1();
        let deg = ResidualDegreeTarget {
            out: vec![2, 2],
            in_: vec![2, 2],
            edge_count: 4,
        };
        let (target, domain) = me_target(&deg);
        let s1 = || StrengthState::new(2, vec![((0, 0), 1), ((0, 1), 2), ((1, 0), 2), ((1, 1), 1)]);
        let s2_pairs = || -> Vec<((u64, u64), OccNum)> {
            let mut p: Vec<_> =
                StrengthState::new(2, vec![((0, 0), 2), ((0, 1), 1), ((1, 0), 1), ((1, 1), 2)])
                    .iter_occupied()
                    .collect();
            p.sort_unstable();
            p
        };

        let trials = 200_000usize;
        let mut rng = StdRng::seed_from_u64(2026);
        let mut counters = FixedEdgeCounters::default();
        let mut record = Vec::new();
        let mut landed_s2 = 0usize;
        for _ in 0..trials {
            let mut state = s1();
            let mut running = degree_distance_raw(&state.row_occ_count, &state.col_occ_count, &deg);
            let start = record.len();
            let accepted = degree_auxiliary_step(
                &mut state,
                &target,
                &domain,
                &mut rng,
                deg.edge_count,
                &BridgeConfig::default(),
                &mut counters,
                &deg,
                1.0,
                &mut record,
                &mut running,
            );
            assert!(accepted, "in-fiber ΔD=0 endpoint must always be accepted");
            record.truncate(start);
            let mut pairs: Vec<_> = state.iter_occupied().collect();
            pairs.sort_unstable();
            if pairs == s2_pairs() {
                landed_s2 += 1;
            } else {
                assert_eq!(pairs.len(), 4, "unexpected endpoint");
            }
        }
        let empirical = landed_s2 as f64 / trials as f64;
        assert!(
            (empirical - p_to_s2).abs() < 0.01,
            "empirical P(->s2) {empirical:.4} vs exact {p_to_s2:.4}"
        );
        assert!(
            (1.0 - empirical - p_self).abs() < 0.01,
            "empirical P(->s1) {} vs exact {p_self:.4}",
            1.0 - empirical
        );
    }
}
