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

use super::errors::FixedStrengthError;
use super::fixed_edges::validate_edge_target;
use super::problem::ResidualStrengthProblem;
use crate::OccNum;

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
}
