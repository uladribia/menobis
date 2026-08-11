//! Validate production repair heuristics against the Dinic max-flow oracle.
//!
//! For each repair type (loop, B capacity, admissibility), generate small
//! feasible problems, run the oracle to confirm feasibility, run the
//! production path, and verify the repair succeeds.

use std::collections::HashSet;

use menobis_core::generation::microcanonical::occupation_mcmc::domain::PairDomain;
use menobis_core::generation::microcanonical::occupation_mcmc::repair::{
    loopless_feasibility_check, repair_capacity, repair_inadmissible_pairs, repair_self_loops,
    RepairConfig,
};
use menobis_core::generation::microcanonical::occupation_mcmc::state::StrengthState;
use menobis_core::model::family::OccupationFamily;
use menobis_core::OccNum;
use menobis_test_oracles::legacy_max_flow::feasibility_max_flow;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Build a complete-list admissible pair set: all N×N pairs optionally excluding self-loops.
fn complete_admissible(n: usize, self_loops: bool) -> Vec<(u64, u64)> {
    (0..n as u64)
        .flat_map(|i| (0..n as u64).map(move |j| (i, j)))
        .filter(|&(i, j)| self_loops || i != j)
        .collect()
}

// ══════════════════════════════════════════════════════════════════════
// Loop repair
// ══════════════════════════════════════════════════════════════════════

#[test]
fn repair_self_loops_oracle_crosscheck() {
    // ME, N=3, self-loops forbidden.
    // Strengths: out=[3, 4, 3], in=[5, 4, 1], total=10.
    // These pass the loopless feasibility condition and the max-flow
    // oracle on the complete loopless domain.
    let n = 3usize;
    let strengths_out: Vec<OccNum> = vec![3, 4, 3];
    let strengths_in: Vec<OccNum> = vec![5, 4, 1];

    // --- 1. Verify loopless feasibility condition ---
    assert!(
        loopless_feasibility_check(&strengths_out, &strengths_in),
        "loopless feasibility check failed for known-feasible strengths"
    );

    // --- 2. Verify via max-flow oracle (complete loopless domain) ---
    let admissible = complete_admissible(n, false);
    let oracle_result = feasibility_max_flow(
        &strengths_out,
        &strengths_in,
        OccupationFamily::ME,
        &admissible,
        OccNum::MAX,
    );
    assert!(
        oracle_result.is_ok(),
        "max-flow oracle must confirm feasibility: {:?}",
        oracle_result
    );
    let oracle_table = oracle_result.unwrap();
    assert!(
        oracle_table.is_some(),
        "oracle should return a non-empty table for non-zero total"
    );

    // --- 3. Construct a state WITH self-loops that realises the same strengths ---
    //
    //   Occupation matrix with self-loops:
    //      0   1   2  | out
    //   0  2   1   0  |  3
    //   1  2   1   1  |  4
    //   2  1   2   0  |  3
    //   --------------
    // in: 5   4   1  | 10
    let pairs_with_loops: Vec<((u64, u64), OccNum)> = vec![
        ((0, 0), 2),
        ((0, 1), 1),
        ((1, 0), 2),
        ((1, 1), 1),
        ((1, 2), 1),
        ((2, 0), 1),
        ((2, 1), 2),
    ];
    let mut state = StrengthState::new(n, pairs_with_loops);
    let out_before = state.out_strengths.clone();
    let in_before = state.in_strengths.clone();

    // Confirm the state matches target strengths.
    assert_eq!(
        state.out_strengths, strengths_out,
        "out-strengths must match"
    );
    assert_eq!(state.in_strengths, strengths_in, "in-strengths must match");

    // Confirm self-loops exist.
    let initial_self_loops: Vec<_> = state.iter_occupied().filter(|((s, t), _)| s == t).collect();
    assert!(
        !initial_self_loops.is_empty(),
        "must have self-loops to repair"
    );

    // --- 4. Apply loop repair ---
    let domain = PairDomain::Complete {
        node_count: n,
        self_loops: false,
    };
    let config = RepairConfig::default();
    let mut rng = StdRng::seed_from_u64(42);

    repair_self_loops(&mut state, &domain, &config, &mut rng)
        .expect("loop repair must succeed for feasible problem");

    // --- 5. Verify no self-loops remain and strengths preserved ---
    let remaining_self_loops: Vec<_> = state.iter_occupied().filter(|((s, t), _)| s == t).collect();
    assert!(
        remaining_self_loops.is_empty(),
        "self-loops remain after repair: {:?}",
        remaining_self_loops
    );
    assert_eq!(
        state.out_strengths, out_before,
        "out-strengths must be preserved"
    );
    assert_eq!(
        state.in_strengths, in_before,
        "in-strengths must be preserved"
    );
}

#[test]
fn repair_self_loops_infeasible_rejected() {
    // Verify that the loopless feasibility check correctly rejects
    // a strength sequence where some node has s_out + s_in > T.
    // N=3, out=[10,0,0], in=[10,0,0], T=10.
    // Node 0: 10 + 10 = 20 > 10 → infeasible.
    let out = vec![10u64, 0, 0];
    let inp = vec![10u64, 0, 0];
    assert!(
        !loopless_feasibility_check(&out, &inp),
        "must reject infeasible strengths"
    );
}

// ══════════════════════════════════════════════════════════════════════
// B capacity repair
// ══════════════════════════════════════════════════════════════════════

#[test]
fn repair_capacity_oracle_crosscheck() {
    // B{layers=3}, N=3, strengths out=[5,5,0], in=[5,5,0], total=10.
    // The max-flow oracle confirms feasibility with capacity=3.
    // We construct a state with self-loop violations:
    //   t_00 = 5 (>3), t_11 = 5 (>3)
    // After repair all occupations must be ≤ 3 and strengths preserved.
    let n = 3usize;
    let strengths_out: Vec<OccNum> = vec![5, 5, 0];
    let strengths_in: Vec<OccNum> = vec![5, 5, 0];
    let family = OccupationFamily::B { layers: 3 };
    let cap = 3u64;

    // --- 1. Verify feasibility via max-flow oracle (complete, capacity=3) ---
    let admissible = complete_admissible(n, true);
    let oracle_result =
        feasibility_max_flow(&strengths_out, &strengths_in, family, &admissible, cap);
    assert!(
        oracle_result.is_ok(),
        "oracle must confirm feasibility: {:?}",
        oracle_result
    );
    let oracle_table = oracle_result.unwrap();
    assert!(oracle_table.is_some(), "oracle should return a table");

    // --- 2. Construct state with capacity violations ---
    //   t_00 = 5 (exceeds M=3), t_11 = 5 (exceeds M=3)
    //   No other cells occupied.
    //   Strengths: out=[5,5,0], in=[5,5,0]
    let pairs: Vec<((u64, u64), OccNum)> = vec![((0, 0), 5), ((1, 1), 5)];
    let mut state = StrengthState::new(n, pairs);
    let out_before = state.out_strengths.clone();
    let in_before = state.in_strengths.clone();

    // Verify violations exist.
    let violations: Vec<_> = state
        .iter_occupied()
        .filter(|(_, occ)| *occ > cap)
        .collect();
    assert_eq!(violations.len(), 2, "expected 2 capacity violations");

    // --- 3. Repair ---
    let domain = PairDomain::Complete {
        node_count: n,
        self_loops: true,
    };
    let config = RepairConfig::default();
    let mut rng = StdRng::seed_from_u64(42);

    repair_capacity(&mut state, family, &domain, &config, &mut rng)
        .expect("capacity repair must succeed for feasible problem");

    // --- 4. Verify ---
    for ((src, tgt), occ) in state.iter_occupied() {
        assert!(
            occ <= cap,
            "occupation {occ} at ({src},{tgt}) exceeds capacity {cap}"
        );
    }
    assert_eq!(
        state.out_strengths, out_before,
        "out-strengths must be preserved"
    );
    assert_eq!(
        state.in_strengths, in_before,
        "in-strengths must be preserved"
    );
}

// ══════════════════════════════════════════════════════════════════════
// Admissibility repair (sparse domain)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn repair_inadmissible_pairs_oracle_crosscheck() {
    // ME, N=3, sparse domain with only off-diagonal pairs allowed
    // (no self-loops).  Strengths: out=[3,4,3], in=[5,4,1], total=10
    // (same as the loop-repair test).
    // The state has self-loops which are inadmissible.  The oracle
    // confirms feasibility with the admissible set.  After repair,
    // no mass may remain on inadmissible pairs.
    let n = 3usize;
    let strengths_out: Vec<OccNum> = vec![3, 4, 3];
    let strengths_in: Vec<OccNum> = vec![5, 4, 1];
    let family = OccupationFamily::ME;

    // Build sparse domain: only off-diagonal pairs allowed.
    let mut allowed_set = HashSet::new();
    for i in 0..n as u64 {
        for j in 0..n as u64 {
            if i != j {
                allowed_set.insert((i, j));
            }
        }
    }
    let admissible_oracle: Vec<(u64, u64)> = allowed_set.iter().copied().collect();
    let domain = PairDomain::Sparse {
        node_count: n,
        allowed: allowed_set.clone(),
    };

    // --- 1. Verify feasibility via max-flow oracle ---
    let oracle_result = feasibility_max_flow(
        &strengths_out,
        &strengths_in,
        family,
        &admissible_oracle,
        OccNum::MAX,
    );
    assert!(
        oracle_result.is_ok(),
        "oracle must confirm feasibility: {:?}",
        oracle_result
    );

    // --- 2. Construct state with inadmissible self-loop mass ---
    //
    //   Same occupation matrix as the loop-repair test:
    //      0   1   2  | out
    //   0  2   1   0  |  3
    //   1  2   1   1  |  4
    //   2  1   2   0  |  3
    //   --------------
    // in: 5   4   1  | 10
    //
    //   Cells (0,0) and (1,1) are self-loops → inadmissible.
    let pairs: Vec<((u64, u64), OccNum)> = vec![
        ((0, 0), 2),
        ((0, 1), 1),
        ((1, 0), 2),
        ((1, 1), 1),
        ((1, 2), 1),
        ((2, 0), 1),
        ((2, 1), 2),
    ];
    let mut state = StrengthState::new(n, pairs);
    let out_before = state.out_strengths.clone();
    let in_before = state.in_strengths.clone();

    // Verify inadmissible mass exists.
    let inadmissible: Vec<_> = state
        .iter_occupied()
        .filter(|((s, t), _)| !domain.is_admissible(*s, *t))
        .collect();
    assert!(!inadmissible.is_empty(), "must have inadmissible mass");

    // --- 3. Repair ---
    let config = RepairConfig::default();
    let mut rng = StdRng::seed_from_u64(42);

    repair_inadmissible_pairs(&mut state, family, &domain, &config, &mut rng)
        .expect("admissibility repair must succeed for feasible problem");

    // --- 4. Verify ---
    for ((s, t), _) in state.iter_occupied() {
        assert!(
            domain.is_admissible(s, t),
            "pair ({s},{t}) remains occupied but is inadmissible"
        );
    }
    assert_eq!(
        state.out_strengths, out_before,
        "out-strengths must be preserved"
    );
    assert_eq!(
        state.in_strengths, in_before,
        "in-strengths must be preserved"
    );
}
