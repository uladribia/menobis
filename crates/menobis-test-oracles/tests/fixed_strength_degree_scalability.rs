//! Heavy: N=1000 scalability gate for the fixed-(s,k) sampler (§42–§43).
//!
//! # Status: STOPPED (algorithmic limitation, §43.1 policy)
//!
//! The N=1000 degree-repair gate **fails as designed**: the shared
//! degree-biased auxiliary step (one recorded `K_E` transition + outer
//! degree-distance MH, `λ = 1.0`) cannot bring a randomized exact-E
//! start to the exact degree target at scale.  The distance trajectory
//! shows a geometric tail that **floors at a strictly positive D** that
//! grows roughly linearly with N (probed: N=30 floor D≈21, N=100
//! D≈79, N=200 D≈154 under default budgets; `λ ∈ {0.5, 1.0, 2.0}` all
//! floor).  Degree repair converges only on tiny enumerable fibers
//! (N ≤ 3) where the exact-E state space is small enough for the walk to
//! find the target.
//!
//! Per §43.1 the plan mandates STOP: do not blindly raise the repair
//! budget, and per §52 do not add a new move family or a second repair
//! policy in this task.  The exact trace-matrix oracle (§22), the
//! production-vs-oracle correspondence (§39), the degree repair on tiny
//! fibers (§41), and the tiny-N end-to-end gates all pass — the failure
//! is specifically the **initialization repair at scale**, not the
//! stationary trace kernel.
//!
//! These tests pin the documented behavior so the STOP artifact is
//! reproducible and the diagnostics are captured as permanent evidence:
//! `[ignore]`d because each case runs a large repair budget.

use std::collections::HashSet;

use menobis_core::generation::microcanonical::mcmc::McmcConfig;
use menobis_core::generation::microcanonical::occupation_mcmc::chain::sample_fixed_strength_degree_bench;
use menobis_core::generation::microcanonical::occupation_mcmc::domain::PairDomain;
use menobis_core::generation::microcanonical::occupation_mcmc::fixed_degrees::DegreeTraceConfig;
use menobis_core::generation::microcanonical::occupation_mcmc::problem::FixedStrengthProblem;
use menobis_core::model::family::OccupationFamily;
use menobis_core::OccNum;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

fn synthetic_feasible_table(
    n: usize,
    d: usize,
    low: u64,
    high: u64,
    seed: u64,
) -> Vec<((u64, u64), OccNum)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut table: Vec<((u64, u64), OccNum)> = Vec::new();
    for i in 0..n {
        let mut chosen = HashSet::new();
        while chosen.len() < d {
            let j = rng.random_range(0..n);
            if j != i && chosen.insert(j) {
                let occ = rng.random_range(low..=high);
                table.push(((i as u64, j as u64), occ));
            }
        }
    }
    table
}

fn derive_constraints(
    table: &[((u64, u64), OccNum)],
    n: usize,
) -> (Vec<OccNum>, Vec<OccNum>, Vec<u32>, Vec<u32>, usize) {
    let mut s_out = vec![0u64; n];
    let mut s_in = vec![0u64; n];
    let mut k_out = vec![0u32; n];
    let mut k_in = vec![0u32; n];
    for &((s, t), o) in table {
        s_out[s as usize] += o;
        s_in[t as usize] += o;
        k_out[s as usize] += 1;
        k_in[t as usize] += 1;
    }
    (s_out, s_in, k_out, k_in, table.len())
}

/// §43.1 STOP artifact: N=1000 repair must exhaust with the structured
/// `DegreeRepairExhausted` error (never an inexact sample), and the
/// diagnostics must be sane.  This pins the documented algorithmic
/// limitation and its reproducible diagnostics.
#[test]
#[ignore]
fn n1000_degree_repair_stop_artifact() {
    let n = 1000usize;
    let table = synthetic_feasible_table(n, 8, 1, 3, 42);
    let (out, inp, k_out, k_in, _) = derive_constraints(&table, n);
    let domain = PairDomain::Complete {
        node_count: n,
        self_loops: false,
    };
    let problem =
        FixedStrengthProblem::new(OccupationFamily::ME, out, inp, domain, vec![]).unwrap();
    let config = McmcConfig {
        burn_in_sweeps: 2,
        sweeps_per_sample: 2,
        proposals_per_sweep: Some(2000),
        seed: 1,
    };
    match sample_fixed_strength_degree_bench(
        problem,
        k_out,
        k_in,
        config,
        DegreeTraceConfig::default(),
    ) {
        Err(menobis_core::generation::microcanonical::occupation_mcmc::errors::FixedStrengthError::DegreeRepairExhausted {
            best_degree_distance,
            restarts,
            total_steps,
            target_edges,
        }) => {
            eprintln!(
                "[STOP artifact] N={n} degree repair exhausted: best D={best_degree_distance}, \
                 restarts={restarts}, steps={total_steps}, target edges={target_edges}"
            );
            assert!(best_degree_distance > 0, "best distance must be positive");
            assert_eq!(target_edges, 8 * n);
            assert!(restarts >= 1);
            assert!(total_steps > 0);
        }
        Ok(_) => panic!(
            "N=1000 degree repair unexpectedly succeeded — the STOP artifact is stale; \
             re-evaluate §43.1 before enabling production use"
        ),
        Err(other) => panic!("expected DegreeRepairExhausted (documented STOP), got {other:?}"),
    }
}

/// Mixed-size floor characterization (fast, the inspect-step evidence of
/// §43.1): the repair's floor grows with N while the budget is held
/// fixed — pins the geometric-tail / floor observation in the report.
#[test]
#[ignore]
fn degree_repair_floor_scales_with_n() {
    for n in [60usize, 120] {
        let table = synthetic_feasible_table(n, 8, 1, 3, 42);
        let (out, inp, k_out, k_in, _) = derive_constraints(&table, n);
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: false,
        };
        let problem =
            FixedStrengthProblem::new(OccupationFamily::ME, out, inp, domain, vec![]).unwrap();
        let config = McmcConfig {
            burn_in_sweeps: 0,
            sweeps_per_sample: 0,
            proposals_per_sweep: Some(1000),
            seed: 3,
        };
        match sample_fixed_strength_degree_bench(
            problem,
            k_out,
            k_in,
            config,
            DegreeTraceConfig::default(),
        ) {
            Err(menobis_core::generation::microcanonical::occupation_mcmc::errors::FixedStrengthError::DegreeRepairExhausted {
                best_degree_distance,
                ..
            }) => eprintln!("[floor] N={n}: best D={best_degree_distance} (default repair budget)"),
            other => panic!("N={n}: expected repair exhaustion, got {other:?}"),
        }
    }
}
