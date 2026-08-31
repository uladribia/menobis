//! Heavy: N=1000 scalability gate for the fixed-(s,k) sampler (§49).
//!
//! # Status: extras-first constructor active (Gate C/D passed)
//!
//! These tests used to pin the degree-repair STOP artifact
//! (`DegreeRepairExhausted` at N=1000) and the repair floor scaling
//! with N.  That evidence is preserved in the decision record
//! `docs/decisions/microcanonical-fixed-sk-stop.md`; per plan Part I
//! §62 the live tests now require **extras-first success**: the
//! production one-shot must construct an exact `D = 0` state directly
//! and sample without any degree repair (§45–§47).
//!
//! The N=1000 sampler gate runs burn-in + thinning on the trace with a
//! small budget (exactness is the gate; mixing is measured separately
//! by `fixed_strength_degree_trace_constructed`).  `[ignore]`d because
//! each case runs an N=1000 MCMC sweep.

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

/// §49: N=1000 one-shot fixed-(s,k) sampling on a heterogeneous
/// random-support synthetic instance.  The extras-first constructor must
/// build an exact D=0 state directly (no degree repair), the trace
/// sweeps must end on the fiber, and the final output must reproduce the
/// exact constraints.  (This test previously pinned `DegreeRepairExhausted`;
/// the STOP artifact lives in the decision record now.)
#[test]
#[ignore]
fn n1000_extras_first_one_shot() {
    let n = 1000usize;
    let table = synthetic_feasible_table(n, 8, 1, 3, 42);
    let (out, inp, k_out, k_in, e) = derive_constraints(&table, n);
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
    let (_net, bench) = sample_fixed_strength_degree_bench(
        problem,
        k_out,
        k_in,
        config,
        DegreeTraceConfig::default(),
    )
    .unwrap_or_else(|e| panic!("N=1000 extras-first one-shot failed: {e}"));
    eprintln!(
        "[one-shot] N={n} E={e} direct_init={:.3}s extras_attempts={} extras_edges={} \
         filler_edges={} completion={} occ1={:.3} mcmc={:.3}s",
        bench.direct_init_time_s,
        bench.extras_attempts,
        bench.extras_edges,
        bench.filler_edges,
        bench.completion_attempts,
        bench.occupation_one_fraction,
        bench.mcmc_time_s,
    );
    assert_eq!(bench.target_edges, e);
    assert!(
        bench.extras_edges >= 1,
        "heterogeneous residual needs extras"
    );
    // The sampled network itself was validated inside the sampler
    // (exact full strengths/degrees/E, §26).
}

/// Mixed-size N∈{60,120} one-shot gate: the extras-first constructor
/// succeeds regardless of scale (this previously pinned the repair
/// floor growing with N).
#[test]
#[ignore]
fn n60_n120_extras_first_success() {
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
            burn_in_sweeps: 1,
            sweeps_per_sample: 1,
            proposals_per_sweep: Some(500),
            seed: 3,
        };
        match sample_fixed_strength_degree_bench(
            problem,
            k_out,
            k_in,
            config,
            DegreeTraceConfig::default(),
        ) {
            Ok((_, bench)) => eprintln!(
                "[scale] N={n}: direct_init={:.3}s extras={} occ1={:.3}",
                bench.direct_init_time_s, bench.extras_edges, bench.occupation_one_fraction
            ),
            other => panic!("N={n}: expected extras-first success, got {other:?}"),
        }
    }
}
