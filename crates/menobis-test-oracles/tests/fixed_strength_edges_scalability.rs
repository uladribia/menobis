//! Heavy: N=1000+ scalability gates for the fixed-(s,E) sampler (§32–§35).
//!
//! Builds sparse feasible networks deterministically (never hand-picked
//! arbitrary strengths), derives constraints from the actual table, and
//! runs the full production pipeline: construct → structural repair →
//! edge repair → mixed-kernel burn-in → exact-E sample.  Asserts:
//!
//! - exact output strengths and exact occupied-pair count;
//! - no self-loop violation when the domain forbids them;
//! - sparse `CompleteMinus` residualization for fixed pairs (no `N²`
//!   materialization);
//! - nonzero MCMC movement on known non-singleton fibers (§35);
//! - repair diagnostics are sane (exact E reached before MCMC).
//!
//! Marked `#[ignore]` because each case runs thousands of sparse MCMC
//! proposals; run explicitly with `cargo test -- --ignored` for the gate.

use std::collections::HashSet;

use menobis_core::generation::microcanonical::mcmc::McmcConfig;
use menobis_core::generation::microcanonical::occupation_mcmc::chain::sample_fixed_strength_edges_bench;
use menobis_core::generation::microcanonical::occupation_mcmc::domain::PairDomain;
use menobis_core::generation::microcanonical::occupation_mcmc::fixed_edges::BridgeConfig;
use menobis_core::generation::microcanonical::occupation_mcmc::problem::FixedStrengthProblem;
use menobis_core::model::family::OccupationFamily;
use menobis_core::OccNum;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

/// Deterministic sparse feasible network: every node `i` picks `d` random
/// distinct targets and places `events_per_edge..=3·events_per_edge`
/// events per edge.  The resulting table is loopless (self-loops are
/// rejected when drawing targets), sparse (`E ≈ N·d`), and feasible —
/// constraints are derived from it, never hand-picked.
///
/// Returns the generated table; callers derive `(s_out, s_in, E)` and
/// (for fixed-pair scenarios) pick occupied / absent coordinates from it.
fn synthetic_feasible_table(
    n: usize,
    d: usize,
    events_per_edge: u64,
    seed: u64,
) -> Vec<((u64, u64), OccNum)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut table: Vec<((u64, u64), OccNum)> = Vec::new();
    for i in 0..n {
        let mut chosen = HashSet::new();
        while chosen.len() < d {
            let j = rng.random_range(0..n);
            if j != i && chosen.insert(j) {
                let occ = rng.random_range(events_per_edge..=3 * events_per_edge);
                table.push(((i as u64, j as u64), occ));
            }
        }
    }
    table
}

fn derive_constraints(
    table: &[((u64, u64), OccNum)],
    n: usize,
) -> (Vec<OccNum>, Vec<OccNum>, usize) {
    let mut s_out = vec![0u64; n];
    let mut s_in = vec![0u64; n];
    for &((s, t), o) in table {
        s_out[s as usize] += o;
        s_in[t as usize] += o;
    }
    let e = table.len();
    (s_out, s_in, e)
}

/// Run the full production pipeline for one family/scenario and assert the
/// §32 invariants.
#[allow(clippy::too_many_arguments)]
fn run_n1000_scenario(
    family: OccupationFamily,
    n: usize,
    target_e: usize,
    s_out: &[OccNum],
    s_in: &[OccNum],
    fixed: Vec<(u64, u64, OccNum)>,
    seed: u64,
) {
    let domain = PairDomain::Complete {
        node_count: n,
        self_loops: false,
    };
    let problem =
        FixedStrengthProblem::new(family, s_out.to_vec(), s_in.to_vec(), domain, fixed.clone())
            .expect("problem construction");

    // §32 fixed-pair case: residualization must stay CompleteMinus (O(F)).
    if !fixed.is_empty() {
        let residual = problem.clone().into_residual().expect("residualization");
        assert!(
            matches!(residual.domain, PairDomain::CompleteMinus { .. }),
            "fixed-pair residual domain must be CompleteMinus, not an N^2 set"
        );
        // Loopless complete policy count minus distinct fixed coordinates.
        let expected = n * n - n - fixed.len();
        assert_eq!(residual.domain.admissible_pair_count(), expected);
    }

    let config = McmcConfig {
        burn_in_sweeps: 50,
        sweeps_per_sample: 10,
        proposals_per_sweep: None,
        seed,
    };
    let (network, bench) =
        sample_fixed_strength_edges_bench(problem, target_e, config, BridgeConfig::default())
            .unwrap_or_else(|e| panic!("{family:?} N={n} E={target_e}: {e}"));

    // Exact strengths and exact E.
    assert_eq!(network.sources.len(), target_e, "{family:?} E drift");
    let mut out = vec![0u64; n];
    let mut inp = vec![0u64; n];
    for ((&s, &t), &o) in network
        .sources
        .iter()
        .zip(network.targets.iter())
        .zip(network.occ_nums.iter())
    {
        out[s as usize] += o;
        inp[t as usize] += o;
    }
    assert_eq!(out, s_out, "{family:?} out-strength drift");
    assert_eq!(inp, s_in, "{family:?} in-strength drift");

    // No self-loop violation.
    for (&s, &t) in network.sources.iter().zip(network.targets.iter()) {
        assert_ne!(s, t, "{family:?}: self-loop appeared");
    }

    // §35: at least some movement on this non-singleton fiber.
    assert!(
        bench.counters.local_accepted + bench.counters.bridge_successful_returns > 0,
        "{family:?}: no MCMC movement: counters={:?}",
        bench.counters
    );
    assert_eq!(
        bench.counters.outer_proposals as usize,
        (50 + 10) * bench.target_edges.max(2 * n),
        "{family:?}: sweep count mismatch"
    );

    // §34: repair reached exact residual E before MCMC (invariant inside
    // the orchestrator) and did not exhaust.  target_edges is the
    // residual target; the caller checks the full merged E separately.
    assert_eq!(bench.best_edges, bench.target_edges);
    eprintln!(
        "[scalability] {family:?} N={n} E={} T={} fixed={} const={:.2}s st_repr={:.2}s edge_repr={:.3}s (steps {}, restarts {}) mcmc={:.2}s local_acc={} bridge={}/{}/{}",
        target_e,
        s_out.iter().sum::<OccNum>(),
        fixed.len(),
        bench.construction_time_s,
        bench.structural_repair_time_s,
        bench.edge_repair_time_s,
        bench.edge_repair_steps,
        bench.edge_repair_restarts,
        bench.mcmc_time_s,
        bench.counters.local_accepted,
        bench.counters.bridge_successful_returns,
        bench.counters.bridge_departures,
        bench.counters.bridge_timeouts,
    );
}

/// §32 Scenario A/B/C: ME, W, and B at N=1000, E ≈ 10·N, T > E.
#[test]
#[ignore]
fn n1000_scenarios_me_w_b() {
    let n = 1000;
    let me_table = synthetic_feasible_table(n, 10, 5, 42);
    let (me_out, me_in, me_e) = derive_constraints(&me_table, n);
    run_n1000_scenario(OccupationFamily::ME, n, me_e, &me_out, &me_in, vec![], 1);
    run_n1000_scenario(OccupationFamily::ME, n, me_e, &me_out, &me_in, vec![], 2);

    let w_table = synthetic_feasible_table(n, 10, 5, 43);
    let (w_out, w_in, w_e) = derive_constraints(&w_table, n);
    run_n1000_scenario(
        OccupationFamily::W { layers: 3 },
        n,
        w_e,
        &w_out,
        &w_in,
        vec![],
        3,
    );

    // B: occupancy must fit within M=5 per pair.  events_per_edge=1 draws
    // occupancies in [1, 3] ≤ M.
    let b_table = synthetic_feasible_table(n, 10, 1, 44);
    let (b_out, b_in, b_e) = derive_constraints(&b_table, n);
    assert!(
        b_table.iter().all(|&(_, o)| o <= 5),
        "B scenario must keep per-cell occupancy within M=5"
    );
    run_n1000_scenario(
        OccupationFamily::B { layers: 5 },
        n,
        b_e,
        &b_out,
        &b_in,
        vec![],
        4,
    );
}

/// §32 fixed-pair scalability case: a modest number of positive and zero
/// fixed pairs must residualize to a sparse CompleteMinus domain and
/// sample successfully.
#[test]
#[ignore]
fn n1000_fixed_pair_scalability() {
    let n = 1000;
    let me_table = synthetic_feasible_table(n, 10, 5, 42);
    let (me_out, me_in, me_e) = derive_constraints(&me_table, n);
    let occupied: std::collections::HashSet<(u64, u64)> =
        me_table.iter().map(|&(p, _)| p).collect();
    let mut fixed = Vec::new();
    // 20 positive fixed pairs taken from actually occupied coordinates
    // (guarantees the fixed occupation fits the derived margins), at
    // occupation 1 each.
    let n_positive_fixed = 20usize;
    for &(coord, _) in me_table.iter().take(n_positive_fixed) {
        fixed.push((coord.0, coord.1, 1));
    }
    // 8 zero fixed pairs on coordinates guaranteed absent from the table.
    let mut zero_found = 0usize;
    let mut k = 0u64;
    while zero_found < 8 {
        let coord = (2 * k, 2 * k + 1);
        if !occupied.contains(&coord) {
            fixed.push((coord.0, coord.1, 0));
            zero_found += 1;
        }
        k += 1;
    }
    run_n1000_scenario(OccupationFamily::ME, n, me_e, &me_out, &me_in, fixed, 7);
    let _ = n_positive_fixed;
}

/// §33 optional smoke: N=5000, E = O(N), ME loopless.
#[test]
#[ignore]
fn n5000_smoke() {
    let n = 5000;
    let table = synthetic_feasible_table(n, 8, 4, 5);
    let (out, inp, e) = derive_constraints(&table, n);
    let domain = PairDomain::Complete {
        node_count: n,
        self_loops: false,
    };
    let problem = FixedStrengthProblem::new(
        OccupationFamily::ME,
        out.clone(),
        inp.clone(),
        domain,
        vec![],
    )
    .expect("problem construction");
    let config = McmcConfig {
        burn_in_sweeps: 5,
        sweeps_per_sample: 5,
        proposals_per_sweep: None,
        seed: 5,
    };
    let (network, bench) =
        sample_fixed_strength_edges_bench(problem, e, config, BridgeConfig::default())
            .unwrap_or_else(|err| panic!("N=5000 failed: {err}"));
    assert_eq!(network.sources.len(), e);
    let mut co = vec![0u64; n];
    let mut ci = vec![0u64; n];
    for ((&s, &t), &o) in network
        .sources
        .iter()
        .zip(network.targets.iter())
        .zip(network.occ_nums.iter())
    {
        co[s as usize] += o;
        ci[t as usize] += o;
    }
    assert_eq!(co, out);
    assert_eq!(ci, inp);
    assert!(bench.counters.local_accepted + bench.counters.bridge_successful_returns > 0);
    eprintln!(
        "[scalability] N=5000 OK const={:.2}s repair={:.2}s edge_repr={:.3}s mcmc={:.2}s",
        bench.construction_time_s,
        bench.structural_repair_time_s,
        bench.edge_repair_time_s,
        bench.mcmc_time_s
    );
}
