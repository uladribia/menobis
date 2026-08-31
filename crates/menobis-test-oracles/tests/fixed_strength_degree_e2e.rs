//! Part G (§49–§52): N=1000 fixed-(s,k) end-to-end gates.
//!
//! Run the **actual production one-shot sampler**
//! (`sample_fixed_strength_degree`) on realistic PA-geographic
//! instances and fixed-pair scenarios at N=1000.  The sampler performs
//! the full pipeline: residualize s,k,fixed pairs → extras-first exact
//! construction (D=0) → degree-trace burn-in/thinning → merge fixed
//! pairs → full exact output validation (§83 architecture).  Exactness
//! is the gate; the trace counters report init vs MCMC times separately.
//!
//! All tests are `#[ignore]`d (N=1000), run with
//! `--release -- --ignored --nocapture`.

use menobis_core::generation::microcanonical::mcmc::McmcConfig;
use menobis_core::generation::microcanonical::occupation_mcmc::chain::sample_fixed_strength_degree_bench;
use menobis_core::generation::microcanonical::occupation_mcmc::domain::PairDomain;
use menobis_core::generation::microcanonical::occupation_mcmc::fixed_degrees::DegreeTraceConfig;
use menobis_core::generation::microcanonical::occupation_mcmc::problem::FixedStrengthProblem;
use menobis_core::model::family::OccupationFamily;
use menobis_core::OccNum;
use menobis_test_oracles::pa_geographic::{pa_geographic_witness, OccupationPattern, PaGeoConfig};
use std::collections::HashSet;

fn pa_geo(n: usize, d: f64, sl: bool) -> PaGeoConfig {
    PaGeoConfig {
        node_count: n,
        average_degree: d,
        self_loops: sl,
        seed: 42,
    }
}

fn mcmc_budget(seed: u64) -> McmcConfig {
    // One full E-sweep of burn-in + one full E-sweep of thinning; the
    // constructed start is already on the fiber with ~0.83 occ-1 so a
    // sweep yields thousands of support changes (Gate D evidence).
    McmcConfig {
        burn_in_sweeps: 1,
        sweeps_per_sample: 1,
        proposals_per_sweep: Some(8000),
        seed,
    }
}

fn trace_config() -> DegreeTraceConfig {
    DegreeTraceConfig {
        lambda: 1.0,
        max_steps: 16,
    }
}

/// Run the one-shot sampler and report init/MCMC times split.
fn run_e2e(
    family: OccupationFamily,
    cfg: &PaGeoConfig,
    pattern: OccupationPattern,
    fixed: Vec<(u64, u64, OccNum)>,
    seed: u64,
) -> Result<(menobis_core::generation::output::SampledNetwork, String), String> {
    let w = pa_geographic_witness(cfg, pattern);
    let n = w.n;
    // The sampler takes the FULL degree vectors and subtracts the
    // fixed-pair degrees internally (§45 residualization order); the
    // FULL targets are what the final merged output must reproduce.
    let problem = FixedStrengthProblem::new(
        family,
        w.strength_out.clone(),
        w.strength_in.clone(),
        PairDomain::Complete {
            node_count: n,
            self_loops: cfg.self_loops,
        },
        fixed.clone(),
    )
    .map_err(|e| e.to_string())?;
    let (net, bench) = sample_fixed_strength_degree_bench(
        problem,
        w.degree_out.clone(),
        w.degree_in.clone(),
        mcmc_budget(seed),
        trace_config(),
    )
    .map_err(|e| e.to_string())?;
    let label = format!(
        "{family:?} N={n} T/E={:.2} fixed={} | direct_init={:.3}s extras_attempts={} \
         extras_edges={} filler_edges={} occ1={:.3} | mcmc={:.3}s trace_diff={} support={}",
        pattern.t_over_e(),
        fixed.len(),
        bench.direct_init_time_s,
        bench.extras_attempts,
        bench.extras_edges,
        bench.filler_edges,
        bench.occupation_one_fraction,
        bench.mcmc_time_s,
        bench.degree_trace.different_state_returns,
        bench.degree_trace.support_changed_returns,
    );
    eprintln!("[e2e] {label}");
    Ok((net, label))
}

/// §49: actual one-shot sampler on realistic PA-geographic ME N=1000
/// (T/E=8) and Balanced12 (T/E=1.5).  The sampler's internal
/// `validate_fixed_sk_output` is an O(E) exact check of full s/k/E;
/// we additionally assert the returned network's cardinality and B-free
/// occupancy sanity.
#[test]
#[ignore]
fn n1000_e2e_me() {
    let cfg = pa_geo(1000, 8.0, false);
    let mut failures = Vec::new();
    for (pattern, tag) in [
        (
            OccupationPattern::PaGeographic {
                events_per_edge: 8.0,
            },
            "realistic T/E=8",
        ),
        (OccupationPattern::Balanced12, "balanced T/E=1.5"),
    ] {
        match run_e2e(OccupationFamily::ME, &cfg, pattern, vec![], 7) {
            Ok((net, label)) => {
                assert_eq!(
                    net.sources.len(),
                    8000,
                    "{tag}: sampled E must equal the degree sum"
                );
                assert!(
                    net.occ_nums.iter().all(|&o| o > 0),
                    "{tag}: no zero occupations"
                );
                eprintln!("[{tag}] OK {label}");
            }
            Err(e) => failures.push(format!("[{tag}] {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "ME E2E failures:\n{}",
        failures.join("\n")
    );
}

/// §50: heterogeneous W N=1000 (realistic PA-geographic, M=1 geometric).
#[test]
#[ignore]
fn n1000_e2e_w() {
    let cfg = pa_geo(1000, 8.0, false);
    match run_e2e(
        OccupationFamily::W { layers: 1 },
        &cfg,
        OccupationPattern::PaGeographic {
            events_per_edge: 8.0,
        },
        vec![],
        7,
    ) {
        Ok((net, label)) => {
            assert_eq!(net.sources.len(), 8000, "W E must equal the degree sum");
            eprintln!("[W realistic] OK {label}");
        }
        Err(e) => panic!("W E2E failed: {e}"),
    }
}

/// §51: heterogeneous B N=1000 (M=5, Balanced12, occupations ≤ 2 ≤ M)
/// plus the capacity-corner regression (Uniform(5) = every edge at
/// M; note the trace is immobile there — Gate A start-state pathology —
/// so only a tiny budget is used; exactness is still guaranteed).
#[test]
#[ignore]
fn n1000_e2e_b() {
    let cfg = pa_geo(1000, 8.0, false);
    let mut failures = Vec::new();
    match run_e2e(
        OccupationFamily::B { layers: 5 },
        &cfg,
        OccupationPattern::Balanced12,
        vec![],
        7,
    ) {
        Ok((net, label)) => {
            assert_eq!(net.sources.len(), 8000);
            assert!(
                net.occ_nums.iter().all(|&o| o <= 5),
                "B capacity must hold in the output"
            );
            eprintln!("[B M=5 Balanced12] OK {label}");
        }
        Err(e) => failures.push(format!("B M=5 Balanced12: {e}")),
    }
    // Capacity corner: every edge at M=5 (occ-1 fraction 0).  The
    // constructed state is exact; the trace self-loops are acceptable
    // for this regression (no mixing is claimed on this degenerate
    // fiber, §44).
    match run_e2e(
        OccupationFamily::B { layers: 5 },
        &cfg,
        OccupationPattern::Uniform(5),
        vec![],
        7,
    ) {
        Ok((net, label)) => {
            assert_eq!(net.sources.len(), 8000);
            assert!(
                net.occ_nums.iter().all(|&o| o == 5),
                "capacity corner must stay at M=5"
            );
            eprintln!("[B M=5 at-capacity corner] OK {label}");
        }
        Err(e) => failures.push(format!("B M=5 at-capacity: {e}")),
    }
    assert!(
        failures.is_empty(),
        "B E2E failures:\n{}",
        failures.join("\n")
    );
}

/// §52: fixed pairs end-to-end at N=1000 — positive fixed pairs,
/// zero fixed pairs, and CompleteMinus exclusion.  The merged output
/// must reproduce the full strengths/degrees (internal validation) and
/// the fixed occupations at their coordinates.
#[test]
#[ignore]
fn n1000_e2e_fixed_pairs() {
    let cfg = pa_geo(1000, 8.0, false);
    let w = pa_geographic_witness(
        &cfg,
        OccupationPattern::PaGeographic {
            events_per_edge: 8.0,
        },
    );
    let mut failures = Vec::new();

    // Positive fixed pairs: first 800 (10%) of the witness support.
    let fixed_pos: Vec<(u64, u64, OccNum)> = w.table[..800]
        .iter()
        .map(|&((s, t), o)| (s, t, o))
        .collect();
    match run_e2e(
        OccupationFamily::ME,
        &cfg,
        OccupationPattern::PaGeographic {
            events_per_edge: 8.0,
        },
        fixed_pos.clone(),
        7,
    ) {
        Ok((net, label)) => {
            let table: HashSet<(u64, u64, OccNum)> = net
                .sources
                .iter()
                .zip(net.targets.iter())
                .zip(net.occ_nums.iter())
                .map(|((&s, &t), &o)| (s, t, o))
                .collect();
            for &(s, t, o) in &fixed_pos {
                assert!(
                    table.contains(&(s, t, o)),
                    "positive fixed pair ({s},{t}) must keep occupation {o}"
                );
            }
            eprintln!("[positive-fixed E2E] OK {label}");
        }
        Err(e) => failures.push(format!("positive-fixed: {e}")),
    }

    // Zero fixed pairs / CompleteMinus: 1000 coordinates absent from
    // the witness support must stay unoccupied in the merged output.
    let support: HashSet<(u64, u64)> = w.table.iter().map(|&((s, t), _)| (s, t)).collect();
    let mut fixed_zero = Vec::new();
    for i in 0..w.n as u64 {
        for j in 0..w.n as u64 {
            if fixed_zero.len() >= 1000 {
                break;
            }
            if !support.contains(&(i, j)) && i != j {
                fixed_zero.push((i, j, 0));
            }
        }
    }
    match run_e2e(
        OccupationFamily::ME,
        &cfg,
        OccupationPattern::PaGeographic {
            events_per_edge: 8.0,
        },
        fixed_zero.clone(),
        7,
    ) {
        Ok((net, label)) => {
            let present: HashSet<(u64, u64)> = net
                .sources
                .iter()
                .zip(net.targets.iter())
                .map(|(&s, &t)| (s, t))
                .collect();
            for &(s, t, _) in &fixed_zero {
                assert!(
                    !present.contains(&(s, t)),
                    "zero fixed pair ({s},{t}) must stay unoccupied"
                );
            }
            eprintln!("[zero-fixed E2E] OK {label}");
        }
        Err(e) => failures.push(format!("zero-fixed: {e}")),
    }
    assert!(
        failures.is_empty(),
        "fixed-pair E2E failures:\n{}",
        failures.join("\n")
    );
}
