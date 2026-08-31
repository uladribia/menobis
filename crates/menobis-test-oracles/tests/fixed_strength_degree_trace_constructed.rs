//! Gate D (plan §42–§44): N=1000 fixed-(s,k) trace mobility from the
//! **actual extras-first constructor output**, not the witness.
//!
//! Gate A proved the exact first-return degree trace is mobile from
//! *witness* states (realistic PA-geographic, occupation-1 fraction
//! ≈ 0.18).  Gate D must not assume that transfers: the extras-first
//! constructor (§6–§35) produces a different occupation pattern
//! (extras concentrated on few edges, rest occupation-1 fillers).  This
//! gate constructs the state from witness-derived *constraints only*
//! via `initialize_exact_sk_extras_first`, then runs the existing
//! `benchmark_fixed_sk_trace_from_exact_table` on that constructed
//! table.
//!
//! Mandatory cases (§43): ME realistic PA N=1000, W realistic PA
//! N=1000, B Balanced12 N=1000.  Initial trace config: `λ = 1`, cap 16,
//! 100,000 attempts for final evidence.
//!
//! Gate D interpretation (§44):
//! - realistic ME/W require approximately `different-state rate >= 1e-2`
//!   with clear nonzero support movement (engineering gate);
//! - degenerate targets that force no occupation-1 edges may remain
//!   mobility warnings rather than constructor errors;
//! - if realistic constructed states are effectively immobile, stop
//!   before public exposure.

use menobis_core::generation::microcanonical::occupation_mcmc::chain::{
    benchmark_fixed_sk_trace_from_exact_table, FixedSkTraceBenchmark,
};
use menobis_core::generation::microcanonical::occupation_mcmc::domain::PairDomain;
use menobis_core::generation::microcanonical::occupation_mcmc::fixed_degree_init::{
    initialize_exact_sk_extras_first, ExactSkInitConfig,
};
use menobis_core::generation::microcanonical::occupation_mcmc::fixed_degrees::DegreeTraceConfig;
use menobis_core::generation::microcanonical::occupation_mcmc::problem::FixedStrengthProblem;
use menobis_core::model::family::OccupationFamily;
use menobis_core::OccNum;
use menobis_test_oracles::pa_geographic::{pa_geographic_witness, OccupationPattern, PaGeoConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn pa_geo_n1000(sl: bool) -> PaGeoConfig {
    PaGeoConfig {
        node_count: 1000,
        average_degree: 8.0,
        self_loops: sl,
        seed: 42,
    }
}

/// Construct an exact (s,k) state with the extras-first constructor from
/// witness-derived constraints (never the witness table), then run the
/// existing trace on it.  Returns the benchmark + constructed one-fraction.
fn run_constructed_case(
    family: OccupationFamily,
    cfg: &PaGeoConfig,
    pattern: OccupationPattern,
    attempts: usize,
    lambda: f64,
    cap: usize,
    seed: u64,
) -> Result<(FixedSkTraceBenchmark, f64), String> {
    let w = pa_geographic_witness(cfg, pattern);
    if let OccupationFamily::B { layers } = family {
        assert!(
            w.table.iter().all(|&(_, o)| o <= layers as OccNum),
            "witness occupations must fit in B M={layers}"
        );
    }
    let domain = PairDomain::Complete {
        node_count: w.n,
        self_loops: cfg.self_loops,
    };
    // 1. Construct from constraints only.
    let full = FixedStrengthProblem::new(
        family,
        w.strength_out.clone(),
        w.strength_in.clone(),
        domain.clone(),
        vec![],
    )
    .map_err(|e| e.to_string())?;
    let residual = full.clone().into_residual().map_err(|e| e.to_string())?;
    let mut rng = StdRng::seed_from_u64(seed);
    let (state, diag) = initialize_exact_sk_extras_first(
        &residual,
        &w.degree_out,
        &w.degree_in,
        &mut rng,
        &ExactSkInitConfig::default(),
    )
    .map_err(|e| e.to_string())?;
    let constructed_table: Vec<((u64, u64), OccNum)> = state.iter_occupied().collect();
    let one_fraction = state.iter_occupied().filter(|&(_, o)| o == 1).count() as f64
        / state.occupied_count() as f64;
    eprintln!(
        "[constructed] {family:?} extras_attempts={} extras_edges={} filler_edges={} occ1={:.3} diag_occ1={:.3} residual_total={}",
        diag.extras_attempts,
        diag.extras_edges,
        diag.filler_edges,
        one_fraction,
        diag.occupation_one_fraction,
        diag.residual_total,
    );
    // 2. Run the existing trace on the constructed table.
    let bench = benchmark_fixed_sk_trace_from_exact_table(
        full,
        w.degree_out.clone(),
        w.degree_in.clone(),
        constructed_table,
        attempts,
        seed,
        DegreeTraceConfig {
            lambda,
            max_steps: cap,
        },
    )
    .map_err(|e| e.to_string())?;
    Ok((bench, one_fraction))
}

fn report(
    label: &str,
    family: OccupationFamily,
    one_fraction: f64,
    bench: &FixedSkTraceBenchmark,
    lambda: f64,
    cap: usize,
) -> String {
    let t = bench.trace.trace_attempts.max(1) as f64;
    let diff = bench.trace.different_state_returns;
    let sup = bench.trace.support_changed_returns;
    let aux = bench.trace.auxiliary_steps as f64;
    let div0 = |x: u64| if x == 0 { f64::NAN } else { aux / x as f64 };
    let diff_rate = diff as f64 / t;
    let class = if diff_rate < 1e-4 || diff == 0 || sup == 0 {
        "RED"
    } else if diff_rate >= 1e-2 {
        "GREEN"
    } else {
        "YELLOW"
    };
    let s = format!(
        "[{label}]\n  {family:?} N={} E={} T/E={:.3} constructed_occ1={:.4} (λ,cap)=({lambda:.2},{cap})\n  \
         attempts={} step1={} departures={} successful={} diff_state={diff} support_changed={sup} \
         timeouts={} aux_steps={} accepts={} rejects={} max_exc={}\n  \
         wall={:.2}s class={class}\n  \
         diff_return_rate={:.2e} support_return_rate={:.2e} timeout_rate={:.2e}\n  \
         aux/attempt={:.2} aux/diff_return={:.2e} aux/support_return={:.2e}",
        bench.n,
        bench.e,
        bench.total_strength as f64 / bench.e as f64,
        one_fraction,
        bench.trace.trace_attempts,
        bench.trace.step1_returns,
        bench.trace.departures,
        bench.trace.successful_returns,
        bench.trace.timeouts,
        bench.trace.auxiliary_steps,
        bench.trace.outer_accepts,
        bench.trace.outer_rejects,
        bench.trace.max_excursion_distance,
        bench.wall_time_s,
        diff_rate,
        sup as f64 / t,
        bench.trace.timeouts as f64 / t,
        aux / t,
        div0(diff),
        div0(sup),
    );
    eprintln!("{s}");
    s
}

/// Gate D smoke: 10,000 attempts on the three mandatory cases (§43 says
/// the final evidence needs 100,000; a smoke run is allowed first).
#[test]
#[ignore]
fn gate_d_constructed_smoke_10k() {
    let cfg = pa_geo_n1000(false);
    let cases = [
        (
            "ME realistic PA-geo T/E=8 ".to_string(),
            OccupationFamily::ME,
            OccupationPattern::PaGeographic {
                events_per_edge: 8.0,
            },
        ),
        (
            "W M=1 realistic PA-geo T/E=8".to_string(),
            OccupationFamily::W { layers: 1 },
            OccupationPattern::PaGeographic {
                events_per_edge: 8.0,
            },
        ),
        (
            "B M=5 Balanced12 T/E=1.5 ".to_string(),
            OccupationFamily::B { layers: 5 },
            OccupationPattern::Balanced12,
        ),
    ];
    for (label, family, pattern) in cases {
        match run_constructed_case(family, &cfg, pattern, 10_000, 1.0, 16, 7) {
            Ok((bench, one_fraction)) => {
                report(&label, family, one_fraction, &bench, 1.0, 16);
            }
            Err(e) => panic!("{label} failed: {e}"),
        }
    }
}

/// Gate D decision run: 100,000 attempts on the three mandatory cases.
#[test]
#[ignore]
fn gate_d_constructed_100k() {
    let cfg = pa_geo_n1000(false);
    let results: Vec<String> = vec![
        (
            "ME realistic PA-geo T/E=8 ".to_string(),
            OccupationFamily::ME,
            OccupationPattern::PaGeographic {
                events_per_edge: 8.0,
            },
        ),
        (
            "W M=1 realistic PA-geo T/E=8".to_string(),
            OccupationFamily::W { layers: 1 },
            OccupationPattern::PaGeographic {
                events_per_edge: 8.0,
            },
        ),
        (
            "B M=5 Balanced12 T/E=1.5 ".to_string(),
            OccupationFamily::B { layers: 5 },
            OccupationPattern::Balanced12,
        ),
    ]
    .into_iter()
    .map(|(label, family, pattern)| {
        match run_constructed_case(family, &cfg, pattern, 100_000, 1.0, 16, 7) {
            Ok((bench, one_fraction)) => report(&label, family, one_fraction, &bench, 1.0, 16),
            Err(e) => panic!("{label} failed: {e}"),
        }
    })
    .collect();
    for r in &results {
        eprintln!("{r}");
    }
}
