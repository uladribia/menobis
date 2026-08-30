//! Gate A: N=1000 fixed-(s,k) trace mobility from **exact** witnesses
//! (recovery plan §5–§12).
//!
//! # What this measures
//!
//! The old STOP artifact showed that the **initialization** degree repair
//! cannot reach `D = 0` at N=1000 (floor ~O(N)).  Gate A instead asks:
//! *given an already exact `(s,k)` state on the degree fiber*, is the
//! existing capped first-return degree trace practically mobile?  It
//! never runs construction/repair — witnesses come from the PA
//! geographic generator (`menobis_test_oracles::pa_geographic`), which
//! builds a heterogeneous preferential-attachment support with
//! plan-mandated occupation patterns (§8, §11).
//!
//! # Classification policy (§9) — engineering diagnostics, not
//! correctness thresholds
//!
//! - GREEN:  `different_state_returns / trace_attempts >= 1e-2` with
//!   clear nonzero support movement.
//! - YELLOW: `1e-4 <= rate < 1e-2`, or support movement much rarer than
//!   occupation-only movement → run the §10 tuning grid before
//!   redesigning.
//! - RED:    rate `< 1e-4`, or `different_state_returns == 0`, or
//!   `support_changed_returns == 0` on a fiber known to contain multiple
//!   supports, or `timeout_rate > 0.95` (cap tuning failing), or
//!   `aux / different_state_return > 100_000`.
//!
//! All tests are `#[ignore]`d (each runs 10k–100k top-level traces).
//! Run with `--release`:
//!
//! ```text
//! cargo test -p menobis-test-oracles \
//!   --test fixed_strength_degree_trace_gate --release -- --ignored --nocapture
//! ```

use menobis_core::generation::microcanonical::occupation_mcmc::chain::{
    benchmark_fixed_sk_trace_from_exact_table, FixedSkTraceBenchmark,
};
use menobis_core::generation::microcanonical::occupation_mcmc::domain::PairDomain;
use menobis_core::generation::microcanonical::occupation_mcmc::fixed_degrees::DegreeTraceConfig;
use menobis_core::generation::microcanonical::occupation_mcmc::problem::FixedStrengthProblem;
use menobis_core::model::family::OccupationFamily;
use menobis_core::OccNum;
use menobis_test_oracles::pa_geographic::{pa_geographic_witness, OccupationPattern, PaGeoConfig};

/// Runs one Gate A case and returns the benchmark.
fn run_case(
    family: OccupationFamily,
    cfg: &PaGeoConfig,
    pattern: OccupationPattern,
    attempts: usize,
    lambda: f64,
    cap: usize,
    seed: u64,
) -> Result<FixedSkTraceBenchmark, String> {
    let w = pa_geographic_witness(cfg, pattern);
    // Family capacity: B occupations must fit in M layers.
    if let OccupationFamily::B { layers } = family {
        for &(_, o) in &w.table {
            assert!(
                o <= layers as OccNum,
                "witness occupation {o} exceeds B layers {layers}"
            );
        }
    }
    let domain = PairDomain::Complete {
        node_count: w.n,
        self_loops: cfg.self_loops,
    };
    let problem = FixedStrengthProblem::new(
        family,
        w.strength_out.clone(),
        w.strength_in.clone(),
        domain,
        vec![],
    )
    .map_err(|e| e.to_string())?;
    benchmark_fixed_sk_trace_from_exact_table(
        problem,
        w.degree_out.clone(),
        w.degree_in.clone(),
        w.table.clone(),
        attempts,
        seed,
        DegreeTraceConfig {
            lambda,
            max_steps: cap,
        },
    )
    .map_err(|e| e.to_string())
}

/// Derived rates + GREEN/YELLOW/RED classification (§8, §9).
#[derive(Clone, Debug)]
struct CaseReport {
    label: String,
    family: String,
    n: usize,
    e: usize,
    total_strength: OccNum,
    frac_occ1: f64,
    trace_attempts: u64,
    step1_returns: u64,
    departures: u64,
    successful_returns: u64,
    different_state_returns: u64,
    support_changed_returns: u64,
    timeouts: u64,
    auxiliary_steps: u64,
    outer_accepts: u64,
    outer_rejects: u64,
    max_excursion_distance: u64,
    wall_time_s: f64,
    classification: &'static str,
}

fn classify(bench: &FixedSkTraceBenchmark, fiber_has_multiple_supports: bool) -> &'static str {
    let t = bench.trace.trace_attempts.max(1) as f64;
    let diff_rate = bench.trace.different_state_returns as f64 / t;
    let support = bench.trace.support_changed_returns;
    let timeout_rate = bench.trace.timeouts as f64 / t;
    let aux = bench.trace.auxiliary_steps as f64;
    let aux_per_diff = if bench.trace.different_state_returns == 0 {
        f64::INFINITY
    } else {
        aux / bench.trace.different_state_returns as f64
    };
    let red = diff_rate < 1e-4
        || bench.trace.different_state_returns == 0
        || (fiber_has_multiple_supports && support == 0)
        || timeout_rate > 0.95
        || aux_per_diff > 100_000.0;
    if red {
        "RED"
    } else if diff_rate >= 1e-2 {
        "GREEN"
    } else {
        "YELLOW"
    }
}

fn report_case(
    label: &str,
    family: OccupationFamily,
    cfg: &PaGeoConfig,
    pattern: OccupationPattern,
    lambda: f64,
    cap: usize,
    bench: &FixedSkTraceBenchmark,
) -> CaseReport {
    let w = pa_geographic_witness(cfg, pattern);
    let t = bench.trace.trace_attempts.max(1) as f64;
    // The witness itself is on the fiber; a realistic N=1000 d=8 PA
    // support fiber contains many supports, so support movement is
    // expected unless mobility collapses.
    let rep = CaseReport {
        label: label.into(),
        family: format!("{family:?}"),
        n: bench.n,
        e: bench.e,
        total_strength: bench.total_strength,
        frac_occ1: w.fraction_occupation_1(),
        trace_attempts: bench.trace.trace_attempts,
        step1_returns: bench.trace.step1_returns,
        departures: bench.trace.departures,
        successful_returns: bench.trace.successful_returns,
        different_state_returns: bench.trace.different_state_returns,
        support_changed_returns: bench.trace.support_changed_returns,
        timeouts: bench.trace.timeouts,
        auxiliary_steps: bench.trace.auxiliary_steps,
        outer_accepts: bench.trace.outer_accepts,
        outer_rejects: bench.trace.outer_rejects,
        max_excursion_distance: bench.trace.max_excursion_distance,
        wall_time_s: bench.wall_time_s,
        classification: classify(bench, true),
    };
    // ── print the raw counters + derived rates (§8 record) ──
    let diff = bench.trace.different_state_returns;
    let sup = bench.trace.support_changed_returns;
    let aux = bench.trace.auxiliary_steps as f64;
    let div0 = |x: u64| if x == 0 { f64::NAN } else { aux / x as f64 };
    eprintln!(
        "\n[{}]\n  N={} E={} T={} T/E={:.3} frac_occ1={:.4} (λ,cap)=({:.2},{})\n  \
         attempts={} step1={} departures={} successful={} diff_state={} support_changed={} \
         timeouts={} aux_steps={} accepts={} rejects={} max_exc={}\n  \
         wall={:.2}s\n  \
         class={} family={}\n  \
         diff_return_rate={:.2e} support_return_rate={:.2e} timeout_rate={:.2e}\n  \
         aux/attempt={:.2} aux/diff_return={:.2e} aux/support_return={:.2e}",
        rep.label,
        rep.n,
        rep.e,
        rep.total_strength,
        rep.total_strength as f64 / rep.e as f64,
        rep.frac_occ1,
        lambda,
        cap,
        rep.trace_attempts,
        rep.step1_returns,
        rep.departures,
        rep.successful_returns,
        rep.different_state_returns,
        rep.support_changed_returns,
        rep.timeouts,
        rep.auxiliary_steps,
        rep.outer_accepts,
        rep.outer_rejects,
        rep.max_excursion_distance,
        rep.wall_time_s,
        rep.classification,
        rep.family,
        diff as f64 / t,
        sup as f64 / t,
        rep.timeouts as f64 / t,
        aux / t,
        div0(diff),
        div0(sup),
    );
    rep
}

/// The mandatory ME witness configurations (§8).
fn me_cases() -> Vec<(String, OccupationPattern)> {
    vec![
        (
            "A1 all-1        T/E=1 ".into(),
            OccupationPattern::Uniform(1),
        ),
        (
            "A2 balanced 1/2 T/E=1.5".into(),
            OccupationPattern::Balanced12,
        ),
        (
            "A3 all-2        T/E=2 ".into(),
            OccupationPattern::Uniform(2),
        ),
        (
            "A4 all-3        T/E=3 ".into(),
            OccupationPattern::Uniform(3),
        ),
        (
            "A5 all-5        T/E=5 ".into(),
            OccupationPattern::Uniform(5),
        ),
        (
            "A6 all-10       T/E=10".into(),
            OccupationPattern::Uniform(10),
        ),
    ]
}

fn pa_geo_n1000() -> PaGeoConfig {
    PaGeoConfig {
        node_count: 1000,
        average_degree: 8.0,
        self_loops: false,
        seed: 42,
    }
}

/// 10,000-attempt smoke over all mandatory ME cases (§8: smoke allowed
/// before the final decision run).
#[test]
#[ignore]
fn gate_a_me_smoke_10k() {
    let cfg = pa_geo_n1000();
    for (label, pattern) in me_cases() {
        match run_case(OccupationFamily::ME, &cfg, pattern, 10_000, 1.0, 16, 7) {
            Ok(bench) => {
                report_case(&label, OccupationFamily::ME, &cfg, pattern, 1.0, 16, &bench);
            }
            Err(e) => panic!("case {label} failed: {e}"),
        }
    }
}

/// 100,000-attempt decision run over all mandatory ME cases (§8: the
/// final decision requires 100k attempts on the key cases).
#[test]
#[ignore]
fn gate_a_me_grid_100k() {
    let cfg = pa_geo_n1000();
    for (label, pattern) in me_cases() {
        match run_case(OccupationFamily::ME, &cfg, pattern, 100_000, 1.0, 16, 7) {
            Ok(bench) => {
                report_case(&label, OccupationFamily::ME, &cfg, pattern, 1.0, 16, &bench);
            }
            Err(e) => panic!("case {label} failed: {e}"),
        }
    }
}

/// W representative cases: `T/E ∈ {1, 2, 5}` (§11).
#[test]
#[ignore]
fn gate_a_w_grid() {
    let cfg = pa_geo_n1000();
    for c in [1u64, 2, 5] {
        let label = format!("W M=1 all-{c}   T/E={c} ");
        let pattern = OccupationPattern::Uniform(c);
        match run_case(
            OccupationFamily::W { layers: 1 },
            &cfg,
            pattern,
            100_000,
            1.0,
            16,
            7,
        ) {
            Ok(bench) => {
                report_case(
                    &label,
                    OccupationFamily::W { layers: 1 },
                    &cfg,
                    pattern,
                    1.0,
                    16,
                    &bench,
                );
            }
            Err(e) => panic!("case {label} failed: {e}"),
        }
    }
}

/// B representative cases: `M = 5`, `T/E ∈ {1, 2, 3, 5}` (§11).  The
/// `T/E = M` case is intentionally stressful (every edge at capacity).
#[test]
#[ignore]
fn gate_a_b_grid() {
    let cfg = pa_geo_n1000();
    for c in [1u64, 2, 3, 5] {
        let label = format!("B M=5 all-{c}   T/E={c} ");
        let pattern = OccupationPattern::Uniform(c);
        match run_case(
            OccupationFamily::B { layers: 5 },
            &cfg,
            pattern,
            100_000,
            1.0,
            16,
            7,
        ) {
            Ok(bench) => {
                report_case(
                    &label,
                    OccupationFamily::B { layers: 5 },
                    &cfg,
                    pattern,
                    1.0,
                    16,
                    &bench,
                );
            }
            Err(e) => panic!("case {label} failed: {e}"),
        }
    }
}

/// Realistic PA-geographic instance: score-based mixed occupations
/// (mean `events_per_edge = 8` per edge) on the heterogeneous PA support.
/// This is the "proper instance" the trace gate must answer for — not a
/// uniform corner pattern (§8 note + user refinement).  ME and W (M=1).
#[test]
#[ignore]
fn gate_a_me_and_w_pa_geographic_realistic() {
    let cfg = pa_geo_n1000();
    let pattern = OccupationPattern::PaGeographic {
        events_per_edge: 8.0,
    };
    for (family, label) in [
        (OccupationFamily::ME, "ME realistic PA-geo T/E=8 "),
        (
            OccupationFamily::W { layers: 1 },
            "W M=1 realistic PA-geo T/E=8",
        ),
    ] {
        match run_case(family, &cfg, pattern, 100_000, 1.0, 16, 7) {
            Ok(bench) => {
                report_case(label, family, &cfg, pattern, 1.0, 16, &bench);
            }
            Err(e) => panic!("case {label} failed: {e}"),
        }
    }
}

/// §10 tuning grid for YELLOW cases: fixed (λ, cap) configurations on
/// the ME A5 (T/E=5) case, choosing by support movement per K_E step.
#[test]
#[ignore]
fn gate_a_tuning_grid_yellow() {
    let cfg = pa_geo_n1000();
    for (lambda, cap) in [
        (1.0, 16),
        (0.5, 16),
        (2.0, 16),
        (1.0, 32),
        (0.5, 32),
        (1.0, 64),
    ] {
        let label = format!("ME A3 T/E=2 λ={lambda} cap={cap}");
        let pattern = OccupationPattern::Uniform(2);
        match run_case(OccupationFamily::ME, &cfg, pattern, 50_000, lambda, cap, 7) {
            Ok(bench) => {
                report_case(
                    &label,
                    OccupationFamily::ME,
                    &cfg,
                    pattern,
                    lambda,
                    cap,
                    &bench,
                );
            }
            Err(e) => panic!("case {label} failed: {e}"),
        }
    }
}
