//! Fixed-(s,k) end-to-end scale + memory benchmark (perf note for the
//! thesis experiments; recorded in
//! `docs/decisions/microcanonical-fixed-sk-performance.md`).
//!
//! Runs the **production one-shot sampler** (`sample_fixed_strength_degree`)
//! on realistic PA-geographic constraints at N ∈ {100, 500, 1000, 2000,
//! 5000} and across ME/W/B at N=1000, reporting:
//!
//! - extras-first constructor diagnostics (attempts, extras/filler
//!   edges, occupation-1 fraction) and wall time;
//! - trace MCMC wall time and mobility (support-changing returns per
//!   sweep, auxiliary `K_E` steps per support change);
//! - peak RSS (`VmHWM`) to evidence the O(E) memory claim.
//!
//! Budget: `burn_in_sweeps = 3`, `sweeps_per_sample = 1` (4 full E-sweeps
//! per case).  Per-sweep mobility rates and `K_E`/support are
//! budget-independent diagnostics; wall times scale linearly with the
//! sweep count, so the recorded numbers extrapolate to any budget.
//! Exactness is asserted independently (sampler-internal full validation
//! + E check).
//!
//! Run in release mode (`--release -- --ignored --nocapture`); the full
//! sweep takes roughly 10–15 minutes.

use menobis_core::generation::microcanonical::mcmc::McmcConfig;
use menobis_core::generation::microcanonical::occupation_mcmc::chain::sample_fixed_strength_degree_bench;
use menobis_core::generation::microcanonical::occupation_mcmc::domain::PairDomain;
use menobis_core::generation::microcanonical::occupation_mcmc::fixed_degrees::DegreeTraceConfig;
use menobis_core::generation::microcanonical::occupation_mcmc::problem::FixedStrengthProblem;
use menobis_core::model::family::OccupationFamily;
use menobis_core::OccNum;
use menobis_test_oracles::pa_geographic::{pa_geographic_witness, OccupationPattern, PaGeoConfig};

/// Peak resident set size from `/proc/self/status` (Linux); `None` off-Linux.
fn peak_rss_mib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

fn pa_geo(n: usize, d: f64, sl: bool) -> PaGeoConfig {
    PaGeoConfig {
        node_count: n,
        average_degree: d,
        self_loops: sl,
        seed: 42,
    }
}

fn run_case(
    family: OccupationFamily,
    cfg: &PaGeoConfig,
    pattern: OccupationPattern,
    seed: u64,
) -> Result<String, String> {
    let w = pa_geographic_witness(cfg, pattern);
    if let OccupationFamily::B { layers } = family {
        assert!(
            w.table.iter().all(|&(_, o)| o <= layers as OccNum),
            "witness occupations must fit B M={layers}"
        );
    }
    let n = w.n;
    let e = w.degree_out.iter().map(|&k| k as usize).sum::<usize>();
    let problem = FixedStrengthProblem::new(
        family,
        w.strength_out.clone(),
        w.strength_in.clone(),
        PairDomain::Complete {
            node_count: n,
            self_loops: cfg.self_loops,
        },
        vec![],
    )
    .map_err(|e| e.to_string())?;
    let config = McmcConfig {
        burn_in_sweeps: 3,
        sweeps_per_sample: 1,
        proposals_per_sweep: Some(e),
        seed,
    };
    let (network, bench) = sample_fixed_strength_degree_bench(
        problem,
        w.degree_out.clone(),
        w.degree_in.clone(),
        config,
        DegreeTraceConfig::default(),
    )
    .map_err(|e| e.to_string())?;

    // Independent exactness checks.
    assert_eq!(network.sources.len(), e, "E exact");
    let mut ko = vec![0u32; n];
    let mut ki = vec![0u32; n];
    for ((&s, &t), _) in network
        .sources
        .iter()
        .zip(network.targets.iter())
        .zip(network.occ_nums.iter())
    {
        ko[s as usize] += 1;
        ki[t as usize] += 1;
        if !cfg.self_loops {
            assert_ne!(s, t, "self-loop");
        }
    }
    assert_eq!(ko, w.degree_out, "out-degrees exact");
    assert_eq!(ki, w.degree_in, "in-degrees exact");

    let total_proposals = bench.degree_trace.trace_attempts.max(1);
    let sweeps = 4u64; // burn_in 3 + thinning 1
    let support_per_sweep = bench.degree_trace.support_changed_returns as f64 / sweeps as f64;
    let aux_per_support = if bench.degree_trace.support_changed_returns == 0 {
        f64::NAN
    } else {
        bench.degree_trace.auxiliary_steps as f64
            / bench.degree_trace.support_changed_returns as f64
    };
    let ms_per_trace = if total_proposals == 0 {
        f64::NAN
    } else {
        bench.mcmc_time_s * 1000.0 / total_proposals as f64
    };
    let rss = peak_rss_mib()
        .map(|m| format!("{m:.0} MiB"))
        .unwrap_or_else(|| "n/a".into());
    let label = format!(
        "\n[fixed-sk bench]\n  {family:?} N={n} E={e} T/E={:.2} loops={} seed={seed}\n  \
         init: {:.3}s (extras_attempts={} extras_edges={} filler_edges={} completion={} occ1={:.3})\n  \
         mcmc: {:.2}s  proposals={total_proposals}  sweeps={sweeps}  support/sweep={support_per_sweep:.0}\n  \
         aux/support={aux_per_support:.2}  ms/trace={ms_per_trace:.2}  timeouts={}  peak_rss={rss}",
        w.strength_out.iter().sum::<OccNum>() as f64 / e as f64,
        cfg.self_loops,
        bench.direct_init_time_s,
        bench.extras_attempts,
        bench.extras_edges,
        bench.filler_edges,
        bench.completion_attempts,
        bench.occupation_one_fraction,
        bench.mcmc_time_s,
        bench.degree_trace.timeouts,
    );
    eprintln!("{label}");
    Ok(label)
}

/// ME realistic PA-geographic scale sweep: N ∈ {100, 500, 1000, 2000, 5000}.
#[test]
#[ignore]
fn fixed_sk_scale_sweep_me() {
    let mut failures = Vec::new();
    for n in [100usize, 500, 1000, 2000, 5000] {
        let cfg = pa_geo(n, 8.0, false);
        if let Err(e) = run_case(
            OccupationFamily::ME,
            &cfg,
            OccupationPattern::PaGeographic {
                events_per_edge: 8.0,
            },
            7,
        ) {
            failures.push(format!("N={n}: {e}"));
        }
    }
    assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
}

/// Family coverage at N=1000: W (realistic PA-geo) and B M=5 (Balanced12).
#[test]
#[ignore]
fn fixed_sk_family_sweep_n1000() {
    let cfg = pa_geo(1000, 8.0, false);
    let mut failures = Vec::new();
    if let Err(e) = run_case(
        OccupationFamily::W { layers: 1 },
        &cfg,
        OccupationPattern::PaGeographic {
            events_per_edge: 8.0,
        },
        7,
    ) {
        failures.push(format!("W: {e}"));
    }
    if let Err(e) = run_case(
        OccupationFamily::B { layers: 5 },
        &cfg,
        OccupationPattern::Balanced12,
        7,
    ) {
        failures.push(format!("B: {e}"));
    }
    assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
}
