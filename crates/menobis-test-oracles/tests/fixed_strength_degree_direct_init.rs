//! Gate C (plan §36–§40): N=1000 extras-first constructor gates.
//!
//! The extras-first exact-(s,k) constructor
//! (`initialize_exact_sk_extras_first`) must construct an exact residual
//! `(s,k)` state at N=1000 from **witness-derived constraints only** (the
//! witness table/support is never passed into the constructor) on:
//!
//! - C1/Gate C1: realistic ME PA-geographic (T/E=8, heterogeneous);
//! - C2: Balanced12 (T/E=1.5, heterogeneous);
//! - C3: the uniform regression grid (ME d∈{4,8,16} × {1,2,5,10}, W d=8,
//!   B M=5 d=8);
//! - C4: structural variants (loops on, CompleteMinus positive fixed
//!   pairs, CompleteMinus zero fixed pairs);
//! - §40: heterogeneous B (M=5 Balanced12) and W (PA-geographic).
//!
//! Requirements per case: exact `s_out`, `s_in`, `k_out`, `k_in`, `E`,
//! domain exactness, `D = 0`, B capacity, and (for structural variants)
//! fixed-pair exclusion — with diagnostics (extras attempts/edges,
//! filler edges, completion attempts, occupation-1 fraction, wall time).
//!
//! The constructor is combinatorial (§2 of the recovery plan): no
//! detailed balance is claimed; exactness is the gate.  All tests are
//! `#[ignore]`d (N=1000), run with
//! `--release -- --ignored --nocapture`.

use std::collections::HashSet;

use menobis_core::generation::microcanonical::occupation_mcmc::fixed_degree_init::{
    initialize_exact_sk_extras_first, ExactSkInitConfig,
};
use menobis_core::generation::microcanonical::occupation_mcmc::problem::FixedStrengthProblem;
use menobis_core::model::family::OccupationFamily;
use menobis_core::OccNum;
use menobis_test_oracles::pa_geographic::{pa_geographic_witness, OccupationPattern, PaGeoConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn pa_geo(n: usize, d: f64, sl: bool) -> PaGeoConfig {
    PaGeoConfig {
        node_count: n,
        average_degree: d,
        self_loops: sl,
        seed: 42,
    }
}

/// Exact (s,k) extras-first residual gate runner: derive s,k from a
/// PA-geo witness, residualize (with fixed pairs), construct, and
/// independently verify every invariant from the public state API.
/// Returns a one-line diagnostics label.
fn run_init_gate(
    family: OccupationFamily,
    cfg: &PaGeoConfig,
    pattern: OccupationPattern,
    fixed: Vec<(u64, u64, OccNum)>,
    seed: u64,
) -> Result<String, String> {
    let w = pa_geographic_witness(cfg, pattern);
    let n = w.n;
    // Full problem (Complete domain + fixed pairs) -> residual.
    let full = FixedStrengthProblem::new(
        family,
        w.strength_out.clone(),
        w.strength_in.clone(),
        menobis_core::generation::microcanonical::occupation_mcmc::domain::PairDomain::Complete {
            node_count: n,
            self_loops: cfg.self_loops,
        },
        fixed.clone(),
    )
    .map_err(|e| e.to_string())?;
    let residual = full.clone().into_residual().map_err(|e| e.to_string())?;

    // Residual degree target (subtract fixed pairs).
    let mut k_out = w.degree_out.clone();
    let mut k_in = w.degree_in.clone();
    for &(s, t, occ) in &fixed {
        if occ > 0 {
            k_out[s as usize] -= 1;
            k_in[t as usize] -= 1;
        }
    }

    let t0 = std::time::Instant::now();
    let mut rng = StdRng::seed_from_u64(seed);
    let (state, diag) = initialize_exact_sk_extras_first(
        &residual,
        &k_out,
        &k_in,
        &mut rng,
        &ExactSkInitConfig::default(),
    )
    .map_err(|e| e.to_string())?;
    let wall = t0.elapsed().as_secs_f64();

    // Independent verification (public API only).
    if state.out_strengths != residual.strength_out {
        return Err("out strengths mismatch".into());
    }
    if state.in_strengths != residual.strength_in {
        return Err("in strengths mismatch".into());
    }
    let mut ko = vec![0u32; n];
    let mut ki = vec![0u32; n];
    let mut seen = HashSet::with_capacity(state.occupied_count());
    let cap = match family {
        OccupationFamily::B { layers } => layers as OccNum,
        _ => OccNum::MAX,
    };
    for pair in state.iter_occupied() {
        let ((s, t), o) = pair;
        if !seen.insert((s, t)) {
            return Err("duplicate coordinate".into());
        }
        if o == 0 {
            return Err("zero occupation".into());
        }
        if !cfg.self_loops && s == t {
            return Err("self-loop".into());
        }
        if o > cap {
            return Err("B capacity".into());
        }
        if !residual.domain.is_admissible(s, t) {
            return Err("inadmissible pair (fixed coordinate reoccupied)".into());
        }
        ko[s as usize] += 1;
        ki[t as usize] += 1;
    }
    if ko != k_out || ki != k_in {
        return Err("degree mismatch".into());
    }
    if state.occupied_count() != k_out.iter().map(|&k| k as usize).sum::<usize>() {
        return Err("E mismatch".into());
    }
    // D_raw = 0 by exact degrees.
    let d_raw: u64 = (0..n as usize)
        .map(|i| (ko[i] as u64).abs_diff(k_out[i] as u64) + (ki[i] as u64).abs_diff(k_in[i] as u64))
        .sum();
    if d_raw != 0 {
        return Err(format!("D = {d_raw} != 0"));
    }

    let label = format!(
        "{family:?} N={n} d={} T/E={:.2} loops={} fixed={} | \
         attempts={} extras_failures={} extras_edges={} filler_edges={} \
         completion_attempts={} completion_failures={} occ1={} ({:.3}) \
         residual_total={} wall={wall:.2}s",
        cfg.average_degree,
        pattern.t_over_e(),
        cfg.self_loops,
        fixed.len(),
        diag.extras_attempts,
        diag.extras_failed_attempts,
        diag.extras_edges,
        diag.filler_edges,
        diag.completion_attempts,
        diag.completion_failed_attempts,
        diag.occupation_one_edges,
        diag.occupation_one_fraction,
        diag.residual_total,
    );
    eprintln!("[init gate] {label}");
    Ok(label)
}

/// Deterministic fixed pairs: take the first `count` witness support
/// edges (sorted order) with their occupations (positive fixed pairs).
fn positive_fixed_pairs(
    witness_table: &[((u64, u64), OccNum)],
    count: usize,
) -> Vec<(u64, u64, OccNum)> {
    witness_table[..count.min(witness_table.len())]
        .iter()
        .map(|&((s, t), o)| (s, t, o))
        .collect()
}

/// Deterministic zero-occupation fixed pairs: `count` pairs absent from
/// the witness support.
fn zero_fixed_pairs(
    w: &menobis_test_oracles::pa_geographic::PaGeoWitness,
    count: usize,
) -> Vec<(u64, u64, OccNum)> {
    let support: HashSet<(u64, u64)> = w.table.iter().map(|&((s, t), _)| (s, t)).collect();
    let mut out = Vec::new();
    let n = w.n as u64;
    for i in 0u64..n {
        for j in 0u64..n {
            if out.len() >= count {
                return out;
            }
            if !support.contains(&(i, j)) && (w.n == 1 || i != j) {
                out.push((i, j, 0));
            }
        }
    }
    out
}

/// Gate C1 + C2 (§36–§37): realistic ME PA-geographic and Balanced12 at
/// N=1000 must **succeed** (this test previously pinned the support-first
/// blocker; the extras-first constructor replaces it, plan Part I §62).
#[test]
#[ignore]
fn n1000_extras_first_initialization() {
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
        match run_init_gate(OccupationFamily::ME, &cfg, pattern, vec![], 7) {
            Ok(label) => eprintln!("[{tag}] OK {label}"),
            Err(e) => failures.push(format!("[{tag}] {e}")),
        }
    }

    // All-1 (T/E = 1) has zero residual: the constructor must succeed
    // with the exact all-ones state (residual total 0, §20).
    let w = pa_geographic_witness(&cfg, OccupationPattern::Uniform(1));
    let full = FixedStrengthProblem::new(
        OccupationFamily::ME,
        w.strength_out.clone(),
        w.strength_in.clone(),
        menobis_core::generation::microcanonical::occupation_mcmc::domain::PairDomain::Complete {
            node_count: w.n,
            self_loops: false,
        },
        vec![],
    )
    .unwrap();
    let residual = full.into_residual().unwrap();
    let mut rng = StdRng::seed_from_u64(7);
    let (state, diag) = initialize_exact_sk_extras_first(
        &residual,
        &w.degree_out,
        &w.degree_in,
        &mut rng,
        &ExactSkInitConfig::default(),
    )
    .unwrap();
    assert_eq!(diag.residual_total, 0);
    assert!(diag.extras_edges == 0, "no residual => no extras");
    assert!(
        state.iter_occupied().all(|(_, o)| o == 1),
        "all-ones state expected"
    );
    assert_eq!(state.out_strengths, w.strength_out);
    assert_eq!(state.in_strengths, w.strength_in);

    assert!(
        failures.is_empty(),
        "Gate C1/C2 failures:\n{}",
        failures.join("\n")
    );
}

/// Gate C3 (§38): uniform regression grid — ME d∈{4,8,16} × T/E∈{1,2,5,10};
/// W d=8; B M=5 d=8.  Uniform extras are trivially transportable; this
/// grid must stay green with the extras-first constructor.
#[test]
#[ignore]
fn n1000_constructor_stress_grid() {
    let mut failures = Vec::new();
    // ME: d × T/E
    for d in [4.0, 8.0, 16.0] {
        let cfg = pa_geo(1000, d, false);
        for c in [1u64, 2, 5, 10] {
            let pattern = OccupationPattern::Uniform(c);
            if let Err(e) = run_init_gate(OccupationFamily::ME, &cfg, pattern, vec![], 7) {
                failures.push(format!("ME d={d} T/E={c}: {e}"));
            }
        }
    }
    // W: d=8, T/E∈{1,2,5}
    {
        let cfg = pa_geo(1000, 8.0, false);
        for c in [1u64, 2, 5] {
            let pattern = OccupationPattern::Uniform(c);
            if let Err(e) =
                run_init_gate(OccupationFamily::W { layers: 1 }, &cfg, pattern, vec![], 7)
            {
                failures.push(format!("W T/E={c}: {e}"));
            }
        }
    }
    // B M=5: d=8, T/E∈{1,2,3,5}
    {
        let cfg = pa_geo(1000, 8.0, false);
        for c in [1u64, 2, 3, 5] {
            let pattern = OccupationPattern::Uniform(c);
            if let Err(e) =
                run_init_gate(OccupationFamily::B { layers: 5 }, &cfg, pattern, vec![], 7)
            {
                failures.push(format!("B M=5 T/E={c}: {e}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "stress grid failures:\n{}",
        failures.join("\n")
    );
}

/// Gate C4 (§39): structural variants at N=1000 — loops on, CompleteMinus
/// positive fixed pairs, CompleteMinus zero fixed pairs — all must
/// construct exactly (this test previously pinned exhaustion).
#[test]
#[ignore]
fn n1000_structural_variants() {
    let mut failures = Vec::new();

    // Loops allowed (the support includes the diagonal).
    let cfg_loops = pa_geo(1000, 8.0, true);
    match run_init_gate(
        OccupationFamily::ME,
        &cfg_loops,
        OccupationPattern::PaGeographic {
            events_per_edge: 8.0,
        },
        vec![],
        7,
    ) {
        Ok(label) => eprintln!("[loops-on] OK {label}"),
        Err(e) => failures.push(format!("loops-on: {e}")),
    }

    // CompleteMinus with positive fixed pairs (15% of the support).
    let cfg = pa_geo(1000, 8.0, false);
    let w = pa_geographic_witness(
        &cfg,
        OccupationPattern::PaGeographic {
            events_per_edge: 8.0,
        },
    );
    let fixed_pos = positive_fixed_pairs(&w.table, w.table.len() / 6);
    match run_init_gate(
        OccupationFamily::ME,
        &cfg,
        OccupationPattern::PaGeographic {
            events_per_edge: 8.0,
        },
        fixed_pos,
        7,
    ) {
        Ok(label) => eprintln!("[positive-fixed] OK {label}"),
        Err(e) => failures.push(format!("positive-fixed: {e}")),
    }

    // CompleteMinus with zero fixed pairs (forbidden coordinates absent
    // from the support).
    let fixed_zero = zero_fixed_pairs(&w, 1000);
    match run_init_gate(
        OccupationFamily::ME,
        &cfg,
        OccupationPattern::PaGeographic {
            events_per_edge: 8.0,
        },
        fixed_zero,
        7,
    ) {
        Ok(label) => eprintln!("[zero-fixed] OK {label}"),
        Err(e) => failures.push(format!("zero-fixed: {e}")),
    }

    assert!(
        failures.is_empty(),
        "structural-variant failures:\n{}",
        failures.join("\n")
    );
}

/// §40: heterogeneous B and W at N=1000 — B M=5 + Balanced12 (occupations
/// 1/2, well inside capacity M) and a representative W PA-geographic case.
#[test]
#[ignore]
fn n1000_extras_first_heterogeneous_b_w() {
    let cfg = pa_geo(1000, 8.0, false);
    let mut failures = Vec::new();
    // B M=5, Balanced12 (T/E = 1.5; every occupation ≤ 2 ≤ M).
    match run_init_gate(
        OccupationFamily::B { layers: 5 },
        &cfg,
        OccupationPattern::Balanced12,
        vec![],
        7,
    ) {
        Ok(label) => eprintln!("[B M=5 Balanced12] OK {label}"),
        Err(e) => failures.push(format!("B M=5 Balanced12: {e}")),
    }
    // W M=1 (geometric), realistic PA-geographic (T/E = 8).
    match run_init_gate(
        OccupationFamily::W { layers: 1 },
        &cfg,
        OccupationPattern::PaGeographic {
            events_per_edge: 8.0,
        },
        vec![],
        7,
    ) {
        Ok(label) => eprintln!("[W realistic] OK {label}"),
        Err(e) => failures.push(format!("W realistic: {e}")),
    }
    assert!(
        failures.is_empty(),
        "heterogeneous B/W failures:\n{}",
        failures.join("\n")
    );
}
