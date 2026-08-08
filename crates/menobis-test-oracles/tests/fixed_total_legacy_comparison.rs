//! Level 2 validation: Gibbs vs legacy exact backends — observables.
//!
//! For medium `E, T` where exact state enumeration is infeasible but the
//! legacy rejection/DP backends still run, compare the distribution of
//! occupation observables between the legacy exact backend and the new
//! pair-Gibbs chain.  Both target the identical mathematical law, so
//! means must agree within Monte Carlo tolerance.

use menobis_core::generation::microcanonical::conditional::fixed_total::chain::sample_fixed_total;
use menobis_core::generation::microcanonical::mcmc::McmcConfig;
use menobis_core::model::family::OccupationFamily;
use menobis_test_oracles::legacy_fixed_et::{
    sample_positive_occupations, BFamily, FixedETOccupancy, MeFamily, WFamily,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Mean of (max occupation, Σ t², mean occupation) over samples.
fn mean_observables(samples: &[Vec<u64>]) -> (f64, f64, f64) {
    let n = samples.len() as f64;
    let mut sum_max = 0.0;
    let mut sum_sq = 0.0;
    let mut sum_mean = 0.0;
    for s in samples {
        sum_max += *s.iter().max().unwrap() as f64;
        sum_sq += s.iter().map(|&t| (t * t) as f64).sum::<f64>();
        sum_mean += s.iter().sum::<u64>() as f64 / s.len() as f64;
    }
    (sum_max / n, sum_sq / n, sum_mean / n)
}

/// Compare legacy backend vs Gibbs on the same (E, T) fiber.
///
/// Tolerances are relative; `rel_tol` applies to max and mean,
/// `rel_tol_sq` to Σt² (higher variance observable).
fn compare_backends<F: FixedETOccupancy>(
    family: OccupationFamily,
    legacy: &F,
    e: usize,
    t: u64,
    trials: u64,
    rel_tol: f64,
    rel_tol_sq: f64,
) {
    // Gibbs samples (independent chains per seed)
    let mut gibbs = Vec::with_capacity(trials as usize);
    for seed in 0..trials {
        let config = McmcConfig {
            burn_in_sweeps: 50,
            sweeps_per_sample: 10,
            proposals_per_sweep: None,
            seed,
        };
        gibbs.push(sample_fixed_total(family, e, t, &config).expect("gibbs failed"));
    }

    // Legacy exact samples
    let mut legacy_samples = Vec::with_capacity(trials as usize);
    for seed in 0..trials {
        let mut rng = StdRng::seed_from_u64(seed);
        legacy_samples
            .push(sample_positive_occupations(legacy, t, e, &mut rng).expect("legacy failed"));
    }

    let (g_max, g_sq, g_mean) = mean_observables(&gibbs);
    let (l_max, l_sq, l_mean) = mean_observables(&legacy_samples);

    let ok_max = (g_max - l_max).abs() <= rel_tol * l_max.max(1.0);
    let ok_sq = (g_sq - l_sq).abs() <= rel_tol_sq * l_sq.max(1.0);
    let ok_mean = (g_mean - l_mean).abs() <= rel_tol * l_mean.max(1.0);

    assert!(
        ok_max && ok_sq && ok_mean,
        "{family:?} E={e} T={t}: gibbs(max={g_max:.3},sq={g_sq:.3},mean={g_mean:.3}) \
         vs legacy(max={l_max:.3},sq={l_sq:.3},mean={l_mean:.3})",
    );
}

#[test]
fn me_gibbs_matches_legacy_e10_t30() {
    compare_backends(OccupationFamily::ME, &MeFamily, 10, 30, 4000, 0.05, 0.08);
}

#[test]
fn b_gibbs_matches_legacy_e10_t30() {
    compare_backends(
        OccupationFamily::B { layers: 5 },
        &BFamily { layers: 5 },
        10,
        30,
        4000,
        0.05,
        0.08,
    );
}

#[test]
fn w_gibbs_matches_legacy_e10_t25() {
    compare_backends(
        OccupationFamily::W { layers: 2 },
        &WFamily { layers: 2 },
        10,
        25,
        4000,
        0.05,
        0.08,
    );
}
