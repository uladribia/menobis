//! Fixed-total pair-Gibbs stationarity vs exact enumeration.
//!
//! Heavy validation: compare empirical Gibbs state probabilities against
//! exact enumeration of the fixed-(E,T) fiber (total variation distance).
//! These tests are intentionally more expensive than the fast unit tests
//! in `menobis-core` and live in the oracle crate by design.

use menobis_core::generation::microcanonical::conditional::fixed_total::chain::sample_fixed_total;
use menobis_core::generation::microcanonical::mcmc::McmcConfig;
use menobis_core::model::family::OccupationFamily;
use menobis_test_oracles::enumeration::{enumerate_fixed_total, normalize_states, WeightedState};
use std::collections::HashMap;

/// Total variation distance between the empirical distribution over the
/// enumerated states and the exact probabilities.
fn tv_distance(
    observed: &HashMap<Vec<u64>, u64>,
    states: &[WeightedState],
    exact: &[f64],
    trials: f64,
) -> f64 {
    let mut tv = 0.0;
    for (state, p) in states.iter().zip(exact) {
        let emp = observed.get(&state.occupations).copied().unwrap_or(0) as f64 / trials;
        tv += (emp - p).abs();
    }
    tv / 2.0
}

/// Run independent Gibbs chains (one per seed) and check the empirical
/// state distribution against exact enumeration.
fn run_stationarity_check(family: OccupationFamily, e: usize, t: u64, trials: u64, tv_max: f64) {
    let states = enumerate_fixed_total(family, e, t);
    let exact = normalize_states(&states);

    let mut counts: HashMap<Vec<u64>, u64> = HashMap::new();
    for seed in 0..trials {
        let config = McmcConfig {
            burn_in_sweeps: 20,
            sweeps_per_sample: 5,
            proposals_per_sweep: None,
            seed,
        };
        let occ = sample_fixed_total(family, e, t, &config).expect("sampling failed");
        *counts.entry(occ).or_insert(0) += 1;
    }

    let tv = tv_distance(&counts, &states, &exact, trials as f64);
    assert!(
        tv < tv_max,
        "{family:?} E={e} T={t}: TV={tv:.4} exceeds {tv_max}"
    );
}

#[test]
fn me_stationarity() {
    run_stationarity_check(OccupationFamily::ME, 3, 6, 12_000, 0.03);
}

#[test]
fn b_stationarity() {
    run_stationarity_check(OccupationFamily::B { layers: 4 }, 3, 8, 12_000, 0.03);
}

#[test]
fn b_near_saturation_single_state() {
    // B(3), E=3, T=9 → all cells at capacity → exactly one state.
    run_stationarity_check(OccupationFamily::B { layers: 3 }, 3, 9, 2_000, 1e-12);
}

#[test]
fn w_stationarity() {
    run_stationarity_check(OccupationFamily::W { layers: 2 }, 3, 6, 12_000, 0.03);
}

#[test]
fn w_m1_uniform_stationarity() {
    // W(1): all compositions equally weighted → uniform over states.
    run_stationarity_check(OccupationFamily::W { layers: 1 }, 3, 6, 12_000, 0.03);
}
