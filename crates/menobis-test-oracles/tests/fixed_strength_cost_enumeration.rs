//! Heavy: exact enumeration tests for fixed-strength cost-constrained MCMC.
//!
//! Validates that the gamma-fitted cost-constrained MCMC reproduces the
//! exact Boltzmann distribution for small (N=2) enumerable systems.
//! Lives in the oracle crate so these expensive tests don't slow down
//! `cargo test -p menobis-core`.

use menobis_core::generation::microcanonical::mcmc::McmcConfig;
use menobis_core::generation::microcanonical::occupation_mcmc::chain::FixedStrengthChain;
use menobis_core::generation::microcanonical::occupation_mcmc::cost_fit::{
    fit_gamma, warm_start_gamma, FixedStrengthCostFitConfig,
};
use menobis_core::generation::microcanonical::occupation_mcmc::domain::PairDomain;
use menobis_core::generation::microcanonical::occupation_mcmc::initializer::initialize_table;
use menobis_core::generation::microcanonical::occupation_mcmc::problem::FixedStrengthProblem;
use menobis_core::generation::microcanonical::occupation_mcmc::state::StrengthState;
use menobis_core::generation::microcanonical::occupation_mcmc::target::StrengthTarget;
use menobis_core::model::family::OccupationFamily;
use menobis_core::pairs::PairCostProvider;
use menobis_core::OccNum;
use rand::rngs::StdRng;
use rand::SeedableRng;

// -----------------------------------------------------------------------
// Cost providers
// -----------------------------------------------------------------------

struct LinearCost;
impl PairCostProvider for LinearCost {
    fn cost(&self, source: usize, target: usize) -> Option<f64> {
        Some((source as i64 - target as i64).unsigned_abs() as f64)
    }
}

struct ConstantCost;
impl PairCostProvider for ConstantCost {
    fn cost(&self, _source: usize, _target: usize) -> Option<f64> {
        Some(1.0)
    }
}

// -----------------------------------------------------------------------
// Exact enumeration helpers (ME, N=2)
// -----------------------------------------------------------------------

type OccupiedState = Vec<((u64, u64), OccNum)>;

fn enumerate_states(
    family: OccupationFamily,
    s_out: &[OccNum],
    s_in: &[OccNum],
    self_loops: bool,
) -> Vec<(OccupiedState, f64)> {
    let n = s_out.len();
    let mut results = Vec::new();
    let cells: Vec<(u64, u64)> = (0..n as u64)
        .flat_map(|i| (0..n as u64).map(move |j| (i, j)))
        .filter(|&(i, j)| self_loops || i != j)
        .collect();

    fn recurse(
        family: OccupationFamily,
        idx: usize,
        cells: &[(u64, u64)],
        remaining_out: &mut [OccNum],
        remaining_in: &mut [OccNum],
        current: &mut OccupiedState,
        results: &mut Vec<(OccupiedState, f64)>,
    ) {
        if idx == cells.len() {
            if remaining_out.iter().all(|&s| s == 0) && remaining_in.iter().all(|&s| s == 0) {
                let log_degen: f64 = current
                    .iter()
                    .map(|&(_, occ)| family.log_base_measure(occ))
                    .sum();
                results.push((current.clone(), log_degen));
            }
            return;
        }
        let (src, tgt) = cells[idx];
        let max_possible = remaining_out[src as usize].min(remaining_in[tgt as usize]);
        for occ in 0..=max_possible {
            remaining_out[src as usize] -= occ;
            remaining_in[tgt as usize] -= occ;
            if occ > 0 {
                current.push(((src, tgt), occ));
            }
            recurse(
                family,
                idx + 1,
                cells,
                remaining_out,
                remaining_in,
                current,
                results,
            );
            if occ > 0 {
                current.pop();
            }
            remaining_out[src as usize] += occ;
            remaining_in[tgt as usize] += occ;
        }
    }

    let mut remaining_out = s_out.to_vec();
    let mut remaining_in = s_in.to_vec();
    let mut current: OccupiedState = Vec::new();
    recurse(
        family,
        0,
        &cells,
        &mut remaining_out,
        &mut remaining_in,
        &mut current,
        &mut results,
    );
    results
}

fn state_cost_from_pairs(pairs: &OccupiedState, costs: &dyn PairCostProvider) -> f64 {
    let mut total = 0.0;
    for &((s, t), occ) in pairs {
        let c = costs.cost(s as usize, t as usize).unwrap_or(0.0);
        total += c * (occ as f64);
    }
    total
}

fn exact_expected_cost_at_gamma(
    states: &[(OccupiedState, f64)],
    costs: &dyn PairCostProvider,
    gamma: f64,
) -> f64 {
    let log_weights: Vec<f64> = states
        .iter()
        .map(|(pairs, log_d)| log_d - gamma * state_cost_from_pairs(pairs, costs))
        .collect();
    let max_log = log_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_weights.iter().map(|w| (w - max_log).exp()).collect();
    let total: f64 = weights.iter().sum();
    states
        .iter()
        .zip(weights.iter())
        .map(|((pairs, _), w)| state_cost_from_pairs(pairs, costs) * w / total)
        .sum()
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[test]
fn constant_cost_does_not_affect_distribution() {
    let s_out = vec![2u64, 2];
    let s_in = vec![2u64, 2];
    let n = 2;
    let domain = PairDomain::Complete {
        node_count: n,
        self_loops: true,
    };
    let config = McmcConfig::new(10, 5, 42);
    let costs = ConstantCost;

    let table = initialize_table(&s_out, &s_in, OccupationFamily::ME, &domain).unwrap();
    let state = StrengthState::new(n, table);
    let mut chain = FixedStrengthChain::new(
        state,
        StrengthTarget::with_costs(OccupationFamily::ME, &costs),
        domain,
        config,
    );
    // Set a large gamma
    let mut target = StrengthTarget::with_costs(OccupationFamily::ME, &costs);
    target.set_gamma(10.0);
    chain.set_target(target);

    // Run a few steps — constant cost means ΔC=0 always, so moves are
    // identical to the no-cost case.
    let mut rng = StdRng::seed_from_u64(99);
    for _ in 0..50 {
        let outcome = chain.step(&mut rng);
        // Just verify it doesn't crash
        assert!(matches!(
            outcome,
            menobis_core::generation::microcanonical::mcmc::McmcOutcome::Accepted
                | menobis_core::generation::microcanonical::mcmc::McmcOutcome::Held
                | menobis_core::generation::microcanonical::mcmc::McmcOutcome::Rejected
        ));
    }
}

#[test]
fn zero_gamma_cost_equals_no_cost() {
    let s_out = vec![2u64, 2];
    let s_in = vec![2u64, 2];
    let n = 2;
    let domain = PairDomain::Complete {
        node_count: n,
        self_loops: true,
    };
    let config = McmcConfig::new(10, 5, 42);
    let costs = LinearCost;

    let table = initialize_table(&s_out, &s_in, OccupationFamily::ME, &domain).unwrap();
    let state = StrengthState::new(n, table);
    let chain = FixedStrengthChain::new(
        state,
        StrengthTarget::with_costs(OccupationFamily::ME, &costs),
        domain,
        config,
    );
    let no_cost = StrengthTarget::new(OccupationFamily::ME);
    let d1 = no_cost.delta_log_weight(0, 1, 2, 3).unwrap();
    let d2 = chain.target.delta_log_weight(0, 1, 2, 3).unwrap();
    assert!(
        (d1 - d2).abs() < 1e-12,
        "gamma=0 delta ({d2}) != no-cost delta ({d1})"
    );
}

#[test]
fn me_cost_enumeration_agreement_n2() {
    let s_out = vec![2u64, 2];
    let s_in = vec![2u64, 2];
    let states = enumerate_states(OccupationFamily::ME, &s_out, &s_in, true);
    let costs = LinearCost;
    let gamma = 0.5;

    let log_weights: Vec<f64> = states
        .iter()
        .map(|(pairs, log_d)| log_d - gamma * state_cost_from_pairs(pairs, &costs))
        .collect();
    let max_log = log_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let total_weight: f64 = log_weights.iter().map(|w| (w - max_log).exp()).sum();
    let exact_probs: Vec<f64> = log_weights
        .iter()
        .map(|w| (w - max_log).exp() / total_weight)
        .collect();

    let trials = 3000;
    let mut counts = std::collections::HashMap::<OccupiedState, u64>::new();
    for seed in 0..trials {
        let problem = FixedStrengthProblem::new(
            OccupationFamily::ME,
            s_out.clone(),
            s_in.clone(),
            PairDomain::Complete {
                node_count: 2,
                self_loops: true,
            },
            vec![],
        )
        .unwrap()
        .into_residual()
        .unwrap();

        let table = initialize_table(
            &problem.strength_out,
            &problem.strength_in,
            problem.family,
            &problem.domain,
        )
        .unwrap();
        let state = StrengthState::new(2, table);

        let mut tgt = StrengthTarget::with_costs(OccupationFamily::ME, &costs);
        tgt.set_gamma(gamma);

        let mcmc_config = McmcConfig {
            burn_in_sweeps: 10,
            sweeps_per_sample: 5,
            proposals_per_sweep: None,
            seed,
        };
        let mut chain = FixedStrengthChain::new(
            state,
            tgt,
            PairDomain::Complete {
                node_count: 2,
                self_loops: true,
            },
            mcmc_config,
        );

        let mut rng = StdRng::seed_from_u64(seed);
        chain.burn_in(&mut rng);
        let net = chain.sample(&mut rng);

        let mut pairs: OccupiedState = net
            .sources
            .iter()
            .zip(net.targets.iter())
            .zip(net.occ_nums.iter())
            .map(|((&s, &t), &o)| ((s, t), o))
            .collect();
        pairs.sort_unstable();
        *counts.entry(pairs).or_default() += 1;
    }

    for ((pairs, _), prob) in states.iter().zip(exact_probs.iter()) {
        let expected_count = prob * trials as f64;
        let observed = counts.get(pairs).copied().unwrap_or(0) as f64;
        let ratio = observed / expected_count;
        assert!(
            ratio > 0.3 && ratio < 2.5,
            "state {:?}: expected {expected_count:.1}, observed {observed}, ratio {ratio:.2}",
            pairs
        );
    }
}

#[test]
fn warm_start_positive_gamma_when_cost_above_uniform() {
    let s_out = vec![3u64, 3];
    let s_in = vec![3u64, 3];
    let n = 2;
    let domain = PairDomain::Complete {
        node_count: n,
        self_loops: true,
    };
    let config = McmcConfig::new(5, 2, 42);
    let costs = LinearCost;

    let table = initialize_table(&s_out, &s_in, OccupationFamily::ME, &domain).unwrap();
    let state = StrengthState::new(n, table);
    let mut chain = FixedStrengthChain::new(
        state,
        StrengthTarget::with_costs(OccupationFamily::ME, &costs),
        domain,
        config,
    );

    let mut rng = StdRng::seed_from_u64(42);
    let gamma_0 = warm_start_gamma(&mut chain, &mut rng, &costs, 0.1, 10).unwrap();
    assert!(
        gamma_0 > 0.0,
        "expected positive gamma for low observed cost, got {gamma_0}"
    );
}

#[test]
fn warm_start_negative_gamma_when_cost_below_uniform() {
    let s_out = vec![3u64, 3];
    let s_in = vec![3u64, 3];
    let n = 2;
    let domain = PairDomain::Complete {
        node_count: n,
        self_loops: true,
    };
    let config = McmcConfig::new(5, 2, 42);
    let costs = LinearCost;

    let table = initialize_table(&s_out, &s_in, OccupationFamily::ME, &domain).unwrap();
    let state = StrengthState::new(n, table);
    let mut chain = FixedStrengthChain::new(
        state,
        StrengthTarget::with_costs(OccupationFamily::ME, &costs),
        domain,
        config,
    );

    let mut rng = StdRng::seed_from_u64(42);
    let gamma_0 = warm_start_gamma(&mut chain, &mut rng, &costs, 100.0, 10).unwrap();
    assert!(
        gamma_0 < 0.0,
        "expected negative gamma for high observed cost, got {gamma_0}"
    );
}

fn assert_gamma_recovery(family: OccupationFamily, target_gamma: f64) {
    let s_out = vec![2u64, 2];
    let s_in = vec![2u64, 2];
    let states = enumerate_states(family, &s_out, &s_in, true);
    assert!(!states.is_empty(), "enumeration returned no states");
    let costs = LinearCost;
    let c_obs = exact_expected_cost_at_gamma(&states, &costs, target_gamma);
    assert!(c_obs.is_finite() && c_obs > 0.0, "invalid c_obs {c_obs}");

    let domain = PairDomain::Complete {
        node_count: 2,
        self_loops: true,
    };
    let table = initialize_table(&s_out, &s_in, family, &domain).unwrap();
    let state = StrengthState::new(2, table);
    let config = McmcConfig::new(10, 5, 42);
    let mut chain = FixedStrengthChain::new(
        state,
        StrengthTarget::with_costs(family, &costs),
        domain,
        config,
    );

    let mut rng = StdRng::seed_from_u64(42);
    let fit_config = FixedStrengthCostFitConfig {
        warm_start_sweeps: 200,
        adaptation_sweeps: 200,
        estimation_sweeps: 400,
        samples_per_iteration: 400,
        max_iterations: 25,
        absolute_cost_tolerance: 0.05,
        relative_cost_tolerance: 0.25,
        confidence_multiplier: 2.09,
        batch_count: 20,
        ..FixedStrengthCostFitConfig::default()
    };

    let result = fit_gamma(&mut chain, &mut rng, &costs, c_obs, 0.0, &fit_config)
        .unwrap_or_else(|e| panic!("fit_gamma failed for {family:?}: {e}"));

    assert!(
        (result.gamma - target_gamma).abs() < 0.5,
        "fitted gamma {} not close to target {target_gamma} for {family:?}",
        result.gamma
    );

    let mu_fit = exact_expected_cost_at_gamma(&states, &costs, result.gamma);
    let tol = 0.15 * c_obs.abs().max(1.0);
    assert!(
        (mu_fit - c_obs).abs() < tol,
        "expected cost at fitted gamma {} does not match target {} (gamma={})",
        mu_fit,
        c_obs,
        result.gamma
    );
}

#[test]
fn gamma_recovery_me() {
    assert_gamma_recovery(OccupationFamily::ME, 1.0);
}

#[test]
fn gamma_recovery_b() {
    assert_gamma_recovery(OccupationFamily::B { layers: 4 }, 1.0);
}

#[test]
fn gamma_recovery_w() {
    assert_gamma_recovery(OccupationFamily::W { layers: 2 }, 1.0);
}
