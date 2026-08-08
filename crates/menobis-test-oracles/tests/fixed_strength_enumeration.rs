//! Heavy: exhaustive enumeration tests for fixed-strength MCMC.
//!
//! Enumerates all occupation states for tiny (N=2) systems and
//! validates that the 4-cycle Metropolis chain visits each state
//! with the correct frequency.  Lives in the oracle crate so these
//! expensive tests don't slow down `cargo test -p menobis-core`.

use std::collections::HashMap;

use menobis_core::generation::microcanonical::mcmc::McmcConfig;
use menobis_core::generation::microcanonical::occupation_mcmc::domain::PairDomain;
use menobis_core::generation::microcanonical::occupation_mcmc::problem::FixedStrengthProblem;
use menobis_core::generation::microcanonical::occupation_mcmc::sample_fixed_strength;
use menobis_core::model::family::OccupationFamily;
use menobis_core::OccNum;

type OccupiedState = Vec<((u64, u64), OccNum)>;
type WeightedState = (OccupiedState, f64);

fn make_problem(
    family: OccupationFamily,
    out: Vec<OccNum>,
    inp: Vec<OccNum>,
    self_loops: bool,
) -> menobis_core::generation::microcanonical::occupation_mcmc::problem::ResidualStrengthProblem {
    let domain = PairDomain::Complete {
        node_count: out.len(),
        self_loops,
    };
    FixedStrengthProblem::new(family, out, inp, domain, vec![])
        .unwrap()
        .into_residual()
        .unwrap()
}

fn enumerate_me_states(s_out: &[OccNum], s_in: &[OccNum], self_loops: bool) -> Vec<WeightedState> {
    let n = s_out.len();
    let mut results = Vec::new();
    let cells: Vec<(u64, u64)> = (0..n as u64)
        .flat_map(|i| (0..n as u64).map(move |j| (i, j)))
        .filter(|&(i, j)| self_loops || i != j)
        .collect();

    fn recurse(
        idx: usize,
        cells: &[(u64, u64)],
        remaining_out: &mut [OccNum],
        remaining_in: &mut [OccNum],
        current: &mut OccupiedState,
        results: &mut Vec<WeightedState>,
    ) {
        if idx == cells.len() {
            if remaining_out.iter().all(|&s| s == 0) && remaining_in.iter().all(|&s| s == 0) {
                let log_weight: f64 = current
                    .iter()
                    .map(|&(_, occ)| -libm::lgamma((occ as f64) + 1.0))
                    .sum();
                results.push((current.clone(), log_weight));
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
        0,
        &cells,
        &mut remaining_out,
        &mut remaining_in,
        &mut current,
        &mut results,
    );
    results
}

#[test]
fn me_mcmc_enumeration_agreement_n2() {
    let s_out = vec![2u64, 2];
    let s_in = vec![2u64, 2];
    let states = enumerate_me_states(&s_out, &s_in, true);
    let total_weight: f64 = states.iter().map(|(_, w)| w.exp()).sum();

    let trials = 2000;
    let mut counts: HashMap<OccupiedState, u64> = HashMap::new();
    for seed in 0..trials {
        let prob = make_problem(OccupationFamily::ME, s_out.clone(), s_in.clone(), true);
        let config = McmcConfig {
            burn_in_sweeps: 10,
            sweeps_per_sample: 5,
            proposals_per_sweep: None,
            seed,
        };
        let (net, _backend) = sample_fixed_strength(prob, config, false).unwrap();
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

    for (state, log_weight) in &states {
        let weight = log_weight.exp();
        let expected_prob = weight / total_weight;
        let expected_count = expected_prob * trials as f64;
        let observed = counts.get(state).copied().unwrap_or(0) as f64;
        let ratio = observed / expected_count;
        assert!(
            ratio > 0.25 && ratio < 2.5,
            "state {:?}: expected {expected_count:.1}, observed {observed}, ratio {ratio:.2}",
            state
        );
    }
}

#[test]
fn me_mcmc_enumeration_agreement_n2_no_self_loops() {
    let s_out = vec![2u64, 2];
    let s_in = vec![2u64, 2];
    let states = enumerate_me_states(&s_out, &s_in, false);
    let total_weight: f64 = states.iter().map(|(_, w)| w.exp()).sum();

    let trials = 2000;
    let mut counts: HashMap<OccupiedState, u64> = HashMap::new();
    for seed in 0..trials {
        let prob = make_problem(OccupationFamily::ME, s_out.clone(), s_in.clone(), false);
        let config = McmcConfig {
            burn_in_sweeps: 10,
            sweeps_per_sample: 5,
            proposals_per_sweep: None,
            seed,
        };
        let (net, _backend) = sample_fixed_strength(prob, config, false).unwrap();
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

    for (state, log_weight) in &states {
        let weight = log_weight.exp();
        let expected_prob = weight / total_weight;
        let expected_count = expected_prob * trials as f64;
        let observed = counts.get(state).copied().unwrap_or(0) as f64;
        let ratio = observed / expected_count;
        assert!(
            ratio > 0.25 && ratio < 2.5,
            "state {:?}: expected {expected_count:.1}, observed {observed}, ratio {ratio:.2}",
            state
        );
    }
}
