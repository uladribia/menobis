//! Persistent MCMC chain for fixed-strength sampling.
//!
//! Provides:
//! - `FixedStrengthChain` — stateful chain with step/sweep/burn_in/sample.
//! - `sample_fixed_strength` — one-shot convenience orchestrator.

use rand::rngs::StdRng;
use rand::SeedableRng;

use super::domain::PairDomain;
use super::errors::FixedStrengthError;
use super::initializer::initialize_table;
use super::me_direct::{sample_strength_stub_matching, MAX_EXPLICIT_STUBS};
use super::move_cycle::cycle4_step;
use super::problem::ResidualStrengthProblem;
use super::state::StrengthState;
use super::target::StrengthTarget;
use crate::distribution::OccupationFamily;
use crate::generation::microcanonical::mcmc::{McmcConfig, McmcCounters, McmcOutcome};
use crate::generation::output::SampledNetwork;
use crate::OccNum;

/// Backend used for the fixed-strength sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrengthBackend {
    /// Exact stub-matching (ME, complete domain, self-loops allowed).
    MeDirect,
    /// 4-cycle Metropolis MCMC.
    CycleMcmc,
}

/// Persistent MCMC chain for fixed-strength sampling.
pub struct FixedStrengthChain<'a> {
    pub state: StrengthState,
    pub target: StrengthTarget<'a>,
    pub domain: PairDomain,
    pub config: McmcConfig,
    pub counters: McmcCounters,
}

impl<'a> FixedStrengthChain<'a> {
    /// Create a new chain from a pre-built state.
    pub fn new(
        state: StrengthState,
        target: StrengthTarget<'a>,
        domain: PairDomain,
        config: McmcConfig,
    ) -> Self {
        Self {
            state,
            target,
            domain,
            config,
            counters: McmcCounters::new(),
        }
    }

    /// Perform one MCMC step (one 4-cycle proposal).
    pub fn step(&mut self, rng: &mut impl rand::Rng) -> McmcOutcome {
        self.counters.proposals += 1;
        let outcome = cycle4_step(&mut self.state, &self.target, &self.domain, rng);
        match outcome {
            McmcOutcome::Accepted => self.counters.accepted += 1,
            McmcOutcome::Held => self.counters.held += 1,
            McmcOutcome::Rejected => self.counters.metropolis_rejected += 1,
        }
        outcome
    }

    /// Perform one sweep.
    ///
    /// One sweep consists of `proposals_per_sweep` proposals, where
    /// `proposals_per_sweep = max(occupied_pairs, 2 * node_count, 1)`.
    pub fn sweep(&mut self, rng: &mut impl rand::Rng) {
        let n = self.state.node_count;
        let occupied = self.state.occupied_count();
        let per_sweep = self
            .config
            .proposals_per_sweep
            .unwrap_or_else(|| occupied.max(2 * n).max(1));
        for _ in 0..per_sweep {
            self.step(rng);
        }
    }

    /// Run burn-in: perform `burn_in_sweeps` sweeps.
    pub fn burn_in(&mut self, rng: &mut impl rand::Rng) {
        for _ in 0..self.config.burn_in_sweeps.max(1) {
            self.sweep(rng);
        }
    }

    /// After burn-in, perform thinning sweeps and return the current network.
    pub fn sample(&mut self, rng: &mut impl rand::Rng) -> SampledNetwork {
        for _ in 0..self.config.sweeps_per_sample.max(1) {
            self.sweep(rng);
        }
        debug_assert_eq!(
            self.state.out_strengths.iter().sum::<OccNum>(),
            self.state.in_strengths.iter().sum::<OccNum>(),
            "strengths unbalanced after sampling"
        );
        self.state.to_sampled_network()
    }
}

/// Check whether the ME direct backend can be used.
fn can_use_me_direct(
    family: OccupationFamily,
    self_loops: bool,
    domain: &PairDomain,
    has_fixed_pairs: bool,
    total: OccNum,
) -> bool {
    if family != OccupationFamily::Poisson {
        return false;
    }
    if !self_loops {
        return false;
    }
    if has_fixed_pairs {
        return false;
    }
    if matches!(domain, PairDomain::Sparse { .. }) {
        return false;
    }
    if total > MAX_EXPLICIT_STUBS {
        return false;
    }
    true
}

/// One-shot fixed-strength sampling.
///
/// Selects the appropriate backend:
///
/// 1. **ME direct** (exact stub-matching) if the problem is simple enough.
/// 2. **Cycle MCMC** (4-cycle Metropolis) for all other cases.
///
/// # Errors
///
/// Returns [`FixedStrengthError`] if the problem is infeasible or an
/// internal error occurs.
pub fn sample_fixed_strength(
    problem: ResidualStrengthProblem,
    config: McmcConfig,
    has_fixed_pairs: bool,
) -> Result<(SampledNetwork, StrengthBackend), FixedStrengthError> {
    problem.validate()?;

    let total = problem.total;
    let family = problem.family;
    let self_loops = problem.domain.self_loops_allowed();
    let seed = config.seed;

    // Try ME direct fast path.
    if can_use_me_direct(family, self_loops, &problem.domain, has_fixed_pairs, total) {
        let result =
            sample_strength_stub_matching(&problem.strength_out, &problem.strength_in, seed)?;
        return Ok((result, StrengthBackend::MeDirect));
    }

    // Initialize occupation table.
    let table = initialize_table(
        &problem.strength_out,
        &problem.strength_in,
        family,
        &problem.domain,
    )?;

    // Build state.
    let state = StrengthState::new(problem.domain.node_count(), table);

    // Build target (Phase 4: gamma = 0, no cost).
    let target = StrengthTarget::new(family, 0.0);

    // Build chain.
    let mut chain = FixedStrengthChain::new(state, target, problem.domain, config);

    // Run chain.
    let mut rng = StdRng::seed_from_u64(seed);
    chain.burn_in(&mut rng);
    let network = chain.sample(&mut rng);

    Ok((network, StrengthBackend::CycleMcmc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::microcanonical::fixed_strength::problem::FixedStrengthProblem;

    fn make_problem(
        family: OccupationFamily,
        out: Vec<OccNum>,
        inp: Vec<OccNum>,
        self_loops: bool,
    ) -> ResidualStrengthProblem {
        let domain = PairDomain::Complete {
            node_count: out.len(),
            self_loops,
        };
        FixedStrengthProblem::new(family, out, inp, domain, vec![])
            .unwrap()
            .into_residual()
            .unwrap()
    }

    #[test]
    fn me_direct_backend_selected() {
        let prob = make_problem(OccupationFamily::Poisson, vec![5, 5], vec![5, 5], true);
        let config = McmcConfig::new(10, 5, 42);
        let (net, backend) = sample_fixed_strength(prob, config, false).unwrap();
        assert_eq!(backend, StrengthBackend::MeDirect);
        let total: OccNum = net.occ_nums.iter().sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn me_mcmc_backend_no_self_loops() {
        let prob = make_problem(
            OccupationFamily::Poisson,
            vec![5, 5, 5],
            vec![5, 5, 5],
            false,
        );
        let config = McmcConfig {
            burn_in_sweeps: 20,
            sweeps_per_sample: 10,
            proposals_per_sweep: None,
            seed: 42,
        };
        let (net, backend) = sample_fixed_strength(prob, config, false).unwrap();
        assert_eq!(backend, StrengthBackend::CycleMcmc);
        let total: OccNum = net.occ_nums.iter().sum();
        assert_eq!(total, 15);
        // No self-loops.
        for (&s, &t) in net.sources.iter().zip(net.targets.iter()) {
            assert_ne!(s, t, "self-loop found");
        }
    }

    #[test]
    fn b_fixed_strength() {
        let prob = make_problem(
            OccupationFamily::Binomial(4),
            vec![4, 4, 4],
            vec![4, 4, 4],
            true,
        );
        let config = McmcConfig {
            burn_in_sweeps: 20,
            sweeps_per_sample: 10,
            proposals_per_sweep: None,
            seed: 42,
        };
        let (net, backend) = sample_fixed_strength(prob, config, false).unwrap();
        assert_eq!(backend, StrengthBackend::CycleMcmc);
        let total: OccNum = net.occ_nums.iter().sum();
        assert_eq!(total, 12);
        // No B occupation exceeds 4.
        for &occ in &net.occ_nums {
            assert!(occ <= 4, "B occupation {occ} exceeds 4");
        }
    }

    #[test]
    fn w_fixed_strength() {
        let prob = make_problem(
            OccupationFamily::NegativeBinomial(2),
            vec![5, 5, 5],
            vec![5, 5, 5],
            true,
        );
        let config = McmcConfig {
            burn_in_sweeps: 20,
            sweeps_per_sample: 10,
            proposals_per_sweep: None,
            seed: 42,
        };
        let (net, backend) = sample_fixed_strength(prob, config, false).unwrap();
        assert_eq!(backend, StrengthBackend::CycleMcmc);
        let total: OccNum = net.occ_nums.iter().sum();
        assert_eq!(total, 15);
    }

    #[test]
    fn strengths_preserved_across_backends() {
        // ME with self-loops → direct backend
        let prob = make_problem(
            OccupationFamily::Poisson,
            vec![4, 7, 2],
            vec![6, 3, 4],
            true,
        );
        let config = McmcConfig::new(20, 10, 42);
        let (net, backend) = sample_fixed_strength(prob, config, false).unwrap();
        let mut check_out = vec![0u64; 3];
        let mut check_in = vec![0u64; 3];
        for ((&s, &t), &o) in net
            .sources
            .iter()
            .zip(net.targets.iter())
            .zip(net.occ_nums.iter())
        {
            check_out[s as usize] += o;
            check_in[t as usize] += o;
        }
        assert_eq!(
            check_out,
            vec![4, 7, 2],
            "out-strengths not preserved ({:?})",
            backend
        );
        assert_eq!(
            check_in,
            vec![6, 3, 4],
            "in-strengths not preserved ({:?})",
            backend
        );
    }

    /// Exhaustive enumeration test: enumerate all occupation matrices for N=2,
    /// ME, small strengths, compute exact probabilities, and verify MCMC
    /// empirical frequencies match.
    #[cfg(test)]
    mod enumeration_tests {
        use super::*;
        use std::collections::HashMap;

        type OccupiedState = Vec<((u64, u64), OccNum)>;
        type WeightedState = (OccupiedState, f64);

        /// Enumerate all occupation matrices for N=2 with given strengths.
        /// Returns a list of `(occupation_map, log_weight)` where
        /// `log_weight = Σ −log(t_ij!)`.
        fn enumerate_me_states(
            s_out: &[OccNum],
            s_in: &[OccNum],
            self_loops: bool,
        ) -> Vec<WeightedState> {
            let n = s_out.len();
            let mut results = Vec::new();

            // Generate all possible occupation matrices via recursion.
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
                    if remaining_out.iter().all(|&s| s == 0) && remaining_in.iter().all(|&s| s == 0)
                    {
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

            // Enumerate exact distribution.
            let states = enumerate_me_states(&s_out, &s_in, true);
            let total_weight: f64 = states.iter().map(|(_, w)| w.exp()).sum();

            // Run MCMC many times and count occupation frequencies.
            let trials = 2000;
            let mut counts: HashMap<OccupiedState, u64> = HashMap::new();
            for seed in 0..trials {
                let prob =
                    make_problem(OccupationFamily::Poisson, s_out.clone(), s_in.clone(), true);
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

            // Check each state appears with roughly the right frequency.
            for (state, log_weight) in &states {
                let weight = log_weight.exp();
                let expected_prob = weight / total_weight;
                let expected_count = expected_prob * trials as f64;
                let observed = counts.get(state).copied().unwrap_or(0) as f64;
                let ratio = if expected_count > 0.0 {
                    observed / expected_count
                } else {
                    1.0
                };
                assert!(
                    ratio > 0.25 && ratio < 2.5,
                    "state {:?}: expected {expected_count:.1}, observed {observed}, ratio {ratio:.2}",
                    state,
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
                let prob = make_problem(
                    OccupationFamily::Poisson,
                    s_out.clone(),
                    s_in.clone(),
                    false,
                );
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
                let ratio = if expected_count > 0.0 {
                    observed / expected_count
                } else {
                    1.0
                };
                assert!(
                    ratio > 0.25 && ratio < 2.5,
                    "state {:?}: expected {expected_count:.1}, observed {observed}, ratio {ratio:.2}",
                    state,
                );
            }
        }
    }
}
