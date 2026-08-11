//! Persistent MCMC chain for fixed-strength sampling.
//!
//! Provides:
//! - `FixedStrengthChain` — stateful chain with step/sweep/burn_in/sample.
//! - `sample_fixed_strength` — one-shot convenience orchestrator (Phase 4).
//! - `sample_fixed_strength_with_cost` — cost-constrained entry point (Phase 5).

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::time::Instant;

use super::cost_fit;
use super::domain::PairDomain;
use super::errors::FixedStrengthCostError;
use super::errors::FixedStrengthError;
use super::initializer::initialize_table;
use super::move_cycle::occupied_cycle4_step;
use super::problem::ResidualStrengthProblem;
use super::repair;
use super::state::StrengthState;
use super::target::StrengthTarget;
use crate::generation::microcanonical::mcmc::{McmcConfig, McmcCounters, McmcOutcome};
use crate::generation::output::SampledNetwork;
use crate::model::family::OccupationFamily;
use crate::pairs::PairCostProvider;
use crate::OccNum;

/// Per-stage benchmark metrics for the fixed-strength pipeline (§35).
#[derive(Clone, Debug)]
pub struct FixedStrengthBenchMetrics {
    /// Wall time for construction (initialize_table + State::new).
    pub construction_time_s: f64,
    /// Wall time for all repairs (loop + capacity + forbidden/inadmissible).
    pub repair_time_s: f64,
    /// Total rectangle-repair steps across all repair phases.
    pub repair_steps: u64,
    /// Number of reconstruction restarts performed (0 = first try).
    pub repair_restarts: u32,
    /// Number of occupied pairs after all repairs.
    pub occupied_pairs: usize,
    /// Total MCMC proposals during burn-in + thinning.
    pub mcmc_proposals: u64,
    /// Total MCMC acceptances during burn-in + thinning.
    pub mcmc_accepted: u64,
    /// Total MCMC held (structurally invalid) during burn-in + thinning.
    pub mcmc_held: u64,
    /// Wall time for burn-in + thinning sweeps.
    pub mcmc_time_s: f64,
    /// Wall time for the final sample() call alone.
    pub final_sampling_time_s: f64,
    /// Wall time for gamma fitting (None if no cost).
    pub gamma_fit_time_s: Option<f64>,
    /// Effective sample size of cost samples at fitted gamma (None if no cost).
    pub cost_ess: Option<f64>,
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

    /// Replace the target (e.g., with a new gamma after fitting).
    ///
    /// The state and domain are unchanged.  Use this to change gamma
    /// without rebuilding the chain or running feasibility again.
    pub fn set_target(&mut self, target: StrengthTarget<'a>) {
        self.target = target;
    }

    /// Perform one MCMC step (one 4-cycle proposal).
    pub fn step(&mut self, rng: &mut impl rand::Rng) -> McmcOutcome {
        self.counters.proposals += 1;
        let outcome = occupied_cycle4_step(&mut self.state, &self.target, &self.domain, rng);
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

    /// Run sweeps and collect a cost sample after each sweep.
    ///
    /// Returns a vector of total costs, one per sweep.  The target's gamma
    /// must already be set to the desired value before calling this.
    ///
    /// # Errors
    ///
    /// Propagates cost errors from [`state_cost`](super::cost::state_cost).
    pub fn measure_cost(
        &mut self,
        rng: &mut impl rand::Rng,
        sweeps: usize,
        costs: &dyn PairCostProvider,
    ) -> Result<Vec<f64>, super::errors::FixedStrengthCostError> {
        let mut samples = Vec::with_capacity(sweeps);
        for _ in 0..sweeps {
            self.sweep(rng);
            let total = super::cost::state_cost(&self.state, costs)?;
            samples.push(total);
        }
        Ok(samples)
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

// --------------------------------------------------------------------------
// Shared initialization and repair helpers
// --------------------------------------------------------------------------

/// Initialize a state from a residual strength problem (no repair).
fn init_state(problem: &ResidualStrengthProblem) -> Result<StrengthState, FixedStrengthError> {
    problem.validate()?;
    let table = initialize_table(
        &problem.strength_out,
        &problem.strength_in,
        problem.family,
        &problem.domain,
    )?;
    Ok(StrengthState::new(problem.domain.node_count(), table))
}

/// Apply all repair phases to a state.
///
/// Returns `(total_repair_steps, repair_restarts, occupied_pairs)`.
fn repair_state(
    state: &mut StrengthState,
    problem: &ResidualStrengthProblem,
    rng: &mut impl Rng,
) -> Result<(u64, u32, usize), FixedStrengthError> {
    let family = problem.family;
    let mut total_repair_steps: u64 = 0;
    let mut repair_restarts: u32 = 0;

    // Phase D: loop repair for complete loopless ME/W/B (spec 14-17).
    if !problem.domain.self_loops_allowed()
        && (family == OccupationFamily::ME
            || matches!(family, OccupationFamily::W { .. })
            || matches!(family, OccupationFamily::B { .. }))
    {
        if !repair::loopless_feasibility_check(&problem.strength_out, &problem.strength_in) {
            return Err(FixedStrengthError::InitializationFailed(
                "loopless feasibility check failed: \
                    s_i^out + s_i^in > T for some node"
                    .into(),
            ));
        }
        total_repair_steps += repair::repair_self_loops(
            state,
            &problem.domain,
            &repair::RepairConfig::default(),
            rng,
        )?;
    }

    // Phase E: B capacity repair (spec 18).
    if matches!(family, OccupationFamily::B { .. }) {
        let (steps, restarts) = repair::repair_all_violations(
            state,
            family,
            &problem.domain,
            None,
            &repair::RepairConfig::default(),
            rng,
            &problem.strength_out,
            &problem.strength_in,
        )?;
        total_repair_steps += steps;
        repair_restarts = restarts;
    }

    // Admissibility repair for sparse domains (spec §19).
    if matches!(problem.domain, PairDomain::Sparse { .. }) {
        total_repair_steps += repair::repair_inadmissible_pairs(
            state,
            family,
            &problem.domain,
            &repair::RepairConfig::default(),
            rng,
        )?;
    }

    let occupied_pairs = state.occupied_count();
    Ok((total_repair_steps, repair_restarts, occupied_pairs))
}

/// One-shot fixed-strength sampling (Phase 4, no cost).
///
/// Uses the production pipeline: compressed constructor → repair →
/// occupied-cell MCMC.
///
/// # Errors
///
/// Returns [`FixedStrengthError`] if the problem is infeasible or an
/// internal error occurs.
pub fn sample_fixed_strength(
    problem: ResidualStrengthProblem,
    config: McmcConfig,
) -> Result<SampledNetwork, FixedStrengthError> {
    let family = problem.family;
    let seed = config.seed;
    let mut rng = StdRng::seed_from_u64(seed);

    let mut state = init_state(&problem)?;
    let (_, _, _) = repair_state(&mut state, &problem, &mut rng)?;

    let target = StrengthTarget::new(family);
    let mut chain = FixedStrengthChain::new(state, target, problem.domain, config);
    chain.burn_in(&mut rng);
    let network = chain.sample(&mut rng);

    Ok(network)
}

/// Fixed-strength sampling with a cost provider (Phase 5).
///
/// The target starts with gamma=0.  The caller should call
/// `chain.set_target(...)` with a fitted gamma before sampling.
///
/// The cost-constrained chain always runs the occupied-cell MCMC kernel
/// (a direct exact sampler is not applicable when costs are present).
///
/// # Errors
///
/// Returns [`FixedStrengthError`] if the problem is infeasible.
pub fn sample_fixed_strength_with_cost<'a>(
    problem: ResidualStrengthProblem,
    costs: &'a dyn PairCostProvider,
    config: McmcConfig,
) -> Result<FixedStrengthChain<'a>, FixedStrengthError> {
    let family = problem.family;
    let seed = config.seed;
    let mut rng = StdRng::seed_from_u64(seed);

    let mut state = init_state(&problem)?;
    let (_, _, _) = repair_state(&mut state, &problem, &mut rng)?;

    let target = StrengthTarget::with_costs(family, costs);
    let chain = FixedStrengthChain::new(state, target, problem.domain, config);

    Ok(chain)
}

// --------------------------------------------------------------------------
// Benchmark entry points (§35)
// --------------------------------------------------------------------------

/// Fixed-strength sampling with per-stage benchmark instrumentation.
///
/// See [`sample_fixed_strength`] for semantics.  This variant additionally
/// reports wall-time, repair, and MCMC counters useful for benchmarking.
pub fn sample_fixed_strength_bench(
    problem: ResidualStrengthProblem,
    config: McmcConfig,
) -> Result<(SampledNetwork, FixedStrengthBenchMetrics), FixedStrengthError> {
    let family = problem.family;
    let seed = config.seed;
    let mut rng = StdRng::seed_from_u64(seed);

    // Phase 1: Construction (initialize_table + State::new).
    let t0 = Instant::now();
    let mut state = init_state(&problem)?;
    let construction_time_s = t0.elapsed().as_secs_f64();

    // Phase 2: Repair.
    let t0 = Instant::now();
    let (repair_steps, repair_restarts, occupied_pairs) =
        repair_state(&mut state, &problem, &mut rng)?;
    let repair_time_s = t0.elapsed().as_secs_f64();

    // Phase 3: MCMC burn-in + thinning.
    let target = StrengthTarget::new(family);
    let sweeps = config.sweeps_per_sample.max(1);
    let mut chain = FixedStrengthChain::new(state, target, problem.domain, config);
    let t0 = Instant::now();
    chain.burn_in(&mut rng);
    for _ in 0..sweeps {
        chain.sweep(&mut rng);
    }
    let mcmc_time_s = t0.elapsed().as_secs_f64();
    let mcmc_proposals = chain.counters.proposals;
    let mcmc_accepted = chain.counters.accepted;
    let mcmc_held = chain.counters.held;

    // Phase 4: Final sampling (convert to network).
    let t0 = Instant::now();
    let network = chain.state.to_sampled_network();
    let final_sampling_time_s = t0.elapsed().as_secs_f64();

    Ok((
        network,
        FixedStrengthBenchMetrics {
            construction_time_s,
            repair_time_s,
            repair_steps,
            repair_restarts,
            occupied_pairs,
            mcmc_proposals,
            mcmc_accepted,
            mcmc_held,
            mcmc_time_s,
            final_sampling_time_s,
            gamma_fit_time_s: None,
            cost_ess: None,
        },
    ))
}

/// Fixed-strength cost-constrained sampling with per-stage benchmark instrumentation.
///
/// See [`sample_fixed_strength_with_cost`] for semantics.  This variant
/// additionally reports wall-time, repair, gamma-fitting, and MCMC counters.
pub fn sample_fixed_strength_with_cost_bench(
    problem: ResidualStrengthProblem,
    costs: &dyn PairCostProvider,
    config: McmcConfig,
    fit_config: &super::cost_fit::FixedStrengthCostFitConfig,
    observed_total_cost: f64,
    fixed_cost: f64,
) -> Result<(SampledNetwork, FixedStrengthBenchMetrics), FixedStrengthCostError> {
    let family = problem.family;
    let seed = config.seed;
    let mut rng = StdRng::seed_from_u64(seed);

    // Phase 1: Construction.
    let t0 = Instant::now();
    let mut state = init_state(&problem).map_err(FixedStrengthCostError::FixedStrength)?;
    let construction_time_s = t0.elapsed().as_secs_f64();

    // Phase 2: Repair.
    let t0 = Instant::now();
    let (repair_steps, repair_restarts, occupied_pairs) =
        repair_state(&mut state, &problem, &mut rng)
            .map_err(FixedStrengthCostError::FixedStrength)?;
    let repair_time_s = t0.elapsed().as_secs_f64();

    // Phase 3: Cost chain preparation.
    let target = StrengthTarget::with_costs(family, costs);
    let sweeps = config.sweeps_per_sample.max(1);
    let mut chain = FixedStrengthChain::new(state, target, problem.domain, config);

    // Phase 4: Gamma fitting.
    let t0 = Instant::now();
    let fit_result = cost_fit::fit_gamma(
        &mut chain,
        &mut rng,
        costs,
        observed_total_cost,
        fixed_cost,
        fit_config,
    )?;
    let gamma_fit_time_s = t0.elapsed().as_secs_f64();
    let cost_ess = Some(cost_fit::effective_sample_size(&fit_result.best_samples));

    // Phase 5: Set fitted gamma and MCMC burn-in + thinning.
    {
        let mut target = StrengthTarget::with_costs(family, costs);
        target.set_gamma(fit_result.gamma);
        chain.set_target(target);
    }
    chain.counters.reset();
    let t0 = Instant::now();
    chain.burn_in(&mut rng);
    for _ in 0..sweeps {
        chain.sweep(&mut rng);
    }
    let mcmc_time_s = t0.elapsed().as_secs_f64();
    let mcmc_proposals = chain.counters.proposals;
    let mcmc_accepted = chain.counters.accepted;
    let mcmc_held = chain.counters.held;

    // Phase 6: Final sampling.
    let t0 = Instant::now();
    let network = chain.state.to_sampled_network();
    let final_sampling_time_s = t0.elapsed().as_secs_f64();

    Ok((
        network,
        FixedStrengthBenchMetrics {
            construction_time_s,
            repair_time_s,
            repair_steps,
            repair_restarts,
            occupied_pairs,
            mcmc_proposals,
            mcmc_accepted,
            mcmc_held,
            mcmc_time_s,
            final_sampling_time_s,
            gamma_fit_time_s: Some(gamma_fit_time_s),
            cost_ess,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::microcanonical::occupation_mcmc::problem::FixedStrengthProblem;

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
    fn cycle_mcmc_backend_selected() {
        let prob = make_problem(OccupationFamily::ME, vec![5, 5], vec![5, 5], true);
        let config = McmcConfig::new(10, 5, 42);
        let net = sample_fixed_strength(prob, config).unwrap();
        let total: OccNum = net.occ_nums.iter().sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn me_mcmc_backend_no_self_loops() {
        let prob = make_problem(OccupationFamily::ME, vec![5, 5, 5], vec![5, 5, 5], false);
        let config = McmcConfig::new(20, 10, 42);
        let net = sample_fixed_strength(prob, config).unwrap();
        let total: OccNum = net.occ_nums.iter().sum();
        assert_eq!(total, 15);
        // Looplessness will be guaranteed by Phase D (loop repair).
        // For now, verify strengths are preserved.
        let (mut co, mut ci) = (vec![0u64; 3], vec![0u64; 3]);
        for ((&s, &t), &o) in net
            .sources
            .iter()
            .zip(net.targets.iter())
            .zip(net.occ_nums.iter())
        {
            co[s as usize] += o;
            ci[t as usize] += o;
        }
        assert_eq!(co, [5, 5, 5]);
        assert_eq!(ci, [5, 5, 5]);
    }

    #[test]
    fn b_fixed_strength() {
        let prob = make_problem(
            OccupationFamily::B { layers: 4 },
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
        let net = sample_fixed_strength(prob, config).unwrap();
        let total: OccNum = net.occ_nums.iter().sum();
        assert_eq!(total, 12);
        for &occ in &net.occ_nums {
            assert!(occ <= 4, "B occupation {occ} exceeds 4");
        }
    }

    #[test]
    fn w_fixed_strength() {
        let prob = make_problem(
            OccupationFamily::W { layers: 2 },
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
        let net = sample_fixed_strength(prob, config).unwrap();
        let total: OccNum = net.occ_nums.iter().sum();
        assert_eq!(total, 15);
    }

    #[test]
    fn strengths_preserved() {
        let prob = make_problem(OccupationFamily::ME, vec![4, 7, 2], vec![6, 3, 4], true);
        let config = McmcConfig::new(20, 10, 42);
        let net = sample_fixed_strength(prob, config).unwrap();
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
        assert_eq!(check_out, vec![4, 7, 2], "out-strengths not preserved");
        assert_eq!(check_in, vec![6, 3, 4], "in-strengths not preserved");
    }

    #[test]
    fn chain_deterministic() {
        // Reproduce the exact Python test: B M=3, strengths=[3,3,3], self_loops=true, seed=42.
        // Two consecutive runs must produce identical output.
        let config = McmcConfig::new(50, 10, 42);

        let run = || -> Vec<OccNum> {
            let prob = make_problem(
                OccupationFamily::B { layers: 3 },
                vec![3, 3, 3],
                vec![3, 3, 3],
                true,
            );
            let net = sample_fixed_strength(prob, config.clone()).unwrap();
            let mut all: Vec<OccNum> = Vec::new();
            for i in 0..net.sources.len() {
                all.push(net.sources[i]);
                all.push(net.targets[i]);
                all.push(net.occ_nums[i]);
            }
            all
        };

        let a = run();
        let b = run();
        assert_eq!(
            a, b,
            "two runs with same seed must produce identical output"
        );
    }

    #[test]
    fn set_target_updates_gamma() {
        let prob = make_problem(OccupationFamily::ME, vec![5, 5], vec![5, 5], true);
        let config = McmcConfig::new(10, 5, 42);
        let mut chain = {
            let target = StrengthTarget::new(OccupationFamily::ME);
            let table = initialize_table(
                &prob.strength_out,
                &prob.strength_in,
                prob.family,
                &prob.domain,
            )
            .unwrap();
            let state = StrengthState::new(prob.domain.node_count(), table);
            FixedStrengthChain::new(state, target, prob.domain, config)
        };

        assert!((chain.target.gamma() - 0.0).abs() < 1e-12);
        let new_target = {
            let mut t = StrengthTarget::new(OccupationFamily::ME);
            t.set_gamma(2.5);
            t
        };
        chain.set_target(new_target);
        assert!((chain.target.gamma() - 2.5).abs() < 1e-12);
    }
}
