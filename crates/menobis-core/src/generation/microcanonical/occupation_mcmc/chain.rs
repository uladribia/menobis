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
use super::fixed_edges::{
    fixed_edge_sweep, validate_edge_target, BridgeConfig, EdgeRepairConfig, FixedEdgeCounters,
};
use super::fixed_edges::{repair_to_edge_target, residual_edge_target};
use super::initializer::initialize_table;
use super::move_cycle::{occupied_cycle4_step, Cycle4Proposal};
use super::problem::{FixedStrengthProblem, ResidualStrengthProblem};
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
///
/// Uses the caller's RNG so the initial construction is reproducible per
/// seed and every reconstruction restart differs (§13.4).
fn init_state(
    problem: &ResidualStrengthProblem,
    rng: &mut impl Rng,
) -> Result<StrengthState, FixedStrengthError> {
    problem.validate()?;
    let table = initialize_table(
        &problem.strength_out,
        &problem.strength_in,
        problem.family,
        &problem.domain,
        rng,
    )?;
    Ok(StrengthState::new(problem.domain.node_count(), table))
}

/// Merge positive fixed pairs into a residual sampled network.
///
/// The residual domain excludes every fixed coordinate, so merged output
/// coordinates are unique without downstream deduplication (§25).  Zero-
/// occupation fixed pairs are dropped (they never appear in output).
pub(crate) fn merge_fixed_pairs(
    residual: SampledNetwork,
    fixed_pairs: &[(u64, u64, OccNum)],
) -> SampledNetwork {
    let mut all: Vec<(u64, u64, OccNum)> =
        Vec::with_capacity(residual.sources.len() + fixed_pairs.len());
    for &(s, t, o) in fixed_pairs {
        if o > 0 {
            all.push((s, t, o));
        }
    }
    for i in 0..residual.sources.len() {
        all.push((
            residual.sources[i],
            residual.targets[i],
            residual.occ_nums[i],
        ));
    }
    all.sort_unstable();
    SampledNetwork {
        sources: all.iter().map(|&(s, _, _)| s).collect(),
        targets: all.iter().map(|&(_, t, _)| t).collect(),
        occ_nums: all.iter().map(|&(_, _, o)| o).collect(),
    }
}

/// One-shot fixed-strength + fixed-occupied-pair-count sampling (§26).
///
/// Pipeline: Rust residualizes the full problem once (validating fixed
/// pairs and subtracting strengths/edges/domain), validates the residual
/// edge target, constructs a fresh randomized fixed-strength state,
/// applies structural repair, repairs the state to the exact residual
/// occupied-pair count, burn-in + thinning with the exact mixed
/// local/bridge kernel, and merges positive fixed pairs back into the
/// full network.
///
/// # Errors
///
/// Propagates fixed-pair validation (duplicates, bounds), edge-target
/// infeasibility, repair exhaustion, or initialization failures.  An
/// inexact-E state never enters sampling (§26): if the repair cannot
/// reach the target, a structured error is returned.
pub fn sample_fixed_strength_edges(
    problem: FixedStrengthProblem,
    full_target_edges: usize,
    config: McmcConfig,
    bridge_config: BridgeConfig,
) -> Result<SampledNetwork, FixedStrengthError> {
    sample_fixed_strength_edges_bench(problem, full_target_edges, config, bridge_config)
        .map(|(network, _)| network)
}

/// Per-stage diagnostics for the fixed-(s,E) pipeline (§34–§35).
#[derive(Clone, Debug)]
pub struct FixedStrengthEdgesBench {
    /// Wall time for compressed construction.
    pub construction_time_s: f64,
    /// Wall time for structural repair (loops, capacity, inadmissible).
    pub structural_repair_time_s: f64,
    /// Wall time for edge-count repair.
    pub edge_repair_time_s: f64,
    /// Edge-repair steps used.
    pub edge_repair_steps: u64,
    /// Edge-repair reconstruction restarts.
    pub edge_repair_restarts: u32,
    /// Occupied-pair count of the first constructed state.
    pub initial_edges: usize,
    /// Best occupied-pair count reached during repair.
    pub best_edges: usize,
    /// Residual edge target.
    pub target_edges: usize,
    /// Wall time for burn-in + thinning sweeps.
    pub mcmc_time_s: f64,
    /// Mixed-kernel counters.
    pub counters: FixedEdgeCounters,
}

/// Benchmark-instrumented one-shot fixed-(s,E) sampling (§34–§35).
pub fn sample_fixed_strength_edges_bench(
    problem: FixedStrengthProblem,
    full_target_edges: usize,
    config: McmcConfig,
    bridge_config: BridgeConfig,
) -> Result<(SampledNetwork, FixedStrengthEdgesBench), FixedStrengthError> {
    let fixed_pairs = problem.fixed_pairs.clone();
    let family = problem.family;

    let seed = config.seed;
    let mut rng = StdRng::seed_from_u64(seed);

    // ---- Residualize once (§16) and validate the edge target (§14) ----
    let residual = problem.into_residual()?;
    let residual_target = residual_edge_target(full_target_edges, &fixed_pairs)?;
    validate_edge_target(&residual, residual_target)?;

    // ---- Construction + structural repair ----
    let n = residual.domain.node_count();
    let t0 = Instant::now();
    let table = initialize_table(
        &residual.strength_out,
        &residual.strength_in,
        residual.family,
        &residual.domain,
        &mut rng,
    )?;
    let mut state = StrengthState::new(n, table);
    let construction_time_s = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let (_, _, _) = repair_state(&mut state, &residual, &mut rng)?;
    let structural_repair_time_s = t0.elapsed().as_secs_f64();

    // ---- Edge-count repair (§13) ----
    let t0 = Instant::now();
    let edge_outcome = repair_to_edge_target(
        &mut state,
        &residual,
        &mut rng,
        residual_target,
        &EdgeRepairConfig::default(),
    )?;
    let edge_repair_time_s = t0.elapsed().as_secs_f64();

    // ---- Runtime invariant (§26): no inexact state enters sampling ----
    if state.occupied_count() != residual_target {
        return Err(FixedStrengthError::EdgeRepairExhausted {
            best_edges: state.occupied_count(),
            target_edges: residual_target,
            best_distance: state.occupied_count().abs_diff(residual_target),
            restarts: edge_outcome.restarts,
            total_steps: edge_outcome.steps,
        });
    }

    // ---- Burn-in + thinning with the exact mixed kernel ----
    let target = StrengthTarget::new(family);
    let mut counters = FixedEdgeCounters::default();
    let t0 = Instant::now();
    for _ in 0..config.burn_in_sweeps.max(1) {
        fixed_edge_sweep(
            &mut state,
            &target,
            &residual.domain,
            &mut rng,
            residual_target,
            &bridge_config,
            &mut counters,
            config.proposals_per_sweep,
        );
    }
    for _ in 0..config.sweeps_per_sample.max(1) {
        fixed_edge_sweep(
            &mut state,
            &target,
            &residual.domain,
            &mut rng,
            residual_target,
            &bridge_config,
            &mut counters,
            config.proposals_per_sweep,
        );
    }
    let mcmc_time_s = t0.elapsed().as_secs_f64();

    // ---- Exact-E residual network + merge fixed pairs ----
    let residual_network = state.to_sampled_network();
    let network = merge_fixed_pairs(residual_network, &fixed_pairs);

    Ok((
        network,
        FixedStrengthEdgesBench {
            construction_time_s,
            structural_repair_time_s,
            edge_repair_time_s,
            edge_repair_steps: edge_outcome.steps,
            edge_repair_restarts: edge_outcome.restarts,
            initial_edges: edge_outcome.initial_edges,
            best_edges: edge_outcome.best_edges,
            target_edges: residual_target,
            mcmc_time_s,
            counters,
        },
    ))
}

/// Apply all repair phases to a state.
///
/// Returns `(total_repair_steps, repair_restarts, occupied_pairs)`.
/// Exposed to sibling modules so the fixed-(s,E) edge repair can
/// reconstruct fresh states exactly as the fixed-strength pipeline does.
pub(crate) fn repair_state(
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

    // Admissibility repair for sparse/excluded domains (spec §15.3, §19).
    if problem.domain.requires_admissibility_repair() {
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

    let mut state = init_state(&problem, &mut rng)?;
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

    let mut state = init_state(&problem, &mut rng)?;
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
    let mut state = init_state(&problem, &mut rng)?;
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
    let mut state =
        init_state(&problem, &mut rng).map_err(FixedStrengthCostError::FixedStrength)?;
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
            let mut rng = StdRng::seed_from_u64(42);
            let target = StrengthTarget::new(OccupationFamily::ME);
            let table = initialize_table(
                &prob.strength_out,
                &prob.strength_in,
                prob.family,
                &prob.domain,
                &mut rng,
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

    // -----------------------------------------------------------------
    // Phase 8: one-shot fixed-(s,E) core sampler
    // -----------------------------------------------------------------

    fn full_problem(
        family: OccupationFamily,
        so: Vec<OccNum>,
        si: Vec<OccNum>,
        sl: bool,
        fixed: Vec<(u64, u64, OccNum)>,
    ) -> FixedStrengthProblem {
        let domain = PairDomain::Complete {
            node_count: so.len(),
            self_loops: sl,
        };
        FixedStrengthProblem::new(family, so, si, domain, fixed).unwrap()
    }

    fn full_strengths(net: &SampledNetwork, n: usize) -> (Vec<OccNum>, Vec<OccNum>) {
        let mut out = vec![0u64; n];
        let mut inp = vec![0u64; n];
        for ((&s, &t), &o) in net
            .sources
            .iter()
            .zip(net.targets.iter())
            .zip(net.occ_nums.iter())
        {
            out[s as usize] += o;
            inp[t as usize] += o;
        }
        (out, inp)
    }

    #[test]
    fn fixed_se_all_families_exact_strengths_and_edges() {
        // Proven-feasible full problems (explicit constructions): the
        // gate demands exact strengths and exact E for all three families.
        type Case = (OccupationFamily, Vec<OccNum>, Vec<OccNum>, bool, usize);
        let cases: Vec<Case> = vec![
            (OccupationFamily::ME, vec![2, 2], vec![2, 2], true, 2),
            (OccupationFamily::ME, vec![2, 2], vec![2, 2], true, 4),
            (
                OccupationFamily::ME,
                vec![3, 3, 3, 3],
                vec![3, 3, 3, 3],
                true,
                6,
            ),
            (
                OccupationFamily::B { layers: 2 },
                vec![2, 2],
                vec![2, 2],
                true,
                2,
            ),
            (
                OccupationFamily::B { layers: 2 },
                vec![2, 2],
                vec![2, 2],
                true,
                4,
            ),
            (
                OccupationFamily::W { layers: 1 },
                vec![2, 2],
                vec![2, 2],
                true,
                4,
            ),
        ];
        for (family, so, si, sl, e) in cases {
            let problem = full_problem(family, so.clone(), si.clone(), sl, vec![]);
            let config = McmcConfig::new(20, 10, 42);
            let net = sample_fixed_strength_edges(problem, e, config, BridgeConfig::default())
                .unwrap_or_else(|err| panic!("{family:?} E={e}: {err}"));
            assert_eq!(net.sources.len(), e, "{family:?} E={e}");
            let n = so.len();
            let (out, inp) = full_strengths(&net, n);
            assert_eq!(out, so, "{family:?} E={e}: out-strength drift");
            assert_eq!(inp, si, "{family:?} E={e}: in-strength drift");
            assert!(
                net.occ_nums.iter().all(|&o| o > 0),
                "zero occupations in output"
            );
        }
    }

    #[test]
    fn fixed_se_reproducible_by_seed() {
        let run = |seed: u64| -> Vec<OccNum> {
            let problem = full_problem(
                OccupationFamily::ME,
                vec![3, 3, 3, 3],
                vec![3, 3, 3, 3],
                true,
                vec![],
            );
            let net = sample_fixed_strength_edges(
                problem,
                6,
                McmcConfig::new(20, 10, seed),
                BridgeConfig::default(),
            )
            .unwrap();
            let mut triples: Vec<_> = net
                .sources
                .iter()
                .zip(net.targets.iter())
                .zip(net.occ_nums.iter())
                .map(|((&s, &t), &o)| (s, t, o))
                .collect();
            triples.sort_unstable();
            triples
                .into_iter()
                .flat_map(|(s, t, o)| vec![s, t, o])
                .collect()
        };
        assert_eq!(run(1), run(1), "same seed must reproduce the sample");
    }

    #[test]
    fn fixed_se_positive_fixed_pair_merged_exactly() {
        // ME N=2 s=[3,3]/[3,3] sl=true, full E=4 with fixed pair (0,1,1):
        // residual s=[2,3]/[3,2], domain minus (0,1), residual E=3
        // (feasible: {(0,0)=2,(1,0)=1,(1,1)=2}).
        let problem = full_problem(
            OccupationFamily::ME,
            vec![3, 3],
            vec![3, 3],
            true,
            vec![(0, 1, 1)],
        );
        let net = sample_fixed_strength_edges(
            problem,
            4,
            McmcConfig::new(20, 10, 7),
            BridgeConfig::default(),
        )
        .unwrap();
        assert_eq!(net.sources.len(), 4, "full E must include the fixed pair");
        let (out, inp) = full_strengths(&net, 2);
        assert_eq!(out, vec![3, 3]);
        assert_eq!(inp, vec![3, 3]);
        // Fixed pair present with the right occupation under a unique key.
        let keys: Vec<(u64, u64)> = net
            .sources
            .iter()
            .zip(net.targets.iter())
            .map(|(&s, &t)| (s, t))
            .collect();
        let has_fixed = keys.contains(&(0, 1));
        let occ01 = net
            .sources
            .iter()
            .zip(net.targets.iter())
            .zip(net.occ_nums.iter())
            .find(|((&s, &t), _)| s == 0 && t == 1)
            .map(|(_, &o)| o);
        assert!(
            has_fixed && occ01 == Some(1),
            "fixed pair missing/misvalued"
        );
        // No duplicate output coordinates.
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), keys.len(), "duplicate output coordinates");
    }

    #[test]
    fn fixed_se_zero_fixed_pair_never_reoccupied() {
        // ME N=3 s=[2,2,2] sl=true, zero fixed pair (0,1,0): coordinate
        // (0,1) must never appear in output and full E must be exact.
        let problem = full_problem(
            OccupationFamily::ME,
            vec![2, 2, 2],
            vec![2, 2, 2],
            true,
            vec![(0, 1, 0)],
        );
        let net = sample_fixed_strength_edges(
            problem,
            3,
            McmcConfig::new(20, 10, 11),
            BridgeConfig::default(),
        )
        .unwrap();
        assert_eq!(net.sources.len(), 3, "full E must be exact");
        assert!(
            !net.sources
                .iter()
                .zip(net.targets.iter())
                .any(|(&s, &t)| s == 0 && t == 1),
            "zero fixed pair was reoccupied"
        );
        let (out, inp) = full_strengths(&net, 3);
        assert_eq!(out, vec![2, 2, 2]);
        assert_eq!(inp, vec![2, 2, 2]);
    }

    #[test]
    fn fixed_se_duplicate_fixed_pair_rejected() {
        let problem = full_problem(
            OccupationFamily::ME,
            vec![3, 3],
            vec![3, 3],
            true,
            vec![(0, 1, 1), (0, 1, 1)],
        );
        match sample_fixed_strength_edges(
            problem,
            4,
            McmcConfig::new(5, 5, 1),
            BridgeConfig::default(),
        ) {
            Err(FixedStrengthError::InvalidResidual(msg)) => {
                assert!(msg.contains("duplicate fixed pair"), "{msg}")
            }
            other => panic!("expected duplicate error, got {other:?}"),
        }
    }

    #[test]
    fn fixed_se_infeasible_targets_rejected() {
        // E above total occupation.
        let problem = full_problem(
            OccupationFamily::ME,
            vec![4, 4, 4, 4],
            vec![4, 4, 4, 4],
            true,
            vec![],
        );
        match sample_fixed_strength_edges(
            problem,
            17,
            McmcConfig::new(5, 5, 1),
            BridgeConfig::default(),
        ) {
            Err(FixedStrengthError::InvalidEdgeTarget(msg)) => {
                assert!(msg.contains("exceeds total occupation"), "{msg}")
            }
            other => panic!("expected InvalidEdgeTarget, got {other:?}"),
        }
        // E below the positive fixed-pair count.
        let problem = full_problem(
            OccupationFamily::ME,
            vec![3, 3],
            vec![3, 3],
            true,
            vec![(0, 1, 1)],
        );
        match sample_fixed_strength_edges(
            problem,
            0,
            McmcConfig::new(5, 5, 1),
            BridgeConfig::default(),
        ) {
            Err(FixedStrengthError::InvalidEdgeTarget(msg)) => {
                assert!(msg.contains("positive fixed pairs"), "{msg}")
            }
            other => panic!("expected InvalidEdgeTarget, got {other:?}"),
        }
    }
}

// --------------------------------------------------------------------------
// Phase 9: one-shot fixed-(s,k) core sampler (§25–§27)
// --------------------------------------------------------------------------

use super::fixed_degrees::{
    degree_distance, degree_trace_step, degree_trace_sweep, repair_to_degree_target,
    residualize_degree_target, validate_degree_target, DegreeRepairConfig, DegreeTraceConfig,
    DegreeTraceCounters,
};

/// Per-stage diagnostics for the fixed-(s,k) pipeline (§27, §35-style).
#[derive(Clone, Debug)]
pub struct FixedStrengthDegreeBench {
    /// Wall time for compressed construction.
    pub construction_time_s: f64,
    /// Wall time for structural repair (loops, capacity, inadmissible).
    pub structural_repair_time_s: f64,
    /// Wall time for edge-count repair to exact residual E.
    pub edge_repair_time_s: f64,
    /// Edge-repair steps used.
    pub edge_repair_steps: u64,
    /// Edge-repair reconstruction restarts.
    pub edge_repair_restarts: u32,
    /// Wall time for degree repair to the exact residual degrees.
    pub degree_repair_time_s: f64,
    /// Degree-repair steps used.
    pub degree_repair_steps: u64,
    /// Degree-repair reconstruction restarts.
    pub degree_repair_restarts: u32,
    /// Half-normalized degree distance of the initial exact-E state.
    pub initial_degree_distance: u64,
    /// Residual degree target edge count.
    pub target_edges: usize,
    /// Wall time for burn-in + thinning sweeps.
    pub mcmc_time_s: f64,
    /// Nested degree-trace mobility counters (§27).
    pub degree_trace: DegreeTraceCounters,
    /// Underlying fixed-(s,E) kernel counters (§27: to attribute slow
    /// performance to K_E itself vs the outer degree trace).
    pub fixed_edge: FixedEdgeCounters,
}

/// Benchmark-instrumented one-shot fixed-(s,k) sampling (§25).
///
/// Pipeline: Rust residualizes strengths/fixed pairs exactly once,
/// subtracts the fixed-pair degree contribution, validates the combined
/// residual (s,k) target, constructs an exact residual `E` state (edge
/// repair), repairs it to the exact residual degree vectors with the
/// shared degree-biased auxiliary step, then burn-in + thinning with the
/// capped first-return degree trace.  The returned network carries exact
/// full strengths, exact full degrees, and exact `E`.
///
/// # Errors
///
/// - invalid fixed pairs / residual (propagated from
///   `FixedStrengthProblem::into_residual`);
/// - [`InvalidDegreeTarget`](FixedStrengthError::InvalidDegreeTarget)
///   from residualization or combined-target validation;
/// - degree-repair exhaustion never returns an inexact sample.
#[allow(clippy::too_many_arguments)]
pub fn sample_fixed_strength_degree_bench(
    problem: FixedStrengthProblem,
    full_degree_out: Vec<u32>,
    full_degree_in: Vec<u32>,
    config: McmcConfig,
    degree_config: DegreeTraceConfig,
) -> Result<(SampledNetwork, FixedStrengthDegreeBench), FixedStrengthError> {
    let fixed_pairs = problem.fixed_pairs.clone();
    let family = problem.family;
    let full_strength_out = problem.strength_out.clone();
    let full_strength_in = problem.strength_in.clone();

    let seed = config.seed;
    let mut rng = StdRng::seed_from_u64(seed);

    // ---- Residualize once (§7.2 ordering: strengths first, then degrees) ----
    let residual = problem.into_residual()?;
    let degree = residualize_degree_target(&full_degree_out, &full_degree_in, &fixed_pairs)?;
    validate_degree_target(&residual, &degree)?;
    let e_res = degree.edge_count;

    // ---- Construction + structural repair ----
    let n = residual.domain.node_count();
    let t0 = Instant::now();
    let table = initialize_table(
        &residual.strength_out,
        &residual.strength_in,
        residual.family,
        &residual.domain,
        &mut rng,
    )?;
    let mut state = StrengthState::new(n, table);
    let construction_time_s = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let (_, _, _) = repair_state(&mut state, &residual, &mut rng)?;
    let structural_repair_time_s = t0.elapsed().as_secs_f64();

    // ---- Edge-count repair to exact residual E (§14) ----
    let t0 = Instant::now();
    let edge_outcome = repair_to_edge_target(
        &mut state,
        &residual,
        &mut rng,
        e_res,
        &EdgeRepairConfig::default(),
    )?;
    let edge_repair_time_s = t0.elapsed().as_secs_f64();

    // ---- Degree repair with the shared degree-biased auxiliary step (§15) ----
    let t0 = Instant::now();
    let degree_outcome = repair_to_degree_target(
        &mut state,
        &residual,
        &mut rng,
        e_res,
        &BridgeConfig::default(),
        &degree,
        &DegreeRepairConfig::default(),
    )?;
    let degree_repair_time_s = t0.elapsed().as_secs_f64();
    if degree_outcome.best_distance != 0 {
        return Err(FixedStrengthError::DegreeRepairExhausted {
            best_degree_distance: degree_outcome.best_distance,
            restarts: degree_outcome.restarts,
            total_steps: degree_outcome.steps,
            target_edges: e_res,
        });
    }

    // ---- Runtime invariant (§26): no inexact-degree state enters sampling ----
    debug_assert_eq!(
        state.row_occ_count,
        degree.out.iter().map(|&k| k as usize).collect::<Vec<_>>(),
        "out-degree caches must equal the residual target before sampling"
    );
    debug_assert_eq!(
        state.col_occ_count,
        degree.in_.iter().map(|&k| k as usize).collect::<Vec<_>>()
    );

    // ---- Burn-in + thinning with the capped first-return degree trace ----
    let target = StrengthTarget::new(family);
    let mut f_e_counters = FixedEdgeCounters::default();
    let mut trace_counters = DegreeTraceCounters::default();
    let mut record: Vec<
        crate::generation::microcanonical::occupation_mcmc::move_cycle::Cycle4Proposal,
    > = Vec::new();
    let mut running_raw = 0u64; // start exactly on the degree fiber
    let t0 = Instant::now();
    for _ in 0..config.burn_in_sweeps.max(1) {
        degree_trace_sweep(
            &mut state,
            &target,
            &residual.domain,
            &mut rng,
            e_res,
            &BridgeConfig::default(),
            &mut f_e_counters,
            &degree,
            &degree_config,
            &mut record,
            &mut running_raw,
            &mut trace_counters,
            config.proposals_per_sweep,
        );
    }
    for _ in 0..config.sweeps_per_sample.max(1) {
        degree_trace_sweep(
            &mut state,
            &target,
            &residual.domain,
            &mut rng,
            e_res,
            &BridgeConfig::default(),
            &mut f_e_counters,
            &degree,
            &degree_config,
            &mut record,
            &mut running_raw,
            &mut trace_counters,
            config.proposals_per_sweep,
        );
    }
    let mcmc_time_s = t0.elapsed().as_secs_f64();
    debug_assert_eq!(running_raw, 0, "sampling must end on the degree fiber");

    // ---- Final residual network + merge fixed pairs + full validation (§26) ----
    let residual_network = state.to_sampled_network();
    let network = merge_fixed_pairs(residual_network, &fixed_pairs);
    validate_fixed_sk_output(
        &network,
        n,
        &full_strength_out,
        &full_strength_in,
        &full_degree_out,
        &full_degree_in,
        family,
        residual.domain.self_loops_allowed(),
    )?;

    Ok((
        network,
        FixedStrengthDegreeBench {
            construction_time_s,
            structural_repair_time_s,
            edge_repair_time_s,
            edge_repair_steps: edge_outcome.steps,
            edge_repair_restarts: edge_outcome.restarts,
            degree_repair_time_s,
            degree_repair_steps: degree_outcome.steps,
            degree_repair_restarts: degree_outcome.restarts,
            initial_degree_distance: degree_outcome.initial_distance,
            target_edges: e_res,
            mcmc_time_s,
            degree_trace: trace_counters,
            fixed_edge: f_e_counters,
        },
    ))
}

/// One-shot fixed-(s,k) sampling without diagnostics (§25).
pub fn sample_fixed_strength_degree(
    problem: FixedStrengthProblem,
    full_degree_out: Vec<u32>,
    full_degree_in: Vec<u32>,
    config: McmcConfig,
    degree_config: DegreeTraceConfig,
) -> Result<SampledNetwork, FixedStrengthError> {
    sample_fixed_strength_degree_bench(
        problem,
        full_degree_out,
        full_degree_in,
        config,
        degree_config,
    )
    .map(|(network, _)| network)
}

// --------------------------------------------------------------------------
// Gate A: trace mobility from an exact (s,k) witness (§5–§7)
// --------------------------------------------------------------------------

/// Diagnostics-only output of one fixed-(s,k) trace run started directly
/// on the exact degree fiber (`D = 0`) with no construction/repair.
#[derive(Clone, Debug)]
pub struct FixedSkTraceBenchmark {
    /// Node count.
    pub n: usize,
    /// Occupied-pair count (residual `E`).
    pub e: usize,
    /// Total residual strength (sum of all witness occupations).
    pub total_strength: OccNum,
    /// Nested degree-trace mobility counters (§27).
    pub trace: DegreeTraceCounters,
    /// Underlying fixed-(s,E) kernel counters (to attribute slow
    /// performance to `K_E` itself vs the outer degree trace).
    pub fixed_edge: FixedEdgeCounters,
    /// Wall time for the whole trace run.
    pub wall_time_s: f64,
    /// Maximum half-normalized degree distance seen by the periodic
    /// independent full scan.  Must stay `0` for a valid run.
    pub max_checkpoint_distance: u64,
}

/// Gate A diagnostic: run the existing capped first-return degree trace
/// starting from an already exact `(s,k)` state (`D = 0`), bypassing all
/// fixed-s construction, structural/edge repair, and degree repair.
///
/// The caller supplies a **witness table** — an already feasible exact
/// occupation table for `problem` whose row/column degrees equal the
/// full degree targets.  The helper validates the witness against the
/// residual problem (strengths, degrees, `E`, family capacity, domain
/// admissibility, duplicates, loop policy), recomputes `D` by an
/// independent full scan and requires `D = 0`, then executes exactly
/// `trace_attempts` top-level [`degree_trace_step`] calls of the
/// existing production trace kernel and returns counters plus wall time.
///
/// `seed` drives the trace randomness only (the witness is provided).
///
/// # Errors
///
/// - [`InvalidResidual`](FixedStrengthError::InvalidResidual) if fixed
///   pairs are present (Gate A requires none), or the witness does not
///   realize the residual strengths, violates the domain, duplicates
///   coordinates, or uses zero occupations.
/// - [`InvalidDegreeTarget`](FixedStrengthError::InvalidDegreeTarget)
///   if the residualized degree target is unbalanced or the witness
///   degrees differ from it (including `D != 0` at start or at any
///   periodic checkpoint).
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub fn benchmark_fixed_sk_trace_from_exact_table(
    problem: FixedStrengthProblem,
    full_degree_out: Vec<u32>,
    full_degree_in: Vec<u32>,
    exact_table: Vec<((u64, u64), OccNum)>,
    trace_attempts: usize,
    seed: u64,
    degree_config: DegreeTraceConfig,
) -> Result<FixedSkTraceBenchmark, FixedStrengthError> {
    let fixed_pairs = problem.fixed_pairs.clone();
    let family = problem.family;
    let full_strength_out = problem.strength_out.clone();
    let full_strength_in = problem.strength_in.clone();

    // Gate A: the diagnostic takes an already exact table; fixed-pair
    // residualization is a Part B concern (§6.2).
    if !fixed_pairs.is_empty() {
        return Err(FixedStrengthError::InvalidResidual(
            "Gate A trace diagnostic requires no fixed pairs".into(),
        ));
    }

    // Same residualization/validation as the production one-shot (§7
    // ordering: strengths first, then degrees).
    let residual = problem.into_residual()?;
    let degree = residualize_degree_target(&full_degree_out, &full_degree_in, &fixed_pairs)?;
    validate_degree_target(&residual, &degree)?;
    let e_res = degree.edge_count;

    // ---- Construct the state directly from the witness (no repair) ----
    let n = residual.domain.node_count();
    let occ_sum: OccNum = exact_table.iter().map(|&(_, o)| o).sum();
    if occ_sum != residual.total {
        return Err(FixedStrengthError::InvalidResidual(format!(
            "witness total occupation {occ_sum} != residual total {}",
            residual.total
        )));
    }
    let mut state = StrengthState::new(n, exact_table);

    // Witness validation, O(E) once, before the trace (§6.5).
    for ((s, t), _) in state.iter_occupied() {
        if !residual.domain.is_admissible(s, t) {
            return Err(FixedStrengthError::InvalidResidual(format!(
                "witness pair ({s}, {t}) is not admissible in the residual domain"
            )));
        }
    }
    validate_fixed_sk_output(
        &state.to_sampled_network(),
        n,
        &residual.strength_out,
        &residual.strength_in,
        &degree.out,
        &degree.in_,
        family,
        residual.domain.self_loops_allowed(),
    )?;
    if state.occupied_count() != e_res {
        return Err(FixedStrengthError::InvalidDegreeTarget(format!(
            "witness occupied count {} != residual edge count {e_res}",
            state.occupied_count()
        )));
    }
    let d0 = degree_distance(&state.row_occ_count, &state.col_occ_count, &degree);
    if d0 != 0 {
        return Err(FixedStrengthError::InvalidDegreeTarget(format!(
            "witness is not on the exact degree fiber: D = {d0}"
        )));
    }

    // ---- Run the existing trace kernel (§6.8–§6.10) ----
    let target = StrengthTarget::new(family);
    let mut f_e_counters = FixedEdgeCounters::default();
    let mut trace_counters = DegreeTraceCounters::default();
    let mut record: Vec<Cycle4Proposal> = Vec::new();
    let mut running_raw = 0u64; // start exactly on the degree fiber
    let mut rng = StdRng::seed_from_u64(seed);
    let mut max_checkpoint_distance = 0u64;

    let t0 = Instant::now();
    for attempt in 0..trace_attempts {
        degree_trace_step(
            &mut state,
            &target,
            &residual.domain,
            &mut rng,
            e_res,
            &BridgeConfig::default(),
            &mut f_e_counters,
            &degree,
            &degree_config,
            &mut record,
            &mut running_raw,
            &mut trace_counters,
        );
        // Every top-level trace must end on the fiber (§6.10).  The
        // kernel restores the origin on timeout, so `running_raw == 0`
        // is the O(1) invariant; verify it explicitly.
        debug_assert_eq!(running_raw, 0, "trace must end on the degree fiber");
        record.clear();
        // Periodic independent full-scan D check (O(N), once per 1000
        // traces — never per K_E step).
        if attempt % 1000 == 0 {
            let d = degree_distance(&state.row_occ_count, &state.col_occ_count, &degree);
            if d > max_checkpoint_distance {
                max_checkpoint_distance = d;
            }
        }
    }
    let wall_time_s = t0.elapsed().as_secs_f64();
    if max_checkpoint_distance != 0 {
        return Err(FixedStrengthError::InvalidDegreeTarget(format!(
            "trace left the degree fiber: max checkpoint D = {max_checkpoint_distance}"
        )));
    }

    // Final full-target validation against the stored full vectors
    // (§6.1: full targets are kept before residualization): with no
    // fixed pairs, residual == full, so this re-verifies the endpoint
    // is still an exact full (s,k) state after all traces.
    validate_fixed_sk_output(
        &state.to_sampled_network(),
        n,
        &full_strength_out,
        &full_strength_in,
        &degree.out,
        &degree.in_,
        family,
        residual.domain.self_loops_allowed(),
    )?;

    Ok(FixedSkTraceBenchmark {
        n,
        e: e_res,
        total_strength: residual.total,
        trace: trace_counters,
        fixed_edge: f_e_counters,
        wall_time_s,
        max_checkpoint_distance,
    })
}

/// O(E) boundary validation of the merged full network (§26): exact full
/// strengths, exact full degrees, unique coordinates, positive
/// occupations, family capacity, and the self-loop policy.
#[allow(clippy::too_many_arguments)]
fn validate_fixed_sk_output(
    net: &SampledNetwork,
    n: usize,
    full_s_out: &[OccNum],
    full_s_in: &[OccNum],
    full_k_out: &[u32],
    full_k_in: &[u32],
    family: OccupationFamily,
    self_loops: bool,
) -> Result<(), FixedStrengthError> {
    use std::collections::HashSet;
    let mut co = vec![0u64; n];
    let mut ci = vec![0u64; n];
    let mut ko = vec![0u32; n];
    let mut ki = vec![0u32; n];
    let mut seen = HashSet::with_capacity(net.sources.len());
    let cap = match family {
        OccupationFamily::B { layers } => layers as OccNum,
        _ => OccNum::MAX,
    };
    for ((&s, &t), &o) in net
        .sources
        .iter()
        .zip(net.targets.iter())
        .zip(net.occ_nums.iter())
    {
        let coord = (s, t);
        if !seen.insert(coord) {
            return Err(FixedStrengthError::InvalidResidual(format!(
                "duplicate output coordinate {coord:?}"
            )));
        }
        if o == 0 {
            return Err(FixedStrengthError::InvalidResidual(format!(
                "zero occupation in output at {coord:?}"
            )));
        }
        if !self_loops && s == t {
            return Err(FixedStrengthError::InvalidResidual(format!(
                "self-loop {coord:?} violates the loopless policy"
            )));
        }
        if o > cap {
            return Err(FixedStrengthError::InvalidResidual(format!(
                "occupation {o} at {coord:?} exceeds B capacity {cap}"
            )));
        }
        co[s as usize] += o;
        ci[t as usize] += o;
        ko[s as usize] += 1;
        ki[t as usize] += 1;
    }
    if co != full_s_out || ci != full_s_in {
        return Err(FixedStrengthError::InvalidResidual(format!(
            "full strengths not reproduced: out {co:?} != {full_s_out:?}, in {ci:?} != {full_s_in:?}"
        )));
    }
    if ko != full_k_out || ki != full_k_in {
        return Err(FixedStrengthError::InvalidResidual(format!(
            "full degrees not reproduced: out {ko:?} != {full_k_out:?}, in {ki:?} != {full_k_in:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod phase9_tests {
    use super::*;

    fn full_problem(
        family: OccupationFamily,
        so: Vec<OccNum>,
        si: Vec<OccNum>,
        sl: bool,
        fixed: Vec<(u64, u64, OccNum)>,
    ) -> FixedStrengthProblem {
        let domain = PairDomain::Complete {
            node_count: so.len(),
            self_loops: sl,
        };
        FixedStrengthProblem::new(family, so, si, domain, fixed).unwrap()
    }

    fn trace_config() -> DegreeTraceConfig {
        DegreeTraceConfig::default()
    }

    fn degrees_from(net: &SampledNetwork, n: usize) -> (Vec<u32>, Vec<u32>) {
        let mut ko = vec![0u32; n];
        let mut ki = vec![0u32; n];
        for ((&s, &t), _) in net
            .sources
            .iter()
            .zip(net.targets.iter())
            .zip(net.occ_nums.iter())
        {
            ko[s as usize] += 1;
            ki[t as usize] += 1;
        }
        (ko, ki)
    }

    #[test]
    fn fixed_sk_all_families_exact() {
        // Feasible (s,k) derived from explicit constructed networks
        // (§49: generated/constructed networks guarantee feasibility).
        // ME N=2 s=[3,3] E=4 k=(2,2): the all-ones state.
        // B M=2 N=2 s=[2,2] E=4 k=(2,2): all-ones with occ=1 each.
        // W M=1 N=2 s=[3,3] E=4 k=(2,2): all-ones.
        // ME N=3 s=[4,3,2] E=5 k=(1,2,2): state B (nontrivial degree repair
        // if the constructor lands elsewhere).
        type Case = (
            OccupationFamily,
            Vec<OccNum>,
            Vec<OccNum>,
            Vec<u32>,
            Vec<u32>,
            bool,
        );
        let cases: Vec<Case> = vec![
            (
                OccupationFamily::ME,
                vec![3, 3],
                vec![3, 3],
                vec![2, 2],
                vec![2, 2],
                true,
            ),
            (
                OccupationFamily::B { layers: 2 },
                vec![2, 2],
                vec![2, 2],
                vec![2, 2],
                vec![2, 2],
                true,
            ),
            (
                OccupationFamily::W { layers: 1 },
                vec![3, 3],
                vec![3, 3],
                vec![2, 2],
                vec![2, 2],
                true,
            ),
            (
                OccupationFamily::ME,
                vec![4, 3, 2],
                vec![2, 3, 4],
                vec![1, 2, 2],
                vec![2, 2, 1],
                true,
            ),
        ];
        let mut repaired_nontrivially = 0usize;
        for (family, so, si, ko, ki, sl) in cases {
            let problem = full_problem(family, so.clone(), si.clone(), sl, vec![]);
            let deg_cfg = trace_config();
            let bench = sample_fixed_strength_degree_bench(
                problem,
                ko.clone(),
                ki.clone(),
                McmcConfig::new(3, 2, 42),
                deg_cfg,
            )
            .unwrap_or_else(|e| panic!("{family:?}: {e}"))
            .1;
            assert_eq!(bench.target_edges, ko.iter().sum::<u32>() as usize);
            if bench.initial_degree_distance > 0 {
                repaired_nontrivially += 1;
            }
        }
        assert!(
            repaired_nontrivially >= 1,
            "expected at least one nontrivial degree repair among the cases"
        );
    }

    #[test]
    fn fixed_sk_reproducible_by_seed() {
        let so = vec![4u64, 3, 2];
        let si = vec![2u64, 3, 4];
        let ko = vec![1u32, 2, 2];
        let ki = vec![2u32, 2, 1];
        let run = |seed: u64| -> Vec<OccNum> {
            let problem = full_problem(OccupationFamily::ME, so.clone(), si.clone(), true, vec![]);
            let net = sample_fixed_strength_degree(
                problem,
                ko.clone(),
                ki.clone(),
                McmcConfig::new(5, 3, seed),
                trace_config(),
            )
            .unwrap();
            let mut triples: Vec<_> = net
                .sources
                .iter()
                .zip(net.targets.iter())
                .zip(net.occ_nums.iter())
                .map(|((&s, &t), &o)| (s, t, o))
                .collect();
            triples.sort_unstable();
            triples
                .into_iter()
                .flat_map(|(s, t, o)| vec![s, t, o])
                .collect()
        };
        assert_eq!(run(1), run(1), "same seed must reproduce the sample");
        let _ = run(7);
    }

    #[test]
    fn fixed_sk_positive_fixed_pair_merged_exactly() {
        // ME N=3 s=[3,3,3] k=(2,2,2)/(2,2,2) with fixed (0,1,1): residual
        // strengths [2,3,3]/[3,2,3], residual k=(1,2,2)/(2,1,2), residual
        // E=5 (explicit feasible construction above).
        let problem = full_problem(
            OccupationFamily::ME,
            vec![3, 3, 3],
            vec![3, 3, 3],
            true,
            vec![(0, 1, 1)],
        );
        let net = sample_fixed_strength_degree(
            problem,
            vec![2, 2, 2],
            vec![2, 2, 2],
            McmcConfig::new(3, 2, 7),
            trace_config(),
        )
        .unwrap();
        // Full degrees sum to 6; residual E=5 plus the fixed pair => 6.
        assert_eq!(net.sources.len(), 6, "full E = residual 5 + fixed pair");
        // Fixed pair present with the right occupation; no duplicates.
        let mut co = vec![0u64; 3];
        let mut ci = vec![0u64; 3];
        let mut keys = Vec::new();
        for ((&s, &t), &o) in net
            .sources
            .iter()
            .zip(net.targets.iter())
            .zip(net.occ_nums.iter())
        {
            co[s as usize] += o;
            ci[t as usize] += o;
            keys.push((s, t));
        }
        assert_eq!(co, vec![3, 3, 3]);
        assert_eq!(ci, vec![3, 3, 3]);
        let (ko, ki) = degrees_from(&net, 3);
        assert_eq!(ko, vec![2, 2, 2]);
        assert_eq!(ki, vec![2, 2, 2]);
        assert!(keys.contains(&(0, 1)));
        let occ01 = net
            .sources
            .iter()
            .zip(net.targets.iter())
            .zip(net.occ_nums.iter())
            .find(|((&s, &t), _)| s == 0 && t == 1)
            .map(|(_, &o)| o);
        assert_eq!(occ01, Some(1));
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), keys.len(), "duplicate output coordinates");
    }

    #[test]
    fn fixed_sk_zero_fixed_pair_never_reoccupied() {
        // ME N=3 s=[2,2,2] k=(1,1,1)/(1,1,1) with fixed (0,1,0):
        // coordinate (0,1) excluded; output never occupies it and the
        // full degrees/strengths stay exact.
        let problem = full_problem(
            OccupationFamily::ME,
            vec![2, 2, 2],
            vec![2, 2, 2],
            true,
            vec![(0, 1, 0)],
        );
        let net = sample_fixed_strength_degree(
            problem,
            vec![1, 1, 1],
            vec![1, 1, 1],
            McmcConfig::new(3, 2, 11),
            trace_config(),
        )
        .unwrap();
        assert_eq!(net.sources.len(), 3);
        assert!(
            !net.sources
                .iter()
                .zip(net.targets.iter())
                .any(|(&s, &t)| s == 0 && t == 1),
            "zero fixed pair was reoccupied"
        );
        let (ko, ki) = degrees_from(&net, 3);
        assert_eq!(ko, vec![1, 1, 1]);
        assert_eq!(ki, vec![1, 1, 1]);
    }

    #[test]
    fn fixed_sk_invalid_degree_target_rejected() {
        // k_out[0] = 3 > s_out[0] = 2 must fail validation before sampling.
        let problem = full_problem(OccupationFamily::ME, vec![2, 2], vec![2, 2], true, vec![]);
        match sample_fixed_strength_degree(
            problem,
            vec![3, 1],
            vec![2, 2],
            McmcConfig::new(2, 1, 1),
            trace_config(),
        ) {
            Err(FixedStrengthError::InvalidDegreeTarget(msg)) => {
                assert!(msg.contains("exceeds strength_out"), "{msg}")
            }
            other => panic!("expected InvalidDegreeTarget, got {other:?}"),
        }
    }
}

// --------------------------------------------------------------------------
// Gate A unit tests: trace-from-exact-witness diagnostic (§6)
// --------------------------------------------------------------------------

#[cfg(test)]
mod gate_a_tests {
    use super::*;

    /// N=3 ME witness: `s_out=(3,1,1)`, `s_in=(1,3,1)`, `k_out=(2,1,1)`,
    /// `k_in=(1,2,1)`, `E=4`, `D=0` by construction.
    #[allow(clippy::type_complexity)]
    fn me_3_witness() -> (
        FixedStrengthProblem,
        Vec<u32>,
        Vec<u32>,
        Vec<((u64, u64), OccNum)>,
    ) {
        let table = vec![((0, 1), 2), ((0, 2), 1), ((1, 0), 1), ((2, 1), 1)];
        let problem = FixedStrengthProblem::new(
            OccupationFamily::ME,
            vec![3, 1, 1],
            vec![1, 3, 1],
            PairDomain::Complete {
                node_count: 3,
                self_loops: true,
            },
            vec![],
        )
        .unwrap();
        (problem, vec![2, 1, 1], vec![1, 2, 1], table)
    }

    #[test]
    fn trace_from_exact_witness_runs_and_stays_on_fiber() {
        let (problem, k_out, k_in, table) = me_3_witness();
        let bench = benchmark_fixed_sk_trace_from_exact_table(
            problem,
            k_out,
            k_in,
            table,
            2000,
            7,
            DegreeTraceConfig {
                lambda: 1.0,
                max_steps: 16,
            },
        )
        .unwrap();
        assert_eq!(bench.n, 3);
        assert_eq!(bench.e, 4);
        assert_eq!(bench.total_strength, 5);
        assert_eq!(bench.trace.trace_attempts, 2000);
        assert_eq!(
            bench.max_checkpoint_distance, 0,
            "trace must never leave D=0"
        );
        // Every top-level attempt is exactly one of: timeout (self-loop),
        // step-1 return (in-fiber move), or departed successful return
        // (returned to D=0 after leaving).  Disjoint by construction.
        assert_eq!(
            bench.trace.trace_attempts,
            bench.trace.timeouts + bench.trace.step1_returns + bench.trace.successful_returns,
        );
        assert!(bench.wall_time_s >= 0.0);
    }

    #[test]
    fn trace_from_exact_witness_strength_check() {
        // Witness with wrong total occupation must be rejected before the
        // trace runs.
        let (problem, k_out, k_in, _) = me_3_witness();
        let bad_table = vec![((0, 1), 2), ((1, 0), 1), ((2, 1), 1)];
        match benchmark_fixed_sk_trace_from_exact_table(
            problem,
            k_out,
            k_in,
            bad_table,
            16,
            7,
            DegreeTraceConfig::default(),
        ) {
            Err(FixedStrengthError::InvalidResidual(msg)) => {
                assert!(msg.contains("total occupation"), "{msg}")
            }
            other => panic!("expected InvalidResidual, got {other:?}"),
        }
    }

    #[test]
    fn trace_from_exact_witness_degree_check() {
        // A table with the same strengths but a different degree vector
        // (5 all-ones edges vs the 4-edge witness) must be rejected: it
        // has the correct total mass and strengths, but E/k differ.
        let (problem, k_out, k_in, _) = me_3_witness();
        let bad_table = vec![
            ((0, 0), 1),
            ((0, 1), 1),
            ((0, 2), 1),
            ((1, 1), 1),
            ((2, 1), 1),
        ];
        match benchmark_fixed_sk_trace_from_exact_table(
            problem,
            k_out,
            k_in,
            bad_table,
            16,
            7,
            DegreeTraceConfig::default(),
        ) {
            Err(e) => {
                let msg = e.to_string();
                assert!(msg.contains("degree"), "unexpected error: {msg}")
            }
            other => panic!("expected a degree error, got {other:?}"),
        }
    }
}
