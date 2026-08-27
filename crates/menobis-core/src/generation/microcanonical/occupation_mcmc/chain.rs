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
use super::move_cycle::occupied_cycle4_step;
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
