//! Persistent MCMC chain for fixed-strength sampling.
//!
//! Provides:
//! - `FixedStrengthChain` — stateful chain with step/sweep/burn_in/sample.
//! - `sample_fixed_strength` — one-shot convenience orchestrator (Phase 4).
//! - `sample_fixed_strength_with_cost` — cost-constrained entry point (Phase 5).

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
use crate::pairs::PairCostProvider;
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

/// Check whether the ME direct backend can be used.
fn can_use_me_direct(
    family: OccupationFamily,
    self_loops: bool,
    domain: &PairDomain,
    has_fixed_pairs: bool,
    total: OccNum,
    target: &StrengthTarget,
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
    // Cost-constrained problems must use MCMC (Phase 5).
    if target.costs.is_some() {
        return false;
    }
    true
}

/// One-shot fixed-strength sampling (Phase 4, no cost).
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

    // Phase 4 target: no cost.
    let target = StrengthTarget::new(family);

    // Try ME direct fast path.
    if can_use_me_direct(
        family,
        self_loops,
        &problem.domain,
        has_fixed_pairs,
        total,
        &target,
    ) {
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

    // Build chain.
    let mut chain = FixedStrengthChain::new(state, target, problem.domain, config);

    // Run chain.
    let mut rng = StdRng::seed_from_u64(seed);
    chain.burn_in(&mut rng);
    let network = chain.sample(&mut rng);

    Ok((network, StrengthBackend::CycleMcmc))
}

/// Fixed-strength sampling with a cost provider (Phase 5).
///
/// The target starts with gamma=0.  The caller should call
/// `chain.set_target(...)` with a fitted gamma before sampling.
///
/// Unlike [`sample_fixed_strength`], this always uses the MCMC backend
/// because ME direct is incompatible with cost constraints.
///
/// # Errors
///
/// Returns [`FixedStrengthError`] if the problem is infeasible.
pub fn sample_fixed_strength_with_cost<'a>(
    problem: ResidualStrengthProblem,
    costs: &'a dyn PairCostProvider,
    config: McmcConfig,
    _has_fixed_pairs: bool,
) -> Result<(FixedStrengthChain<'a>, StrengthBackend), FixedStrengthError> {
    problem.validate()?;

    let family = problem.family;

    // Initialize occupation table.
    let table = initialize_table(
        &problem.strength_out,
        &problem.strength_in,
        family,
        &problem.domain,
    )?;

    // Build state.
    let state = StrengthState::new(problem.domain.node_count(), table);

    // Build target with cost provider (gamma starts at 0.0).
    let target = StrengthTarget::with_costs(family, costs);

    // Build chain.
    let chain = FixedStrengthChain::new(state, target, problem.domain, config);

    Ok((chain, StrengthBackend::CycleMcmc))
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

    #[test]
    fn set_target_updates_gamma() {
        let prob = make_problem(OccupationFamily::Poisson, vec![5, 5], vec![5, 5], true);
        let config = McmcConfig::new(10, 5, 42);
        let mut chain = {
            let target = StrengthTarget::new(OccupationFamily::Poisson);
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
            let mut t = StrengthTarget::new(OccupationFamily::Poisson);
            t.set_gamma(2.5);
            t
        };
        chain.set_target(new_target);
        assert!((chain.target.gamma() - 2.5).abs() < 1e-12);
    }
}
