//! Support MCMC sampler for directed fixed-degree sequences.
//!
//! Provides:
//! - `FixedDegreeMcmcConfig` — burn-in and thinning parameters.
//! - `FixedDegreeChain` — persistent MCMC chain with step/sweep/sample.
//! - `sample_fixed_degree_support` — one-shot convenience function.

use rand::Rng;
use rand::SeedableRng;

use super::diagnostics::{self, DegreeHeterogeneity, FixedDegreeDiagnostics, RepresentationMode};
use super::errors::FixedKTError;
use super::initializer::greedy_directed_initialize;
use super::state::DegreeSupportState;
use super::switch::{directed_switch_step, SwitchOutcome};

/// MCMC configuration for the fixed-degree support sampler.
#[derive(Clone, Debug)]
pub struct FixedDegreeMcmcConfig {
    /// Number of sweeps for burn-in.
    pub burn_in_sweeps: usize,
    /// Number of sweeps between output samples.
    pub sweeps_per_sample: usize,
    /// Optional override for proposals per sweep.  `None` means auto.
    pub proposals_per_sweep: Option<usize>,
    /// Seed for the RNG (deterministic reproducibility).
    pub seed: u64,
}

impl FixedDegreeMcmcConfig {
    /// Default configuration based on heterogeneity classification.
    pub fn default_for_heterogeneity(het: &DegreeHeterogeneity) -> Self {
        match het {
            DegreeHeterogeneity::Light => Self {
                burn_in_sweeps: 20,
                sweeps_per_sample: 5,
                seed: 0,
                proposals_per_sweep: None,
            },
            DegreeHeterogeneity::Heterogeneous => Self {
                burn_in_sweeps: 50,
                sweeps_per_sample: 10,
                seed: 0,
                proposals_per_sweep: None,
            },
            DegreeHeterogeneity::HubDominated => Self {
                burn_in_sweeps: 100,
                sweeps_per_sample: 20,
                seed: 0,
                proposals_per_sweep: None,
            },
        }
    }
}

impl Default for FixedDegreeMcmcConfig {
    fn default() -> Self {
        Self {
            burn_in_sweeps: 50,
            sweeps_per_sample: 10,
            seed: 0,
            proposals_per_sweep: None,
        }
    }
}

/// Persistent MCMC chain for fixed-degree support sampling.
pub struct FixedDegreeChain {
    pub state: DegreeSupportState,
    pub diagnostics: FixedDegreeDiagnostics,
    pub config: FixedDegreeMcmcConfig,
}

impl FixedDegreeChain {
    /// Create a new chain from pre-built support state.
    pub fn new(state: DegreeSupportState, config: FixedDegreeMcmcConfig) -> Self {
        Self {
            state,
            diagnostics: FixedDegreeDiagnostics::new(),
            config,
        }
    }

    /// Perform one MCMC step (one directed double-edge switch proposal).
    pub fn step(
        &mut self,
        rng: &mut impl Rng,
        admissible_pairs: Option<&[(u64, u64)]>,
    ) -> SwitchOutcome {
        self.diagnostics.mcmc.proposals += 1;
        let outcome = directed_switch_step(&mut self.state, false, rng, admissible_pairs);
        match outcome {
            SwitchOutcome::Switched => {
                self.diagnostics.mcmc.accepted += 1;
            }
            SwitchOutcome::Hold => {
                // We don't track hold subtypes here for simplicity;
                // the switch module doesn't return subtypes.
            }
        }
        outcome
    }

    /// Perform one sweep (E proposal attempts).
    pub fn sweep(&mut self, rng: &mut impl Rng, admissible_pairs: Option<&[(u64, u64)]>) {
        let m = self.state.edge_count().max(1);
        for _ in 0..m {
            self.step(rng, admissible_pairs);
        }
    }

    /// Run burn-in.
    pub fn burn_in(&mut self, rng: &mut impl Rng, admissible_pairs: Option<&[(u64, u64)]>) {
        let sweeps = self.config.burn_in_sweeps.max(1);
        for _ in 0..sweeps {
            self.sweep(rng, admissible_pairs);
        }
    }

    /// After burn-in, perform thinning sweeps and return a reference to the
    /// current support.
    pub fn sample_support(
        &mut self,
        rng: &mut impl Rng,
        admissible_pairs: Option<&[(u64, u64)]>,
    ) -> &[(u64, u64)] {
        let sweeps = self.config.sweeps_per_sample.max(1);
        for _ in 0..sweeps {
            self.sweep(rng, admissible_pairs);
        }
        &self.state.edges
    }
}

/// One-shot convenience: build a support chain, burn in, and return one sample.
///
/// This is the primary entry point for the fixed-degree support kernel.
///
/// # Arguments
///
/// * `out_degrees` — target out-degree sequence.
/// * `in_degrees` — target in-degree sequence.
/// * `self_loops` — whether self-loops are allowed.
/// * `config` — MCMC configuration (burn-in, thinning, seed).
///
/// # Returns
///
/// A `(DegreeSupportState, FixedDegreeDiagnostics)` tuple containing the
/// final support state and diagnostics.
pub fn sample_fixed_degree_support(
    out_degrees: &[u32],
    in_degrees: &[u32],
    self_loops: bool,
    config: &FixedDegreeMcmcConfig,
    admissible_pairs: Option<&[(u64, u64)]>,
) -> Result<(DegreeSupportState, FixedDegreeDiagnostics), FixedKTError> {
    let n = out_degrees.len();
    if n == 0 {
        return Err(FixedKTError::InvalidResidual(
            "empty degree sequence".into(),
        ));
    }

    // Heterogeneity classification
    let het = diagnostics::classify_heterogeneity(out_degrees, in_degrees);

    // Complement mode decision
    let (work_out, work_in, repr, _work_e) =
        diagnostics::maybe_complement(out_degrees, in_degrees, self_loops);

    // Initialize
    let state = greedy_directed_initialize(&work_out, &work_in, self_loops, admissible_pairs)?;

    // Build chain
    let effective_config = FixedDegreeMcmcConfig {
        burn_in_sweeps: config.burn_in_sweeps,
        sweeps_per_sample: config.sweeps_per_sample,
        seed: config.seed,
        proposals_per_sweep: None,
    };
    let mut chain = FixedDegreeChain::new(state, effective_config);
    chain.diagnostics.representation = repr;
    chain.diagnostics.heterogeneity = het;

    // RNG
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

    // Burn in
    chain.burn_in(&mut rng, admissible_pairs);

    // Sample (one after burn-in)
    chain.sample_support(&mut rng, admissible_pairs);

    // If complement mode, invert back to original representation
    if repr == RepresentationMode::Complement {
        invert_complement(&mut chain.state, self_loops);
    }

    Ok((chain.state, chain.diagnostics))
}

/// Invert a complement representation back to the original support.
///
/// Builds the complement edge list directly and replaces the state's
/// edges, dropping the stale adjacency structures (O(N²) memory).  The
/// resulting state is only valid for edge extraction, not for further
/// MCMC steps or `contains()` queries.
fn invert_complement(state: &mut DegreeSupportState, self_loops: bool) {
    let n = state.node_count;

    // Build the complement: all ordered pairs not in the current state.
    let mut new_edges = Vec::with_capacity(n.saturating_mul(n.saturating_sub(1)));
    for i in 0..(n as u64) {
        for j in 0..(n as u64) {
            if !self_loops && i == j {
                continue;
            }
            if !state.contains(&(i, j)) {
                new_edges.push((i, j));
            }
        }
    }
    state.replace_edges(new_edges);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directed_cycle_chain() {
        let out = vec![1u32, 1, 1, 1];
        let inp = vec![1u32, 1, 1, 1];
        let config = FixedDegreeMcmcConfig {
            burn_in_sweeps: 10,
            sweeps_per_sample: 5,
            seed: 42,
            proposals_per_sweep: None,
        };
        let (state, diag) = sample_fixed_degree_support(&out, &inp, false, &config, None).unwrap();
        assert_eq!(state.edge_count(), 4);
        assert_eq!(state.out_degree_sequence(), out);
        assert!(diag.acceptance_rate() >= 0.0);
    }

    #[test]
    fn star_chain() {
        let n = 6;
        let mut out = vec![0u32; n];
        out[0] = (n - 1) as u32;
        let mut inp = vec![0u32; n];
        for item in inp.iter_mut().skip(1) {
            *item = 1;
        }
        let config = FixedDegreeMcmcConfig {
            burn_in_sweeps: 10,
            sweeps_per_sample: 5,
            seed: 99,
            proposals_per_sweep: None,
        };
        let (state, _diag) = sample_fixed_degree_support(&out, &inp, false, &config, None).unwrap();
        assert_eq!(state.edge_count(), n - 1);
        assert_eq!(state.out_degree_sequence(), out);
    }

    #[test]
    fn complement_mode() {
        // Near-complete directed graph: complement should be used.
        let n = 5;
        let max_allowed = n - 1;
        // Each node has out-degree = max_allowed - 1 (only 1 missing)
        let out = vec![(max_allowed - 1) as u32; n];
        let inp = vec![(max_allowed - 1) as u32; n];
        let config = FixedDegreeMcmcConfig {
            burn_in_sweeps: 5,
            sweeps_per_sample: 2,
            seed: 7,
            proposals_per_sweep: None,
        };
        let (state, diag) = sample_fixed_degree_support(&out, &inp, false, &config, None).unwrap();
        assert_eq!(state.edge_count(), n * (n - 1) - n);
        assert_eq!(diag.representation, RepresentationMode::Complement);
    }

    #[test]
    fn deterministic_reproducibility() {
        let out = vec![2u32, 1, 1];
        let inp = vec![1u32, 2, 1];
        let config = FixedDegreeMcmcConfig {
            burn_in_sweeps: 5,
            sweeps_per_sample: 2,
            seed: 42,
            proposals_per_sweep: None,
        };
        let (state_a, _) = sample_fixed_degree_support(&out, &inp, false, &config, None).unwrap();
        let (state_b, _) = sample_fixed_degree_support(&out, &inp, false, &config, None).unwrap();
        assert_eq!(state_a.edges, state_b.edges);
    }
}
