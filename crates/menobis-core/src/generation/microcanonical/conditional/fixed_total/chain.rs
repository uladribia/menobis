//! Pair-Gibbs chain for fixed-total occupation sampling.
//!
//! The chain redraws the split of two uniformly chosen cells from the
//! exact family conditional.  Each step is a Gibbs update: it preserves
//! total `T`, positivity, and B capacity, has acceptance probability
//! one, and satisfies detailed balance exactly.
//!
//! Memory is `O(E)`.  A sweep performs `E` pair-Gibbs steps.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use super::errors::FixedTotalError;
use super::initializer::initialize_balanced;
use super::pair_conditional::sample_split;
use super::state::FixedTotalState;
use crate::generation::microcanonical::mcmc::{McmcConfig, McmcCounters};
use crate::model::family::OccupationFamily;
use crate::OccNum;

/// Persistent pair-Gibbs chain for a fixed-total occupation vector.
pub struct FixedTotalChain {
    /// Current occupation state.
    pub state: FixedTotalState,
    /// Family law used by the conditionals.
    pub family: OccupationFamily,
    /// Burn-in / thinning configuration.
    pub config: McmcConfig,
    /// Proposal and acceptance counters.
    pub counters: McmcCounters,
}

impl FixedTotalChain {
    /// Create a chain from an existing feasible state.
    pub fn new(state: FixedTotalState, family: OccupationFamily, config: McmcConfig) -> Self {
        Self {
            state,
            family,
            config,
            counters: McmcCounters::new(),
        }
    }

    /// One pair-Gibbs step: redraw the split of two uniformly chosen cells.
    ///
    /// Always accepted (Gibbs).  For `E < 2` the step is a no-op.
    pub fn step(&mut self, rng: &mut impl Rng) {
        self.counters.proposals += 1;
        let e = self.state.len();
        if e < 2 {
            return;
        }
        let i = rng.random_range(0..e);
        let mut j = rng.random_range(0..e - 1);
        if j >= i {
            j += 1;
        }
        let q = self.state.get(i) + self.state.get(j);
        let (a, b) = sample_split(self.family, q, rng);
        self.state.set_pair(i, j, a, b);
        self.counters.accepted += 1;
    }

    /// One sweep: `E` pair-Gibbs steps (or the config override).
    pub fn sweep(&mut self, rng: &mut impl Rng) {
        let e = self.state.len().max(1);
        let per_sweep = self.config.proposals_per_sweep.unwrap_or(e);
        for _ in 0..per_sweep {
            self.step(rng);
        }
    }

    /// Run `burn_in_sweeps` sweeps.
    pub fn burn_in(&mut self, rng: &mut impl Rng) {
        for _ in 0..self.config.burn_in_sweeps.max(1) {
            self.sweep(rng);
        }
    }

    /// Run `sweeps_per_sample` thinning sweeps and return a copy of the
    /// current occupation vector.
    pub fn sample(&mut self, rng: &mut impl Rng) -> Vec<OccNum> {
        for _ in 0..self.config.sweeps_per_sample.max(1) {
            self.sweep(rng);
        }
        self.state.occupations().to_vec()
    }
}

/// One-shot fixed-total sampling: initialize, burn in, return one vector.
///
/// Handles the trivial cases `E = 0` (empty) and `E = 1` (single cell)
/// directly without running a chain.
///
/// # Arguments
///
/// * `family` — ME, B, or W.
/// * `e` — number of occupied cells.
/// * `t` — total occupation `T`.
/// * `config` — MCMC configuration (seed, burn-in, thinning).
pub fn sample_fixed_total(
    family: OccupationFamily,
    e: usize,
    t: OccNum,
    config: &McmcConfig,
) -> Result<Vec<OccNum>, FixedTotalError> {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let occ = initialize_balanced(family, e, t, &mut rng)?;
    if e <= 1 {
        return Ok(occ);
    }
    let mut chain = FixedTotalChain::new(FixedTotalState::new(occ), family, config.clone());
    chain.burn_in(&mut rng);
    Ok(chain.sample(&mut rng))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(seed: u64) -> McmcConfig {
        McmcConfig {
            burn_in_sweeps: 10,
            sweeps_per_sample: 5,
            proposals_per_sweep: None,
            seed,
        }
    }

    #[test]
    fn total_and_positivity_preserved() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut chain = FixedTotalChain::new(
            FixedTotalState::new(vec![1, 3, 2, 4]),
            OccupationFamily::ME,
            config(1),
        );
        for _ in 0..200 {
            chain.step(&mut rng);
            assert_eq!(chain.state.total(), 10);
            assert!(chain.state.occupations().iter().all(|&t| t >= 1));
        }
    }

    #[test]
    fn b_cap_preserved() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut chain = FixedTotalChain::new(
            FixedTotalState::new(vec![2, 2, 3, 3]),
            OccupationFamily::B { layers: 3 },
            config(2),
        );
        for _ in 0..200 {
            chain.step(&mut rng);
            assert_eq!(chain.state.total(), 10);
            assert!(chain
                .state
                .occupations()
                .iter()
                .all(|&t| (1..=3).contains(&t)));
        }
    }

    #[test]
    fn reproducible_one_shot() {
        let a = sample_fixed_total(OccupationFamily::ME, 5, 15, &config(42)).unwrap();
        let b = sample_fixed_total(OccupationFamily::ME, 5, 15, &config(42)).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.iter().sum::<OccNum>(), 15);
        assert!(a.iter().all(|&t| t >= 1));
    }

    #[test]
    fn trivial_cases() {
        assert_eq!(
            sample_fixed_total(OccupationFamily::ME, 0, 0, &config(1)).unwrap(),
            vec![]
        );
        assert_eq!(
            sample_fixed_total(OccupationFamily::ME, 1, 7, &config(1)).unwrap(),
            vec![7]
        );
        assert!(sample_fixed_total(OccupationFamily::ME, 0, 3, &config(1)).is_err());
    }

    #[test]
    fn one_cell_step_is_noop() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut chain = FixedTotalChain::new(
            FixedTotalState::new(vec![9]),
            OccupationFamily::ME,
            config(3),
        );
        chain.step(&mut rng);
        assert_eq!(chain.state.occupations(), &[9]);
    }
}
