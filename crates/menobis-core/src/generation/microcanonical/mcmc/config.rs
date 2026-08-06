//! Shared MCMC configuration.

/// Shared MCMC configuration for microcanonical chains.
///
/// Controls burn-in length, thinning interval, proposals per sweep, and RNG
/// seeding.
///
/// The number of proposals per sweep defaults to
/// `max(occupied_pairs, 2 * node_count, 1)` when `proposals_per_sweep` is
/// `None`.
#[derive(Clone, Debug)]
pub struct McmcConfig {
    /// Number of sweeps during burn-in.
    pub burn_in_sweeps: usize,
    /// Number of sweeps between successive samples (thinning).
    pub sweeps_per_sample: usize,
    /// Optional override for proposals per sweep.  `None` means auto.
    pub proposals_per_sweep: Option<usize>,
    /// Seed for deterministic RNG initialization.
    pub seed: u64,
}

impl McmcConfig {
    pub fn new(burn_in_sweeps: usize, sweeps_per_sample: usize, seed: u64) -> Self {
        Self {
            burn_in_sweeps,
            sweeps_per_sample,
            proposals_per_sweep: None,
            seed,
        }
    }
}

impl Default for McmcConfig {
    fn default() -> Self {
        Self {
            burn_in_sweeps: 50,
            sweeps_per_sample: 10,
            proposals_per_sweep: None,
            seed: 0,
        }
    }
}
