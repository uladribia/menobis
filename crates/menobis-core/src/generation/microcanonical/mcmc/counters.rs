//! Shared MCMC proposal / acceptance counters.

/// Counters collected during MCMC execution.
///
/// These are updated by the chain's `step()` method and reported through
/// diagnostics or benchmarks.
#[derive(Clone, Debug, Default)]
pub struct McmcCounters {
    /// Total number of proposal attempts.
    pub proposals: u64,
    /// Number of accepted proposals.
    pub accepted: u64,
    /// Number of held proposals (invalid moves that preserve the state).
    pub held: u64,
    /// Number of Metropolis-rejected proposals (valid moves that were
    /// stochastically rejected).
    pub metropolis_rejected: u64,
}

impl McmcCounters {
    /// Create a new zeroed counter set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Acceptance rate: `accepted / proposals`.
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposals == 0 {
            0.0
        } else {
            self.accepted as f64 / self.proposals as f64
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&mut self) {
        self.proposals = 0;
        self.accepted = 0;
        self.held = 0;
        self.metropolis_rejected = 0;
    }
}
