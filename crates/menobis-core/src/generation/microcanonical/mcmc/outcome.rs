//! Outcome of a single MCMC step.

/// The outcome of a single MCMC proposal step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum McmcOutcome {
    /// The proposal was accepted and the state was updated.
    Accepted,
    /// The proposal was invalid; the state is retained (switch-and-hold).
    Held,
    /// The proposal was valid but rejected by the Metropolis criterion.
    Rejected,
}
