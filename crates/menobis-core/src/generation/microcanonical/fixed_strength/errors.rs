//! Errors for the fixed-strength microcanonical sampler.

use std::fmt;

/// Errors that can occur during fixed-strength sampling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedStrengthError {
    /// The residual problem is infeasible.
    InvalidResidual(String),
    /// Too large for the direct stub-matching backend.
    TooLargeForDirect(u64),
    /// The direct stub-matching backend cannot handle this configuration
    /// (e.g., self-loops forbidden, masked domain, or fixed cells).
    DirectNotApplicable(String),
    /// Total stub count overflowed usize.
    ArithmeticOverflow(String),
    /// Initialization via max flow or greedy construction failed.
    InitializationFailed(String),
    /// An internal error occurred during MCMC.
    McmcError(String),
}

impl fmt::Display for FixedStrengthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidResidual(msg) => write!(f, "invalid residual problem: {msg}"),
            Self::TooLargeForDirect(t) => {
                write!(f, "total stubs {t} exceeds maximum for direct backend")
            }
            Self::DirectNotApplicable(msg) => {
                write!(f, "direct stub-matching not applicable: {msg}")
            }
            Self::ArithmeticOverflow(msg) => write!(f, "arithmetic overflow: {msg}"),
            Self::InitializationFailed(msg) => write!(f, "initialization failed: {msg}"),
            Self::McmcError(msg) => write!(f, "MCMC error: {msg}"),
        }
    }
}

impl std::error::Error for FixedStrengthError {}
