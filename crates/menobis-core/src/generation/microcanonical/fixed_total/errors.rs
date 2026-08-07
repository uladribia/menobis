//! Error type for the fixed-total occupation sampler.

use std::fmt;

/// Errors produced by the fixed-total occupation sampler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedTotalError {
    /// The residual problem is infeasible (e.g., T < E, B capacity exceeded).
    InvalidResidual(String),
}

impl fmt::Display for FixedTotalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidResidual(msg) => write!(f, "invalid residual fixed-total problem: {msg}"),
        }
    }
}

impl std::error::Error for FixedTotalError {}
