//! Errors for the fixed-\((\mathbf k,T)\) microcanonical sampler.

use std::fmt;

/// Errors that can occur during fixed-\((\mathbf k,T)\) sampling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedKTError {
    /// The residual problem is infeasible.
    InvalidResidual(String),
    /// Not enough edges to perform a switch.
    TooFewEdges,
    /// No feasible support could be constructed.
    InitializationFailed(String),
    /// Occupation allocation failed (delegated from FixedETError).
    OccupationError(String),
    /// Configuration error.
    ConfigError(String),
}

impl fmt::Display for FixedKTError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidResidual(msg) => write!(f, "invalid residual problem: {msg}"),
            Self::TooFewEdges => write!(f, "need at least 2 edges for a switch"),
            Self::InitializationFailed(msg) => write!(f, "support initialization failed: {msg}"),
            Self::OccupationError(msg) => write!(f, "occupation allocation failed: {msg}"),
            Self::ConfigError(msg) => write!(f, "configuration error: {msg}"),
        }
    }
}

impl std::error::Error for FixedKTError {}