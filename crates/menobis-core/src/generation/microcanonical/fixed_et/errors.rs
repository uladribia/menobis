//! Shared errors for the fixed-(E,T) microcanonical family.

/// Errors that can occur during fixed-(E,T) sampling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedETError {
    /// The residual problem is infeasible.
    InvalidResidual(String),
}

impl std::fmt::Display for FixedETError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidResidual(msg) => write!(f, "invalid residual problem: {msg}"),
        }
    }
}

impl std::error::Error for FixedETError {}
