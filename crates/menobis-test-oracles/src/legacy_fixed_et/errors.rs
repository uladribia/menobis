//! Shared errors for the fixed-(E,T) microcanonical family.

/// Errors that can occur during fixed-(E,T) sampling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedETError {
    /// The fallback DP table would require too much memory.
    TableTooLarge {
        t: usize,
        e: usize,
        required_cells: usize,
        max_cells: usize,
        family: &'static str,
    },
    /// The residual problem is infeasible.
    InvalidResidual(String),
    /// No backend available (rejection exhausted, table too large).
    NoBackendAvailable,
}

impl std::fmt::Display for FixedETError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TableTooLarge {
                t,
                e,
                required_cells,
                max_cells,
                family,
            } => {
                write!(
                    f,
                    "{family} table {t}×{e} requires {required_cells} cells (limit {max_cells})"
                )
            }
            Self::InvalidResidual(msg) => write!(f, "invalid residual problem: {msg}"),
            Self::NoBackendAvailable => write!(f, "no sampling backend available"),
        }
    }
}

impl std::error::Error for FixedETError {}
