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

// ---------------------------------------------------------------------------
// Cost errors — not PartialEq/Eq because they carry f64 fields.
// ---------------------------------------------------------------------------

/// Errors during fixed-strength cost-constrained sampling.
#[derive(Clone, Debug)]
pub enum FixedStrengthCostError {
    /// A cost provider returned `None` for an admissible pair.
    MissingCost { source: u64, target: u64 },
    /// A cost provider returned a non-finite value (NaN, +inf, -inf).
    NonFiniteCost {
        source: u64,
        target: u64,
        value: f64,
    },
    /// The observed (target) cost is non-finite.
    NonFiniteObservedCost { value: f64 },
    /// Cost is constant over the feasible state space — gamma is not
    /// identifiable.
    CostNotIdentifiable,
    /// A gamma value is invalid (e.g., W family with gamma that would
    /// push q outside (0,1) — currently unused but reserved).
    InvalidGamma { value: f64, message: String },
    /// The bracketing interval does not contain the target expected cost.
    InvalidBracket {
        lower: f64,
        upper: f64,
        message: String,
    },
    /// Could not find a valid bracket after maximum expansions.
    BracketNotFound,
    /// The fit did not converge to the required tolerance.
    FitDidNotConverge { iterations: usize, residual: f64 },
    /// Fixed-pair cost does not match the residual/total decomposition.
    ResidualCostInconsistent {
        total: f64,
        fixed: f64,
        residual: f64,
    },
    /// Wraps a non-cost [`FixedStrengthError`].
    FixedStrength(FixedStrengthError),
}

impl fmt::Display for FixedStrengthCostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCost { source, target } => {
                write!(
                    f,
                    "cost provider returned None for admissible pair ({source}, {target})"
                )
            }
            Self::NonFiniteCost {
                source,
                target,
                value,
            } => {
                write!(
                    f,
                    "non-finite cost {value} from provider for pair ({source}, {target})"
                )
            }
            Self::NonFiniteObservedCost { value } => {
                write!(f, "observed (target) cost is non-finite: {value}")
            }
            Self::CostNotIdentifiable => {
                write!(
                    f,
                    "cost is constant over the feasible state space; gamma is not identifiable"
                )
            }
            Self::InvalidGamma { value, message } => {
                write!(f, "invalid gamma {value}: {message}")
            }
            Self::InvalidBracket {
                lower,
                upper,
                message,
            } => {
                write!(f, "invalid bracket [{lower}, {upper}]: {message}")
            }
            Self::BracketNotFound => {
                write!(
                    f,
                    "could not find a valid gamma bracket after maximum expansions"
                )
            }
            Self::FitDidNotConverge {
                iterations,
                residual,
            } => {
                write!(f, "gamma fit did not converge after {iterations} iterations (residual {residual:.4e})")
            }
            Self::ResidualCostInconsistent {
                total,
                fixed,
                residual,
            } => {
                write!(f, "cost residual mismatch: total {total:.6e} != fixed {fixed:.6e} + residual {residual:.6e}")
            }
            Self::FixedStrength(e) => write!(f, "fixed-strength error: {e}"),
        }
    }
}

impl std::error::Error for FixedStrengthCostError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FixedStrength(e) => Some(e),
            _ => None,
        }
    }
}
