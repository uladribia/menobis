//! Errors for the fixed-strength microcanonical sampler.

use std::fmt;

use crate::OccNum;

/// Errors that can occur during fixed-strength sampling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedStrengthError {
    /// The residual problem is infeasible.
    InvalidResidual(String),
    /// The residual edge target violates a necessary feasibility bound
    /// (§14 of the fixed-sE plan).
    InvalidEdgeTarget(String),
    /// The residual degree target violates a necessary feasibility bound
    /// (§10 of the fixed-(s,k) plan).
    InvalidDegreeTarget(String),
    /// The extras-first exact-(s,k) constructor exhausted its retry
    /// budgets (plan §35).  Retry exhaustion is **not** mathematical
    /// infeasibility (§21, §27) — a larger budget or different
    /// randomization may succeed.
    ExactSkExtrasFirstExhausted {
        /// Slot-aware extras transport attempts consumed.
        extras_attempts: usize,
        /// Extras attempts that stranded positive mass (slot/domain
        /// limits) plus extras tables discarded for completion failure.
        extras_failures: usize,
        /// Binary completion failures across all kept extras tables.
        completion_failures: usize,
        /// `Σ (s_out − k_out) = Σ (s_in − k_in)`.
        residual_total: OccNum,
    },
    /// Edge-count repair exhausted its restart budget without reaching the
    /// exact edge target (§13.3).
    EdgeRepairExhausted {
        /// The best occupied-pair count reached.
        best_edges: usize,
        /// The requested residual edge target.
        target_edges: usize,
        /// Absolute distance `|best − target|` of the best state.
        best_distance: usize,
        /// Number of reconstruction restarts performed.
        restarts: u32,
        /// Total attempted repair steps across all restarts.
        total_steps: u64,
    },
    /// Initialization via max flow or greedy construction failed.
    InitializationFailed(String),
    /// The repair step did not converge within the configured bounds (spec 21).
    RepairDidNotConverge {
        remaining_loops: usize,
        remaining_capacity_violations: usize,
        remaining_forbidden_occupations: usize,
        restart_count: u32,
        steps: u64,
    },
}

impl fmt::Display for FixedStrengthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidResidual(msg) => write!(f, "invalid residual problem: {msg}"),
            Self::InvalidEdgeTarget(msg) => write!(f, "invalid edge target: {msg}"),
            Self::InvalidDegreeTarget(msg) => write!(f, "invalid degree target: {msg}"),
            Self::ExactSkExtrasFirstExhausted {
                extras_attempts,
                extras_failures,
                completion_failures,
                residual_total,
            } => {
                write!(
                    f,
                    "extras-first exact-(s,k) initialization exhausted after {extras_attempts} extras attempts \
                     ({extras_failures} extras/completion failures, {completion_failures} completion failures) of \
                     residual total {residual_total} (target not proven infeasible)"
                )
            }
            Self::EdgeRepairExhausted {
                best_edges,
                target_edges,
                best_distance,
                restarts,
                total_steps,
            } => {
                write!(
                    f,
                    "edge repair exhausted after {total_steps} steps and {restarts} restarts: \
                     best E {best_edges}, target E {target_edges}, best distance {best_distance}"
                )
            }
            Self::InitializationFailed(msg) => write!(f, "initialization failed: {msg}"),
            Self::RepairDidNotConverge {
                remaining_loops,
                remaining_capacity_violations,
                remaining_forbidden_occupations,
                restart_count,
                steps,
            } => {
                write!(
                    f,
                    "repair did not converge after {steps} steps and {restart_count} restarts: \
                     {remaining_loops} loops, {remaining_capacity_violations} capacity violations, \
                     {remaining_forbidden_occupations} forbidden occupations remaining"
                )
            }
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
    /// Could not find a valid bracket after maximum expansions.
    BracketNotFound,
    /// The MCMC chain did not produce enough accepted transitions for a
    /// reliable cost estimate (spec §27).
    InsufficientMobility {
        proposals: u64,
        accepted: u64,
        min_accepted: u64,
    },
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
            Self::InsufficientMobility {
                proposals,
                accepted,
                min_accepted,
            } => {
                write!(
                    f,
                    "insufficient MCMC mobility: {accepted}/{proposals} accepted, need at least {min_accepted}"
                )
            }
            Self::BracketNotFound => {
                write!(
                    f,
                    "could not find a valid gamma bracket after maximum expansions"
                )
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
