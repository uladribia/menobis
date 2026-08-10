//! Fixed-strength microcanonical samplers for ME, B, and W.
//!
//! This module provides exact and MCMC-based sampling from the
//! microcanonical ensemble with fixed out- and in-strength sequences.
//!
//! # Backend
//!
//! - **Cycle MCMC**: generic 4-cycle Metropolis chain for all families
//!   and restricted domains.

pub mod chain;
pub mod compressed;
pub mod cost;
pub mod cost_fit;
pub mod domain;
pub mod errors;
pub mod initializer;
pub mod move_cycle;
pub mod problem;
pub mod rectangle;
pub mod repair;
pub mod state;
pub mod target;

pub use chain::{
    sample_fixed_strength, sample_fixed_strength_bench, sample_fixed_strength_with_cost,
    sample_fixed_strength_with_cost_bench, FixedStrengthBenchMetrics, StrengthBackend,
};
pub use compressed::FlowTable;
pub use cost::{residual_cost_target, state_cost};
pub use cost_fit::{
    effective_sample_size, fit_gamma, FixedStrengthCostFitConfig, FixedStrengthCostFitResult,
};
pub use errors::FixedStrengthCostError;
pub use repair::{repair_all_violations, repair_capacity, RepairConfig};
