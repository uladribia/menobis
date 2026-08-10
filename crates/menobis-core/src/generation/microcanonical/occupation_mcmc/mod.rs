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
pub mod feasibility;
pub mod initializer;
pub mod move_cycle;
pub mod problem;
pub mod rectangle;
pub mod repair;
pub mod state;
pub mod target;

pub use chain::{sample_fixed_strength, sample_fixed_strength_with_cost, StrengthBackend};
pub use cost::{residual_cost_target, state_cost};
pub use errors::FixedStrengthCostError;
pub use repair::{repair_all_violations, repair_capacity, RepairConfig};
