//! Fixed-strength microcanonical samplers for ME, B, and W.
//!
//! This module provides exact and MCMC-based sampling from the
//! microcanonical ensemble with fixed out- and in-strength sequences.
//!
//! # Backends
//!
//! - **ME direct** (`me_direct`): exact stub-matching for the simple case
//!   (ME family, self-loops allowed, complete pair domain, no fixed cells,
//!   total stubs ≤ 10M).
//! - **Cycle MCMC** (future steps): generic 4-cycle Metropolis chain for
//!   all families and restricted domains.

pub mod chain;
pub mod compressed;
pub mod cost;
pub mod cost_fit;
pub mod domain;
pub mod errors;
pub mod feasibility;
pub mod initializer;
pub mod me_direct;
pub mod move_cycle;
pub mod problem;
pub mod rectangle;
pub mod repair;
pub mod state;
pub mod target;

pub use chain::{sample_fixed_strength, sample_fixed_strength_with_cost, StrengthBackend};
pub use cost::{residual_cost_target, state_cost};
pub use errors::FixedStrengthCostError;
pub use me_direct::sample_strength_stub_matching;
pub use repair::{repair_all_violations, repair_capacity, RepairConfig};
