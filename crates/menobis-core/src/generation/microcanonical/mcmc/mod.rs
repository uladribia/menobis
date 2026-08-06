//! Shared MCMC infrastructure reused by multiple microcanonical chains.
//!
//! This module provides only **data types and small helper functions** that
//! are genuinely shared across constraint families.  It deliberately avoids
//! a trait-based generic chain framework — each concrete chain
//! (`FixedDegreeChain`, `FixedStrengthChain`, …) remains in its own module
//! and embeds these shared types.
//!
//! # Contents
//!
//! - [`McmcConfig`]: burn-in, thinning, and seed configuration.
//! - [`McmcCounters`]: proposal / acceptance / hold / reject counters.
//! - [`McmcOutcome`]: per-step outcome enum.

mod config;
mod counters;
mod outcome;

pub use config::McmcConfig;
pub use counters::McmcCounters;
pub use outcome::McmcOutcome;
