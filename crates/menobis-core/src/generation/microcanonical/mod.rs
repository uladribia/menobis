//! Microcanonical generation: exact-constraint direct samplers and MCMC.
//!
//! - `fixed_strength`: fixed-strength samplers for ME, B, and W.
//! - `fixed_et`: fixed-(E,T) samplers for ME, B, and W.
//! - `fixed_kt`: fixed-degree-event samplers for ME, B, and W.
//! - `fixed_total`: shared pair-Gibbs fixed-total occupation sampler.
//! - `mcmc`: shared MCMC infrastructure (config, counters, outcome).
//! - `support`: shared support-sampling utilities.

pub mod fixed_et;
pub mod fixed_kt;
pub mod fixed_strength;
pub mod fixed_total;
pub mod mcmc;
pub mod support;

pub use fixed_et::{
    sample_b_fixed_et, sample_b_fixed_et_explicit, sample_me_fixed_et, sample_me_fixed_et_explicit,
    sample_w_fixed_et, sample_w_fixed_et_explicit,
};
pub use fixed_kt::core::sample_fixed_kt_core;
pub use fixed_kt::core::FixedKTConfig;
pub use fixed_kt::diagnostics::FixedDegreeDiagnostics;
pub use fixed_kt::sampler::FixedDegreeMcmcConfig;
pub use fixed_strength::sample_fixed_strength;
pub use fixed_strength::sample_strength_stub_matching;
pub use fixed_total::{sample_fixed_total, FixedTotalChain, FixedTotalState};
