//! Microcanonical generation: exact-constraint direct samplers and MCMC.
//!
//! - `conditional/fixed_total`: shared pair-Gibbs fixed-total occupation sampler.
//! - `binary`: fixed-degree binary-support MCMC.
//! - `occupation_mcmc`: fixed-strength occupation MCMC.
//! - `route`: unified router (constraint structure → sampling plan → backend).
//! - `mcmc`: shared MCMC infrastructure (config, counters, outcome).
//! - `support`: shared support-sampling utilities.

pub mod binary;
pub mod conditional;
pub mod mcmc;
pub mod occupation_mcmc;
pub mod route;
pub mod support;

// Re-exports from conditional/fixed_total
pub use conditional::fixed_total::{sample_fixed_total, FixedTotalChain, FixedTotalState};

// Re-exports from binary
pub use binary::core::sample_fixed_kt_core;
pub use binary::core::FixedKTConfig;
pub use binary::diagnostics::FixedDegreeDiagnostics;
pub use binary::sampler::FixedDegreeMcmcConfig;

// Re-exports from conditional/fixed_et
pub use conditional::fixed_et::{
    sample_b_fixed_et, sample_b_fixed_et_explicit, sample_me_fixed_et, sample_me_fixed_et_explicit,
    sample_w_fixed_et, sample_w_fixed_et_explicit,
};

// Re-exports from occupation_mcmc
pub use occupation_mcmc::sample_fixed_strength;
pub use occupation_mcmc::sample_strength_stub_matching;
