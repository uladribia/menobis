//! Legacy exact fixed-(E,T) occupation backends — reference oracle.
//!
//! These are the exact rejection/DP samplers that were the production
//! implementation before the scalable pair-Gibbs chain
//! ([`menobis_core::generation::microcanonical::conditional::fixed_total`]).
//!
//! They are **scientifically valid** and remain valuable as a reference
//! oracle at medium `E`, `T` where state enumeration is infeasible:
//!
//! - ME: multinomial rejection + Stirling-numbers DP fallback;
//! - B:  binary-cell rejection + bounded DP fallback;
//! - W:  weak-composition rejection + unbounded DP fallback.
//!
//! This module is **test/reference infrastructure only**.  It must never
//! become a production dependency of `menobis-core`.

pub mod b;
pub mod core;
pub mod errors;
pub mod me;
pub mod w;

pub use b::BFamily;
pub use core::{
    sample_fixed_et_core, sample_positive_occupations, FixedETOccupancy, MAX_DP_CELLS,
    MAX_REJECTION_ATTEMPTS, MIN_ESTIMATED_P_ACC, REJECTION_THRESHOLD,
};
pub use errors::FixedETError;
pub use me::MeFamily;
pub use w::WFamily;
