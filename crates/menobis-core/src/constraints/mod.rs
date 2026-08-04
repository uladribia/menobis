//! Shared constraint ontology and preprocessing.
//!
//! This module owns the concepts that all ensemble backends share:
//! the sparse pair mask, fixed-pair accounting, residualization, and
//! common validation. Fitting, generation, filtering, and analysis
//! consume these types; no backend may re-implement them.

pub mod fixed_pairs;
pub mod mask;
pub mod validation;

pub use fixed_pairs::{FixedContributions, FixedPairs, ResidualConstraints};
pub use mask::PairMask;
