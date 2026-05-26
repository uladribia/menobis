//! Core Rust functionality for MENoBiS.
//!
//! This crate contains the performance-sensitive kernels for
//! Max Entropy NOn Binary Suite network analysis, fitting, and generation.

pub mod clustering;
pub mod distribution;
pub mod filter;
pub mod fitting;
pub mod generation;
pub mod graph;
pub mod pairs;
pub mod stats;

/// Current MENoBiS core version, synchronized with the Python package during releases.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::VERSION;

    #[test]
    fn exposes_version() {
        assert!(!VERSION.is_empty());
    }
}
