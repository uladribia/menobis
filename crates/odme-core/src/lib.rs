//! Core Rust functionality for ODME.
//!
//! This crate contains the performance-sensitive kernels for
//! Origin-Destination Multi-Edge network analysis, fitting, and generation.

pub mod clustering;
pub mod cost;
pub mod filter;
pub mod fitting;
pub mod generation;
pub mod graph;
pub mod stats;

/// Current ODME core version, synchronized with the Python package during releases.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::VERSION;

    #[test]
    fn exposes_version() {
        assert!(!VERSION.is_empty());
    }
}
