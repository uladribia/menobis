//! Fixed-\((\mathbf k,T)\) microcanonical sampler for directed graphs.
//!
//! Provides a directed fixed-degree support MCMC and composes it with the
//! existing fixed-\((E,T)\) occupation allocators (ME, B, W).

pub mod core;
pub mod diagnostics;
pub mod errors;
pub mod feasibility;
pub mod initializer;
pub mod sampler;
pub mod state;
pub mod switch;