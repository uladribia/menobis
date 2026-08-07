//! Fixed-total occupation sampling: scalable pair-Gibbs chain.
//!
//! Samples positive occupations `t_e ≥ 1` with `Σ t_e = T` over `E`
//! cells under the family-specific degeneracy law, using a two-cell
//! Gibbs kernel with exact conditionals.
//!
//! - Memory is `O(E)` — no `O(E·T)` DP tables, no exponential
//!   whole-allocation rejection.
//! - This is the shared occupation backend for fixed-\((E,T)\) and
//!   fixed-\((\mathbf k,T)\).
//!
//! # Pipeline
//!
//! ```text
//! validate (T ≥ E, B capacity)
//!     ↓
//! balanced feasible initializer  (O(E))
//!     ↓
//! pair-Gibbs burn-in
//!     ↓
//! thinned sample
//! ```

pub mod chain;
pub mod errors;
pub mod initializer;
pub mod pair_conditional;
pub mod state;

pub use chain::{sample_fixed_total, FixedTotalChain};
pub use errors::FixedTotalError;
pub use pair_conditional::sample_split;
pub use state::FixedTotalState;
