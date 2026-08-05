//! Fixed-(E,T) microcanonical samplers for ME, B, and W families.
//!
//! Each family exports two public entry points:
//!
//! * `sample_{family}_fixed_et(node_count, self_loops, e, t, ...)` — fast
//!   index-mapped path (no N² materialisation).
//! * `sample_{family}_fixed_et_explicit(sources, targets, e, t, ...)` —
//!   explicit admissible-pair arrays (for masked/preprocessed problems).
//!
//! The shared infrastructure lives in submodules; only the entry points
//! and re-exports are public.

pub mod b;
pub mod core;
pub mod errors;
pub mod me;
pub mod pairs;
pub mod support;
pub mod w;

use rand::rngs::StdRng;
use rand::SeedableRng;

use super::super::output::SampledNetwork;
use crate::OccNum;

use b::BFamily;
use core::sample_fixed_et_core;
use me::MeFamily;
use pairs::{linear_to_pair, total_admissible_pairs};
use w::WFamily;

// ---------------------------------------------------------------------------
// ME family
// ---------------------------------------------------------------------------

/// Exact ME microcanonical sampler with fixed (E,T).
pub fn sample_me_fixed_et(
    node_count: usize,
    self_loops: bool,
    residual_edges: usize,
    residual_total: OccNum,
    seed: u64,
) -> Result<SampledNetwork, errors::FixedETError> {
    let l = total_admissible_pairs(node_count, self_loops);
    let mut rng = StdRng::seed_from_u64(seed);
    sample_fixed_et_core(
        &MeFamily,
        l,
        residual_edges,
        residual_total,
        &mut rng,
        |idx| {
            let (i, j) = linear_to_pair(idx, node_count, self_loops);
            (i as u64, j as u64)
        },
    )
}

/// Same as [`sample_me_fixed_et`] but with explicit pair arrays.
pub fn sample_me_fixed_et_explicit(
    admissible_sources: &[u64],
    admissible_targets: &[u64],
    residual_edges: usize,
    residual_total: OccNum,
    seed: u64,
) -> Result<SampledNetwork, errors::FixedETError> {
    let l = admissible_sources.len();
    let mut rng = StdRng::seed_from_u64(seed);
    sample_fixed_et_core(
        &MeFamily,
        l,
        residual_edges,
        residual_total,
        &mut rng,
        |idx| (admissible_sources[idx], admissible_targets[idx]),
    )
}

// ---------------------------------------------------------------------------
// B family
// ---------------------------------------------------------------------------

/// Exact B microcanonical sampler with fixed (E,T) and M layers.
pub fn sample_b_fixed_et(
    node_count: usize,
    self_loops: bool,
    layers: OccNum,
    residual_edges: usize,
    residual_total: OccNum,
    seed: u64,
) -> Result<SampledNetwork, errors::FixedETError> {
    let l = total_admissible_pairs(node_count, self_loops);
    let mut rng = StdRng::seed_from_u64(seed);
    sample_fixed_et_core(
        &BFamily { layers },
        l,
        residual_edges,
        residual_total,
        &mut rng,
        |idx| {
            let (i, j) = linear_to_pair(idx, node_count, self_loops);
            (i as u64, j as u64)
        },
    )
}

/// Same as [`sample_b_fixed_et`] but with explicit pair arrays.
pub fn sample_b_fixed_et_explicit(
    admissible_sources: &[u64],
    admissible_targets: &[u64],
    layers: OccNum,
    residual_edges: usize,
    residual_total: OccNum,
    seed: u64,
) -> Result<SampledNetwork, errors::FixedETError> {
    let l = admissible_sources.len();
    let mut rng = StdRng::seed_from_u64(seed);
    sample_fixed_et_core(
        &BFamily { layers },
        l,
        residual_edges,
        residual_total,
        &mut rng,
        |idx| (admissible_sources[idx], admissible_targets[idx]),
    )
}

// ---------------------------------------------------------------------------
// W family
// ---------------------------------------------------------------------------

/// Exact W microcanonical sampler with fixed (E,T) and M layers.
pub fn sample_w_fixed_et(
    node_count: usize,
    self_loops: bool,
    layers: OccNum,
    residual_edges: usize,
    residual_total: OccNum,
    seed: u64,
) -> Result<SampledNetwork, errors::FixedETError> {
    let l = total_admissible_pairs(node_count, self_loops);
    let mut rng = StdRng::seed_from_u64(seed);
    sample_fixed_et_core(
        &WFamily { layers },
        l,
        residual_edges,
        residual_total,
        &mut rng,
        |idx| {
            let (i, j) = linear_to_pair(idx, node_count, self_loops);
            (i as u64, j as u64)
        },
    )
}

/// Same as [`sample_w_fixed_et`] but with explicit pair arrays.
pub fn sample_w_fixed_et_explicit(
    admissible_sources: &[u64],
    admissible_targets: &[u64],
    layers: OccNum,
    residual_edges: usize,
    residual_total: OccNum,
    seed: u64,
) -> Result<SampledNetwork, errors::FixedETError> {
    let l = admissible_sources.len();
    let mut rng = StdRng::seed_from_u64(seed);
    sample_fixed_et_core(
        &WFamily { layers },
        l,
        residual_edges,
        residual_total,
        &mut rng,
        |idx| (admissible_sources[idx], admissible_targets[idx]),
    )
}
