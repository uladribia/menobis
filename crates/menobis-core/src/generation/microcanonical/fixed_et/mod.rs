//! Fixed-(E,T) microcanonical samplers for ME, B, and W families.
//!
//! Each family exports two public entry points:
//!
//! * `sample_{family}_fixed_et(node_count, self_loops, e, t, ...)` — fast
//!   index-mapped path (no N² materialisation).
//! * `sample_{family}_fixed_et_explicit(sources, targets, e, t, ...)` —
//!   explicit admissible-pair arrays (for masked/preprocessed problems).
//!
//! Occupation allocation uses the shared pair-Gibbs chain in
//! [`super::fixed_total`] (O(E) memory, no DP tables).  The legacy
//! rejection/DP backends were archived on
//! `archive/exact-fixed-total-pre-gibbs` (tag `exact-fixed-total-v1`).

pub mod core;
pub mod errors;
pub mod pairs;

use rand::rngs::StdRng;
use rand::SeedableRng;

use super::super::output::SampledNetwork;
use crate::OccNum;

use crate::generation::microcanonical::mcmc::McmcConfig;
use crate::model::family::OccupationFamily;
use core::sample_fixed_et_core;
use pairs::{linear_to_pair, total_admissible_pairs};

/// Default MCMC configuration for the fixed-(E,T) Gibbs occupation backend.
fn default_config(seed: u64) -> McmcConfig {
    McmcConfig {
        burn_in_sweeps: 20,
        sweeps_per_sample: 5,
        proposals_per_sweep: None,
        seed,
    }
}

// ---------------------------------------------------------------------------
// ME family
// ---------------------------------------------------------------------------

/// ME microcanonical sampler with fixed (E,T).
pub fn sample_me_fixed_et(
    node_count: usize,
    self_loops: bool,
    residual_edges: usize,
    residual_total: OccNum,
    seed: u64,
) -> Result<SampledNetwork, errors::FixedETError> {
    let l = total_admissible_pairs(node_count, self_loops);
    let mut rng = StdRng::seed_from_u64(seed);
    let config = default_config(seed);
    sample_fixed_et_core(
        OccupationFamily::ME,
        l,
        residual_edges,
        residual_total,
        &config,
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
    let config = default_config(seed);
    sample_fixed_et_core(
        OccupationFamily::ME,
        l,
        residual_edges,
        residual_total,
        &config,
        &mut rng,
        |idx| (admissible_sources[idx], admissible_targets[idx]),
    )
}

// ---------------------------------------------------------------------------
// B family
// ---------------------------------------------------------------------------

/// B microcanonical sampler with fixed (E,T) and M layers.
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
    let config = default_config(seed);
    sample_fixed_et_core(
        OccupationFamily::B {
            layers: layers as u32,
        },
        l,
        residual_edges,
        residual_total,
        &config,
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
    let config = default_config(seed);
    sample_fixed_et_core(
        OccupationFamily::B {
            layers: layers as u32,
        },
        l,
        residual_edges,
        residual_total,
        &config,
        &mut rng,
        |idx| (admissible_sources[idx], admissible_targets[idx]),
    )
}

// ---------------------------------------------------------------------------
// W family
// ---------------------------------------------------------------------------

/// W microcanonical sampler with fixed (E,T) and M layers.
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
    let config = default_config(seed);
    sample_fixed_et_core(
        OccupationFamily::W {
            layers: layers as u32,
        },
        l,
        residual_edges,
        residual_total,
        &config,
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
    let config = default_config(seed);
    sample_fixed_et_core(
        OccupationFamily::W {
            layers: layers as u32,
        },
        l,
        residual_edges,
        residual_total,
        &config,
        &mut rng,
        |idx| (admissible_sources[idx], admissible_targets[idx]),
    )
}
