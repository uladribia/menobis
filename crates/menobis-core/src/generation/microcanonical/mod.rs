//! Microcanonical generation: exact-constraint direct samplers.
//!
//! - `sample_strength_stub_matching`: exact ME fixed-strength via stub
//!   matching (uniform over compatible labelled matchings).
//! - `fixed_et`: fixed-(E,T) samplers for ME, B, and W families.

pub mod fixed_et;
pub mod fixed_kt;
pub mod support;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use super::output::SampledNetwork;

pub use fixed_et::{
    sample_b_fixed_et, sample_b_fixed_et_explicit, sample_me_fixed_et, sample_me_fixed_et_explicit,
    sample_w_fixed_et, sample_w_fixed_et_explicit,
};
pub use fixed_kt::core::sample_fixed_kt_core;
pub use fixed_kt::core::FixedKTConfig;
pub use fixed_kt::diagnostics::FixedDegreeDiagnostics;
pub use fixed_kt::sampler::FixedDegreeMcmcConfig;

pub fn sample_strength_stub_matching(
    strength_out: &[u64],
    strength_in: &[u64],
    seed: u64,
) -> SampledNetwork {
    let n = strength_out.len();
    let total_out: u64 = strength_out.iter().sum();
    let total_in: u64 = strength_in.iter().sum();
    assert_eq!(
        total_out, total_in,
        "stub_matching requires balanced strengths"
    );
    let t = total_out as usize;

    // Build outgoing stubs: node i appears s_out[i] times.
    let mut out_stubs: Vec<u64> = Vec::with_capacity(t);
    for (i, &s) in strength_out.iter().enumerate() {
        for _ in 0..s {
            out_stubs.push(i as u64);
        }
    }

    // Build incoming stubs: node j appears s_in[j] times.
    let mut in_stubs: Vec<u64> = Vec::with_capacity(t);
    for (j, &s) in strength_in.iter().enumerate() {
        for _ in 0..s {
            in_stubs.push(j as u64);
        }
    }

    // Shuffle incoming stubs.
    let mut rng = StdRng::seed_from_u64(seed);
    in_stubs.shuffle(&mut rng);

    // Count pair occupations from stub matchings.
    let mut weight_map = std::collections::HashMap::new();
    for (&src, &tgt) in out_stubs.iter().zip(in_stubs.iter()) {
        *weight_map.entry((src, tgt)).or_insert(0u64) += 1;
    }

    let mut result = SampledNetwork::default();
    let mut pairs: Vec<_> = weight_map.into_iter().collect();
    pairs.sort_unstable();
    for ((src, tgt), w) in pairs {
        result.sources.push(src);
        result.targets.push(tgt);
        result.occ_nums.push(w);
    }
    let _ = n; // used only in assert context
    result
}
