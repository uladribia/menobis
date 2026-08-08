//! Exact ME fixed-strength sampler via stub matching.
//!
//! # Algorithm
//!
//! Materialise the incoming stubs (each target node `j` appears `s_in[j]`
//! times), shuffle them uniformly, then iterate outgoing strengths
//! source-by-source while consuming the shuffled targets in order.
//! Pair occupations are counted in a sparse hash map.
//!
//! This produces a uniform sample from the ME microcanonical ensemble
//! with exact out- and in-strengths.  Memory is \(O(T)\) node identifiers
//! for the single shuffled incoming-stub vector.
//!
//! # Limitations
//!
//! - Only valid for ME (Poisson occupation statistics at fixed strength).
//! - Always allows self-loops (the stub-matching procedure cannot enforce
//!   looplessness without biasing the distribution).
//! - Only valid for a complete pair domain (no masks, no fixed cells).
//! - Total stub count must fit in `usize` and be below the configurable
//!   work limit [`MAX_EXPLICIT_STUBS`].

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use super::errors::FixedStrengthError;
use crate::generation::output::SampledNetwork;

/// Maximum number of explicit stubs for the direct stub-matching backend.
///
/// Above this limit, the function returns [`TooLargeForDirect`] and the
/// caller should fall back to the MCMC backend.
///
/// [`TooLargeForDirect`]: FixedStrengthError::TooLargeForDirect
pub(crate) const MAX_EXPLICIT_STUBS: u64 = 10_000_000;

/// Exact ME microcanonical fixed-strength sample via stub matching.
///
/// Returns a [`SampledNetwork`] with exact out- and in-strengths.
///
/// # Errors
///
/// - [`FixedStrengthError::InvalidResidual`] if strengths are unbalanced.
/// - [`FixedStrengthError::ArithmeticOverflow`] if total stubs don't fit in
///   `usize`.
/// - [`FixedStrengthError::TooLargeForDirect`] if the total stub count
///   exceeds [`MAX_EXPLICIT_STUBS`].
///
/// # Panics
///
/// Panics if `strength_out` and `strength_in` have different lengths
/// (this is a precondition violation, reported as an assertion).
pub fn sample_strength_stub_matching(
    strength_out: &[u64],
    strength_in: &[u64],
    seed: u64,
) -> Result<SampledNetwork, FixedStrengthError> {
    let n = strength_out.len();
    assert_eq!(
        n,
        strength_in.len(),
        "strength_out and strength_in must have the same length"
    );

    let total_out: u64 = strength_out.iter().sum();
    let total_in: u64 = strength_in.iter().sum();

    if total_out != total_in {
        return Err(FixedStrengthError::InvalidResidual(format!(
            "total out-strength ({total_out}) != total in-strength ({total_in})"
        )));
    }

    let t: usize = total_out.try_into().map_err(|_| {
        FixedStrengthError::ArithmeticOverflow(format!(
            "total stub count {total_out} does not fit in usize"
        ))
    })?;

    if total_out > MAX_EXPLICIT_STUBS {
        return Err(FixedStrengthError::TooLargeForDirect(total_out));
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

    // Stream outgoing strengths and consume shuffled targets.
    let mut weight_map = std::collections::HashMap::new();
    let mut in_idx = 0usize;
    for (src, &s) in strength_out.iter().enumerate() {
        let src_u64 = src as u64;
        for _ in 0..s {
            let tgt = in_stubs[in_idx];
            in_idx += 1;
            *weight_map.entry((src_u64, tgt)).or_insert(0u64) += 1;
        }
    }
    debug_assert_eq!(in_idx, t, "all incoming stubs must be consumed");

    // Build sorted output.
    let mut result = SampledNetwork::default();
    let mut pairs: Vec<_> = weight_map.into_iter().collect();
    pairs.sort_unstable();
    for ((src, tgt), w) in pairs {
        result.sources.push(src);
        result.targets.push(tgt);
        result.occ_nums.push(w);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_exact_strengths() {
        let s_out = vec![10, 20, 30];
        let s_in = vec![15, 25, 20];
        let result = sample_strength_stub_matching(&s_out, &s_in, 42).unwrap();
        let total: u64 = result.occ_nums.iter().sum();
        assert_eq!(total, 60);
        let mut actual_out = vec![0u64; 3];
        let mut actual_in = vec![0u64; 3];
        for ((&src, &tgt), &w) in result
            .sources
            .iter()
            .zip(result.targets.iter())
            .zip(result.occ_nums.iter())
        {
            actual_out[src as usize] += w;
            actual_in[tgt as usize] += w;
        }
        assert_eq!(actual_out, s_out);
        assert_eq!(actual_in, s_in);
    }

    #[test]
    fn deterministic_reproducibility() {
        let s_out = vec![5, 7, 3];
        let s_in = vec![4, 6, 5];
        let a = sample_strength_stub_matching(&s_out, &s_in, 42).unwrap();
        let b = sample_strength_stub_matching(&s_out, &s_in, 42).unwrap();
        assert_eq!(a.sources, b.sources);
        assert_eq!(a.targets, b.targets);
        assert_eq!(a.occ_nums, b.occ_nums);
    }

    #[test]
    fn rejects_unbalanced_strengths() {
        let s_out = vec![10, 10];
        let s_in = vec![5, 5];
        assert!(sample_strength_stub_matching(&s_out, &s_in, 0).is_err());
    }

    #[test]
    fn zero_strengths() {
        let s_out = vec![0u64; 5];
        let s_in = vec![0u64; 5];
        let result = sample_strength_stub_matching(&s_out, &s_in, 0).unwrap();
        assert!(result.sources.is_empty());
        assert!(result.targets.is_empty());
        assert!(result.occ_nums.is_empty());
    }

    #[test]
    fn too_large_returns_error() {
        // Construct a total exceeding MAX_EXPLICIT_STUBS.
        let limit = MAX_EXPLICIT_STUBS + 1;
        let s_out = vec![limit, 0];
        let s_in = vec![limit, 0];
        let result = sample_strength_stub_matching(&s_out, &s_in, 0);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(FixedStrengthError::TooLargeForDirect(_))
        ));
    }

    #[test]
    fn small_case_succeeds() {
        // A small case well below the limit.
        let s_out = vec![5u64, 5];
        let s_in = vec![5u64, 5];
        let result = sample_strength_stub_matching(&s_out, &s_in, 42);
        assert!(result.is_ok());
    }
}
