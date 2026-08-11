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
//! [`super::conditional::fixed_total`].

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use crate::generation::microcanonical::conditional::fixed_total::sample_fixed_total;
use crate::generation::microcanonical::mcmc::McmcConfig;
use crate::generation::microcanonical::support::uniform_edges::sample_uniform_support;
use crate::generation::output::SampledNetwork;
use crate::model::family::OccupationFamily;
use crate::OccNum;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during fixed-(E,T) sampling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedETError {
    /// The residual problem is infeasible.
    InvalidResidual(String),
}

impl std::fmt::Display for FixedETError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidResidual(msg) => write!(f, "invalid residual problem: {msg}"),
        }
    }
}

impl std::error::Error for FixedETError {}

// ---------------------------------------------------------------------------
// Pair-index helpers
// ---------------------------------------------------------------------------

/// Total number of admissible pairs given node count and self-loop policy.
fn total_admissible_pairs(n: usize, self_loops: bool) -> usize {
    if self_loops {
        n.saturating_mul(n)
    } else {
        n.saturating_mul(n.saturating_sub(1))
    }
}

/// Map a linear index `idx` in `[0, L)` to a source/target node pair `(i, j)`.
fn linear_to_pair(idx: usize, n: usize, self_loops: bool) -> (usize, usize) {
    if self_loops {
        (idx / n, idx % n)
    } else {
        let i = idx / (n - 1);
        let j_raw = idx % (n - 1);
        let j = if j_raw >= i { j_raw + 1 } else { j_raw };
        (i, j)
    }
}

// ---------------------------------------------------------------------------
// Core orchestrator
// ---------------------------------------------------------------------------

/// Run the full fixed-(E,T) sampling pipeline.
///
/// `get_pair(idx)` maps a linear index to a `(source, target)` pair.
fn sample_fixed_et_core<G>(
    family: OccupationFamily,
    l: usize,
    e: usize,
    t: OccNum,
    config: &McmcConfig,
    rng: &mut StdRng,
    get_pair: G,
) -> Result<SampledNetwork, FixedETError>
where
    G: Fn(usize) -> (u64, u64),
{
    if e > l {
        return Err(FixedETError::InvalidResidual(format!(
            "residual_edges ({e}) exceeds admissible pair count ({l})"
        )));
    }
    if (e == 0) != (t == 0) {
        return Err(FixedETError::InvalidResidual(format!(
            "inconsistent (E,T) = ({e},{t}): both must be zero or both positive"
        )));
    }
    if e > 0 && t < e as OccNum {
        return Err(FixedETError::InvalidResidual(format!(
            "residual total {t} < residual edges {e} (each edge needs ≥1 event)"
        )));
    }
    if let Some(max) = family.max_occupation() {
        let capacity = max.saturating_mul(e as OccNum);
        if t > capacity {
            return Err(FixedETError::InvalidResidual(format!(
                "total {t} exceeds B capacity {max} × {e} = {capacity}"
            )));
        }
    }

    if e == 0 {
        return Ok(SampledNetwork::default());
    }
    if e == 1 {
        let idx = rng.random_range(0..l);
        let (i, j) = get_pair(idx);
        return Ok(SampledNetwork {
            sources: vec![i],
            targets: vec![j],
            occ_nums: vec![t],
        });
    }
    if t == e as OccNum {
        let indices = sample_uniform_support(l, e, rng);
        let mut sources = Vec::with_capacity(e);
        let mut targets = Vec::with_capacity(e);
        for &idx in &indices {
            let (i, j) = get_pair(idx);
            sources.push(i);
            targets.push(j);
        }
        return Ok(SampledNetwork {
            sources,
            targets,
            occ_nums: vec![1; e],
        });
    }

    let support = sample_uniform_support(l, e, rng);
    let occupations = sample_fixed_total(family, e, t, config)
        .map_err(|e| FixedETError::InvalidResidual(e.to_string()))?;

    let mut sources = Vec::with_capacity(e);
    let mut targets = Vec::with_capacity(e);
    let mut occ_nums = Vec::with_capacity(e);
    for (&idx, &occ) in support.iter().zip(occupations.iter()) {
        debug_assert!(occ > 0, "occupation allocator returned a zero");
        let (i, j) = get_pair(idx);
        sources.push(i);
        targets.push(j);
        occ_nums.push(occ);
    }
    let result = SampledNetwork {
        sources,
        targets,
        occ_nums,
    };

    debug_assert_eq!(result.sources.len(), e);
    debug_assert_eq!(result.occ_nums.iter().copied().sum::<OccNum>(), t);

    Ok(result)
}

// ---------------------------------------------------------------------------
// Default MCMC config
// ---------------------------------------------------------------------------

fn default_config(seed: u64) -> McmcConfig {
    McmcConfig {
        burn_in_sweeps: 20,
        sweeps_per_sample: 5,
        proposals_per_sweep: None,
        seed,
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// ME microcanonical sampler with fixed (E,T).
pub fn sample_me_fixed_et(
    node_count: usize,
    self_loops: bool,
    residual_edges: usize,
    residual_total: OccNum,
    seed: u64,
) -> Result<SampledNetwork, FixedETError> {
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
) -> Result<SampledNetwork, FixedETError> {
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

/// B microcanonical sampler with fixed (E,T) and M layers.
pub fn sample_b_fixed_et(
    node_count: usize,
    self_loops: bool,
    layers: OccNum,
    residual_edges: usize,
    residual_total: OccNum,
    seed: u64,
) -> Result<SampledNetwork, FixedETError> {
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
) -> Result<SampledNetwork, FixedETError> {
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

/// W microcanonical sampler with fixed (E,T) and M layers.
pub fn sample_w_fixed_et(
    node_count: usize,
    self_loops: bool,
    layers: OccNum,
    residual_edges: usize,
    residual_total: OccNum,
    seed: u64,
) -> Result<SampledNetwork, FixedETError> {
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
) -> Result<SampledNetwork, FixedETError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn config(seed: u64) -> McmcConfig {
        McmcConfig {
            burn_in_sweeps: 5,
            sweeps_per_sample: 2,
            proposals_per_sweep: None,
            seed,
        }
    }

    fn identity_pair(idx: usize) -> (u64, u64) {
        (idx as u64, idx as u64 + 1)
    }

    #[test]
    fn e_zero_t_zero() {
        let mut rng = StdRng::seed_from_u64(1);
        let net = sample_fixed_et_core(
            OccupationFamily::ME,
            10,
            0,
            0,
            &config(1),
            &mut rng,
            identity_pair,
        )
        .unwrap();
        assert!(net.sources.is_empty());
    }

    #[test]
    fn invalid_e_above_l() {
        let mut rng = StdRng::seed_from_u64(1);
        assert!(sample_fixed_et_core(
            OccupationFamily::ME,
            10,
            11,
            12,
            &config(1),
            &mut rng,
            identity_pair,
        )
        .is_err());
    }

    #[test]
    fn invalid_t_below_e() {
        let mut rng = StdRng::seed_from_u64(1);
        assert!(sample_fixed_et_core(
            OccupationFamily::ME,
            10,
            5,
            4,
            &config(1),
            &mut rng,
            identity_pair,
        )
        .is_err());
    }

    #[test]
    fn invalid_b_capacity() {
        let mut rng = StdRng::seed_from_u64(1);
        assert!(sample_fixed_et_core(
            OccupationFamily::B { layers: 2 },
            10,
            5,
            11,
            &config(1),
            &mut rng,
            identity_pair,
        )
        .is_err());
    }

    #[test]
    fn general_case_exact_constraints() {
        let mut rng = StdRng::seed_from_u64(42);
        let net = sample_fixed_et_core(
            OccupationFamily::ME,
            100,
            8,
            20,
            &config(42),
            &mut rng,
            identity_pair,
        )
        .unwrap();
        assert_eq!(net.sources.len(), 8);
        assert_eq!(net.occ_nums.iter().sum::<OccNum>(), 20);
        assert!(net.occ_nums.iter().all(|&t| t >= 1));
    }

    #[test]
    fn reproducible() {
        let a = sample_fixed_et_core(
            OccupationFamily::ME,
            100,
            8,
            20,
            &config(7),
            &mut StdRng::seed_from_u64(7),
            identity_pair,
        )
        .unwrap();
        let b = sample_fixed_et_core(
            OccupationFamily::ME,
            100,
            8,
            20,
            &config(7),
            &mut StdRng::seed_from_u64(7),
            identity_pair,
        )
        .unwrap();
        assert_eq!(a.sources, b.sources);
        assert_eq!(a.targets, b.targets);
        assert_eq!(a.occ_nums, b.occ_nums);
    }

    #[test]
    fn total_pairs_with_self_loops() {
        assert_eq!(total_admissible_pairs(4, true), 16);
        assert_eq!(total_admissible_pairs(4, false), 12);
    }

    #[test]
    fn linear_to_pair_roundtrip() {
        for &sl in &[true, false] {
            for n in [2, 3, 5] {
                let l = if sl { n * n } else { n * (n - 1) };
                let mut found = std::collections::HashSet::new();
                for idx in 0..l {
                    let (i, j) = linear_to_pair(idx, n, sl);
                    assert!(i < n && j < n);
                    if !sl {
                        assert_ne!(i, j);
                    }
                    assert!(found.insert((i, j)));
                }
                assert_eq!(found.len(), l);
            }
        }
    }
}
