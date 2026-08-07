//! Orchestrator for fixed-(E,T) microcanonical generation.
//!
//! Pipeline: validation → uniform support sampling → fixed-total Gibbs
//! occupation allocation → output assembly.
//!
//! The legacy rejection/DP backends (archived on
//! `archive/exact-fixed-total-pre-gibbs`, tag `exact-fixed-total-v1`)
//! have been replaced by the shared pair-Gibbs chain in
//! [`crate::generation::microcanonical::fixed_total`].

use rand::rngs::StdRng;
use rand::Rng;

use super::super::super::output::SampledNetwork;
use super::errors::FixedETError;
use crate::generation::microcanonical::fixed_total::sample_fixed_total;
use crate::generation::microcanonical::mcmc::McmcConfig;
use crate::generation::microcanonical::support::uniform_edges::sample_uniform_support;
use crate::model::family::OccupationFamily;
use crate::OccNum;

/// Run the full fixed-(E,T) sampling pipeline.
///
/// `get_pair(idx)` maps a linear index to a `(source, target)` pair.
///
/// The Gibbs occupation backend consumes `config` (burn-in, thinning,
/// seed); `rng` drives support sampling only.
pub fn sample_fixed_et_core<G>(
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
    // ---- basic validation (shared across families) ----
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

    // ---- special cases ----
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

    // ---- general case ----
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

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
            11, // > 2*5
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
}
