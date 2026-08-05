//! Core orchestrator for fixed-\((\mathbf k,T)\) microcanonical sampling.
//!
//! The pipeline:
//! 1. Validate residual out-degree / in-degree sequences.
//! 2. Sample a directed support via MCMC.
//! 3. Compute \(E = \sum k^{\mathrm{out}} = \sum k^{\mathrm{in}}\).
//! 4. Allocate positive occupations via the family-specific allocator.
//! 5. Pair occupations with support → build `SampledNetwork`.

use rand::rngs::StdRng;
use rand::SeedableRng;

use super::super::super::output::SampledNetwork;
use super::errors::FixedKTError;
use super::feasibility::DirectedDegreeSequence;
use super::sampler::{sample_fixed_degree_support, FixedDegreeMcmcConfig};
use crate::generation::microcanonical::fixed_et::core::{
    sample_positive_occupations, FixedETOccupancy,
};
use crate::OccNum;

/// Configuration for fixed-\((\mathbf k,T)\) sampling.
#[derive(Clone, Debug)]
pub struct FixedKTConfig {
    pub mcmc: FixedDegreeMcmcConfig,
    pub self_loops: bool,
}

impl FixedKTConfig {
    pub fn new(mcmc: FixedDegreeMcmcConfig, self_loops: bool) -> Self {
        Self { mcmc, self_loops }
    }
}

impl Default for FixedKTConfig {
    fn default() -> Self {
        Self {
            mcmc: FixedDegreeMcmcConfig::default(),
            self_loops: false,
        }
    }
}

/// Run the full fixed-\((\mathbf k,T)\) sampling pipeline.
///
/// # Arguments
///
/// * `family` — the family-specific occupancy implementation (ME, B, or W).
/// * `out_degrees` — target out-degree sequence (length N).
/// * `in_degrees` — target in-degree sequence (length N).
/// * `total` — target total occupation \(T\).
/// * `config` — sampling configuration (MCMC params, self-loop policy).
///
/// # Returns
///
/// A `SampledNetwork` with exactly `E = Σout = Σin` edges and total
/// occupation `T`.
pub fn sample_fixed_kt_core<F: FixedETOccupancy>(
    family: &F,
    out_degrees: &[u32],
    in_degrees: &[u32],
    total: OccNum,
    config: &FixedKTConfig,
) -> Result<SampledNetwork, FixedKTError> {
    // ---- Step 1: Validate degree sequences ----
    let seq = DirectedDegreeSequence::new(
        out_degrees.to_vec(),
        in_degrees.to_vec(),
        config.self_loops,
    )?;

    let e = seq.edge_count;

    // ---- Special cases ----
    if e == 0 {
        return if total == 0 {
            Ok(SampledNetwork::default())
        } else {
            Err(FixedKTError::InvalidResidual(format!(
                "zero edges but total occupation {total} > 0"
            )))
        };
    }

    if total < e as OccNum {
        return Err(FixedKTError::InvalidResidual(format!(
            "total occupation {total} < edge count {e} (each edge needs ≥1 event)"
        )));
    }

    // Family-specific residual validation
    family
        .validate_residual(e, total)
        .map_err(|e| FixedKTError::OccupationError(e.to_string()))?;

    // ---- Step 2: Sample directed support ----
    let mut rng = StdRng::seed_from_u64(config.mcmc.seed);
    let (state, _diag) = sample_fixed_degree_support(
        &seq.out_degrees,
        &seq.in_degrees,
        config.self_loops,
        &config.mcmc,
    )?;

    // ---- Steps 3-4: Allocate occupations ----
    let occupations = if e == 1 {
        vec![total]
    } else if total == e as OccNum {
        vec![1; e]
    } else {
        sample_positive_occupations(family, total, e, &mut rng)
            .map_err(|e| FixedKTError::OccupationError(e.to_string()))?
    };

    // ---- Step 5: Build output ----
    let mut sources = Vec::with_capacity(e);
    let mut targets = Vec::with_capacity(e);
    let mut occ_nums = Vec::with_capacity(e);

    for (&(src, tgt), &occ) in state.edges.iter().zip(occupations.iter()) {
        sources.push(src);
        targets.push(tgt);
        occ_nums.push(occ);
    }

    let result = SampledNetwork {
        sources,
        targets,
        occ_nums,
    };

    debug_assert_eq!(result.sources.len(), e);
    debug_assert_eq!(
        result.occ_nums.iter().copied().sum::<OccNum>(),
        total
    );

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::microcanonical::fixed_et::me::MeFamily;
    use crate::generation::microcanonical::fixed_et::b::BFamily;
    use crate::generation::microcanonical::fixed_et::w::WFamily;

    #[test]
    fn me_directed_cycle() {
        // N=4, each node out=1, in=1, T=8 (2 events per edge)
        let out = vec![1u32, 1, 1, 1];
        let inp = vec![1u32, 1, 1, 1];
        let config = FixedKTConfig {
            mcmc: FixedDegreeMcmcConfig {
                burn_in_sweeps: 10,
                sweeps_per_sample: 5,
                seed: 42,
            },
            self_loops: false,
        };
        let result = sample_fixed_kt_core(&MeFamily, &out, &inp, 8, &config).unwrap();
        assert_eq!(result.sources.len(), 4);
        assert_eq!(result.occ_nums.iter().sum::<OccNum>(), 8);
        // Verify support out-degree
        let mut support_out = vec![0u32; 4];
        for &s in &result.sources {
            support_out[s as usize] += 1;
        }
        assert_eq!(support_out, out);
    }

    #[test]
    fn me_out_star() {
        let n = 5;
        let mut out = vec![0u32; n];
        out[0] = (n - 1) as u32;
        let mut inp = vec![0u32; n];
        for i in 1..n {
            inp[i] = 1;
        }
        let config = FixedKTConfig {
            mcmc: FixedDegreeMcmcConfig {
                burn_in_sweeps: 10,
                sweeps_per_sample: 5,
                seed: 42,
            },
            self_loops: false,
        };
        let t = (n as OccNum - 1) * 3;
        let result = sample_fixed_kt_core(&MeFamily, &out, &inp, t, &config).unwrap();
        assert_eq!(result.sources.len(), (n - 1) as usize);
        assert_eq!(result.occ_nums.iter().sum::<OccNum>(), t);
        let mut support_out = vec![0u32; n];
        for &s in &result.sources {
            support_out[s as usize] += 1;
        }
        assert_eq!(support_out, out);
    }

    #[test]
    fn b_directed_cycle() {
        let out = vec![1u32, 1, 1, 1];
        let inp = vec![1u32, 1, 1, 1];
        let config = FixedKTConfig {
            mcmc: FixedDegreeMcmcConfig {
                burn_in_sweeps: 10,
                sweeps_per_sample: 5,
                seed: 42,
            },
            self_loops: false,
        };
        let result = sample_fixed_kt_core(
            &BFamily { layers: 4 },
            &out,
            &inp,
            6,
            &config,
        )
        .unwrap();
        assert_eq!(result.sources.len(), 4);
        assert_eq!(result.occ_nums.iter().sum::<OccNum>(), 6);
    }

    #[test]
    fn w_directed_cycle() {
        let out = vec![1u32, 1, 1, 1];
        let inp = vec![1u32, 1, 1, 1];
        let config = FixedKTConfig {
            mcmc: FixedDegreeMcmcConfig {
                burn_in_sweeps: 10,
                sweeps_per_sample: 5,
                seed: 42,
            },
            self_loops: false,
        };
        let result = sample_fixed_kt_core(
            &WFamily { layers: 2 },
            &out,
            &inp,
            10,
            &config,
        )
        .unwrap();
        assert_eq!(result.sources.len(), 4);
        assert_eq!(result.occ_nums.iter().sum::<OccNum>(), 10);
    }

    #[test]
    fn infeasible_t_below_e() {
        let out = vec![2u32, 2];
        let inp = vec![2u32, 2];
        let config = FixedKTConfig::default();
        let result = sample_fixed_kt_core(&MeFamily, &out, &inp, 3, &config);
        assert!(result.is_err());
    }

    #[test]
    fn reproducible() {
        let out = vec![1u32, 1, 1, 1];
        let inp = vec![1u32, 1, 1, 1];
        let config = FixedKTConfig {
            mcmc: FixedDegreeMcmcConfig {
                burn_in_sweeps: 5,
                sweeps_per_sample: 2,
                seed: 42,
            },
            self_loops: false,
        };
        let a = sample_fixed_kt_core(&MeFamily, &out, &inp, 8, &config).unwrap();
        let b = sample_fixed_kt_core(&MeFamily, &out, &inp, 8, &config).unwrap();
        assert_eq!(a.sources, b.sources);
        assert_eq!(a.targets, b.targets);
        assert_eq!(a.occ_nums, b.occ_nums);
    }
}