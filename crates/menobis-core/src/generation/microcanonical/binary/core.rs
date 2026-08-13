//! Core orchestrator for fixed-\((\mathbf k,T)\) microcanonical sampling.
//!
//! The pipeline:
//! 1. Validate residual out-degree / in-degree sequences.
//! 2. Sample a directed support via MCMC.
//! 3. Compute \(E = \sum k^{\mathrm{out}} = \sum k^{\mathrm{in}}\).
//! 4. Allocate positive occupations via the shared fixed-total Gibbs sampler.
//! 5. Pair occupations with support → build `SampledNetwork`.

use super::super::super::output::SampledNetwork;
use super::errors::FixedKTError;
use super::feasibility::DirectedDegreeSequence;
use super::sampler::{sample_fixed_degree_support, FixedDegreeMcmcConfig};
use crate::generation::microcanonical::conditional::fixed_total::sample_fixed_total;
use crate::generation::microcanonical::mcmc::McmcConfig;
use crate::model::family::OccupationFamily;
use crate::OccNum;

/// Configuration for fixed-\((\mathbf k,T)\) sampling.
#[derive(Clone, Debug, Default)]
pub struct FixedKTConfig {
    pub mcmc: FixedDegreeMcmcConfig,
    pub self_loops: bool,
    /// Optional admissible-pair list for masked support (fixed-pair residualization).
    /// When `Some`, the support sampler restricts to these ordered pairs.
    pub admissible_pairs: Option<Vec<(u64, u64)>>,
}

impl FixedKTConfig {
    pub fn new(mcmc: FixedDegreeMcmcConfig, self_loops: bool) -> Self {
        Self {
            mcmc,
            self_loops,
            admissible_pairs: None,
        }
    }
}

/// Run the full fixed-\((\mathbf k,T)\) sampling pipeline.
///
/// # Arguments
///
/// * `family` — the occupation family (ME, B, or W).
/// * `out_degrees` — target out-degree sequence (length N).
/// * `in_degrees` — target in-degree sequence (length N).
/// * `total` — target total occupation \(T\).
/// * `config` — sampling configuration (MCMC params, self-loop policy).
///
/// # Returns
///
/// A `SampledNetwork` with exactly `E = Σout = Σin` edges and total
/// occupation `T`.
pub fn sample_fixed_kt_core(
    family: OccupationFamily,
    out_degrees: &[u32],
    in_degrees: &[u32],
    total: OccNum,
    config: &FixedKTConfig,
) -> Result<SampledNetwork, FixedKTError> {
    // ---- Step 1: Validate degree sequences ----
    let seq =
        DirectedDegreeSequence::new(out_degrees.to_vec(), in_degrees.to_vec(), config.self_loops)?;

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

    // Family-specific residual validation (B capacity)
    if let Some(max) = family.max_occupation() {
        let capacity = max.saturating_mul(e as OccNum);
        if total > capacity {
            return Err(FixedKTError::InvalidResidual(format!(
                "total occupation {total} exceeds B capacity {max} × {e} = {capacity}"
            )));
        }
    }

    // Gibbs occupation backend configuration (derived from the support MCMC config)
    let gibbs_config = McmcConfig {
        burn_in_sweeps: config.mcmc.burn_in_sweeps,
        sweeps_per_sample: config.mcmc.sweeps_per_sample,
        proposals_per_sweep: config.mcmc.proposals_per_sweep,
        seed: config.mcmc.seed,
    };

    // ---- Step 2: Sample directed support ----
    let (state, _diag) = sample_fixed_degree_support(
        &seq.out_degrees,
        &seq.in_degrees,
        config.self_loops,
        &config.mcmc,
        config.admissible_pairs.as_deref(),
    )?;

    // ---- Steps 3-4: Allocate occupations ----
    let occupations = if e == 1 {
        vec![total]
    } else if total == e as OccNum {
        vec![1; e]
    } else {
        sample_fixed_total(family, e, total, &gibbs_config)
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
    debug_assert_eq!(result.occ_nums.iter().copied().sum::<OccNum>(), total);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::family::OccupationFamily;

    #[test]
    fn me_directed_cycle() {
        // N=4, each node out=1, in=1, T=8 (2 events per edge)
        let out = vec![1u32, 1, 1, 1];
        let inp = vec![1u32, 1, 1, 1];
        let config = FixedKTConfig {
            mcmc: FixedDegreeMcmcConfig {
                burn_in_sweeps: 10,
                sweeps_per_sample: 5,
                proposals_per_sweep: None,
                seed: 42,
                self_loops: false,
            },
            self_loops: false,
            admissible_pairs: None,
        };
        let result = sample_fixed_kt_core(OccupationFamily::ME, &out, &inp, 8, &config).unwrap();
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
        for item in inp.iter_mut().skip(1) {
            *item = 1;
        }
        let config = FixedKTConfig {
            mcmc: FixedDegreeMcmcConfig {
                burn_in_sweeps: 10,
                sweeps_per_sample: 5,
                proposals_per_sweep: None,
                seed: 42,
                self_loops: false,
            },
            self_loops: false,
            admissible_pairs: None,
        };
        let t = (n as OccNum - 1) * 3;
        let result = sample_fixed_kt_core(OccupationFamily::ME, &out, &inp, t, &config).unwrap();
        assert_eq!(result.sources.len(), n - 1);
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
                proposals_per_sweep: None,
                seed: 42,
                self_loops: false,
            },
            self_loops: false,
            admissible_pairs: None,
        };
        let result =
            sample_fixed_kt_core(OccupationFamily::B { layers: 4 }, &out, &inp, 6, &config)
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
                proposals_per_sweep: None,
                seed: 42,
                self_loops: false,
            },
            self_loops: false,
            admissible_pairs: None,
        };
        let result =
            sample_fixed_kt_core(OccupationFamily::W { layers: 2 }, &out, &inp, 10, &config)
                .unwrap();
        assert_eq!(result.sources.len(), 4);
        assert_eq!(result.occ_nums.iter().sum::<OccNum>(), 10);
    }

    #[test]
    fn infeasible_t_below_e() {
        let out = vec![2u32, 2];
        let inp = vec![2u32, 2];
        let config = FixedKTConfig::default();
        let result = sample_fixed_kt_core(OccupationFamily::ME, &out, &inp, 3, &config);
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
                proposals_per_sweep: None,
                seed: 42,
                self_loops: false,
            },
            self_loops: false,
            admissible_pairs: None,
        };
        let a = sample_fixed_kt_core(OccupationFamily::ME, &out, &inp, 8, &config).unwrap();
        let b = sample_fixed_kt_core(OccupationFamily::ME, &out, &inp, 8, &config).unwrap();
        assert_eq!(a.sources, b.sources);
        assert_eq!(a.targets, b.targets);
        assert_eq!(a.occ_nums, b.occ_nums);
    }

    #[test]
    fn me_directed_self_loops() {
        // N=2 with full degree 1 each, self_loops=true, total=2.
        // Possible supports with out=[1,1], in=[1,1], self_loops=true:
        //   {(0,0),(1,1)}  or  {(0,1),(1,0)}
        // Both are valid and the chain should explore both.
        let out = vec![1u32, 1];
        let inp = vec![1u32, 1];
        let config = FixedKTConfig {
            mcmc: FixedDegreeMcmcConfig {
                burn_in_sweeps: 20,
                sweeps_per_sample: 10,
                proposals_per_sweep: None,
                seed: 42,
                self_loops: true,
            },
            self_loops: true,
            admissible_pairs: None,
        };
        let result = sample_fixed_kt_core(OccupationFamily::ME, &out, &inp, 2, &config).unwrap();
        assert_eq!(result.sources.len(), 2);
        assert_eq!(result.occ_nums.iter().sum::<OccNum>(), 2);
        // Verify support degrees
        let mut support_out = vec![0u32; 2];
        for &s in &result.sources {
            support_out[s as usize] += 1;
        }
        assert_eq!(support_out, out);
        // Self-loops may or may not be present in this particular sample,
        // but the chain should have run without panic and with correct degrees.
    }

    #[test]
    fn me_self_loops_with_occupation() {
        // N=2, out=[2,1], in=[1,2], total=5 → E=3
        // With self_loops=true this is realizable.
        let out = vec![2u32, 1];
        let inp = vec![1u32, 2];
        let config = FixedKTConfig {
            mcmc: FixedDegreeMcmcConfig {
                burn_in_sweeps: 20,
                sweeps_per_sample: 10,
                proposals_per_sweep: None,
                seed: 42,
                self_loops: true,
            },
            self_loops: true,
            admissible_pairs: None,
        };
        let result = sample_fixed_kt_core(OccupationFamily::ME, &out, &inp, 5, &config).unwrap();
        assert_eq!(result.sources.len(), 3);
        assert_eq!(result.occ_nums.iter().sum::<OccNum>(), 5);
        // Verify support degrees
        let mut support_out = vec![0u32; 2];
        for &s in &result.sources {
            support_out[s as usize] += 1;
        }
        assert_eq!(support_out, out);
    }
}
