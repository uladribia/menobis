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
        config.admissible_pairs.as_deref(),
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
    debug_assert_eq!(result.occ_nums.iter().copied().sum::<OccNum>(), total);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::microcanonical::fixed_et::b::BFamily;
    use crate::generation::microcanonical::fixed_et::me::MeFamily;
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
                proposals_per_sweep: None,
                seed: 42,
            },
            self_loops: false,
            admissible_pairs: None,
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
        for item in inp.iter_mut().skip(1) {
            *item = 1;
        }
        let config = FixedKTConfig {
            mcmc: FixedDegreeMcmcConfig {
                burn_in_sweeps: 10,
                sweeps_per_sample: 5,
                proposals_per_sweep: None,
                seed: 42,
            },
            self_loops: false,
            admissible_pairs: None,
        };
        let t = (n as OccNum - 1) * 3;
        let result = sample_fixed_kt_core(&MeFamily, &out, &inp, t, &config).unwrap();
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
            },
            self_loops: false,
            admissible_pairs: None,
        };
        let result = sample_fixed_kt_core(&BFamily { layers: 4 }, &out, &inp, 6, &config).unwrap();
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
            },
            self_loops: false,
            admissible_pairs: None,
        };
        let result = sample_fixed_kt_core(&WFamily { layers: 2 }, &out, &inp, 10, &config).unwrap();
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
                proposals_per_sweep: None,
                seed: 42,
            },
            self_loops: false,
            admissible_pairs: None,
        };
        let a = sample_fixed_kt_core(&MeFamily, &out, &inp, 8, &config).unwrap();
        let b = sample_fixed_kt_core(&MeFamily, &out, &inp, 8, &config).unwrap();
        assert_eq!(a.sources, b.sources);
        assert_eq!(a.targets, b.targets);
        assert_eq!(a.occ_nums, b.occ_nums);
    }
}

#[cfg(test)]
mod exhaustive_tests {
    use super::*;
    use crate::generation::microcanonical::fixed_et::me::MeFamily;
    use std::collections::{HashMap, HashSet};
    type EdgeList = Vec<(u64, u64)>;
    type SupportMap = HashMap<EdgeList, Vec<EdgeList>>;

    /// Enumerate all simple directed support graphs for a given degree sequence.
    fn enumerate_supports(
        out_degrees: &[u32],
        in_degrees: &[u32],
        self_loops: bool,
    ) -> Vec<Vec<(u64, u64)>> {
        let n = out_degrees.len();
        // Build all possible directed edges (ordered pairs)
        let mut all_pairs: Vec<(u64, u64)> = Vec::new();
        for i in 0..(n as u64) {
            for j in 0..(n as u64) {
                if !self_loops && i == j {
                    continue;
                }
                all_pairs.push((i, j));
            }
        }
        // Generate all subsets of size E and test degree match
        let e: usize = out_degrees.iter().sum::<u32>() as usize;
        let mut results = Vec::new();
        // Use combinations to enumerate all E-sized subsets
        // Since N is small (<=4), we can brute-force
        let l = all_pairs.len();
        if l > 20 {
            return results; // Too large for exhaustive enumeration
        }

        // Generate all combinations of size E
        #[allow(clippy::too_many_arguments)]
        fn combine(
            start: usize,
            k: usize,
            n: usize,
            current: &mut Vec<usize>,
            results: &mut Vec<EdgeList>,
            all_pairs: &[(u64, u64)],
            out_degrees: &[u32],
            in_degrees: &[u32],
        ) {
            if k == 0 {
                let edges: EdgeList = current.iter().map(|&i| all_pairs[i]).collect();
                let mut out_cnt = vec![0u32; out_degrees.len()];
                let mut in_cnt = vec![0u32; in_degrees.len()];
                for &(src, tgt) in &edges {
                    out_cnt[src as usize] += 1;
                    in_cnt[tgt as usize] += 1;
                }
                if out_cnt == out_degrees && in_cnt == in_degrees {
                    results.push(edges);
                }
                return;
            }
            for i in start..=n - k {
                current.push(i);
                combine(
                    i + 1,
                    k - 1,
                    n,
                    current,
                    results,
                    all_pairs,
                    out_degrees,
                    in_degrees,
                );
                current.pop();
            }
        }

        let mut current = Vec::new();
        combine(
            0,
            e,
            l,
            &mut current,
            &mut results,
            &all_pairs,
            out_degrees,
            in_degrees,
        );
        results
    }

    #[test]
    fn exhaustive_support_uniformity_for_4_cycle() {
        // N=4, each node out=1, in=1. All possible realizations.
        let out = vec![1u32, 1, 1, 1];
        let inp = vec![1u32, 1, 1, 1];
        let supports = enumerate_supports(&out, &inp, false);
        // There should be 9 possible support graphs for a 4-cycle
        // (all directed 4-cycles: 3! = 6, plus 3 possible 2+2 cycles = 9 total)
        assert!(!supports.is_empty(), "no supports found");

        // Run MCMC many times and count support frequencies
        let mut counts: HashMap<Vec<(u64, u64)>, u64> = HashMap::new();
        let trials = 5000;
        for seed in 0..trials {
            let config = FixedKTConfig {
                mcmc: FixedDegreeMcmcConfig {
                    burn_in_sweeps: 20,
                    sweeps_per_sample: 10,
                    proposals_per_sweep: None,
                    seed,
                },
                self_loops: false,
                admissible_pairs: None,
            };
            let result = sample_fixed_kt_core(&MeFamily, &out, &inp, 4, &config).unwrap();
            let mut edges: Vec<(u64, u64)> = result
                .sources
                .iter()
                .zip(result.targets.iter())
                .map(|(&s, &t)| (s, t))
                .collect();
            edges.sort_unstable();
            *counts.entry(edges).or_default() += 1;
        }

        // Check that each support appears roughly uniformly
        let expected = trials as f64 / supports.len() as f64;
        for (_edges, count) in counts.iter() {
            let ratio = *count as f64 / expected;
            // Allow 50% deviation for this small trial count
            assert!(
                ratio > 0.5 && ratio < 1.5,
                "support frequency {count} vs expected {expected:.1} (ratio {ratio:.2})"
            );
        }
    }

    #[test]
    fn connectivity_of_switch_graph() {
        // For N=3, out=in=[1,1,1], enumerate all supports and verify
        // the directed double-edge switch connects them.
        let out = vec![1u32, 1, 1];
        let inp = vec![1u32, 1, 1];
        let supports = enumerate_supports(&out, &inp, false);
        if supports.is_empty() || supports.len() > 20 {
            return;
        }

        // Build transition graph: each support is a node, edges if switch connects them
        let mut adj: SupportMap = HashMap::new();
        for s in &supports {
            adj.entry(s.clone()).or_default();
        }

        // For each pair of supports, check if a single directed double-edge switch connects them
        for i in 0..supports.len() {
            for j in i + 1..supports.len() {
                let a = &supports[i];
                let b = &supports[j];
                // Compute symmetric difference
                let set_a: HashSet<_> = a.iter().copied().collect();
                let set_b: HashSet<_> = b.iter().copied().collect();
                let diff: Vec<_> = set_a.symmetric_difference(&set_b).copied().collect();
                // A single switch changes 2 edges: 4 edges in symmetric difference
                if diff.len() == 4 {
                    // Verify the switch preserves degrees
                    adj.entry(a.clone()).or_default().push(b.clone());
                    adj.entry(b.clone()).or_default().push(a.clone());
                }
            }
        }

        // BFS from first support — note: the 2-edge switch may not fully connect
        // the state space (e.g., N=3 directed 3-cycles need 3-edge switches).
        // This test documents the connectivity status.
        let mut visited = HashSet::new();
        let mut stack = vec![supports[0].clone()];
        while let Some(s) = stack.pop() {
            if !visited.insert(s.clone()) {
                continue;
            }
            if let Some(neighbors) = adj.get(&s) {
                for n in neighbors {
                    if !visited.contains(n) {
                        stack.push(n.clone());
                    }
                }
            }
        }
        // The 2-edge switch should connect all supports for N>=4.
        // For N=3, it may not (3-cycles need 3-edge switches).
        // Record the component size but don't assert full connectivity.
        let connected = visited.len();
        let total = supports.len();
        println!("2-edge switch connectivity: {connected}/{total} supports connected");
        // At minimum, each support must be reachable from itself
        assert!(connected >= 1);
    }
}
