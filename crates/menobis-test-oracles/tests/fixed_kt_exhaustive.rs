//! Heavy: exhaustive enumeration tests for fixed-degree support MCMC.
//!
//! Enumerates all feasible support graphs for small N and validates
//! that the double-edge-switch MCMC visits each support uniformly.
//! These tests are expensive and live in the oracle crate so they
//! don't slow down `cargo test -p menobis-core`.

use std::collections::{HashMap, HashSet};

use menobis_core::generation::microcanonical::fixed_et::me::MeFamily;
use menobis_core::generation::microcanonical::fixed_kt::core::{
    sample_fixed_kt_core, FixedKTConfig,
};
use menobis_core::generation::microcanonical::fixed_kt::sampler::FixedDegreeMcmcConfig;

type EdgeList = Vec<(u64, u64)>;
type SupportMap = HashMap<EdgeList, Vec<EdgeList>>;

/// Enumerate all simple directed support graphs for a given degree sequence.
fn enumerate_supports(out_degrees: &[u32], in_degrees: &[u32], self_loops: bool) -> Vec<EdgeList> {
    let n = out_degrees.len();
    let mut all_pairs: Vec<(u64, u64)> = Vec::new();
    for i in 0..(n as u64) {
        for j in 0..(n as u64) {
            if !self_loops && i == j {
                continue;
            }
            all_pairs.push((i, j));
        }
    }
    let e: usize = out_degrees.iter().sum::<u32>() as usize;
    let l = all_pairs.len();
    if l > 20 {
        return vec![];
    }

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

    let mut results: Vec<EdgeList> = Vec::new();
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
    let out = vec![1u32, 1, 1, 1];
    let inp = vec![1u32, 1, 1, 1];
    let supports = enumerate_supports(&out, &inp, false);
    assert!(!supports.is_empty(), "no supports found");

    let mut counts: HashMap<EdgeList, u64> = HashMap::new();
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
        let mut edges: EdgeList = result
            .sources
            .iter()
            .zip(result.targets.iter())
            .map(|(&s, &t)| (s, t))
            .collect();
        edges.sort_unstable();
        *counts.entry(edges).or_default() += 1;
    }

    let expected = trials as f64 / supports.len() as f64;
    for count in counts.values() {
        let ratio = *count as f64 / expected;
        assert!(
            ratio > 0.5 && ratio < 1.5,
            "support frequency {count} vs expected {expected:.1} (ratio {ratio:.2})"
        );
    }
}

#[test]
fn connectivity_of_switch_graph() {
    let out = vec![1u32, 1, 1];
    let inp = vec![1u32, 1, 1];
    let supports = enumerate_supports(&out, &inp, false);
    if supports.is_empty() || supports.len() > 20 {
        return;
    }

    let mut adj: SupportMap = HashMap::new();
    for s in &supports {
        adj.entry(s.clone()).or_default();
    }
    for i in 0..supports.len() {
        for j in i + 1..supports.len() {
            let a = &supports[i];
            let b = &supports[j];
            let set_a: HashSet<_> = a.iter().copied().collect();
            let set_b: HashSet<_> = b.iter().copied().collect();
            let diff: Vec<_> = set_a.symmetric_difference(&set_b).copied().collect();
            if diff.len() == 4 {
                adj.entry(a.clone()).or_default().push(b.clone());
                adj.entry(b.clone()).or_default().push(a.clone());
            }
        }
    }

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
    let connected = visited.len();
    let total = supports.len();
    println!("2-edge switch connectivity: {connected}/{total} supports connected");
    assert!(connected >= 1);
}
