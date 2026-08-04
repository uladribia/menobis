//! Network statistics for non-binary directed networks.

use crate::graph::OccupiedPair;
use std::collections::HashMap;

/// All per-node statistics computed in a single pass.
#[derive(Clone, Debug)]
pub struct NodeStats {
    pub node_count: usize,
    pub strength_out: Vec<u64>,
    pub strength_in: Vec<u64>,
    pub degree_out: Vec<u64>,
    pub degree_in: Vec<u64>,
    pub y2_out: Vec<f64>,
    pub y2_in: Vec<f64>,
    pub s_nn_out: Vec<f64>,
    pub s_nn_in: Vec<f64>,
    pub k_nn_out: Vec<f64>,
    pub k_nn_in: Vec<f64>,
}

/// Occupation-number distribution entry.
#[derive(Clone, Debug)]
pub struct OccupationDistribution {
    pub occ_nums: Vec<crate::OccNum>,
    pub counts: Vec<u64>,
}

/// Compute all per-node statistics in a single pass over edges.
#[must_use]
pub fn compute_all_node_stats(node_count: usize, edges: &[OccupiedPair]) -> NodeStats {
    let mut s_out = vec![0_u64; node_count];
    let mut s_in = vec![0_u64; node_count];
    let mut k_out = vec![0_u64; node_count];
    let mut k_in = vec![0_u64; node_count];
    let mut sum_w2_out = vec![0_u64; node_count];
    let mut sum_w2_in = vec![0_u64; node_count];

    for e in edges {
        s_out[e.source] += e.occ_num;
        s_in[e.target] += e.occ_num;
        k_out[e.source] += 1;
        k_in[e.target] += 1;
        sum_w2_out[e.source] += e.occ_num * e.occ_num;
        sum_w2_in[e.target] += e.occ_num * e.occ_num;
    }

    // Y2
    let y2_out: Vec<f64> = (0..node_count)
        .map(|i| {
            if s_out[i] > 0 {
                sum_w2_out[i] as f64 / (s_out[i] as f64 * s_out[i] as f64)
            } else {
                0.0
            }
        })
        .collect();
    let y2_in: Vec<f64> = (0..node_count)
        .map(|j| {
            if s_in[j] > 0 {
                sum_w2_in[j] as f64 / (s_in[j] as f64 * s_in[j] as f64)
            } else {
                0.0
            }
        })
        .collect();

    // s^w_nn: need sum_j t_ij * s_in(j) for each i, and sum_i t_ij * s_out(i) for each j
    let mut weighted_s_in_sum_out = vec![0.0_f64; node_count]; // for s_nn_out
    let mut weighted_s_out_sum_in = vec![0.0_f64; node_count]; // for s_nn_in
    let mut weighted_k_in_sum_out = vec![0.0_f64; node_count]; // for k_nn_out
    let mut weighted_k_out_sum_in = vec![0.0_f64; node_count]; // for k_nn_in

    for e in edges {
        weighted_s_in_sum_out[e.source] += e.occ_num as f64 * s_in[e.target] as f64;
        weighted_s_out_sum_in[e.target] += e.occ_num as f64 * s_out[e.source] as f64;
        weighted_k_in_sum_out[e.source] += k_in[e.target] as f64;
        weighted_k_out_sum_in[e.target] += k_out[e.source] as f64;
    }

    let s_nn_out: Vec<f64> = (0..node_count)
        .map(|i| {
            if s_out[i] > 0 {
                weighted_s_in_sum_out[i] / s_out[i] as f64
            } else {
                0.0
            }
        })
        .collect();
    let s_nn_in: Vec<f64> = (0..node_count)
        .map(|j| {
            if s_in[j] > 0 {
                weighted_s_out_sum_in[j] / s_in[j] as f64
            } else {
                0.0
            }
        })
        .collect();
    let k_nn_out: Vec<f64> = (0..node_count)
        .map(|i| {
            if k_out[i] > 0 {
                weighted_k_in_sum_out[i] / k_out[i] as f64
            } else {
                0.0
            }
        })
        .collect();
    let k_nn_in: Vec<f64> = (0..node_count)
        .map(|j| {
            if k_in[j] > 0 {
                weighted_k_out_sum_in[j] / k_in[j] as f64
            } else {
                0.0
            }
        })
        .collect();

    NodeStats {
        node_count,
        strength_out: s_out,
        strength_in: s_in,
        degree_out: k_out,
        degree_in: k_in,
        y2_out,
        y2_in,
        s_nn_out,
        s_nn_in,
        k_nn_out,
        k_nn_in,
    }
}

/// Compute the occupation-number distribution P(t_ij).
#[must_use]
pub fn occupation_distribution(edges: &[OccupiedPair]) -> OccupationDistribution {
    let mut counts: HashMap<u64, u64> = HashMap::new();
    for e in edges {
        *counts.entry(e.occ_num).or_insert(0) += 1;
    }
    let mut pairs: Vec<(u64, u64)> = counts.into_iter().collect();
    pairs.sort_unstable_by_key(|&(occ, _)| occ);
    OccupationDistribution {
        occ_nums: pairs.iter().map(|&(occ, _)| occ).collect(),
        counts: pairs.iter().map(|&(_, c)| c).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::OccupiedPair;

    #[test]
    fn all_stats_on_small_graph() {
        let edges = [
            OccupiedPair::new(0, 1, 3),
            OccupiedPair::new(0, 2, 4),
            OccupiedPair::new(1, 2, 5),
        ];

        let stats = compute_all_node_stats(3, &edges);

        assert_eq!(stats.strength_out, vec![7, 5, 0]);
        assert_eq!(stats.strength_in, vec![0, 3, 9]);
        assert_eq!(stats.degree_out, vec![2, 1, 0]);
        assert_eq!(stats.degree_in, vec![0, 1, 2]);
        assert!(stats.y2_out[0] > 0.0);
        assert_eq!(stats.y2_out[2], 0.0);
    }

    #[test]
    fn occupation_distribution_counts() {
        let edges = [
            OccupiedPair::new(0, 1, 3),
            OccupiedPair::new(0, 2, 3),
            OccupiedPair::new(1, 2, 5),
        ];

        let dist = occupation_distribution(&edges);

        assert_eq!(dist.occ_nums, vec![3, 5]);
        assert_eq!(dist.counts, vec![2, 1]);
    }
}
