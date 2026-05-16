//! Clustering coefficients for undirected projections.

use crate::graph::WeightedEdge;
use std::collections::{HashMap, HashSet};

/// Binary clustering coefficient per node.
#[must_use]
pub fn clustering_coefficients(node_count: usize, edges: &[WeightedEdge]) -> Vec<f64> {
    let adj = build_undirected_adjacency(node_count, edges);
    let mut result = vec![0.0_f64; node_count];

    for i in 0..node_count {
        let neighbors: Vec<usize> = adj[i].iter().copied().collect();
        let k = neighbors.len();
        if k < 2 {
            continue;
        }
        let mut triangles = 0_u64;
        for (idx, &n1) in neighbors.iter().enumerate() {
            for &n2 in &neighbors[idx + 1..] {
                if adj[n1].contains(&n2) {
                    triangles += 1;
                }
            }
        }
        result[i] = 2.0 * triangles as f64 / (k * (k - 1)) as f64;
    }

    result
}

/// Weighted clustering coefficient per node.
///
/// c^w_i = sum_{jk} (w_ij + w_ik) * Theta(ij) * Theta(jk) * Theta(ki)
///         / (2 * s_i * (k_i - 1))
#[must_use]
pub fn weighted_clustering_coefficients(node_count: usize, edges: &[WeightedEdge]) -> Vec<f64> {
    let (adj, weights) = build_undirected_adjacency_weighted(node_count, edges);
    let mut result = vec![0.0_f64; node_count];

    // Compute strength and degree
    let mut s = vec![0.0_f64; node_count];
    let mut k = vec![0_usize; node_count];
    for i in 0..node_count {
        k[i] = adj[i].len();
        s[i] = adj[i]
            .iter()
            .map(|&j| *weights.get(&(i, j)).unwrap_or(&0) as f64)
            .sum();
    }

    for i in 0..node_count {
        if k[i] < 2 || s[i] == 0.0 {
            continue;
        }
        let neighbors: Vec<usize> = adj[i].iter().copied().collect();
        let mut num = 0.0_f64;
        for (idx, &n1) in neighbors.iter().enumerate() {
            let w_ij = *weights.get(&(i, n1)).unwrap_or(&0) as f64;
            for &n2 in &neighbors[idx + 1..] {
                if adj[n1].contains(&n2) {
                    let w_ik = *weights.get(&(i, n2)).unwrap_or(&0) as f64;
                    num += w_ij + w_ik;
                }
            }
        }
        result[i] = num / (2.0 * s[i] * (k[i] - 1) as f64);
    }

    result
}

fn build_undirected_adjacency(node_count: usize, edges: &[WeightedEdge]) -> Vec<HashSet<usize>> {
    let mut adj = vec![HashSet::new(); node_count];
    for e in edges {
        adj[e.source].insert(e.target);
        adj[e.target].insert(e.source);
    }
    adj
}

type UndirectedWeighted = (Vec<HashSet<usize>>, HashMap<(usize, usize), u64>);

fn build_undirected_adjacency_weighted(
    node_count: usize,
    edges: &[WeightedEdge],
) -> UndirectedWeighted {
    let mut adj = vec![HashSet::new(); node_count];
    let mut weights: HashMap<(usize, usize), u64> = HashMap::new();
    for e in edges {
        adj[e.source].insert(e.target);
        adj[e.target].insert(e.source);
        // Keep max weight for undirected projection
        weights
            .entry((e.source, e.target))
            .and_modify(|w| *w = (*w).max(e.weight))
            .or_insert(e.weight);
        weights
            .entry((e.target, e.source))
            .and_modify(|w| *w = (*w).max(e.weight))
            .or_insert(e.weight);
    }
    (adj, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::WeightedEdge;

    #[test]
    fn triangle_has_clustering_one() {
        let edges = [
            WeightedEdge::new(0, 1, 1),
            WeightedEdge::new(1, 2, 1),
            WeightedEdge::new(2, 0, 1),
        ];
        let c = clustering_coefficients(3, &edges);
        for &ci in &c {
            assert!((ci - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn star_has_zero_clustering() {
        let edges = [
            WeightedEdge::new(0, 1, 1),
            WeightedEdge::new(0, 2, 1),
            WeightedEdge::new(0, 3, 1),
        ];
        let c = clustering_coefficients(4, &edges);
        assert_eq!(c[0], 0.0);
    }

    #[test]
    fn weighted_triangle_positive() {
        let edges = [
            WeightedEdge::new(0, 1, 2),
            WeightedEdge::new(1, 2, 3),
            WeightedEdge::new(2, 0, 4),
        ];
        let wc = weighted_clustering_coefficients(3, &edges);
        for &ci in &wc {
            assert!(ci > 0.0);
        }
    }
}
