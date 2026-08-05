//! Sparse directed support state for the fixed-degree MCMC.

use std::collections::{HashMap, HashSet};

/// Sparse state for a directed simple graph support.
///
/// Maintains three synchronized structures:
/// - `edges`:  `Vec` of ordered `(src, tgt)` pairs for O(1) uniform sampling.
/// - `edge_positions`:  maps each pair to its index in `edges` for O(1) swap-remove.
/// - `out_adjacency`:  per-node outgoing neighbour set for O(1) duplicate checks.
///
/// Memory: \(O(N + E)\).  No dense adjacency matrix.
#[derive(Clone, Debug)]
pub struct DegreeSupportState {
    pub node_count: usize,
    pub edges: Vec<(u64, u64)>,
    edge_positions: HashMap<(u64, u64), usize>,
    out_adjacency: Vec<HashSet<u64>>,
    // Cache of in-degrees for debug validation (computed from edges on demand).
}

impl DegreeSupportState {
    /// Create a new support state from a list of edges.
    ///
    /// The edges are assumed to be a valid simple directed graph with no
    /// self-loops (when `self_loops` is false) and no parallel edges.
    pub fn new(node_count: usize, edges: Vec<(u64, u64)>, self_loops: bool) -> Self {
        let m = edges.len();
        let mut edge_positions = HashMap::with_capacity(m);
        let mut out_adjacency: Vec<HashSet<u64>> = (0..node_count).map(|_| HashSet::new()).collect();

        for (idx, &(src, tgt)) in edges.iter().enumerate() {
            debug_assert!(
                self_loops || src != tgt,
                "self-loop found but self_loops=false"
            );
            debug_assert!(
                (src as usize) < node_count && (tgt as usize) < node_count,
                "pair out of range"
            );
            edge_positions.insert((src, tgt), idx);
            out_adjacency[src as usize].insert(tgt);
        }
        debug_assert_eq!(edge_positions.len(), m);
        debug_assert_eq!(
            out_adjacency.iter().map(|s| s.len()).sum::<usize>(),
            m
        );

        Self {
            node_count,
            edges,
            edge_positions,
            out_adjacency,
        }
    }

    /// Number of edges in the current support.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Check whether an ordered pair is currently occupied.
    #[inline]
    pub fn contains(&self, pair: &(u64, u64)) -> bool {
        self.edge_positions.contains_key(pair)
    }

    /// Add an ordered pair to the support.
    ///
    /// # Panics
    ///
    /// Panics if the pair is already present in debug mode.
    #[inline]
    pub fn insert(&mut self, pair: (u64, u64)) {
        debug_assert!(!self.contains(&pair), "inserting duplicate edge {:?}", pair);
        let idx = self.edges.len();
        self.edges.push(pair);
        self.edge_positions.insert(pair, idx);
        self.out_adjacency[pair.0 as usize].insert(pair.1);
    }

    /// Remove an ordered pair from the support (swap-remove).
    ///
    /// # Panics
    ///
    /// Panics if the pair is not present in debug mode.
    #[inline]
    pub fn remove(&mut self, pair: &(u64, u64)) {
        let idx = self.edge_positions.remove(pair).expect("removing non-existent edge");
        let last = self.edges.pop().unwrap();
        if idx < self.edges.len() {
            self.edges[idx] = last;
            self.edge_positions.insert(last, idx);
        }
        self.out_adjacency[pair.0 as usize].remove(&pair.1);
    }

    /// Return the current out-degree of a node.
    #[inline]
    pub fn out_degree(&self, node: usize) -> usize {
        self.out_adjacency[node].len()
    }

    /// Compute the in-degree of a node by scanning edges (for debug validation).
    #[inline]
    pub fn in_degree(&self, node: usize) -> usize {
        self.edges.iter().filter(|&&(_, tgt)| tgt as usize == node).count()
    }

    /// Compute the full out-degree sequence.
    #[inline]
    pub fn out_degree_sequence(&self) -> Vec<u32> {
        (0..self.node_count)
            .map(|i| self.out_adjacency[i].len() as u32)
            .collect()
    }

    /// Debug validation: check internal consistency.
    #[cfg(debug_assertions)]
    pub fn debug_validate(&self) {
        // Edge vector and map agreement
        assert_eq!(self.edges.len(), self.edge_positions.len());
        for (idx, &(src, tgt)) in self.edges.iter().enumerate() {
            assert_eq!(self.edge_positions[&(src, tgt)], idx);
            assert!(self.out_adjacency[src as usize].contains(&tgt));
        }
        // Out-adjacency agrees with edge count
        let total_out: usize = self.out_adjacency.iter().map(|s| s.len()).sum();
        assert_eq!(total_out, self.edges.len());
    }

    /// Debug build validation skipped in release.
    #[cfg(not(debug_assertions))]
    pub fn debug_validate(&self) {}
}

impl std::fmt::Display for DegreeSupportState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DegreeSupportState(n={}, E={})", self.node_count, self.edges.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_state() {
        let state = DegreeSupportState::new(5, vec![], false);
        assert_eq!(state.edge_count(), 0);
        assert!(!state.contains(&(0, 1)));
    }

    #[test]
    fn insert_and_remove() {
        let mut state = DegreeSupportState::new(5, vec![], false);
        state.insert((0, 1));
        assert!(state.contains(&(0, 1)));
        assert_eq!(state.edge_count(), 1);
        assert_eq!(state.out_degree(0), 1);
        assert_eq!(state.in_degree(1), 1);

        state.remove(&(0, 1));
        assert!(!state.contains(&(0, 1)));
        assert_eq!(state.edge_count(), 0);
    }

    #[test]
    fn swap_remove_preserves_correctness() {
        let mut state = DegreeSupportState::new(4, vec![(0, 1), (1, 2), (2, 3)], false);
        assert_eq!(state.edge_count(), 3);
        state.remove(&(1, 2));
        assert_eq!(state.edge_count(), 2);
        assert!(state.contains(&(0, 1)));
        assert!(state.contains(&(2, 3)));
        assert!(!state.contains(&(1, 2)));
    }

    #[test]
    fn build_from_edges() {
        let edges = vec![(0, 1), (1, 0), (0, 2), (2, 1)];
        let state = DegreeSupportState::new(3, edges, false);
        assert_eq!(state.edge_count(), 4);
        assert_eq!(state.out_degree(0), 2);
        assert_eq!(state.in_degree(0), 1);
    }
}