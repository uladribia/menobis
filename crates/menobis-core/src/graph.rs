//! Occupied-pair graph primitives for MENoBiS.

use crate::OccNum;

/// An occupied directed pair (i→j) with positive integer occupation number.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OccupiedPair {
    /// Source node identifier.
    pub source: usize,
    /// Target node identifier.
    pub target: usize,
    /// Pair occupation number (t_ij > 0).
    pub occ_num: OccNum,
}

impl OccupiedPair {
    /// Create an occupied pair.
    #[must_use]
    pub const fn new(source: usize, target: usize, occ_num: OccNum) -> Self {
        Self {
            source,
            target,
            occ_num,
        }
    }
}

/// Lazy sparse view over an occupied-pair edge list.
///
/// Carries the node count and the edge slice; adjacency is only built by
/// the clustering routines that actually need it (O(N + E) memory).
#[derive(Clone, Copy, Debug)]
pub struct SparseGraphView<'a> {
    pub node_count: usize,
    pub edges: &'a [OccupiedPair],
}

impl<'a> SparseGraphView<'a> {
    /// Build a view, deriving the node count from the edges.
    #[must_use]
    pub fn from_edges(edges: &'a [OccupiedPair]) -> Self {
        let node_count = edges
            .iter()
            .map(|e| e.source.max(e.target))
            .max()
            .map_or(0, |m| m + 1);
        Self { node_count, edges }
    }
}

/// Directed node sequences for occupied origin-destination pairs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectedNodeSequence {
    /// Outgoing value per node.
    pub out: Vec<u64>,
    /// Incoming value per node.
    pub incoming: Vec<u64>,
}

/// Compute directed incoming and outgoing strengths.
#[must_use]
pub fn directed_strengths(node_count: usize, edges: &[OccupiedPair]) -> DirectedNodeSequence {
    let mut out = vec![0_u64; node_count];
    let mut incoming = vec![0_u64; node_count];

    for edge in edges {
        out[edge.source] += edge.occ_num;
        incoming[edge.target] += edge.occ_num;
    }

    DirectedNodeSequence { out, incoming }
}

/// Compute directed incoming and outgoing binary degrees.
#[must_use]
pub fn directed_degrees(node_count: usize, edges: &[OccupiedPair]) -> DirectedNodeSequence {
    let mut out = vec![0_u64; node_count];
    let mut incoming = vec![0_u64; node_count];

    for edge in edges {
        out[edge.source] += 1;
        incoming[edge.target] += 1;
    }

    DirectedNodeSequence { out, incoming }
}

#[cfg(test)]
mod tests {
    use super::{directed_degrees, directed_strengths, OccupiedPair, SparseGraphView};

    #[test]
    fn sparse_graph_view_derives_node_count() {
        let edges = [OccupiedPair::new(0, 1, 3), OccupiedPair::new(2, 4, 1)];
        let view = SparseGraphView::from_edges(&edges);
        assert_eq!(view.node_count, 5);
        assert_eq!(view.edges.len(), 2);
        assert_eq!(SparseGraphView::from_edges(&[]).node_count, 0);
    }

    #[test]
    fn directed_strengths_conserve_total_weight() {
        let edges = [OccupiedPair::new(0, 1, 3), OccupiedPair::new(1, 2, 4)];

        let strengths = directed_strengths(3, &edges);

        assert_eq!(strengths.out, vec![3, 4, 0]);
        assert_eq!(strengths.incoming, vec![0, 3, 4]);
        assert_eq!(strengths.out.iter().sum::<u64>(), 7);
        assert_eq!(strengths.incoming.iter().sum::<u64>(), 7);
    }

    #[test]
    fn directed_degrees_count_binary_edges() {
        let edges = [
            OccupiedPair::new(0, 1, 3),
            OccupiedPair::new(0, 2, 4),
            OccupiedPair::new(1, 2, 5),
        ];

        let degrees = directed_degrees(3, &edges);

        assert_eq!(degrees.out, vec![2, 1, 0]);
        assert_eq!(degrees.incoming, vec![0, 1, 2]);
        assert_eq!(degrees.out.iter().sum::<u64>(), 3);
        assert_eq!(degrees.incoming.iter().sum::<u64>(), 3);
    }
}
