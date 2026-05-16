//! Weighted edge-list graph primitives for ODME.

/// A weighted directed edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WeightedEdge {
    /// Source node identifier.
    pub source: usize,
    /// Target node identifier.
    pub target: usize,
    /// Positive integer edge weight.
    pub weight: u64,
}

impl WeightedEdge {
    /// Create a weighted edge.
    #[must_use]
    pub const fn new(source: usize, target: usize, weight: u64) -> Self {
        Self {
            source,
            target,
            weight,
        }
    }
}

/// Directed node sequences for weighted origin-destination edges.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectedNodeSequence {
    /// Outgoing value per node.
    pub out: Vec<u64>,
    /// Incoming value per node.
    pub incoming: Vec<u64>,
}

/// Compute directed incoming and outgoing strengths.
#[must_use]
pub fn directed_strengths(node_count: usize, edges: &[WeightedEdge]) -> DirectedNodeSequence {
    let mut out = vec![0_u64; node_count];
    let mut incoming = vec![0_u64; node_count];

    for edge in edges {
        out[edge.source] += edge.weight;
        incoming[edge.target] += edge.weight;
    }

    DirectedNodeSequence { out, incoming }
}

/// Compute directed incoming and outgoing binary degrees.
#[must_use]
pub fn directed_degrees(node_count: usize, edges: &[WeightedEdge]) -> DirectedNodeSequence {
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
    use super::{directed_degrees, directed_strengths, WeightedEdge};

    #[test]
    fn directed_strengths_conserve_total_weight() {
        let edges = [WeightedEdge::new(0, 1, 3), WeightedEdge::new(1, 2, 4)];

        let strengths = directed_strengths(3, &edges);

        assert_eq!(strengths.out, vec![3, 4, 0]);
        assert_eq!(strengths.incoming, vec![0, 3, 4]);
        assert_eq!(strengths.out.iter().sum::<u64>(), 7);
        assert_eq!(strengths.incoming.iter().sum::<u64>(), 7);
    }

    #[test]
    fn directed_degrees_count_binary_edges() {
        let edges = [
            WeightedEdge::new(0, 1, 3),
            WeightedEdge::new(0, 2, 4),
            WeightedEdge::new(1, 2, 5),
        ];

        let degrees = directed_degrees(3, &edges);

        assert_eq!(degrees.out, vec![2, 1, 0]);
        assert_eq!(degrees.incoming, vec![0, 1, 2]);
        assert_eq!(degrees.out.iter().sum::<u64>(), 3);
        assert_eq!(degrees.incoming.iter().sum::<u64>(), 3);
    }
}
