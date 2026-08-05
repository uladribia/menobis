//! Directed degree-sequence validation and graphicality.
//!
//! MENoBiS models directed graphs: every pair \((i,j)\) is an ordered pair,
//! and degree constraints come as separate out- and in-degree vectors.
//!
//! # Graphicality
//!
//! We use the **Fulkerson–Chen–Anstee** theorem (equivalently the bipartite
//! Gale–Ryser theorem for 0–1 matrices) to check whether a pair of out- and
//! in-degree sequences can be realized as a simple directed graph without
//! self-loops.
//!
//! A necessary and sufficient condition for a simple directed graph
//! (no parallel arcs, no self-loops) with out-degrees \(d^+_i\) and
//! in-degrees \(d^-_i\) is:
//!
//! 1. \(\sum_i d^+_i = \sum_i d^-_i\).
//! 2. For every subset \(S \subseteq \{1,\dots,N\}\):
//!
//!    \[
//!    \sum_{i\in S} d^+_i \le
//!    \sum_{j=1}^N \min(d^-_j, |S|) - \max_{j\notin S}(0, d^-_j - (N - |S|))
//!    \]
//!
//!    (simplified: \(\sum_{i\in S} d^+_i \le \sum_{j=1}^N \min(d^-_j, |S|)\)).
//!
//! In practice we implement the constructive greedy test (sufficient for our
//! generator) rather than the full subset enumeration, since the constructor
//! itself will detect failure for non-graphical sequences.

use super::errors::FixedKTError;

/// A validated directed degree-sequence pair.
#[derive(Clone, Debug)]
pub struct DirectedDegreeSequence {
    pub out_degrees: Vec<u32>,
    pub in_degrees: Vec<u32>,
    pub edge_count: usize,
    pub max_out_degree: u32,
    pub max_in_degree: u32,
}

impl DirectedDegreeSequence {
    /// Validate and build a `DirectedDegreeSequence` from raw out/in vectors.
    ///
    /// Checks:
    /// - lengths match
    /// - all degrees non-negative (enforced by u32)
    /// - sum(out) == sum(in)
    /// - no out-degree or in-degree exceeds N-1 (loopless constraint)
    /// - the sequence pair is graphical (constructive check)
    pub fn new(
        out_degrees: Vec<u32>,
        in_degrees: Vec<u32>,
        self_loops: bool,
    ) -> Result<Self, FixedKTError> {
        let n = out_degrees.len();
        if in_degrees.len() != n {
            return Err(FixedKTError::InvalidResidual(format!(
                "out_degrees length ({}) != in_degrees length ({})",
                n,
                in_degrees.len()
            )));
        }
        if n == 0 {
            return Err(FixedKTError::InvalidResidual(
                "empty degree sequence".into(),
            ));
        }

        let max_allowed = if self_loops { n as u32 } else { n as u32 - 1 };

        let sum_out: u64 = out_degrees.iter().map(|&d| d as u64).sum();
        let sum_in: u64 = in_degrees.iter().map(|&d| d as u64).sum();

        if sum_out != sum_in {
            return Err(FixedKTError::InvalidResidual(format!(
                "out-degree sum ({sum_out}) != in-degree sum ({sum_in})"
            )));
        }
        let edge_count = sum_out as usize;

        let max_out = *out_degrees.iter().max().unwrap_or(&0);
        let max_in = *in_degrees.iter().max().unwrap_or(&0);

        // Check per-node bounds
        for (i, &d) in out_degrees.iter().enumerate() {
            if d > max_allowed {
                return Err(FixedKTError::InvalidResidual(format!(
                    "out_degree[{i}] = {d} exceeds max allowed {max_allowed}"
                )));
            }
        }
        for (i, &d) in in_degrees.iter().enumerate() {
            if d > max_allowed {
                return Err(FixedKTError::InvalidResidual(format!(
                    "in_degree[{i}] = {d} exceeds max allowed {max_allowed}"
                )));
            }
        }

        // Quick check: degree sum zero consistency
        if edge_count == 0 {
            // All degrees must be zero
            let all_zero = out_degrees.iter().all(|&d| d == 0)
                && in_degrees.iter().all(|&d| d == 0);
            if !all_zero {
                return Err(FixedKTError::InvalidResidual(
                    "zero edge count but non-zero degrees".into(),
                ));
            }
            return Ok(Self {
                out_degrees,
                in_degrees,
                edge_count: 0,
                max_out_degree: 0,
                max_in_degree: 0,
            });
        }

        // Constructive graphicality check: attempt to build a support.
        // If the greedy constructor can build one, the sequence is graphical.
        // (This is sufficient — if it fails, the sequence may still be
        // graphical via a different construction, but we treat it as
        // infeasible for this implementation.)
        match check_directed_graphicality(&out_degrees, &in_degrees, self_loops) {
            Ok(()) => {}
            Err(e) => return Err(FixedKTError::InvalidResidual(e)),
        }

        Ok(Self {
            out_degrees,
            in_degrees,
            edge_count,
            max_out_degree: max_out,
            max_in_degree: max_in,
        })
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.out_degrees.len()
    }
}

/// Quick constructive check for directed graphicality.
///
/// Tries the greedy initializer (with randomized fallback). If it can build
/// a valid support, the sequence is declared graphical.
fn check_directed_graphicality(
    out_degrees: &[u32],
    in_degrees: &[u32],
    self_loops: bool,
) -> Result<(), String> {
    // Delegate to the initializer (which includes randomized fallback).
    // If the initializer can construct a support, the sequence is graphical.
    match super::initializer::greedy_directed_initialize(out_degrees, in_degrees, self_loops) {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("graphicality check failed: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_directed_cycle() {
        // Out: each node has out-degree 1, in-degree 1 → directed cycle.
        let out = vec![1u32, 1, 1, 1];
        let inp = vec![1u32, 1, 1, 1];
        let seq = DirectedDegreeSequence::new(out, inp, false).unwrap();
        assert_eq!(seq.edge_count, 4);
    }

    #[test]
    fn valid_out_star() {
        // Node 0 connects to all others.
        let n = 5;
        let mut out = vec![0u32; n];
        out[0] = (n - 1) as u32;
        let mut inp = vec![0u32; n];
        for i in 1..n {
            inp[i] = 1;
        }
        let seq = DirectedDegreeSequence::new(out, inp, false).unwrap();
        assert_eq!(seq.edge_count, (n - 1) as usize);
    }

    #[test]
    fn sum_mismatch_rejected() {
        let out = vec![2u32, 0];
        let inp = vec![1u32, 0];
        assert!(DirectedDegreeSequence::new(out, inp, false).is_err());
    }

    #[test]
    fn degree_exceeds_n_rejected() {
        let out = vec![10u32, 0];
        let inp = vec![5u32, 5];
        assert!(DirectedDegreeSequence::new(out, inp, false).is_err());
    }

    #[test]
    fn empty_degree_zero_ok() {
        let out = vec![0u32, 0, 0];
        let inp = vec![0u32, 0, 0];
        let seq = DirectedDegreeSequence::new(out, inp, false).unwrap();
        assert_eq!(seq.edge_count, 0);
    }

    #[test]
    fn complete_directed() {
        // Every ordered pair (i,j) with i!=j is an edge.
        let n = 4;
        let out = vec![(n - 1) as u32; n];
        let inp = vec![(n - 1) as u32; n];
        let seq = DirectedDegreeSequence::new(out, inp, false).unwrap();
        assert_eq!(seq.edge_count, n * (n - 1));
    }

    #[test]
    fn len_mismatch_rejected() {
        let out = vec![1u32, 2];
        let inp = vec![1u32];
        assert!(DirectedDegreeSequence::new(out, inp, false).is_err());
    }
}