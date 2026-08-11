//! Prepared problem — validated residual constraints ready for sampling.
//!
//! [`PreparedProblem`] is the output of the preparation pipeline: raw
//! user constraints minus fixed-pair contributions, mask-filtered, and
//! validated for internal consistency.  Sampler backends receive only
//! residual problems.

use super::family::OccupationFamily;
use crate::OccNum;

/// A fully prepared residual problem for a microcanonical or canonical
/// sampling backend.
///
/// All hard constraints are residual (fixed-pair contributions already
/// subtracted).  The sampler must receive the residual problem, not the
/// raw user input.
///
/// Fields are optional because different ensembles constrain different
/// observables (e.g., fixed-(E,T) constrains edges and total occupation;
/// fixed-strength constrains strengths but not edges).
#[derive(Clone, Debug)]
pub struct PreparedProblem {
    /// Occupation family (ME, B, W).
    pub family: OccupationFamily,
    /// Number of nodes.
    pub node_count: usize,
    /// Whether self-loops are allowed on admissible pairs.
    pub self_loops: bool,
    /// Number of admissible free pairs (after mask and fixed pairs).
    pub admissible_pair_count: usize,
    /// Residual binary edge count E_res (uniform support).
    pub residual_edges: Option<usize>,
    /// Residual total occupation T_res.
    pub residual_total: Option<OccNum>,
    /// Residual out-degree sequence.
    pub residual_out_degrees: Option<Vec<u32>>,
    /// Residual in-degree sequence.
    pub residual_in_degrees: Option<Vec<u32>>,
    /// Residual out-strength sequence.
    pub residual_out_strengths: Option<Vec<OccNum>>,
    /// Residual in-strength sequence.
    pub residual_in_strengths: Option<Vec<OccNum>>,
}

impl PreparedProblem {
    /// Create a new prepared problem with the given parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        family: OccupationFamily,
        node_count: usize,
        self_loops: bool,
        admissible_pair_count: usize,
        residual_edges: Option<usize>,
        residual_total: Option<OccNum>,
        residual_out_degrees: Option<Vec<u32>>,
        residual_in_degrees: Option<Vec<u32>>,
        residual_out_strengths: Option<Vec<OccNum>>,
        residual_in_strengths: Option<Vec<OccNum>>,
    ) -> Self {
        Self {
            family,
            node_count,
            self_loops,
            admissible_pair_count,
            residual_edges,
            residual_total,
            residual_out_degrees,
            residual_in_degrees,
            residual_out_strengths,
            residual_in_strengths,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_problem() {
        let p = PreparedProblem::new(
            OccupationFamily::ME,
            0,
            false,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert_eq!(p.node_count, 0);
        assert_eq!(p.family, OccupationFamily::ME);
    }

    #[test]
    fn fixed_et_problem() {
        let p = PreparedProblem::new(
            OccupationFamily::B { layers: 3 },
            10,
            false,
            90,
            Some(15),
            Some(30),
            None,
            None,
            None,
            None,
        );
        assert_eq!(p.residual_edges, Some(15));
        assert_eq!(p.residual_total, Some(30));
    }
}
