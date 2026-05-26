//! Shared fitting support helpers.

/// Result of saturation detection: which nodes are saturated and what mask
/// and excess constraints to use for the residual sub-problem.
#[derive(Clone, Debug)]
pub(crate) struct SaturationPeeling {
    /// Dense mask: `true` = pair is fixed (excluded from the free sub-problem).
    pub mask: Vec<bool>,
    /// Excess outgoing constraint after removing known contributions.
    pub excess_out: Vec<f64>,
    /// Excess incoming constraint after removing known contributions.
    pub excess_in: Vec<f64>,
    /// Whether any saturation was detected.
    pub has_saturation: bool,
}

/// Detect degree-saturated nodes and peel them into a mask.
///
/// A node is degree-saturated when its target degree equals the number of
/// candidate partners (N with self-loops, N-1 without). For such nodes, all
/// outgoing (or incoming) edges are deterministically occupied.
///
/// `contribution_per_pair` is the known weight each saturated pair contributes
/// to the partner's constraint (1.0 for degree models, M for B strength).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub(crate) fn peel_degree_saturation(
    out_seq: &[f64],
    in_seq: &[f64],
    self_loops: bool,
    contribution_per_pair: f64,
) -> SaturationPeeling {
    let n = out_seq.len();
    let capacity = if self_loops {
        n as f64
    } else {
        (n.saturating_sub(1)) as f64
    };
    let mut mask = self_loop_mask(n, self_loops);
    let mut excess_out = out_seq.to_vec();
    let mut excess_in = in_seq.to_vec();
    let mut saturated_out = 0usize;
    let mut saturated_in = 0usize;

    // Peel out-saturated nodes (all outgoing pairs are fixed)
    for i in 0..n {
        if out_seq[i] >= capacity - 1e-9 {
            saturated_out += 1;
            for j in 0..n {
                if !mask[i * n + j] {
                    mask[i * n + j] = true;
                    excess_in[j] -= contribution_per_pair;
                }
            }
            excess_out[i] = 0.0;
        }
    }
    // Peel in-saturated nodes (all incoming pairs are fixed)
    for j in 0..n {
        if in_seq[j] >= capacity - 1e-9 {
            saturated_in += 1;
            for i in 0..n {
                if !mask[i * n + j] {
                    mask[i * n + j] = true;
                    excess_out[i] -= contribution_per_pair;
                }
            }
            excess_in[j] = 0.0;
        }
    }
    // Clamp negative excess from double-peeling
    for v in excess_out.iter_mut() {
        *v = v.max(0.0);
    }
    for v in excess_in.iter_mut() {
        *v = v.max(0.0);
    }
    // Rebalance: ensure sum(excess_out) == sum(excess_in)
    let sum_out: f64 = excess_out.iter().sum();
    let sum_in: f64 = excess_in.iter().sum();
    let diff = sum_out - sum_in;
    if diff.abs() > 1e-9 {
        if diff > 0.0 {
            // Excess out > excess in: scale down excess_out
            if sum_out > 0.0 {
                let factor = sum_in / sum_out;
                for v in excess_out.iter_mut() {
                    *v *= factor;
                }
            }
        } else {
            // Excess in > excess out: scale down excess_in
            if sum_in > 0.0 {
                let factor = sum_out / sum_in;
                for v in excess_in.iter_mut() {
                    *v *= factor;
                }
            }
        }
    }
    let has_saturation = saturated_out > 0 || saturated_in > 0;
    SaturationPeeling {
        mask,
        excess_out,
        excess_in,
        has_saturation,
    }
}

/// Detect B-strength-saturated nodes (strength = M * capacity) and peel.
///
/// Each pair from a saturated node contributes weight M to partner constraints.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub(crate) fn peel_b_strength_saturation(
    strength_out: &[f64],
    strength_in: &[f64],
    layers: u32,
    self_loops: bool,
) -> SaturationPeeling {
    let n = strength_out.len();
    let capacity = if self_loops {
        n as f64
    } else {
        (n.saturating_sub(1)) as f64
    };
    let m = f64::from(layers);
    let max_strength = m * capacity;
    let mut mask = self_loop_mask(n, self_loops);
    let mut excess_out = strength_out.to_vec();
    let mut excess_in = strength_in.to_vec();
    let mut saturated_out = 0usize;
    let mut saturated_in = 0usize;

    for i in 0..n {
        if strength_out[i] >= max_strength - 1e-9 {
            saturated_out += 1;
            for j in 0..n {
                if !mask[i * n + j] {
                    mask[i * n + j] = true;
                    excess_in[j] -= m;
                }
            }
            excess_out[i] = 0.0;
        }
    }
    for j in 0..n {
        if strength_in[j] >= max_strength - 1e-9 {
            saturated_in += 1;
            for i in 0..n {
                if !mask[i * n + j] {
                    mask[i * n + j] = true;
                    excess_out[i] -= m;
                }
            }
            excess_in[j] = 0.0;
        }
    }
    for v in excess_out.iter_mut() {
        *v = v.max(0.0);
    }
    for v in excess_in.iter_mut() {
        *v = v.max(0.0);
    }
    // Rebalance
    let sum_out: f64 = excess_out.iter().sum();
    let sum_in: f64 = excess_in.iter().sum();
    let diff = sum_out - sum_in;
    if diff.abs() > 1e-9 {
        if diff > 0.0 {
            if sum_out > 0.0 {
                let factor = sum_in / sum_out;
                for v in excess_out.iter_mut() {
                    *v *= factor;
                }
            }
        } else if sum_in > 0.0 {
            let factor = sum_out / sum_in;
            for v in excess_in.iter_mut() {
                *v *= factor;
            }
        }
    }
    let has_saturation = saturated_out > 0 || saturated_in > 0;
    SaturationPeeling {
        mask,
        excess_out,
        excess_in,
        has_saturation,
    }
}

/// Check if total edges saturate the candidate pair space.
#[must_use]
#[allow(dead_code)]
pub(crate) fn is_edge_saturated(n: usize, target_edges: f64, self_loops: bool) -> bool {
    let capacity = if self_loops {
        (n * n) as f64
    } else {
        (n * n.saturating_sub(1)) as f64
    };
    target_edges >= capacity - 1e-9
}

/// Build a dense pair mask from self-loop policy.
///
/// `true` means the pair is excluded from fitting sums.
#[must_use]
pub(crate) fn self_loop_mask(n: usize, self_loops: bool) -> Vec<bool> {
    let mut mask = vec![false; n * n];
    if !self_loops {
        for i in 0..n {
            mask[i * n + i] = true;
        }
    }
    mask
}

/// Maximum absolute coordinate change between two same-length vectors.
#[must_use]
pub(crate) fn max_abs_delta(current: &[f64], previous: &[f64]) -> f64 {
    current
        .iter()
        .zip(previous.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

/// Maximum absolute coordinate change across two fitted multiplier pairs.
#[must_use]
pub(crate) fn max_pair_delta(x: &[f64], old_x: &[f64], y: &[f64], old_y: &[f64]) -> f64 {
    max_abs_delta(x, old_x).max(max_abs_delta(y, old_y))
}

/// Euclidean distance between projected XY coordinates of nodes i and j.
#[inline]
#[must_use]
pub(crate) fn coord_distance(coord_x: &[f64], coord_y: &[f64], i: usize, j: usize) -> f64 {
    (coord_x[i] - coord_x[j]).hypot(coord_y[i] - coord_y[j])
}
