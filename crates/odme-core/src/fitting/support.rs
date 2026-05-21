//! Shared fitting support helpers.

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
