//! Lagrange multiplier fitting for fixed-strength ME models.

/// Result of iterative proportional fitting.
#[derive(Clone, Debug)]
pub struct FitResult {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
}

/// Iterative proportional fitting for ME fixed-strength without self-loops.
///
/// Solves: s_out_i = sum_{j != i} x_i * y_j  and  s_in_j = sum_{i != j} x_i * y_j
#[must_use]
pub fn balance_no_self_loops(
    s_out: &[f64],
    s_in: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> FitResult {
    let n = s_out.len();
    let total: f64 = s_out.iter().sum();
    let sqrt_t = total.sqrt();

    let mut x: Vec<f64> = s_out
        .iter()
        .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
        .collect();
    let mut y: Vec<f64> = s_in
        .iter()
        .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
        .collect();

    for iter in 0..max_iterations {
        let sum_x: f64 = x.iter().sum();
        let mut y_new = vec![0.0; n];
        for j in 0..n {
            if s_in[j] == 0.0 {
                continue;
            }
            let denom = sum_x - x[j];
            y_new[j] = if denom > 0.0 { s_in[j] / denom } else { 0.0 };
        }

        let sum_y: f64 = y_new.iter().sum();
        let mut x_new = vec![0.0; n];
        for i in 0..n {
            if s_out[i] == 0.0 {
                continue;
            }
            let denom = sum_y - y_new[i];
            x_new[i] = if denom > 0.0 { s_out[i] / denom } else { 0.0 };
        }

        let dx = x_new
            .iter()
            .zip(x.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let dy = y_new
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        x = x_new;
        y = y_new;

        if dx < tolerance && dy < tolerance {
            return FitResult {
                x,
                y,
                converged: true,
                iterations: iter + 1,
            };
        }
    }

    FitResult {
        x,
        y,
        converged: false,
        iterations: max_iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::balance_no_self_loops;

    #[test]
    fn recovers_symmetric_strengths() {
        let s_out = vec![10.0, 20.0, 30.0];
        let s_in = vec![15.0, 25.0, 20.0];

        let result = balance_no_self_loops(&s_out, &s_in, 1e-10, 50000);

        assert!(result.converged);
        // Check: sum_{j!=i} x_i * y_j ≈ s_out_i
        for (i, &s_out_i) in s_out.iter().enumerate() {
            let row_sum: f64 = (0..3)
                .filter(|&j| j != i)
                .map(|j| result.x[i] * result.y[j])
                .sum();
            assert!(
                (row_sum - s_out_i).abs() < 1e-6,
                "s_out[{i}]: expected {s_out_i}, got {row_sum}",
            );
        }
        for (j, &s_in_j) in s_in.iter().enumerate() {
            let col_sum: f64 = (0..3)
                .filter(|&i| i != j)
                .map(|i| result.x[i] * result.y[j])
                .sum();
            assert!(
                (col_sum - s_in_j).abs() < 1e-6,
                "s_in[{j}]: expected {s_in_j}, got {col_sum}",
            );
        }
    }
}
