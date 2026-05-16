//! Lagrange multiplier fitting for fixed-strength ME models.

/// Result of iterative proportional fitting.
#[derive(Clone, Debug)]
pub struct FitResult {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
}

fn solve_binary_multiplier(target: f64, other: &[f64], skip: Option<usize>) -> f64 {
    if target <= 0.0 {
        return 0.0;
    }

    let mut high = 1.0;
    loop {
        let sum: f64 = other
            .iter()
            .enumerate()
            .filter(|(idx, _)| Some(*idx) != skip)
            .map(|(_, &value)| {
                let z = high * value;
                z / (1.0 + z)
            })
            .sum();
        if sum >= target || high > 1e150 {
            break;
        }
        high *= 2.0;
    }

    let mut low = 0.0;
    for _ in 0..100 {
        let mid = 0.5 * (low + high);
        let sum: f64 = other
            .iter()
            .enumerate()
            .filter(|(idx, _)| Some(*idx) != skip)
            .map(|(_, &value)| {
                let z = mid * value;
                z / (1.0 + z)
            })
            .sum();
        if sum < target {
            low = mid;
        } else {
            high = mid;
        }
    }

    0.5 * (low + high)
}

/// Alternating coordinate fitting for directed binary fixed-degree models.
///
/// Solves: k_out_i = sum_j p_ij and k_in_j = sum_i p_ij with
/// p_ij = x_i * y_j / (1 + x_i * y_j).
#[must_use]
pub fn balance_binary_degrees(
    degree_out: &[f64],
    degree_in: &[f64],
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> FitResult {
    let n = degree_out.len();
    let mut x = vec![1.0; n];
    let mut y = vec![1.0; n];

    for i in 0..n {
        if degree_out[i] <= 0.0 {
            x[i] = 0.0;
        }
        if degree_in[i] <= 0.0 {
            y[i] = 0.0;
        }
    }

    for iter in 0..max_iterations {
        let old_x = x.clone();
        let old_y = y.clone();

        for i in 0..n {
            x[i] =
                solve_binary_multiplier(degree_out[i], &y, if self_loops { None } else { Some(i) });
        }
        for j in 0..n {
            y[j] =
                solve_binary_multiplier(degree_in[j], &x, if self_loops { None } else { Some(j) });
        }

        let dx = x
            .iter()
            .zip(old_x.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let dy = y
            .iter()
            .zip(old_y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

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

fn binary_probability(x: f64, y: f64) -> f64 {
    let z = x * y;
    z / (1.0 + z)
}

/// Iterative proportional fitting for weighted factor constraints.
///
/// Solves excess_out_i = sum_j p_ij * a_i * b_j and excess_in_j = sum_i p_ij * a_i * b_j.
#[must_use]
pub fn balance_weighted_factors(
    excess_out: &[f64],
    excess_in: &[f64],
    degree_x: &[f64],
    degree_y: &[f64],
    self_loops: bool,
    tolerance: f64,
    max_iterations: usize,
) -> FitResult {
    let n = excess_out.len();
    let total: f64 = excess_out.iter().sum();
    let sqrt_t = total.sqrt().max(1.0);
    let mut a: Vec<f64> = excess_out
        .iter()
        .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
        .collect();
    let mut b: Vec<f64> = excess_in
        .iter()
        .map(|&s| if s > 0.0 { s / sqrt_t } else { 0.0 })
        .collect();

    for iter in 0..max_iterations {
        let mut b_new = vec![0.0; n];
        for j in 0..n {
            if excess_in[j] == 0.0 {
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&i| self_loops || i != j)
                .map(|i| binary_probability(degree_x[i], degree_y[j]) * a[i])
                .sum();
            b_new[j] = if denom > 0.0 {
                excess_in[j] / denom
            } else {
                0.0
            };
        }

        let mut a_new = vec![0.0; n];
        for i in 0..n {
            if excess_out[i] == 0.0 {
                continue;
            }
            let denom: f64 = (0..n)
                .filter(|&j| self_loops || i != j)
                .map(|j| binary_probability(degree_x[i], degree_y[j]) * b_new[j])
                .sum();
            a_new[i] = if denom > 0.0 {
                excess_out[i] / denom
            } else {
                0.0
            };
        }

        let da = a_new
            .iter()
            .zip(a.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        let db = b_new
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);

        a = a_new;
        b = b_new;

        if da < tolerance && db < tolerance {
            return FitResult {
                x: a,
                y: b,
                converged: true,
                iterations: iter + 1,
            };
        }
    }

    FitResult {
        x: a,
        y: b,
        converged: false,
        iterations: max_iterations,
    }
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
    use super::{balance_binary_degrees, balance_no_self_loops, binary_probability};

    #[test]
    fn recovers_binary_degrees() {
        let k_out = vec![0.8, 1.2, 1.0];
        let k_in = vec![1.1, 0.9, 1.0];
        let result = balance_binary_degrees(&k_out, &k_in, true, 1e-12, 50000);

        assert!(result.converged);
        for (i, &expected) in k_out.iter().enumerate() {
            let row_sum: f64 = (0..3)
                .map(|j| binary_probability(result.x[i], result.y[j]))
                .sum();
            assert!((row_sum - expected).abs() < 1e-6);
        }
        for (j, &expected) in k_in.iter().enumerate() {
            let col_sum: f64 = (0..3)
                .map(|i| binary_probability(result.x[i], result.y[j]))
                .sum();
            assert!((col_sum - expected).abs() < 1e-6);
        }
    }

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
