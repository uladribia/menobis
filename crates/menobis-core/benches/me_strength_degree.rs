//! Benchmark: ME and B strength-degree L-BFGS solver at N=100, 500, 1000.
//!
//! Run with: cargo bench --package menobis-core --bench me_strength_degree

use std::time::Instant;

use menobis_core::fitting::b_lbfgs::fit_strength_degree_binomial_lbfgs;
use menobis_core::fitting::mask::PairMask;
use menobis_core::fitting::me_lbfgs::fit_strength_degree_poisson_lbfgs;

/// Generate targets from heterogeneous multipliers for ME.
fn generate_me_targets(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Heterogeneous multipliers varying ~10x (mimics real networks)
    let x: Vec<f64> = (0..n)
        .map(|i| 0.05 + 0.95 * (i as f64 / n as f64).powf(0.7))
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|j| 0.1 + 0.9 * ((n - 1 - j) as f64 / n as f64).powf(0.8))
        .collect();
    let z: Vec<f64> = (0..n)
        .map(|i| 0.2 + 0.6 * (((i * 7) % n) as f64 / n as f64))
        .collect();
    let w: Vec<f64> = (0..n)
        .map(|j| 0.3 + 0.5 * (((j * 3) % n) as f64 / n as f64))
        .collect();

    let mask = PairMask::from_self_loops(n, false);
    let mut s_out = vec![0.0; n];
    let mut s_in = vec![0.0; n];
    let mut k_out = vec![0.0; n];
    let mut k_in = vec![0.0; n];

    for i in 0..n {
        for j in 0..n {
            if mask.is_masked(i, j) {
                continue;
            }
            let q = x[i] * y[j];
            let v = z[i] * w[j];
            let exp_q = q.exp();
            let exp_q_m1 = q.exp_m1();
            if exp_q_m1 <= 0.0 {
                continue;
            }
            let v_g = v * exp_q_m1;
            let zz = 1.0 + v_g;
            let occ = v_g / zz;
            let weight = v * q * exp_q / zz;
            k_out[i] += occ;
            k_in[j] += occ;
            s_out[i] += weight;
            s_in[j] += weight;
        }
    }

    (s_out, s_in, k_out, k_in)
}

fn bench_me(n: usize) {
    let (s_out, s_in, k_out, k_in) = generate_me_targets(n);
    let mask = PairMask::from_self_loops(n, false);

    // Warmup
    let _ = fit_strength_degree_poisson_lbfgs(&s_out, &s_in, &k_out, &k_in, &mask, 1e-6, 5000);

    // Timed runs
    let n_runs = 5;
    let mut times = Vec::with_capacity(n_runs);
    let mut iters_total = 0;
    let mut converged_count = 0;

    for _ in 0..n_runs {
        let start = Instant::now();
        let result =
            fit_strength_degree_poisson_lbfgs(&s_out, &s_in, &k_out, &k_in, &mask, 1e-6, 5000);
        let elapsed = start.elapsed();
        times.push(elapsed.as_secs_f64());
        iters_total += result.iterations;
        if result.converged {
            converged_count += 1;
        }
    }

    let mean_time = times.iter().sum::<f64>() / n_runs as f64;
    let min_time = times.iter().copied().fold(f64::INFINITY, f64::min);
    let max_time = times.iter().copied().fold(0.0_f64, f64::max);
    let avg_iters = iters_total as f64 / n_runs as f64;

    println!("ME  N={n:>5} | mean={mean_time:.4}s  min={min_time:.4}s  max={max_time:.4}s | iters={avg_iters:.0} | converged={converged_count}/{n_runs}");
}

/// Generate targets from heterogeneous multipliers for B(M).
fn generate_b_targets(n: usize, m: u32) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n)
        .map(|i| 0.05 + 0.95 * (i as f64 / n as f64).powf(0.7))
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|j| 0.1 + 0.9 * ((n - 1 - j) as f64 / n as f64).powf(0.8))
        .collect();
    let z: Vec<f64> = (0..n)
        .map(|i| 0.2 + 0.6 * (((i * 7) % n) as f64 / n as f64))
        .collect();
    let w: Vec<f64> = (0..n)
        .map(|j| 0.3 + 0.5 * (((j * 3) % n) as f64 / n as f64))
        .collect();

    let mask = PairMask::from_self_loops(n, false);
    let mut s_out = vec![0.0; n];
    let mut s_in = vec![0.0; n];
    let mut k_out = vec![0.0; n];
    let mut k_in = vec![0.0; n];
    let m_f = f64::from(m);

    for i in 0..n {
        for j in 0..n {
            if mask.is_masked(i, j) {
                continue;
            }
            let q = x[i] * y[j];
            let v = z[i] * w[j];
            let one_plus_q = 1.0 + q;
            let g = one_plus_q.powi(m as i32) - 1.0;
            if g <= 0.0 {
                continue;
            }
            let vg = v * g;
            let zz = 1.0 + vg;
            let occ = vg / zz;
            let weight = v * m_f * q * one_plus_q.powi(m as i32 - 1) / zz;
            k_out[i] += occ;
            k_in[j] += occ;
            s_out[i] += weight;
            s_in[j] += weight;
        }
    }

    (s_out, s_in, k_out, k_in)
}

fn bench_b(n: usize, m: u32) {
    let (s_out, s_in, k_out, k_in) = generate_b_targets(n, m);
    let mask = PairMask::from_self_loops(n, false);

    // Warmup
    let _ = fit_strength_degree_binomial_lbfgs(&s_out, &s_in, &k_out, &k_in, m, &mask, 1e-6, 5000);

    let n_runs = 5;
    let mut times = Vec::with_capacity(n_runs);
    let mut iters_total = 0;
    let mut converged_count = 0;

    for _ in 0..n_runs {
        let start = Instant::now();
        let result =
            fit_strength_degree_binomial_lbfgs(&s_out, &s_in, &k_out, &k_in, m, &mask, 1e-6, 5000);
        let elapsed = start.elapsed();
        times.push(elapsed.as_secs_f64());
        iters_total += result.iterations;
        if result.converged {
            converged_count += 1;
        }
    }

    let mean_time = times.iter().sum::<f64>() / n_runs as f64;
    let min_time = times.iter().copied().fold(f64::INFINITY, f64::min);
    let max_time = times.iter().copied().fold(0.0_f64, f64::max);
    let avg_iters = iters_total as f64 / n_runs as f64;

    println!("B{m:<2} N={n:>5} | mean={mean_time:.4}s  min={min_time:.4}s  max={max_time:.4}s | iters={avg_iters:.0} | converged={converged_count}/{n_runs}");
}

fn main() {
    println!("ME & B Strength-Degree L-BFGS Benchmark (no self-loops, tol=1e-6)");
    println!("=================================================================");
    println!();
    println!("--- ME (Poisson) ---");
    bench_me(100);
    bench_me(500);
    bench_me(1000);
    println!();
    println!("--- B (Binomial, M=3) ---");
    bench_b(100, 3);
    bench_b(500, 3);
    bench_b(1000, 3);
}
