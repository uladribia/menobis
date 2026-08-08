//! Criterion benchmarks for the fixed-total pair-Gibbs chain.
//!
//! Measures:
//! - `initialize_balanced` (ME / B / W)
//! - `sample_split` (ME / B / W) at various sums
//! - chain sweep throughput at E = 10, 100, 1000
//! - end-to-end sample (burn-in + thin) at E = 100

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::SeedableRng;

use menobis_core::generation::microcanonical::conditional::fixed_total::chain::sample_fixed_total;
use menobis_core::generation::microcanonical::conditional::fixed_total::initializer::initialize_balanced;
use menobis_core::generation::microcanonical::conditional::fixed_total::pair_conditional::sample_split;
use menobis_core::generation::microcanonical::conditional::fixed_total::state::FixedTotalState;
use menobis_core::generation::microcanonical::conditional::fixed_total::FixedTotalChain;
use menobis_core::generation::microcanonical::mcmc::McmcConfig;
use menobis_core::model::family::OccupationFamily;
use menobis_core::OccNum;

// ---------------------------------------------------------------------------
// Helper: standard MCMC config (quick burn-in, 1 sample)
// ---------------------------------------------------------------------------
fn short_config(seed: u64) -> McmcConfig {
    McmcConfig {
        burn_in_sweeps: 5,
        sweeps_per_sample: 3,
        proposals_per_sweep: None,
        seed,
    }
}

fn chain_for_e(family: OccupationFamily, e: usize, t: OccNum, seed: u64) -> FixedTotalChain {
    let occ = initialize_balanced(family, e, t, &mut StdRng::seed_from_u64(seed)).unwrap();
    FixedTotalChain::new(FixedTotalState::new(occ), family, short_config(seed))
}

// ---------------------------------------------------------------------------
// Sample split benchmarks (microbenchmarks of the core conditional)
// ---------------------------------------------------------------------------
fn bench_sample_split(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_split");

    for (label, family, q) in [
        ("ME/q=10", OccupationFamily::ME, 10u64),
        ("ME/q=100", OccupationFamily::ME, 100),
        ("ME/q=1000", OccupationFamily::ME, 1000),
        ("B(4)/q=6", OccupationFamily::B { layers: 4 }, 6),
        ("W(2)/q=10", OccupationFamily::W { layers: 2 }, 10),
        ("W(2)/q=100", OccupationFamily::W { layers: 2 }, 100),
        ("W(5)/q=10", OccupationFamily::W { layers: 5 }, 10),
        ("W(5)/q=100", OccupationFamily::W { layers: 5 }, 100),
    ] {
        group.bench_with_input(label, &(family, q), |b, &(fam, qv)| {
            let mut rng = StdRng::seed_from_u64(42);
            b.iter(|| {
                let (a, b) = sample_split(black_box(fam), black_box(qv), &mut rng);
                black_box((a, b))
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Initialization benchmarks
// ---------------------------------------------------------------------------
fn bench_initialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("initialize_balanced");

    for (label, family, e, t) in [
        ("ME/E=10/T=50", OccupationFamily::ME, 10usize, 50u64),
        ("ME/E=100/T=500", OccupationFamily::ME, 100, 500),
        ("ME/E=1000/T=5000", OccupationFamily::ME, 1000, 5000),
        ("B(4)/E=10/T=20", OccupationFamily::B { layers: 4 }, 10, 20),
        (
            "B(4)/E=100/T=200",
            OccupationFamily::B { layers: 4 },
            100,
            200,
        ),
        ("W(2)/E=10/T=50", OccupationFamily::W { layers: 2 }, 10, 50),
        (
            "W(2)/E=100/T=500",
            OccupationFamily::W { layers: 2 },
            100,
            500,
        ),
    ] {
        group.bench_with_input(label, &(family, e, t), |b, &(fam, e_cnt, t_sum)| {
            b.iter_batched(
                || StdRng::seed_from_u64(42),
                |mut rng| {
                    let occ = initialize_balanced(
                        black_box(fam),
                        black_box(e_cnt),
                        black_box(t_sum),
                        &mut rng,
                    );
                    black_box(occ.unwrap())
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Step throughput (sweep of `e` pair-Gibbs updates)
// ---------------------------------------------------------------------------
fn bench_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("sweep_throughput");

    // ME at various scales
    for (label, family, e, t) in [
        ("ME/E=10/T=50", OccupationFamily::ME, 10usize, 50u64),
        ("ME/E=100/T=500", OccupationFamily::ME, 100, 500),
        ("ME/E=1000/T=5000", OccupationFamily::ME, 1000, 5000),
        (
            "B(4)/E=100/T=300",
            OccupationFamily::B { layers: 4 },
            100,
            300,
        ),
        (
            "W(2)/E=100/T=500",
            OccupationFamily::W { layers: 2 },
            100,
            500,
        ),
    ] {
        group.throughput(Throughput::Elements(e as u64));
        group.bench_with_input(label, &(family, e, t), |b, &(fam, e_cnt, t_sum)| {
            b.iter_batched(
                || chain_for_e(fam, e_cnt, t_sum, 42),
                |mut chain| {
                    let mut rng = StdRng::seed_from_u64(99);
                    chain.sweep(&mut rng);
                    black_box(chain.state.total())
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// End-to-end: initialize + burn-in + sample
// ---------------------------------------------------------------------------
fn bench_e2e(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_sample");

    for (label, family, e, t) in [
        ("ME/E=100/T=500", OccupationFamily::ME, 100usize, 500u64),
        (
            "B(4)/E=100/T=300",
            OccupationFamily::B { layers: 4 },
            100,
            300,
        ),
        (
            "W(2)/E=100/T=500",
            OccupationFamily::W { layers: 2 },
            100,
            500,
        ),
    ] {
        group.bench_with_input(label, &(family, e, t), |b, &(fam, e_cnt, t_sum)| {
            b.iter_batched(
                || short_config(42),
                |cfg| {
                    let result = sample_fixed_total(
                        black_box(fam),
                        black_box(e_cnt),
                        black_box(t_sum),
                        black_box(&cfg),
                    );
                    black_box(result.unwrap())
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(30)
}

criterion_group!(
    name = fixed_total;
    config = bench_config();
    targets = bench_sample_split, bench_initialize, bench_sweep, bench_e2e
);
criterion_main!(fixed_total);
