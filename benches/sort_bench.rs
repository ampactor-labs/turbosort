use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use voracious_radix_sort::RadixSort;

fn gen_random_u32(n: usize) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
    (0..n).map(|_| rng.gen()).collect()
}

fn gen_random_u64(n: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
    (0..n).map(|_| rng.gen()).collect()
}

fn gen_random_f32(n: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
    (0..n).map(|_| rng.gen_range(-1e6f32..1e6)).collect()
}

fn gen_random_i32(n: usize) -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
    (0..n).map(|_| rng.gen()).collect()
}

fn bench_turbosort_u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbosort/u32");
    for &size in &[16, 128, 512, 4096, 65536, 1_000_000, 10_000_000] {
        let data = gen_random_u32(size);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("turbosort", size), &data, |b, data| {
            b.iter_batched_ref(
                || data.clone(),
                |d| turbosort::sort(d),
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("std_unstable", size), &data, |b, data| {
            b.iter_batched_ref(
                || data.clone(),
                |d| d.sort_unstable(),
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("voracious", size), &data, |b, data| {
            b.iter_batched_ref(
                || data.clone(),
                |d| d.voracious_sort(),
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn bench_turbosort_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbosort/u64");
    for &size in &[128, 4096, 1_000_000] {
        let data = gen_random_u64(size);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("turbosort", size), &data, |b, data| {
            b.iter_batched_ref(
                || data.clone(),
                |d| turbosort::sort(d),
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("std_unstable", size), &data, |b, data| {
            b.iter_batched_ref(
                || data.clone(),
                |d| d.sort_unstable(),
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("voracious", size), &data, |b, data| {
            b.iter_batched_ref(
                || data.clone(),
                |d| d.voracious_sort(),
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn bench_turbosort_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbosort/f32");
    for &size in &[128, 4096, 1_000_000] {
        let data = gen_random_f32(size);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("turbosort", size), &data, |b, data| {
            b.iter_batched_ref(
                || data.clone(),
                |d| turbosort::sort(d),
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("std_unstable", size), &data, |b, data| {
            b.iter_batched_ref(
                || data.clone(),
                |d| d.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap()),
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("voracious", size), &data, |b, data| {
            b.iter_batched_ref(
                || data.clone(),
                |d| d.voracious_sort(),
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn bench_turbosort_i32(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbosort/i32");
    for &size in &[128, 4096, 1_000_000] {
        let data = gen_random_i32(size);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("turbosort", size), &data, |b, data| {
            b.iter_batched_ref(
                || data.clone(),
                |d| turbosort::sort(d),
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("std_unstable", size), &data, |b, data| {
            b.iter_batched_ref(
                || data.clone(),
                |d| d.sort_unstable(),
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("voracious", size), &data, |b, data| {
            b.iter_batched_ref(
                || data.clone(),
                |d| d.voracious_sort(),
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

#[cfg(feature = "parallel")]
fn bench_parallel_u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel/u32");
    for &size in &[1_000_000, 10_000_000] {
        let data = gen_random_u32(size);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("turbosort_par", size), &data, |b, data| {
            b.iter_batched_ref(
                || data.clone(),
                |d| turbosort::sort_parallel(d),
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("turbosort_seq", size), &data, |b, data| {
            b.iter_batched_ref(
                || data.clone(),
                |d| turbosort::sort(d),
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("std_unstable", size), &data, |b, data| {
            b.iter_batched_ref(
                || data.clone(),
                |d| d.sort_unstable(),
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_turbosort_u32,
    bench_turbosort_u64,
    bench_turbosort_f32,
    bench_turbosort_i32,
);

#[cfg(feature = "parallel")]
criterion_group!(parallel_benches, bench_parallel_u32);

#[cfg(not(feature = "parallel"))]
criterion_main!(benches);

#[cfg(feature = "parallel")]
criterion_main!(benches, parallel_benches);
