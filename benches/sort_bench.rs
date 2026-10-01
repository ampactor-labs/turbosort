use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
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

/// Ordered and repetitive inputs, which uniform random data never produces.
/// In 0.2.1 the 129-512 quicksort went quadratic on most of them, reversed
/// input ran every radix pass, and a permutation's equal-sized buckets
/// alias in the cache.
const PATTERNS: [&str; 6] = [
    "sorted",
    "reversed",
    "few_unique",
    "pipe_organ",
    "sawtooth",
    "permutation",
];

fn gen_pattern(pattern: &str, n: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
    let n = n as u64;
    match pattern {
        "sorted" => (0..n).collect(),
        "reversed" => (0..n).rev().collect(),
        "few_unique" => (0..n).map(|i| i % 4).collect(),
        "pipe_organ" => (0..n).map(|i| if i < n / 2 { i } else { n - i }).collect(),
        "sawtooth" => (0..n).map(|i| i % (n / 8)).collect(),
        "permutation" => {
            let mut v: Vec<u64> = (0..n).collect();
            v.shuffle(&mut rng);
            v
        }
        _ => unreachable!("unknown pattern {pattern}"),
    }
}

fn bench_pair<T: turbosort::SortableKey + Ord>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    data: &[T],
) {
    let size = data.len();
    group.bench_with_input(
        BenchmarkId::new(format!("turbosort/{name}"), size),
        data,
        |b, data| {
            b.iter_batched_ref(
                || data.to_vec(),
                |d| turbosort::sort(d),
                criterion::BatchSize::LargeInput,
            )
        },
    );
    group.bench_with_input(
        BenchmarkId::new(format!("std_unstable/{name}"), size),
        data,
        |b, data| {
            b.iter_batched_ref(
                || data.to_vec(),
                |d| d.sort_unstable(),
                criterion::BatchSize::LargeInput,
            )
        },
    );
}

fn bench_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("patterns");
    for &size in &[512, 65_536] {
        group.throughput(Throughput::Elements(size as u64));
        for pattern in PATTERNS {
            let data = gen_pattern(pattern, size);
            let narrow: Vec<u32> = data.iter().map(|&x| x as u32).collect();
            bench_pair(&mut group, &format!("u32/{pattern}"), &narrow);
            bench_pair(&mut group, &format!("u64/{pattern}"), &data);
        }
    }
    group.finish();
}

/// Many distinct short arrays per iteration. The groups above sort one input
/// over and over, so the branch predictor learns it, which flatters
/// insertion sort and quicksort at these lengths; this group times a batch
/// of 16,384 random keys cut into slices of each size.
fn bench_small_batches(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_batches");
    let data = gen_random_u32(16_384);
    group.throughput(Throughput::Elements(data.len() as u64));
    for &size in &[4usize, 8, 16, 17, 32, 64, 128, 512] {
        group.bench_with_input(BenchmarkId::new("turbosort", size), &data, |b, data| {
            b.iter_batched_ref(
                || data.clone(),
                |d| d.chunks_exact_mut(size).for_each(turbosort::sort),
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("std_unstable", size), &data, |b, data| {
            b.iter_batched_ref(
                || data.clone(),
                |d| d.chunks_exact_mut(size).for_each(<[u32]>::sort_unstable),
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

/// A 32-byte record keyed by one field: what `sort_by_key` is for.
#[derive(Clone, Copy)]
struct Record<K> {
    key: K,
    // Never read: it makes the record 32 bytes.
    #[allow(dead_code)]
    payload: [u64; 3],
}

fn bench_by_key_pair<K: turbosort::SortableKey + Ord>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    data: &[Record<K>],
) {
    let size = data.len();
    group.bench_with_input(
        BenchmarkId::new(format!("turbosort/{name}"), size),
        data,
        |b, data| {
            b.iter_batched_ref(
                || data.to_vec(),
                |d| turbosort::sort_by_key(d, |r| r.key),
                criterion::BatchSize::LargeInput,
            )
        },
    );
    group.bench_with_input(
        BenchmarkId::new(format!("std_stable/{name}"), size),
        data,
        |b, data| {
            b.iter_batched_ref(
                || data.to_vec(),
                |d| d.sort_by_key(|r| r.key),
                criterion::BatchSize::LargeInput,
            )
        },
    );
}

/// Stable sorting of records by a random key, against the standard
/// library's stable `sort_by_key`.
fn bench_by_key(c: &mut Criterion) {
    let mut group = c.benchmark_group("by_key");
    for &size in &[1_000, 65_536, 1_000_000] {
        let keys = gen_random_u64(size);
        let narrow: Vec<Record<u32>> = keys
            .iter()
            .map(|&k| Record {
                key: k as u32,
                payload: [k; 3],
            })
            .collect();
        let wide: Vec<Record<u64>> = keys
            .iter()
            .map(|&k| Record {
                key: k,
                payload: [k; 3],
            })
            .collect();
        group.throughput(Throughput::Elements(size as u64));
        bench_by_key_pair(&mut group, "u32", &narrow);
        bench_by_key_pair(&mut group, "u64", &wide);
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
    bench_patterns,
    bench_small_batches,
    bench_by_key,
);

#[cfg(feature = "parallel")]
criterion_group!(parallel_benches, bench_parallel_u32);

#[cfg(not(feature = "parallel"))]
criterion_main!(benches);

#[cfg(feature = "parallel")]
criterion_main!(benches, parallel_benches);
