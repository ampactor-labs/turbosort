// Phase 2: benchmark harness placeholder
use criterion::criterion_main;

mod benchmarks {
    use criterion::{criterion_group, BenchmarkId, Criterion};

    fn sort_benchmarks(_c: &mut Criterion) {
        // Populated in Phase 2
    }

    criterion_group!(benches, sort_benchmarks);
}

criterion_main!(benchmarks::benches);
