# turbosort

SIMD-accelerated radix sort for primitive types in Rust.

## Performance

Benchmarked on Intel i5-6300U (2C/4T, AVX2) with random `u32` data. `cargo bench` to reproduce.

### Serial (`turbosort::sort`)

| Size | std | turbosort | voracious | ts vs std | ts vs voracious |
|------|-----|-----------|-----------|-----------|-----------------|
| 16 | TBD | TBD | TBD | TBD | TBD |
| 1M | TBD | TBD | TBD | TBD | TBD |
| 10M | TBD | TBD | TBD | TBD | TBD |

The O(n) radix sort dominates O(n log n) comparison sort as arrays grow.
At 10M the gap narrows — both algorithms become memory-bandwidth-bound.

Small arrays (n ≤ 16) use AVX2 sorting networks for branch-free sorting.

### Parallel (`turbosort::sort_parallel`, requires `parallel` feature)

| Size | std | turbosort (serial) | turbosort (parallel) | par vs std | par vs serial |
|------|-----|--------------------|----------------------|------------|---------------|
| 1M | TBD | TBD | TBD | TBD | TBD |
| 10M | TBD | TBD | TBD | TBD | TBD |

`sort_parallel()` is a separate function — `sort()` is always single-threaded.
Parallel radix sort splits histogram and scatter phases across cores (lock-free, no atomics).

## Usage

```rust
// Sort any primitive numeric type
let mut data = vec![5u32, 3, 8, 1, 9, 2, 7, 4, 6];
turbosort::sort(&mut data);
assert_eq!(data, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

// Signed integers
let mut v = vec![3i32, -1, 4, -1, 5, -9];
turbosort::sort(&mut v);
assert_eq!(v, vec![-9, -1, -1, 3, 4, 5]);

// Floats (total order: -inf < -0.0 < +0.0 < +inf < NaN)
let mut v = vec![1.0f32, f32::NAN, -0.0, 0.0, f32::NEG_INFINITY];
turbosort::sort(&mut v);
assert_eq!(v[0], f32::NEG_INFINITY);
assert!(v[4].is_nan());

// Zero-allocation sorting with caller-provided buffer
let mut data = [42u64, 7, 99, 1, 0];
let mut buf = [0u64; 5];
turbosort::sort_with_buffer(&mut data, &mut buf);
```

## Supported Types

All 10 primitive numeric types: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `f32`, `f64`.

## Algorithm Dispatch

| Input Size | Algorithm | Complexity |
|-----------|-----------|------------|
| 0–1 | no-op | — |
| 2–16 | SIMD sorting network (AVX2, NEON ≤8) | O(n) |
| 17–512 | Quicksort with SIMD leaf nodes | O(n log n) |
| 513+ | LSD radix sort | O(n) |

With the `parallel` feature, `sort_parallel()` uses rayon for arrays >131K — see [Parallel](#parallel-turbosortsor_parallel-requires-parallel-feature) benchmarks above.

## Features

```toml
[dependencies]
turbosort = "0.1"

# For parallel sorting:
turbosort = { version = "0.1", features = ["parallel"] }
```

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Heap allocation + CPUID detection |
| `alloc` | yes | Radix sort buffer allocation |
| `parallel` | no | `sort_parallel()` via rayon |

`no_std` is supported — disable default features and use `sort_with_buffer()`.

## Architecture Support

| Arch | SIMD | Status |
|------|------|--------|
| x86_64 | AVX2 | Sorting networks, quicksort leaf acceleration |
| x86_64 | SSE4.2 | Planned |
| aarch64 | NEON | Sorting networks (4-lane) |
| Other | — | Scalar fallback (still uses radix sort) |

Runtime CPUID dispatch on x86_64 — same binary works on all CPUs.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE).
