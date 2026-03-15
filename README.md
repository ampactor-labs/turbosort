# turbosort

SIMD-accelerated radix sort for primitive types in pure Rust.

No FFI, no nightly, no unsafe trait implementations. Just fast sorting.

## Performance

Sorting 10M random `u32` values:

| Algorithm | Time | vs std |
|-----------|------|--------|
| `std::sort_unstable` | 301 ms | 1.0x |
| `turbosort::sort` | 182 ms | **1.7x** |

At 1M elements, turbosort is **3.1x faster** than std. The gap widens with array size
as O(n) radix sort dominates O(n log n) comparison sort.

Small arrays (n ≤ 16) use AVX2 sorting networks: **1.7x faster** than std.

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
| 2–16 | SIMD sorting network (AVX2/NEON) | O(n) |
| 17–512 | Quicksort with SIMD leaf nodes | O(n log n) |
| 513+ | LSD radix sort | O(n) |
| 131K+ | Parallel radix sort (rayon) | O(n/p) |

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
