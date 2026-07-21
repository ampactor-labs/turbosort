# turbosort

[![Crates.io](https://img.shields.io/crates/v/turbosort.svg)](https://crates.io/crates/turbosort)
[![Documentation](https://docs.rs/turbosort/badge.svg)](https://docs.rs/turbosort)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/turbosort.svg)](#license)
[![Downloads](https://img.shields.io/crates/d/turbosort.svg)](https://crates.io/crates/turbosort)

SIMD-accelerated radix sort for primitive types in Rust.

## Performance

Benchmarked on an Intel i7-8665U (4C/8T, AVX2) with random data; one coherent criterion run at 30 samples per point, all three sorts measured back to back per size. Absolute times swing with this 15W chip's thermal state; the within-run ratios are the stable signal. Run `cargo bench` (add `--features parallel` for the parallel benchmarks) to reproduce.

### Serial: `u32`

| Size | std | turbosort | voracious | ts vs std | ts vs voracious |
|------|-----|-----------|-----------|-----------|-----------------|
| 16 | 66.6 ns | 47.1 ns | 87.3 ns | 1.42x | 1.86x |
| 128 | 899 ns | 388 ns | 999 ns | 2.32x | 2.58x |
| 512 | 4.76 µs | 3.50 µs | 7.72 µs | 1.36x | 2.20x |
| 4K | 42.8 µs | 26.7 µs | 35.4 µs | 1.61x | 1.33x |
| 64K | 1.28 ms | 496 µs | 622 µs | 2.58x | 1.25x |
| 1M | 21.8 ms | 8.84 ms | 15.98 ms | 2.47x | 1.81x |
| 10M | 249 ms | 105 ms | 162 ms | 2.38x | 1.55x |

Sizes 17–128 sort entirely in AVX2 registers through padded bitonic networks (tiers of 32/64/128), and 129–512 runs quicksort with those networks as leaves; this tier lost to `std` by ~35% in 0.1.x and now wins it by 1.4–2.3x. Above 512 the LSD radix path takes over, with per-type histograms sized to the actual pass count and a scratch buffer that is never zeroed. Already-sorted input is detected in one scan and returned untouched.

### Serial: `u64`

| Size | std | turbosort | voracious | ts vs std | ts vs voracious |
|------|-----|-----------|-----------|-----------|-----------------|
| 128 | 869 ns | 1.04 µs | 868 ns | 0.84x | 0.84x |
| 4K | 51.3 µs | 71.9 µs | 43.0 µs | 0.71x | 0.60x |
| 1M | 23.1 ms | 29.9 ms | 20.0 ms | 0.77x | 0.67x |

`u64` is the honest weak spot. Random 64-bit keys force eight byte passes; above 64K elements turbosort switches to 11-bit digits (six passes, measured faster than both 8-bit and 16-bit variants), which cut 1M-element time by 1.4x over the byte path, but voracious's diverting MSD strategy touches memory roughly half as much and still wins. If your workload is dominated by large arrays of random `u64`/`i64`/`f64`, use voracious. Keys with dead high bytes (timestamps, small IDs) are a different story: constant-digit passes are skipped outright, so real-world `u64` often runs in two or three passes.

### Serial: `f32`

| Size | std | turbosort | voracious | ts vs std | ts vs voracious |
|------|-----|-----------|-----------|-----------|-----------------|
| 128 | 1.48 µs | 485 ns | 1.40 µs | 3.05x | 2.88x |
| 4K | 81.1 µs | 42.4 µs | 63.9 µs | 1.91x | 1.51x |
| 1M | 34.8 ms | 10.8 ms | 15.5 ms | 3.23x | 1.44x |

Floats map to unsigned keys through the IEEE 754 bit trick (negative values flip entirely, positive values flip the sign bit), so they ride the same integer networks and radix passes: **3.2x over `std`** at 1M.

### Serial: `i32`

| Size | std | turbosort | voracious | ts vs std | ts vs voracious |
|------|-----|-----------|-----------|-----------|-----------------|
| 128 | 942 ns | 361 ns | 1.02 µs | 2.61x | 2.83x |
| 4K | 55.4 µs | 31.1 µs | 71.4 µs | 1.78x | 2.29x |
| 1M | 22.8 ms | 9.19 ms | 12.7 ms | 2.48x | 1.38x |

Signed integers XOR the sign bit into unsigned key order: one instruction, absorbed into the key transform.

### Parallel (`turbosort::sort_parallel`, requires `parallel` feature)

| Size | std | turbosort (serial) | turbosort (parallel) | par vs std | par vs serial |
|------|-----|--------------------|----------------------|------------|---------------|
| 1M | 23.5 ms | 8.60 ms | 8.36 ms | 2.81x | 1.03x |
| 10M | 270 ms | 116 ms | 73.4 ms | 3.67x | 1.58x |

`sort()` is always single-threaded; `sort_parallel()` is the opt-in multi-core path. It histograms every pass in one parallel scan up front (reused directly for the first scatter), then scatters chunks to disjoint destination ranges without locks or atomics. At 1M the setup cost still cancels the gain; at 10M it reaches **1.58x over serial** and **3.7x over `std`** on these four cores. The scatter is memory-bandwidth-bound, so scaling stays sub-linear. A fused variant that computed next-pass histograms during the scatter benchmarked slower, because the accumulation competed with the scatter for cache. It was dropped; the module docs record the measurement so it is not retried blindly.

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

## Supported types

All 10 primitive numeric types: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `f32`, `f64`.

## Algorithm dispatch

| Input Size | Algorithm | Complexity |
|-----------|-----------|------------|
| 0-1 | no-op | O(1) |
| 2-16 | SIMD sorting network (AVX2 and NEON) | O(n) |
| 17-128 | Padded bitonic network in registers (AVX2) | O(n log²n) |
| 129-512 | Quicksort with SIMD partition, network leaves | O(n log n) |
| 513+ | LSD radix sort | O(n) |

Details that cut across the tiers:

- Radix inputs get a one-scan sorted check first; sorted data returns in O(n).
- `u8`/`i8` above 512 elements use a plain counting sort with no scratch buffer.
- 8-byte keys (`u64`/`i64`/`f64`) at 64K+ elements use 11-bit digits (six passes) instead of eight byte passes.
- Passes where every key shares the same digit are skipped, so keys with constant high bytes pay only for the bytes that vary.
- With the `parallel` feature, `sort_parallel()` hands arrays larger than 131K to rayon.

## Features

```toml
[dependencies]
turbosort = "0.2"

# For parallel sorting:
turbosort = { version = "0.2", features = ["parallel"] }
```

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Heap allocation + CPUID detection |
| `alloc` | yes | Radix sort buffer allocation |
| `parallel` | no | `sort_parallel()` via rayon |

For `no_std`, disable default features and call `sort_with_buffer()` with your own scratch buffer.

## Architecture support

| Arch | SIMD | Status |
|------|------|--------|
| x86_64 | AVX2 | Networks to 128 elements, SIMD partition |
| x86_64 | SSE4.2 | Planned |
| aarch64 | NEON | Networks to 16 elements |
| Other | none | Scalar fallback (still uses radix sort) |

On x86_64, turbosort detects AVX2 at runtime via CPUID, so one binary runs on any CPU. The NEON paths run in CI on Apple Silicon and under Miri's aarch64 interpreter; the unsafe core (raw-pointer scatter into uninitialized scratch) is Miri-checked on every push.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE).
