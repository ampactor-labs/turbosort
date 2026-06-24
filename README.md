# turbosort

SIMD-accelerated radix sort for primitive types in Rust.

## Performance

Benchmarked on an Intel i7-8665U (4C/8T, AVX2) with random data, full-sample criterion runs. Large arrays carry roughly ±15% run-to-run variance from thermal throttling on this 15W chip, so the speedup ratios are steadier than the absolute times. Run `cargo bench` (add `--features parallel` for the parallel benchmarks) to reproduce.

### Serial — `u32`

| Size | std | turbosort | voracious | ts vs std | ts vs voracious |
|------|-----|-----------|-----------|-----------|-----------------|
| 16 | 39.6 ns | 34.1 ns | 36.8 ns | 1.16x | 1.08x |
| 128 | 739 ns | 1072 ns | 695 ns | 0.69x | 0.65x |
| 512 | 3.90 µs | 5.85 µs | 7.56 µs | 0.67x | 0.77x |
| 4K | 42.0 µs | 30.5 µs | 33.2 µs | 1.38x | 1.09x |
| 64K | 1090 µs | 485 µs | 556 µs | 2.25x | 1.15x |
| 1M | 19.2 ms | 8.49 ms | 12.6 ms | 2.26x | 1.48x |
| 10M | 221 ms | 92.4 ms | 129 ms | 2.39x | 1.40x |

The algorithm dispatch table (see below) explains the mid-range dip: sizes 128–512 fall in the quicksort + SIMD-leaf tier, where `std::sort_unstable` still wins because its branch predictor warms up quickly on sorted sub-runs. Once arrays exceed ~4K elements the LSD radix pass takes over and turbosort pulls ahead, reaching **2.4x over `std`** and **1.4x over voracious** at 10M. Small arrays (n ≤ 16) sort through branch-free AVX2 sorting networks.

### Serial — `u64`

| Size | std | turbosort | voracious | ts vs std | ts vs voracious |
|------|-----|-----------|-----------|-----------|-----------------|
| 128 | 736 ns | 712 ns | 738 ns | 1.03x | 1.04x |
| 4K | 42.4 µs | 69.7 µs | 38.7 µs | 0.61x | 0.56x |
| 1M | 20.5 ms | 22.9 ms | 16.6 ms | 0.90x | 0.73x |

`u64` is a 2-pass radix sort (two 32-bit halves). At 4K–1M the double-scatter cost shows: turbosort trails `std` by ~10% and voracious (which uses a wider radix for 64-bit keys) by ~37%. If you sort large volumes of `u64`, prefer voracious or `std::sort_unstable`.

### Serial — `f32`

| Size | std | turbosort | voracious | ts vs std | ts vs voracious |
|------|-----|-----------|-----------|-----------|-----------------|
| 128 | 1150 ns | 1175 ns | 1206 ns | 0.98x | 1.03x |
| 4K | 67.5 µs | 42.6 µs | 52.7 µs | 1.58x | 1.24x |
| 1M | 28.3 ms | 9.80 ms | 12.5 ms | 2.89x | 1.27x |

Floats require a sign-flip step (IEEE 754 negative numbers sort backwards in bit-order), but turbosort absorbs that into the radix pass. The result is a strong win: **2.9x over `std`** at 1M.

### Serial — `i32`

| Size | std | turbosort | voracious | ts vs std | ts vs voracious |
|------|-----|-----------|-----------|-----------|-----------------|
| 128 | 814 ns | 1104 ns | 887 ns | 0.74x | 0.80x |
| 4K | 44.4 µs | 30.9 µs | 55.4 µs | 1.44x | 1.79x |
| 1M | 19.5 ms | 7.67 ms | 12.1 ms | 2.55x | 1.57x |

Signed integers use a bias step to remap the sign bit before radix sorting. At 1M elements turbosort reaches **2.55x over `std`** and **1.57x over voracious**.

### Parallel (`turbosort::sort_parallel`, requires `parallel` feature)

| Size | std | turbosort (serial) | turbosort (parallel) | par vs std | par vs serial |
|------|-----|--------------------|----------------------|------------|---------------|
| 1M | 18.0 ms | 7.53 ms | 7.18 ms | 2.51x | 1.05x |
| 10M | 225 ms | 87.1 ms | 71.8 ms | 3.13x | 1.21x |

`sort()` is always single-threaded; `sort_parallel()` is the opt-in multi-core path. It splits the histogram and scatter phases across cores without locks or atomics. On these four cores the gain is real but sub-linear: about break-even at 1M (setup cost cancels the work saved), rising to **1.21x over serial** at 10M, where it reaches **3.1x over `std`**. The scatter phase is bound by memory bandwidth rather than compute, so extra cores help less than they would for a CPU-heavy workload.

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
| 2-16 | SIMD sorting network (AVX2; NEON ≤8) | O(n) |
| 17-512 | Quicksort with SIMD leaf nodes | O(n log n) |
| 513+ | LSD radix sort | O(n) |

With the `parallel` feature, `sort_parallel()` hands arrays larger than 131K to rayon. See the parallel benchmarks above.

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

For `no_std`, disable default features and call `sort_with_buffer()` with your own scratch buffer.

## Architecture support

| Arch | SIMD | Status |
|------|------|--------|
| x86_64 | AVX2 | Sorting networks, quicksort leaf acceleration |
| x86_64 | SSE4.2 | Planned |
| aarch64 | NEON | Sorting networks (4-lane) |
| Other | none | Scalar fallback (still uses radix sort) |

On x86_64, turbosort detects AVX2 at runtime via CPUID, so one binary runs on any CPU.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE).
