# turbosort

[![Crates.io](https://img.shields.io/crates/v/turbosort.svg)](https://crates.io/crates/turbosort)
[![Documentation](https://docs.rs/turbosort/badge.svg)](https://docs.rs/turbosort)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/turbosort.svg)](#license)
[![Downloads](https://img.shields.io/crates/d/turbosort.svg)](https://crates.io/crates/turbosort)

SIMD-accelerated radix sort for primitive types in Rust.

```toml
[dependencies]
turbosort = "0.2"
```

**Status: shipping.** SSE4.2 is unimplemented, so pre-AVX2 x86 falls back to scalar, and large random `u64` loses to voracious (see Weak spots).

## Measured

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

`u64` is the honest weak spot; the mechanism and the recommendation are in Weak spots.

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

### Profiling

Where the cycles and cache misses actually go, measured with Linux `perf` on the same i7-8665U. Reproduce with:

```sh
RUSTFLAGS="-C force-frame-pointers=yes" cargo build --profile profiling --example profile
perf record --call-graph fp -F 997 -e cycles:u -- target/profiling/examples/profile
perf script --inline | inferno-collapse-perf | inferno-flamegraph > docs/flamegraph.svg
```

[![Flamegraph of the default profiling workload](https://raw.githubusercontent.com/ampactor-labs/turbosort/master/docs/flamegraph.svg)](https://github.com/ampactor-labs/turbosort/blob/master/docs/flamegraph.svg)

On the default workload (thirty 10M-element sorts each of random `u32` and `f32`), the scatter passes take 73% of cycles and the histogram scans 22%; the rest is the harness's per-iteration clone (3%) plus startup. The sorted-input check never registers, since on random data it abandons within the first few elements.

The cache story, from a `perf stat` sweep of `examples/profile` (mean of 30/10/3 back-to-back sorts per size, so absolute times differ slightly from the criterion tables above; the ratios agree). Counters cover the whole process, including data generation and the per-iteration clone, identically for both sorts. Per-key counters do not depend on clock speed, which makes them the stable metric on a 15W chip:

| Random `u32` | 1M | 10M | 100M |
|---|---|---|---|
| turbosort | 5.6 ms | 84 ms | 0.80 s |
| `std::sort_unstable` | 15.7 ms | 188 ms | 2.53 s |
| speedup | 2.8x | 2.2x | 3.2x |
| turbosort LLC misses/key | 0.32 | 0.95 | 0.99 |
| turbosort instructions/key | 53 | 55 | 61 |
| `std` instructions/key | 180 | 210 | 245 |
| turbosort IPC | 2.1 | 1.9 | 1.9 |
| `std` IPC | 2.9 | 2.8 | 2.8 |

Two regimes. At 1M keys the 8MB working set (data plus scratch) half-fits this chip's 8MB L3, so only a third of keys miss. From 10M up every key costs one LLC miss and the sort runs against memory, IPC pinned near 1.9. `std`'s pdqsort is the mirror image: compute-bound at IPC 2.8, with a per-key instruction count that grows with log n (180 to 245 from 1M to 100M) while radix stays flat near 55. That is why the gap widens to ~3x at 100M keys; rerunning the pair in the opposite order gives 2.97x, so it is not a thermal artifact. There is no large-N cliff: the ratio is at its best at the largest size tested.

Recording cache misses instead of cycles (`perf record -e cache-misses:u`, 100M keys) puts 88% of the sort's LLC misses in the scatter pass; histograms and the copy-back are sequential streams the prefetcher covers. dTLB misses stay under 0.007/key at every size with plain 4K pages, because the scatter writes 256 destination streams, each sequential, so the hot page set stays near 256 no matter how large the array grows.

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

## How it works

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

## Supported types

All 10 primitive numeric types: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `f32`, `f64`.

## Weak spots

Large random `u64`, `i64`, and `f64` are the honest loss. Random 64-bit keys force eight byte passes; above 64K elements turbosort switches to 11-bit digits (six passes, measured faster than both 8-bit and 16-bit variants), which cut 1M-element time by 1.4x over the byte path, and voracious still wins by touching memory roughly half as much with its diverting MSD strategy. At 1M `u64` it runs 0.77x against `std` and 0.67x against voracious. If your workload is dominated by large arrays of random 64-bit keys, use voracious. Keys with dead high bytes (timestamps, small IDs) are a different story: constant-digit passes are skipped outright, so real-world `u64` often runs in two or three passes.

The 1M-key numbers are cache-flattered. At that size the working set half-fits this chip's 8MB L3 and only about a third of keys miss, which is why the honest floor is the 2.2x at 10M rather than the headline. Absolute times swing with a 15W laptop's thermal state; the per-key counters are the stable metric and the run-order control is in the profiling section.

SSE4.2 is listed as planned and is not written, so x86 without AVX2 gets the scalar path with the radix sort on top. `sort()` is not stable and it is not comparison-based: it sorts primitive numeric slices only, which is why `std::sort_unstable` is the comparison target throughout.

## Verification

The unsafe core is a raw-pointer scatter into uninitialized scratch, and it runs under Miri on every push, on x86_64 and under Miri's aarch64 interpreter, so the NEON paths are checked on a machine that does not have NEON. CI also runs `cargo clippy --all-features -- -D warnings`, a `thumbv7em-none-eabihf` no-default-features build to prove the `no_std` path, an `aarch64-unknown-linux-gnu` cross build, and `cargo publish --dry-run --locked` so a packaging break fails before a release rather than during one.

Correctness beyond the unit tests: proptest generates random slices and asserts the result matches `std`'s sort, and the tiny-network paths are checked with 0-1 principle tests, which is the standard argument that a sorting network correct on every binary input is correct on every input. Every Rust block in this README is compiled by `cargo test` through a doctest include, so an example that stops compiling fails the build.

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

On x86_64, turbosort detects AVX2 at runtime via CPUID, so one binary runs on any CPU.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE).
