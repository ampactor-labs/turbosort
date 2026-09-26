# turbosort

[![CI](https://img.shields.io/github/actions/workflow/status/ampactor-labs/turbosort/ci.yml?branch=master&label=ci)](https://github.com/ampactor-labs/turbosort/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/turbosort.svg)](https://crates.io/crates/turbosort)
[![Documentation](https://docs.rs/turbosort/badge.svg)](https://docs.rs/turbosort)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/turbosort.svg)](#license)

A Rust library that sorts slices of primitive numbers with SIMD sorting networks and radix sort. Short slices of 4-byte keys go through AVX2 or NEON sorting networks, and slices over 512 elements use least-significant-digit radix sort, with an optional parallel path on the rayon thread pool. On the author's Intel i7-8665U laptop it sorts 10M random `u32` 2.4x faster than `std`'s `sort_unstable`, but random 64-bit keys sort slower than `std`. It also runs in `no_std` with a caller-provided buffer.

**Status: shipping.** The SSE4.2 path is not written, so x86 CPUs without AVX2 get no SIMD code, and random 64-bit keys sort slower than `std` at every size measured.

Package: https://crates.io/crates/turbosort · Docs: https://docs.rs/turbosort

## Usage

Add the crate to `Cargo.toml`:

```toml
[dependencies]
turbosort = "0.2"
```

`turbosort::sort` sorts a slice of any supported type in ascending order:

```rust
let mut data = vec![5u32, 3, 8, 1, 9, 2, 7, 4, 6];
turbosort::sort(&mut data);
assert_eq!(data, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

// Signed integers
let mut v = vec![3i32, -1, 4, -1, 5, -9];
turbosort::sort(&mut v);
assert_eq!(v, vec![-9, -1, -1, 3, 4, 5]);

// Floats use the IEEE 754 total order: -0.0 before +0.0, f32::NAN last
let mut v = vec![1.0f32, f32::NAN, -0.0, 0.0, f32::NEG_INFINITY];
turbosort::sort(&mut v);
assert_eq!(v[0], f32::NEG_INFINITY);
assert_eq!(v[1].to_bits(), (-0.0f32).to_bits());
assert!(v[4].is_nan());

// No allocation: pass a scratch buffer at least as long as the slice
let mut data = [42u64, 7, 99, 1, 0];
let mut buf = [0u64; 5];
turbosort::sort_with_buffer(&mut data, &mut buf);
assert_eq!(data, [0, 1, 7, 42, 99]);
```

| Function                               | What it does                                                                                                   |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `sort(&mut [T])`                       | Sorts in place on one thread. Above 512 elements it may allocate a scratch buffer the size of the slice.       |
| `sort_with_buffer(&mut [T], &mut [T])` | The same, with your scratch buffer. It never allocates, and it panics if the buffer is shorter than the slice. |
| `sort_parallel(&mut [T])`              | Radix sort on rayon threads for 131,072 elements or more; shorter slices get the serial radix sort. Needs the `parallel` feature. |

`T` is one of the ten primitive number types: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `f32` and `f64`. They implement the sealed `SortableKey` trait, which no other type can implement.

### Cargo features

| Feature    | Default            | What it enables                                             |
| ---------- | ------------------ | ----------------------------------------------------------- |
| `std`      | yes                | Heap allocation and run-time AVX2 detection                 |
| `alloc`    | yes, through `std` | The scratch buffer that `sort` allocates above 512 elements |
| `parallel` | no                 | `sort_parallel`, through rayon                              |

```toml
turbosort = { version = "0.2", features = ["parallel"] }
```

For `no_std`, turn off default features and call `sort_with_buffer` with your own buffer. Without `std` there is no run-time CPU detection, so x86 builds get no SIMD code. Without `alloc`, `sort` falls back to quicksort above 512 elements.

## How it works

Every supported type maps to an unsigned integer key with the same order. Unsigned integers are their own key, and signed integers flip the sign bit. Floats use the IEEE 754 bit trick: a positive value flips its sign bit and a negative value flips every bit. The sort works on keys and converts back, so all ten types share one code path per key width.

The slice length then picks the algorithm. A sorting network is a fixed sequence of compare-and-swap steps with no branches. It suits SIMD (single instruction, multiple data) registers, which compare 4 or 8 values in one instruction.

| Length     | Algorithm                                                                                         |
| ---------- | ------------------------------------------------------------------------------------------------- |
| 0 to 1     | Nothing to do                                                                                     |
| 2 to 16    | Sorting network in AVX2 or NEON registers for 4-byte keys, insertion sort otherwise               |
| 17 to 128  | Bitonic sorting network in AVX2 registers, padded to 32, 64 or 128 elements, for 4-byte keys      |
| 129 to 512 | Quicksort with a vectorized AVX2 partition and the 128-element network as its leaf, 4-byte keys   |
| 17 to 512  | Scalar quicksort, when AVX2 is missing or the key is not 4 bytes                                  |
| 513 and up | LSD radix sort                                                                                    |

LSD (least-significant-digit) radix sort makes one pass per digit of the key, lowest digit first. Each pass counts the keys per digit value (the histogram), turns the counts into output offsets and scatters every key to its slot. A pass keeps keys with equal digits in their current order, so the slice is sorted after the last pass. The cost grows linearly with the length. The radix passes are scalar code: SIMD only runs on slices of 512 elements or fewer.

- One scan checks for already-sorted input and returns early.
- `u8` and `i8` use a counting sort with no scratch buffer.
- One scan builds the histograms for every pass, sized to the key's pass count. From 8,192 elements the scan keeps four interleaved copies, so neighbouring keys with the same digit do not wait on the same counter.
- A pass where every key has the same digit is skipped, so keys with constant high bytes pay only for the bytes that vary.
- The scratch buffer is never zeroed, because every scatter pass writes every position.
- 8-byte keys (`u64`, `i64`, `f64`) at 65,536 elements or more use 11-bit digits: six passes instead of eight. That made 1M-element `u64` 1.4x faster than the byte path. 13-bit and 16-bit digits were also measured and lost; the 16-bit offset tables fall out of L1 cache and lose by more than 2x at 1M elements.

### Parallel sort

`sort` is always single-threaded, and `sort_parallel` is the opt-in multi-core path. It builds every pass's histogram in one parallel scan and reuses the per-chunk counts for the first scatter. Each thread then scatters its chunk into disjoint destination ranges, with no locks or atomics. Later passes recount each chunk, because the previous scatter moved keys across chunk boundaries. A fused variant that built the next pass's histogram during the scatter measured slower, because the counting competed with the scatter for cache. The module docs in `src/parallel.rs` record that result.

### Architecture support

| Arch    | SIMD   | What it covers                                                       |
| ------- | ------ | -------------------------------------------------------------------- |
| x86_64  | AVX2   | Networks up to 128 elements and the quicksort partition, 4-byte keys |
| x86_64  | SSE4.2 | Planned, with no code yet                                            |
| aarch64 | NEON   | Networks up to 16 elements, 4-byte keys                              |
| Other   | none   | Scalar insertion sort and quicksort up to 512 elements, radix above  |

On x86_64 the crate checks for AVX2 at run time (the CPUID instruction, through `is_x86_feature_detected!`), so one binary runs on any x86_64 CPU. Every aarch64 CPU has NEON, so there is no check.

## Benchmarks

The author ran these on an Intel i7-8665U laptop (4 cores, 8 threads, AVX2, 8 MB L3, 15 W) for the 0.2.0 release. The sort code has not changed since, and the numbers have not been re-run. The data is random values from a seeded generator (`benches/sort_bench.rs`); `f32` values are uniform between -1e6 and 1e6. One run of criterion (a Rust benchmarking library) took 30 samples per point and measured the three sorts back to back at each size. The baselines are `std`'s `sort_unstable` (`sort_unstable_by` with `partial_cmp` for `f32`) and `voracious_radix_sort` 1.2, another Rust radix sort. An unstable sort is the fair baseline: equal keys are bit-identical values, so the order of equal elements cannot be observed. Absolute times move with the laptop's thermal state, so compare the ratios within the run. To reproduce:

```sh
cargo bench                      # u32, u64, f32 and i32
cargo bench --features parallel  # adds the parallel u32 group
```

A ratio above 1 means turbosort is faster.

| Type  | Size | `std`   | turbosort | voracious | vs `std` | vs voracious |
| ----- | ---- | ------- | --------- | --------- | -------- | ------------ |
| `u32` | 16   | 66.6 ns | 47.1 ns   | 87.3 ns   | 1.42x    | 1.86x        |
| `u32` | 128  | 899 ns  | 388 ns    | 999 ns    | 2.32x    | 2.58x        |
| `u32` | 512  | 4.76 µs | 3.50 µs   | 7.72 µs   | 1.36x    | 2.20x        |
| `u32` | 4K   | 42.8 µs | 26.7 µs   | 35.4 µs   | 1.61x    | 1.33x        |
| `u32` | 64K  | 1.28 ms | 496 µs    | 622 µs    | 2.58x    | 1.25x        |
| `u32` | 1M   | 21.8 ms | 8.84 ms   | 15.98 ms  | 2.47x    | 1.81x        |
| `u32` | 10M  | 249 ms  | 105 ms    | 162 ms    | 2.38x    | 1.55x        |
| `u64` | 128  | 869 ns  | 1.04 µs   | 868 ns    | 0.84x    | 0.84x        |
| `u64` | 4K   | 51.3 µs | 71.9 µs   | 43.0 µs   | 0.71x    | 0.60x        |
| `u64` | 1M   | 23.1 ms | 29.9 ms   | 20.0 ms   | 0.77x    | 0.67x        |
| `f32` | 128  | 1.48 µs | 485 ns    | 1.40 µs   | 3.05x    | 2.88x        |
| `f32` | 4K   | 81.1 µs | 42.4 µs   | 63.9 µs   | 1.91x    | 1.51x        |
| `f32` | 1M   | 34.8 ms | 10.8 ms   | 15.5 ms   | 3.23x    | 1.44x        |
| `i32` | 128  | 942 ns  | 361 ns    | 1.02 µs   | 2.61x    | 2.83x        |
| `i32` | 4K   | 55.4 µs | 31.1 µs   | 71.4 µs   | 1.78x    | 2.29x        |
| `i32` | 1M   | 22.8 ms | 9.19 ms   | 12.7 ms   | 2.48x    | 1.38x        |

| Parallel `u32` | `std`   | `sort`  | `sort_parallel` | vs `std` | vs `sort` |
| -------------- | ------- | ------- | --------------- | -------- | --------- |
| 1M             | 23.5 ms | 8.60 ms | 8.36 ms         | 2.81x    | 1.03x     |
| 10M            | 270 ms  | 116 ms  | 73.4 ms         | 3.67x    | 1.58x     |

`u64` loses to both baselines at every size measured, and Limitations explains why. Below 350 elements voracious calls `sort_unstable` for `u64`, so those two columns match at 128. In 0.1.x the 128- and 512-element `u32` sizes ran at 0.69x and 0.67x of `std`, and the AVX2 networks brought them to 2.3x and 1.4x. The parallel sort gains only 1.03x at 1M, where its setup cost cancels most of the gain, and reaches 1.58x over the serial sort at 10M. The scatter is limited by memory bandwidth, so the gain stays well below the core count.

### Profiling

Linux `perf` on the same laptop puts 73% of cycles in the scatter passes and 22% in the histogram scans. The radix sort stays near 55 instructions per key from 1M to 100M keys, while `std` grows from 180 to 245, so the gap widens to 3.2x at 100M random `u32`. [docs/PROFILING.md](docs/PROFILING.md) has the flamegraph, the cache-miss counts and the commands.

## Testing

```sh
cargo test --all-features
```

On x86_64 this runs 24 unit tests, 54 integration tests in `tests/correctness.rs` and 7 doctests. One doctest compiles the Rust example in this README through `include_str!` in `src/lib.rs`. The integration tests take several minutes in a debug build (451 s on a shared 4-core container), mostly in the property tests.

- Property tests (proptest) sort 2,000 random slices per type in each of three length bands (0 to 16, 17 to 512, 513 to 10,000) for all ten types, and compare with `std`'s `sort_by` on the same keys.
- Exhaustive tests sort every permutation of up to 8 distinct `u8` values and up to 6 `i8` values.
- Other tests cover lengths at the tier boundaries, float and signed extremes, and adversarial `u32` patterns (sorted, reversed, pipe organ, sawtooth, few unique, median-of-three killer) up to 200,000 elements.
- The sorting networks get 0-1 principle tests: a network that sorts every input of 0s and 1s sorts every input. The NEON tests try every 0-1 input up to 16 elements, and the AVX2 tests sample 500 random 0-1 inputs at each of ten lengths from 17 to 128.
- Unit tests call each radix core directly at small sizes, so Miri can run them.
- `sort_parallel` tests use 300,000 elements: random `u32`, `u64` and `f32`, few unique values, sorted and reversed input, and `u8`.

CI (`.github/workflows/ci.yml`) runs on every push to master and every pull request:

- Format, clippy with `-D warnings`, build and tests on Ubuntu x86_64 and on macOS Apple Silicon, where the NEON paths run natively.
- Miri, an interpreter that detects undefined behaviour in unsafe code. It runs the unit and boundary tests on x86_64, and the unit tests for the aarch64 target, which checks the NEON code on an x86 runner.
- A `no_std` build for `thumbv7em-none-eabihf`, a host build with `alloc` only, and clippy for `thumbv7em-none-eabihf` and `aarch64-unknown-linux-gnu`.
- A build with the minimum supported Rust version (1.75), `cargo publish --dry-run --locked`, and a docs build with the docs.rs settings.

Gaps: SIMD dispatch is off under Miri on x86_64, so Miri never runs the AVX2 code. The AVX2 unit tests return early, and pass, on a CPU without AVX2. The exhaustive NEON 0-1 test for 11 to 16 elements is skipped under Miri. The `no_std` targets are built but never run, and CI does not run the benchmarks.

## Limitations

turbosort only sorts slices of the ten primitive number types in ascending order. It has no comparator or key-function API, so it cannot sort structs by a field, and it does not support `usize`, `isize`, `u128` or `i128`. Its weak case is random 64-bit keys: random `u64` ran at 0.71x to 0.84x the speed of `std`'s `sort_unstable` at every size measured, and at 0.60x to 0.84x the speed of `voracious_radix_sort`. On x86 CPUs without AVX2 it runs scalar code, because the planned SSE4.2 path is not written. All published numbers come from one x86 laptop.

- **Random 64-bit keys.** Below 513 elements, 8-byte keys get no SIMD network. Above that, random 64-bit keys need eight byte passes, or six 11-bit passes from 65,536 elements. At 1M `u64` turbosort runs at 0.77x of `std` and 0.67x of voracious. For 64-bit keys voracious uses a diverting LSD radix sort: it sorts on the top digits until the buckets are small, then finishes with insertion sort, so random keys need fewer passes over memory. For large arrays of random 64-bit keys, use voracious. Keys whose high bytes never change, such as small IDs, skip those passes in turbosort. The benchmarks measure `u64` only; `i64` and `f64` run the same 8-byte code.
- **Cache-friendly 1M results.** At 1M keys the working set half fits in the laptop's 8 MB L3 cache. The 10M results (2.2x to 2.4x over `std` for `u32`) are the fairer measure of large inputs.
- **Parallel scaling.** `sort_parallel` gains 1.03x over `sort` at 1M elements and 1.58x at 10M on four cores.
- **`no_std`.** Without `std`, x86 builds skip AVX2 detection and use scalar code. `sort_with_buffer` never uses the 11-bit path, so large 8-byte keys take eight passes, and it uses up to about 64 KiB of stack for their histograms.
- **No aarch64 numbers.** CI runs the NEON code, but there are no published benchmarks for it.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your option. `Cargo.toml` declares `MIT OR Apache-2.0`.
