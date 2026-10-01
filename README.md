# turbosort

[![CI](https://img.shields.io/github/actions/workflow/status/ampactor-labs/turbosort/ci.yml?branch=master&label=ci)](.github/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/turbosort.svg)](https://crates.io/crates/turbosort)
[![Documentation](https://docs.rs/turbosort/badge.svg)](https://docs.rs/turbosort)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/turbosort.svg)](#license)

A Rust crate that sorts slices of primitive numbers, switching to radix sort (which orders keys one digit at a time) above 512 elements. Short slices of 32-bit types go through sorting networks, fixed compare-and-swap sequences run with SIMD instructions that handle several values at once (AVX2 on x86, NEON on ARM). On an Intel i7-8665U, release 0.2.1 sorted random 32-bit integers 2.2 to 3.2 times as fast as the standard library's unstable sort, from 1M to 100M keys. Random 64-bit keys are its weak spot: on a cloud VM the current code beats the standard library on them at 128 and 4K keys and loses at 1M.

**Status: working.** Version 0.2.1 is on crates.io, `master` carries unreleased changes ([CHANGELOG](CHANGELOG.md)), and the API can still change before 1.0; x86 CPUs without AVX2 run scalar code, because the planned SSE4.2 path is not written.

Package: https://crates.io/crates/turbosort · Docs: https://docs.rs/turbosort

## Usage

Add the dependency to `Cargo.toml`:

```toml
[dependencies]
turbosort = "0.2"
```

Then sort a slice in place. `cargo test` compiles and runs this block, so it matches the current API:

```rust
// Unsigned and signed integers
let mut data = vec![5u32, 3, 8, 1, 9, 2, 7, 4, 6];
turbosort::sort(&mut data);
assert_eq!(data, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

let mut v = vec![3i32, -1, 4, -1, 5, -9];
turbosort::sort(&mut v);
assert_eq!(v, vec![-9, -1, -1, 3, 4, 5]);

// Floats sort in IEEE 754 total order, the order of f32::total_cmp:
// -inf < -0.0 < +0.0 < +inf < NaN, for a NaN whose sign bit is clear.
let mut v = vec![1.0f32, f32::NAN, -0.0, 0.0, f32::NEG_INFINITY];
turbosort::sort(&mut v);
assert_eq!(v[0], f32::NEG_INFINITY);
assert!(v[1] == 0.0 && v[1].is_sign_negative());
assert!(v[4].is_nan());

// No allocation: the caller provides a scratch buffer at least as long as the slice.
let mut data = [42u64, 7, 99, 1, 0];
let mut buf = [0u64; 5];
turbosort::sort_with_buffer(&mut data, &mut buf);
assert_eq!(data, [0, 1, 7, 42, 99]);
```

`sort` allocates one scratch buffer the length of the slice when it runs the radix sort on a type wider than one byte. `sort_with_buffer` uses the caller's buffer instead and panics if that buffer is shorter than the slice. For the multi-core sort, enable the `parallel` feature and call `turbosort::sort_parallel(&mut data)`:

```toml
[dependencies]
turbosort = { version = "0.2", features = ["parallel"] }
```

### Cargo features

| Feature | Default | What it enables |
| --- | --- | --- |
| `std` | yes | Runtime AVX2 detection through CPUID; turns on `alloc` |
| `alloc` | yes, through `std` | The scratch buffer that `sort` allocates for the radix sort |
| `parallel` | no | `sort_parallel`, a multi-core radix sort on [rayon](https://crates.io/crates/rayon) |

With `default-features = false` the crate is `no_std`. Call `sort_with_buffer` there: without `alloc`, `sort` uses the comparison sort from `core` above 512 elements instead of the radix sort (see [Limitations](#limitations)).

### Supported types

`sort` accepts slices of the ten primitive number types: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `f32` and `f64`. They implement the sealed `SortableKey` trait, which other types cannot implement; use the standard library's `sort_unstable` for anything else.

## How it works

`sort` picks an algorithm by slice length (`src/dispatch.rs`):

| Length | Algorithm |
| --- | --- |
| 0 to 1 | Nothing to do |
| 2 to 16 | Sorting network: AVX2 on x86_64, NEON on aarch64 |
| 17 to 128 | Sorting networks padded to 32, 64 or 128 elements, built by merging sorted runs inside AVX2 registers |
| 129 to 512 | 128-element blocks sorted by those networks, then merged pairwise with an 8-wide AVX2 bitonic merge |
| 513 and up | LSD radix sort |

A sorting network is a fixed sequence of compare-and-swap steps, so it has no data-dependent branches and maps onto SIMD min and max instructions. The merges do not depend on the data's order either, so in the SIMD rows every input pattern costs the same. Those rows apply only to types with 4-byte keys (`u32`, `i32`, `f32`) on a CPU that has the instructions. Every other case uses insertion sort up to 16 elements and the standard library's `sort_unstable` (from `core`, so it allocates nothing) up to 512; on aarch64 that includes 17 to 512 elements, because NEON only covers the 16-element networks. `u8` and `i8` switch to a counting sort at 64 elements.

Above 16 elements, one scan first checks whether the slice is already in order (`src/presorted.rs`). An ascending slice returns after it and a descending one is reversed, which sorts it because equal keys are identical values for these types. Random input fails the check within a comparison or two; the rest of the scan runs in fixed blocks that the compiler vectorizes.

Every comparison and every radix digit works on an unsigned integer key with the same order as the value (`src/key.rs`). Unsigned types use the value itself, signed types flip the sign bit, and floats flip the sign bit when positive and every bit when negative. The networks and radix passes only ever see unsigned integers.

The radix sort works from the least significant digit (LSD) up. Each pass distributes the keys into 256 buckets by one byte and keeps their order within a bucket, so after the pass over the most significant byte the slice is sorted (`src/radix/`):

- One scan builds the histograms for every pass at once. A pass where every key has the same digit is skipped, so keys whose high bytes never vary skip those passes.
- The scratch buffer that `sort` allocates is never zeroed, because every scatter pass writes every position.
- For 8-byte keys (`u64`, `i64`, `f64`) of 65,536 elements or more, `sort` uses 11-bit digits: six passes instead of eight. That made 1M-element `u64` 1.4x faster than the byte path in 0.2.0, and 8-, 13- and 16-bit digits were measured and lost ([CHANGELOG](CHANGELOG.md), 0.2.0). `sort_with_buffer` stays on byte digits, because the wide path allocates its histograms.
- Wide keys divert (`src/radix/divert.rs`). When the top digits alone carry log2(n) + 2 bits, as they do for random 64-bit keys, only those digits are radix-sorted; one scan then sorts each run of keys that agree on them, by insertion sort up to 16 keys. At 1M random `u64`, `sort` runs three of its six 11-bit passes and `sort_with_buffer` three of its eight byte passes. The digit count comes from the histograms; if it is too optimistic, the runs come out longer and the scan sorts them with `sort_unstable`. Diverting only happens when it skips at least three digits, which in practice means 8-byte keys.

`sort_parallel` hands slices of 131,072 elements or more to rayon and runs the serial `sort` below that (`src/parallel.rs`). It does not divert: 8-byte keys run all six 11-bit passes. One parallel scan builds per-chunk histograms for every pass, and the first pass reuses them as its scatter offsets. Each thread then scatters its chunk into its own ranges of the destination, so no locks or atomics are needed. A fused variant that built the next pass's histograms during the scatter measured slower and was dropped; the module docs record why.

### Architecture support

| Arch | SIMD | What it covers |
| --- | --- | --- |
| x86_64 | AVX2 | Networks up to 128 elements, networks and merges up to 512 |
| x86_64 | SSE4.2 | Planned; no code yet |
| aarch64 | NEON | Networks up to 16 elements |
| Other | none | Insertion sort and `core`'s `sort_unstable`, the same radix sort |

On x86_64 the crate detects AVX2 at runtime through CPUID when the `std` feature is on, so one binary runs on any x86_64 CPU. NEON is part of every aarch64 CPU, so it needs no check.

## Benchmarks

The tables come from one criterion run of release 0.2.1 on an Intel i7-8665U (4 cores, 8 threads, AVX2, 8 MB L3, a 15 W laptop chip) at 30 samples per point, with the three sorts measured back to back at each size. The data is random from a fixed seed (`benches/sort_bench.rs`): `u32`, `u64` and `i32` over their full range, `f32` uniform in [-1e6, 1e6). The baselines are the standard library's `sort_unstable` (`sort_unstable_by` with `partial_cmp` for `f32`) and `voracious_sort` from voracious_radix_sort 1.2.0. Each ratio is the other sort's time divided by turbosort's, so values below 1 are losses. Absolute times swing with the chip's thermal state; the ratios within the run are the stable signal. This command runs the same benchmarks at the same sample count:

```sh
cargo bench --bench sort_bench --features parallel -- --sample-size 30
```

Serial `u32`:

| Size | std | turbosort | voracious | vs std | vs voracious |
| --- | --- | --- | --- | --- | --- |
| 16 | 66.6 ns | 47.1 ns | 87.3 ns | 1.42x | 1.86x |
| 128 | 899 ns | 388 ns | 999 ns | 2.32x | 2.58x |
| 512 | 4.76 µs | 3.50 µs | 7.72 µs | 1.36x | 2.20x |
| 4K | 42.8 µs | 26.7 µs | 35.4 µs | 1.61x | 1.33x |
| 64K | 1.28 ms | 496 µs | 622 µs | 2.58x | 1.25x |
| 1M | 21.8 ms | 8.84 ms | 15.98 ms | 2.47x | 1.81x |
| 10M | 249 ms | 105 ms | 162 ms | 2.38x | 1.55x |

Serial `u64`, `f32` and `i32`:

| Type | Size | std | turbosort | voracious | vs std | vs voracious |
| --- | --- | --- | --- | --- | --- | --- |
| `u64` | 128 | 869 ns | 1.04 µs | 868 ns | 0.84x | 0.84x |
| `u64` | 4K | 51.3 µs | 71.9 µs | 43.0 µs | 0.71x | 0.60x |
| `u64` | 1M | 23.1 ms | 29.9 ms | 20.0 ms | 0.77x | 0.67x |
| `f32` | 128 | 1.48 µs | 485 ns | 1.40 µs | 3.05x | 2.88x |
| `f32` | 4K | 81.1 µs | 42.4 µs | 63.9 µs | 1.91x | 1.51x |
| `f32` | 1M | 34.8 ms | 10.8 ms | 15.5 ms | 3.23x | 1.44x |
| `i32` | 128 | 942 ns | 361 ns | 1.02 µs | 2.61x | 2.83x |
| `i32` | 4K | 55.4 µs | 31.1 µs | 71.4 µs | 1.78x | 2.29x |
| `i32` | 1M | 22.8 ms | 9.19 ms | 12.7 ms | 2.48x | 1.38x |

Parallel `u32`, with `sort_parallel` against the serial `sort` and the standard library:

| Size | std | serial | parallel | parallel vs std | parallel vs serial |
| --- | --- | --- | --- | --- | --- |
| 1M | 23.5 ms | 8.60 ms | 8.36 ms | 2.81x | 1.03x |
| 10M | 270 ms | 116 ms | 73.4 ms | 3.67x | 1.58x |

In 0.1.x, before the register networks, the 17 to 512 range ran at 0.69x of the standard library's speed at 128 elements and 0.67x at 512 ([CHANGELOG](CHANGELOG.md), 0.2.0). In 0.2.1 random `u64` lost at all three sizes: 64-bit keys got no SIMD code below 513 elements and ran eight byte passes at 4K and six 11-bit passes at 1M. For `u64`, voracious 1.2.0 hands slices under 350 elements to `sort_unstable`, which is why the two tie at 128, and runs its diverting LSD radix sort above that; turbosort now diverts too ([Since 0.2.1](#since-021)). The parallel sort's setup cancels its gain at 1M; at 10M it reaches 1.58x over serial on four cores.

A `perf` profile of `examples/profile.rs` on the same chip, sorting 10M random `u32` and `f32` thirty times each, recorded 73% of cycles in the scatter passes and 22% in the histogram scans ([CHANGELOG](CHANGELOG.md), 0.2.1). A `perf stat` sweep measured turbosort at 2.8x, 2.2x and 3.2x the speed of `sort_unstable` on 1M, 10M and 100M random `u32`. At 1M the data and scratch (8 MB together) partly fit in the 8 MB L3, so only a third of keys miss the cache; from 10M up every key costs about one miss. [docs/BENCHMARKS.md](docs/BENCHMARKS.md) has the counters and the commands. The committed flamegraph does not show the cycle split, because no turbosort function appears in it.

A second machine did not reproduce the large-array margins. On a shared cloud VM (4 vCPUs of an Intel Xeon at 2.8 GHz), short runs of the same benchmarks matched the 4K ratios (`u32` 1.61x, `u64` 0.71x against `sort_unstable`), while random `u64` at 1M ran at 0.30x to 0.32x and random `u32` at 1M ranged from 1.62x to 0.86x across three runs. The runs and their commands are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md#a-second-machine).

### Since 0.2.1

The unreleased changes on `master` have not been measured on the laptop. On a cloud VM (a KVM guest with 4 vCPUs of an Intel Xeon at 2.1 GHz, Rust 1.97.0), criterion runs of the same benchmark file before the changes (commit 0ccf74e, with the new `patterns` group copied in) and after gave these speeds relative to `sort_unstable`, each from one run at 20 samples per point:

| Input | Size | 0.2.1 | now |
| --- | --- | --- | --- |
| `u32`, sorted | 512 | 0.05x | 2.98x |
| `u32`, reversed | 512 | 0.05x | 1.91x |
| `u32`, four distinct values | 512 | 0.06x | 0.84x |
| `u32`, sawtooth | 512 | 0.11x | 1.78x |
| `u32`, pipe organ | 512 | 0.52x | 2.25x |
| `u32`, random | 512 | 1.46x | 2.15x |
| `u64`, four distinct values | 512 | 0.05x | 1.05x |
| `u64`, pipe organ | 512 | 0.10x | 1.03x |
| `u32`, reversed | 65,536 | 0.04x | 1.67x |
| `u64`, reversed | 65,536 | 0.03x | 1.27x |
| `u64`, random | 128 | 0.89x | 1.07x |
| `u64`, random | 4,096 | 0.63x | 1.18x |
| `u64`, random | 1M | 0.65x | 0.90x |

The 17 to 512 rows changed because the two quicksorts are gone, the reversed rows because one scan now finishes presorted input, and the random `u64` rows because the radix sort diverts. Random `u32` from 4K to 1M did not change. [docs/BENCHMARKS.md](docs/BENCHMARKS.md#since-021) has every row with its times and the commands.

## Testing

```sh
cargo test                           # unit tests, tests/correctness.rs and the README doctest
cargo test --release --all-features  # the same plus the parallel tests, optimized
```

With default features on x86_64, `cargo test` runs 94 tests: 35 unit tests, 53 in `tests/correctness.rs` and 6 doctests, one of which is this README (`src/lib.rs` includes it). `--all-features` adds eight `sort_parallel` tests, seven of them at 300,000 elements, and the `sort_parallel` doctest. In a debug build the proptest cases dominate: on the 2.1 GHz cloud VM the integration suite took 73 seconds, against about 5 seconds with `--release`.

- `tests/correctness.rs` checks every permutation of up to 8 `u8` and 6 `i8` values; the lengths around the switch points (0 to 3, 15 to 17, 128, 511 to 513, 1024 and 4096); `sort_with_buffer` on 70,000 and 200,000 random `u64`; structured `u64` (cleared high bits, a split top bit, correlated top bytes, 1,000 distinct values, a heavy tail) through `sort` and `sort_with_buffer` at the same sizes; float and signed edge cases; sorted, reversed, pipe-organ, sawtooth, few-unique and median-of-three-killer inputs up to 200,000 elements, and descending input with duplicates in every tier; 2,000 proptest cases per type for each of three length ranges (0 to 16, 17 to 512, 513 to 10,000); and 500 cases each of structured `u64` and `f64` inputs. Results are compared with the standard library's sort on the same key order.
- The unit tests cover the key transforms, each radix core called directly at sizes Miri can run, including diverting on random, correlated and duplicate keys; the diverting digit choice and its finishing scan; the presorted scan around its block boundaries; the AVX2 networks at every length from 17 to 128 and the network-and-merge tier at every length from 129 to 512, with random, structured and random 0-1 inputs; the bitonic merge at every pair of run lengths up to 96; and, on aarch64, the NEON networks against every 0-1 input up to 16 elements. By the 0-1 principle, a sorting network that sorts every input of zeros and ones sorts every input.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on pushes to master and on pull requests. It checks formatting, runs `cargo clippy --all-features --all-targets -- -D warnings` and `cargo test --all-features` on ubuntu-latest and macos-latest (an Apple Silicon runner, so the NEON code runs natively), builds without default features for `thumbv7em-none-eabihf` and with only `alloc`, runs the test suite in both of those configurations on x86_64 (where, without `std`, AVX2 detection is off, so these runs also cover the scalar small sorts), runs clippy for `thumbv7em-none-eabihf` and `aarch64-unknown-linux-gnu`, builds on the 1.75 minimum supported Rust version, builds the docs the way docs.rs does, and runs `cargo publish --dry-run --locked`. Miri, an interpreter that detects undefined behaviour in unsafe code, runs the unit tests and the boundary tests on x86_64, which covers the raw-pointer scatter into uninitialized scratch. It runs the unit tests again for aarch64, which checks the NEON networks on a machine without NEON. The release workflow runs the tests once more before it publishes.

Miri reports AVX2 as unavailable, so the AVX2 code never runs under it, and the Miri jobs build without the `parallel` feature, so Miri does not check `sort_parallel` either. The AVX2 0-1 tests sample random inputs and do not cover all of them. No CI job measures speed, so a performance regression would pass.

## Limitations

The published speedups were measured on one laptop chip, an Intel i7-8665U, for release 0.2.1, and a second machine did not reproduce them for large arrays. Random 64-bit keys are the weak spot. In 0.2.1 they ran at 0.84x of the standard library's speed at 128 elements, 0.71x at 4K and 0.77x at 1M on the laptop, and voracious, another Rust radix sort, beat turbosort at all three sizes; on a shared cloud VM, 1M of them ran at 0.30x to 0.32x. The current code, on another cloud VM, reaches 1.07x at 128 and 1.18x at 4K but 0.90x at 1M. For large arrays of random 64-bit keys, measure voracious and the standard library against it.

- x86 CPUs without AVX2 get insertion sort and `core`'s `sort_unstable` below 513 elements, because the planned SSE4.2 path is not written. Above 512 elements every CPU runs the same scalar radix sort.
- AVX2 detection needs the `std` feature, so a `no_std` build on x86_64 never uses AVX2.
- Without the `alloc` feature, `sort` uses `core`'s `sort_unstable` above 512 elements (and the counting sort for `u8` and `i8`): O(n log n) on every input, but no radix sort. Use `sort_with_buffer` in `no_std` code.
- `sort_with_buffer` never allocates, so 8-byte keys stay on byte digits there. On skewed keys that cannot divert, that is up to 2x slower than `sort`'s 11-bit digits, and large 8-byte inputs use up to 64 KiB of stack for histograms.
- Inputs that fill every radix bucket equally run slower than random keys, because the equal-sized buckets put the scatter's write positions in the same cache sets. In the `patterns` benchmark a random permutation of 65,536 `u32` took 758 µs against 430 µs for random keys, and a scratch harness on the same VM put a permutation of 2^20 `u32` at 0.97x of `sort_unstable`. Write-combining buffers fixed this in a prototype (2.4x on that input) but made random input that fits in cache 1.5x slower, so they are not in yet.
- Large inputs with few distinct values trail the standard library: in the `patterns` benchmark, 65,536 `u64` with four distinct values took 415 µs against 223 µs, and `u32` ran at 0.89x. The standard library partitions equal keys away; the radix sort still runs every live pass.
- From 17 to about 22 elements, 4-byte keys run the 32-element network, at 0.6x to 0.9x of `sort_unstable`'s speed on the cloud VM.
- It sorts the ten primitive number types only. There is no comparator or key function, so it cannot sort structs or sort by a field, and `usize`, `isize`, `u128` and `i128` are not supported.
- A NaN's sign bit decides where it sorts. `f32::NAN` sorts last, but on x86_64 a NaN produced by an invalid operation (`0.0 / 0.0` gives `0xffc00000`) has the sign bit set and sorts first.
- `sort` is not a stable sort. For these types equal keys are identical bit patterns, so the order of equal elements cannot be observed.
- The `parallel` feature depends on rayon, whose releases from 1.11 on need Rust 1.80. CI builds it on Rust 1.75 against the lockfile's rayon 1.10; on Rust 1.75 to 1.79, pin it with `cargo update -p rayon --precise 1.10.0`.

## License

Licensed under either the [MIT license](LICENSE-MIT) or the [Apache License 2.0](LICENSE-APACHE), at your option (`MIT OR Apache-2.0` in `Cargo.toml`).
