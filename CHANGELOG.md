# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- From 17 to 512 elements, both quicksorts (AVX2 for 4-byte keys from 129
  elements, scalar for every other case) handed the whole slice to
  insertion sort when median-of-three picked the minimum or the depth limit
  hit. Sorted, reversed, few-unique, pipe-organ and sawtooth input ran at
  0.05x to 0.52x of `sort_unstable` at 512 elements (up to 43 µs against
  2-5 µs).
- `sort` without the `alloc` feature went quadratic above 512 elements on
  the same patterns: 795 ms for 80,000 `u32` of `i % 4`. It now uses
  `core`'s `sort_unstable`.
- Reversed input above 512 elements ran every radix pass, at 0.03x to 0.11x
  of `sort_unstable` from 1K to 64K elements; it is now reversed in one scan.
- The rustdoc said NaN sorts last. Only a NaN with the sign bit clear does;
  one with it set (x86's `0.0 / 0.0`) sorts first, as `f32::total_cmp` and
  the README say.
- `sort_parallel` below 131,072 elements ran the radix core directly, even
  for slices the networks handle; it now runs the serial `sort`.

### Changed
- AVX2, 4-byte keys, 129 to 512 elements: 128-element network blocks merged
  with an 8-wide bitonic merge replace the quicksort. 512 random `u32` went
  from 1.46x to 2.15x `sort_unstable` on a 4-vCPU Xeon VM, and no input
  pattern is slower than another.
- Every other type and platform from 17 to 512 elements uses `core`'s
  `sort_unstable_by_key` on the radix key instead of the scalar quicksort.
- Above 16 elements, one vectorized scan finishes ascending and descending
  input before any tier runs; before, only the radix tier checked, and only
  for ascending input.
- `u8` and `i8` use the counting sort from 64 elements instead of 513, in
  every build, with interleaved counters from 8,192 elements.
- Diverting LSD for wide keys: the radix sort sorts only the top digits that
  carry log2(n) + 2 bits, then one scan finishes the short runs left. Random
  `u64` went from 0.63x to 1.18x `sort_unstable` at 4K and from 0.65x to
  0.90x at 1M on the VM. `sort_parallel` does not divert yet.
- Byte radix passes whose buckets are many and nearly equal in size
  (permutations, pipe organs, sawtooths) stage each bucket's writes in a
  cache-line buffer and write whole lines. With equal buckets the plain
  scatter's write positions contend for a few cache sets. A random
  permutation of 65,536 `u32` went from 758 µs to 344 µs (1.48x to 3.14x
  `sort_unstable`); random keys never take this path.
- The crate description no longer says "SIMD-accelerated radix sort": the
  radix passes are scalar; SIMD is in the small-slice networks and merges.

### Added
- A `patterns` benchmark group: sorted, reversed, few-unique, pipe-organ,
  sawtooth and permutation inputs for `u32` and `u64` at 512 and 65,536.
- Tests for the merge tier at every length from 129 to 512, the presorted
  scan, the diverting digit choice and finishing scan, presorted input in
  every tier, and structured `u64` and `f64` inputs.
- CI runs the test suite without default features and with only `alloc`,
  and lints tests and benches; the release workflow runs the tests before
  publishing.

### Removed
- The AVX2 partition (`src/arch/x86_64/partition.rs`) and its lookup table
  (`src/lut.rs`), which only the quicksort used.

## [0.2.1] - 2026-07-29

### Added
- `examples/profile.rs`, a perf/flamegraph workload with size, type, and
  algorithm arguments, plus a `profiling` build profile (release codegen with
  debug info). Frame-pointer builds are the documented recipe; dwarf
  unwinding loses the AVX2 hot loops under `perf script`.
- README "Profiling" section with the measured story: 73% of cycles in the
  scatter passes and 22% in histograms; ~3x over `std::sort_unstable` at 100M
  random `u32` (no large-N cliff); 88% of LLC misses in the scatter pass; dTLB
  misses under 0.007/key at every size because the 256 scatter destination
  streams keep the hot page set small. Flamegraph committed at
  `docs/flamegraph.svg`, excluded from the published package.

## [0.2.0] - 2026-07-23

### Fixed
- aarch64: sorting 9–16 element `u32`/`i32`/`f32` slices panicked with an
  index out of bounds — the NEON tiny path used an 8-element buffer but
  received up to 16 elements. The NEON networks were rewritten branch-free
  and extended to 16 elements (four 4-lane registers, bitonic merges),
  verified under Miri's aarch64 interpreter and 0-1-principle tests.
- The README claimed `u64` sorts in "two 32-bit halves"; the code has always
  done one pass per key byte. Docs now match the implementation.

### Changed
- Mid-range sorts (17–128 elements) go straight to padded AVX2 bitonic
  networks (tiers of 32/64/128 in registers), and quicksort's leaf grew from
  16 to 128. The 128-element tier went from 0.69x vs `std::sort_unstable`
  to 2.3x; 512 from 0.67x to 1.4x.
- Radix histograms are sized to the key's actual pass count and computed
  with a single copy below 8K elements (four interleaved copies above),
  replacing a fixed 64KB of zeroed stack per call. The 4K `u32` tier is
  about 2.5x faster.
- The radix scratch buffer is no longer zeroed: scatter passes write every
  position, so the allocation stays uninitialized until first scatter.
- 8-byte keys at 64K+ elements switch to 11-bit digits — six passes instead
  of eight. 1M-element `u64` improved 1.4x over the byte path; 8-bit,
  13-bit, and 16-bit digit widths were benchmarked and lost.
- `u8`/`i8` slices above 512 elements use a counting sort with no scratch
  buffer.
- Radix and parallel sorts detect already-sorted input in one scan and
  return early.
- `sort_parallel` reuses its initial all-passes histogram scan for the first
  scatter pass, uses the wide-digit path for 8-byte keys, skips the scratch
  zeroing, and copies back in parallel: 1.24x at 10M `u32` over 0.1.1 under
  identical conditions. A fused scatter+histogram variant was measured
  slower and rejected (see the module docs for why).

### Added
- CI now tests on Apple Silicon (native NEON), cross-checks clippy for
  aarch64 and a `thumbv7em` no_std target, builds with the 1.75 MSRV, and
  runs the unsafe core under Miri on both x86_64 and aarch64. The 0.1.1
  aarch64 panic could not have shipped with this matrix.
- Tests: NEON 0-1-principle and plumbing tests, direct unit tests for every
  radix core at Miri-friendly sizes, and a parallel test block covering
  random, few-unique, sorted, and counting-sort inputs for 300K elements.

## [0.1.1] - 2026-06-30

### Fixed
- `sort_parallel` now recomputes per-chunk histograms on each pass, fixing
  incorrect ordering for some inputs.

### Changed
- 4-way histogram interleaving and a parallel histogram scan for faster radix
  passes.
- Extracted and corrected the AVX2 SIMD partition (Bramas neutralize).

### Added
- `voracious_radix_sort` comparison benchmarks and full benchmark results in the
  README.
- crates.io / docs.rs / license / downloads badges.

## [0.1.0] - 2026-03-25

### Added
- Initial release: SIMD-accelerated LSD radix sort for the 10 primitive numeric
  types (`u8`–`u64`, `i8`–`i64`, `f32`, `f64`).
- `sort`, `sort_with_buffer`, and the feature-gated `sort_parallel` (rayon).
- AVX2 sorting networks for tiny inputs, quicksort + SIMD leaf for mid-range
  arrays, and LSD radix for large arrays.

[Unreleased]: https://github.com/ampactor-labs/turbosort/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/ampactor-labs/turbosort/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ampactor-labs/turbosort/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/ampactor-labs/turbosort/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ampactor-labs/turbosort/releases/tag/v0.1.0
