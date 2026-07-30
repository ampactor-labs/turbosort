# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.1]: https://github.com/ampactor-labs/turbosort/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ampactor-labs/turbosort/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/ampactor-labs/turbosort/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ampactor-labs/turbosort/releases/tag/v0.1.0
