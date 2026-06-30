# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.1]: https://github.com/ampactor-labs/turbosort/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ampactor-labs/turbosort/releases/tag/v0.1.0
