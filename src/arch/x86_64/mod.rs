//! x86_64 SIMD backends (AVX2, SSE4.2).
//!
//! Runtime CPUID dispatch selects the best available instruction set.
//! Falls back to scalar if no SIMD is detected.

// Phase 3+: AVX2 sorting networks, SIMD partition, SIMD histogram
