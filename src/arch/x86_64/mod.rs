//! x86_64 SIMD backends (AVX2, SSE4.2).
//!
//! Runtime CPUID dispatch selects the best available instruction set.
//! Falls back to scalar if no SIMD is detected.

#[allow(dead_code)]
pub mod avx2;
