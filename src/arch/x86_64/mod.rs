//! x86_64 SIMD backend (AVX2).
//!
//! A runtime CPUID check picks AVX2 when the CPU has it; otherwise the
//! scalar paths run.

pub mod avx2;
