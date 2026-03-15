//! Architecture-specific backend selection.
//!
//! At compile time, cfg gates select which SIMD backends are available.
//! At runtime, CPUID checks pick the best available instruction set.

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

use crate::key::SortableKey;

/// Sort a small slice (n ≤ 16) using the best available backend.
#[inline]
pub fn sort_tiny<T: SortableKey>(slice: &mut [T]) {
    // TODO: Phase 3 adds SIMD sorting networks here
    scalar::insertion_sort(slice);
}

/// Sort a medium slice (17..=512) using the best available backend.
#[inline]
pub fn sort_small<T: SortableKey>(slice: &mut [T]) {
    // TODO: Phase 4 adds SIMD quicksort here
    scalar::quicksort(slice);
}
