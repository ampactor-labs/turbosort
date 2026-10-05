//! Sorting networks for very small arrays (n ≤ 16).
//!
//! 4-byte keys go to the SIMD networks: AVX2 from 4 elements, NEON at every
//! length. Everything else uses `core`'s unstable sort, which is an insertion
//! sort at these lengths.

use crate::key::SortableKey;

/// Sort a slice of at most 16 elements.
///
/// # Panics
///
/// Debug-asserts that `slice.len() <= 16`.
#[inline]
pub fn sort<T: SortableKey>(slice: &mut [T]) {
    debug_assert!(slice.len() <= 16);
    crate::arch::sort_tiny(slice);
}
