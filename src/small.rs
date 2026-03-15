//! Quicksort for medium arrays (17 ≤ n ≤ 512).
//!
//! Dispatches to SIMD-accelerated partition when available (Phase 4),
//! falls back to scalar Hoare partition.

use crate::key::SortableKey;

/// Sort a slice of 17..=512 elements.
#[inline]
pub fn sort<T: SortableKey>(slice: &mut [T]) {
    debug_assert!((17..=512).contains(&slice.len()));
    crate::arch::sort_small(slice);
}
