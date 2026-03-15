//! Sorting networks for very small arrays (n ≤ 16).
//!
//! Dispatches to SIMD sorting networks when available (Phase 3),
//! falls back to insertion sort on scalar.

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
