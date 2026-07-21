//! Medium arrays (17 ≤ n ≤ 512).
//!
//! With AVX2, sizes up to 128 go straight to padded sorting networks and
//! larger sizes run quicksort with SIMD partition and network leaves.
//! Falls back to scalar quicksort with Hoare partition.

use crate::key::SortableKey;

/// Sort a slice of 17..=512 elements.
#[inline]
pub fn sort<T: SortableKey>(slice: &mut [T]) {
    debug_assert!((17..=512).contains(&slice.len()));
    crate::arch::sort_small(slice);
}
