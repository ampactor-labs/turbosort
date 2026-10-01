//! Medium arrays (17 ≤ n ≤ 512).
//!
//! With AVX2 and a 4-byte key, sizes up to 128 go straight to padded sorting
//! networks and larger sizes sort 128-element blocks and merge them. Every
//! other case uses `core`'s unstable sort on the radix keys.

use crate::key::SortableKey;

/// Sort a slice of 17..=512 elements.
#[inline]
pub fn sort<T: SortableKey>(slice: &mut [T]) {
    debug_assert!((17..=512).contains(&slice.len()));
    crate::arch::sort_small(slice);
}
