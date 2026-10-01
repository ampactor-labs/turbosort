//! Size-based algorithm dispatch.
//!
//! Routes to the optimal algorithm based on input length:
//! - 0-1: no-op
//! - 2-16: sorting network or insertion sort ([`crate::tiny`])
//! - 17-512: sorting networks and merges, or `core`'s unstable sort
//!   ([`crate::small`])
//! - >512: LSD radix sort ([`crate::radix`])
//!
//! [`sort_parallel`](crate::sort_parallel) adds a parallel radix tier for
//! arrays over 131K elements.

use crate::key::SortableKey;

/// Sort a slice using the best algorithm for its size.
///
/// This is the core dispatch function called by the public API.
#[inline]
pub fn sort<T: SortableKey>(slice: &mut [T]) {
    let len = slice.len();

    if len <= 1 {
        return;
    }

    if len <= 16 {
        crate::tiny::sort(slice);
        return;
    }

    if len <= 512 {
        crate::small::sort(slice);
        return;
    }

    #[cfg(feature = "alloc")]
    {
        crate::radix::sort(slice);
    }

    // no_std without alloc: no scratch buffer for the radix sort.
    #[cfg(not(feature = "alloc"))]
    {
        crate::arch::scalar::comparison_sort(slice);
    }
}

/// Sort a slice using the best algorithm, with a pre-allocated buffer.
///
/// For `no_std` or allocation-sensitive callers. The buffer must be at least
/// as long as `slice`. Elements in `buffer` are overwritten.
///
/// # Panics
///
/// Panics if `buffer.len() < slice.len()`.
pub fn sort_with_buffer<T: SortableKey>(slice: &mut [T], buffer: &mut [T]) {
    assert!(
        buffer.len() >= slice.len(),
        "buffer too small: need {}, got {}",
        slice.len(),
        buffer.len()
    );

    let len = slice.len();

    if len <= 1 {
        return;
    }

    if len <= 16 {
        crate::tiny::sort(slice);
        return;
    }

    if len <= 512 {
        crate::small::sort(slice);
        return;
    }

    crate::radix::sort_with_buffer(slice, buffer);
}
