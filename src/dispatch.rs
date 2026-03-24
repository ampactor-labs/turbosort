//! Size-based algorithm dispatch.
//!
//! Routes to the optimal algorithm based on input length:
//! - 0-1: no-op
//! - 2-16: sorting network ([`tiny`])
//! - 17-512: quicksort ([`small`])
//! - >512: radix sort (Phase 2)
//! - >131K + `parallel` feature: parallel radix sort (Phase 6)

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

    // Phase 2: radix sort for large arrays
    #[cfg(feature = "alloc")]
    {
        crate::radix::sort(slice);
    }

    // no_std without alloc: fall back to quicksort
    #[cfg(not(feature = "alloc"))]
    {
        crate::arch::scalar::quicksort(slice);
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

    // Phase 2: radix sort with caller-provided buffer
    crate::radix::sort_with_buffer(slice, buffer);
}
