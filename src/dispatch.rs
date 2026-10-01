//! Size-based algorithm dispatch.
//!
//! Routes to the optimal algorithm based on input length:
//! - 0-1: no-op
//! - 2-16: sorting networks for 4-byte keys (AVX2 from 4 elements, NEON),
//!   `core`'s unstable sort otherwise ([`crate::tiny`])
//! - 17-512: sorting networks and merges, or `core`'s unstable sort
//!   ([`crate::small`])
//! - >512: LSD radix sort ([`crate::radix`])
//!
//! Above 16 elements, one scan first finishes input that is already sorted
//! either way ([`crate::presorted`]). One-byte keys (`u8`, `i8`) use a
//! counting sort from 64 elements, which needs no scratch buffer.
//!
//! [`sort_parallel`](crate::sort_parallel) adds a parallel radix tier for
//! arrays over 131K elements.

use crate::key::{SortableKey, UnsignedKey};

/// One-byte keys switch to counting sort at this length. Below it, the
/// counting sort's fixed cost (256 counters to clear and walk) loses to the
/// comparison sorts; above it, it wins by 1.6x at 128 and 5x at 512.
const COUNTING_MIN: usize = 64;

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

    if crate::presorted::finish_presorted(slice) {
        return;
    }

    if T::Key::BYTES == 1 && len >= COUNTING_MIN {
        crate::radix::counting_sort(slice);
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

    if crate::presorted::finish_presorted(slice) {
        return;
    }

    if T::Key::BYTES == 1 && len >= COUNTING_MIN {
        crate::radix::counting_sort(slice);
        return;
    }

    if len <= 512 {
        crate::small::sort(slice);
        return;
    }

    crate::radix::sort_with_buffer(slice, buffer);
}
