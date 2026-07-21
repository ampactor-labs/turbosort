//! Stable LSD scatter passes.
//!
//! Moves elements from source to destination according to their radix digit
//! and the pre-computed offset table. Stability is guaranteed because elements
//! with the same digit are written in their original order.
//!
//! Both scatters work on raw pointers so the destination may be uninitialized
//! memory: a scatter writes every position `0..len` exactly once (the offset
//! ranges partition the destination), so the destination is fully initialized
//! afterwards.

use crate::key::{SortableKey, UnsignedKey};

/// Scatter `len` elements from `src` into `dst` by the byte digit at `pass`.
///
/// Each offset advances as elements are placed, so after the call
/// `offsets[d]` points one past the last element with digit `d`.
///
/// # Safety
///
/// - `src` must be valid for `len` reads; `dst` for `len` writes.
/// - `src` and `dst` must not overlap.
/// - `offsets` must be the exclusive prefix sum of the digit histogram of
///   `src[..len]` for this pass — this guarantees every write lands in
///   `0..len` and no position is written twice.
pub unsafe fn scatter_pass<T: SortableKey>(
    src: *const T,
    dst: *mut T,
    len: usize,
    offsets: &mut [usize; 256],
    pass: usize,
) {
    for i in 0..len {
        let elem = *src.add(i);
        let digit = elem.to_radix_key().radix_digit(pass) as usize;
        let pos = offsets[digit];
        *dst.add(pos) = elem;
        offsets[digit] = pos + 1;
    }
}

/// Scatter by an arbitrary-width digit: bits `shift..shift+width` of the key,
/// with `mask == (1 << width) - 1` and `offsets.len() == mask + 1`.
///
/// # Safety
///
/// Same contract as [`scatter_pass`], with `offsets` the exclusive prefix sum
/// of the wide-digit histogram of `src[..len]`.
#[cfg(feature = "alloc")]
pub unsafe fn scatter_pass_wide<T: SortableKey>(
    src: *const T,
    dst: *mut T,
    len: usize,
    offsets: &mut [usize],
    shift: u32,
    mask: usize,
) {
    debug_assert_eq!(offsets.len(), mask + 1);
    for i in 0..len {
        let elem = *src.add(i);
        let digit = elem.to_radix_key().wide_digit(shift, mask);
        let pos = *offsets.get_unchecked(digit);
        *dst.add(pos) = elem;
        *offsets.get_unchecked_mut(digit) = pos + 1;
    }
}
