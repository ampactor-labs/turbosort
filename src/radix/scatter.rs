//! Stable LSD scatter pass.
//!
//! Moves elements from source to destination according to their radix digit
//! and the pre-computed offset table. Stability is guaranteed because elements
//! with the same digit are written in their original order.

use crate::key::{SortableKey, UnsignedKey};

/// Scatter elements from `src` into `dst` based on the radix digit at `pass`.
///
/// `offsets` is a 256-element array of mutable write positions. Each offset is
/// advanced as elements are placed, so after the call, `offsets[d]` points past
/// the last element with digit `d`.
///
/// # Safety
///
/// - `dst` must be at least as long as `src`.
/// - `offsets` must contain valid indices into `dst`.
/// - Each offset range must be non-overlapping (guaranteed by correct prefix sum).
pub fn scatter_pass<T: SortableKey>(
    src: &[T],
    dst: &mut [T],
    offsets: &mut [usize; 256],
    pass: usize,
) {
    for &elem in src {
        let digit = elem.to_radix_key().radix_digit(pass) as usize;
        let pos = offsets[digit];
        dst[pos] = elem;
        offsets[digit] = pos + 1;
    }
}
