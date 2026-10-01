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

/// Bytes staged per bucket by [`scatter_pass_combined`]: one cache line.
const LINE: usize = 64;

/// One staging line per bucket, aligned so each is exactly one cache line.
#[repr(C, align(64))]
struct Lines([[u8; LINE]; 256]);

/// Like [`scatter_pass`], but stages each bucket's elements in a cache-line
/// buffer and writes them to `dst` a whole line at a time (software write
/// combining).
///
/// When every bucket holds nearly the same number of keys (a permutation of
/// 0..n, for one), the plain scatter's 256 write positions sit at the same
/// offsets within their pages, contend for a few cache sets and miss on
/// almost every write. Here every element is written to one of 256 lines
/// that stay in L1, and `dst` receives full lines, aligned to its own cache
/// lines. On inputs without that shape the staging is pure overhead, so
/// callers choose this only for passes [`wants_combining`] picks.
///
/// Kept out of line so its 16 KiB of staging lines only take stack space in
/// the passes that use them.
///
/// # Safety
///
/// Same contract as [`scatter_pass`]; `offsets` is read, not advanced.
#[inline(never)]
pub unsafe fn scatter_pass_combined<T: SortableKey>(
    src: *const T,
    dst: *mut T,
    len: usize,
    offsets: &[usize; 256],
    pass: usize,
) {
    let size = core::mem::size_of::<T>();
    debug_assert!(size.is_power_of_two() && size <= LINE);
    let per_line = LINE / size;
    let mut lines = core::mem::MaybeUninit::<Lines>::uninit();
    let lines = lines.as_mut_ptr() as *mut T;

    // dst[i] sits in slot (lead + i) % per_line of its cache line. Bucket d
    // starts its staging line at the slot its first position has, so every
    // flush after the first covers exactly one destination cache line.
    // `base[d]` is the dst index of slot 0 (it wraps below 0 for a bucket that
    // starts mid-line at index 0); slots before `first[d]` are never copied.
    let lead = (dst as usize % LINE) / size;
    let mut base = [0usize; 256];
    let mut fill = [0u8; 256];
    let mut first = [0u8; 256];
    for d in 0..256 {
        let slot = (lead + offsets[d]) % per_line;
        base[d] = offsets[d].wrapping_sub(slot);
        fill[d] = slot as u8;
        first[d] = slot as u8;
    }

    for i in 0..len {
        let elem = *src.add(i);
        let d = elem.to_radix_key().radix_digit(pass) as usize;
        let f = fill[d] as usize;
        lines.add(d * per_line + f).write(elem);
        if f + 1 < per_line {
            fill[d] = (f + 1) as u8;
            continue;
        }
        // The line is full: slots first..per_line hold the bucket's next
        // keys, which belong at dst[base + first..base + per_line].
        let start = first[d] as usize;
        core::ptr::copy_nonoverlapping(
            lines.add(d * per_line + start),
            dst.add(base[d].wrapping_add(start)),
            per_line - start,
        );
        base[d] = base[d].wrapping_add(per_line);
        fill[d] = 0;
        first[d] = 0;
    }

    for d in 0..256 {
        let (start, end) = (first[d] as usize, fill[d] as usize);
        if end > start {
            core::ptr::copy_nonoverlapping(
                lines.add(d * per_line + start),
                dst.add(base[d].wrapping_add(start)),
                end - start,
            );
        }
    }
}

/// Inputs below this length scatter plainly: their buckets are too short for
/// the cache-set contention that write combining avoids.
const COMBINE_MIN: usize = 16_384;

/// Whether a pass with histogram `hist` has the shape write combining is
/// for: at least 64 non-empty buckets whose sizes are within about 3% of each
/// other. Random keys spread their bucket sizes far wider than that, so they
/// keep the plain scatter.
pub fn wants_combining(hist: &[usize], len: usize) -> bool {
    if len < COMBINE_MIN {
        return false;
    }
    let (mut buckets, mut min, mut max) = (0usize, usize::MAX, 0usize);
    for &count in hist {
        if count > 0 {
            buckets += 1;
            min = min.min(count);
            max = max.max(count);
        }
    }
    buckets >= 64 && max - min <= len / buckets / 32 + 2
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::radix::{histogram, prefix_sum};

    /// One combined pass over `src` into a destination that starts `skew`
    /// elements into an allocation, against the plain scatter.
    fn check_pass<T: SortableKey + PartialEq + core::fmt::Debug>(
        src: &[T],
        pass: usize,
        skew: usize,
    ) {
        let [hist] = histogram::compute_bytes_one(src, pass);
        let mut offsets = hist;
        prefix_sum::exclusive_prefix_sum(&mut offsets);

        let mut expected: Vec<T> = Vec::with_capacity(src.len());
        let mut plain_offsets = offsets;
        unsafe {
            scatter_pass(
                src.as_ptr(),
                expected.as_mut_ptr(),
                src.len(),
                &mut plain_offsets,
                pass,
            );
            expected.set_len(src.len());
        }

        let mut out: Vec<T> = Vec::with_capacity(src.len() + skew);
        unsafe {
            let dst = out.as_mut_ptr().add(skew);
            scatter_pass_combined(src.as_ptr(), dst, src.len(), &offsets, pass);
            out.set_len(src.len() + skew);
        }
        assert_eq!(&out[skew..], &expected[..], "pass {pass}, skew {skew}");
    }

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn combined_matches_plain_for_every_width_and_alignment() {
        let mut state = 0x9E3779B97F4A7C15u64;
        // Skewed bytes leave some buckets empty and many shorter than a line.
        let n = 1500;
        let u16s: Vec<u16> = (0..n)
            .map(|_| (xorshift(&mut state) % 700) as u16)
            .collect();
        let u32s: Vec<u32> = (0..n).map(|_| xorshift(&mut state) as u32).collect();
        let i64s: Vec<i64> = (0..n).map(|_| xorshift(&mut state) as i64 >> 7).collect();
        let f64s: Vec<f64> = (0..n).map(|i| i as f64 * 0.37 - 200.0).collect();
        for skew in 0..4 {
            check_pass(&u16s, 0, skew);
            check_pass(&u16s, 1, skew);
            check_pass(&u32s, 0, skew);
            check_pass(&u32s, 2, skew);
            check_pass(&i64s, 0, skew);
            check_pass(&i64s, 7, skew);
            check_pass(&f64s, 6, skew);
        }
    }

    #[test]
    fn combining_is_chosen_for_equal_buckets_only() {
        let len = 1 << 16;
        let equal = [256usize; 256];
        assert!(wants_combining(&equal, len));
        assert!(!wants_combining(&equal, 4096), "too short to bother");

        let mut state = 0x2545F4914F6CDD1Du64;
        let mut random = [0usize; 256];
        for _ in 0..len {
            random[(xorshift(&mut state) & 0xFF) as usize] += 1;
        }
        assert!(!wants_combining(&random, len));

        let mut few = [0usize; 256];
        few[..4].fill(len / 4);
        assert!(!wants_combining(&few, len), "four streams cannot alias");
    }
}
