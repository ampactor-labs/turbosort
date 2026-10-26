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
/// Every element is written to one of 256 lines that stay in L1, and `dst`
/// receives full lines, aligned to its own cache lines. Two shapes call for
/// it ([`wants_combining`] and [`streams`] choose):
///
/// - When every bucket holds nearly the same number of keys (a permutation
///   of 0..n, for one), the plain scatter's 256 write positions sit at the
///   same offsets within their pages, contend for a few cache sets and miss
///   on almost every write.
/// - When `dst` is larger than the caches, each line the plain scatter
///   touches is first read from memory and only then overwritten, about one
///   memory round trip per 64 bytes. With `stream` set, full lines go out
///   as non-temporal stores, which write memory without reading it first
///   and keep the destination out of the caches. On the 4-vCPU Xeon VM one
///   out-of-cache pass over 1M random `u32` went from 5.3-6.3 ns to 2.2-2.6
///   ns per key; inside the caches the plain scatter stayed faster.
///
/// On other inputs the staging is overhead. Kept out of line so its 16 KiB of
/// staging lines only take stack space in the passes that use them.
///
/// # Safety
///
/// Same contract as [`scatter_pass`]; `offsets` is read, not advanced. `dst`
/// must be aligned for `T`, as any pointer to `T` must.
#[inline(never)]
pub unsafe fn scatter_pass_combined<T: SortableKey>(
    src: *const T,
    dst: *mut T,
    len: usize,
    offsets: &[usize; 256],
    pass: usize,
    stream: bool,
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
        if stream && start == 0 {
            // A whole destination line: dst[base] starts a cache line (see
            // `lead` above), and so does the staging line.
            stream_line(
                lines.add(d * per_line) as *const u8,
                dst.add(base[d]) as *mut u8,
            );
        } else {
            core::ptr::copy_nonoverlapping(
                lines.add(d * per_line + start),
                dst.add(base[d].wrapping_add(start)),
                per_line - start,
            );
        }
        base[d] = base[d].wrapping_add(per_line);
        fill[d] = 0;
        first[d] = 0;
    }

    if stream {
        stream_fence();
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

/// Write the 64-byte line at `src` to `dst` with non-temporal stores. Both
/// must be 64-byte aligned. Elsewhere than x86_64 it is a plain copy, and so
/// under Miri, which cannot run the stores; the alignment is checked there.
#[inline(always)]
unsafe fn stream_line(src: *const u8, dst: *mut u8) {
    debug_assert!(src as usize % LINE == 0 && dst as usize % LINE == 0);
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        use core::arch::x86_64::{__m128i, _mm_load_si128, _mm_stream_si128};
        // SSE2 is part of the x86_64 baseline.
        let (s, d) = (src as *const __m128i, dst as *mut __m128i);
        _mm_stream_si128(d, _mm_load_si128(s));
        _mm_stream_si128(d.add(1), _mm_load_si128(s.add(1)));
        _mm_stream_si128(d.add(2), _mm_load_si128(s.add(2)));
        _mm_stream_si128(d.add(3), _mm_load_si128(s.add(3)));
    }
    #[cfg(any(not(target_arch = "x86_64"), miri))]
    core::ptr::copy_nonoverlapping(src, dst, LINE);
}

/// Order the non-temporal stores before anything that follows, as their
/// weak ordering requires. Miri cannot run the fence and, interpreting one
/// thread, has nothing for it to order, so it skips it.
#[inline(always)]
fn stream_fence() {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    // SAFETY: SSE is part of the x86_64 baseline.
    unsafe {
        core::arch::x86_64::_mm_sfence()
    };
}

/// Destinations of at least this many bytes take the streaming scatter on
/// x86_64. Below it the destination fits in the caches, where non-temporal
/// stores only lose: at 1 MiB the two tied on the VM, at 4 MiB streaming
/// was twice as fast.
const STREAM_MIN_BYTES: usize = 2 << 20;

/// Whether a scatter of `len` elements of `T` should use
/// [`scatter_pass_combined`] with non-temporal stores. Only x86_64 has them
/// here; elsewhere combining without them measured slower than the plain
/// scatter, so the answer is no.
pub fn streams<T>(len: usize) -> bool {
    cfg!(target_arch = "x86_64")
        && len.saturating_mul(core::mem::size_of::<T>()) >= STREAM_MIN_BYTES
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

        for stream in [false, true] {
            let mut out: Vec<T> = Vec::with_capacity(src.len() + skew);
            unsafe {
                let dst = out.as_mut_ptr().add(skew);
                scatter_pass_combined(src.as_ptr(), dst, src.len(), &offsets, pass, stream);
                out.set_len(src.len() + skew);
            }
            assert_eq!(
                &out[skew..],
                &expected[..],
                "pass {pass}, skew {skew}, stream {stream}"
            );
        }
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
