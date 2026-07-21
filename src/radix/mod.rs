//! LSD (Least-Significant-Digit) radix sort.
//!
//! Sorts by processing digits from least-significant to most-significant.
//! Each pass is a stable counting sort on one digit, so after all passes
//! the array is fully sorted. Uses a ping-pong scratch buffer to avoid copies.
//!
//! # Structure
//!
//! - One-byte keys (`u8`/`i8`) use a plain counting sort: histogram the 256
//!   values and rewrite the slice from the counts. No scratch buffer at all.
//! - Multi-byte keys run byte-digit passes ([`sort_core_bytes`]) with
//!   histograms for all passes computed in one scan. Passes where every key
//!   shares the same digit are skipped.
//! - Eight-byte keys on large inputs switch to wider digits
//!   ([`sort_core_wide`]): fewer, more expensive passes. The width is chosen
//!   from benchmarks (see `WIDE_BINS`).
//!
//! The scratch buffer never needs to be initialized: every scatter pass
//! writes all `len` positions of its destination.

pub mod histogram;
pub mod prefix_sum;
pub mod scatter;

#[cfg(feature = "alloc")]
extern crate alloc;

use crate::key::{SortableKey, UnsignedKey};

use histogram::INTERLEAVE_MIN;

/// Minimum length before eight-byte keys use the wide-digit path.
///
/// Below this the byte-digit path wins: the wide path's heap-allocated
/// histograms and larger offset tables only amortize on big inputs.
#[cfg(feature = "alloc")]
const WIDE_MIN: usize = 65_536;

/// Bins per wide-digit pass (2^11). With 64-bit keys this gives six passes
/// of 11 bits instead of eight passes of 8 bits. Measured against 16-bit
/// digits (four passes, 65536 bins): the 16-bit offset tables fall out of L1
/// and lose by more than 2x at 1M elements on AVX2-class hardware.
#[cfg(feature = "alloc")]
const WIDE_BINS: usize = 2048;

/// Pass count for the wide-digit path: ceil(64 / 11).
#[cfg(feature = "alloc")]
const WIDE_PASSES: usize = 6;

/// Sort a slice using LSD radix sort with an internally allocated buffer.
///
/// Requires the `alloc` feature. The scratch buffer is allocated but never
/// zeroed — scatter passes fully overwrite it.
#[cfg(feature = "alloc")]
pub fn sort<T: SortableKey>(slice: &mut [T]) {
    if slice.len() <= 1 || sorted_ascending(slice) {
        return;
    }
    if T::Key::BYTES == 1 {
        counting_sort(slice);
        return;
    }
    let mut buffer: alloc::vec::Vec<T> = alloc::vec::Vec::with_capacity(slice.len());
    // SAFETY: the Vec's capacity is valid for slice.len() writes; T is Copy so
    // nothing needs dropping, and the Vec's length stays 0. `sort()` already
    // allocates, so the wide-digit path (which allocates histograms) is allowed.
    unsafe { sort_raw::<T, true>(slice, buffer.as_mut_ptr()) }
}

/// Sort a slice using LSD radix sort with a caller-provided buffer.
///
/// The buffer must be at least as long as `slice`. Unlike [`sort`], this
/// entry point performs no heap allocation, so it is usable in `no_std`
/// environments that manage their own memory: 8-byte keys stay on the
/// byte-digit path rather than the wide-digit path (which would allocate
/// histograms), trading some speed on large `u64`/`i64`/`f64` inputs for the
/// allocation-free guarantee.
///
/// For large 8-byte-key inputs it uses up to ~64 KiB of stack for the
/// interleaved histograms; prefer [`sort`] where an allocator is available.
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
    if slice.len() <= 1 || sorted_ascending(slice) {
        return;
    }
    if T::Key::BYTES == 1 {
        counting_sort(slice);
        return;
    }
    // SAFETY: buffer is a live &mut [T] of at least slice.len() elements and
    // cannot alias slice. WIDE = false keeps this path allocation-free.
    unsafe { sort_raw::<T, false>(slice, buffer.as_mut_ptr()) }
}

/// One scan; bails at the first inversion. Sorted inputs cost O(n) total
/// instead of running every radix pass.
pub(crate) fn sorted_ascending<T: SortableKey>(slice: &[T]) -> bool {
    slice
        .windows(2)
        .all(|w| w[0].to_radix_key() <= w[1].to_radix_key())
}

/// Counting sort for one-byte keys: histogram, then rewrite from counts.
pub(crate) fn counting_sort<T: SortableKey>(slice: &mut [T]) {
    debug_assert_eq!(T::Key::BYTES, 1);
    let mut counts = [0usize; 256];
    for elem in slice.iter() {
        counts[elem.to_radix_key().radix_digit(0) as usize] += 1;
    }
    let mut i = 0;
    for (d, &count) in counts.iter().enumerate() {
        if count > 0 {
            let val = T::from_radix_key(T::Key::from_digit(d as u8));
            slice[i..i + count].fill(val);
            i += count;
        }
    }
}

/// Dispatch to the monomorphized core for this key width.
///
/// `WIDE` permits the wide-digit path for 8-byte keys, which allocates its
/// histograms on the heap; callers that must not allocate pass `false`.
///
/// # Safety
///
/// `scratch` must be valid for `slice.len()` writes and must not alias `slice`.
/// It may be uninitialized.
unsafe fn sort_raw<T: SortableKey, const WIDE: bool>(slice: &mut [T], scratch: *mut T) {
    match T::Key::BYTES {
        2 => sort_core_bytes::<T, 2>(slice, scratch),
        4 => sort_core_bytes::<T, 4>(slice, scratch),
        8 => {
            #[cfg(feature = "alloc")]
            if WIDE && slice.len() >= WIDE_MIN {
                sort_core_wide::<T, WIDE_BINS, WIDE_PASSES>(slice, scratch);
                return;
            }
            sort_core_bytes::<T, 8>(slice, scratch)
        }
        _ => unreachable!("SortableKey key widths are 1, 2, 4, or 8 bytes"),
    }
}

/// Byte-digit LSD passes with stack histograms sized to the pass count.
///
/// # Safety
///
/// See [`sort_raw`].
unsafe fn sort_core_bytes<T: SortableKey, const PASSES: usize>(slice: &mut [T], scratch: *mut T) {
    let len = slice.len();
    debug_assert_eq!(PASSES, T::Key::BYTES);

    let hist = if len >= INTERLEAVE_MIN {
        histogram::compute_bytes::<T, PASSES, 4>(slice)
    } else {
        histogram::compute_bytes::<T, PASSES, 1>(slice)
    };

    let mut in_scratch = false;
    for (pass, h) in hist.iter().enumerate() {
        if histogram::is_pass_trivial(h, len) {
            continue;
        }
        let mut offsets = *h;
        prefix_sum::exclusive_prefix_sum(&mut offsets);

        let (src, dst) = if in_scratch {
            (scratch as *const T, slice.as_mut_ptr())
        } else {
            (slice.as_ptr(), scratch)
        };
        // SAFETY: offsets is the exclusive prefix sum of this pass's histogram
        // of src[..len]; src/dst validity comes from the sort_raw contract.
        scatter::scatter_pass(src, dst, len, &mut offsets, pass);
        in_scratch = !in_scratch;
    }

    if in_scratch {
        core::ptr::copy_nonoverlapping(scratch as *const T, slice.as_mut_ptr(), len);
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    // The dispatch thresholds (WIDE_MIN, INTERLEAVE_MIN) pick a core by input
    // size, but every core must be correct at any size — these call the cores
    // directly so small, miri-friendly inputs cover them all.

    #[test]
    fn counting_sort_u8_all_values() {
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut data: Vec<u8> = (0..2000).map(|_| xorshift(&mut state) as u8).collect();
        let mut expected = data.clone();
        expected.sort_unstable();
        counting_sort(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn wide_core_u64_small_input() {
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut data: Vec<u64> = (0..2000).map(|_| xorshift(&mut state)).collect();
        let mut expected = data.clone();
        expected.sort_unstable();
        let mut scratch: Vec<u64> = Vec::with_capacity(data.len());
        unsafe { sort_core_wide::<u64, WIDE_BINS, WIDE_PASSES>(&mut data, scratch.as_mut_ptr()) };
        assert_eq!(data, expected);
    }

    #[test]
    fn byte_core_u64_interleaved_histogram() {
        // Above INTERLEAVE_MIN so the 4-copy histogram fold runs.
        let mut state = 0xDA942042E4DD58B5u64;
        let mut data: Vec<u64> = (0..INTERLEAVE_MIN + 17)
            .map(|_| xorshift(&mut state))
            .collect();
        let mut expected = data.clone();
        expected.sort_unstable();
        let mut scratch: Vec<u64> = Vec::with_capacity(data.len());
        unsafe { sort_core_bytes::<u64, 8>(&mut data, scratch.as_mut_ptr()) };
        assert_eq!(data, expected);
    }
}

/// Wide-digit LSD passes with heap histograms; fewer passes over the data.
///
/// # Safety
///
/// See [`sort_raw`].
#[cfg(feature = "alloc")]
unsafe fn sort_core_wide<T: SortableKey, const BINS: usize, const PASSES: usize>(
    slice: &mut [T],
    scratch: *mut T,
) {
    let len = slice.len();
    let bits = BINS.trailing_zeros();
    let mask = BINS - 1;

    let mut hist = alloc::vec![0usize; PASSES * BINS];
    for elem in slice.iter() {
        let key = elem.to_radix_key();
        for pass in 0..PASSES {
            hist[pass * BINS + key.wide_digit(pass as u32 * bits, mask)] += 1;
        }
    }

    let mut offsets = alloc::vec![0usize; BINS];
    let mut in_scratch = false;
    for pass in 0..PASSES {
        let h = &hist[pass * BINS..(pass + 1) * BINS];
        if histogram::is_pass_trivial(h, len) {
            continue;
        }
        offsets.copy_from_slice(h);
        prefix_sum::exclusive_prefix_sum(&mut offsets);

        let (src, dst) = if in_scratch {
            (scratch as *const T, slice.as_mut_ptr())
        } else {
            (slice.as_ptr(), scratch)
        };
        // SAFETY: offsets is the exclusive prefix sum of this pass's histogram
        // of src[..len]; src/dst validity comes from the sort_raw contract.
        scatter::scatter_pass_wide(src, dst, len, &mut offsets, pass as u32 * bits, mask);
        in_scratch = !in_scratch;
    }

    if in_scratch {
        core::ptr::copy_nonoverlapping(scratch as *const T, slice.as_mut_ptr(), len);
    }
}
