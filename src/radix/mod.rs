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
//! - Both cores divert ([`divert`]): when the top digits alone carry enough
//!   bits to nearly order the keys, the low-digit passes are replaced by one
//!   scan that sorts the short runs left over.
//!
//! The scratch buffer never needs to be initialized: every scatter pass
//! writes all `len` positions of its destination.

pub mod divert;
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
    if slice.len() <= 1 {
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
    if slice.len() <= 1 {
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

/// Counting sort for one-byte keys: histogram, then rewrite from counts.
///
/// Needs no scratch buffer, so it also serves `no_std` builds without
/// `alloc`. Large inputs count into four interleaved histograms, like the
/// radix passes, so runs of equal bytes do not serialize on one counter.
pub(crate) fn counting_sort<T: SortableKey>(slice: &mut [T]) {
    debug_assert_eq!(T::Key::BYTES, 1);
    let [counts] = if slice.len() >= INTERLEAVE_MIN {
        histogram::compute_bytes::<T, 1, 4>(slice)
    } else {
        histogram::compute_bytes::<T, 1, 1>(slice)
    };
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

    // Bytes that never vary need no histogram: keys holding small values in a
    // wide type often vary in only one or two. The scan that finds them stops
    // within the first block when every byte varies.
    let varying = histogram::varying_bits(slice);
    let mut counted = [0usize; PASSES];
    let mut n_counted = 0;
    for pass in 0..PASSES {
        if (varying >> (8 * pass)) & 0xFF != 0 {
            counted[n_counted] = pass;
            n_counted += 1;
        }
    }
    let counted = &counted[..n_counted];
    let hist = match (n_counted == PASSES, len >= INTERLEAVE_MIN) {
        (true, true) => histogram::compute_bytes::<T, PASSES, 4>(slice),
        (true, false) => histogram::compute_bytes::<T, PASSES, 1>(slice),
        (false, true) => histogram::compute_bytes_live::<T, PASSES, 4>(slice, counted),
        (false, false) => histogram::compute_bytes_live::<T, PASSES, 1>(slice, counted),
    };

    // Live passes: digits that vary. A digit is constant exactly when the
    // first key's bucket holds every key, and its scatter would be a no-op.
    let first = slice[0].to_radix_key();
    let mut live = [0usize; PASSES];
    let mut n_live = 0;
    for (pass, h) in hist.iter().enumerate() {
        if h[first.radix_digit(pass) as usize] < len {
            live[n_live] = pass;
            n_live += 1;
        }
    }
    let skip = divert::skippable_digits(n_live, len, |i| {
        hist[live[i]].iter().copied().max().unwrap_or(0)
    });

    let mut in_scratch = false;
    for &pass in &live[skip..n_live] {
        let mut offsets = hist[pass];
        prefix_sum::exclusive_prefix_sum(&mut offsets);

        let (src, dst) = if in_scratch {
            (scratch as *const T, slice.as_mut_ptr())
        } else {
            (slice.as_ptr(), scratch)
        };
        // SAFETY: offsets is the exclusive prefix sum of this pass's histogram
        // of src[..len]; src/dst validity comes from the sort_raw contract.
        if scatter::wants_combining(&hist[pass], len) {
            scatter::scatter_pass_combined(src, dst, len, &offsets, pass);
        } else {
            scatter::scatter_pass(src, dst, len, &mut offsets, pass);
        }
        in_scratch = !in_scratch;
    }

    if in_scratch {
        core::ptr::copy_nonoverlapping(scratch as *const T, slice.as_mut_ptr(), len);
    }
    if skip > 0 {
        divert::finish_runs(slice, 8 * live[skip] as u32);
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
    fn counting_sort_i8_interleaved_runs() {
        // Above INTERLEAVE_MIN, with long runs of one value (the case the
        // interleaving exists for) and every i8 value present.
        let mut data: Vec<i8> = (0..INTERLEAVE_MIN as i32 * 2)
            .map(|i| ((i / 97) % 256 - 128) as i8)
            .collect();
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

    /// Keys whose three top bytes are one random byte repeated: the
    /// histograms promise 24 bits, the keys carry 8, so diverting leaves
    /// runs of about len/256 for the finishing sort.
    fn correlated_top_bytes(len: usize, state: &mut u64) -> Vec<u64> {
        (0..len)
            .map(|_| {
                let b = xorshift(state) & 0xFF;
                b << 56 | b << 48 | b << 40 | xorshift(state) >> 24
            })
            .collect()
    }

    fn check_cores(data: Vec<u64>) {
        let mut expected = data.clone();
        expected.sort_unstable();
        let mut scratch: Vec<u64> = Vec::with_capacity(data.len());

        let mut bytes = data.clone();
        unsafe { sort_core_bytes::<u64, 8>(&mut bytes, scratch.as_mut_ptr()) };
        assert_eq!(bytes, expected, "byte core, len {}", data.len());

        let mut wide = data;
        unsafe { sort_core_wide::<u64, WIDE_BINS, WIDE_PASSES>(&mut wide, scratch.as_mut_ptr()) };
        assert_eq!(wide, expected, "wide core, len {}", wide.len());
    }

    #[test]
    fn cores_divert_on_random_keys() {
        let mut state = 0x853C49E6748FEA9Bu64;
        // Short runs: the finishing pass is mostly insertion sorts of 1-2 keys.
        check_cores((0..3000).map(|_| xorshift(&mut state)).collect());
        // Signed keys go through the key transform both ways.
        let mut data: Vec<i64> = (0..3000).map(|_| xorshift(&mut state) as i64).collect();
        let mut expected = data.clone();
        expected.sort_unstable();
        let mut scratch: Vec<i64> = Vec::with_capacity(data.len());
        unsafe { sort_core_bytes::<i64, 8>(&mut data, scratch.as_mut_ptr()) };
        assert_eq!(data, expected);
    }

    #[test]
    fn cores_divert_with_long_runs() {
        let mut state = 0xC0FFEE123456789u64;
        check_cores(correlated_top_bytes(5000, &mut state));
        // Duplicates: runs of identical keys.
        check_cores(
            (0..5000)
                .map(|_| (xorshift(&mut state) % 300).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                .collect(),
        );
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

    // As in the byte core, count only the digits that vary.
    let varying = histogram::varying_bits(slice);
    let mut counted = [0usize; PASSES];
    let mut n_counted = 0;
    for pass in 0..PASSES {
        if (varying >> (pass as u32 * bits)) & mask as u64 != 0 {
            counted[n_counted] = pass;
            n_counted += 1;
        }
    }
    let first = slice[0].to_radix_key();
    let mut hist = alloc::vec![0usize; PASSES * BINS];
    if n_counted == PASSES {
        for elem in slice.iter() {
            let key = elem.to_radix_key();
            for pass in 0..PASSES {
                hist[pass * BINS + key.wide_digit(pass as u32 * bits, mask)] += 1;
            }
        }
    } else {
        for elem in slice.iter() {
            let key = elem.to_radix_key();
            for &pass in &counted[..n_counted] {
                hist[pass * BINS + key.wide_digit(pass as u32 * bits, mask)] += 1;
            }
        }
        for pass in 0..PASSES {
            if !counted[..n_counted].contains(&pass) {
                hist[pass * BINS + first.wide_digit(pass as u32 * bits, mask)] = len;
            }
        }
    }

    let mut live = [0usize; PASSES];
    let mut n_live = 0;
    for pass in 0..PASSES {
        if hist[pass * BINS + first.wide_digit(pass as u32 * bits, mask)] < len {
            live[n_live] = pass;
            n_live += 1;
        }
    }
    let skip = divert::skippable_digits(n_live, len, |i| {
        let pass = live[i];
        hist[pass * BINS..(pass + 1) * BINS]
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
    });

    let mut offsets = alloc::vec![0usize; BINS];
    let mut in_scratch = false;
    for &pass in &live[skip..n_live] {
        let h = &hist[pass * BINS..(pass + 1) * BINS];
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
    if skip > 0 {
        divert::finish_runs(slice, live[skip] as u32 * bits);
    }
}
