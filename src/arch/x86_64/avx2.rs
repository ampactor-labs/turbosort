//! AVX2 SIMD implementations for sorting networks, partition, and histogram.
//!
//! All functions are `#[target_feature(enable = "avx2")]` and must only be
//! called after runtime CPUID verification via `is_x86_feature_detected!("avx2")`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::key::SortableKey;

use super::partition::partition_u32;

// ============================================================================
// Phase 3: Sorting Networks (n ≤ 16)
// ============================================================================

/// Reverse the lanes within a 256-bit vector of 32-bit elements.
/// [0,1,2,3,4,5,6,7] → [7,6,5,4,3,2,1,0]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn reverse_u32(v: __m256i) -> __m256i {
    let shuffled = _mm256_shuffle_epi32::<0b00_01_10_11>(v);
    _mm256_permute2x128_si256::<0x01>(shuffled, shuffled)
}

/// Bitonic merge network for two sorted 8-element vectors.
///
/// Assumes `lo` is sorted ascending and `hi` is sorted ascending.
/// Produces a fully sorted 16-element sequence across `lo` (lower 8) and `hi` (upper 8).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn bitonic_merge_8x2(lo: &mut __m256i, hi: &mut __m256i) {
    let hi_rev = reverse_u32(*hi);
    let new_lo = _mm256_min_epu32(*lo, hi_rev);
    let new_hi = _mm256_max_epu32(*lo, hi_rev);
    *lo = new_lo;
    *hi = new_hi;

    merge_within_register(lo);
    merge_within_register(hi);
}

/// Ascending merge network within a single 8-element register.
///
/// Assumes the register contains a bitonic sequence (first half ascending,
/// second half descending or vice versa) and produces an ascending sorted result.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn merge_within_register(v: &mut __m256i) {
    // Distance-4 (cross 128-bit lane boundary)
    {
        let swapped = _mm256_permute2x128_si256::<0x01>(*v, *v);
        let lo = _mm256_min_epu32(*v, swapped);
        let hi = _mm256_max_epu32(*v, swapped);
        *v = _mm256_blend_epi32::<0xF0>(lo, hi);
    }
    // Distance-2
    {
        let shuffled = _mm256_shuffle_epi32::<0b01_00_11_10>(*v);
        let lo = _mm256_min_epu32(*v, shuffled);
        let hi = _mm256_max_epu32(*v, shuffled);
        *v = _mm256_blend_epi32::<0b1100_1100>(lo, hi);
    }
    // Distance-1
    {
        let shuffled = _mm256_shuffle_epi32::<0b10_11_00_01>(*v);
        let lo = _mm256_min_epu32(*v, shuffled);
        let hi = _mm256_max_epu32(*v, shuffled);
        *v = _mm256_blend_epi32::<0b1010_1010>(lo, hi);
    }
}

/// Sort exactly 8 u32 elements using Batcher's odd-even mergesort network.
///
/// 6 levels, 19 comparators. Each level maps to one SIMD shuffle + min/max + blend.
///
/// ```text
/// L1: (0,1) (2,3) (4,5) (6,7)
/// L2: (0,2) (1,3) (4,6) (5,7)
/// L3: (1,2) (5,6)
/// L4: (0,4) (1,5) (2,6) (3,7)
/// L5: (2,4) (3,5)
/// L6: (1,2) (3,4) (5,6)
/// ```
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sort_network_8(v: __m256i) -> __m256i {
    let mut r = v;

    // L1: CAS (0,1),(2,3),(4,5),(6,7)
    {
        let s = _mm256_shuffle_epi32::<0b10_11_00_01>(r); // [1,0,3,2,5,4,7,6]
        let lo = _mm256_min_epu32(r, s);
        let hi = _mm256_max_epu32(r, s);
        r = _mm256_blend_epi32::<0b1010_1010>(lo, hi); // even=min, odd=max
    }

    // L2: CAS (0,2),(1,3),(4,6),(5,7)
    {
        let s = _mm256_shuffle_epi32::<0b01_00_11_10>(r); // [2,3,0,1,6,7,4,5]
        let lo = _mm256_min_epu32(r, s);
        let hi = _mm256_max_epu32(r, s);
        r = _mm256_blend_epi32::<0b1100_1100>(lo, hi); // 0,1=min; 2,3=max
    }

    // L3: CAS (1,2),(5,6) — fixup within quads
    {
        let s = _mm256_shuffle_epi32::<0b11_01_10_00>(r); // [0,2,1,3,4,6,5,7]
        let lo = _mm256_min_epu32(r, s);
        let hi = _mm256_max_epu32(r, s);
        r = _mm256_blend_epi32::<0b0100_0100>(lo, hi); // hi at positions 2,6
    }

    // L4: CAS (0,4),(1,5),(2,6),(3,7) — cross 128-bit lane merge
    {
        let s = _mm256_permute2x128_si256::<0x01>(r, r); // swap halves
        let lo = _mm256_min_epu32(r, s);
        let hi = _mm256_max_epu32(r, s);
        r = _mm256_blend_epi32::<0xF0>(lo, hi); // 0-3=min, 4-7=max
    }

    // L5: CAS (2,4),(3,5) — cross-lane fixup
    {
        let perm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
        let s = _mm256_permutevar8x32_epi32(r, perm);
        let lo = _mm256_min_epu32(r, s);
        let hi = _mm256_max_epu32(r, s);
        r = _mm256_blend_epi32::<0b0011_0000>(lo, hi); // hi at positions 4,5
    }

    // L6: CAS (1,2),(3,4),(5,6) — final cleanup
    {
        let perm = _mm256_setr_epi32(0, 2, 1, 4, 3, 6, 5, 7);
        let s = _mm256_permutevar8x32_epi32(r, perm);
        let lo = _mm256_min_epu32(r, s);
        let hi = _mm256_max_epu32(r, s);
        r = _mm256_blend_epi32::<0b0101_0100>(lo, hi); // hi at positions 2,4,6
    }

    r
}

/// Sort up to 16 u32 elements using AVX2 sorting networks.
///
/// # Safety
///
/// Caller must ensure AVX2 is available (checked via `is_x86_feature_detected!`).
#[target_feature(enable = "avx2")]
pub unsafe fn sort_u32_16(slice: &mut [u32]) {
    let len = slice.len();
    debug_assert!(len <= 16);

    if len <= 1 {
        return;
    }

    if len <= 8 {
        let mut buf = [u32::MAX; 8];
        buf[..len].copy_from_slice(slice);
        let v = _mm256_loadu_si256(buf.as_ptr() as *const __m256i);
        let sorted = sort_network_8(v);
        _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, sorted);
        slice.copy_from_slice(&buf[..len]);
    } else {
        let mut buf_lo = [u32::MAX; 8];
        let mut buf_hi = [u32::MAX; 8];
        buf_lo.copy_from_slice(&slice[..8]);
        let hi_len = len - 8;
        buf_hi[..hi_len].copy_from_slice(&slice[8..]);

        let mut lo = sort_network_8(_mm256_loadu_si256(buf_lo.as_ptr() as *const __m256i));
        let mut hi = sort_network_8(_mm256_loadu_si256(buf_hi.as_ptr() as *const __m256i));
        bitonic_merge_8x2(&mut lo, &mut hi);

        _mm256_storeu_si256(buf_lo.as_mut_ptr() as *mut __m256i, lo);
        _mm256_storeu_si256(buf_hi.as_mut_ptr() as *mut __m256i, hi);
        slice[..8].copy_from_slice(&buf_lo);
        slice[8..].copy_from_slice(&buf_hi[..hi_len]);
    }
}

// ============================================================================
// Dispatch wrappers
// ============================================================================

/// Check if AVX2 is available at runtime.
///
/// Requires `std` for CPUID detection. Returns `false` in `no_std` builds.
#[inline]
pub fn is_available() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "std")))]
    {
        false
    }
}

/// Sort up to 16 elements of any 4-byte SortableKey type using AVX2.
///
/// Converts to u32 radix keys, sorts via SIMD network, converts back.
///
/// # Safety
///
/// Caller must ensure:
/// - AVX2 is available
/// - `T` and `T::Key` are both 4 bytes
#[target_feature(enable = "avx2")]
pub unsafe fn sort_tiny_u32_keys_generic<T: SortableKey>(slice: &mut [T]) {
    let len = slice.len();
    debug_assert!(core::mem::size_of::<T>() == 4);
    debug_assert!(core::mem::size_of::<T::Key>() == 4);
    if len <= 1 {
        return;
    }

    let mut keys: [u32; 16] = [u32::MAX; 16];
    for (i, elem) in slice.iter().enumerate() {
        let key = elem.to_radix_key();
        // SAFETY: T::Key is guaranteed to be 4 bytes (checked by debug_assert above),
        // same as u32, so this reinterprets the key's bytes as u32 for SIMD sorting.
        keys[i] = core::ptr::read(&key as *const T::Key as *const u32);
    }

    sort_u32_16(&mut keys[..len]);

    for (i, elem) in slice.iter_mut().enumerate() {
        // SAFETY: The sorted u32 values are valid T::Key representations (they were
        // originally produced by to_radix_key), so reinterpreting back is sound.
        let key = core::ptr::read(&keys[i] as *const u32 as *const T::Key);
        *elem = T::from_radix_key(key);
    }
}

/// Quicksort for u32 key arrays using SIMD sorting network for leaf nodes.
///
/// Uses AVX2 SIMD partition (Bramas neutralize strategy) for splitting and
/// falls back to the AVX2 sorting network for partitions <= 16.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub unsafe fn quicksort_u32(slice: &mut [u32]) {
    quicksort_u32_impl(slice, 2 * log2_usize(slice.len()));
}

#[target_feature(enable = "avx2")]
unsafe fn quicksort_u32_impl(slice: &mut [u32], depth_limit: usize) {
    let len = slice.len();

    if len <= 16 {
        sort_u32_16(slice);
        return;
    }

    if depth_limit == 0 {
        // Use sorting network on chunks to avoid O(n²)
        for chunk in slice.chunks_mut(16) {
            sort_u32_16(chunk);
        }
        // Then insertion sort to merge the sorted chunks
        for i in 1..len {
            let mut j = i;
            while j > 0 && slice[j - 1] > slice[j] {
                slice.swap(j, j - 1);
                j -= 1;
            }
        }
        return;
    }

    let pivot = median_of_three(slice[0], slice[len / 2], slice[len - 1]);
    let mid = partition_u32(slice, pivot);

    if mid == 0 || mid == len {
        // All elements equal or bad pivot
        for i in 1..len {
            let mut j = i;
            while j > 0 && slice[j - 1] > slice[j] {
                slice.swap(j, j - 1);
                j -= 1;
            }
        }
        return;
    }

    let (left, right) = slice.split_at_mut(mid);
    quicksort_u32_impl(left, depth_limit - 1);
    quicksort_u32_impl(right, depth_limit - 1);
}

#[inline]
fn median_of_three(a: u32, b: u32, c: u32) -> u32 {
    if a <= b {
        if b <= c {
            b
        } else if a <= c {
            c
        } else {
            a
        }
    } else if a <= c {
        a
    } else if b <= c {
        c
    } else {
        b
    }
}

#[inline]
fn log2_usize(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    usize::BITS as usize - 1 - n.leading_zeros() as usize
}

/// Sort 17-512 elements of any 4-byte SortableKey type using SIMD quicksort.
///
/// Converts to u32 keys, sorts via SIMD partition + sorting network, converts back.
///
/// # Safety
///
/// Caller must ensure AVX2 is available and `T`/`T::Key` are 4 bytes.
#[target_feature(enable = "avx2")]
pub unsafe fn quicksort_u32_keys_generic<T: SortableKey>(slice: &mut [T]) {
    let len = slice.len();
    debug_assert!(core::mem::size_of::<T>() == 4);
    debug_assert!(core::mem::size_of::<T::Key>() == 4);

    // Convert to keys in a stack buffer (max 512 × 4 = 2KB)
    let mut keys = [0u32; 512];
    for (i, elem) in slice.iter().enumerate() {
        let key = elem.to_radix_key();
        // SAFETY: T::Key is guaranteed to be 4 bytes (checked by debug_assert above),
        // same as u32, so this reinterprets the key's bytes as u32 for SIMD sorting.
        keys[i] = core::ptr::read(&key as *const T::Key as *const u32);
    }

    quicksort_u32(&mut keys[..len]);

    for (i, elem) in slice.iter_mut().enumerate() {
        // SAFETY: The sorted u32 values are valid T::Key representations (they were
        // originally produced by to_radix_key), so reinterpreting back is sound.
        let key = core::ptr::read(&keys[i] as *const u32 as *const T::Key);
        *elem = T::from_radix_key(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_network_8_basic() {
        if !is_available() {
            return;
        }
        unsafe {
            let mut data = [5u32, 3, 8, 1, 9, 2, 7, 4];
            sort_u32_16(&mut data);
            assert_eq!(data, [1, 2, 3, 4, 5, 7, 8, 9]);
        }
    }

    #[test]
    fn sort_network_8_with_padding() {
        if !is_available() {
            return;
        }
        unsafe {
            let mut data = [5u32, 3, 1];
            sort_u32_16(&mut data);
            assert_eq!(data, [1, 3, 5]);
        }
    }

    #[test]
    fn sort_network_16() {
        if !is_available() {
            return;
        }
        unsafe {
            let mut data = [16u32, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
            sort_u32_16(&mut data);
            assert_eq!(
                data,
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            );
        }
    }

    #[test]
    fn sort_network_all_equal() {
        if !is_available() {
            return;
        }
        unsafe {
            let mut data = [42u32; 8];
            sort_u32_16(&mut data);
            assert_eq!(data, [42; 8]);
        }
    }

    #[test]
    fn sort_network_already_sorted() {
        if !is_available() {
            return;
        }
        unsafe {
            let mut data = [1u32, 2, 3, 4, 5, 6, 7, 8];
            sort_u32_16(&mut data);
            assert_eq!(data, [1, 2, 3, 4, 5, 6, 7, 8]);
        }
    }

    #[test]
    fn sort_network_reversed() {
        if !is_available() {
            return;
        }
        unsafe {
            let mut data = [8u32, 7, 6, 5, 4, 3, 2, 1];
            sort_u32_16(&mut data);
            assert_eq!(data, [1, 2, 3, 4, 5, 6, 7, 8]);
        }
    }
}
