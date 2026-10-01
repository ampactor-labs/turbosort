//! AVX2 sorting networks and bitonic merges for 4-byte keys.
//!
//! All functions are `#[target_feature(enable = "avx2")]` and must only be
//! called after runtime CPUID verification via `is_x86_feature_detected!("avx2")`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::key::SortableKey;

// ============================================================================
// Sorting networks (n ≤ 16)
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
// Multi-register bitonic networks (32/64/128 elements)
//
// Each sorted run occupies consecutive registers. Merging two adjacent runs:
// lane-reverse the second run (register order and lanes), which makes the
// combined span bitonic, then half-clean with min/max at register distance
// and recurse into halves down to the single-register merge.
// ============================================================================

/// Merge a bitonic 16-element span (2 registers) into ascending order.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn merge_bitonic_2(r: &mut [__m256i; 2]) {
    let lo = _mm256_min_epu32(r[0], r[1]);
    let hi = _mm256_max_epu32(r[0], r[1]);
    r[0] = lo;
    r[1] = hi;
    merge_within_register(&mut r[0]);
    merge_within_register(&mut r[1]);
}

/// Merge a bitonic 32-element span (4 registers) into ascending order.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn merge_bitonic_4(r: &mut [__m256i; 4]) {
    for i in 0..2 {
        let lo = _mm256_min_epu32(r[i], r[i + 2]);
        let hi = _mm256_max_epu32(r[i], r[i + 2]);
        r[i] = lo;
        r[i + 2] = hi;
    }
    let mut a = [r[0], r[1]];
    let mut b = [r[2], r[3]];
    merge_bitonic_2(&mut a);
    merge_bitonic_2(&mut b);
    *r = [a[0], a[1], b[0], b[1]];
}

/// Merge a bitonic 64-element span (8 registers) into ascending order.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn merge_bitonic_8(r: &mut [__m256i; 8]) {
    for i in 0..4 {
        let lo = _mm256_min_epu32(r[i], r[i + 4]);
        let hi = _mm256_max_epu32(r[i], r[i + 4]);
        r[i] = lo;
        r[i + 4] = hi;
    }
    let mut a = [r[0], r[1], r[2], r[3]];
    let mut b = [r[4], r[5], r[6], r[7]];
    merge_bitonic_4(&mut a);
    merge_bitonic_4(&mut b);
    *r = [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]];
}

/// Reverse a sorted run of `K` registers in place: lane-reverse each register
/// and reverse the register order.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn reverse_run<const K: usize>(run: &mut [__m256i; K]) {
    for i in 0..K / 2 {
        let a = reverse_u32(run[i]);
        run[i] = reverse_u32(run[K - 1 - i]);
        run[K - 1 - i] = a;
    }
    if K % 2 == 1 {
        run[K / 2] = reverse_u32(run[K / 2]);
    }
}

/// Sort 4 registers (32 elements): sort each, merge pairs, merge the halves.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sort_4regs(r: &mut [__m256i; 4]) {
    for reg in r.iter_mut() {
        *reg = sort_network_8(*reg);
    }
    let (mut r0, mut r1, mut r2, mut r3) = (r[0], r[1], r[2], r[3]);
    bitonic_merge_8x2(&mut r0, &mut r1);
    bitonic_merge_8x2(&mut r2, &mut r3);
    *r = [r0, r1, r2, r3];

    // Merge the two sorted 16-runs: reverse the second, half-clean, recurse.
    let mut hi_run = [r[2], r[3]];
    reverse_run(&mut hi_run);
    let mut lo = [r[0], r[1]];
    let l0 = _mm256_min_epu32(lo[0], hi_run[0]);
    let l1 = _mm256_min_epu32(lo[1], hi_run[1]);
    let h0 = _mm256_max_epu32(lo[0], hi_run[0]);
    let h1 = _mm256_max_epu32(lo[1], hi_run[1]);
    lo = [l0, l1];
    let mut hi = [h0, h1];
    merge_bitonic_2(&mut lo);
    merge_bitonic_2(&mut hi);
    *r = [lo[0], lo[1], hi[0], hi[1]];
}

/// Sort 8 registers (64 elements).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sort_8regs(r: &mut [__m256i; 8]) {
    let (a, b) = r.split_at_mut(4);
    let a: &mut [__m256i; 4] = a.try_into().unwrap();
    let b: &mut [__m256i; 4] = b.try_into().unwrap();
    sort_4regs(a);
    sort_4regs(b);
    reverse_run(b);

    let mut lo = [_mm256_setzero_si256(); 4];
    let mut hi = [_mm256_setzero_si256(); 4];
    for i in 0..4 {
        lo[i] = _mm256_min_epu32(a[i], b[i]);
        hi[i] = _mm256_max_epu32(a[i], b[i]);
    }
    merge_bitonic_4(&mut lo);
    merge_bitonic_4(&mut hi);
    *a = lo;
    *b = hi;
}

/// Sort 16 registers (128 elements).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sort_16regs(r: &mut [__m256i; 16]) {
    let (a, b) = r.split_at_mut(8);
    let a: &mut [__m256i; 8] = a.try_into().unwrap();
    let b: &mut [__m256i; 8] = b.try_into().unwrap();
    sort_8regs(a);
    sort_8regs(b);
    reverse_run(b);

    let mut lo = [_mm256_setzero_si256(); 8];
    let mut hi = [_mm256_setzero_si256(); 8];
    for i in 0..8 {
        lo[i] = _mm256_min_epu32(a[i], b[i]);
        hi[i] = _mm256_max_epu32(a[i], b[i]);
    }
    merge_bitonic_8(&mut lo);
    merge_bitonic_8(&mut hi);
    *a = lo;
    *b = hi;
}

/// Sort up to 128 u32 elements with padded bitonic networks.
///
/// Pads to the next network tier (16/32/64/128) with `u32::MAX`, which sorts
/// past every real element and is discarded on the copy back.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub unsafe fn sort_u32_128(slice: &mut [u32]) {
    let len = slice.len();
    debug_assert!(len <= 128);

    if len <= 16 {
        sort_u32_16(slice);
        return;
    }

    let mut buf = [u32::MAX; 128];
    buf[..len].copy_from_slice(slice);
    let p = buf.as_mut_ptr() as *mut __m256i;

    if len <= 32 {
        let mut r = [_mm256_setzero_si256(); 4];
        for (i, reg) in r.iter_mut().enumerate() {
            *reg = _mm256_loadu_si256(p.add(i));
        }
        sort_4regs(&mut r);
        for (i, reg) in r.iter().enumerate() {
            _mm256_storeu_si256(p.add(i), *reg);
        }
    } else if len <= 64 {
        let mut r = [_mm256_setzero_si256(); 8];
        for (i, reg) in r.iter_mut().enumerate() {
            *reg = _mm256_loadu_si256(p.add(i));
        }
        sort_8regs(&mut r);
        for (i, reg) in r.iter().enumerate() {
            _mm256_storeu_si256(p.add(i), *reg);
        }
    } else {
        let mut r = [_mm256_setzero_si256(); 16];
        for (i, reg) in r.iter_mut().enumerate() {
            *reg = _mm256_loadu_si256(p.add(i));
        }
        sort_16regs(&mut r);
        for (i, reg) in r.iter().enumerate() {
            _mm256_storeu_si256(p.add(i), *reg);
        }
    }

    slice.copy_from_slice(&buf[..len]);
}

// ============================================================================
// Block merges (129-512 elements)
//
// Sort 128-key blocks with the register networks, then merge the sorted
// blocks pairwise. Like the networks, nothing here depends on the data's
// order, so every input pattern costs the same.
// ============================================================================

/// Load `run[i..i + 8]` (bounds-checked) into a register.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_8(run: &[u32], i: usize) -> __m256i {
    _mm256_loadu_si256(run[i..i + 8].as_ptr() as *const __m256i)
}

/// Merge two sorted runs into `out` with an 8-wide bitonic merge.
///
/// Both run lengths must be non-zero multiples of 8. `hi` always holds the
/// 8 largest keys loaded so far; each step merges it with the next 8 keys
/// from whichever run has the smaller head and emits the lower 8, which no
/// unloaded key can undercut.
#[target_feature(enable = "avx2")]
unsafe fn merge_runs(a: &[u32], b: &[u32], out: &mut [u32]) {
    debug_assert!(!a.is_empty() && a.len() % 8 == 0);
    debug_assert!(!b.is_empty() && b.len() % 8 == 0);
    debug_assert_eq!(out.len(), a.len() + b.len());
    let mut lo = load_8(a, 0);
    let mut hi = load_8(b, 0);
    let (mut ia, mut ib, mut o) = (8, 8, 0);
    loop {
        bitonic_merge_8x2(&mut lo, &mut hi);
        _mm256_storeu_si256(out[o..o + 8].as_mut_ptr() as *mut __m256i, lo);
        o += 8;
        let take_a = if ia < a.len() {
            ib == b.len() || a[ia] <= b[ib]
        } else if ib < b.len() {
            false
        } else {
            break;
        };
        if take_a {
            lo = load_8(a, ia);
            ia += 8;
        } else {
            lo = load_8(b, ib);
            ib += 8;
        }
    }
    _mm256_storeu_si256(out[o..o + 8].as_mut_ptr() as *mut __m256i, hi);
}

/// Sort 129..=512 keys whose count is a multiple of 8: 128-key blocks with
/// the register networks, then pairwise merges through a stack buffer.
#[target_feature(enable = "avx2")]
unsafe fn sort_u32_512(keys: &mut [u32]) {
    let n = keys.len();
    debug_assert!(n > 128 && n <= 512 && n % 8 == 0);
    for block in keys.chunks_mut(128) {
        sort_u32_128(block);
    }
    let mut tmp = [0u32; 512];
    if n <= 256 {
        merge_runs(&keys[..128], &keys[128..], &mut tmp[..n]);
        keys.copy_from_slice(&tmp[..n]);
    } else {
        merge_runs(&keys[..128], &keys[128..256], &mut tmp[..256]);
        if n <= 384 {
            tmp[256..n].copy_from_slice(&keys[256..]);
        } else {
            merge_runs(&keys[256..384], &keys[384..], &mut tmp[256..n]);
        }
        merge_runs(&tmp[..256], &tmp[256..n], keys);
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

/// Copy the radix keys of `slice` into the front of `keys`.
///
/// # Safety
///
/// `T::Key` must be 4 bytes.
#[inline(always)]
unsafe fn load_keys<T: SortableKey>(slice: &[T], keys: &mut [u32]) {
    debug_assert!(keys.len() >= slice.len());
    for (k, elem) in keys.iter_mut().zip(slice) {
        let key = elem.to_radix_key();
        // SAFETY: T::Key is 4 bytes (caller contract), the same size as u32,
        // so this reinterprets the key's bytes for SIMD sorting.
        *k = core::ptr::read(&key as *const T::Key as *const u32);
    }
}

/// Write the front of `keys` back into `slice` as values of `T`.
///
/// # Safety
///
/// `T::Key` must be 4 bytes, and every copied key must have come from
/// [`load_keys`] (sorting only permutes them).
#[inline(always)]
unsafe fn store_keys<T: SortableKey>(keys: &[u32], slice: &mut [T]) {
    for (elem, k) in slice.iter_mut().zip(keys) {
        // SAFETY: the u32 was produced by to_radix_key, so it is a valid
        // T::Key representation.
        let key = core::ptr::read(k as *const u32 as *const T::Key);
        *elem = T::from_radix_key(key);
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
    debug_assert!(len <= 16);
    if len <= 1 {
        return;
    }

    let mut keys = [0u32; 16];
    load_keys(slice, &mut keys);
    sort_u32_16(&mut keys[..len]);
    store_keys(&keys, slice);
}

/// Sort 17..=512 elements of any 4-byte SortableKey type using AVX2.
///
/// Up to 128 elements run the padded register networks; larger slices sort
/// 128-element blocks and merge them. No pivots, so the cost does not depend
/// on the input's order.
///
/// # Safety
///
/// Caller must ensure AVX2 is available and `T`/`T::Key` are 4 bytes.
#[target_feature(enable = "avx2")]
pub unsafe fn sort_small_u32_keys_generic<T: SortableKey>(slice: &mut [T]) {
    let len = slice.len();
    debug_assert!(core::mem::size_of::<T>() == 4);
    debug_assert!(core::mem::size_of::<T::Key>() == 4);
    debug_assert!(len <= 512);

    if len <= 128 {
        let mut keys = [0u32; 128];
        load_keys(slice, &mut keys);
        sort_u32_128(&mut keys[..len]);
        store_keys(&keys, slice);
    } else {
        // Pad to a multiple of 8 for the merges. Padding keys sort after
        // every real key, so they end up past `len` and are never copied out.
        let mut keys = [u32::MAX; 512];
        load_keys(slice, &mut keys);
        sort_u32_512(&mut keys[..len.next_multiple_of(8)]);
        store_keys(&keys, slice);
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

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn sort_128_all_lengths_random() {
        if !is_available() {
            return;
        }
        let mut state = 0x853C49E6748FEA9Bu64;
        for n in 17..=128usize {
            for _ in 0..20 {
                let mut data: Vec<u32> = (0..n).map(|_| xorshift(&mut state) as u32).collect();
                let mut expected = data.clone();
                expected.sort_unstable();
                unsafe { sort_u32_128(&mut data) };
                assert_eq!(data, expected, "n={n}");
            }
        }
    }

    #[test]
    fn sort_128_zero_one_patterns() {
        // Randomized 0-1 principle sampling across every network tier.
        if !is_available() {
            return;
        }
        let mut state = 0xC0FFEE123456789u64;
        for n in [17, 31, 32, 33, 63, 64, 65, 100, 127, 128] {
            for _ in 0..500 {
                let mut data: Vec<u32> =
                    (0..n).map(|_| (xorshift(&mut state) & 1) as u32).collect();
                let mut expected = data.clone();
                expected.sort_unstable();
                unsafe { sort_u32_128(&mut data) };
                assert_eq!(data, expected, "n={n}");
            }
        }
    }

    #[test]
    fn sort_128_structured_patterns() {
        if !is_available() {
            return;
        }
        for n in [17usize, 32, 33, 64, 65, 128] {
            let cases: [Vec<u32>; 4] = [
                (0..n as u32).collect(),
                (0..n as u32).rev().collect(),
                vec![7; n],
                (0..n).map(|i| (i % 3) as u32).collect(),
            ];
            for case in cases {
                let mut data = case.clone();
                let mut expected = case;
                expected.sort_unstable();
                unsafe { sort_u32_128(&mut data) };
                assert_eq!(data, expected, "n={n}");
            }
        }
    }
    #[test]
    fn merge_runs_all_run_lengths() {
        if !is_available() {
            return;
        }
        let mut state = 0x2545F4914F6CDD1Du64;
        for la in (8..=96).step_by(8) {
            for lb in (8..=96).step_by(8) {
                // Narrow value range so equal keys straddle the two runs.
                let mut a: Vec<u32> = (0..la)
                    .map(|_| (xorshift(&mut state) % 64) as u32)
                    .collect();
                let mut b: Vec<u32> = (0..lb)
                    .map(|_| (xorshift(&mut state) % 64) as u32)
                    .collect();
                a.sort_unstable();
                b.sort_unstable();
                let mut expected = [a.clone(), b.clone()].concat();
                expected.sort_unstable();
                let mut out = vec![0u32; la + lb];
                unsafe { merge_runs(&a, &b, &mut out) };
                assert_eq!(out, expected, "la={la} lb={lb}");
            }
        }
    }

    /// The 129..=512 tier through the public-facing generic entry point.
    fn check_small(data: Vec<u32>) {
        let mut expected = data.clone();
        expected.sort_unstable();
        let mut sorted = data;
        unsafe { sort_small_u32_keys_generic(&mut sorted) };
        assert_eq!(sorted, expected, "n={}", sorted.len());
    }

    #[test]
    fn sort_512_all_lengths_random() {
        if !is_available() {
            return;
        }
        let mut state = 0x9E3779B97F4A7C15u64;
        for n in 129..=512usize {
            for _ in 0..3 {
                check_small((0..n).map(|_| xorshift(&mut state) as u32).collect());
            }
        }
    }

    #[test]
    fn sort_512_zero_one_patterns() {
        if !is_available() {
            return;
        }
        let mut state = 0xDA942042E4DD58B5u64;
        for n in [129, 135, 136, 255, 256, 257, 384, 385, 511, 512] {
            for _ in 0..200 {
                check_small((0..n).map(|_| (xorshift(&mut state) & 1) as u32).collect());
            }
        }
    }

    #[test]
    fn sort_512_structured_patterns() {
        if !is_available() {
            return;
        }
        for n in [129usize, 200, 256, 300, 384, 448, 511, 512] {
            let half = n / 2;
            let cases: [Vec<u32>; 8] = [
                (0..n as u32).collect(),
                (0..n as u32).rev().collect(),
                vec![u32::MAX; n],
                (0..n).map(|i| (i % 4) as u32).collect(),
                (0..half)
                    .chain((0..n - half).rev())
                    .map(|x| x as u32)
                    .collect(),
                (0..n).map(|i| (i % (n / 8)) as u32).collect(),
                (0..n)
                    .map(|i| {
                        if i == 0 || i == n - 1 {
                            0
                        } else {
                            1 + i as u32
                        }
                    })
                    .collect(),
                (0..n).map(|i| u32::MAX - (i % 3) as u32).collect(),
            ];
            for case in cases {
                check_small(case);
            }
        }
    }
}
