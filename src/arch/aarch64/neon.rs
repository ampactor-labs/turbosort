//! NEON SIMD implementations for aarch64 (Apple Silicon, Graviton).
//!
//! 4-lane (128-bit) sorting networks, composed into 8- and 16-element sorts
//! via bitonic merges. All compare-exchange stages are branch-free vector ops.
//! NEON is mandatory on aarch64, so no runtime check is needed.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::key::SortableKey;

// ============================================================================
// 4-lane compare-exchange stages
// ============================================================================

/// Byte-shuffle indices for the lane permutation [0,2,1,3].
#[cfg(target_arch = "aarch64")]
const PERM_0213: [u8; 16] = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15];

/// CAS (0,1),(2,3): compare adjacent pairs, min to even lanes, max to odd.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn cas_adjacent(r: uint32x4_t) -> uint32x4_t {
    let s = vrev64q_u32(r); // [1,0,3,2]
    let lo = vminq_u32(r, s);
    let hi = vmaxq_u32(r, s);
    // [lo0, hi0, lo2, hi2] = [min01, max01, min23, max23]
    vtrn1q_u32(lo, hi)
}

/// CAS (0,2),(1,3): compare across the register halves.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn cas_distance_2(r: uint32x4_t) -> uint32x4_t {
    let s = vextq_u32(r, r, 2); // [2,3,0,1]
    let lo = vminq_u32(r, s);
    let hi = vmaxq_u32(r, s);
    // lanes 0,1 from lo, lanes 2,3 from hi
    vcombine_u32(vget_low_u32(lo), vget_high_u32(hi))
}

/// CAS (1,2): the final fixup comparator of the 4-element network.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn cas_middle(r: uint32x4_t) -> uint32x4_t {
    let idx = vld1q_u8(PERM_0213.as_ptr());
    let s = vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(r), idx)); // [0,2,1,3]
    let lo = vminq_u32(r, s);
    let hi = vmaxq_u32(r, s);
    // lane 0: min(r0,r0)=r0, lane 1: min12, lane 2: max12, lane 3: max(r3,r3)=r3
    vcombine_u32(vget_low_u32(lo), vget_high_u32(hi))
}

/// Reverse all four lanes: [0,1,2,3] → [3,2,1,0].
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn reverse_4(r: uint32x4_t) -> uint32x4_t {
    let s = vrev64q_u32(r); // [1,0,3,2]
    vextq_u32(s, s, 2)
}

// ============================================================================
// Sorting networks
// ============================================================================

/// Sort exactly 4 u32 elements: Batcher odd-even network, 5 comparators.
///
/// ```text
/// L1: (0,1) (2,3)
/// L2: (0,2) (1,3)
/// L3: (1,2)
/// ```
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn sort_network_4(v: uint32x4_t) -> uint32x4_t {
    cas_middle(cas_distance_2(cas_adjacent(v)))
}

/// Merge a bitonic 4-element sequence into ascending order.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn merge_within_4(v: uint32x4_t) -> uint32x4_t {
    cas_adjacent(cas_distance_2(v))
}

/// Merge two ascending 4-element registers into one ascending 8-sequence
/// spanning `(lo, hi)`.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn merge_sorted_4x2(lo: uint32x4_t, hi: uint32x4_t) -> (uint32x4_t, uint32x4_t) {
    let hi_rev = reverse_4(hi);
    let new_lo = vminq_u32(lo, hi_rev);
    let new_hi = vmaxq_u32(lo, hi_rev);
    (merge_within_4(new_lo), merge_within_4(new_hi))
}

/// Merge a bitonic 8-element sequence spanning `(r0, r1)` into ascending order.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn merge_bitonic_8(r0: uint32x4_t, r1: uint32x4_t) -> (uint32x4_t, uint32x4_t) {
    // Distance-4 half-cleaner across the two registers
    let lo = vminq_u32(r0, r1);
    let hi = vmaxq_u32(r0, r1);
    (merge_within_4(lo), merge_within_4(hi))
}

/// Sort up to 8 u32 elements using NEON sorting networks.
///
/// # Safety
///
/// Only called on aarch64 where NEON is architecturally guaranteed.
#[cfg(target_arch = "aarch64")]
pub unsafe fn sort_u32_8(slice: &mut [u32]) {
    let len = slice.len();
    debug_assert!(len <= 8);
    if len <= 1 {
        return;
    }

    if len <= 4 {
        let mut buf = [u32::MAX; 4];
        buf[..len].copy_from_slice(slice);
        let sorted = sort_network_4(vld1q_u32(buf.as_ptr()));
        vst1q_u32(buf.as_mut_ptr(), sorted);
        slice.copy_from_slice(&buf[..len]);
    } else {
        let mut buf = [u32::MAX; 8];
        buf[..len].copy_from_slice(slice);
        let lo = sort_network_4(vld1q_u32(buf.as_ptr()));
        let hi = sort_network_4(vld1q_u32(buf.as_ptr().add(4)));
        let (lo, hi) = merge_sorted_4x2(lo, hi);
        vst1q_u32(buf.as_mut_ptr(), lo);
        vst1q_u32(buf.as_mut_ptr().add(4), hi);
        slice.copy_from_slice(&buf[..len]);
    }
}

/// Sort up to 16 u32 elements using NEON sorting networks.
///
/// For n ≤ 8, defers to [`sort_u32_8`]. For 9–16, sorts four 4-lane registers
/// and merges them with bitonic half-cleaners.
///
/// # Safety
///
/// Only called on aarch64 where NEON is architecturally guaranteed.
#[cfg(target_arch = "aarch64")]
pub unsafe fn sort_u32_16(slice: &mut [u32]) {
    let len = slice.len();
    debug_assert!(len <= 16);
    if len <= 8 {
        sort_u32_8(slice);
        return;
    }

    let mut buf = [u32::MAX; 16];
    buf[..len].copy_from_slice(slice);

    let a = sort_network_4(vld1q_u32(buf.as_ptr()));
    let b = sort_network_4(vld1q_u32(buf.as_ptr().add(4)));
    let c = sort_network_4(vld1q_u32(buf.as_ptr().add(8)));
    let d = sort_network_4(vld1q_u32(buf.as_ptr().add(12)));

    // Two sorted 8-sequences: (a,b) and (c,d)
    let (a, b) = merge_sorted_4x2(a, b);
    let (c, d) = merge_sorted_4x2(c, d);

    // Half-cleaner across 16: lanes 0-3 pair with 15-12, lanes 4-7 with 11-8
    let rd = reverse_4(d);
    let rc = reverse_4(c);
    let lo0 = vminq_u32(a, rd);
    let lo1 = vminq_u32(b, rc);
    let hi0 = vmaxq_u32(a, rd);
    let hi1 = vmaxq_u32(b, rc);

    let (a, b) = merge_bitonic_8(lo0, lo1);
    let (c, d) = merge_bitonic_8(hi0, hi1);

    vst1q_u32(buf.as_mut_ptr(), a);
    vst1q_u32(buf.as_mut_ptr().add(4), b);
    vst1q_u32(buf.as_mut_ptr().add(8), c);
    vst1q_u32(buf.as_mut_ptr().add(12), d);
    slice.copy_from_slice(&buf[..len]);
}

/// Sort up to 16 elements of any 4-byte SortableKey type using NEON.
///
/// # Safety
///
/// Caller must ensure `T` and `T::Key` are both 4 bytes.
#[cfg(target_arch = "aarch64")]
pub unsafe fn sort_tiny_u32_keys_generic<T: SortableKey>(slice: &mut [T]) {
    let len = slice.len();
    debug_assert!(len <= 16);
    debug_assert!(core::mem::size_of::<T>() == 4);
    debug_assert!(core::mem::size_of::<T::Key>() == 4);
    if len <= 1 {
        return;
    }

    let mut keys: [u32; 16] = [u32::MAX; 16];
    for (i, elem) in slice.iter().enumerate() {
        let key = elem.to_radix_key();
        // SAFETY: T::Key is 4 bytes (debug-asserted), same layout class as u32.
        keys[i] = core::ptr::read(&key as *const T::Key as *const u32);
    }

    sort_u32_16(&mut keys[..len]);

    for (i, elem) in slice.iter_mut().enumerate() {
        // SAFETY: the sorted u32 values were produced by to_radix_key, so they
        // are valid T::Key representations.
        let key = core::ptr::read(&keys[i] as *const u32 as *const T::Key);
        *elem = T::from_radix_key(key);
    }
}

/// NEON is always available on aarch64.
#[inline]
pub fn is_available() -> bool {
    cfg!(target_arch = "aarch64")
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    // 0-1 principle: a comparator network sorts every input iff it sorts
    // every 0-1 input. `zero_one_full` is exhaustive over all 2^n patterns;
    // miri is too slow for that, so it runs a reduced range there.
    fn check_zero_one(n: usize) {
        for pattern in 0u32..(1 << n) {
            let mut data: Vec<u32> = (0..n).map(|i| (pattern >> i) & 1).collect();
            let mut expected = data.clone();
            expected.sort_unstable();
            unsafe { sort_u32_16(&mut data) };
            assert_eq!(data, expected, "n={n} pattern={pattern:#b}");
        }
    }

    #[test]
    fn zero_one_small() {
        for n in 2..=10 {
            check_zero_one(n);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn zero_one_full() {
        for n in 11..=16 {
            check_zero_one(n);
        }
    }

    #[test]
    fn distinct_values_all_lengths() {
        // Catches load/store plumbing bugs (dropped or duplicated elements)
        // that 0-1 inputs cannot distinguish.
        for n in 2..=16usize {
            let mut data: Vec<u32> = (0..n as u32)
                .rev()
                .map(|x| x.wrapping_mul(2654435761))
                .collect();
            let mut expected = data.clone();
            expected.sort_unstable();
            unsafe { sort_u32_16(&mut data) };
            assert_eq!(data, expected, "n={n}");
        }
    }
}
