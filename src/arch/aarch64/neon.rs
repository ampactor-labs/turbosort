//! NEON SIMD implementations for aarch64 (Apple Silicon, Graviton).
//!
//! 4-lane (128-bit) implementations of sorting networks.
//! All functions require NEON, which is mandatory on aarch64 — no runtime check needed.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::key::SortableKey;

// ============================================================================
// Sorting Network (n ≤ 8)
// ============================================================================

/// Sort exactly 4 u32 elements using a NEON sorting network.
///
/// Batcher odd-even mergesort for 4 elements: 5 comparators in 3 levels.
/// ```text
/// L1: (0,1) (2,3)
/// L2: (0,2) (1,3)
/// L3: (1,2)
/// ```
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn sort_network_4(v: uint32x4_t) -> uint32x4_t {
    let mut r = v;

    // L1: CAS (0,1),(2,3) — swap adjacent pairs
    {
        let s = vrev64q_u32(r); // [1,0,3,2]
        let lo = vminq_u32(r, s);
        let hi = vmaxq_u32(r, s);
        // lanes 0,2 from lo, lanes 1,3 from hi
        let mask = vcombine_u32(vcreate_u32(0), vcreate_u32(0xFFFFFFFF_00000000));
        r = vbslq_u32(
            vreinterpretq_u32_u64(vdupq_n_u64(0x00000000_FFFFFFFF)),
            lo,
            hi,
        );
        // Actually: use element-wise selection
        // lane 0: min(r[0],r[1]), lane 1: max(r[0],r[1]), lane 2: min(r[2],r[3]), lane 3: max(r[2],r[3])
        let lo0 = vgetq_lane_u32(lo, 0);
        let hi1 = vgetq_lane_u32(hi, 1);
        let lo2 = vgetq_lane_u32(lo, 2);
        let hi3 = vgetq_lane_u32(hi, 3);
        r = vsetq_lane_u32(lo0, r, 0);
        r = vsetq_lane_u32(hi1, r, 1);
        r = vsetq_lane_u32(lo2, r, 2);
        r = vsetq_lane_u32(hi3, r, 3);
    }

    // L2: CAS (0,2),(1,3) — compare distance-2
    {
        // Swap halves: [2,3,0,1]
        let s = vextq_u32(r, r, 2);
        let lo = vminq_u32(r, s);
        let hi = vmaxq_u32(r, s);
        // lanes 0,1 from lo, lanes 2,3 from hi
        let lo_low = vget_low_u32(vreinterpretq_u32_u64(vreinterpretq_u64_u32(lo)));
        let hi_high = vget_high_u32(vreinterpretq_u32_u64(vreinterpretq_u64_u32(hi)));
        r = vcombine_u32(vget_low_u32(lo), vget_high_u32(hi));
    }

    // L3: CAS (1,2) — final fixup
    {
        // Shuffle to [0,2,1,3]
        let s = vextq_u32(r, r, 0); // identity, need [0,2,1,3]
                                    // NEON doesn't have arbitrary 4-lane permute like AVX2
                                    // Extract, compare, reinsert
        let v1 = vgetq_lane_u32(r, 1);
        let v2 = vgetq_lane_u32(r, 2);
        if v1 > v2 {
            r = vsetq_lane_u32(v2, r, 1);
            r = vsetq_lane_u32(v1, r, 2);
        }
    }

    r
}

/// Sort up to 8 u32 elements using NEON sorting networks.
///
/// For n ≤ 4: single 4-lane sort.
/// For 5-8: sort two 4-lane halves, then bitonic merge.
#[cfg(target_arch = "aarch64")]
pub unsafe fn sort_u32_8(slice: &mut [u32]) {
    let len = slice.len();
    if len <= 1 {
        return;
    }

    if len <= 4 {
        let mut buf = [u32::MAX; 4];
        buf[..len].copy_from_slice(slice);
        let v = vld1q_u32(buf.as_ptr());
        let sorted = sort_network_4(v);
        vst1q_u32(buf.as_mut_ptr(), sorted);
        slice.copy_from_slice(&buf[..len]);
    } else {
        // 5-8 elements: sort two halves, then merge
        let mut buf_lo = [u32::MAX; 4];
        let mut buf_hi = [u32::MAX; 4];
        buf_lo.copy_from_slice(&slice[..4]);
        let hi_len = len - 4;
        buf_hi[..hi_len].copy_from_slice(&slice[4..]);

        let mut lo = sort_network_4(vld1q_u32(buf_lo.as_ptr()));
        let mut hi = sort_network_4(vld1q_u32(buf_hi.as_ptr()));

        // Bitonic merge: reverse hi, compare with lo
        let hi_rev = vrev64q_u32(hi);
        let hi_rev = vextq_u32(hi_rev, hi_rev, 2); // full reverse: [3,2,1,0]
        let new_lo = vminq_u32(lo, hi_rev);
        let new_hi = vmaxq_u32(lo, hi_rev);

        // Merge within each register
        lo = merge_within_4(new_lo);
        hi = merge_within_4(new_hi);

        vst1q_u32(buf_lo.as_mut_ptr(), lo);
        vst1q_u32(buf_hi.as_mut_ptr(), hi);
        slice[..4].copy_from_slice(&buf_lo);
        slice[4..].copy_from_slice(&buf_hi[..hi_len]);
    }
}

/// Merge a bitonic 4-element sequence into sorted order.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn merge_within_4(v: uint32x4_t) -> uint32x4_t {
    let mut r = v;

    // Distance-2: swap halves and compare
    {
        let s = vextq_u32(r, r, 2);
        let lo = vminq_u32(r, s);
        let hi = vmaxq_u32(r, s);
        r = vcombine_u32(vget_low_u32(lo), vget_high_u32(hi));
    }

    // Distance-1: swap adjacent and compare
    {
        let v1 = vgetq_lane_u32(r, 1);
        let v0 = vgetq_lane_u32(r, 0);
        if v0 > v1 {
            r = vsetq_lane_u32(v1, r, 0);
            r = vsetq_lane_u32(v0, r, 1);
        }
        let v2 = vgetq_lane_u32(r, 2);
        let v3 = vgetq_lane_u32(r, 3);
        if v2 > v3 {
            r = vsetq_lane_u32(v3, r, 2);
            r = vsetq_lane_u32(v2, r, 3);
        }
    }

    r
}

/// Sort up to 8 elements of any 4-byte SortableKey type using NEON.
#[cfg(target_arch = "aarch64")]
pub unsafe fn sort_tiny_u32_keys_generic<T: SortableKey>(slice: &mut [T]) {
    let len = slice.len();
    if len <= 1 {
        return;
    }

    let mut keys: [u32; 8] = [u32::MAX; 8];
    for (i, elem) in slice.iter().enumerate() {
        let key = elem.to_radix_key();
        keys[i] = core::ptr::read(&key as *const T::Key as *const u32);
    }

    sort_u32_8(&mut keys[..len]);

    for (i, elem) in slice.iter_mut().enumerate() {
        let key = core::ptr::read(&keys[i] as *const u32 as *const T::Key);
        *elem = T::from_radix_key(key);
    }
}

/// NEON is always available on aarch64.
#[inline]
pub fn is_available() -> bool {
    cfg!(target_arch = "aarch64")
}
