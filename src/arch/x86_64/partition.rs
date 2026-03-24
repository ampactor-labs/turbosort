//! AVX2 SIMD partition for u32 quicksort.
//!
//! Implements the Bramas neutralize strategy: vectorized comparison + compress-store
//! via LUT, reading from whichever side has less room between read/write pointers.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::lut::COMPRESS_LUT;

/// Partition a slice of u32 around a pivot using AVX2 vectorized comparison.
///
/// Returns the number of elements less than the pivot (partition point).
/// Uses the Bramas neutralize strategy: always read from the side with less
/// remaining room between the read and write pointers.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub unsafe fn partition_u32(slice: &mut [u32], pivot: u32) -> usize {
    let len = slice.len();
    if len < 16 {
        return scalar_partition_u32(slice, pivot);
    }

    let pivot_vec = _mm256_set1_epi32(pivot as i32);
    let mut write_left = 0usize;
    let mut write_right = len;
    let mut read_left = 0usize;
    let mut read_right = len - 8;

    // Save the first and last 8 elements — process them after the main loop.
    // This gives the write pointers room to operate in the middle without
    // clobbering elements we haven't read yet.
    let vec_left = _mm256_loadu_si256(slice.as_ptr().add(read_left) as *const __m256i);
    read_left += 8;
    let vec_right = _mm256_loadu_si256(slice.as_ptr().add(read_right) as *const __m256i);
    // read_right stays pointing at the start of the saved right block

    while read_left + 8 <= read_right {
        // Neutralize: read from whichever side has less room to avoid the
        // write pointer catching the read pointer.
        let left_room = read_left - write_left;
        let right_room = write_right - read_right;
        let v = if left_room <= right_room {
            let v = _mm256_loadu_si256(slice.as_ptr().add(read_left) as *const __m256i);
            read_left += 8;
            v
        } else {
            read_right -= 8;
            _mm256_loadu_si256(slice.as_ptr().add(read_right) as *const __m256i)
        };
        partition_vector(v, pivot_vec, slice, &mut write_left, &mut write_right);
        debug_assert!(write_left <= write_right, "write pointers crossed");
    }

    // Scalar tail for the remaining gap between read pointers.
    // The read pointers may have crossed (read_right < read_left) when data
    // is heavily skewed to one side, so we need to handle the full unread range.
    let tail_start = read_left.min(read_right);
    let tail_end = read_left.max(read_right);
    // Save the tail elements first, since the write pointers may overlap this region.
    let mut tail_buf = [0u32; 15];
    let tail_len = tail_end - tail_start;
    tail_buf[..tail_len].copy_from_slice(&slice[tail_start..tail_end]);
    for &elem in &tail_buf[..tail_len] {
        if elem < pivot {
            slice[write_left] = elem;
            write_left += 1;
        } else {
            write_right -= 1;
            slice[write_right] = elem;
        }
    }

    // Now process the saved first and last vectors
    partition_vector(
        vec_left,
        pivot_vec,
        slice,
        &mut write_left,
        &mut write_right,
    );
    partition_vector(
        vec_right,
        pivot_vec,
        slice,
        &mut write_left,
        &mut write_right,
    );

    write_left
}

/// Partition 8 elements from a vector into left (< pivot) and right (>= pivot).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn partition_vector(
    v: __m256i,
    pivot: __m256i,
    slice: &mut [u32],
    write_left: &mut usize,
    write_right: &mut usize,
) {
    // Unsigned comparison via bias trick: add i32::MIN to both, then use signed cmpgt
    let bias = _mm256_set1_epi32(i32::MIN);
    let v_biased = _mm256_add_epi32(v, bias);
    let p_biased = _mm256_add_epi32(pivot, bias);
    let cmp = _mm256_cmpgt_epi32(p_biased, v_biased);
    let mask = _mm256_movemask_epi8(cmp);

    let lane_mask = compress_byte_mask_to_lane_mask(mask);
    let count_left = lane_mask.count_ones() as usize;
    let count_right = 8 - count_left;

    let perm = _mm256_loadu_si256(COMPRESS_LUT[lane_mask as usize].as_ptr() as *const __m256i);
    let permuted = _mm256_permutevar8x32_epi32(v, perm);

    // Store left elements contiguously
    let ptr = slice.as_mut_ptr();
    let mut buf = [0u32; 8];
    _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, permuted);
    core::ptr::copy_nonoverlapping(buf.as_ptr(), ptr.add(*write_left), count_left);
    *write_left += count_left;

    // Store right elements at the right end
    *write_right -= count_right;
    core::ptr::copy_nonoverlapping(
        buf.as_ptr().add(count_left),
        ptr.add(*write_right),
        count_right,
    );
}

/// Convert a byte-granularity movemask to a lane-granularity mask.
#[inline(always)]
fn compress_byte_mask_to_lane_mask(byte_mask: i32) -> u8 {
    let m = byte_mask as u32;
    let mut lane_mask = 0u8;
    let mut i = 0;
    while i < 8 {
        if m & (1 << (i * 4)) != 0 {
            lane_mask |= 1 << i;
        }
        i += 1;
    }
    lane_mask
}

/// Scalar partition fallback for < 16 elements.
pub fn scalar_partition_u32(slice: &mut [u32], pivot: u32) -> usize {
    let mut left = 0;
    let mut right = slice.len();
    loop {
        while left < right && slice[left] < pivot {
            left += 1;
        }
        while left < right && slice[right - 1] >= pivot {
            right -= 1;
        }
        if left >= right {
            return left;
        }
        slice.swap(left, right - 1);
        left += 1;
        right -= 1;
    }
}
