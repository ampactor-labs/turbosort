//! LSD (Least-Significant-Digit) radix sort.
//!
//! Sorts by processing bytes from least-significant to most-significant.
//! Each pass is a stable counting sort on one byte, so after all passes
//! the array is fully sorted. Uses a ping-pong buffer to avoid copies.
//!
//! # Algorithm
//!
//! 1. Compute histograms for all passes in a single scan.
//! 2. For each pass (skipping trivial ones):
//!    a. Convert histogram to exclusive prefix sum (offsets).
//!    b. Scatter elements from source to destination.
//!    c. Swap source/destination roles (ping-pong).
//! 3. If the final result is in the buffer, copy back to the input slice.

pub mod histogram;
pub mod prefix_sum;
pub mod scatter;

#[cfg(feature = "alloc")]
extern crate alloc;

use crate::key::{SortableKey, UnsignedKey};

/// Sort a slice using LSD radix sort with an internally allocated buffer.
///
/// Requires the `alloc` feature.
#[cfg(feature = "alloc")]
pub fn sort<T: SortableKey>(slice: &mut [T]) {
    if slice.len() <= 1 {
        return;
    }
    // SAFETY: T is one of the 10 primitive numeric types (sealed by SortableKey);
    // all-zero bytes is a valid representation for all of them.
    let mut buffer = alloc::vec![unsafe { core::mem::zeroed() }; slice.len()];
    sort_with_buffer(slice, &mut buffer);
}

/// Sort a slice using LSD radix sort with a caller-provided buffer.
///
/// The buffer must be at least as long as `slice`. This entry point
/// is usable in `no_std` environments that manage their own memory.
///
/// # Panics
///
/// Panics if `buffer.len() < slice.len()`.
pub fn sort_with_buffer<T: SortableKey>(slice: &mut [T], buffer: &mut [T]) {
    assert!(buffer.len() >= slice.len());
    let len = slice.len();
    if len <= 1 {
        return;
    }

    let passes = T::Key::BYTES;

    // Step 1: 4-way interleaved histogram reduces store-forwarding stalls
    // on random-access increments. Each copy handles independent elements.
    // Stack-allocated: 4 × passes × 256 entries. For u64 (8 passes): 64KB.
    let mut h = [[0usize; 8 * 256]; 4];
    let chunks = len / 4;
    for i in 0..chunks {
        let k0 = slice[i * 4].to_radix_key();
        let k1 = slice[i * 4 + 1].to_radix_key();
        let k2 = slice[i * 4 + 2].to_radix_key();
        let k3 = slice[i * 4 + 3].to_radix_key();
        for pass in 0..passes {
            h[0][pass * 256 + k0.radix_digit(pass) as usize] += 1;
            h[1][pass * 256 + k1.radix_digit(pass) as usize] += 1;
            h[2][pass * 256 + k2.radix_digit(pass) as usize] += 1;
            h[3][pass * 256 + k3.radix_digit(pass) as usize] += 1;
        }
    }
    // Scalar tail
    for elem in slice[chunks * 4..].iter() {
        let key = elem.to_radix_key();
        for pass in 0..passes {
            h[0][pass * 256 + key.radix_digit(pass) as usize] += 1;
        }
    }
    // Merge 4 copies
    let mut histograms = [0usize; 8 * 256];
    for copy in &h {
        for (g, c) in histograms.iter_mut().zip(copy.iter()) {
            *g += c;
        }
    }

    // Step 2: for each pass, scatter from src to dst, then swap roles
    let mut in_buffer = false;

    for pass in 0..passes {
        let hist_slice = &histograms[pass * 256..(pass + 1) * 256];

        // Skip trivial passes (all elements have the same digit)
        if histogram::is_pass_trivial(hist_slice, len) {
            continue;
        }

        // Build offsets from histogram
        let mut offsets = [0usize; 256];
        offsets.copy_from_slice(hist_slice);
        prefix_sum::exclusive_prefix_sum(&mut offsets);

        // Scatter
        if in_buffer {
            scatter::scatter_pass(buffer, slice, &mut offsets, pass);
        } else {
            scatter::scatter_pass(slice, buffer, &mut offsets, pass);
        }
        in_buffer = !in_buffer;
    }

    // Step 3: if final result is in buffer, copy back
    if in_buffer {
        slice.copy_from_slice(buffer);
    }
}
