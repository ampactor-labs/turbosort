//! Histogram computation for radix sort.

use crate::key::{SortableKey, UnsignedKey};

/// Below this length, a single histogram copy wins: the fixed cost of zeroing
/// and folding four copies outweighs the store-forwarding stalls it avoids.
pub const INTERLEAVE_MIN: usize = 8192;

/// Compute per-pass byte histograms with `COPIES`-way interleaving.
///
/// Interleaved copies let consecutive elements increment independent counters,
/// avoiding store-forwarding stalls when neighboring elements share a digit.
/// The stack cost is `COPIES × PASSES × 2KB`, so callers pick `COPIES = 1` for
/// small inputs and `4` for large ones (see [`INTERLEAVE_MIN`]).
pub fn compute_bytes<T: SortableKey, const PASSES: usize, const COPIES: usize>(
    slice: &[T],
) -> [[usize; 256]; PASSES] {
    let mut h = [[[0usize; 256]; PASSES]; COPIES];

    let mut chunks = slice.chunks_exact(COPIES);
    for chunk in &mut chunks {
        for c in 0..COPIES {
            let key = chunk[c].to_radix_key();
            for pass in 0..PASSES {
                h[c][pass][key.radix_digit(pass) as usize] += 1;
            }
        }
    }
    for elem in chunks.remainder() {
        let key = elem.to_radix_key();
        for pass in 0..PASSES {
            h[0][pass][key.radix_digit(pass) as usize] += 1;
        }
    }

    let mut out = h[0];
    for copy in h.iter().skip(1) {
        for pass in 0..PASSES {
            for d in 0..256 {
                out[pass][d] += copy[pass][d];
            }
        }
    }
    out
}

/// Check if a pass can be skipped (all elements have the same digit).
///
/// A pass where one bin has count == total length means that digit position
/// is constant across all keys — the scatter would be a no-op. (The serial
/// cores read the same fact off the largest bucket, which diverting needs.)
#[cfg(feature = "parallel")]
#[inline]
pub fn is_pass_trivial(histogram: &[usize], total: usize) -> bool {
    histogram.contains(&total)
}
