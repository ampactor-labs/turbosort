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

/// The bits in which some key of `slice` (which must not be empty) differs
/// from the first key.
///
/// Digits that never differ need no histogram and no pass. The scan stops as
/// soon as every byte of the key has differed somewhere, which random keys do
/// within the first block; every histogram is needed then anyway.
pub fn varying_bits<T: SortableKey>(slice: &[T]) -> u64 {
    let first = slice[0].to_radix_key().to_u64();
    let every_byte = |diff: u64| (0..T::Key::BYTES).all(|p| (diff >> (8 * p)) & 0xFF != 0);
    let mut diff = 0;
    for block in slice.chunks(64) {
        for x in block {
            diff |= x.to_radix_key().to_u64() ^ first;
        }
        if every_byte(diff) {
            break;
        }
    }
    diff
}

/// Like [`compute_bytes`], for the digits in `live` only. Every other digit
/// is the same in all keys, so its row gets every key in the first key's
/// bucket, as a full count would.
pub fn compute_bytes_live<T: SortableKey, const PASSES: usize, const COPIES: usize>(
    slice: &[T],
    live: &[usize],
) -> [[usize; 256]; PASSES] {
    let mut h = [[[0usize; 256]; PASSES]; COPIES];

    let mut chunks = slice.chunks_exact(COPIES);
    for chunk in &mut chunks {
        for c in 0..COPIES {
            let key = chunk[c].to_radix_key();
            for &pass in live {
                h[c][pass][key.radix_digit(pass) as usize] += 1;
            }
        }
    }
    for elem in chunks.remainder() {
        let key = elem.to_radix_key();
        for &pass in live {
            h[0][pass][key.radix_digit(pass) as usize] += 1;
        }
    }

    let mut out = h[0];
    for copy in h.iter().skip(1) {
        for &pass in live {
            for d in 0..256 {
                out[pass][d] += copy[pass][d];
            }
        }
    }
    let first = slice[0].to_radix_key();
    for (pass, row) in out.iter_mut().enumerate() {
        if !live.contains(&pass) {
            row[first.radix_digit(pass) as usize] = slice.len();
        }
    }
    out
}

/// The histogram of the byte digit at `pass` alone, for tests.
#[cfg(test)]
pub fn compute_bytes_one<T: SortableKey>(slice: &[T], pass: usize) -> [[usize; 256]; 1] {
    let mut h = [0usize; 256];
    for elem in slice {
        h[elem.to_radix_key().radix_digit(pass) as usize] += 1;
    }
    [h]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varying_bits_finds_the_bytes_that_differ() {
        // Only byte 2 and the top bit vary.
        let keys: Vec<u64> = (0..300u64)
            .map(|i| 0x1111_2222_0000_4455 | (i % 256) << 16 | (i / 256) << 63)
            .collect();
        assert_eq!(varying_bits(&keys), 0x8000_0000_00FF_0000);
        assert_eq!(varying_bits(&[7u32; 5]), 0);
    }

    #[test]
    fn live_histograms_match_full_ones() {
        let keys: Vec<u64> = (0..10_000u64)
            .map(|i| (i * 2654435761 % 4099) << 24)
            .collect();
        let live: Vec<usize> = (0..8)
            .filter(|&p| (varying_bits(&keys) >> (8 * p)) & 0xFF != 0)
            .collect();
        assert_eq!(live, vec![3, 4]);
        assert_eq!(
            compute_bytes_live::<u64, 8, 4>(&keys, &live),
            compute_bytes::<u64, 8, 4>(&keys)
        );
        assert_eq!(
            compute_bytes_live::<u64, 8, 1>(&keys, &live),
            compute_bytes::<u64, 8, 1>(&keys)
        );
    }
}
